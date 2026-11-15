# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import csv
import errno
import json
import logging
import math
import os
import random
import re
import signal
import sys
import traceback
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from codebleu import calc_codebleu
from safetensors import safe_open
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoConfig, AutoTokenizer

from types import SimpleNamespace

logger = logging.getLogger(__name__)
CHECKPOINTING_ENABLED = True

# constants
SEP_TOKEN = "<sep>"
TOKENS = [
    "<vul-start>",
    "<vul-end>",
    "<mod-start>",
    "<mod-end>",
    # SEP_TOKEN,
    "<blank>",
    "<null>",
]
EDGE_TYPES = ["DDG", "CDG", "CFG", "REF", "AST"]


def relative_position_bucket(distances, num_buckets, max_distance):
    half = max(1, num_buckets // 2)
    max_exact = max(1, half // 2)
    magnitude = distances.abs()
    large = (
        max_exact
        + (
            torch.log(magnitude.float().clamp_min(max_exact) / max_exact)
            / math.log(max(max_distance / max_exact, 1.0001))
            * (half - max_exact)
        ).long()
    )
    magnitude_bucket = torch.where(magnitude < max_exact, magnitude, large)
    magnitude_bucket = magnitude_bucket.clamp(max=half - 1)
    direction = (distances > 0).long() * half
    return (magnitude_bucket + direction).clamp(max=num_buckets - 1)


def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()

    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    return logger


def init_seeds(rank, seed=0):
    import torch.backends.cudnn as cudnn

    process_seed = seed + rank

    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(process_seed)

    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    cudnn.benchmark = False
    # cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_worker_seed():
    g = torch.Generator()
    g.manual_seed(0)

    return g, seed_worker


def init_tokenizer(opt):
    model_name = "Salesforce/codet5-" + opt.model_size
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.add_tokens(TOKENS)

    return tokenizer


def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, checkpoint_exists


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, step, best_eval_metric, opt, dir_path, name):
    if not CHECKPOINTING_ENABLED:
        return

    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
        "best_eval_metric": best_eval_metric,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)


def resolve_symlink(path):
    if os.path.islink(path):
        return Path(os.readlink(path))
    return Path(path)


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = resolve_symlink(dir_path)

    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)

    config = AutoConfig.from_pretrained(epoch_path)
    config.graph_attention_layers = opt.graph_attention_layers
    config.no_learned_gate = bool(
        opt.no_learned_gate or getattr(config, "no_learned_gate", False)
    )
    config.disable_edge_attention = bool(
        opt.disable_edge_attention or getattr(config, "disable_edge_attention", False)
    )
    config.add_consistency_loss = opt.add_consistency_loss
    config.drop_t5_passages = opt.drop_t5_passages

    model = model_class(config)
    with safe_open(os.path.join(epoch_path, "model.safetensors"), framework="pt") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        missing_keys = [
            k
            for k in missing_keys
            if not any(
                re.match(p, k) for p in model._keys_to_ignore_on_load_missing or []
            )
        ]
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not any(
                re.match(p, k) for p in model._keys_to_ignore_on_load_unexpected or []
            )
        ]

        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Unexpected weight issues:\nMissing: {missing_keys}\nUnexpected: {unexpected_keys}"
            )

    model = model.to(opt.device)

    step = 0
    if Path(optimizer_path).exists() and not reset_params:
        logger.info("loading checkpoint %s" % optimizer_path)
        checkpoint = torch.load(optimizer_path, map_location=opt.device)
        opt_checkpoint = checkpoint["opt"]
        step = checkpoint["step"]
    else:
        logger.warning("optimizer checkpoint does not exist, starting from scratch")
        checkpoint, opt_checkpoint = {}, {}

    best_eval_metric = 0.0
    if "best_eval_metric" in checkpoint:
        best_eval_metric = checkpoint["best_eval_metric"]
    if not reset_params:
        optimizer, scheduler = set_optim(SimpleNamespace(**opt_checkpoint), model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def average_all(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x = x / opt.world_size
    return x


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


def weighted_average_all(x, count, opt):
    if not opt.is_distributed:
        return x, count

    t_value = torch.tensor([x * count], device=opt.device)
    t_count = torch.tensor([count], device=opt.device)

    dist.all_reduce(t_value, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_count, op=dist.ReduceOp.SUM)

    weighted_avg = (t_value / t_count).item()
    total_count = t_count.item()

    return weighted_avg, total_count


def write_output(glob_path, output_path):
    files = list(glob_path.glob("*.txt"))
    files.sort()
    with open(output_path, "w") as outfile:
        for path in files:
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / "tmp_dir"
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f"{opt.global_rank}.json"
    with open(tmp_path, "w") as fw:
        json.dump(data, fw)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / "dataset_wscores.json"
        logger.info(f"Writing dataset with scores at {final_path}")
        results_path = write_path.glob("*.json")
        alldata = []
        for path in results_path:
            with open(path, "r") as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, "w") as fout:
            json.dump(alldata, fout, indent=4)
        write_path.rmdir()


def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter="\t")
        for k, row in enumerate(reader):
            if not row[0] == "id":
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    logger.warning(
                        f"The following input line has not been correctly loaded: {row}"
                    )
    return passages


def calc_codebleu_safe(golds, preds, **kwargs):
    _, bleu, error = run_in_process(calc_codebleu, golds, preds, **kwargs)

    if not bleu:
        logger.error(
            f"An error occurred while trying to calculcate BLEU metrics: {error}"
        )
        return {
            "codebleu": 0.0,
            "ngram_match_score": 0.0,
            "weighted_ngram_match_score": 0.0,
            "syntax_match_score": 0.0,
            "dataflow_match_score": 0.0,
        }

    return bleu


def process_wrapper(queue, func, *args, **kwargs):
    try:
        signal.signal(signal.SIGSEGV, signal.SIG_DFL)

        result = func(*args, **kwargs)
        queue.put(("success", result, None))
    except Exception as e:
        error = f"Error: {str(e)}\n{traceback.format_exc()}"
        queue.put(("error", None, error))


def run_in_process(func, *args, **kwargs):
    """
    Runs a function in a separate process and handles crashes gracefully.
    We use this to avoid runtime exceptions in Python, bringing the entire process group down during training because there is a bug in the CodeBLEU library, crashing the parser in certain edge cases (I was not able to pinpoint).

    Args:
        func: The function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        tuple: (success: bool, result: Any, error: str)
    """

    queue = Queue()
    process = Process(
        target=process_wrapper,
        args=(queue, func) + args,
        kwargs=kwargs,
    )
    process.start()

    try:
        status, result, error = queue.get(timeout=60)
        process.join()

        if status == "success":
            return True, result, None
        else:
            return False, None, error

    except Exception as e:
        process.terminate()
        process.join()
        return False, None, f"Process failed: {str(e)}"


GRAPH_PARAMETER_MARKERS = (
    "graph_encoder.",
    "EncDecAttentionGraph.",
    "cross_gate",
)


def is_graph_parameter(name):
    return any(marker in name for marker in GRAPH_PARAMETER_MARKERS)


def enable_t5_updates_at_step(model, step, freeze_until_step, is_main=True):
    if freeze_until_step <= 0 or step < freeze_until_step:
        return False

    unfrozen_count = sum(
        1 for name, _ in model.named_parameters() if not is_graph_parameter(name)
    )

    if is_main:
        logger.info(
            f"Enabled updates for {unfrozen_count} T5 parameters at step {step}"
        )

    return True


def set_optim(opt, model):
    vdg_params = []
    t5_params = []

    if opt.freeze_t5 is not None and opt.freeze_t5 != 0 and opt.is_main:
        if opt.freeze_t5 == -1:
            logger.warning(
                "--freeze_t5=-1 was provided! T5 modules will be permanently frozen during training."
            )
        else:
            logger.warning(
                f"--freeze_t5={opt.freeze_t5} was provided! T5 modules will be frozen until step {opt.freeze_t5}."
            )

    for name, param in model.named_parameters():
        graph_parameter = is_graph_parameter(name)
        if graph_parameter:
            param.requires_grad = True
            vdg_params.append(param)
        else:
            param.requires_grad = opt.freeze_t5 != -1
            t5_params.append(param)

        if opt.freeze_t5 is not None and opt.freeze_t5 != 0 and graph_parameter:
            logger.info(f"Module {name} will receive gradient updates. ")

    param_groups = [
        {"params": t5_params, "lr": opt.lr},
        {
            "params": vdg_params,
            "lr": opt.lr_vdg,
            "weight_decay": opt.weight_decay_vdg,
        },
    ]

    optimizer, scheduler = None, None
    if opt.optim == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=opt.lr)
    elif opt.optim == "adamw":
        use_fused = torch.cuda.is_available()
        optimizer = torch.optim.AdamW(
            param_groups, lr=opt.lr, weight_decay=opt.weight_decay, fused=use_fused
        )
    if opt.scheduler == "fixed":
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == "linear":
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_steps
        else:
            scheduler_steps = opt.scheduler_steps
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=opt.warmup_steps,
            scheduler_steps=scheduler_steps,
            min_ratio=opt.min_lr_ratio,
            fixed_lr=opt.fixed_lr,
            freeze_t5_steps=opt.freeze_t5,
            t5_unfreeze_warmup_steps=opt.t5_unfreeze_warmup_steps,
            accumulation_steps=opt.accumulation_steps,
            t5_lr_ratio=(opt.t5_unfreeze_lr / opt.lr if opt.freeze_t5 > 0 else 1.0),
        )
    elif opt.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=opt.lr_factor,
            patience=opt.patience,
            threshold=opt.lr_threshold,
            min_lr=0.000005,
        )
    return optimizer, scheduler


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        scheduler_steps,
        min_ratio,
        fixed_lr,
        freeze_t5_steps=0,
        t5_unfreeze_warmup_steps=0,
        accumulation_steps=1,
        t5_lr_ratio=1.0,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        self.freeze_t5_step = max(0, freeze_t5_steps) // accumulation_steps
        self.t5_unfreeze_warmup_steps = t5_unfreeze_warmup_steps // accumulation_steps
        self.t5_lr_ratio = t5_lr_ratio
        super(WarmupLinearScheduler, self).__init__(
            optimizer,
            [self.t5_lr_lambda, self.lr_lambda],
            last_epoch=last_epoch,
        )

    def t5_lr_lambda(self, step):
        if not self.freeze_t5_step:
            return self.lr_lambda(step)
        if step < self.freeze_t5_step:
            return 0.0
        unfreeze_warmup_end = self.freeze_t5_step + self.t5_unfreeze_warmup_steps
        if step < unfreeze_warmup_end:
            progress = (step - self.freeze_t5_step) / float(
                max(1, self.t5_unfreeze_warmup_steps)
            )
            return self.t5_lr_ratio * progress
        if self.fixed_lr:
            return self.t5_lr_ratio
        return max(
            self.t5_lr_ratio * self.min_ratio,
            self.t5_lr_ratio
            * (
                1.0
                + (self.min_ratio - 1)
                * (step - unfreeze_warmup_end)
                / float(max(1, self.scheduler_steps - unfreeze_warmup_end))
            ),
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(
                max(1, self.warmup_steps)
            ) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(
            self.min_ratio,
            1.0
            + (self.min_ratio - 1)
            * (step - self.warmup_steps)
            / float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        return 1.0
