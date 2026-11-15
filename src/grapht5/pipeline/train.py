import logging
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from grapht5.pipeline import evaluation
from grapht5.runtime import util
from grapht5.runtime.slurm import get_device, move_to_device
from grapht5.runtime.util import (
    average_all,
    calc_codebleu_safe,
    init_worker_seed,
    weighted_average_all,
    enable_t5_updates_at_step,
)
from grapht5.runtime.wandb_utils import get_wandb_logger
from grapht5.runtime.vdg_diagnostics import (
    gate_metrics,
    gradient_metrics,
    parameter_snapshot,
    unwrap_model,
    update_metrics,
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self, model, tokenizer, optimizer, scheduler, opt, collator, checkpoint_path
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt = opt
        self.collator = collator
        self.checkpoint_path = checkpoint_path
        self._closed = False
        self.last_grad_norm = None
        self.last_vdg_diagnostic_bucket = 0
        self.epoch = 1

        self.writer = SummaryWriter(checkpoint_path / "runs")
        self.wandb_logger = get_wandb_logger(opt)

    def train(self, step, train_dataset, eval_dataset, best_eval):
        # Initialize wandb model watching
        self.wandb_logger.watch_model(self.model)
        train_sampler = self._get_train_sampler(train_dataset)
        train_dataloader = self._get_train_dataloader(train_dataset, train_sampler)

        curr_loss = 0.0
        self.model.train()

        while step < self.opt.total_steps:
            if self.opt.is_main:
                logger.info(
                    f"Starting epoch {self.epoch} with {len(train_dataloader)} batches/{self.opt.world_size}."
                )

            if self.opt.is_distributed:
                train_sampler.set_epoch(self.epoch)

            start_epoch = time.time()
            step, curr_loss, best_eval = self._train_epoch(
                step, curr_loss, best_eval, train_dataloader, eval_dataset
            )
            epoch_time = time.time() - start_epoch
            if self.opt.is_main:
                logger.info(
                    f"Completed epoch {self.epoch} in {epoch_time:.2f} seconds."
                )

            self.epoch += 1

    def close(self, finish_wandb=True):
        if self._closed:
            return
        if hasattr(self, "writer"):
            self.writer.close()
        if finish_wandb and hasattr(self, "wandb_logger"):
            self.wandb_logger.finish()
        self._closed = True

    def __del__(self):
        if hasattr(self, "_closed"):
            self.close()

    def _get_train_sampler(self, train_dataset):
        if self.opt.is_distributed:
            return DistributedSampler(
                train_dataset,
                num_replicas=self.opt.world_size,
                rank=self.opt.global_rank,
                seed=self.opt.seed,
                shuffle=True,
            )
        return RandomSampler(train_dataset)

    def _get_train_dataloader(self, dataset, sampler):
        g, seed_worker = init_worker_seed()

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.opt.per_gpu_train_batch_size,
            drop_last=True,
            pin_memory=True,
            generator=g,
            worker_init_fn=seed_worker,
            num_workers=self.opt.per_gpu_train_batch_size * 2,
            collate_fn=self.collator,
            persistent_workers=True,
        )

    def _train_epoch(self, step, curr_loss, best_eval, train_dataloader, eval_dataset):
        with tqdm(
            desc="Training...",
            unit="batch",
            file=None if self.opt.is_main else open(os.devnull, "w"),
            total=len(train_dataloader) * self.opt.world_size,
            ascii=True,
            leave=False,
            smoothing=0.1,
        ) as pbar:
            for i, batch in enumerate(train_dataloader):
                batch = move_to_device(batch, get_device(self.opt, self.opt.local_rank))

                step += 1

                if self.opt.freeze_t5 > 0 and step == self.opt.freeze_t5:
                    enabled = enable_t5_updates_at_step(
                        self.model, step, self.opt.freeze_t5, self.opt.is_main
                    )
                    if enabled:
                        self.wandb_logger.log_metrics({"training/t5_unfrozen": 1}, step)

                train_loss, curr_loss = self._train_step(step, batch, curr_loss)
                current_period = step % self.opt.eval_freq or self.opt.eval_freq

                pbar.set_postfix(
                    {
                        "current loss": f"{train_loss:.3f}",
                        "avg. loss (eval)": f"{curr_loss / current_period:.3f}",
                        "LR T5/VDG:": f"{self.scheduler.get_last_lr()[0]:.9f}, {self.scheduler.get_last_lr()[1]:.9f}",
                    }
                )

                pbar.update(self.opt.world_size)

                if step % self.opt.eval_freq == 0:
                    if not self.opt.cpu:
                        torch.cuda.empty_cache()

                    best_eval = self._evaluate_and_save(
                        step, curr_loss, eval_dataset, best_eval
                    )
                    curr_loss = 0.0

                if step % self.opt.save_freq == 0:
                    if self.opt.is_main:
                        logger.info(f"Saving checkpoint at step-{step} ...")
                        self._save_checkpoint(step, best_eval)

                    if self.opt.is_distributed:
                        torch.distributed.barrier()

                if step >= self.opt.total_steps:
                    break

        if not self.opt.cpu:
            torch.cuda.empty_cache()

        return step, curr_loss, best_eval

    def _train_step(self, step, batch, curr_loss):
        (
            _,
            labels,
            _,
            input_ids,
            attention_mask,
            node_ids,
            node_masks,
            node_roles,
            edges,
        ) = batch

        is_accumulation_step = step % self.opt.accumulation_steps != 0
        diagnostic_due = self._vdg_diagnostic_due(step, is_accumulation_step)
        with self._backward_context(not is_accumulation_step):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                node_ids=node_ids,
                node_masks=node_masks,
                node_roles=node_roles,
                edges=edges,
            )
            (output[0] / self.opt.accumulation_steps).backward()
        train_loss = output[0]

        if 0 < self.opt.freeze_t5 and step < self.opt.freeze_t5:
            for param in self.optimizer.param_groups[0]["params"]:
                param.grad = None

        curr_loss += train_loss.item()

        diagnostic_metrics = {}
        if not is_accumulation_step:
            snapshot = None
            if diagnostic_due:
                diagnostic_metrics.update(gradient_metrics(self.model))
                snapshot = parameter_snapshot(self.model)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt.clip
            )
            self.last_grad_norm = grad_norm.item()
            self.optimizer.step()

            if self.opt.scheduler != "plateau":
                self.scheduler.step()

            if diagnostic_due:
                diagnostic_metrics.update(update_metrics(self.model, snapshot))
                diagnostic_metrics.update(gate_metrics(self.model))

            self.model.zero_grad(set_to_none=True)

            curr_loss = torch.tensor(curr_loss, device=self.model.device)
            curr_loss = average_all(curr_loss, self.opt).item()

            if diagnostic_due:
                if step % self.opt.wandb_log_freq:
                    self.wandb_logger.log_metrics(diagnostic_metrics, step)
                self.last_vdg_diagnostic_bucket = step // self.opt.wandb_log_freq

        # Log training metrics to wandb
        if step % self.opt.wandb_log_freq == 0:
            wandb_metrics = {
                "train/loss": train_loss.item(),
                "train/avg_loss": curr_loss / max(1, step % self.opt.eval_freq),
                "train/lr_t5": self.scheduler.get_last_lr()[0],
                "train/lr_vdg": (
                    self.scheduler.get_last_lr()[1]
                    if len(self.scheduler.get_last_lr()) > 1
                    else self.scheduler.get_last_lr()[0]
                ),
                "train/epoch": self.epoch,
            }
            if self.last_grad_norm is not None:
                wandb_metrics["train/grad_norm_pre_clip"] = self.last_grad_norm
                wandb_metrics["train/grad_clipped"] = float(
                    self.last_grad_norm > self.opt.clip
                )
            wandb_metrics.update(diagnostic_metrics)
            self.wandb_logger.log_metrics(wandb_metrics, step)

            # Log parameter statistics
            # self.wandb_logger.log_parameters(self.model, step)

            # Log parameter counts (frozen vs unfrozen)
            self.wandb_logger.log_parameter_counts(self.model, step)

            # Generate encoder heatmaps
            # self.wandb_logger.plot_encoder_heatmaps(self.model, step)
            self.wandb_logger.plot_cross_attention_gates(self.model, step)

        return train_loss, curr_loss

    def _backward_context(self, synchronize):
        if self.opt.is_distributed and not synchronize:
            return self.model.no_sync()
        return nullcontext()

    def _vdg_diagnostic_due(self, step, is_accumulation_step):
        graph_encoder = getattr(unwrap_model(self.model), "graph_encoder", None)
        return (
            self.wandb_logger.enabled
            and graph_encoder is not None
            and not is_accumulation_step
            and step // self.opt.wandb_log_freq > self.last_vdg_diagnostic_bucket
        )

    def _evaluate_and_save(self, step, curr_loss, eval_dataset, best_eval):
        logger.info(f"Start evaluation ...")

        em, cb = evaluate(
            self.model, eval_dataset, self.collator, self.opt, self.wandb_logger, step
        )

        if not self.opt.cpu:
            torch.cuda.empty_cache()

        metrics = []

        if self.opt.with_em:
            metrics.append(em)
        if self.opt.with_codebleu:
            metrics.append(cb["codebleu"])

        curr_eval = sum(metrics) / len(metrics) if metrics else 0

        if self.opt.scheduler == "plateau":
            self.scheduler.step(curr_eval)

        self.model.train()

        if self.opt.is_main:
            if curr_eval > best_eval:
                logger.info(
                    f"New best model found in epoch {self.epoch} (step={step}), "
                    f"improving from {100 * best_eval:.4f} to {100 * curr_eval:.4f} ({self.opt.eval_metric_display_name })."
                )

                best_eval = curr_eval

                util.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    step,
                    best_eval,
                    self.opt,
                    self.checkpoint_path,
                    "best_dev",
                )

            # Log evaluation metrics to wandb
            wandb_eval_metrics = {
                "eval/exact_match": em * 100,
                "eval/codebleu": cb["codebleu"] * 100,
                "eval/bleu_4": cb["weighted_ngram_match_score"] * 100,
                "eval/combined_score": curr_eval * 100,
                "eval/best_score": best_eval * 100,
                "train/avg_loss_at_eval": curr_loss / self.opt.eval_freq,
            }
            self.wandb_logger.log_metrics(wandb_eval_metrics, step)

            log = f"epoch: {self.epoch} | "
            log += f"{step} / {self.opt.total_steps} | "
            log += f"train: {curr_loss / self.opt.eval_freq:.3f} | "
            log += f"{100 * em:.2f} EM | {100 * cb['codebleu']:.2f} CodeBLEU | {100 * cb['weighted_ngram_match_score']:.2f} BLEU-4 | "
            log += f"lr: {self.scheduler.get_last_lr()[0]:.9f}"
            logger.info(log)

        return best_eval

    def _save_checkpoint(self, step, best_dev_em):
        util.save(
            self.model,
            self.optimizer,
            self.scheduler,
            step,
            best_dev_em,
            self.opt,
            self.checkpoint_path,
            f"step-{step}",
        )


def evaluate(model, dataset, collator, opt, wandb_logger=None, step=None):
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=opt.world_size,
            rank=opt.global_rank,
            seed=opt.seed,
            shuffle=False,
        )
        if opt.is_distributed
        else SequentialSampler(dataset)
    )

    # start workers before waiting on other processes
    collator.eval()

    dataloader = _get_eval_dataloader(dataset, collator, sampler, opt)
    loader_iter = iter(dataloader)

    # wait for other processes
    if opt.is_distributed:
        torch.distributed.barrier()

    model.eval()

    total = 0
    exactmatch, golds, preds = [], [], []
    model = model.module if hasattr(model, "module") else model

    # Store individual examples for wandb logging
    eval_examples = []
    with torch.no_grad():
        with tqdm(
            desc="Evaluating",
            unit="batch",
            total=len(dataloader) * opt.world_size,
            ascii=True,
            leave=False,
            file=None if opt.is_main else open(os.devnull, "w"),
        ) as pbar:
            for i, batch in enumerate(loader_iter):
                (
                    idx,
                    labels,
                    target_mask,
                    input_ids,
                    attention_mask,
                    node_ids,
                    node_masks,
                    node_roles,
                    edges,
                ) = move_to_device(batch, get_device(opt, opt.local_rank))

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    node_ids=node_ids,
                    node_masks=node_masks,
                    node_roles=node_roles,
                    edges=edges,
                    max_length=opt.answer_maxlength,
                )

                for k, o in enumerate(outputs):
                    ans = collator.tokenizer.decode(o, skip_special_tokens=True)
                    gold = [dataset.get_example(idx[k]).target]

                    golds.append(gold)
                    preds.append(ans)

                    score = evaluation.ems(ans, gold)
                    total += 1
                    exactmatch.append(score)

                    # Store example for wandb logging
                    eval_examples.append(
                        {
                            "target": gold[0],
                            "generated_target": ans,
                            "exact_match": score,
                        }
                    )

                pbar.set_postfix({"EMs (main)": sum(exactmatch)})
                pbar.update(opt.world_size)

    bleu = calc_codebleu_safe(golds, preds, lang=opt.language)

    if opt.is_distributed:
        torch.distributed.barrier()

    collator.train()

    codebleu = {k: weighted_average_all(v, total, opt)[0] for k, v in bleu.items()}
    em, _ = weighted_average_all(np.mean(exactmatch), total, opt)

    # Log evaluation examples to wandb if logger is provided
    if wandb_logger and eval_examples and step is not None:
        import wandb

        columns = ["target", "generated_target", "exact_match"]
        data = [
            [ex["target"], ex["generated_target"], ex["exact_match"]]
            for ex in eval_examples
        ]
        table = wandb.Table(columns=columns, data=data)
        eval_examples_key = (
            f"eval/examples_r{opt.global_rank}"
            if opt.is_distributed
            else "eval/examples"
        )
        wandb_logger.log_metrics_all_processes({eval_examples_key: table}, step)

    return em, codebleu


def _get_eval_dataloader(dataset, collator, sampler, opt):
    g, seed_worker = init_worker_seed()

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_eval_batch_size,
        drop_last=False,
        pin_memory=True,
        collate_fn=collator,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=opt.per_gpu_eval_batch_size,
    )

    return dataloader
