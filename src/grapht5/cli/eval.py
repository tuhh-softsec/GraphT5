from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from grapht5.config.config import log_config, parse_config
from grapht5.datasets import data
from grapht5.datasets.data import Dataset
from grapht5.modeling.model import FiDT5
from grapht5.pipeline.inference import evaluate
from grapht5.runtime import slurm, util
from grapht5.runtime.util import init_tokenizer, load


def setup_result_directory(opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / "test_results").mkdir(parents=True, exist_ok=True)
    if not directory_exists and opt.is_main:
        log_config(opt)


def validate_generation_options(opt):
    if opt.num_return_sequences < 1:
        raise ValueError("--num_return_sequences must be at least one.")
    if opt.num_return_sequences > opt.beam_size:
        raise ValueError(
            "--beam_size must be at least --num_return_sequences for beam search."
        )


def setup_dataset(opt, tokenizer):
    collator = data.Collator(
        tokenizer,
        text_maxlength=opt.text_maxlength,
        answer_maxlength=opt.answer_maxlength,
        node_max_length=opt.node_max_length,
        max_nodes_per_graph=opt.max_nodes_per_graph,
        n_context=opt.n_context,
    )

    eval_examples = data.load_data(
        opt.eval_data,
        max_nodes_per_graph=opt.max_nodes_per_graph,
    )
    dataset = Dataset(eval_examples, opt)

    if opt.is_distributed:
        eval_sampler = DistributedSampler(
            dataset,
            num_replicas=opt.world_size,
            rank=opt.global_rank,
            shuffle=False,
        )
    else:
        eval_sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_eval_batch_size,
        num_workers=opt.per_gpu_eval_batch_size,
        collate_fn=collator,
    )
    return dataloader, dataset


def main():
    opt = parse_config()

    try:
        slurm.init_distributed_mode(opt)
        slurm.init_signal_handler()

        validate_generation_options(opt)
        util.init_seeds(opt.global_rank, seed=opt.seed)
        setup_result_directory(opt)
        tokenizer = init_tokenizer(opt)

        logger = util.init_logger(
            opt.is_main,
            opt.is_distributed,
            Path(opt.checkpoint_dir) / opt.name / "run.log",
        )
        logger.info(opt.name)

        eval_dataloader, eval_dataset = setup_dataset(opt, tokenizer)

        model = load(FiDT5, opt.model_path, opt, reset_params=True)[0]
        if opt.disable_torch_compile:
            logger.info("Skipping torch.compile")
        elif opt.device.type == "mps":
            model = torch.compile(model, backend="aot_eager_decomp_partition")
        else:
            model = torch.compile(model, dynamic=True)

        logger.info(f"Starting eval for model {opt.name}")

        em, codebleu, total = evaluate(
            model, eval_dataset, eval_dataloader, tokenizer, opt
        )
        log = f"EM {100 * em:.2f} | "
        log += f"{100 * codebleu['codebleu']:.2f} CodeBLEU | "
        log += f"{100 * codebleu['weighted_ngram_match_score']:.2f} BLEU-4 | "
        log += f"total number of examples {total} "
        logger.info(log)
    finally:
        slurm.cleanup_distributed()


if __name__ == "__main__":
    main()
