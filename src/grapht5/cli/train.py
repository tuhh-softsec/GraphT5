import logging
from pathlib import Path

import torch
import transformers
from transformers import logging as tf_logging

from grapht5.config.config import log_config, parse_config, set_training_schedule
from grapht5.datasets import data
from grapht5.modeling.model import FiDT5
from grapht5.pipeline.train import Trainer
from grapht5.runtime import slurm, util
from grapht5.runtime.util import init_tokenizer, resolve_symlink

tf_logging.set_verbosity_error()
torch.set_warn_always(False)
logger = logging.getLogger(__name__)

torch._dynamo.config.cache_size_limit = 64


def setup_datasets(opt, tokenizer):
    collator = data.Collator(
        tokenizer,
        text_maxlength=opt.text_maxlength,
        answer_maxlength=opt.answer_maxlength,
        node_max_length=opt.node_max_length,
        max_nodes_per_graph=opt.max_nodes_per_graph,
        n_context=opt.n_context,
        passage_dropout=opt.passage_dropout,
    )

    train_examples = data.load_data(
        opt.train_data,
        max_nodes_per_graph=opt.max_nodes_per_graph,
        drop_empty_graphs=opt.drop_t5_passages,
    )
    train_dataset = data.Dataset(train_examples, opt)

    eval_examples = data.load_data(
        opt.eval_data,
        max_nodes_per_graph=opt.max_nodes_per_graph,
        drop_empty_graphs=opt.drop_t5_passages,
    )
    eval_dataset = data.Dataset(eval_examples, opt)

    return train_dataset, eval_dataset, collator


def main():
    opt = parse_config()

    trainer = None
    try:
        slurm.init_distributed_mode(opt)
        slurm.init_signal_handler()
        util.init_seeds(opt.global_rank, seed=0)

        checkpoint_path = Path(opt.checkpoint_dir) / opt.name
        load_path = Path(checkpoint_path / "checkpoint" / "latest")
        checkpoint_exists = resolve_symlink(load_path).exists()

        if opt.is_distributed:
            torch.distributed.barrier()
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        run_logger = util.init_logger(
            opt.is_main, opt.is_distributed, checkpoint_path / "run.log"
        )

        tokenizer = init_tokenizer(opt)
        train_dataset, eval_dataset, collator = setup_datasets(opt, tokenizer)
        set_training_schedule(opt, len(train_dataset))
        log_config(opt)

        if opt.model_path == "none" and not checkpoint_exists:
            t5 = transformers.T5ForConditionalGeneration.from_pretrained(opt.base_model)
            t5.resize_token_embeddings(len(tokenizer))

            if opt.use_adapted_model:
                run_logger.info(
                    f"NOTE: --use_adapted_model was specified; "
                    f"using pre-trained T5 model at {opt.adapted_model_path} as base."
                )
                adapted_model = torch.load(opt.adapted_model_path)
                t5.load_state_dict(adapted_model, strict=False)

            t5.config.graph_attention_layers = opt.graph_attention_layers
            t5.config.max_nodes_per_graph = opt.max_nodes_per_graph
            t5.config.node_max_length = opt.node_max_length
            t5.config.no_learned_gate = opt.no_learned_gate
            t5.config.disable_edge_attention = opt.disable_edge_attention
            t5.config.add_consistency_loss = opt.add_consistency_loss
            t5.config.drop_t5_passages = opt.drop_t5_passages
            t5.config.dropout_rate = opt.dropout

            model = FiDT5(t5.config)
            model.config.sep_token_id = tokenizer.convert_tokens_to_ids(util.SEP_TOKEN)
            model.load_t5(t5.state_dict())
            model = model.to(opt.device)

            optimizer, scheduler = util.set_optim(opt, model)
            step, best_eval = 0, 0.0
        elif opt.model_path == "none" and checkpoint_exists:
            model, optimizer, scheduler, _, step, best_eval = util.load(
                FiDT5,
                checkpoint_path / "checkpoint" / "latest",
                opt,
                reset_params=False,
            )
            run_logger.info(
                f"Checkpoint loaded from {checkpoint_path / 'checkpoint' / 'latest'}"
            )
        else:
            model, optimizer, scheduler, _, _, best_eval = util.load(
                FiDT5, opt.model_path, opt, reset_params=True
            )
            step = 0
            run_logger.info(f"Model loaded from {opt.model_path}")

        if opt.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank] if not opt.cpu else None,
                output_device=opt.local_rank if not opt.cpu else None,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )

        if opt.disable_torch_compile:
            run_logger.info("Skipping torch.compile")
        elif opt.device.type == "mps":
            model = torch.compile(model, backend="aot_eager_decomp_partition")
        else:
            model = torch.compile(model, dynamic=True)

        run_logger.info("Start training")

        trainer = Trainer(
            model, tokenizer, optimizer, scheduler, opt, collator, checkpoint_path
        )
        trainer.train(step, train_dataset, eval_dataset, best_eval)
        run_logger.info("Training finished")
    finally:
        if trainer is not None:
            trainer.close(finish_wandb=not slurm.received_exit_signal())
        slurm.cleanup_distributed()


if __name__ == "__main__":
    main()
