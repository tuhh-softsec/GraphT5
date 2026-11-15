import argparse
import tomllib
from pathlib import Path
from types import SimpleNamespace

from loguru import logger


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _read(path):
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    parent = config.pop("extends", None)
    return _merge(_read(path.parent / parent), config) if parent else config


def load_config(path, **overrides):
    config = _read(Path(path).resolve())
    values = {}
    for section in ("run", "data", "model", "optimization", "training", "evaluation", "logging"):
        values.update({
            key: value for key, value in config[section].items()
            if not isinstance(value, dict)
        })
    for section in ("text", "graph"):
        values.update(config["model"][section])
    values.update({key: value for key, value in overrides.items() if value is not None})
    values.setdefault("passage_dropout", None)

    values["base_model"] = "Salesforce/codet5-" + values["model_size"]
    values["eval_metric_display_name"] = (
        "EM + CodeBLEU" if values["with_em"] and values["with_codebleu"]
        else "EM" if values["with_em"]
        else "CodeBLEU" if values["with_codebleu"] else "none"
    )
    return SimpleNamespace(**values)


def set_training_schedule(opt, sample_count):
    steps_per_epoch = sample_count // (opt.per_gpu_train_batch_size * opt.world_size)
    opt.total_steps = opt.epochs * steps_per_epoch
    opt.scheduler_steps = opt.total_steps // opt.accumulation_steps
    if opt.schedule == "finetune":
        opt.warmup_steps = steps_per_epoch // opt.accumulation_steps
        opt.eval_freq = steps_per_epoch
        opt.save_freq = steps_per_epoch
    else:
        opt.save_freq = steps_per_epoch // 4


def parse_config():
    parser = argparse.ArgumentParser(description="Run GraphT5 with a TOML configuration")
    parser.add_argument("--config", type=Path, required=True)
    for name in ("train_data", "eval_data", "model_path", "checkpoint_dir", "name", "language"):
        parser.add_argument("--" + name.replace("_", "-"))
    for name in ("beam_size", "num_return_sequences"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--write-results", action="store_true", default=None)
    args = vars(parser.parse_args())
    path = args.pop("config")
    return load_config(path, **args)


def log_config(opt):
    logger.info("Configuration: {}", vars(opt))
