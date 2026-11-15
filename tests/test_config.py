from pathlib import Path

import pytest

from grapht5.config.config import load_config, set_training_schedule


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_pretraining_matches_paper_schedule():
    opt = load_config(CONFIGS / "pretraining.toml")
    opt.world_size = 2
    set_training_schedule(opt, 458918)
    assert (opt.n_context, opt.graph_attention_layers, opt.freeze_t5) == (3, 6, 20000)
    assert (opt.total_steps, opt.scheduler_steps, opt.save_freq) == (917832, 28682, 28682)
    assert (opt.lr, opt.lr_vdg, opt.warmup_steps) == (0.0003, 0.0004, 625)


@pytest.mark.parametrize(
    ("dataset", "samples", "language"),
    [
        ("bigvul_cvefixes", 3584, "cpp"),
        ("icvul", 3892, "cpp"),
        ("primevul", 3712, "cpp"),
        ("vul4j", 5296, "java"),
        ("vul4c", 3678, "cpp"),
    ],
)
def test_finetuning_configs(dataset, samples, language):
    opt = load_config(CONFIGS / f"finetuning_{dataset}.toml")
    opt.world_size = 1
    set_training_schedule(opt, samples)
    epoch_steps = samples // 2
    assert (opt.total_steps, opt.scheduler_steps) == (20 * epoch_steps, 20 * epoch_steps // 32)
    assert (opt.eval_freq, opt.save_freq, opt.warmup_steps) == (epoch_steps, epoch_steps, epoch_steps // 32)
    assert (opt.n_context, opt.language, opt.lr, opt.lr_vdg) == (15, language, 0.0004, 0.0004)


def test_schedule_uses_actual_dataset_size():
    opt = load_config(CONFIGS / "finetuning_icvul.toml")
    opt.world_size = 2
    set_training_schedule(opt, 101)
    assert (opt.total_steps, opt.eval_freq, opt.save_freq) == (500, 25, 25)
