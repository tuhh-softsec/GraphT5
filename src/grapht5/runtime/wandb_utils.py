import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


def get_commit_sha() -> str:
    commit_sha = os.environ.get("GRAPHT5_COMMIT_SHA")
    if commit_sha:
        return commit_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class WandbLogger:
    def __init__(self, opt):
        self.opt = opt
        self.use_wandb = opt.use_wandb and WANDB_AVAILABLE
        self.run = None

        if not WANDB_AVAILABLE and opt.use_wandb:
            logger.warning("wandb not installed. Install with: pip install wandb")
            return

        if self.use_wandb:
            self._init_wandb()

    @property
    def enabled(self):
        return bool(self.use_wandb and self.run and self.opt.is_main)

    def _init_wandb(self):
        """Initialize wandb run"""
        if not self.opt.is_main:
            return  # only initialize on main process

        # set up run configuration
        config = {
            # training hyperparameters
            "lr": self.opt.lr,
            "lr_vdg": self.opt.lr_vdg,
            "batch_size": self.opt.per_gpu_train_batch_size,
            "total_steps": self.opt.total_steps,
            "warmup_steps": self.opt.warmup_steps,
            "eval_freq": self.opt.eval_freq,
            "dropout": self.opt.dropout,
            "weight_decay": self.opt.weight_decay,
            "weight_decay_vdg": self.opt.weight_decay_vdg,
            "clip": self.opt.clip,
            "scheduler": self.opt.scheduler,
            # model architecture
            "model_size": self.opt.model_size,
            "base_model": self.opt.base_model,
            "text_maxlength": self.opt.text_maxlength,
            "answer_maxlength": self.opt.answer_maxlength,
            "node_max_length": self.opt.node_max_length,
            "max_nodes_per_graph": self.opt.max_nodes_per_graph,
            # graph parameters
            "graph_attention_layers": self.opt.graph_attention_layers,
            "no_learned_gate": self.opt.no_learned_gate,
            "t5_unfreeze_lr": self.opt.t5_unfreeze_lr,
            "t5_unfreeze_warmup_steps": self.opt.t5_unfreeze_warmup_steps,
            # other settings
            "n_context": self.opt.n_context,
            "accumulation_steps": self.opt.accumulation_steps,
            "seed": self.opt.seed,
            "git_commit_sha": get_commit_sha(),
        }

        # initialize wandb
        self.run = wandb.init(
            project=self.opt.wandb_project,
            entity=self.opt.wandb_entity,
            name=self.opt.wandb_run_name or self.opt.name,
            tags=self.opt.wandb_tags,
            config=config,
            dir=str(self.opt.checkpoint_dir),
            resume=(
                "allow"
                if hasattr(self.opt, "model_path") and self.opt.model_path != "none"
                else None
            ),
        )

        logger.info(f"Initialized wandb run: {self.run.name}")

    def watch_model(self, model):
        """Watch model parameters and gradients"""
        if self.use_wandb and self.run and self.opt.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=self.opt.wandb_log_freq)

    def log_metrics(self, metrics: Dict[str, Any], step=None):
        """Log metrics to wandb"""
        if self.use_wandb and self.run and self.opt.is_main:
            wandb.log(metrics, step=step)

    def log_metrics_all_processes(self, metrics: Dict[str, Any], step: int):
        """Log metrics to wandb from all processes (not just main)"""
        if self.use_wandb and self.run:
            wandb.log(metrics, step=step)

    def log_parameters(self, model, step: int):
        """Log all Parameter modules in the model"""
        if not (self.use_wandb and self.run and self.opt.is_main):
            return

        param_stats = {}

        # get the actual model if wrapped in DDP
        actual_model = model.module if hasattr(model, "module") else model

        for name, param in actual_model.named_parameters():
            if isinstance(param, nn.Parameter) and param.requires_grad:
                param_data = param.data.detach().cpu()

                # basic statistics
                param_stats[f"params/{name}/mean"] = param_data.mean().item()
                param_stats[f"params/{name}/std"] = param_data.std().item()
                param_stats[f"params/{name}/norm"] = param_data.norm().item()
                param_stats[f"params/{name}/min"] = param_data.min().item()
                param_stats[f"params/{name}/max"] = param_data.max().item()

                # gradient statistics if available
                if param.grad is not None:
                    grad_data = param.grad.detach().cpu()
                    param_stats[f"gradients/{name}/mean"] = grad_data.mean().item()
                    param_stats[f"gradients/{name}/std"] = grad_data.std().item()
                    param_stats[f"gradients/{name}/norm"] = grad_data.norm().item()
                    param_stats[f"gradients/{name}/min"] = grad_data.min().item()
                    param_stats[f"gradients/{name}/max"] = grad_data.max().item()

        self.log_metrics(param_stats, step)

    def log_parameter_counts(self, model, step: int):
        """Log counts of frozen vs unfrozen parameters"""
        if not (self.use_wandb and self.run and self.opt.is_main):
            return

        frozen_params = 0
        unfrozen_params = 0
        total_params = 0

        # get the actual model if wrapped in DDP
        actual_model = model.module if hasattr(model, "module") else model

        for name, param in actual_model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                unfrozen_params += param.numel()
            else:
                frozen_params += param.numel()

        param_counts = {
            "params/total_parameters": total_params,
            "params/frozen_parameters": frozen_params,
            "params/unfrozen_parameters": unfrozen_params,
            "params/frozen_ratio": frozen_params / max(1, total_params),
            "params/unfrozen_ratio": unfrozen_params / max(1, total_params),
        }

        self.log_metrics(param_counts, step)

    def plot_cross_attention_gates(self, model, step: int):
        """Create and log cross attention gates for text and graph encoders"""
        if not (self.use_wandb and self.run and self.opt.is_main):
            return

        if step % self.opt.wandb_plot_freq != 0:
            return

        try:
            # get the actual model if wrapped in DDP
            actual_model = model.module if hasattr(model, "module") else model

            plots = {}

            # cross-attention gate values
            if hasattr(actual_model, "decoder"):
                gate_values = self._extract_cross_attention_gates(actual_model.decoder)
                if gate_values is not None:
                    plots["cross_attention_gate_baselines"] = self._create_gate_plot(
                        gate_values
                    )

            # log plots
            self.log_metrics(plots, step)

        except Exception as e:
            logger.warning(f"Failed to create cross-attention gates: {e}")

    def plot_encoder_heatmaps(self, model, step: int):
        """Create and log heatmap visualizations for text and graph encoders"""
        if not (self.use_wandb and self.run and self.opt.is_main):
            return

        if step % self.opt.wandb_plot_freq != 0:
            return

        try:
            # get the actual model if wrapped in DDP
            actual_model = model.module if hasattr(model, "module") else model

            plots = {}

            # text encoder attention weights (if available)
            if hasattr(actual_model, "encoder") and hasattr(
                actual_model.encoder, "encoder"
            ):
                text_encoder = actual_model.encoder.encoder
                attention_weights = self._extract_attention_weights(
                    text_encoder, "text_encoder"
                )
                if attention_weights is not None:
                    plots["text_encoder_attention"] = self._create_attention_heatmap(
                        attention_weights, "Text Encoder Attention"
                    )

            # cross-attention gate values
            if hasattr(actual_model, "decoder"):
                gate_values = self._extract_cross_attention_gates(actual_model.decoder)
                if gate_values is not None:
                    plots["cross_attention_gate_baselines"] = self._create_gate_plot(
                        gate_values
                    )

            # log plots
            self.log_metrics(plots, step)

        except Exception as e:
            logger.warning(f"Failed to create encoder heatmaps: {e}")

    def _extract_attention_weights(self, encoder, encoder_type: str):
        """Extract attention weights from encoder layers"""
        try:
            attention_weights = []

            if encoder_type == "text_encoder":
                # extract from T5 blocks
                for i, block in enumerate(encoder.block):
                    if hasattr(block, "layer") and len(block.layer) > 0:
                        self_attn = block.layer[0]
                        if hasattr(self_attn, "SelfAttention") and hasattr(
                            self_attn.SelfAttention, "relative_attention_bias"
                        ):
                            # get relative attention bias as proxy for attention pattern
                            bias = (
                                self_attn.SelfAttention.relative_attention_bias.weight.data.cpu().numpy()
                            )
                            attention_weights.append(bias)

            return attention_weights if attention_weights else None

        except Exception as e:
            logger.warning(
                f"Failed to extract attention weights from {encoder_type}: {e}"
            )
            return None

    def _extract_cross_attention_gates(self, decoder):
        """Extract scalar cross-attention gate baselines from decoder"""
        try:
            gate_values = []

            for block in decoder.block:
                if hasattr(block, "layer") and len(block.layer) > 1:
                    cross_attn_layer = block.layer[1]
                    if hasattr(cross_attn_layer, "cross_gate"):

                        gate_val = torch.sigmoid(cross_attn_layer.cross_gate).item()
                        gate_values.append(gate_val)

            return gate_values if gate_values else None

        except Exception as e:
            logger.warning(f"Failed to extract cross-attention gates: {e}")
            return None

    def _create_attention_heatmap(self, attention_weights, title: str):
        """Create heatmap visualization of attention weights"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, weights in enumerate(attention_weights[:6]):  # show first 6 layers
                ax = axes[i]

                # normalize weights for visualization
                weights_norm = (weights - weights.min()) / (
                    weights.max() - weights.min() + 1e-8
                )

                sns.heatmap(weights_norm, ax=ax, cmap="viridis", cbar=True)
                ax.set_title(f"Layer {i+1}")
                ax.set_xlabel("Key Position")
                ax.set_ylabel("Query Position")

            # hide unused subplots
            for i in range(len(attention_weights), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(title)
            plt.tight_layout()

            # convert to wandb Image
            wandb_image = wandb.Image(fig)
            plt.close(fig)

            return wandb_image

        except Exception as e:
            logger.warning(f"Failed to create attention heatmap: {e}")
            return None

    def _create_gate_plot(self, gate_values):
        """Create plot of scalar cross-attention gate baselines"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            layers = list(range(1, len(gate_values) + 1))
            ax.plot(layers, gate_values, "o-", linewidth=2, markersize=8)
            ax.set_xlabel("Decoder Layer")
            ax.set_ylabel("Baseline Gate Value")
            ax.set_title("Cross-Attention Gate Baselines (0=Graph, 1=Text)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

            # add horizontal lines at 0.5 for reference
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Balanced")
            ax.legend()

            plt.tight_layout()

            # convert to wandb Image
            wandb_image = wandb.Image(fig)
            plt.close(fig)

            return wandb_image

        except Exception as e:
            logger.warning(f"Failed to create gate plot: {e}")
            return None

    def finish(self):
        """Finish the wandb run"""
        if self.use_wandb and self.run:
            wandb.finish()


def get_wandb_logger(opt) -> WandbLogger:
    """Get wandb logger instance"""
    return WandbLogger(opt)
