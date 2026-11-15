import copy
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, GenerationMixin, T5Config
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import (
    DEPARALLELIZE_DOCSTRING,
    PARALLELIZE_DOCSTRING,
    T5_START_DOCSTRING,
    T5PreTrainedModel,
    T5Stack,
    logger,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from grapht5.modeling.dual_cross_attn_t5 import DualCrossAttentionT5Stack
from grapht5.modeling.graph import VDGEncoder


@add_start_docstrings(
    """T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING
)
class T5ForConditionalGeneration(T5PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _keys_to_ignore_on_load_missing = [
        # Embedding ties (graph + decoder)
        r"encoder\.encoder\.embed_tokens\.weight",
        r"graph_encoder\.node_embeddings\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        # Cross-gates
        r"decoder\.block\.\d+\.layer\.1\.cross_gate",
        r"decoder\.block\.\d+\.layer\.1\.cross_gate_proj\.weight",
        r"graph_encoder\.vulnerability_role_emb\.weight",
        r"graph_encoder\.edge_type_bias\.weight",
        r"graph_encoder\.edge_distance_bias\.weight",
        # Graph branch q/k/v/o
        r"decoder\.block\.\d+\.layer\.1\.EncDecAttentionGraph\.(q|k|v|o)\.weight",
    ]
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "graph_encoder.node_embeddings.embed_tokens.weight",
        "lm_head.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = T5Stack(encoder_config, self.shared)

        graph_config = copy.deepcopy(encoder_config)
        self.add_consistency_loss = graph_config.add_consistency_loss
        self.drop_t5_passages = graph_config.drop_t5_passages

        # init graph encoder
        self.graph_encoder = VDGEncoder(
            config=graph_config,
            shared=self.shared,
            hidden_dim=config.d_model,
        )

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # self.decoder = T5Stack(decoder_config, self.shared)
        self.decoder = DualCrossAttentionT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()
        self.graph_encoder.init_role_embeddings(config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
        self.graph_encoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(
                self.graph_encoder.node_embeddings.embed_tokens, self.shared
            )

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # graph parameters
        node_ids: Optional[torch.LongTensor] = None,
        node_masks: Optional[torch.FloatTensor] = None,
        node_roles: Optional[torch.LongTensor] = None,
        edges: Optional[torch.LongTensor] = None,
        # reuse precomputed graph encodings (e.g., during generation)
        encoder_hidden_states_graph: Optional[torch.FloatTensor] = None,  # [B, Lg, H]
        encoder_attention_mask_graph: Optional[torch.FloatTensor] = None,  # [B, Lg]
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        consistency_loss = None

        # ---------------------- TEXT ENCODER ----------------------
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        text_hidden = encoder_outputs.last_hidden_state  # [B, Lt, H]
        text_mask_2d = attention_mask  # [B, Lt] or None

        # ---------------------- GRAPH ENCODER ----------------------
        graph_hidden = encoder_hidden_states_graph  # [B, Lg, H] (maybe None)
        graph_mask_2d = encoder_attention_mask_graph  # [B, Lg]   (maybe None)

        if graph_hidden is None and node_ids is not None:
            graph_outputs, graph_mask = self.graph_encoder(
                node_ids=node_ids,
                node_masks=node_masks,
                node_roles=node_roles,
                edges=edges,
            )

            B, N, S, H = graph_outputs.shape
            graph_hidden = graph_outputs.view(B, N * S, H)  # [B, Lg, H]
            graph_mask_2d = graph_mask.view(B, N * S)  # [B, Lg]

            if self.add_consistency_loss:
                consistency_loss = compute_token_diversity_loss(
                    graph_outputs, graph_mask
                )

        # ---------------------- DEVICE PLACEMENT ----------------------
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            text_hidden = text_hidden.to(self.decoder.first_device)
            if text_mask_2d is not None:
                text_mask_2d = text_mask_2d.to(self.decoder.first_device)
            if graph_hidden is not None:
                graph_hidden = graph_hidden.to(self.decoder.first_device)
            if graph_mask_2d is not None:
                graph_mask_2d = graph_mask_2d.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        text_available = (
            text_mask_2d.bool().any(dim=1)
            if text_mask_2d is not None
            else torch.ones(
                text_hidden.size(0), dtype=torch.bool, device=text_hidden.device
            )
        )
        graph_available = torch.zeros_like(text_available)

        if graph_hidden is not None and graph_mask_2d is not None:
            graph_available = graph_mask_2d.bool().any(dim=1)
            if graph_mask_2d.size(1) == 0:
                graph_hidden = None
                graph_mask_2d = None

        # Prepare decoder inputs
        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            decoder_input_ids = self._shift_right(labels)

        # Build 4D cross-attention masks (T5 expects inverted 4D bias masks)
        text_xattn_mask = (
            self.invert_attention_mask(text_mask_2d)
            if text_mask_2d is not None
            else None
        )
        graph_xattn_mask = (
            self.invert_attention_mask(graph_mask_2d)
            if graph_mask_2d is not None
            else None
        )

        # Optional: drop text completely if requested
        if getattr(self, "drop_t5_passages", False):
            text_hidden = None
            text_xattn_mask = None
            text_available.zero_()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states_text=text_hidden,
            encoder_attention_mask_text=text_xattn_mask,
            encoder_hidden_states_graph=graph_hidden,
            encoder_attention_mask_graph=graph_xattn_mask,
            encoder_available_text=text_available,
            encoder_available_graph=graph_available,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=False if labels is not None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
            if self.add_consistency_loss and consistency_loss is not None:
                loss = loss + consistency_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=(
                text_hidden if text_hidden is not None else graph_hidden
            ),
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor, model_kwargs, model_input_name, generation_config
    ):
        if "encoder_outputs" not in model_kwargs:
            # TEXT encoder
            encoder_kwargs = {
                "input_ids": inputs_tensor,
                "attention_mask": model_kwargs.get("attention_mask", None),
            }
            encoder_outputs = self.encoder(**encoder_kwargs)  # BaseModelOutput
            model_kwargs["encoder_outputs"] = encoder_outputs  # keep text as-is

            # GRAPH encoder (optional)
            if "node_ids" in model_kwargs:
                graph_outputs, graph_mask = self.graph_encoder(
                    node_ids=model_kwargs.get("node_ids"),
                    node_masks=model_kwargs.get("node_masks"),
                    node_roles=model_kwargs.get("node_roles"),
                    edges=model_kwargs.get("edges"),
                )

                B, N, S, H = graph_outputs.shape
                graph_hidden = graph_outputs.view(B, N * S, H)  # [B, Lg, H]
                graph_mask_2d = graph_mask.view(B, N * S)  # [B, Lg]

                # Stash for decoder; we'll invert inside forward
                model_kwargs["encoder_hidden_states_graph"] = graph_hidden
                model_kwargs["encoder_attention_mask_graph"] = graph_mask_2d

        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, attention_mask=attention_mask, **kwargs
        )
        model_inputs["attention_mask"] = attention_mask
        return model_inputs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        # validate graph inputs
        graph_args = [
            "node_ids",
            "node_masks",
            "node_roles",
            "edges",
        ]
        model_kwargs_copy = model_kwargs.copy()

        for arg in graph_args:
            model_kwargs_copy.pop(arg, None)

        # Call parent validation with remaining kwargs
        super()._validate_model_kwargs(model_kwargs_copy)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past


def compute_token_diversity_loss(
    graph_outputs, node_masks, intra_node_weight=4.0, inter_node_weight=4.0
):
    """
    Computes a loss that penalizes the model when tokens within a node are too similar
    and when aggregated node embeddings (inter-node tokens) are too similar.

    Args:
        graph_outputs (torch.Tensor): Tensor of shape [B, num_nodes * seq_len, hidden].
            It will be reshaped into [B, num_nodes, seq_len, hidden] assuming that the node
            structure is encoded in node_masks.
        node_masks (torch.Tensor): Binary mask of shape [B, num_nodes, seq_len] indicating which tokens
            in each node are valid.
        intra_node_weight (float): Weighting factor for the intra-node diversity loss.
        inter_node_weight (float): Weighting factor for the inter-node diversity loss.

    Returns:
        torch.Tensor: A scalar loss value. Higher loss means that tokens within nodes
            and aggregated node embeddings across nodes are too similar.
    """
    B, num_nodes, seq_len = node_masks.shape
    hidden = graph_outputs.size(-1)

    graph_outputs = graph_outputs.view(B, num_nodes, seq_len, hidden)

    intra_loss_total = 0.0
    intra_valid_node_count = 0

    aggregated_node_embeddings = []

    for b in range(B):
        batch_node_embs = []
        for n in range(num_nodes):
            valid_mask = node_masks[b, n].bool()
            tokens = graph_outputs[b, n][valid_mask]

            if tokens.size(0) <= 1:
                continue

            tokens_norm = F.normalize(tokens, p=2, dim=-1)
            sim_matrix = tokens_norm @ tokens_norm.T
            num_valid = tokens_norm.size(0)

            off_diag_mask = torch.ones_like(sim_matrix) - torch.eye(
                num_valid, device=sim_matrix.device
            )
            avg_similarity = (sim_matrix * off_diag_mask).sum() / (
                num_valid * (num_valid - 1)
            )
            intra_loss_total += avg_similarity
            intra_valid_node_count += 1

            aggregated_emb = tokens.mean(dim=0)
            batch_node_embs.append(aggregated_emb)

        if batch_node_embs:
            aggregated_node_embeddings.append(torch.stack(batch_node_embs, dim=0))
        else:
            aggregated_node_embeddings.append(None)

    if intra_valid_node_count == 0:
        return torch.tensor(0.0, device=graph_outputs.device)

    intra_diversity_loss = intra_loss_total / intra_valid_node_count

    inter_loss_total = 0.0
    inter_valid_count = 0

    for batch_node_embs in aggregated_node_embeddings:
        if batch_node_embs is None or batch_node_embs.size(0) <= 1:
            continue
        emb_norm = F.normalize(batch_node_embs, p=2, dim=-1)
        sim_matrix = emb_norm @ emb_norm.T
        num_nodes_valid = emb_norm.size(0)
        off_diag_mask = torch.ones_like(sim_matrix) - torch.eye(
            num_nodes_valid, device=sim_matrix.device
        )
        avg_similarity = (sim_matrix * off_diag_mask).sum() / (
            num_nodes_valid * (num_nodes_valid - 1)
        )
        inter_loss_total += avg_similarity
        inter_valid_count += 1

    if inter_valid_count > 0:
        inter_diversity_loss = inter_loss_total / inter_valid_count
    else:
        inter_diversity_loss = 0.0

    total_loss = (
        intra_node_weight * intra_diversity_loss
        + inter_node_weight * inter_diversity_loss
    )
    return total_loss


def compute_masked_distribution_consistency(
    graph_outputs,
    t5_outputs,
    node_masks,
    attention_mask,
    variance_weight=0.7,
    distribution_weight=0.5,
    range_weight=0.5,
):
    # validity check (>=20 tokens)
    t5_valid_count = attention_mask.sum(dim=1)
    node_valid_count = node_masks.reshape(node_masks.shape[0], -1).sum(dim=1)
    valid_samples = (t5_valid_count >= 20) & (node_valid_count >= 20)

    if not valid_samples.any():
        return torch.tensor(0.0, device=graph_outputs.device)

    batch_size, num_nodes, seq_len = node_masks.shape
    node_masks = node_masks.reshape(batch_size, -1).unsqueeze(-1)
    attention_mask = attention_mask.unsqueeze(-1)

    def compute_masked_statistics(x, mask):
        """
        Args:
            x: [batch, seq_len, hidden]
            mask: [batch, seq_len, 1]
        """
        mask = mask.expand(-1, -1, x.size(-1))  # [batch, seq_len, hidden]
        valid_values = x[mask.bool()].view(-1, x.size(-1))

        percentiles = torch.tensor(
            [0.1, 1.0, 10.0, 25.0, 50.0, 75.0, 90.0, 99.0, 99.9], device=x.device
        )
        stats = torch.quantile(valid_values, percentiles / 100.0, dim=0)

        # variance
        mean = valid_values.mean(dim=0)
        var = ((valid_values - mean) ** 2).mean(dim=0)

        # range
        value_range = valid_values.max(dim=0)[0] - valid_values.min(dim=0)[0]

        return stats, var, value_range

    graph_stats, graph_var, graph_range = compute_masked_statistics(
        graph_outputs, node_masks
    )
    t5_stats, t5_var, t5_range = compute_masked_statistics(t5_outputs, attention_mask)

    # distribution matching losses
    percentile_loss = F.mse_loss(graph_stats, t5_stats)
    range_loss = F.mse_loss(graph_range, t5_range)
    variance_loss = F.mse_loss(graph_var, t5_var)

    # outlier penalty (penalize values outside T5's range)
    def compute_outlier_loss(x, mask, t5_min, t5_max):
        """
        Args:
            x: [batch, seq_len, hidden]
            mask: [batch, seq_len, 1]
            t5_min: [hidden]
            t5_max: [hidden]
        """
        expanded_mask = mask.expand(-1, -1, x.size(-1))  # [batch, seq_len, hidden]

        # losses maintaining dimensions
        below_min = F.relu(
            t5_min.unsqueeze(0).unsqueeze(0) - x
        )  # [batch, seq_len, hidden]
        above_max = F.relu(
            x - t5_max.unsqueeze(0).unsqueeze(0)
        )  # [batch, seq_len, hidden]

        # mask and compute mean
        masked_below = (below_min * expanded_mask).sum() / (expanded_mask.sum() + 1e-9)
        masked_above = (above_max * expanded_mask).sum() / (expanded_mask.sum() + 1e-9)

        return masked_below + masked_above

    t5_min = t5_stats[0]  # 0.1 percentile
    t5_max = t5_stats[-1]  # 99.9 percentile
    outlier_loss = compute_outlier_loss(graph_outputs, node_masks, t5_min, t5_max)

    # cosine similarity
    graph_normalized = F.normalize(graph_outputs, p=2, dim=-1)
    t5_normalized = F.normalize(t5_outputs, p=2, dim=-1)

    graph_pooled = (graph_normalized * node_masks).sum(dim=1) / (
        node_masks.sum(dim=1) + 1e-9
    )
    t5_pooled = (t5_normalized * attention_mask).sum(dim=1) / (
        attention_mask.sum(dim=1) + 1e-9
    )

    similarity = F.cosine_similarity(graph_pooled, t5_pooled, dim=-1)
    cosine_loss = 1 - similarity[valid_samples].mean()

    # combine all losses
    distribution_loss = percentile_loss + range_weight * range_loss + outlier_loss

    total_loss = (
        cosine_loss
        + variance_weight * variance_loss
        + distribution_weight * distribution_loss
    )

    return total_loss
