# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types

import torch
from torch import nn
from transformers import T5Config

from grapht5.modeling.graph_t5 import T5ForConditionalGeneration


class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].view(
                kwargs["input_ids"].size(0), -1
            )
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].view(
                kwargs["attention_mask"].size(0), -1
            )

        return super(FiDT5, self).forward(**kwargs)

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        Seq2SeqLMOutput = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        return Seq2SeqLMOutput

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs,
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self._init_graph_t5()
        self._init_decoder_graph_from_text()
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def overwrite_forward_crossattention(self):
        for mod in self.decoder.block:
            cross = mod.layer[1]
            attn_t = getattr(cross, "EncDecAttention", None)
            attn_g = getattr(cross, "EncDecAttentionGraph", None)
            if attn_t is not None:
                attn_t.forward = types.MethodType(cross_attention_forward, attn_t)
            if attn_g is not None:
                attn_g.forward = types.MethodType(cross_attention_forward, attn_g)

    def reset_score_storage(self):
        for mod in self.decoder.block:
            cross = mod.layer[1]
            cross.gate_storage = None
            for name in ("EncDecAttention", "EncDecAttentionGraph"):
                attn = getattr(cross, name, None)
                if attn is not None:
                    attn.score_storage = None

    def get_crossattention_scores(self, text_mask_2d, node_mask_2d):
        if text_mask_2d is not None:
            text_mask_2d = text_mask_2d.view(text_mask_2d.size(0), -1)
        if node_mask_2d is not None:
            node_mask_2d = node_mask_2d.view(node_mask_2d.size(0), -1)
        # masks: [B, Lt] and [B, Lg]
        B = text_mask_2d.size(0) if text_mask_2d is not None else node_mask_2d.size(0)
        text_total = torch.zeros(B, device=self.device)
        graph_total = torch.zeros(B, device=self.device)

        for mod in self.decoder.block:
            cross = mod.layer[1]
            gate = getattr(cross, "gate_storage", None)
            if gate is None:
                gate = self.decoder.cross_attention_gate(cross, text_total)
            elif gate.ndim == 3:
                gate = gate.unsqueeze(1)
            attn_t = getattr(cross.EncDecAttention, "score_storage", None)
            attn_g = getattr(cross.EncDecAttentionGraph, "score_storage", None)

            if attn_t is not None:
                w_t = torch.softmax(attn_t.float(), dim=-1)  # [B,H,Q,Lt]
                if text_mask_2d is not None:
                    w_t = w_t * text_mask_2d[:, None, None, :].to(w_t.dtype)
                text_total = text_total + (gate * w_t).sum(dim=(1, 2, 3))

            if attn_g is not None:
                w_g = torch.softmax(attn_g.float(), dim=-1)  # [B,H,Q,Lg]
                if node_mask_2d is not None:
                    w_g = w_g * node_mask_2d[:, None, None, :].to(w_g.dtype)
                graph_total = graph_total + ((1 - gate) * w_g).sum(dim=(1, 2, 3))

        denom = (text_total + graph_total).clamp_min(1e-9)
        return text_total / denom, graph_total / denom

    def _init_graph_t5(self):
        num_graph_layers = self.graph_encoder.num_layers
        t5_layers = self.encoder.config.num_layers

        assert t5_layers == 2 * num_graph_layers

        with torch.no_grad():
            for i in range(num_graph_layers):
                pretrained_block_1 = self.encoder.block[2 * i]
                pretrained_block_2 = self.encoder.block[2 * i + 1]

                src_ff_1 = pretrained_block_1.layer[-1]
                src_sa_2 = pretrained_block_2.layer[0]
                src_ff_2 = pretrained_block_2.layer[-1]

                dest_ff1 = self.graph_encoder.ff1_layers[i]
                dest_ff1.DenseReluDense.wi.weight.data.copy_(
                    src_ff_1.DenseReluDense.wi.weight.data
                )
                dest_ff1.DenseReluDense.wo.weight.data.copy_(
                    src_ff_1.DenseReluDense.wo.weight.data
                )
                dest_ff1.layer_norm.weight.data.copy_(src_ff_1.layer_norm.weight.data)

                dest_sa = self.graph_encoder.self_attention_layers[i]
                dest_sa.SelfAttention.q.weight.data.copy_(
                    src_sa_2.SelfAttention.q.weight.data
                )
                dest_sa.SelfAttention.k.weight.data.copy_(
                    src_sa_2.SelfAttention.k.weight.data
                )
                dest_sa.SelfAttention.v.weight.data.copy_(
                    src_sa_2.SelfAttention.v.weight.data
                )
                dest_sa.SelfAttention.o.weight.data.copy_(
                    src_sa_2.SelfAttention.o.weight.data
                )
                if (
                    src_sa_2.SelfAttention.has_relative_attention_bias
                    and dest_sa.SelfAttention.has_relative_attention_bias
                ):
                    dest_sa.SelfAttention.relative_attention_bias.weight.data.copy_(
                        src_sa_2.SelfAttention.relative_attention_bias.weight.data
                    )
                dest_sa.layer_norm.weight.data.copy_(src_sa_2.layer_norm.weight.data)

                dest_ff2 = self.graph_encoder.ff2_layers[i]
                dest_ff2.DenseReluDense.wi.weight.data.copy_(
                    src_ff_2.DenseReluDense.wi.weight.data
                )
                dest_ff2.DenseReluDense.wo.weight.data.copy_(
                    src_ff_2.DenseReluDense.wo.weight.data
                )
                dest_ff2.layer_norm.weight.data.copy_(src_ff_2.layer_norm.weight.data)

            self.graph_encoder.self_attention_layers[
                0
            ].SelfAttention.relative_attention_bias.weight.data.copy_(
                self.encoder.block[0]
                .layer[0]
                .SelfAttention.relative_attention_bias.weight.data
            )
            self.graph_encoder.final_layer_norm.weight.data.copy_(
                self.encoder.final_layer_norm.weight.data
            )

    def _init_decoder_graph_from_text(self):
        with torch.no_grad():
            for blk in self.decoder.block:
                cross = blk.layer[1]
                src = getattr(cross, "EncDecAttention", None)
                dst = cross.EncDecAttentionGraph

                for name in ("q", "k", "v", "o"):
                    getattr(dst, name).weight.copy_(getattr(src, name).weight)

                cross.cross_gate.data.zero_()
                cross.cross_gate_proj.weight.zero_()


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)
        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)

        # rmeove graph-related kwargs before passing to T5Stack
        t5_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "node_ids",
                "node_masks",
                "node_roles",
                "edges",
            ]
        }

        outputs = self.encoder(input_ids, attention_mask, **t5_kwargs)
        outputs["last_hidden_state"] = outputs[0].view(
            bsz, self.n_passages * passage_length, -1
        )
        return outputs

    @property
    def embed_tokens(self):
        return self.encoder.embed_tokens

    @embed_tokens.setter
    def embed_tokens(self, value):
        self.encoder.embed_tokens = value


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [], dtype=torch.float, device=output[0].device, requires_grad=True
                )
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    cache_position=None,
):
    """
    Overwrite for T5Attention cross-attention.

    This is essentially T5's standard `forward(...)` from `T5Attention`, with one extra line:
        if is_cross_attention and (not hasattr(self, "score_storage") or self.score_storage is None):
            self.score_storage = scores
    placed after the dot-product `scores = torch.matmul(query_states, key_states.transpose(3, 2))`.

    This ensures we capture the raw cross-attention logits for later inspection,
    without breaking T5's caching logic or parameter naming.
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
    batch_size, seq_length = hidden_states.shape[:2]

    # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
    is_cross_attention = key_value_states is not None

    query_states = self.q(hidden_states)
    query_states = query_states.view(
        batch_size, -1, self.n_heads, self.key_value_proj_dim
    ).transpose(1, 2)

    if past_key_value is not None:
        is_updated = past_key_value.is_updated.get(self.layer_idx)
        if is_cross_attention:
            # after the first generated id, we can subsequently re-use all key/value_states from cache
            curr_past_key_value = past_key_value.cross_attention_cache
        else:
            curr_past_key_value = past_key_value.self_attention_cache

    current_states = key_value_states if is_cross_attention else hidden_states
    if is_cross_attention and past_key_value is not None and is_updated:
        # reuse k,v, cross_attentions
        key_states = curr_past_key_value.key_cache[self.layer_idx]
        value_states = curr_past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self.k(current_states)
        value_states = self.v(current_states)
        key_states = key_states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                {"cache_position": cache_position},
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True

    # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
    scores = torch.matmul(query_states, key_states.transpose(3, 2))

    if position_bias is None:
        key_length = key_states.shape[-2]
        # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
        real_seq_length = (
            query_length if query_length is not None else cache_position[-1] + 1
        )
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, seq_length, key_length),
                device=scores.device,
                dtype=scores.dtype,
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(
                real_seq_length,
                key_length,
                device=scores.device,
                cache_position=cache_position,
            )
            position_bias = position_bias[:, :, -seq_length:, :]

        if mask is not None:
            causal_mask = mask[:, :, :, : key_states.shape[-2]]
            position_bias = position_bias + causal_mask

    if self.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    scores += position_bias_masked

    if is_cross_attention and (
        not hasattr(self, "score_storage") or self.score_storage is None
    ):
        self.score_storage = scores

    # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, -1, self.inner_dim)
    attn_output = self.o(attn_output)

    outputs = (attn_output, past_key_value, position_bias)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs
