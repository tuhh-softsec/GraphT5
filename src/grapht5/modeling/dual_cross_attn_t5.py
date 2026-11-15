import warnings
import torch
from torch import nn

from transformers.utils import (
    is_torchdynamo_compiling,
    logging,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerNorm,
    T5PreTrainedModel,
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    StaticCache,
    T5Config,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

logger = logging.get_logger(__name__)


class DualCrossAttentionT5Stack(T5PreTrainedModel):
    """
    Drop-in replacement for transformers.models.t5.modeling_t5.T5Stack (decoder side)
    that performs two cross-attention branches (text & graph) and mixes them via
    a token-conditioned gate per layer.

    Notes:
    - Self-attention cache is preserved (fast generation).
    - Cross-attention for text & graph does *not* write to the standard cross cache,
      since HF cache has a single cross-attn slot per layer. We recompute cross K/V.
    - Cross-attention weights returned are those from the *text* branch for API compatibility.
    """

    def __init__(self, config: T5Config, embed_tokens: nn.Embedding):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.use_learned_gate = not getattr(config, "no_learned_gate", False)
        assert self.is_decoder, "DualCrossAttentionT5Stack is intended for the decoder."

        # build standard blocks, then inject dual cross-attn modules
        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i)
                for i in range(config.num_layers)
            ]
        )

        # final norm/dropout identical to vanilla
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # inject graph cross-attn + gate per layer
        for i, blk in enumerate(self.block):
            # t5Block.layer[1] exists only for decoder
            cross = blk.layer[1]
            # clone attention module for the graph branch
            # encDecAttentionText = cross.EncDecAttention
            EncDecAttentionGraph = type(cross.EncDecAttention)(
                config, has_relative_attention_bias=False, layer_idx=i
            )

            # cross.EncDecAttentionText = EncDecAttentionText
            cross.EncDecAttentionGraph = EncDecAttentionGraph

            # The scalar preserves legacy checkpoints; the projection adds token routing.
            cross.cross_gate = nn.Parameter(torch.tensor(-0.85))
            cross.cross_gate_proj = nn.Linear(3 * config.d_model, 1, bias=False)

        # initialize weights and apply final processing
        self.post_init()
        for blk in self.block:
            nn.init.zeros_(blk.layer[1].cross_gate_proj.weight)

        # model parallel flags
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    # -------- model parallel parity with HF T5Stack ----------

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated; prefer `from_pretrained(..., device_map='balanced')`.",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # place blocks
        for k, v in self.device_map.items():
            for layer_idx in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer_idx] = self.block[layer_idx].to(cuda_device)
        # embeddings on first
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # final norm on last
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    # -------- embeddings parity ----------

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # --------- forward (mirrors HF T5Stack, but dual cross-attn) ---------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states_text=None,
        encoder_attention_mask_text=None,
        encoder_hidden_states_graph=None,
        encoder_attention_mask_graph=None,
        encoder_available_text=None,
        encoder_available_graph=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        # model parallel: ensure embeds on first device
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # validate inputs/embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You must specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # gradient checkpointing disables caching
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting to False."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("Decoder must have valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True and not self.is_decoder:
            raise ValueError("`use_cache` can only be True if used as a decoder")

        # ---- init cache wrapper to new API (Cache/EncoderDecoderCache) ----
        return_legacy_cache = False
        return_self_attention_cache = False
        if self.is_decoder and (use_cache or past_key_values is not None):
            if isinstance(past_key_values, Cache) and not isinstance(
                past_key_values, EncoderDecoderCache
            ):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Tuple `past_key_values` is deprecated; pass an EncoderDecoderCache instead."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
        elif not self.is_decoder:
            past_key_values = None

        past_key_values_length = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                device=inputs_embeds.device,
            )

        # build self-attention mask / causal
        if attention_mask is None and not is_torchdynamo_compiling():
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                (
                    past_key_values.self_attention_cache
                    if past_key_values is not None
                    else None
                ),
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # prepare head masks
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )

        # outputs to collect
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        position_bias = None
        # separate encoder-decoder position biases for text and graph
        encoder_decoder_position_bias_text = None
        encoder_decoder_position_bias_graph = None

        hidden_states = self.dropout(inputs_embeds)
        next_decoder_cache = (
            past_key_values if use_cache else None
        )  # we’ll pass/receive the same cache object

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            # model-parallel device hops
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states_text is not None:
                    encoder_hidden_states_text = encoder_hidden_states_text.to(
                        hidden_states.device
                    )
                if encoder_hidden_states_graph is not None:
                    encoder_hidden_states_graph = encoder_hidden_states_graph.to(
                        hidden_states.device
                    )
                if encoder_attention_mask_text is not None:
                    encoder_attention_mask_text = encoder_attention_mask_text.to(
                        hidden_states.device
                    )
                if encoder_attention_mask_graph is not None:
                    encoder_attention_mask_graph = encoder_attention_mask_graph.to(
                        hidden_states.device
                    )
                if encoder_available_text is not None:
                    encoder_available_text = encoder_available_text.to(
                        hidden_states.device
                    )
                if encoder_available_graph is not None:
                    encoder_available_graph = encoder_available_graph.to(
                        hidden_states.device
                    )
                if encoder_decoder_position_bias_text is not None:
                    encoder_decoder_position_bias_text = (
                        encoder_decoder_position_bias_text.to(hidden_states.device)
                    )
                if encoder_decoder_position_bias_graph is not None:
                    encoder_decoder_position_bias_graph = (
                        encoder_decoder_position_bias_graph.to(hidden_states.device)
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device
                    )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # gradient checkpointing path: no cache is passed down
            if self.gradient_checkpointing and self.training:

                def custom_forward(*inputs):
                    return self._layer_forward_dual_cross(
                        layer_module,
                        *inputs,
                        output_attentions,
                        use_cache=False,  # no cache under checkpointing
                    )

                layer_outputs = self._gradient_checkpointing_func(
                    custom_forward,
                    hidden_states,
                    causal_mask,
                    position_bias,
                    encoder_hidden_states_text,
                    encoder_attention_mask_text,
                    encoder_decoder_position_bias_text,
                    encoder_hidden_states_graph,
                    encoder_attention_mask_graph,
                    encoder_available_text,
                    encoder_available_graph,
                    encoder_decoder_position_bias_graph,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    cache_position,
                )
            else:
                layer_outputs = self._layer_forward_dual_cross(
                    layer_module,
                    hidden_states,
                    causal_mask,
                    position_bias,
                    encoder_hidden_states_text,
                    encoder_attention_mask_text,
                    encoder_decoder_position_bias_text,
                    encoder_hidden_states_graph,
                    encoder_attention_mask_graph,
                    encoder_available_text,
                    encoder_available_graph,
                    encoder_decoder_position_bias_graph,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    cache_position,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    past_key_values=next_decoder_cache,  # cache object
                )

            # layer_outputs tuple structure:
            # when output_attentions=True:
            #   0: hidden_states
            #   1: updated_cache (or None)
            #   2: self-attn position_bias
            #   3: self-attn weights
            #   4: enc-dec position bias (TEXT)
            #   5: cross-attn weights (TEXT)
            #   6: enc-dec position bias (GRAPH)
            #   7: None (graph attn weights placeholder)
            # when output_attentions=False:
            #   0: hidden_states
            #   1: updated_cache (or None)
            #   2: self-attn position_bias
            #   3: enc-dec position bias (TEXT) <- shifted!
            #   4: None (cross-attn weights placeholder)
            #   5: enc-dec position bias (GRAPH) <- shifted!
            #   6: None
            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[1] if use_cache else next_decoder_cache
            position_bias = layer_outputs[2]

            if output_attentions:
                # when attention weights are included in tuple
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder and (
                    encoder_hidden_states_text is not None
                    or encoder_hidden_states_graph is not None
                ):
                    encoder_decoder_position_bias_text = layer_outputs[4]
                    # for API compatibility, return the TEXT cross-attn weights
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
                    encoder_decoder_position_bias_graph = layer_outputs[6]
            else:
                # when no attention weights, position biases are at different indices
                if self.is_decoder and (
                    encoder_hidden_states_text is not None
                    or encoder_hidden_states_graph is not None
                ):
                    encoder_decoder_position_bias_text = layer_outputs[3]
                    encoder_decoder_position_bias_graph = layer_outputs[5]

            # model-parallel: hop to next device if needed
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # final norm/dropout
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        # if isinstance(next_cache, EncoderDecoderCache):
        #     if next_cache is not None and isinstance(next_cache.self_attention_cache, Cache) and not isinstance(next_cache, EncoderDecoderCache):
        #         # parity with upstream (rare path)
        #         pass
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    # ---- one-layer forward with dual cross-attn (mirrors T5Block but split branches) ----

    def _layer_forward_dual_cross(
        self,
        layer_module: T5Block,
        hidden_states,
        attention_mask,
        position_bias,
        encoder_hidden_states_text,
        encoder_attention_mask_text,
        encoder_decoder_position_bias_text,
        encoder_hidden_states_graph,
        encoder_attention_mask_graph,
        encoder_available_text,
        encoder_available_graph,
        encoder_decoder_position_bias_graph,
        layer_head_mask,
        cross_attn_layer_head_mask,
        cache_position,
        output_attentions=False,
        use_cache=False,
        past_key_values: EncoderDecoderCache = None,
    ):
        # === Self-attention (same as vanilla) ===
        self_attn_outputs = layer_module.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_values,  # hF cache object
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states, updated_cache = self_attn_outputs[:2]
        position_bias = self_attn_outputs[2]

        # fp16 clamp parity
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        # === Cross-attention (dual) ===
        cross_text_weights = None
        # only run cross-attn if we have at least one source
        if self.is_decoder and (
            encoder_hidden_states_text is not None
            or encoder_hidden_states_graph is not None
        ):
            cross_layer = layer_module.layer[1]

            # layerNorm pre-attn per T5
            normed_hidden = cross_layer.layer_norm(hidden_states)

            # text branch (if present)
            if encoder_hidden_states_text is not None:
                # t5Attention API expects query_length or cache_position for bias slicing
                cross_text = cross_layer.EncDecAttention(
                    normed_hidden,
                    mask=encoder_attention_mask_text,
                    key_value_states=encoder_hidden_states_text,
                    position_bias=encoder_decoder_position_bias_text,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=None,  # do NOT touch cross cache (two branches share one slot)
                    use_cache=False,
                    query_length=(
                        cache_position[-1] + 1 if cache_position is not None else None
                    ),
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )
                cross_text_out = cross_text[0]
                encoder_decoder_position_bias_text = cross_text[2]
                if output_attentions:
                    cross_text_weights = cross_text[3]
            else:
                cross_text_out = torch.zeros_like(normed_hidden)

            # graph branch (if present)
            if encoder_hidden_states_graph is not None:
                cross_graph = cross_layer.EncDecAttentionGraph(
                    normed_hidden,
                    mask=encoder_attention_mask_graph,
                    key_value_states=encoder_hidden_states_graph,
                    position_bias=encoder_decoder_position_bias_graph,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=None,  # do NOT touch cross cache
                    use_cache=False,
                    query_length=(
                        cache_position[-1] + 1 if cache_position is not None else None
                    ),
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )
                cross_graph_out = cross_graph[0]
                encoder_decoder_position_bias_graph = cross_graph[2]
                # we don't return graph weights to keep API identical (one tensor per layer)
            else:
                cross_graph_out = torch.zeros_like(normed_hidden)

            gate = self.cross_attention_gate(
                cross_layer,
                normed_hidden,
                cross_text_out,
                cross_graph_out,
                encoder_available_text,
                encoder_available_graph,
            )
            if (
                hasattr(cross_layer, "gate_storage")
                and cross_layer.gate_storage is None
            ):
                cross_layer.gate_storage = gate.detach()
            mixed = gate * cross_text_out + (1.0 - gate) * cross_graph_out

            # residual + dropout
            hidden_states = hidden_states + cross_layer.dropout(mixed)

            # fp16 clamp parity
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

        # === Feed-forward (vanilla) ===
        hidden_states = layer_module.layer[-1](hidden_states)

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (
            hidden_states,
            updated_cache if use_cache else None,
            position_bias,
        )

        if output_attentions:
            # self-attn weights at index 3 (same as vanilla)
            outputs = outputs + (self_attn_outputs[3],)

        # cross pos-bias & weights (text first to keep API shape stable)
        outputs = outputs + (
            encoder_decoder_position_bias_text,
            cross_text_weights if output_attentions else None,
            encoder_decoder_position_bias_graph,
            None,  # we do not return graph attn weights to preserve HF API expectations
        )
        return outputs

    def cross_attention_gate(
        self,
        cross_layer,
        reference_tensor,
        cross_text_out=None,
        cross_graph_out=None,
        text_available=None,
        graph_available=None,
    ):
        if self.use_learned_gate:
            if cross_text_out is None or cross_graph_out is None:
                gate = torch.sigmoid(cross_layer.cross_gate)
            else:
                gate_input = torch.cat(
                    (reference_tensor, cross_text_out, cross_graph_out), dim=-1
                )
                gate = torch.sigmoid(
                    cross_layer.cross_gate + cross_layer.cross_gate_proj(gate_input)
                )
        else:
            gate = reference_tensor.new_tensor(0.5)
        if text_available is None:
            return gate
        if gate.ndim == 0:
            gate = gate.expand(*reference_tensor.shape[:-1], 1)
        shape = (text_available.size(0),) + (1,) * (gate.ndim - 1)
        gate = torch.where(
            (text_available & ~graph_available).view(shape),
            torch.ones_like(gate),
            gate,
        )
        return torch.where(
            (graph_available & ~text_available).view(shape),
            torch.zeros_like(gate),
            gate,
        )

    # ---- causal mask helpers (copied from HF T5Stack) ----

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask
