import torch
import torch.nn as nn
from torch.compiler import disable
from transformers.models.t5.modeling_t5 import (
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
)

from grapht5.runtime.util import EDGE_TYPES, relative_position_bucket

EDGE_DISTANCE_BUCKETS = 32
EDGE_DISTANCE_MAX = 1024


class VDGEncoder(nn.Module):
    def __init__(self, config, shared, hidden_dim):
        super(VDGEncoder, self).__init__()

        self.num_layers = config.graph_attention_layers
        self.dropout_rate = config.dropout_rate
        self.hidden_dim = hidden_dim
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.disable_edge_attention = getattr(config, "disable_edge_attention", False)
        self.node_embeddings = T5NodeEmbedder(shared, dropout=config.dropout_rate)
        self.vulnerability_role_emb = nn.Embedding(2, hidden_dim)
        self.edge_type_bias = nn.Embedding(len(EDGE_TYPES), config.num_heads)
        self.edge_distance_bias = nn.Embedding(EDGE_DISTANCE_BUCKETS, config.num_heads)

        self.self_attention_layers = nn.ModuleList(
            [
                T5LayerSelfAttention(
                    config,
                    has_relative_attention_bias=bool(i == 0),
                )
                for i in range(self.num_layers)
            ]
        )

        self.ff1_layers = nn.ModuleList(
            [T5LayerFF(config) for _ in range(self.num_layers)]
        )
        self.ff2_layers = nn.ModuleList(
            [T5LayerFF(config) for _ in range(self.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(hidden_dim, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights(config)

    @disable
    def forward(self, node_ids, node_masks=None, node_roles=None, edges=None):
        return self._encode_node_content(node_ids, node_masks, node_roles, edges)

    def _encode_node_content(self, node_ids, node_masks, node_roles, edges):
        node_feats = self.node_embeddings(node_ids, node_masks=node_masks)
        if node_roles is not None:
            node_feats = node_feats + self.vulnerability_role_emb(node_roles).unsqueeze(
                2
            )

        batch_size, num_nodes, seq_len, hidden_dim = node_feats.shape

        if batch_size == 0 or num_nodes == 0:
            return (
                torch.zeros(
                    batch_size,
                    num_nodes,
                    seq_len,
                    self.hidden_dim,
                    device=node_ids.device,
                ),
                node_masks,
            )

        node_feats = node_feats * node_masks.unsqueeze(-1)
        if self.disable_edge_attention or edges is None:
            content = self._encode_content(node_feats, node_masks)
        else:
            content = self._encode_content_unified(node_feats, node_masks, edges)
        baseline_content = self.dropout(content)
        baseline_content, packed_masks = self._pack_content(
            baseline_content, node_masks
        )
        return baseline_content, packed_masks

    def _encode_content(self, node_feats, node_masks):
        batch_size, num_nodes, seq_len, hidden_dim = node_feats.shape
        flat_nodes = node_feats.view(batch_size * num_nodes, seq_len, hidden_dim)
        flat_masks = node_masks.view(batch_size * num_nodes, seq_len)
        valid_mask = flat_masks.any(-1)
        hidden = flat_nodes[valid_mask]
        attention_mask = flat_masks[valid_mask, None, None, :].to(hidden.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden.dtype).min
        position_bias = None

        for sa_layer, ff1_layer, ff2_layer in zip(
            self.self_attention_layers,
            self.ff1_layers,
            self.ff2_layers,
        ):
            hidden = ff1_layer(hidden)
            outputs = sa_layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_bias=position_bias,
                past_key_value=None,
                use_cache=False,
                cache_position=torch.arange(seq_len, device=hidden.device),
            )
            hidden = ff2_layer(outputs[0])
            position_bias = outputs[2]

        content = torch.zeros_like(flat_nodes)
        content[valid_mask] = hidden
        content = content.view(batch_size, num_nodes, seq_len, hidden_dim)
        return self.final_layer_norm(content)

    def _encode_content_unified(self, node_feats, node_masks, edges):
        batch_size, num_nodes, seq_len, hidden_dim = node_feats.shape
        length = num_nodes * seq_len
        hidden = node_feats.view(batch_size, length, hidden_dim)
        position_bias = self._unified_attention_bias(node_masks, edges, hidden.dtype)

        for sa_layer, ff1_layer, ff2_layer in zip(
            self.self_attention_layers,
            self.ff1_layers,
            self.ff2_layers,
        ):
            hidden = ff1_layer(hidden)
            outputs = sa_layer(
                hidden_states=hidden,
                attention_mask=None,
                position_bias=position_bias,
                past_key_value=None,
                use_cache=False,
                cache_position=torch.arange(length, device=hidden.device),
            )
            hidden = ff2_layer(outputs[0])
            position_bias = outputs[2]

        content = hidden.view(batch_size, num_nodes, seq_len, hidden_dim)
        return self.final_layer_norm(content)

    def _unified_attention_bias(self, node_masks, edges, dtype):
        batch_size, num_nodes, seq_len = node_masks.shape
        length = num_nodes * seq_len
        device = node_masks.device
        statement_ids = torch.arange(num_nodes, device=device).repeat_interleave(
            seq_len
        )
        same_statement = statement_ids[:, None].eq(statement_ids[None, :])
        key_valid = node_masks.reshape(batch_size, length).bool()

        attention = self.self_attention_layers[0].SelfAttention
        t5_bias = attention.compute_bias(length, length, device=device).to(dtype)

        edge_allowed, edge_bias = self._edge_bias(edges, num_nodes)
        edge_allowed = edge_allowed[:, statement_ids][:, :, statement_ids]
        edge_bias = edge_bias[:, :, statement_ids][:, :, :, statement_ids].to(dtype)

        same_allowed = same_statement.unsqueeze(0) & key_valid[:, None, :]
        edge_allowed = edge_allowed & key_valid[:, None, :]

        min_value = torch.finfo(dtype).min
        bias = torch.full(
            (batch_size, 1, length, length), min_value, dtype=dtype, device=device
        ).expand(-1, attention.n_heads, -1, -1)
        bias = torch.where(same_allowed[:, None], t5_bias, bias)
        bias = torch.where((edge_allowed & ~same_allowed)[:, None], edge_bias, bias)
        fallback = torch.eye(length, dtype=torch.bool, device=device) & ~same_allowed
        return torch.where(fallback[:, None], torch.zeros((), dtype=dtype, device=device), bias)

    def _edge_bias(self, edges, num_nodes):
        batch_size = edges.size(0)
        num_heads = self.edge_type_bias.embedding_dim
        device = edges.device
        allowed = torch.zeros(
            batch_size, num_nodes, num_nodes, dtype=torch.bool, device=device
        )
        bias = torch.zeros(batch_size, num_nodes, num_nodes, num_heads, device=device)
        if edges.size(1):
            sources, targets, types, offsets = edges.unbind(-1)
            valid = sources.ge(0)
            batch_index = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand_as(sources)[valid]
            )
            sources, targets = sources[valid], targets[valid]
            allowed[batch_index, targets, sources] = True
            values = self.edge_type_bias(types[valid]) + self.edge_distance_bias(
                relative_position_bucket(
                    offsets[valid], EDGE_DISTANCE_BUCKETS, EDGE_DISTANCE_MAX
                )
            )
            bias.index_put_((batch_index, targets, sources), values, accumulate=True)
        return allowed, bias.permute(0, 3, 1, 2)

    @staticmethod
    def _pack_content(content, node_masks):
        batch_size, num_nodes, seq_len, hidden_dim = content.shape
        token_counts = node_masks.sum(-1)
        packed_content = []
        packed_masks = []
        for batch_index in range(batch_size):
            nodes = [
                content[
                    batch_index, node_index, : token_counts[batch_index, node_index]
                ]
                for node_index in range(num_nodes)
            ]
            features = torch.cat(nodes)
            padding = num_nodes * seq_len - features.size(0)
            packed_content.append(
                torch.cat(
                    [
                        features,
                        features.new_zeros((padding, hidden_dim)),
                    ]
                )
            )
            packed_masks.append(
                torch.cat(
                    [
                        node_masks.new_ones(features.size(0)),
                        node_masks.new_zeros(padding),
                    ]
                )
            )
        return (
            torch.stack(packed_content).view(
                batch_size, num_nodes, seq_len, hidden_dim
            ),
            torch.stack(packed_masks).view(batch_size, num_nodes, seq_len),
        )

    def set_input_embeddings(self, input_embeddings):
        self.node_embeddings.embed_tokens = input_embeddings

    def init_weights(self, config):
        factor = config.initializer_factor
        d_model = config.d_model
        d_ff = config.d_ff
        num_heads = config.num_heads
        d_kv = config.d_kv

        self.init_role_embeddings(config)

        self.edge_type_bias.weight.data.zero_()
        self.edge_distance_bias.weight.data.zero_()
        self.final_layer_norm.weight.data.fill_(factor * 1.0)

        for layer in self.self_attention_layers:
            attention = layer.SelfAttention
            attention.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * d_kv) ** -0.5)
            )

            attention.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            attention.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            attention.o.weight.data.normal_(
                mean=0.0, std=factor * ((num_heads * d_kv) ** -0.5)
            )

            if attention.has_relative_attention_bias:
                attention.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * (d_model**-0.5)
                )

        for ff_layer in self.ff1_layers + self.ff2_layers:
            ff_module = (
                ff_layer.DenseReluDense
                if hasattr(ff_layer, "DenseReluDense")
                else ff_layer.DenseGatedGeluDense
            )

            if isinstance(ff_module, T5DenseActDense):
                ff_module.wi.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
                if hasattr(ff_module.wi, "bias") and ff_module.wi.bias is not None:
                    ff_module.wi.bias.data.zero_()

                ff_module.wo.weight.data.normal_(mean=0.0, std=factor * (d_ff**-0.5))
                if hasattr(ff_module.wo, "bias") and ff_module.wo.bias is not None:
                    ff_module.wo.bias.data.zero_()

            elif isinstance(ff_module, T5DenseGatedActDense):
                ff_module.wi_0.weight.data.normal_(
                    mean=0.0, std=factor * (d_model**-0.5)
                )
                if hasattr(ff_module.wi_0, "bias") and ff_module.wi_0.bias is not None:
                    ff_module.wi_0.bias.data.zero_()

                ff_module.wi_1.weight.data.normal_(
                    mean=0.0, std=factor * (d_model**-0.5)
                )
                if hasattr(ff_module.wi_1, "bias") and ff_module.wi_1.bias is not None:
                    ff_module.wi_1.bias.data.zero_()

                ff_module.wo.weight.data.normal_(mean=0.0, std=factor * (d_ff**-0.5))
                if hasattr(ff_module.wo, "bias") and ff_module.wo.bias is not None:
                    ff_module.wo.bias.data.zero_()

    def init_role_embeddings(self, config):
        self.vulnerability_role_emb.weight.data.zero_()
        nn.init.normal_(
            self.vulnerability_role_emb.weight.data[1],
            mean=0.0,
            std=config.initializer_factor * (config.d_model**-0.5),
        )


class T5NodeEmbedder(nn.Module):
    def __init__(self, shared, dropout=0.1):
        super(T5NodeEmbedder, self).__init__()
        self.embed_tokens = shared
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_ids, node_masks=None):
        batch_size, num_nodes, seq_length = node_ids.shape

        if node_ids.size(1) == 0:
            batch_size = node_ids.size(0)
            hidden_dim = self.embed_tokens.embedding_dim
            seq_length = node_ids.size(2)
            return torch.zeros(
                batch_size, 0, seq_length, hidden_dim, device=node_ids.device
            )

        flat_node_ids = node_ids.reshape(-1, seq_length)
        outputs = self.embed_tokens(flat_node_ids)
        outputs = self.dropout(outputs)
        hidden_dim = outputs.shape[-1]
        node_feats = outputs.reshape(batch_size, num_nodes, seq_length, hidden_dim)

        return node_feats
