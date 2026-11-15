import torch
from transformers import T5Config

from grapht5.modeling.model import FiDT5


def test_text_and_graph_paths_share_embeddings_and_train():
    config = T5Config(
        vocab_size=64,
        d_model=32,
        d_ff=64,
        d_kv=8,
        num_heads=4,
        num_layers=2,
        num_decoder_layers=2,
        decoder_start_token_id=0,
    )
    config.graph_attention_layers = 2
    config.max_nodes_per_graph = 2
    config.node_max_length = 4
    config.no_learned_gate = False
    config.disable_edge_attention = False
    config.add_consistency_loss = False
    config.drop_t5_passages = False

    model = FiDT5(config)
    assert model.graph_encoder.node_embeddings.embed_tokens is model.shared
    assert len(model.graph_encoder.self_attention_layers) == 2
    assert len(model.graph_encoder.ff1_layers) == 2
    assert len(model.graph_encoder.ff2_layers) == 2
    assert len(model.decoder.block) == 2

    output = model(
        input_ids=torch.tensor([[[5, 6, 7, 0]]]),
        attention_mask=torch.tensor([[[1, 1, 1, 0]]]),
        node_ids=torch.tensor([[[5, 6, 0, 0], [7, 8, 9, 0]]]),
        node_masks=torch.tensor([[[1, 1, 0, 0], [1, 1, 1, 0]]]),
        node_roles=torch.tensor([[1, 0]]),
        edges=torch.tensor([[[0, 1, 0, 1]]]),
        labels=torch.tensor([[10, 11, 2]]),
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.graph_encoder.edge_type_bias.weight.grad is not None
