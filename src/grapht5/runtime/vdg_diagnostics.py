import torch


def unwrap_model(model):
    while hasattr(model, "_orig_mod") or hasattr(model, "module"):
        model = model._orig_mod if hasattr(model, "_orig_mod") else model.module
    return model


def parameter_groups(model):
    model = unwrap_model(model)
    encoder = model.graph_encoder
    encoder_total = [
        parameter
        for name, parameter in encoder.named_parameters()
        if not name.startswith("node_embeddings.embed_tokens.")
    ]
    decoder_total = [
        parameter
        for name, parameter in model.decoder.named_parameters()
        if "EncDecAttentionGraph." in name or "cross_gate" in name
    ]
    return {
        "total": encoder_total + decoder_total,
        "encoder": encoder_total,
        "decoder": decoder_total,
        "role": list(encoder.vulnerability_role_emb.parameters()),
        "edge_bias": list(encoder.edge_type_bias.parameters())
        + list(encoder.edge_distance_bias.parameters()),
        "self_attention": list(encoder.self_attention_layers.parameters()),
        "feed_forward": list(encoder.ff1_layers.parameters())
        + list(encoder.ff2_layers.parameters()),
    }


def _norm(tensors):
    norms = [tensor.detach().float().norm() for tensor in tensors]
    return torch.stack(norms).norm().item() if norms else 0.0


def gradient_metrics(model):
    return {
        f"vdg/grad_norm_pre_clip/{name}": _norm(
            parameter.grad for parameter in parameters if parameter.grad is not None
        )
        for name, parameters in parameter_groups(model).items()
    }


def parameter_snapshot(model):
    return {
        name: [parameter.detach().clone() for parameter in parameters]
        for name, parameters in parameter_groups(model).items()
        if name != "total"
    }


def update_metrics(model, snapshot):
    groups = parameter_groups(model)
    metrics = {}
    for name, previous in snapshot.items():
        parameters = groups[name]
        update_norm = _norm(
            parameter.detach() - old for parameter, old in zip(parameters, previous)
        )
        weight_norm = _norm(previous)
        metrics[f"vdg/update_norm/{name}"] = update_norm
        metrics[f"vdg/update_relative_norm/{name}"] = update_norm / max(
            weight_norm, torch.finfo(torch.float32).tiny
        )
    return metrics


def gate_metrics(model):
    decoder = unwrap_model(model).decoder
    learned = decoder.use_learned_gate
    metrics = {}
    for index, block in enumerate(decoder.block):
        cross = block.layer[1]
        text_share = torch.sigmoid(cross.cross_gate).item() if learned else 0.5
        metrics[f"vdg/decoder_gate/node_share/layer_{index}"] = 1.0 - text_share
        metrics[f"vdg/decoder_gate/projection_norm/layer_{index}"] = (
            cross.cross_gate_proj.weight.detach().float().norm().item()
        )
    return metrics
