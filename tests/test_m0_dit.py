from __future__ import annotations

import torch
import pytest

from conveyor_bench.conveyorvla.dit import (
    ABOT_DOMAIN_ACTION_REINITIALIZED_KEYS,
    ABOT_DOMAIN_STATE_REINITIALIZED_KEYS,
    GO2_X5_REINITIALIZED_ACTION_KEYS,
    M0DiTActionHead,
    M0DiTConfig,
    transfer_abot_pretrain_domain_weights,
    transfer_robocasa_action_weights,
)


def _tiny_config(
    *,
    num_layers: int = 2,
    action_dim: int = 3,
    state_dim: int = 5,
) -> M0DiTConfig:
    return M0DiTConfig(
        action_dim=action_dim,
        state_dim=state_dim,
        action_horizon=4,
        vlm_hidden_dim=8,
        input_embedding_dim=8,
        hidden_size=12,
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=num_layers,
        dropout=0.0,
        max_seq_len=8,
        num_target_vision_tokens=2,
        num_inference_timesteps=4,
    )


def _official_action_keys(num_layers: int) -> set[str]:
    keys = {
        "model.timestep_encoder.timestep_embedder.linear_1.weight",
        "model.timestep_encoder.timestep_embedder.linear_1.bias",
        "model.timestep_encoder.timestep_embedder.linear_2.weight",
        "model.timestep_encoder.timestep_embedder.linear_2.bias",
        "model.proj_out_1.weight",
        "model.proj_out_1.bias",
        "model.proj_out_2.weight",
        "model.proj_out_2.bias",
        "state_encoder.layer1.weight",
        "state_encoder.layer1.bias",
        "state_encoder.layer2.weight",
        "state_encoder.layer2.bias",
        "action_encoder.layer1.weight",
        "action_encoder.layer1.bias",
        "action_encoder.layer2.weight",
        "action_encoder.layer2.bias",
        "action_encoder.layer3.weight",
        "action_encoder.layer3.bias",
        "action_decoder.layer1.weight",
        "action_decoder.layer1.bias",
        "action_decoder.layer2.weight",
        "action_decoder.layer2.bias",
        "future_tokens.weight",
        "position_embedding.weight",
    }
    for index in range(num_layers):
        prefix = f"model.transformer_blocks.{index}"
        keys.update(
            {
                f"{prefix}.norm1.linear.weight",
                f"{prefix}.norm1.linear.bias",
                f"{prefix}.attn1.to_q.weight",
                f"{prefix}.attn1.to_q.bias",
                f"{prefix}.attn1.to_k.weight",
                f"{prefix}.attn1.to_k.bias",
                f"{prefix}.attn1.to_v.weight",
                f"{prefix}.attn1.to_v.bias",
                f"{prefix}.attn1.to_out.0.weight",
                f"{prefix}.attn1.to_out.0.bias",
                f"{prefix}.ff.net.0.proj.weight",
                f"{prefix}.ff.net.0.proj.bias",
                f"{prefix}.ff.net.2.weight",
                f"{prefix}.ff.net.2.bias",
            }
        )
    return keys


def _inputs(config: M0DiTConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 2
    return (
        torch.randn(batch_size, 3, config.vlm_hidden_dim),
        torch.randn(batch_size, config.action_horizon, config.action_dim),
        torch.randn(batch_size, 1, config.state_dim),
    )


def test_state_dict_exactly_matches_official_action_checkpoint_keys() -> None:
    model = M0DiTActionHead(_tiny_config(num_layers=16))

    assert set(model.state_dict()) == _official_action_keys(16)
    assert len(model.state_dict()) == 248


def test_action_structure_can_be_audited_on_meta_device() -> None:
    with torch.device("meta"):
        model = M0DiTActionHead(_tiny_config())

    assert all(value.device.type == "meta" for value in model.state_dict().values())


def test_tiny_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    model = M0DiTActionHead(config)
    vl_embeddings, actions, state = _inputs(config)

    loss = model(
        vl_embeddings,
        actions,
        state,
        noise=torch.zeros_like(actions),
        time=torch.tensor([0.25, 0.75]),
    )
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_action_valid_prefix_masks_cross_expert_suffix() -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    model = M0DiTActionHead(config)
    vl_embeddings, actions, state = _inputs(config)
    changed = actions.clone()
    changed[:, 2:] += 10_000.0
    valid = torch.tensor([[True, True, False, False]] * 2)
    kwargs = {
        "noise": torch.zeros_like(actions),
        "time": torch.tensor([0.25, 0.75]),
        "action_valid_mask": valid,
    }

    first = model(vl_embeddings, actions, state, **kwargs)
    second = model(vl_embeddings, changed, state, **kwargs)

    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="true prefix"):
        model(
            vl_embeddings,
            actions,
            state,
            action_valid_mask=torch.tensor([[True, False, True, False]] * 2),
        )


def test_four_step_sample_is_finite_and_zeros_masked_dimensions() -> None:
    torch.manual_seed(0)
    config = _tiny_config()
    model = M0DiTActionHead(config).eval()
    vl_embeddings, _, state = _inputs(config)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    model_calls: list[None] = []
    hook = model.model.register_forward_hook(lambda *_: model_calls.append(None))

    sampled = model.sample(
        vl_embeddings,
        state,
        action_dimension_mask=mask,
        noise=torch.ones(2, config.action_horizon, config.action_dim),
    )
    hook.remove()

    assert len(model_calls) == 4
    assert sampled.shape == (2, config.action_horizon, config.action_dim)
    assert torch.isfinite(sampled).all()
    assert torch.count_nonzero(sampled.masked_select(~mask[:, None, :])) == 0


def test_stateless_action_head_uses_vlm_device_and_dimension_reduction() -> None:
    torch.manual_seed(0)
    config = _tiny_config(state_dim=0)
    model = M0DiTActionHead(config)
    vl_embeddings = torch.randn(2, 3, config.vlm_hidden_dim)
    actions = torch.randn(2, config.action_horizon, config.action_dim)

    dimensions = model(
        vl_embeddings,
        actions,
        None,
        noise=torch.zeros_like(actions),
        time=torch.tensor([0.25, 0.75]),
        reduction="dimension_mean",
    )
    sampled = model.sample(
        vl_embeddings,
        None,
        noise=torch.ones_like(actions),
    )

    assert model.state_encoder is None
    assert dimensions.shape == (config.action_dim,)
    assert torch.isfinite(dimensions).all()
    assert sampled.shape == actions.shape
    with pytest.raises(ValueError, match="must not receive state"):
        model(vl_embeddings, actions, torch.zeros(2, 1), time=torch.zeros(2))


def _robocasa_checkpoint(model: M0DiTActionHead) -> dict[str, torch.Tensor]:
    source = {
        f"action_model.{key}": torch.full_like(value, 0.125)
        for key, value in model.state_dict().items()
    }
    hidden_size = model.config.hidden_size
    embedding_dim = model.config.input_embedding_dim
    source.update(
        {
            "action_model.state_encoder.layer1.weight": torch.randn(hidden_size, 58),
            "action_model.state_encoder.layer1.bias": torch.randn(hidden_size),
            "action_model.action_encoder.layer1.weight": torch.randn(embedding_dim, 29),
            "action_model.action_encoder.layer1.bias": torch.randn(embedding_dim),
            "action_model.action_decoder.layer2.weight": torch.randn(29, hidden_size),
            "action_model.action_decoder.layer2.bias": torch.randn(29),
        }
    )
    return source


def test_robocasa_transfer_reinitializes_only_six_boundary_tensors() -> None:
    torch.manual_seed(0)
    model = M0DiTActionHead(_tiny_config())
    checkpoint = _robocasa_checkpoint(model)
    initial_boundary = {
        key: model.state_dict()[key].clone()
        for key in GO2_X5_REINITIALIZED_ACTION_KEYS
    }

    report = transfer_robocasa_action_weights(model, checkpoint)

    assert set(report.reinitialized_keys) == GO2_X5_REINITIALIZED_ACTION_KEYS
    assert len(report.reinitialized_keys) == 6
    assert set(report.loaded_keys) == set(model.state_dict()) - GO2_X5_REINITIALIZED_ACTION_KEYS
    for key, value in model.state_dict().items():
        if key in GO2_X5_REINITIALIZED_ACTION_KEYS:
            assert torch.equal(value, initial_boundary[key])
        else:
            assert torch.equal(value, checkpoint[f"action_model.{key}"])


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_robocasa_transfer_fails_closed_on_unapproved_structure(corruption: str) -> None:
    model = M0DiTActionHead(_tiny_config())
    checkpoint = _robocasa_checkpoint(model)
    if corruption == "missing":
        del checkpoint["action_model.state_encoder.layer1.bias"]
    elif corruption == "unexpected":
        checkpoint["action_model.unapproved.weight"] = torch.ones(1)
    else:
        checkpoint["action_model.state_encoder.layer2.weight"] = torch.ones(1)

    with pytest.raises(RuntimeError, match="checkpoint|shape mismatch"):
        transfer_robocasa_action_weights(model, checkpoint)


def test_abot_pretrain_transfer_reuses_trunk_for_nav_and_mani_boundaries() -> None:
    source_model = M0DiTActionHead(
        _tiny_config(action_dim=14, state_dim=14)
    )
    checkpoint = {
        f"action_model.{key}": torch.full_like(value, 0.125)
        for key, value in source_model.state_dict().items()
    }
    nav = M0DiTActionHead(_tiny_config(action_dim=3, state_dim=0))
    mani = M0DiTActionHead(_tiny_config(action_dim=7, state_dim=13))
    nav_initial = {
        key: nav.state_dict()[key].clone()
        for key in ABOT_DOMAIN_ACTION_REINITIALIZED_KEYS
    }
    mani_reinitialized = (
        ABOT_DOMAIN_ACTION_REINITIALIZED_KEYS
        | ABOT_DOMAIN_STATE_REINITIALIZED_KEYS
    )
    mani_initial = {
        key: mani.state_dict()[key].clone() for key in mani_reinitialized
    }

    nav_report = transfer_abot_pretrain_domain_weights(nav, checkpoint)
    mani_report = transfer_abot_pretrain_domain_weights(mani, checkpoint)

    assert set(nav_report.reinitialized_keys) == ABOT_DOMAIN_ACTION_REINITIALIZED_KEYS
    assert set(nav_report.ignored_source_keys) == {
        key for key in source_model.state_dict() if key.startswith("state_encoder.")
    }
    assert set(mani_report.reinitialized_keys) == mani_reinitialized
    assert not mani_report.ignored_source_keys
    for key, value in nav.state_dict().items():
        if key in nav_initial:
            assert torch.equal(value, nav_initial[key])
        else:
            assert torch.equal(value, checkpoint[f"action_model.{key}"])
    for key, value in mani.state_dict().items():
        if key in mani_initial:
            assert torch.equal(value, mani_initial[key])
        else:
            assert torch.equal(value, checkpoint[f"action_model.{key}"])
