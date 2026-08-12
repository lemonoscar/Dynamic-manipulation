import pytest

torch = pytest.importorskip("torch")

from conveyor_bench.conveyorvla.aml import AMLActionHead, AMLConfig


class _PerfectActionHead(AMLActionHead):
    def __init__(self, config: AMLConfig, target: torch.Tensor) -> None:
        super().__init__(config)
        self.register_buffer("target", target)

    def forward(self, context, state, noisy_actions, time):
        del context, state, time
        return self.target.expand_as(noisy_actions)


def _fixture():
    torch.manual_seed(7)
    config = AMLConfig(
        action_dim=3,
        state_dim=4,
        context_dim=5,
        action_horizon=4,
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        feedforward_dim=32,
        inference_steps=4,
    )
    return (
        config,
        torch.randn(2, 3, config.context_dim),
        torch.randn(2, config.state_dim),
        torch.randn(2, config.action_horizon, config.action_dim),
    )


def test_aml_loss_backward_and_sampler_are_finite() -> None:
    config, context, state, target = _fixture()
    model = AMLActionHead(config)
    noise = torch.randn_like(target)
    time = torch.full((target.shape[0],), 0.25)
    mask = torch.tensor([True, False, True])

    loss = model.aml_loss(
        context,
        state,
        target,
        action_dimension_mask=mask,
        noise=noise,
        time=time,
    )
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    sampled = model.sample(context, state, noise=noise)
    assert sampled.shape == target.shape
    assert torch.isfinite(sampled).all()


def test_aml_checkpoint_reload_is_exact() -> None:
    config, context, state, target = _fixture()
    model = AMLActionHead(config).eval()
    time = torch.full((target.shape[0],), 0.25)
    expected = model(context, state, target, time)

    reloaded = AMLActionHead(config).eval()
    reloaded.load_state_dict(model.state_dict())

    assert torch.equal(reloaded(context, state, target, time), expected)


def test_aml_rejects_an_empty_action_mask() -> None:
    config, context, state, target = _fixture()
    model = AMLActionHead(config)

    with pytest.raises(ValueError, match="enable at least one"):
        model.aml_loss(
            context,
            state,
            target,
            action_dimension_mask=torch.zeros(config.action_dim, dtype=torch.bool),
        )


def test_aml_sampler_rejects_zero_steps_and_reaches_clean_action() -> None:
    config, context, state, noise = _fixture()
    target = torch.full((1, config.action_horizon, config.action_dim), 2.0)
    model = _PerfectActionHead(config, target)

    with pytest.raises(ValueError, match="steps must be positive"):
        model.sample(context, state, noise=noise, steps=0)

    for steps in (4, 32):
        sampled = model.sample(context, state, noise=noise, steps=steps)
        assert sampled == pytest.approx(target.expand_as(noise), abs=1.0e-6)


def test_perfect_clean_action_prediction_has_zero_aml_loss() -> None:
    config, context, state, target = _fixture()
    model = _PerfectActionHead(config, target[:1])
    repeated_target = target[:1].expand_as(target)

    loss = model.aml_loss(
        context,
        state,
        repeated_target,
        noise=torch.randn_like(repeated_target),
        time=torch.full((target.shape[0],), 0.25, dtype=torch.float64),
    )

    assert loss.item() == pytest.approx(0.0, abs=1.0e-12)
