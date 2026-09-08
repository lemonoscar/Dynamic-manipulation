"""Differentiable RGB banks, frozen-path parity, and full-module gradient coverage.

Small CPU tensors validate graph wiring, not Qwen capability or causal labels.
"""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from conveyor_bench.conveyorvla import staged_rgb_backend as module


class Qwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.visual = nn.Linear(4, 8)
        self.core = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 32)
        self.calls = 0
        self.prompts = []
        self.processor = SimpleNamespace(tokenizer=self.tokenize)

    def tokenize(self, text, return_tensors):
        device = self.embed_tokens.weight.device
        return {'input_ids': torch.tensor([[len(text) % 32, sum(map(ord, text)) % 32]], device=device),
            'attention_mask': torch.ones(1, 2, dtype=torch.long, device=device)}

    def build_joint_trajectory_inputs(self, examples, solutions, supervise_solutions):
        self.prompts.append((examples, solutions, supervise_solutions))
        result = self.tokenize(examples[0]['lang'], 'pt')
        frames = examples[0]['video'][0] + examples[0]['video'][1]
        result['pixel_values'] = torch.tensor([frames], dtype=torch.float32, device=self.embed_tokens.weight.device)
        result['labels'] = torch.tensor([[-1]])
        return result

    def forward(self, input_ids, attention_mask, pixel_values=None, **kwargs):
        self.calls += 1
        hidden = self.embed_tokens(input_ids)
        if pixel_values is not None:
            hidden = hidden + self.visual(pixel_values)[:, None]
        hidden = torch.tanh(self.core(hidden))
        return SimpleNamespace(hidden_states=(hidden,), logits=self.lm_head(hidden))


class FullPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen = Qwen()
        self.navigation = nn.Linear(8, 3)
        self.manipulation = nn.Linear(8, 7)


def banks_and_loss(policy):
    banks = module.encode_rgb_training_conditions(policy, 'transfer cola', 'PICK', (1., 2., 3., 4.))
    pooled = torch.cat((banks['task_tokens'], banks['live_tokens']), dim=1).mean(1)
    return banks, policy.navigation(pooled).square().mean() + policy.manipulation(pooled).square().mean()


@pytest.mark.parametrize('primitive', ('PICK', 'PLACE', 'NAV_TO_SOURCE', 'NAV_TO_TARGET'))
def test_training_banks_equal_frozen_encoder_and_preserve_gradients(primitive):
    torch.manual_seed(7)
    policy = FullPolicy().eval()
    images = (1., 2., 3., 4.)
    frozen = module.encode_rgb_conditions(policy, 'transfer cola', primitive, images)
    old_prompt = policy.qwen.prompts[-1]
    live = module.encode_rgb_training_conditions(policy, 'transfer cola', primitive, images)
    assert policy.qwen.prompts[-1] == old_prompt
    assert set(live) == {'task_tokens', 'task_mask', 'live_tokens', 'live_mask'}
    for key in live:
        torch.testing.assert_close(live[key].detach().cpu(), frozen[key], rtol=0, atol=0)
        assert live[key].device == next(policy.qwen.parameters()).device
    for key in ('task_tokens', 'live_tokens'):
        assert live[key].requires_grad and live[key].grad_fn is not None
        assert not live[key].is_inference()
    (live['task_tokens'].square().mean() + live['live_tokens'].square().mean()).backward()
    for component in (policy.qwen.embed_tokens, policy.qwen.visual, policy.qwen.core):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters())
    assert policy.qwen.lm_head.weight.grad is None  # Action hidden loss alone cannot train the LM output head.


def test_full_optimizer_coverage_and_real_update_require_language_objective():
    torch.manual_seed(8)
    policy = FullPolicy().train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    flat = [p for group in optimizer.param_groups for p in group['params']]
    assert len(flat) == len({id(p) for p in flat})
    assert {id(p) for p in flat} == {id(p) for p in policy.parameters() if p.requires_grad}
    _, action_loss = banks_and_loss(policy)
    # A separate causal language query is necessary in the production runner.
    # This fixture only proves the extra objective reaches lm_head parameters.
    inputs = policy.qwen.tokenize('visible observations and public completed history', 'pt')
    language = policy.qwen(**inputs)
    answer_loss = nn.functional.cross_entropy(language.logits[:, -1], torch.tensor([2]))
    (action_loss + answer_loss).backward()
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    missing = [name for name, p in policy.named_parameters() if p.grad is None]
    assert not missing
    for name, parameter in policy.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in policy.named_parameters():
        assert not torch.equal(before[name], parameter), name


def test_training_encoder_rejects_frozen_graph_and_cache():
    policy = FullPolicy()
    args = (policy, 'transfer cola', 'PICK', (1., 2., 3., 4.))
    with torch.no_grad(), pytest.raises(ValueError, match='autograd'):
        module.encode_rgb_training_conditions(*args)
    with torch.inference_mode(), pytest.raises(ValueError, match='autograd'):
        module.encode_rgb_training_conditions(*args)
    with pytest.raises(TypeError):
        module.encode_rgb_training_conditions(*args, task_cache={})
    policy.qwen.requires_grad_(False)
    with pytest.raises(ValueError, match='trainable Qwen'):
        module.encode_rgb_training_conditions(*args)


def test_public_context_has_identical_training_and_deployment_prompt():
    policy = FullPolicy().eval()
    images = (1., 2., 3., 4.)
    context = 'Original goal: transfer cola. Completed: reached source. Current: PICK. Remaining: carry, PLACE.'
    inference = module.encode_rgb_conditions(policy, 'transfer cola', 'PICK', images, context=context)
    prompt = policy.qwen.prompts[-1]
    training = module.encode_rgb_training_conditions(policy, 'transfer cola', 'PICK', images, context=context)
    assert policy.qwen.prompts[-1] == prompt
    assert prompt[0][0]['lang'] == module.rgb_condition_semantic('transfer cola', 'PICK', images, context=context)
    for key in training:
        torch.testing.assert_close(training[key].detach().cpu(), inference[key], rtol=0, atol=0)
    cache = {}
    module.encode_rgb_conditions(policy, 'transfer cola', 'PICK', images, cache, context=context)
    calls = policy.qwen.calls
    module.encode_rgb_conditions(policy, 'transfer cola', 'PICK', images, cache, context=context + ' New observed feedback.')
    assert policy.qwen.calls == calls + 2  # A changed public context invalidates task tokens.
    with pytest.raises(ValueError, match='public text'):
        module.encode_rgb_training_conditions(policy, 'transfer cola', 'PICK', images, context={'teacher_phase': 'PICK'})
