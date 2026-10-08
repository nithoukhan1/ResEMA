from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.vgra import (
    VGRA,
    VGRALevelResidual,
    VGRAPairVisibilityPredictor,
    VGRAViewDescriptor,
    visibility_coefficients,
    visibility_state_from_presence,
    visibility_state_weights,
)


CHANNELS = (128, 256, 512)
NC = 9


def _features(batch=2):
    torch.manual_seed(42)
    return [
        torch.randn(batch, 128, 20, 20),
        torch.randn(batch, 256, 10, 10),
        torch.randn(batch, 512, 5, 5),
    ]


def test_visibility_state_mapping_exact():
    ap = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    lat = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    state = visibility_state_from_presence(ap, lat)
    assert torch.equal(state, torch.tensor([[0, 1, 2, 3]], dtype=torch.long))


def test_visibility_state_weights_have_per_class_mean_one():
    counts = torch.tensor([[100.0, 25.0, 9.0, 64.0], [0.0, 0.0, 0.0, 0.0]])
    w = visibility_state_weights(counts)
    assert w.shape == (2, 4)
    torch.testing.assert_close(w.mean(dim=-1), torch.ones(2))
    assert torch.all(w > 0)


def test_target_specific_gate_truth_table():
    probs = torch.eye(4).unsqueeze(1)
    k_ap, k_lat = visibility_coefficients(probs, detach=True)
    torch.testing.assert_close(k_ap[:, 0], torch.tensor([0.0, 0.0, -1.0, 1.0]))
    torch.testing.assert_close(k_lat[:, 0], torch.tensor([0.0, -1.0, 0.0, 1.0]))
    assert not k_ap.requires_grad
    assert not k_lat.requires_grad


def test_view_descriptor_and_visibility_shapes():
    descriptor = VGRAViewDescriptor(CHANNELS, descriptor_rank=32)
    predictor = VGRAPairVisibilityPredictor(view_dim=96, nc=NC, hidden_dim=128)
    ap = descriptor(_features(batch=3))
    lat = descriptor(_features(batch=3))
    assert ap.shape == (3, 96)
    logits = predictor(ap, lat)
    assert logits.shape == (3, NC, 4)
    q = predictor.probabilities(logits)
    torch.testing.assert_close(q.sum(dim=-1), torch.ones(3, NC), rtol=1e-6, atol=1e-6)


def test_level_compatibility_is_bounded():
    torch.manual_seed(42)
    module = VGRALevelResidual(128, companion_dim=96, nc=NC, rank=16, beta_max=2.0)
    x = torch.randn(2, 128, 12, 12)
    h = torch.randn(2, 96)
    m = module.compatibility(x, h)
    assert m.shape == (2, NC, 12, 12)
    assert torch.isfinite(m).all()
    assert float(m.min()) >= 0.0
    assert float(m.max()) <= 1.0 + 1e-7


def test_rho_zero_is_exact_residual_identity():
    torch.manual_seed(42)
    module = VGRALevelResidual(128, companion_dim=96, nc=NC, rank=16, beta_max=2.0)
    x = torch.randn(2, 128, 8, 8)
    h = torch.randn(2, 96)
    coeff = torch.randn(2, NC).clamp(-1, 1)
    native_logits = torch.randn(2, NC, 8, 8)
    residual = module(x, h, coeff)
    assert torch.count_nonzero(residual).item() == 0
    updated = native_logits + residual
    assert torch.equal(updated, native_logits)
    assert float(module.residual_strength.detach()) == 0.0


def test_residual_is_bounded_by_beta_max_for_valid_coefficients():
    torch.manual_seed(42)
    module = VGRALevelResidual(128, companion_dim=96, nc=NC, rank=16, beta_max=2.0)
    with torch.no_grad():
        module.rho.fill_(10.0)
    x = torch.randn(2, 128, 8, 8)
    h = torch.randn(2, 96)
    probs = torch.softmax(torch.randn(2, NC, 4), dim=-1)
    k_ap, _ = visibility_coefficients(probs)
    residual = module(x, h, k_ap)
    assert torch.isfinite(residual).all()
    assert float(residual.abs().max()) <= 2.0 + 1e-6


def test_stop_gradient_blocks_detection_path_from_visibility_logits():
    torch.manual_seed(42)
    module = VGRALevelResidual(128, companion_dim=96, nc=NC, rank=16, beta_max=2.0)
    with torch.no_grad():
        module.rho.fill_(0.4)

    visibility_logits = torch.randn(2, NC, 4, requires_grad=True)
    probs = torch.softmax(visibility_logits, dim=-1)
    k_ap, _ = visibility_coefficients(probs, detach=True)

    x = torch.randn(2, 128, 8, 8, requires_grad=True)
    h = torch.randn(2, 96, requires_grad=True)
    loss = module(x, h, k_ap).square().mean()
    loss.backward()

    assert visibility_logits.grad is None
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert module.rho.grad is not None and torch.isfinite(module.rho.grad)


def test_visibility_predictor_can_learn_from_visibility_loss():
    torch.manual_seed(42)
    predictor = VGRAPairVisibilityPredictor(view_dim=96, nc=NC, hidden_dim=128)
    ap = torch.randn(4, 96, requires_grad=True)
    lat = torch.randn(4, 96, requires_grad=True)
    target = torch.randint(0, 4, (4, NC))
    logits = predictor(ap, lat)
    loss = F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1))
    loss.backward()
    assert predictor.fc2.weight.grad is not None
    assert torch.isfinite(predictor.fc2.weight.grad).all()
    assert float(predictor.fc2.weight.grad.abs().sum()) > 0.0


def test_full_vgra_shapes_identity_and_finite_backward():
    torch.manual_seed(42)
    model = VGRA(CHANNELS, nc=NC, descriptor_rank=32, pair_hidden_dim=128, cross_view_rank=16, beta_max=2.0)
    ap_features = [x.requires_grad_() for x in _features(batch=2)]
    lat_features = [x.requires_grad_() for x in _features(batch=2)]
    out = model.forward_pair(ap_features, lat_features)

    assert out["ap_descriptor"].shape == (2, 96)
    assert out["lat_descriptor"].shape == (2, 96)
    assert out["visibility_logits"].shape == (2, NC, 4)
    assert out["visibility_probabilities"].shape == (2, NC, 4)
    assert out["k_ap"].shape == (2, NC)
    assert out["k_lat"].shape == (2, NC)
    assert [r.shape for r in out["ap_residuals"]] == [(2, NC, 20, 20), (2, NC, 10, 10), (2, NC, 5, 5)]
    assert all(torch.count_nonzero(r).item() == 0 for r in out["ap_residuals"])
    assert all(torch.count_nonzero(r).item() == 0 for r in out["lat_residuals"])

    targets = torch.randint(0, 4, (2, NC))
    vis_loss = F.cross_entropy(out["visibility_logits"].reshape(-1, 4), targets.reshape(-1))
    vis_loss.backward()
    grads = [p.grad for p in model.visibility.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_frozen_reference_core_parameter_count():
    model = VGRA(CHANNELS, nc=NC, descriptor_rank=32, pair_hidden_dim=128, cross_view_rank=16, beta_max=2.0)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params == 102_567
    assert params < 300_000
