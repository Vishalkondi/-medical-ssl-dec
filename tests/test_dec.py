import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.dec import ClusteringLayer, target_distribution  # noqa: E402
from src.training.losses import (  # noqa: E402
    class_weights, dec_kl_loss, select_pseudo_labels, sigmoid_rampup, supervised_q_loss,
)


def test_soft_assignment_shape_and_rows_sum_to_one():
    layer = ClusteringLayer(2, 8)
    q = layer(torch.randn(16, 8))
    assert q.shape == (16, 2)
    assert torch.allclose(q.sum(1), torch.ones(16), atol=1e-6)
    assert (q > 0).all()


def test_closer_centre_gets_higher_probability():
    layer = ClusteringLayer(2, 2)
    layer.centers.data = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    q = layer(torch.tensor([[0.1, 0.0], [9.9, 10.0]]))
    assert q[0, 0] > 0.9 and q[1, 1] > 0.9


def test_init_centers_uses_class_means():
    layer = ClusteringLayer(2, 2)
    z = torch.tensor([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
    layer.init_centers(z, torch.tensor([0, 0, 1]))
    assert torch.allclose(layer.centers, torch.tensor([[1.0, 1.0], [10.0, 10.0]]))


def test_init_centers_requires_every_class():
    with pytest.raises(ValueError, match="no labelled samples"):
        ClusteringLayer(2, 2).init_centers(torch.zeros(3, 2), torch.tensor([0, 0, 0]))


def test_gradient_finite_when_embedding_equals_centre():
    layer = ClusteringLayer(2, 4)
    z = layer.centers.detach()[:1].clone().requires_grad_(True)
    layer(z)[0, 0].backward()
    assert torch.isfinite(z.grad).all()


def test_target_distribution_is_sharper_and_normalised():
    q = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.55, 0.45]])
    p = target_distribution(q)
    assert torch.allclose(p.sum(1), torch.ones(3))
    assert (p.max(1).values >= q.max(1).values - 1e-6).all()


def test_prior_correction_changes_cluster_balance():
    q = torch.full((100, 2), 0.5)
    q[:, 1] += torch.linspace(-0.1, 0.1, 100)
    q[:, 0] = 1 - q[:, 1]
    p_uniform = target_distribution(q)
    p_prior = target_distribution(q, prior=torch.tensor([0.9, 0.1]))
    assert p_prior[:, 0].mean() > p_uniform[:, 0].mean()


def test_kl_loss_zero_when_q_equals_p_and_positive_otherwise():
    p = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    assert dec_kl_loss(p.clone(), p).item() == pytest.approx(0.0, abs=1e-6)
    assert dec_kl_loss(torch.full_like(p, 0.5), p).item() > 0
    mask = torch.tensor([0.0, 0.0])
    assert dec_kl_loss(torch.full_like(p, 0.5), p, mask).item() == 0.0


def test_supervised_q_loss_matches_cross_entropy():
    q = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    y = torch.tensor([0, 1])
    expected = -(np.log(0.8) + np.log(0.7)) / 2
    assert supervised_q_loss(q, y).item() == pytest.approx(expected, rel=1e-5)


def test_class_weights_inverse_frequency():
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert w[1] / w[0] == pytest.approx(3.0)


def test_rampup_bounds():
    assert sigmoid_rampup(0, 10) == pytest.approx(np.exp(-5))
    assert sigmoid_rampup(10, 10) == 1.0
    assert sigmoid_rampup(50, 10) == 1.0


def test_pseudo_label_selection_threshold_and_prior_balance():
    probs = np.array([[0.99, 0.01]] * 50 + [[0.02, 0.98]] * 10 + [[0.6, 0.4]] * 5)
    assert len(select_pseudo_labels(probs, 0.95)) == 60
    idx = select_pseudo_labels(probs, 0.95, prior=np.array([0.5, 0.5]))
    pred = probs[idx].argmax(1)
    assert (pred == 0).sum() == (pred == 1).sum() == 10
    assert len(select_pseudo_labels(np.array([[0.99, 0.01]] * 5), 0.95, np.array([0.5, 0.5]))) == 0
