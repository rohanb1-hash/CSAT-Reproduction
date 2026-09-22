"""Unit tests for LISTA: the ISTA-equivalence at initialisation, and training."""

from __future__ import annotations

import torch

from csat.models.ista import estimate_step_size, ista
from csat.models.lista import LISTA, LISTADecoder, train_lista
from csat.utils.metrics import relative_l2
from csat.utils.tensor_utils import make_sparse_signals


def _problem(p=32, k=64, s=5, n=256, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(p, k) / (p ** 0.5)
    A = A / A.norm(dim=0, keepdim=True)
    alpha = make_sparse_signals(n, k, s)
    return A, alpha, alpha @ A.T


def test_shapes_and_depth():
    A, alpha, y = _problem()
    net = LISTA(p=A.shape[0], k=A.shape[1], n_layers=5, A=A, lam=0.01)
    out, iterates = net(y, return_all=True)
    assert out.shape == alpha.shape
    assert len(iterates) == 5, "one iterate per unrolled layer"


def test_lista_at_initialisation_equals_ista():
    """Establishes the derivation in the module docstring:
        W_s = I - eta A^T A,  W_e = eta A^T,  theta = eta*lambda
    makes the unrolled network EXACTLY t steps of ISTA. This is the test that
    proves LISTA is a re-parameterisation of ISTA rather than a new algorithm.
    """
    A, _, y = _problem()
    lam, layers = 0.01, 6
    net = LISTA(p=A.shape[0], k=A.shape[1], n_layers=layers, A=A, lam=lam,
                init_from_ista=True)
    with torch.no_grad():
        lista_out = net(y)
    ista_out = ista(A, y, lam=lam, n_iters=layers, step_size=estimate_step_size(A))["alpha"]
    assert torch.allclose(lista_out, ista_out, atol=1e-5), \
        (lista_out - ista_out).abs().max().item()


def test_tied_weights_reduce_parameter_count():
    A, _, _ = _problem()
    untied = LISTA(p=A.shape[0], k=A.shape[1], n_layers=6, A=A)
    tied = LISTA(p=A.shape[0], k=A.shape[1], n_layers=6, A=A, tied_weights=True)
    n_untied = sum(p.numel() for p in untied.parameters())
    n_tied = sum(p.numel() for p in tied.parameters())
    assert n_tied * 5 < n_untied <= n_tied * 6 + 1


def test_gradients_reach_every_layer():
    """Establishes: the unrolled network is differentiable end to end, unlike
    classical ISTA.

    One structural exception, which this test pins down rather than papers over:
    the FIRST layer's W_s receives exactly zero gradient, because alpha^(0) = 0
    and the term W_s alpha^(0) is identically zero no matter what W_s is. So an
    untied t-layer LISTA has (t-1) effective W_s matrices, and a k x k block of
    parameters in layer 0 is dead weight. The paper's Section 3 recurrence has
    the same property and does not mention it.
    """
    A, alpha, y = _problem()
    net = LISTA(p=A.shape[0], k=A.shape[1], n_layers=4, A=A)
    loss = torch.nn.functional.mse_loss(net(y), alpha)
    loss.backward()

    assert net.W_s[0].grad is not None
    assert net.W_s[0].grad.abs().sum() == 0, \
        "layer-0 W_s multiplies the zero initial iterate, so it cannot receive gradient"
    for i, w in enumerate(net.W_s[1:], start=1):
        assert w.grad is not None and w.grad.abs().sum() > 0, f"layer {i} W_s"
    for i, w in enumerate(net.W_e):
        assert w.grad is not None and w.grad.abs().sum() > 0, f"layer {i} W_e"


def test_training_improves_on_the_ista_initialisation():
    """Establishes: because training STARTS at exact ISTA, any improvement is a
    genuine gain from learning rather than an artefact of a lucky baseline."""
    A, alpha, y = _problem(n=512)
    tr, va = slice(0, 384), slice(384, 512)
    net = LISTA(p=A.shape[0], k=A.shape[1], n_layers=5, A=A, lam=0.01)
    with torch.no_grad():
        before = relative_l2(net(y[va]), alpha[va])
    train_lista(net, y[tr], alpha[tr], y[va], alpha[va], n_epochs=30,
                batch_size=64, lr=1e-3, verbose=False)
    with torch.no_grad():
        after = relative_l2(net(y[va]), alpha[va])
    assert after < before, f"LISTA did not improve: {before:.4f} -> {after:.4f}"


def test_decoder_wrapper_returns_signal_space():
    """Establishes: LISTADecoder maps measurements to C_hat = Psi alpha_hat with
    the right trailing dimension, including on batched [B,H,N,p] input."""
    A, _, y = _problem()
    psi = torch.randn(48, A.shape[1])
    dec = LISTADecoder(LISTA(p=A.shape[0], k=A.shape[1], n_layers=3, A=A), psi)
    out = dec(y.view(2, 2, -1, A.shape[0]))
    assert out.shape[:-1] == (2, 2, y.shape[0] // 4)
    assert out.shape[-1] == 48
