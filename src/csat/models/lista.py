"""LISTA -- Learned ISTA (Gregor & LeCun 2010; the paper's reference [37]).

THE PAPER'S EQUATION (Section 3)
    alpha_i^{(t+1)} = eta_theta( S alpha_i^{(t)} + B Z_i )
where "S, B are learned weight matrices, eta_theta is a learned soft-thresholding
function, and t is the number of iterations (layers)".

HOW THIS FOLLOWS FROM ISTA  (the derivation the notebook walks through)
    ISTA:   alpha_{t+1} = S_theta( alpha_t - eta A^T(A alpha_t - y) )
                        = S_theta( (I - eta A^T A) alpha_t + eta A^T y )
    Define  W_s := I - eta A^T A   in R^{k x k}
            W_e := eta A^T         in R^{k x p}
    then    alpha_{t+1} = S_theta( W_s alpha_t + W_e y ).
    LISTA drops the constraint that W_s and W_e come from a single A and learns
    them by back-propagation, with theta learned too. The paper's (S, B) are
    exactly (W_s, W_e).

WHAT THE PAPER LEAVES UNSPECIFIED -- all [MISSING]
    * depth t, weight tying, threshold parametrisation;
    * the training objective (supervise alpha? supervise Psi alpha? end-to-end
      through the transformer loss?);
    * the data LISTA is trained on and whether it is retrained per layer/task;
    * the optimiser, learning rate, and schedule.
Every choice below is ours and is flagged in the config.

COST NOTE
    W_s is k x k, so ONE LISTA layer costs O(k^2) per token, whereas one ISTA
    iteration costs O(pk). LISTA is cheaper only because it uses far fewer
    layers than ISTA needs iterations -- not because a layer is cheaper. The
    FLOP counter (utils/flops.py) makes this explicit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ista import estimate_step_size, soft_threshold


class LISTA(nn.Module):
    """Unrolled, learnable ISTA.

    Args:
        p: measurement dimension (rows of A).
        k: number of dictionary atoms (columns of A).
        n_layers: unrolled depth t.
        A: optional [p, k] operator used to initialise W_s, W_e from ISTA. With
            ``init_from_ista=True`` the network STARTS as exact ISTA, so training
            can only improve on it -- this makes the ISTA-vs-LISTA comparison
            meaningful instead of a comparison against a random network.
        lam: the l1 weight whose ISTA threshold initialises the learned threshold.
        tied_weights: share one (W_s, W_e, theta) across all layers.
        learn_threshold: learn theta (per layer, per coordinate) or keep it fixed.
    """

    def __init__(self, p: int, k: int, n_layers: int = 8,
                 A: torch.Tensor | None = None, lam: float = 0.1,
                 tied_weights: bool = False, learn_threshold: bool = True,
                 init_from_ista: bool = True):
        super().__init__()
        self.p, self.k, self.n_layers = p, k, n_layers
        self.tied_weights, self.learn_threshold = tied_weights, learn_threshold

        if init_from_ista and A is not None:
            eta = estimate_step_size(A)
            w_s0 = torch.eye(k, device=A.device, dtype=A.dtype) - eta * (A.T @ A)
            w_e0 = eta * A.T                                     # [k, p]
            theta0 = torch.full((k,), eta * lam, device=A.device, dtype=A.dtype)
        else:
            w_s0 = torch.eye(k) + 0.01 * torch.randn(k, k)
            w_e0 = 0.01 * torch.randn(k, p)
            theta0 = torch.full((k,), 0.01)

        n_sets = 1 if tied_weights else n_layers
        self.W_s = nn.ParameterList(
            [nn.Parameter(w_s0.clone()) for _ in range(n_sets)])
        self.W_e = nn.ParameterList(
            [nn.Parameter(w_e0.clone()) for _ in range(n_sets)])
        thetas = [theta0.clone() for _ in range(n_sets)]
        if learn_threshold:
            self.theta = nn.ParameterList([nn.Parameter(t) for t in thetas])
        else:
            for i, t in enumerate(thetas):
                self.register_buffer(f"theta_{i}", t)
            self.theta = [getattr(self, f"theta_{i}") for i in range(n_sets)]

    def _layer_params(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = 0 if self.tied_weights else layer
        return self.W_s[i], self.W_e[i], self.theta[i]

    def forward(self, y: torch.Tensor, return_all: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """y: [N, p] -> alpha: [N, k]. Differentiable end to end."""
        assert y.shape[-1] == self.p, f"expected measurements of dim {self.p}, got {y.shape[-1]}"
        alpha = torch.zeros(y.shape[0], self.k, device=y.device, dtype=y.dtype)
        iterates = []
        for layer in range(self.n_layers):
            w_s, w_e, theta = self._layer_params(layer)
            alpha = soft_threshold(alpha @ w_s.T + y @ w_e.T, theta.abs())
            if return_all:
                iterates.append(alpha)
        return (alpha, iterates) if return_all else alpha

    def extra_repr(self) -> str:
        return (f"p={self.p}, k={self.k}, layers={self.n_layers}, "
                f"tied={self.tied_weights}, learn_theta={self.learn_threshold}")


class LISTADecoder(nn.Module):
    """LISTA + synthesis, so it is a drop-in replacement for ISTADecoder.

    forward(y: [..., p]) -> C_hat = Psi alpha_hat : [..., d]
    """

    def __init__(self, lista: LISTA, psi: torch.Tensor, learn_psi: bool = False):
        super().__init__()
        self.lista = lista
        if learn_psi:
            self.psi = nn.Parameter(psi.clone())
        else:
            self.register_buffer("psi", psi)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        lead, p = y.shape[:-1], y.shape[-1]
        alpha = self.lista(y.reshape(-1, p))
        return (alpha @ self.psi.T).reshape(*lead, self.psi.shape[0])


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_lista(model: LISTA, y_train: torch.Tensor, target_train: torch.Tensor,
                y_val: torch.Tensor, target_val: torch.Tensor,
                psi: torch.Tensor | None = None,
                supervision: str = "alpha", n_epochs: int = 40,
                batch_size: int = 128, lr: float = 1e-3,
                weight_decay: float = 0.0, verbose: bool = True,
                log_every: int = 10) -> dict[str, object]:
    """Supervised training of LISTA.

    supervision:
        'alpha'  -- MSE against the true sparse code (classical LISTA, requires
                    ground-truth alpha, available only for synthetic data).
        'signal' -- MSE against Psi alpha_true, i.e. supervise the reconstructed
                    signal. Needs ``psi``. This is the setting that would apply
                    inside a transformer, where the true code is unknown.

    [MISSING] The paper never states which of these it used, nor the optimiser.
    We use Adam, the standard choice for LISTA-style unrolled networks.
    """
    assert supervision in ("alpha", "signal")
    if supervision == "signal" and psi is None:
        raise ValueError("supervision='signal' requires the dictionary psi")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    n = y_train.shape[0]

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=y_train.device)
        running, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred_alpha = model(y_train[idx])
            pred = pred_alpha if supervision == "alpha" else pred_alpha @ psi.T
            loss = loss_fn(pred, target_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            pv = model(y_val)
            pv = pv if supervision == "alpha" else pv @ psi.T
            val_loss = loss_fn(pv, target_val).item()

        history["epoch"].append(epoch)
        history["train_loss"].append(running / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d} | train {history['train_loss'][-1]:.6f} "
                  f"| val {val_loss:.6f}")

    return history
