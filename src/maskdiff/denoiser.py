"""A small MLP denoiser for masked sequences and its two training loops."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maskdiff.elbo import loss_targets
from maskdiff.partial import state_digits


class Denoiser(nn.Module):
    """MLP on one-hot tokens (mask = class V) with one logit vector per position."""

    def __init__(self, length: int, vocab: int, width: int, depth: int = 2) -> None:
        super().__init__()
        self.length, self.vocab = length, vocab
        sizes = [length * (vocab + 1)] + [width] * depth
        layers: list[nn.Module] = []
        for size_in, size_out in zip(sizes, sizes[1:], strict=False):
            layers += [nn.Linear(size_in, size_out), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], length * vocab))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(tokens, self.vocab + 1).flatten(1)
        return self.net(onehot.to(self.net[0].weight.dtype)).view(
            -1, self.length, self.vocab
        )


def masked_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    masked: torch.Tensor,
    t: torch.Tensor,
    floor: float,
) -> torch.Tensor:
    """Per sequence: cross-entropy summed over masked positions, over max(t, floor)."""
    ce = F.cross_entropy(logits.flatten(0, 1), x.flatten(), reduction="none")
    return (ce.view_as(x) * masked).sum(dim=1) / t.clamp(min=floor)


def train_sampled(
    data: np.ndarray,
    vocab: int,
    width: int,
    steps: int,
    seed: int,
    batch: int = 512,
    lr: float = 3e-3,
    floor: float = 0.05,
) -> tuple[Denoiser, float]:
    """Adam on sampled rows, t and masks; returns the model and its tail loss."""
    torch.manual_seed(seed)
    rows = torch.as_tensor(data, dtype=torch.long)
    n, length = rows.shape
    model = Denoiser(length, vocab, width)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=steps, eta_min=lr / 100
    )
    tail = torch.zeros(())
    for step in range(steps):
        x = rows[torch.randint(n, (batch,))]
        t = torch.rand(batch)
        masked = torch.rand(batch, length) < t[:, None]
        loss = masked_loss(model(x.masked_fill(masked, vocab)), x, masked, t, floor)
        optimiser.zero_grad(set_to_none=True)
        loss.mean().backward()
        optimiser.step()
        decay.step()
        if step >= steps - 200:
            tail += loss.detach().mean()
    return model, float(tail) / min(200, steps)


def train_exact(
    law: np.ndarray,
    width: int,
    seed: int,
    iterations: int = 2000,
    floor: float = 0.05,
) -> tuple[Denoiser, float]:
    """L-BFGS on the exact expectation of the clipped loss under law, full batch."""
    torch.manual_seed(seed)
    length, vocab = law.ndim, law.shape[0]
    digits = state_digits(vocab, length)
    keep = (digits == vocab).any(axis=1)
    tokens = torch.as_tensor(digits[keep])
    target = torch.as_tensor(loss_targets(law, floor)[keep])
    model = Denoiser(length, vocab, width).double()
    optimiser = torch.optim.LBFGS(
        model.parameters(),
        max_iter=iterations,
        max_eval=2 * iterations,
        history_size=30,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = -(target * F.log_softmax(model(tokens), dim=-1)).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    return model, closure().item()


@torch.no_grad()
def model_table(model: Denoiser) -> np.ndarray:
    """Conditionals of the network on every partial sequence, shape (S, L, V)."""
    tokens = torch.as_tensor(state_digits(model.vocab, model.length))
    return torch.softmax(model(tokens).double(), dim=-1).numpy()
