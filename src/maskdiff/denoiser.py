"""A small MLP denoiser for masked sequences and its training loop."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
        onehot = F.one_hot(tokens, self.vocab + 1).float().flatten(1)
        return self.net(onehot).view(-1, self.length, self.vocab)


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


def train(
    data: np.ndarray,
    vocab: int,
    width: int,
    steps: int,
    seed: int,
    batch: int = 512,
    lr: float = 3e-3,
    floor: float = 0.05,
) -> tuple[Denoiser, float]:
    """Minimise the clipped masked cross-entropy; return the model and the tail loss."""
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


@torch.no_grad()
def model_table(model: Denoiser) -> np.ndarray:
    """Conditionals of the network on every partial sequence, shape (S, L, V)."""
    tokens = torch.as_tensor(state_digits(model.vocab, model.length))
    return torch.softmax(model(tokens).double(), dim=-1).numpy()
