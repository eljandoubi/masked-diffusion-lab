"""Divergences, fits and the exact error of model conditionals."""

from dataclasses import dataclass

import numpy as np

from maskdiff.elbo import pattern_weights
from maskdiff.partial import conditional_table, partial_marginals, state_digits
from maskdiff.sampler import sampler_law, uniform_schedule


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats; q must be positive wherever p is."""
    p, q = p.reshape(-1), q.reshape(-1)
    keep = p > 0
    return float(np.sum(p[keep] * np.log1p((p[keep] - q[keep]) / q[keep])))


def loglog_slope(
    x: np.ndarray, y: np.ndarray, lo: float, hi: float
) -> tuple[float, float]:
    """Least-squares slope of log y on log x over lo <= x <= hi, and the fit R^2."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x >= lo) & (x <= hi) & (y > 0)
    if keep.sum() < 2:
        raise ValueError("need at least two points in the fit range")
    lx, ly = np.log(x[keep]), np.log(y[keep])
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    spread = ((ly - ly.mean()) ** 2).sum()
    r2 = 1.0 - (resid**2).sum() / spread if spread > 0 else 1.0
    return float(slope), float(r2)


@dataclass(frozen=True)
class Reference:
    """A data law with its exact partial marginals and exact conditionals."""

    law: np.ndarray
    mass: np.ndarray
    table: np.ndarray

    @classmethod
    def from_law(cls, law: np.ndarray) -> "Reference":
        marginals = partial_marginals(law)
        return cls(law, marginals.reshape(-1), conditional_table(marginals))

    @property
    def vocab(self) -> int:
        return self.law.shape[0]

    @property
    def length(self) -> int:
        return self.law.ndim

    def conditional_kl(self, model_table: np.ndarray) -> tuple[float, float]:
        """Exact (conditional KL, ELBO gap) of model conditionals of shape (S, L, V).

        The gap is the expected negative ELBO minus the entropy of the law. It sums the
        KL between true and model conditionals over masked positions of every partial
        sequence, weighted by the forward process with weight 1/t. The conditional KL
        averages the same KL over a uniform t and the masked positions of the pattern,
        which is the gap divided by L + 1.
        """
        masked = state_digits(self.vocab, self.length) == self.vocab
        true, model = self.table[masked], model_table[masked]
        per_position = np.zeros(masked.shape)
        per_position[masked] = (true * np.log(true / model)).sum(axis=-1)
        weight = pattern_weights(self.length)[masked.sum(axis=1)]
        gap = float((weight * self.mass * per_position.sum(axis=1)).sum())
        return gap / (self.length + 1), gap

    def sampler_kl(self, model_table: np.ndarray, steps: int) -> float:
        """KL(data || K-step sampler) when the sampler uses the model conditionals."""
        schedule = uniform_schedule(steps, self.length)
        law = sampler_law(model_table, self.vocab, self.length, schedule)
        return kl(self.law, law)
