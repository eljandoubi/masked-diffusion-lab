"""Divergences, fits and the exact error of model conditionals."""

from dataclasses import dataclass

import numpy as np

from maskdiff.elbo import pattern_weights
from maskdiff.partial import conditional_table, partial_marginals, state_digits
from maskdiff.sampler import sampler_law, uniform_schedule


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats, accurate for nearby laws; inf if q = 0 where p > 0."""
    p, q = p.reshape(-1), q.reshape(-1)
    keep = p > 0
    p, q = p[keep], q[keep]
    if np.any(q <= 0):
        return float("inf")
    # log1p keeps nearby laws accurate but rounds to -inf when q is far above p
    near = np.abs(p - q) <= 0.5 * q
    log_ratio = np.empty_like(p)
    log_ratio[near] = np.log1p((p[near] - q[near]) / q[near])
    log_ratio[~near] = np.log(p[~near]) - np.log(q[~near])
    return float(np.sum(p * log_ratio))


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
        """Exact (conditional KL, ELBO gap) of model conditionals, shape (S, L, V)."""
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
