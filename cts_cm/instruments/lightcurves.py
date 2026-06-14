from __future__ import annotations

import numpy as np

from cts_cm.aperture.frames import FloatArray, IntArray

_LOG_2PI = float(np.log(2.0 * np.pi))


def _basis(times: FloatArray, degree: int) -> FloatArray:
    scaled = times / (float(times.max()) or 1.0)
    matrix: FloatArray = np.asarray(
        np.vander(scaled, degree + 1, increasing=True), dtype=np.float64
    )
    return matrix


def _row_softmax(scores: FloatArray) -> FloatArray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    normalised: FloatArray = exponent / exponent.sum(axis=1, keepdims=True)
    return normalised


def _row_logsumexp(scores: FloatArray) -> FloatArray:
    peak = scores.max(axis=1, keepdims=True)
    total = np.log(np.exp(scores - peak).sum(axis=1, keepdims=True)) + peak
    reduced: FloatArray = total[:, 0]
    return reduced


class TrajectoryMixture:
    def __init__(
        self,
        n_groups: int,
        poly_degree: int = 3,
        n_starts: int = 20,
        max_iter: int = 200,
        tol: float = 1e-5,
        min_group_frac: float = 0.02,
        min_posterior: float = 0.70,
        seed: int = 0,
    ) -> None:
        self.n_groups = n_groups
        self.poly_degree = poly_degree
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.tol = tol
        self.min_group_frac = min_group_frac
        self.min_posterior = min_posterior
        self.seed = seed
        self.weights_: FloatArray | None = None
        self.coef_: FloatArray | None = None
        self.sigma2_: FloatArray | None = None
        self.loglik_: float = -np.inf

    def _emission(
        self, pain: FloatArray, basis: FloatArray, coef: FloatArray, sigma2: FloatArray
    ) -> FloatArray:
        means = basis @ coef.T
        diff = pain[:, None, :] - means.T[None, :, :]
        sq = (diff**2).sum(axis=2)
        horizon = pain.shape[1]
        emission: FloatArray = -0.5 * (
            horizon * (_LOG_2PI + np.log(sigma2))[None, :] + sq / sigma2[None, :]
        )
        return emission

    def _fit_coef(self, pain: FloatArray, basis: FloatArray, resp: FloatArray) -> FloatArray:
        gram = basis.T @ basis
        counts = resp.sum(axis=0) + 1e-8
        coef = np.zeros((self.n_groups, basis.shape[1]))
        for group in range(self.n_groups):
            weighted_sum = (resp[:, group][:, None] * pain).sum(axis=0)
            rhs = basis.T @ weighted_sum / counts[group]
            coef[group] = np.linalg.solve(gram + 1e-6 * np.eye(gram.shape[0]), rhs)
        return coef

    def _seed_labels(
        self, pain: FloatArray, gen: np.random.Generator, deterministic: bool
    ) -> IntArray:
        if not deterministic:
            random_labels: IntArray = np.asarray(
                gen.integers(0, self.n_groups, pain.shape[0]), dtype=np.int64
            )
            return random_labels
        level = pain.mean(axis=1)
        edges = np.quantile(level, np.linspace(0.0, 1.0, self.n_groups + 1)[1:-1])
        binned: IntArray = np.asarray(np.digitize(level, edges), dtype=np.int64)
        return binned

    def _update_sigma2(
        self, pain: FloatArray, basis: FloatArray, coef: FloatArray, resp: FloatArray
    ) -> FloatArray:
        residual = pain[:, None, :] - (basis @ coef.T).T[None, :, :]
        numerator = (resp[:, :, None] * residual**2).sum(axis=(0, 2))
        denominator = resp.sum(axis=0) * pain.shape[1] + 1e-8
        sigma2: FloatArray = numerator / denominator + 1e-4
        return sigma2

    def _single_start(
        self, pain: FloatArray, basis: FloatArray, gen: np.random.Generator, deterministic: bool
    ) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        n = pain.shape[0]
        labels = self._seed_labels(pain, gen, deterministic)
        resp = np.zeros((n, self.n_groups))
        resp[np.arange(n), labels] = 1.0
        coef = self._fit_coef(pain, basis, resp)
        sigma2 = self._update_sigma2(pain, basis, coef, resp)
        weights = resp.mean(axis=0)
        previous = -np.inf
        loglik = previous
        for _ in range(self.max_iter):
            emission = self._emission(pain, basis, coef, sigma2)
            scored = np.log(weights + 1e-12)[None, :] + emission
            loglik = float(_row_logsumexp(scored).sum())
            resp = _row_softmax(scored)
            weights = resp.mean(axis=0)
            coef = self._fit_coef(pain, basis, resp)
            sigma2 = self._update_sigma2(pain, basis, coef, resp)
            if loglik - previous < self.tol:
                break
            previous = loglik
        return weights, coef, sigma2, loglik

    def fit(self, pain: FloatArray, times: FloatArray) -> TrajectoryMixture:
        basis = _basis(times, self.poly_degree)
        best: tuple[FloatArray, FloatArray, FloatArray, float] | None = None
        for start in range(self.n_starts):
            gen = np.random.default_rng(self.seed + start)
            candidate = self._single_start(pain, basis, gen, deterministic=start == 0)
            if best is None or candidate[3] > best[3]:
                best = candidate
        assert best is not None
        weights, coef, sigma2, loglik = best
        level = (basis @ coef.T).mean(axis=0)
        order = np.argsort(level)
        self.weights_ = weights[order]
        self.coef_ = coef[order]
        self.sigma2_ = sigma2[order]
        self.loglik_ = loglik
        return self

    def responsibilities(self, pain: FloatArray, times: FloatArray) -> FloatArray:
        assert self.weights_ is not None and self.coef_ is not None and self.sigma2_ is not None
        basis = _basis(times, self.poly_degree)
        emission = self._emission(pain, basis, self.coef_, self.sigma2_)
        scored = np.log(self.weights_ + 1e-12)[None, :] + emission
        return _row_softmax(scored)

    def predict(self, pain: FloatArray, times: FloatArray) -> IntArray:
        labels: IntArray = self.responsibilities(pain, times).argmax(axis=1).astype(np.int64)
        return labels

    def n_parameters(self) -> int:
        return self.n_groups * (self.poly_degree + 1) + (self.n_groups - 1) + self.n_groups

    def bic(self, pain: FloatArray) -> float:
        return 2.0 * self.loglik_ - self.n_parameters() * float(np.log(pain.shape[0]))

    def admissible(self, resp: FloatArray) -> bool:
        fractions = resp.mean(axis=0)
        mean_posterior = float(resp.max(axis=1).mean())
        return bool(fractions.min() >= self.min_group_frac and mean_posterior >= self.min_posterior)


def select_group_count(
    pain: FloatArray,
    times: FloatArray,
    grid: tuple[int, ...],
    poly_degree: int = 3,
    n_starts: int = 20,
    seed: int = 0,
    min_group_frac: float = 0.02,
    min_posterior: float = 0.70,
) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    admissible_scores: dict[int, float] = {}
    for k in grid:
        model = TrajectoryMixture(
            n_groups=k,
            poly_degree=poly_degree,
            n_starts=n_starts,
            seed=seed,
            min_group_frac=min_group_frac,
            min_posterior=min_posterior,
        ).fit(pain, times)
        value = model.bic(pain)
        scores[k] = value
        if model.admissible(model.responsibilities(pain, times)):
            admissible_scores[k] = value
    pool = admissible_scores or scores
    best = max(pool, key=lambda key: pool[key])
    return best, scores
