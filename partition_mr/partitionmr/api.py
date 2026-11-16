"""Validated array inputs and structured outputs around the evaluated engines."""
from dataclasses import dataclass
import numpy as np
from ._core import (adaptive_posterior, barycentric_cdf_interval, confidence_set,
                    lattice_cdf_interval, score_p, score_terms, selection)


def _vector(value, length, name, *, positive=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must contain positive values")
    return result


def _ld(value, length):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (length, length) or not np.all(np.isfinite(matrix)):
        raise ValueError("LD must be a finite square matrix matching the variants")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-10):
        raise ValueError("LD must be symmetric")
    if not np.allclose(np.diag(matrix), 1., rtol=0, atol=1e-8):
        raise ValueError("LD must be a correlation matrix with unit diagonal")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("LD must be positive definite") from error
    return matrix


@dataclass(frozen=True)
class PosteriorConfig:
    """Exposure-only sign-mixture parameters and adaptive search settings."""
    mu: float = 2.0
    sigma: float = 1.0
    mass_tolerance: float = 1e-6
    max_nodes: int = 100000

    def __post_init__(self):
        if (not np.isfinite(self.mu) or self.mu < 0 or not np.isfinite(self.sigma)
                or self.sigma <= 0 or not np.isfinite(self.mass_tolerance)
                or not 0 < self.mass_tolerance < 1
                or not isinstance(self.max_nodes, int) or self.max_nodes < 2):
            raise ValueError("Invalid prior parameters or adaptive search settings")


@dataclass(frozen=True)
class ConditionalBlock:
    """An independent candidate block, aligned to one effect-allele orientation.

    z is the complete candidate exposure z-score vector; y is the outcome
    effect vector, not its z score. sx and sy are effect standard errors.
    selected is the exposure-derived instrument mask, with all candidates
    still retained in z, y and ld.
    """
    z: np.ndarray
    y: np.ndarray
    ld: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    selected: np.ndarray | None = None
    psi: float = 0.0
    name: str = "block"


def _block_arrays(block, threshold, clump_r2):
    z = np.asarray(block.z, dtype=float)
    if z.ndim != 1 or len(z) == 0 or not np.all(np.isfinite(z)):
        raise ValueError("z must be a nonempty finite vector")
    n = len(z)
    y = _vector(block.y, n, "y")
    sx = _vector(block.sx, n, "sx", positive=True)
    sy = _vector(block.sy, n, "sy", positive=True)
    R = _ld(block.ld, n)
    if not np.isfinite(block.psi) or abs(block.psi) >= 1:
        raise ValueError("psi must be finite and strictly between -1 and 1")
    if block.selected is None:
        selected = selection(z, R, threshold, clump_r2)[0]
    else:
        mask = np.asarray(block.selected)
        if mask.shape != (n,) or not np.all(np.isin(mask, [False, True])):
            raise ValueError("selected must be a Boolean mask matching the candidates")
        selected = mask.astype(bool)
    return z, y, R, sx, sy, selected


def conditional_test(blocks, beta=0.0, *, config=None, threshold=2.5, clump_r2=0.1,
                     tolerance=1e-3, engine="barycentric", max_work=2000000):
    """Return a two-sided conditional probability bound for a fixed beta.

    Blocks are mutually independent under the stated model. Each block uses
    every supplied candidate in its exposure posterior. A selected mask only
    controls the score. Prior and selection settings are specified without
    outcome-driven fitting. The result includes attained numerical width,
    unexplored posterior mass and separate convergence indicators.
    """
    config = PosteriorConfig() if config is None else config
    if not isinstance(config, PosteriorConfig):
        raise TypeError("config must be a PosteriorConfig")
    blocks = list(blocks)
    if not blocks or not np.isfinite(beta):
        raise ValueError("Provide independent blocks and a finite beta")
    if (not np.isfinite(threshold) or threshold < 0 or not np.isfinite(clump_r2)
            or not 0 <= clump_r2 <= 1 or not np.isfinite(tolerance)
            or not 0 < tolerance < 1 or not isinstance(max_work, int) or max_work < 1):
        raise ValueError("Invalid selection, tolerance or working-memory settings")
    if engine not in {"barycentric", "lattice"}:
        raise ValueError("engine must be barycentric or lattice")
    weights, centers, bounds, block_reports = [], [], [], []
    observed, variance = 0., 0.
    for index, block in enumerate(blocks):
        if not isinstance(block, ConditionalBlock):
            raise TypeError("Each block must be a ConditionalBlock")
        z, y, R, sx, sy, selected = _block_arrays(block, threshold, clump_r2)
        report = {"name": str(block.name), "index": index + 1,
                  "candidates": len(z), "selected": int(selected.sum())}
        if not selected.any():
            report.update(status="empty selection", retained_states=0,
                          visited_nodes=0, omitted_mass_bound=0., converged=True)
            block_reports.append(report)
            continue
        post = adaptive_posterior(z, R, config.mu, config.sigma,
                                  config.mass_tolerance, config.max_nodes)
        t = z * sx / sy ** 2 * selected
        direction = t * (beta * sx - block.psi * sy)
        observed += float(t @ (y - block.psi * sy * z))
        variance += float(direction @ post.within_covariance @ direction
                          + (1 - block.psi ** 2) * (t * sy) @ R @ (t * sy))
        weights.append(post.weights)
        centers.append(post.component_means @ direction)
        bounds.append(post.omitted_mass_bound)
        report.update(status="evaluated", retained_states=len(post.weights),
                      visited_nodes=post.visited_nodes,
                      omitted_mass_bound=post.omitted_mass_bound,
                      converged=bool(post.converged))
        block_reports.append(report)
    if not weights:
        raise ValueError("No instruments selected; lower the exposure threshold or supply a mask")
    function = barycentric_cdf_interval if engine == "barycentric" else lattice_cdf_interval
    argument = "max_work_atoms" if engine == "barycentric" else "max_cells"
    result = function(observed, weights, centers, variance, mass_bounds=bounds,
                      tolerance=tolerance, **{argument: max_work})
    return {"model": "full-candidate sign-mixture conditional score", "engine": engine,
            "beta_tested": float(beta), "requested_cdf_width": float(tolerance),
            "prior": {"mu": config.mu, "sigma": config.sigma},
            "selection": {"threshold": threshold, "clump_r2": clump_r2},
            "candidate_count": sum(r["candidates"] for r in block_reports),
            "selected_count": sum(r["selected"] for r in block_reports),
            "posterior_converged": all(r["converged"] for r in block_reports),
            "blocks": block_reports, **result}


def summary_score(beta_x, se_x, beta_y, se_y, ld, *, prior_variance,
                  psi=0.0, alpha=0.05, prior_covariance="ld"):
    """Evaluate the selected-summary Gaussian score model used in the application.

    prior_variance is the exposure z-scale variance, specified independently
    of outcome data. The LD prior uses C=prior_variance*R; the independent
    prior uses C=prior_variance*I. The returned confidence set can be unbounded
    or disconnected and is preserved as a list of intervals.
    """
    bx = np.asarray(beta_x, dtype=float)
    if bx.ndim != 1 or not len(bx) or not np.all(np.isfinite(bx)):
        raise ValueError("beta_x must be a nonempty finite vector")
    n = len(bx)
    sx, sy = _vector(se_x, n, "se_x", positive=True), _vector(se_y, n, "se_y", positive=True)
    by = _vector(beta_y, n, "beta_y")
    R = _ld(ld, n)
    if (not np.isfinite(prior_variance) or prior_variance <= 0
            or not np.isfinite(psi) or abs(psi) >= 1 or not 0 < alpha < 1):
        raise ValueError("Invalid prior variance, overlap or confidence level")
    if prior_covariance not in {"ld", "independent"}:
        raise ValueError("prior_covariance must be ld or independent")
    C = prior_variance * (R if prior_covariance == "ld" else np.eye(n))
    B = np.linalg.solve((C + R).T, C.T).T
    z = bx / sx
    mean = B @ z
    covariance = B @ R
    covariance = (covariance + covariance.T) / 2
    terms = score_terms(z, by, mean[None], covariance[None], R,
                        np.ones((1, n), bool), sx, sy, psi)[0]
    return {"model": "selected-summary Gaussian conditional score", "n_instruments": n,
            "prior_variance": float(prior_variance), "prior_covariance": prior_covariance,
            "psi": float(psi), "alpha": float(alpha),
            "beta": float(terms[0] / terms[1]) if terms[1] != 0 else None,
            "p": float(score_p(terms[None], 0.)[0]),
            "confidence_set": [list(map(float, interval)) for interval in confidence_set(terms, alpha)],
            "score_coefficients": terms.tolist(), "posterior_mean": mean.tolist(),
            "posterior_sd": np.sqrt(np.maximum(np.diag(covariance), 0.)).tolist()}
