"""Adaptive sign-partition evaluation with an omitted-mass upper bound.

The reported error certificate bounds the omitted posterior state mass.
"""
from dataclasses import dataclass
import heapq
import itertools
import numpy as np
from scipy.special import logsumexp, ndtr


@dataclass
class AdaptivePosterior:
    mean: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    component_means: np.ndarray
    within_covariance: np.ndarray
    states: np.ndarray
    omitted_mass_bound: float
    visited_nodes: int
    converged: bool


def adaptive_posterior(z, R, mu=2., sigma=1., tolerance=1e-4,
                       max_nodes=100000, order=None, check_every=32):
    z = np.asarray(z, float)
    R = np.asarray(R, float)
    m = len(z)
    if z.ndim != 1 or R.shape != (m, m) or m < 1:
        raise ValueError('A vector and its square covariance are required')
    if not 0 < tolerance < 1 or sigma <= 0 or mu < 0:
        raise ValueError('Invalid model or tolerance')
    if max_nodes < m + 1 or check_every < 1:
        raise ValueError('Search budget must permit at least one complete state')
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(R)):
        raise ValueError('Finite inputs required')
    if not np.allclose(R, R.T, atol=1e-12):
        raise ValueError('Symmetric covariance required')
    np.linalg.cholesky(R)
    inv = np.linalg.inv(R + sigma ** 2 * np.eye(m))
    B = sigma ** 2 * inv
    # Multiplication avoids subtracting nearly equal broad-prior covariances.
    Q = B @ R
    Q = (Q + Q.T) / 2
    if mu == 0:
        return AdaptivePosterior(B @ z, Q, np.ones(1), (B @ z)[None], Q,
                                 np.zeros((1, m)), 0., 0, True)
    h = mu * inv @ z
    J = -mu ** 2 * inv
    np.fill_diagonal(J, 0.)
    if order is None:
        order = np.argsort(-np.abs(h), kind='stable')
    order = np.asarray(order, int)
    if sorted(order.tolist()) != list(range(m)):
        raise ValueError('Order must be a candidate permutation')
    h, J = h[order], J[np.ix_(order, order)]
    remaining_coupling = np.array([np.abs(np.triu(J[k:, k:], 1)).sum()
                                   for k in range(m + 1)])

    def upper(energy, fields, depth):
        return float(energy + np.logaddexp(fields, -fields).sum()
                     + remaining_coupling[depth])

    serial = itertools.count()
    heap = []

    def push(energy, fields, depth, prefix):
        maximum = float(energy + np.abs(fields).sum() + remaining_coupling[depth])
        heapq.heappush(heap, (-maximum, -depth, next(serial), prefix, fields,
                             energy, upper(energy, fields, depth)))

    # A greedy leaf seeds the denominator; its sibling subtrees partition all
    # remaining states, including when a diffuse model exhausts the budget.
    prefix, fields, energy = (), h.copy(), 0.
    for depth in range(m):
        preferred = 1. if fields[0] >= 0 else -1.
        other = -preferred
        push(energy + other * fields[0], fields[1:] + other * J[depth, depth + 1:],
             depth + 1, prefix + (other,))
        energy += preferred * fields[0]
        fields = fields[1:] + preferred * J[depth, depth + 1:]
        prefix += (preferred,)
    signs, logweights = [prefix], [energy]
    log_kept, residual = energy, 1.
    visited = m + 1
    # Heap nodes partition the unexplored sign cube without overlap.
    while heap and visited < max_nodes:
        _, negative_depth, _, prefix, fields, energy, _ = heapq.heappop(heap)
        depth = -negative_depth
        visited += 1
        if depth == m:
            signs.append(prefix)
            logweights.append(energy)
            log_kept = np.logaddexp(log_kept, energy)
        else:
            for sign in (-1., 1.):
                child_energy = energy + sign * fields[0]
                child_fields = fields[1:] + sign * J[depth, depth + 1:]
                push(child_energy, child_fields, depth + 1, prefix + (sign,))
        if signs and (visited % check_every == 0 or not heap):
            log_upper = logsumexp([node[-1] for node in heap]) if heap else -np.inf
            residual = float(np.exp(log_upper - np.logaddexp(log_kept, log_upper)))
            # Round upward to avoid advertising a smaller floating-point bound.
            residual = min(1., np.nextafter(residual, np.inf) * (1 + 1e-12))
            if residual <= tolerance:
                break
    if not signs:
        raise ArithmeticError('No complete state reached within the search budget')
    if heap:
        log_upper = logsumexp([node[-1] for node in heap])
        residual = min(1., np.nextafter(float(np.exp(log_upper - np.logaddexp(
            log_kept, log_upper))), np.inf) * (1 + 1e-12))
    else:
        residual = 0.
    state_array = np.empty((len(signs), m))
    state_array[:, order] = np.asarray(signs)
    weights = np.exp(np.asarray(logweights) - logsumexp(logweights))
    means = z @ B.T + mu * state_array @ (R @ inv).T
    mean = weights @ means
    delta = means - mean
    covariance = Q + (delta * weights[:, None]).T @ delta
    return AdaptivePosterior(mean, covariance, weights, means, Q, state_array,
                             residual, visited, residual <= tolerance)


def cdf_interval(result, direction, observed, residual_variance):
    direction = np.asarray(direction, float)
    variance = float(direction @ result.within_covariance @ direction + residual_variance)
    if variance <= 0:
        raise ValueError('Positive within-component score variance required')
    centers = result.component_means @ direction
    value = float(result.weights @ ndtr((observed - centers) / np.sqrt(variance)))
    r = result.omitted_mass_bound
    # F_full=(1-r_true)*F_kept+r_true*F_discarded, where r_true <= r.
    return value, max(0., (1 - r) * value), min(1., (1 - r) * value + r)


def conservative_two_sided(cdf_low, cdf_high):
    if not 0 <= cdf_low <= cdf_high <= 1:
        raise ValueError('CDF endpoints must be ordered probabilities')
    if cdf_low <= .5 <= cdf_high:
        return 1.
    return float(min(1., 2 * min(cdf_high, 1 - cdf_low)))


def combined_omitted_mass(bounds):
    values = np.asarray(bounds, float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('Mass bounds must lie in [0, 1]')
    return float(-np.expm1(np.log1p(-values).sum()))


def lattice_cdf_interval(observed, weights, component_means, within_variance,
                         mass_bounds=None, tolerance=1e-3, max_cells=2000000,
                         max_refinements=18):
    """Enclose a Gaussian-smoothed mixture convolution by interval rounding.

    Each block mean is rounded down on a lattice. Summed true means lie between
    that lattice sum and the sum plus B*step. Gaussian CDF monotonicity gives
    lower/upper bounds; unknown omitted states enlarge the interval by their
    joint mass upper bound. Refinement uses this mass bound to quantify
    truncation error.
    """
    from scipy.signal import fftconvolve
    import math
    weights = [np.asarray(w, float) for w in weights]
    means = [np.asarray(x, float) for x in component_means]
    if (len(weights) != len(means) or not weights or within_variance <= 0
            or not np.isfinite(within_variance) or not np.isfinite(observed)):
        raise ValueError('Mixture blocks and positive residual variance required')
    for w, x in zip(weights, means):
        if (len(w) != len(x) or not len(w) or np.any(w < 0)
                or np.any(~np.isfinite(w)) or np.any(~np.isfinite(x))
                or not np.isclose(w.sum(), 1)):
            raise ValueError('Each mixture needs matching means and normalized weights')
    bounds = [0.] * len(weights) if mass_bounds is None else mass_bounds
    if len(bounds) != len(weights):
        raise ValueError('One omitted-mass bound is required per mixture block')
    r = combined_omitted_mass(bounds)
    sd = np.sqrt(within_variance)
    origins = [float(x.min()) for x in means]
    observation = math.fsum([float(observed), *[-a for a in origins]]) / sd
    means = [(x - a) / sd for x, a in zip(means, origins)]
    minima = [0.] * len(means)
    base = 0.
    step = .5
    result = None
    for refinement in range(max_refinements + 1):
        indices = [np.floor(np.maximum(0., x - a) / step).astype(np.int64)
                   for x, a in zip(means, minima)]
        cells = 1 + sum(int(index.max()) for index in indices)
        if cells > max_cells:
            if result is None:
                raise ArithmeticError('Initial convolution exceeds the lattice budget')
            break
        distribution = np.ones(1)
        negative_mass = 0.
        for w, index in zip(weights, indices):
            block = np.bincount(index, weights=w)
            if len(distribution) * len(block) <= 16384:
                distribution = np.convolve(distribution, block)
            else:
                distribution = fftconvolve(distribution, block)
            negative_mass += float(-distribution[distribution < 0].sum())
            distribution = np.maximum(distribution, 0.)
            distribution /= distribution.sum()
        centers = base + np.arange(len(distribution)) * step
        lower = float(distribution @ ndtr(observation - centers - len(weights) * step))
        upper = float(distribution @ ndtr(observation - centers))
        # Conservative numerical guard is tracked separately from rounding.
        rounding_guard = 2e-12 * (1 + len(weights)) + 2 * negative_mass
        low = max(0., (1 - r) * lower - rounding_guard)
        high = min(1., (1 - r) * upper + r + rounding_guard)
        result = {'cdf_low': low, 'cdf_high': high, 'cdf_midpoint': (low + high) / 2,
                  'p_upper': conservative_two_sided(low, high),
                  'width': high - low, 'rounding_step': step * sd,
                  'cells': cells, 'refinements': refinement,
                  'omitted_mass_bound': r, 'numerical_guard': rounding_guard,
                  'negative_fft_mass': negative_mass,
                  'converged': high - low <= tolerance}
        if result['converged']:
            break
        step /= 2
    return result
