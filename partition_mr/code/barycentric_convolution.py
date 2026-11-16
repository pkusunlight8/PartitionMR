"""Positive-mixture convolution with a second-order Gaussian CDF error bound."""
import heapq
import math
import numpy as np
from scipy.special import ndtr
from adaptive_partition import combined_omitted_mass, conservative_two_sided


def compress_positive_mixture(weights, centers, loss_budget):
    """Merge adjacent atoms at their conditional barycenter within an M2 budget."""
    weights, centers = np.asarray(weights, float), np.asarray(centers, float)
    if (weights.ndim != 1 or centers.shape != weights.shape or not len(weights)
            or np.any(~np.isfinite(weights)) or np.any(~np.isfinite(centers))
            or np.any(weights < 0) or not np.isfinite(loss_budget) or loss_budget < 0
            or not np.isclose(weights.sum(), 1., atol=1e-10, rtol=1e-10)):
        raise ValueError('Finite normalized positive weights and an M2 budget required')
    keep = weights > 0
    order = np.argsort(centers[keep], kind='stable')
    x, w = centers[keep][order], weights[keep][order]
    unique, start = np.unique(x, return_index=True)
    w = np.add.reduceat(w, start)
    x = unique
    n = len(x)
    if n == 1 or loss_budget == 0:
        return w / w.sum(), x, 0.
    left, right = np.arange(n) - 1, np.arange(n) + 1
    right[-1] = -1
    generation = np.zeros(n, int)
    alive = np.ones(n, bool)
    heap = []

    def push(a, b):
        if a >= 0 and b >= 0:
            cost = (w[a] / (w[a] + w[b])) * w[b] * (x[b] - x[a]) ** 2
            heapq.heappush(heap, (float(cost), a, b, generation[a], generation[b]))

    for a in range(n - 1):
        push(a, a + 1)
    losses = []
    loss = 0.
    while heap:
        cost, a, b, ga, gb = heapq.heappop(heap)
        if (not alive[a] or not alive[b] or right[a] != b
                or ga != generation[a] or gb != generation[b]):
            continue
        candidate = math.fsum([loss, cost])
        if candidate > loss_budget:
            break
        total = w[a] + w[b]
        x[a] += (w[b] / total) * (x[b] - x[a])
        w[a] = total
        alive[b] = False
        generation[a] += 1
        right[a] = right[b]
        if right[b] >= 0:
            left[right[b]] = a
        losses.append(cost)
        loss = candidate
        push(left[a], a)
        push(a, right[a])
    return w[alive] / w[alive].sum(), x[alive], math.fsum(losses)


def barycentric_cdf_interval(observed, weights, component_means, within_variance,
                             mass_bounds=None, tolerance=1e-3, max_work_atoms=2000000):
    """Enclose a Gaussian-smoothed sum using positive barycentric compression.

    In standardized coordinates, sup |d2 Phi(t-x)/dx2| = 1/sqrt(2*pi*e).
    A conditional barycenter cancels the linear Taylor term. Positive
    convolution preserves this bound, so half that constant times the sum
    of weighted within-cluster variances bounds the absolute CDF error.
    """
    weights = [np.asarray(w, float) for w in weights]
    means = [np.asarray(x, float) for x in component_means]
    if (not weights or len(weights) != len(means) or not np.isfinite(observed)
            or not np.isfinite(within_variance) or within_variance <= 0
            or not np.isfinite(tolerance) or not 0 < tolerance < 1
            or not isinstance(max_work_atoms, int) or max_work_atoms < 1):
        raise ValueError('Finite mixture blocks, positive variance, and valid budgets required')
    for w, x in zip(weights, means):
        if (w.ndim != 1 or x.shape != w.shape or not len(w) or np.any(w < 0)
                or np.any(~np.isfinite(w)) or np.any(~np.isfinite(x))
                or not np.isclose(w.sum(), 1., atol=1e-10, rtol=1e-10)):
            raise ValueError('Each block requires finite normalized positive weights')
    bounds = [0.] * len(weights) if mass_bounds is None else mass_bounds
    if len(bounds) != len(weights):
        raise ValueError('One omitted-mass bound is required per block')
    r = combined_omitted_mass(bounds)
    masks = [w > 0 for w in weights]
    weights = [w[keep] for w, keep in zip(weights, masks)]
    means = [x[keep] for x, keep in zip(means, masks)]
    sd = math.sqrt(within_variance)
    origins = [float(x[0]) for x in means]
    observed = math.fsum([float(observed), *[-a for a in origins]]) / sd
    means = [(x - a) / sd for x, a in zip(means, origins)]
    if not np.isfinite(observed) or any(np.any(~np.isfinite(x)) for x in means):
        raise ValueError('Standardized locations must be finite')
    coefficient = 1. / (2. * math.sqrt(2. * math.pi * math.e))
    # Reserve a small implementation allowance separately from the analytic term.
    guard = 2e-12 * (1 + len(weights))
    available = .99 * max(0., tolerance - r - 2 * guard)
    loss_budget = available / (2 * (1 - r) * coefficient) if r < 1 else 0.
    w, x, loss = np.ones(1), np.zeros(1), 0.
    peak, operations, steps = 1, 0, []
    for b, (wb, xb) in enumerate(zip(weights, means)):
        size = len(w) * len(wb)
        if size > max_work_atoms:
            raise ArithmeticError('Convolution exceeds the explicit working-atom budget')
        wc = (w[:, None] * wb).ravel()
        xc = (x[:, None] + xb).ravel()
        peak, operations = max(peak, size), operations + size
        per_step = max(0., loss_budget - loss) / (len(weights) - b)
        w, x, increment = compress_positive_mixture(wc / wc.sum(), xc, per_step)
        loss = math.fsum([loss, increment])
        steps.append({'block': b + 1, 'input_atoms': size, 'output_atoms': len(w),
                      'local_standardized_m2': increment})
    center = float(w @ ndtr(observed - x))
    error = coefficient * loss
    operation_allowance = np.finfo(float).eps * (64 * operations + 16 * len(weights))
    envelope = 1 + abs(observed) + math.fsum(float(np.max(np.abs(v))) for v in means)
    guard = min(1., guard + operation_allowance * envelope)
    low = max(0., (1 - r) * max(0., center - error) - guard)
    high = min(1., (1 - r) * min(1., center + error) + r + guard)
    return {'cdf_low': low, 'cdf_high': high, 'cdf_midpoint': (low + high) / 2,
            'cdf_barycentric': center, 'p_upper': conservative_two_sided(low, high),
            'width': high - low, 'cdf_error_bound': error,
            'standardized_m2': loss, 'retained_atoms': len(w), 'peak_atoms': peak,
            'convolution_atoms': operations, 'joint_states': math.prod(map(len, weights)),
            'omitted_mass_bound': r, 'numerical_guard': guard,
            'converged': bool(high - low <= tolerance * (1 + 1e-10)), 'steps': steps}


def bernstein_positive_projection(weights, centers, degree):
    """Finite positive Bernstein projection, preserving mass and first moment.

    This is the discrete specialization of the positive Bernstein weights in
    OpenAI (2026), Universal optimality of the triangular lattice, Section 7,
    Lemma Positive mixtures. The finite Gaussian-CDF error calculation is here.
    """
    from scipy.stats import binom
    w, x = np.asarray(weights, float), np.asarray(centers, float)
    if (w.ndim != 1 or x.shape != w.shape or not len(w)
            or np.any(~np.isfinite(w)) or np.any(~np.isfinite(x)) or np.any(w < 0)
            or not np.isclose(w.sum(), 1., atol=1e-10, rtol=1e-10)
            or not isinstance(degree, int) or degree < 1):
        raise ValueError('Finite normalized mixture and a positive integer degree required')
    keep = w > 0
    w, x = w[keep], x[keep]
    a, b = float(x.min()), float(x.max())
    if a == b:
        return np.ones(1), np.array([a]), 0.
    span = b - a
    if not math.isfinite(span):
        raise ValueError('The positive-mass support range must be finite')
    u = (x - a) / span
    probabilities = binom.pmf(np.arange(degree + 1)[None, :], degree, u[:, None])
    projected = w @ probabilities
    projected /= projected.sum()
    nodes = a + span * (np.arange(degree + 1) / degree)
    spread = span * math.sqrt(float(w @ (u * (1 - u))) / degree)
    loss = spread * spread
    if (not np.all(np.isfinite(projected)) or not np.all(np.isfinite(nodes))
            or not math.isfinite(loss)):
        raise ValueError('Projected probabilities, locations, and variance must be finite')
    return projected, nodes, loss
