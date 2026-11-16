from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats


def corrected_denominator(beta: float, se_x: np.ndarray, se_y: np.ndarray, psi: float = 0.0) -> np.ndarray:
    if not np.isfinite(psi) or abs(float(psi)) > 1.0:
        raise ValueError("psi must be a finite correlation in [-1, 1]")
    den = se_y**2 + (beta**2) * se_x**2 - 2.0 * beta * psi * se_x * se_y
    return np.maximum(den, 1e-12)


def score(df: pd.DataFrame, beta: float, psi: float = 0.0) -> float:
    gx = df["beta_x"].to_numpy(float)
    gy = df["beta_y"].to_numpy(float)
    sx = df["se_x"].to_numpy(float)
    sy = df["se_y"].to_numpy(float)
    den = corrected_denominator(beta, sx, sy, psi)
    return float(np.sum(gx * (gy - beta * gx) / den))


def ivw_corrected(df: pd.DataFrame, psi: float = 0.0) -> dict[str, Any]:
    if len(df) < 1:
        return _empty("IVW_corrected", "not_enough_instruments", len(df))
    def fun(b: float) -> float:
        return score(df, b, psi)
    beta0 = ivw_fixed(df)["beta"]
    if not np.isfinite(beta0):
        beta0 = 0.0
    beta, root_diagnostics = _solve_root_with_diagnostics(fun, beta0)
    root_valid = root_diagnostics.get("root_status") == "unique_bracketed_root"
    se = _delta_se(df, beta, psi) if root_valid else np.nan
    status = "ok" if root_valid else f"invalid_root:{root_diagnostics.get('root_status', 'unknown')}"
    return _record(
        "IVW_corrected",
        beta,
        se,
        len(df),
        status,
        inference_status=(
            "estimating_equation_delta_method"
            if np.isfinite(se)
            else f"invalid_root:{root_diagnostics.get('root_status', 'unknown')}"
            if not root_valid
            else "delta_method_not_estimable"
        ),
        **root_diagnostics,
    )


def ivw_fixed(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 1:
        return _empty("IVW", "not_enough_instruments", len(df))
    x = df["beta_x"].to_numpy(float)
    y = df["beta_y"].to_numpy(float)
    w = 1.0 / np.maximum(df["se_y"].to_numpy(float) ** 2, 1e-12)
    den = float(np.sum(w * x * x))
    if den <= 0:
        return _empty("IVW", "zero_weight", len(df))
    beta = float(np.sum(w * x * y) / den)
    se = math.sqrt(1.0 / den)
    return _record("IVW", beta, se, len(df), "ok")


def egger(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 3:
        return _empty("MR_Egger", "not_enough_instruments", len(df))
    x = df["beta_x"].to_numpy(float)
    y = df["beta_y"].to_numpy(float)
    w = 1.0 / np.maximum(df["se_y"].to_numpy(float) ** 2, 1e-12)
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T * w
    try:
        cov = np.linalg.inv(XtW @ X)
        coef = cov @ (XtW @ y)
        resid = y - X @ coef
        dof = max(len(y) - 2, 1)
        sigma2 = float(np.sum(w * resid**2) / dof)
        se = math.sqrt(max(cov[1, 1] * sigma2, 0.0))
        return _record("MR_Egger", float(coef[1]), se, len(df), "ok", intercept=float(coef[0]))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
        return _empty("MR_Egger", f"failed:{exc}", len(df))


def weighted_median(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 2:
        return _empty("Weighted_median", "not_enough_instruments", len(df))
    ratio, w = _ratio_and_weights(df)
    ok = np.isfinite(ratio) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return _empty("Weighted_median", "not_enough_instruments", len(df))
    ratio = ratio[ok]
    w = w[ok] / np.sum(w[ok])
    order = np.argsort(ratio)
    csum = np.cumsum(w[order])
    beta = float(ratio[order][np.searchsorted(csum, 0.5)])
    return _record(
        "Weighted_median",
        beta,
        np.nan,
        int(ok.sum()),
        "point_estimate_only_internal_approximation",
        inference_status="official_bootstrap_inference_required",
    )


def weighted_mode(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 3:
        return _empty("Weighted_mode", "not_enough_instruments", len(df))
    ratio, w = _ratio_and_weights(df)
    ok = np.isfinite(ratio) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return _empty("Weighted_mode", "not_enough_instruments", len(df))
    ratio = ratio[ok]
    w = w[ok]
    grid = np.linspace(np.nanpercentile(ratio, 2), np.nanpercentile(ratio, 98), 200)
    bw = max(np.nanstd(ratio) * (len(ratio) ** -0.2), 1e-6)
    dens = []
    for g in grid:
        dens.append(float(np.sum(w * np.exp(-0.5 * ((ratio - g) / bw) ** 2))))
    beta = float(grid[int(np.argmax(dens))])
    return _record(
        "Weighted_mode",
        beta,
        np.nan,
        int(ok.sum()),
        "point_estimate_only_internal_approximation",
        inference_status="official_bootstrap_inference_required",
    )


def robust_huber(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 3:
        return _empty("MR_RAPS_like_Huber", "not_enough_instruments", len(df))
    x = df["beta_x"].to_numpy(float)
    y = df["beta_y"].to_numpy(float)
    w = 1.0 / np.maximum(df["se_y"].to_numpy(float) ** 2, 1e-12)
    beta = ivw_fixed(df)["beta"]
    if not np.isfinite(beta):
        beta = 0.0
    for _ in range(100):
        resid = y - beta * x
        scale = np.median(np.abs(resid - np.median(resid))) / 0.6745
        scale = max(float(scale), 1e-8)
        u = resid / (1.345 * scale)
        rw = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), 1e-8))
        ww = w * rw
        den = float(np.sum(ww * x * x))
        if den <= 0:
            break
        new_beta = float(np.sum(ww * x * y) / den)
        if abs(new_beta - beta) < 1e-10:
            beta = new_beta
            break
        beta = new_beta
    return _record(
        "MR_RAPS_like_Huber",
        beta,
        np.nan,
        len(df),
        "python_huber_point_estimate_only",
        inference_status="robust_sandwich_or_official_mr_raps_required",
    )


def wald_ratio(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) != 1:
        return _empty("Wald_ratio", "requires_single_instrument", len(df))
    row = df.iloc[0]
    if row["beta_x"] == 0:
        return _empty("Wald_ratio", "zero_exposure_beta", len(df))
    beta = float(row["beta_y"] / row["beta_x"])
    se = float(abs(beta) * math.sqrt((row["se_y"] / row["beta_y"]) ** 2 + (row["se_x"] / row["beta_x"]) ** 2)) if row["beta_y"] != 0 else float(abs(row["se_y"] / row["beta_x"]))
    return _record("Wald_ratio", beta, se, 1, "ok")


def all_baseline_methods(df: pd.DataFrame, psi: float = 0.0) -> list[dict[str, Any]]:
    methods = [wald_ratio(df), ivw_fixed(df), ivw_corrected(df, psi=psi), egger(df), weighted_median(df), weighted_mode(df), robust_huber(df)]
    return methods


def _ratio_and_weights(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return Wald ratios and inverse delta-method variances.

    Weighting a ratio estimate by ``1 / se_y**2`` ignores its exposure-effect
    denominator and can materially overweight weak instruments. The first-order
    variance below retains uncertainty in both association estimates.
    """
    bx = df["beta_x"].to_numpy(float)
    by = df["beta_y"].to_numpy(float)
    sx = df["se_x"].to_numpy(float)
    sy = df["se_y"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = by / bx
        variance = (sy**2) / (bx**2) + ((by**2) * (sx**2)) / (bx**4)
        weights = 1.0 / variance
    weights[~np.isfinite(weights) | (weights <= 0)] = np.nan
    return ratio, weights


def _solve_root(fun, beta0: float) -> float:
    root, _ = _solve_root_with_diagnostics(fun, beta0)
    return root


def _solve_root_with_diagnostics(fun, beta0: float, grid_fun=None) -> tuple[float, dict[str, Any]]:
    grid = np.linspace(beta0 - 5.0, beta0 + 5.0, 301)
    vals = (
        np.asarray(grid_fun(grid), dtype=float)
        if grid_fun is not None
        else np.array([fun(float(g)) for g in grid])
    )
    return _solve_root_from_grid(fun, beta0, grid, vals)


def _solve_root_from_grid(fun, beta0: float, grid: np.ndarray, vals: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Solve a score equation from a precomputed diagnostic grid.

    Block-bootstrap callers can reuse block-level score curves rather than
    recomputing the same 301-point grid for every resample. Brent refinement
    still evaluates the original scalar score inside every detected bracket.
    """
    grid = np.asarray(grid, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if grid.ndim != 1 or vals.ndim != 1 or len(grid) != len(vals):
        raise ValueError("grid and vals must be one-dimensional arrays of equal length")
    finite = np.isfinite(vals)
    grid = grid[finite]
    vals = vals[finite]
    if len(grid) < 2:
        return float(beta0), {
            "root_status": "insufficient_finite_score_grid",
            "root_candidate_count": 0,
            "root_score_at_solution": np.nan,
            "root_derivative": np.nan,
        }
    signs = np.sign(vals)
    exact = np.flatnonzero(signs == 0)
    candidates = [float(grid[index]) for index in exact]
    brackets: list[tuple[float, int]] = []
    for i in range(len(grid) - 1):
        if signs[i] * signs[i + 1] < 0:
            midpoint = 0.5 * (float(grid[i]) + float(grid[i + 1]))
            brackets.append((abs(midpoint - beta0), i))
    for _, i in brackets:
        try:
            candidates.append(float(optimize.brentq(fun, float(grid[i]), float(grid[i + 1]), maxiter=200)))
        except (ValueError, RuntimeError):
            continue
    candidates = sorted(candidates)
    unique_candidates: list[float] = []
    for candidate in candidates:
        if not unique_candidates or abs(candidate - unique_candidates[-1]) > 1e-7:
            unique_candidates.append(candidate)
    candidates = unique_candidates
    if candidates:
        root = min(candidates, key=lambda value: abs(value - beta0))
        root_status = "unique_bracketed_root" if len(candidates) == 1 else "multiple_roots_nearest_initial"
    else:
        root = float(grid[int(np.argmin(np.abs(vals)))])
        root_status = "no_bracket_minimum_absolute_score"
    step = 1e-5 * max(1.0, abs(root))
    try:
        score_at_root = float(fun(root))
        derivative = float((fun(root + step) - fun(root - step)) / (2.0 * step))
    except (ArithmeticError, ValueError, FloatingPointError):
        score_at_root = np.nan
        derivative = np.nan
    if root_status == "unique_bracketed_root" and (not np.isfinite(derivative) or abs(derivative) < 1e-8):
        root_status = "flat_root"
    return root, {
        "root_status": root_status,
        "root_candidate_count": len(candidates),
        "root_score_at_solution": score_at_root,
        "root_derivative": derivative,
    }


def _delta_se(df: pd.DataFrame, beta: float, psi: float) -> float:
    x = df["beta_x"].to_numpy(float)
    y = df["beta_y"].to_numpy(float)
    sx = df["se_x"].to_numpy(float)
    sy = df["se_y"].to_numpy(float)
    den = corrected_denominator(beta, sx, sy, psi)
    numerator = x * (y - beta * x)
    den_derivative = 2.0 * beta * sx**2 - 2.0 * psi * sx * sy
    score_derivative = float(np.sum(-(x**2) / den - numerator * den_derivative / (den**2)))
    score_variance = float(np.sum((x**2) / den))
    if not np.isfinite(score_derivative) or abs(score_derivative) <= 1e-12 or not np.isfinite(score_variance) or score_variance <= 0:
        return np.nan
    return float(math.sqrt(score_variance) / abs(score_derivative))


def _record(method: str, beta: float, se: float, n: int, status: str, **extra: Any) -> dict[str, Any]:
    z = beta / se if se and np.isfinite(se) and se > 0 else np.nan
    p = 2.0 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    rec = {"method": method, "beta": beta, "se": se, "p": p, "n_instruments": n, "status": status}
    rec.update(extra)
    return rec


def _empty(method: str, reason: str, n_instruments: int = 0) -> dict[str, Any]:
    return {
        "method": method,
        "beta": np.nan,
        "se": np.nan,
        "p": np.nan,
        "n_instruments": int(n_instruments),
        "status": reason,
        "skip_reason": reason,
    }
