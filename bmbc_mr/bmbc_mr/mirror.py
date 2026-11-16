from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised by the NumPy fallback environment
    njit = None

from .estimators import (
    _solve_root_with_diagnostics,
    corrected_denominator,
    ivw_corrected,
    score,
)
from .ld_mirror import discovery_conditional_mirror_replicates
from .utils import stable_seed

BOOTSTRAP_ROOT_GRID_POINTS = 61
BOOTSTRAP_ROOT_REFINEMENT_ITERATIONS = 5

if njit is not None:

    @njit(cache=True)
    def _bmbc_score_grid_numba(
        gx: np.ndarray,
        gy: np.ndarray,
        sx: np.ndarray,
        sy: np.ndarray,
        mirror_beta_x: np.ndarray,
        mirror_beta_x_residual: np.ndarray,
        mirror_beta_y: np.ndarray,
        mirror_se_x: np.ndarray,
        mirror_se_y: np.ndarray,
        beta_grid: np.ndarray,
        psi: float,
        constant_mirror_standard_errors: bool,
    ) -> np.ndarray:
        n_grid = len(beta_grid)
        n_instruments, n_mirrors = mirror_beta_x.shape
        values = np.zeros(n_grid, dtype=np.float64)
        if constant_mirror_standard_errors:
            mirror_intercept = np.zeros(n_instruments, dtype=np.float64)
            mirror_slope = np.zeros(n_instruments, dtype=np.float64)
            for instrument_index in range(n_instruments):
                for mirror_index in range(n_mirrors):
                    mirror_intercept[instrument_index] += (
                        mirror_beta_x[instrument_index, mirror_index]
                        * mirror_beta_y[instrument_index, mirror_index]
                    )
                    mirror_slope[instrument_index] += (
                        mirror_beta_x[instrument_index, mirror_index]
                        * mirror_beta_x_residual[instrument_index, mirror_index]
                    )
                mirror_intercept[instrument_index] /= n_mirrors
                mirror_slope[instrument_index] /= n_mirrors
            for grid_index in range(n_grid):
                beta = beta_grid[grid_index]
                score_value = 0.0
                for instrument_index in range(n_instruments):
                    real_denominator = max(
                        sy[instrument_index] * sy[instrument_index]
                        + beta * beta * sx[instrument_index] * sx[instrument_index]
                        - 2.0
                        * beta
                        * psi
                        * sx[instrument_index]
                        * sy[instrument_index],
                        1e-12,
                    )
                    real_component = gx[instrument_index] * (
                        gy[instrument_index] - beta * gx[instrument_index]
                    ) / real_denominator
                    mirror_denominator = max(
                        mirror_se_y[instrument_index, 0]
                        * mirror_se_y[instrument_index, 0]
                        + beta
                        * beta
                        * mirror_se_x[instrument_index, 0]
                        * mirror_se_x[instrument_index, 0]
                        - 2.0
                        * beta
                        * psi
                        * mirror_se_x[instrument_index, 0]
                        * mirror_se_y[instrument_index, 0],
                        1e-12,
                    )
                    mirror_component = (
                        mirror_intercept[instrument_index]
                        - beta * mirror_slope[instrument_index]
                    ) / mirror_denominator
                    score_value += real_component - mirror_component
                values[grid_index] = score_value
            return values

        for grid_index in range(n_grid):
            beta = beta_grid[grid_index]
            real_score = 0.0
            mirror_score = 0.0
            for instrument_index in range(n_instruments):
                real_denominator = max(
                    sy[instrument_index] * sy[instrument_index]
                    + beta * beta * sx[instrument_index] * sx[instrument_index]
                    - 2.0
                    * beta
                    * psi
                    * sx[instrument_index]
                    * sy[instrument_index],
                    1e-12,
                )
                real_score += gx[instrument_index] * (
                    gy[instrument_index] - beta * gx[instrument_index]
                ) / real_denominator
                for mirror_index in range(n_mirrors):
                    mirror_denominator = max(
                        mirror_se_y[instrument_index, mirror_index]
                        * mirror_se_y[instrument_index, mirror_index]
                        + beta
                        * beta
                        * mirror_se_x[instrument_index, mirror_index]
                        * mirror_se_x[instrument_index, mirror_index]
                        - 2.0
                        * beta
                        * psi
                        * mirror_se_x[instrument_index, mirror_index]
                        * mirror_se_y[instrument_index, mirror_index],
                        1e-12,
                    )
                    mirror_score += mirror_beta_x[instrument_index, mirror_index] * (
                        mirror_beta_y[instrument_index, mirror_index]
                        - beta
                        * mirror_beta_x_residual[instrument_index, mirror_index]
                    ) / mirror_denominator
            values[grid_index] = real_score - mirror_score / n_mirrors
        return values

    @njit(cache=True)
    def _block_score_curves_numba(
        gx: np.ndarray,
        gy: np.ndarray,
        sx: np.ndarray,
        sy: np.ndarray,
        mirror_beta_x: np.ndarray,
        mirror_beta_x_residual: np.ndarray,
        mirror_beta_y: np.ndarray,
        mirror_se_x: np.ndarray,
        mirror_se_y: np.ndarray,
        beta_grid: np.ndarray,
        psi: float,
        block_codes: np.ndarray,
        n_blocks: int,
        constant_mirror_standard_errors: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_grid = len(beta_grid)
        n_instruments, n_mirrors = mirror_beta_x.shape
        real_curves = np.zeros((n_blocks, n_grid), dtype=np.float64)
        mirror_curves = np.zeros(
            (n_blocks, n_mirrors, n_grid),
            dtype=np.float64,
        )
        inverse_mirror_denominator = np.empty((n_instruments, n_grid))
        if constant_mirror_standard_errors:
            for instrument_index in range(n_instruments):
                for grid_index in range(n_grid):
                    beta = beta_grid[grid_index]
                    inverse_mirror_denominator[instrument_index, grid_index] = 1.0 / max(
                        mirror_se_y[instrument_index, 0]
                        * mirror_se_y[instrument_index, 0]
                        + beta
                        * beta
                        * mirror_se_x[instrument_index, 0]
                        * mirror_se_x[instrument_index, 0]
                        - 2.0
                        * beta
                        * psi
                        * mirror_se_x[instrument_index, 0]
                        * mirror_se_y[instrument_index, 0],
                        1e-12,
                    )
        for instrument_index in range(n_instruments):
            block = block_codes[instrument_index]
            for grid_index in range(n_grid):
                beta = beta_grid[grid_index]
                real_denominator = max(
                    sy[instrument_index] * sy[instrument_index]
                    + beta * beta * sx[instrument_index] * sx[instrument_index]
                    - 2.0
                    * beta
                    * psi
                    * sx[instrument_index]
                    * sy[instrument_index],
                    1e-12,
                )
                real_curves[block, grid_index] += gx[instrument_index] * (
                    gy[instrument_index] - beta * gx[instrument_index]
                ) / real_denominator
            for mirror_index in range(n_mirrors):
                mirror_intercept = (
                    mirror_beta_x[instrument_index, mirror_index]
                    * mirror_beta_y[instrument_index, mirror_index]
                )
                mirror_slope = (
                    mirror_beta_x[instrument_index, mirror_index]
                    * mirror_beta_x_residual[instrument_index, mirror_index]
                )
                for grid_index in range(n_grid):
                    beta = beta_grid[grid_index]
                    if constant_mirror_standard_errors:
                        inverse_denominator = inverse_mirror_denominator[
                            instrument_index,
                            grid_index,
                        ]
                    else:
                        denominator = max(
                            mirror_se_y[instrument_index, mirror_index]
                            * mirror_se_y[instrument_index, mirror_index]
                            + beta
                            * beta
                            * mirror_se_x[instrument_index, mirror_index]
                            * mirror_se_x[instrument_index, mirror_index]
                            - 2.0
                            * beta
                            * psi
                            * mirror_se_x[instrument_index, mirror_index]
                            * mirror_se_y[instrument_index, mirror_index],
                            1e-12,
                        )
                        inverse_denominator = 1.0 / denominator
                    mirror_curves[block, mirror_index, grid_index] += (
                        mirror_intercept - beta * mirror_slope
                    ) * inverse_denominator
        return real_curves, mirror_curves

    @njit(cache=True)
    def _weighted_block_score_derivative_numba(
        gx: np.ndarray,
        gy: np.ndarray,
        sx: np.ndarray,
        sy: np.ndarray,
        mirror_intercept: np.ndarray,
        mirror_slope: np.ndarray,
        mirror_se_x_squared: np.ndarray,
        mirror_se_y_squared: np.ndarray,
        mirror_se_x_y: np.ndarray,
        beta: np.ndarray,
        psi: float,
        block_codes: np.ndarray,
        block_counts: np.ndarray,
        mirror_weights: np.ndarray,
        compute_derivative: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_roots = len(beta)
        n_instruments, n_mirrors = mirror_intercept.shape
        scores = np.zeros(n_roots, dtype=np.float64)
        derivatives = np.zeros(n_roots, dtype=np.float64)
        for root_index in range(n_roots):
            value = beta[root_index]
            score_value = 0.0
            derivative_value = 0.0
            for instrument_index in range(n_instruments):
                instrument_weight = block_counts[
                    root_index,
                    block_codes[instrument_index],
                ]
                if instrument_weight == 0.0:
                    continue
                se_x_value = sx[instrument_index]
                se_y_value = sy[instrument_index]
                denominator = (
                    se_y_value * se_y_value
                    + value * value * se_x_value * se_x_value
                    - 2.0 * value * psi * se_x_value * se_y_value
                )
                denominator = max(denominator, 1e-12)
                numerator = gx[instrument_index] * (
                    gy[instrument_index] - value * gx[instrument_index]
                )
                real_component = numerator / denominator
                mirror_component = 0.0
                mirror_derivative = 0.0
                for mirror_index in range(n_mirrors):
                    mirror_weight = mirror_weights[root_index, mirror_index]
                    if mirror_weight == 0.0:
                        continue
                    mirror_denominator = (
                        mirror_se_y_squared[instrument_index, mirror_index]
                        + value
                        * value
                        * mirror_se_x_squared[instrument_index, mirror_index]
                        - 2.0
                        * value
                        * psi
                        * mirror_se_x_y[instrument_index, mirror_index]
                    )
                    mirror_denominator = max(mirror_denominator, 1e-12)
                    mirror_numerator = (
                        mirror_intercept[instrument_index, mirror_index]
                        - value * mirror_slope[instrument_index, mirror_index]
                    )
                    mirror_component += (
                        mirror_weight * mirror_numerator / mirror_denominator
                    )
                    if compute_derivative:
                        mirror_denominator_derivative = (
                            2.0
                            * value
                            * mirror_se_x_squared[instrument_index, mirror_index]
                            - 2.0
                            * psi
                            * mirror_se_x_y[instrument_index, mirror_index]
                        )
                        mirror_derivative += mirror_weight * (
                            -mirror_slope[instrument_index, mirror_index]
                            / mirror_denominator
                            - mirror_numerator
                            * mirror_denominator_derivative
                            / (mirror_denominator * mirror_denominator)
                        )
                score_value += instrument_weight * (
                    real_component - mirror_component
                )
                if compute_derivative:
                    denominator_derivative = (
                        2.0 * value * se_x_value * se_x_value
                        - 2.0 * psi * se_x_value * se_y_value
                    )
                    real_derivative = (
                        -(gx[instrument_index] * gx[instrument_index]) / denominator
                        - numerator
                        * denominator_derivative
                        / (denominator * denominator)
                    )
                    derivative_value += instrument_weight * (
                        real_derivative - mirror_derivative
                    )
            scores[root_index] = score_value
            derivatives[root_index] = derivative_value
        return scores, derivatives

else:
    _bmbc_score_grid_numba = None
    _block_score_curves_numba = None
    _weighted_block_score_derivative_numba = None


def bootstrap_score_backend() -> str:
    """Report the score kernel selected by the current runtime contract."""
    use_numba = (
        _weighted_block_score_derivative_numba is not None
        and os.environ.get("BMBC_USE_NUMBA", "1").strip().lower()
        not in {"0", "false", "no"}
    )
    return "numba_fused_single_thread" if use_numba else "numpy_vectorized"


def bootstrap_root_refinement_backend(
    replicates: dict[str, np.ndarray] | None,
) -> str:
    """Report whether exact constant-SE mirror preaggregation is available."""
    if replicates is None:
        return "not_available_no_mirror_replicates"
    mirror_se_x = np.asarray(replicates.get("se_x", np.empty((0, 0))), dtype=float)
    mirror_se_y = np.asarray(replicates.get("se_y", np.empty((0, 0))), dtype=float)
    if _mirror_standard_errors_are_replicate_constant(mirror_se_x, mirror_se_y):
        return "exact_constant_mirror_se_numerator_preaggregation"
    return "general_repeat_specific_mirror_se"


def _mirror_standard_errors_are_replicate_constant(
    mirror_se_x: np.ndarray,
    mirror_se_y: np.ndarray,
) -> bool:
    return bool(
        mirror_se_x.ndim == 2
        and mirror_se_y.ndim == 2
        and len(mirror_se_x) > 0
        and len(mirror_se_y) > 0
        and np.array_equal(mirror_se_x, np.broadcast_to(mirror_se_x[0], mirror_se_x.shape))
        and np.array_equal(mirror_se_y, np.broadcast_to(mirror_se_y[0], mirror_se_y.shape))
    )


def _matching_effect_allele_frequency(real: pd.DataFrame) -> tuple[np.ndarray, str, int]:
    """Use exposure EAF when available, otherwise the aligned outcome EAF."""
    n = len(real)
    exposure = (
        pd.to_numeric(real["eaf_x"], errors="coerce").to_numpy(float)
        if "eaf_x" in real
        else np.full(n, np.nan)
    )
    outcome = (
        pd.to_numeric(real["eaf_y"], errors="coerce").to_numpy(float)
        if "eaf_y" in real
        else np.full(n, np.nan)
    )
    exposure_valid = np.isfinite(exposure) & (exposure > 0.0) & (exposure < 1.0)
    outcome_valid = np.isfinite(outcome) & (outcome > 0.0) & (outcome < 1.0)
    fallback = ~exposure_valid & outcome_valid
    matching_eaf = np.where(exposure_valid, exposure, np.where(outcome_valid, outcome, np.nan))
    fallback_count = int(fallback.sum())
    if not np.isfinite(matching_eaf).all():
        source = "incomplete_exposure_and_aligned_outcome_effect_allele_frequency"
    elif fallback_count == 0:
        source = "exposure_effect_allele_frequency"
    elif fallback_count == n:
        source = "aligned_outcome_effect_allele_frequency_fallback"
    else:
        source = "mixed_exposure_and_aligned_outcome_effect_allele_frequency"
    return matching_eaf, source, fallback_count


def _group_indices_by_block(blocks: pd.Series) -> dict[str, np.ndarray]:
    frame = pd.DataFrame({"block": pd.Series(blocks).astype(str).to_numpy()})
    return {
        str(block): np.asarray(indices, dtype=int)
        for block, indices in frame.groupby("block", sort=False).indices.items()
    }


def make_mirror_replicates(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    repeats: int,
    seed: int,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if harmonized.empty or mirror_pool.empty:
        return [], {"mirror_status": "empty_pool", "mean_abs_eaf_diff": np.nan, "n_mirror_variants": 0}
    real = harmonized.copy()
    real["block"] = real["block"].astype(str)
    real_snps = set(real["SNP"].dropna().astype(str)) if "SNP" in real else set()
    pool = mirror_pool.copy()
    excluded_real_snps = int(pool["SNP"].astype(str).isin(real_snps).sum()) if "SNP" in pool else 0
    if "SNP" in pool and real_snps:
        pool = pool[~pool["SNP"].astype(str).isin(real_snps)].copy()
    if pool.empty:
        return [], {
            "mirror_status": "empty_pool",
            "mean_abs_eaf_diff": np.nan,
            "n_mirror_variants": 0,
            "excluded_real_snps_from_pool": excluded_real_snps,
        }
    pool["block"] = pool["block"].astype(str)
    matching_eaf, matching_frequency_source, fallback_count = _matching_effect_allele_frequency(real)
    if not np.isfinite(matching_eaf).all():
        return [], {
            "mirror_status": "invalid_or_missing_real_maf",
            "mean_abs_eaf_diff": np.nan,
            "mean_abs_maf_diff": np.nan,
            "n_mirror_variants": 0,
            "excluded_real_snps_from_pool": excluded_real_snps,
            "matching_frequency_source": matching_frequency_source,
            "real_frequency_fallback_instruments": fallback_count,
        }
    reps: list[pd.DataFrame] = []
    eaf_diffs: list[float] = []
    maf_diffs: list[float] = []
    for _ in range(int(repeats)):
        rows = []
        for instrument_index, (_, inst) in enumerate(real.iterrows()):
            candidates = pool[pool["block"] == inst["block"]]
            if candidates.empty:
                candidates = pool
            eaf = matching_eaf[instrument_index]
            maf = _minor_allele_frequency(eaf)
            if np.isfinite(maf):
                candidate_maf = pd.to_numeric(candidates["eaf"], errors="coerce").map(_minor_allele_frequency)
                candidates = candidates.assign(_dist=(candidate_maf - float(maf)).abs())
                candidates = candidates.nsmallest(min(50, len(candidates)), "_dist")
            chosen = candidates.sample(n=1, replace=True, random_state=int(rng.integers(1, 2**31 - 1))).iloc[0]
            pseudo = inst.copy()
            pseudo["SNP"] = f"mirror_{chosen['SNP']}"
            pseudo["chr"] = chosen["chr"]
            pseudo["pos"] = chosen["pos"]
            pseudo["eaf_y"] = chosen["eaf"]
            pseudo["beta_y"] = chosen["beta"]
            pseudo["se_y"] = chosen["se"]
            pseudo["block"] = chosen["block"]
            sign = -1.0 if rng.random() < 0.5 else 1.0
            donor_pool = real[real["block"] == inst["block"]]
            if donor_pool.empty:
                donor_pool = real
            donor = donor_pool.sample(n=1, replace=True, random_state=int(rng.integers(1, 2**31 - 1))).iloc[0]
            pseudo["beta_x"] = sign * abs(float(donor["beta_x"]))
            pseudo["se_x"] = float(donor["se_x"])
            pseudo["p_x"] = float(donor["p_x"])
            if np.isfinite(maf):
                try:
                    chosen_eaf = float(chosen["eaf"])
                    eaf_diffs.append(abs(float(eaf) - chosen_eaf))
                    maf_diffs.append(abs(float(maf) - _minor_allele_frequency(chosen_eaf)))
                except (TypeError, ValueError):
                    pass
            rows.append(pseudo)
        reps.append(pd.DataFrame(rows))
    diag = {
        "mirror_status": "ok",
        "mean_abs_eaf_diff": float(np.mean(eaf_diffs)) if eaf_diffs else np.nan,
        "mean_abs_maf_diff": float(np.mean(maf_diffs)) if maf_diffs else np.nan,
        "matching_frequency_metric": "minor_allele_frequency",
        "matching_frequency_source": matching_frequency_source,
        "real_frequency_fallback_instruments": fallback_count,
        "n_mirror_variants": int(sum(len(r) for r in reps)),
        "excluded_real_snps_from_pool": excluded_real_snps,
    }
    return reps, diag


def make_mirror_array_replicates(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    repeats: int,
    seed: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if harmonized.empty or mirror_pool.empty:
        return None, {"mirror_status": "empty_pool", "mean_abs_eaf_diff": np.nan, "n_mirror_variants": 0}
    if int(repeats) < 1:
        return None, {"mirror_status": "invalid_mirror_repeat_count", "mean_abs_eaf_diff": np.nan, "n_mirror_variants": 0}

    real = harmonized.reset_index(drop=True).copy()
    required_real_columns = {"block", "beta_x", "se_x", "beta_y", "se_y"}
    missing_real_columns = sorted(required_real_columns.difference(real.columns))
    if not {"eaf_x", "eaf_y"}.intersection(real.columns):
        missing_real_columns.append("eaf_x_or_eaf_y")
    if missing_real_columns:
        return None, {
            "mirror_status": "invalid_real_schema",
            "mirror_schema_missing": ",".join(missing_real_columns),
            "mean_abs_eaf_diff": np.nan,
            "mean_abs_maf_diff": np.nan,
            "n_mirror_variants": 0,
        }
    real_snps = set(real["SNP"].dropna().astype(str)) if "SNP" in real else set()
    pool = mirror_pool.reset_index(drop=True).copy()
    excluded_real_snps = int(pool["SNP"].astype(str).isin(real_snps).sum()) if "SNP" in pool else 0
    if "SNP" in pool and real_snps:
        pool = pool[~pool["SNP"].astype(str).isin(real_snps)].reset_index(drop=True)
    required_pool_columns = {"block", "eaf", "beta", "se"}
    missing_pool_columns = sorted(required_pool_columns.difference(pool.columns))
    if missing_pool_columns:
        return None, {
            "mirror_status": "invalid_pool_schema",
            "mirror_schema_missing": ",".join(missing_pool_columns),
            "mean_abs_eaf_diff": np.nan,
            "n_mirror_variants": 0,
            "excluded_real_snps_from_pool": excluded_real_snps,
        }
    pool["beta"] = pd.to_numeric(pool["beta"], errors="coerce")
    pool["se"] = pd.to_numeric(pool["se"], errors="coerce")
    pool["eaf"] = pd.to_numeric(pool["eaf"], errors="coerce")
    valid_pool = (
        np.isfinite(pool["beta"])
        & np.isfinite(pool["se"])
        & (pool["se"] > 0)
        & np.isfinite(pool["eaf"])
        & pool["eaf"].between(0.0, 1.0, inclusive="neither")
        & pool["block"].notna()
    )
    has_exposure = {"beta_x", "se_x"}.issubset(pool.columns)
    if has_exposure:
        pool["beta_x"] = pd.to_numeric(pool["beta_x"], errors="coerce")
        pool["se_x"] = pd.to_numeric(pool["se_x"], errors="coerce")
        valid_pool &= np.isfinite(pool["beta_x"]) & np.isfinite(pool["se_x"]) & (pool["se_x"] > 0)
    pool = pool[valid_pool].reset_index(drop=True)
    if pool.empty:
        return None, {
            "mirror_status": "empty_pool_after_validity_filter",
            "mean_abs_eaf_diff": np.nan,
            "n_mirror_variants": 0,
            "excluded_real_snps_from_pool": excluded_real_snps,
        }
    pool_eaf = pd.to_numeric(pool["eaf"], errors="coerce").to_numpy(float)
    pool_maf = np.minimum(pool_eaf, 1.0 - pool_eaf)
    pool_beta_y = pd.to_numeric(pool["beta"], errors="coerce").to_numpy(float)
    pool_se_y = pd.to_numeric(pool["se"], errors="coerce").to_numpy(float)
    pool_ld = pd.to_numeric(pool.get("ld_score", np.nan), errors="coerce").to_numpy(float) if "ld_score" in pool else np.full(len(pool), np.nan)
    pool_beta_x = pd.to_numeric(pool["beta_x"], errors="coerce").to_numpy(float) if has_exposure else np.full(len(pool), np.nan)
    pool_se_x = pd.to_numeric(pool["se_x"], errors="coerce").to_numpy(float) if has_exposure else np.full(len(pool), np.nan)
    has_residual = has_exposure and "beta_x_residual" in pool
    pool_beta_x_residual = pd.to_numeric(pool["beta_x_residual"], errors="coerce").to_numpy(float) if has_residual else pool_beta_x
    groups = _group_indices_by_block(pool["block"])
    global_idx = np.arange(len(pool))

    real_blocks = real["block"].astype(str).to_numpy()
    real_eaf, matching_frequency_source, fallback_count = _matching_effect_allele_frequency(real)
    real_maf = np.minimum(real_eaf, 1.0 - real_eaf)
    if not np.isfinite(real_maf).all() or np.any((real_eaf <= 0.0) | (real_eaf >= 1.0)):
        return None, {
            "mirror_status": "invalid_or_missing_real_maf",
            "mean_abs_eaf_diff": np.nan,
            "mean_abs_maf_diff": np.nan,
            "n_mirror_variants": 0,
            "excluded_real_snps_from_pool": excluded_real_snps,
            "matching_frequency_source": matching_frequency_source,
            "real_frequency_fallback_instruments": fallback_count,
        }
    real_ld = pd.to_numeric(real.get("ld_score", np.nan), errors="coerce").to_numpy(float) if "ld_score" in real else np.full(len(real), np.nan)
    n = len(real)
    r = int(repeats)
    beta_x = np.empty((r, n), dtype=float)
    se_x = np.empty((r, n), dtype=float)
    beta_x_residual = np.empty((r, n), dtype=float)
    beta_y = np.empty((r, n), dtype=float)
    se_y = np.empty((r, n), dtype=float)
    matched_maf: list[float] = []
    matched_eaf: list[float] = []
    matched_se: list[float] = []
    matched_ld: list[float] = []
    matched_z: list[float] = []
    target_maf: list[float] = []
    target_eaf: list[float] = []
    target_se: list[float] = []
    target_ld: list[float] = []
    target_z: list[float] = []
    global_fallback_instruments = 0
    real_abs_beta = np.abs(real["beta_x"].to_numpy(float))
    real_se = real["se_x"].to_numpy(float)
    real_groups = _group_indices_by_block(real["block"])
    for j in range(n):
        idx_pool = groups.get(real_blocks[j])
        if idx_pool is None or len(idx_pool) == 0:
            idx_pool = global_idx
            global_fallback_instruments += 1
        distance_terms = []
        for values, target in [
            (pool_maf[idx_pool], real_maf[j]),
            (pool_se_y[idx_pool], float(real["se_y"].iloc[j])),
            (pool_ld[idx_pool], real_ld[j]),
        ]:
            finite = np.isfinite(values)
            if np.isfinite(target) and finite.any():
                scale = float(np.nanstd(values[finite]))
                scale = scale if np.isfinite(scale) and scale > 0 else 1.0
                term = np.full(len(idx_pool), 10.0)
                term[finite] = np.abs(values[finite] - float(target)) / scale
                distance_terms.append(term)
        if has_exposure:
            mirror_z = np.abs(pool_beta_x[idx_pool] / np.maximum(pool_se_x[idx_pool], 1e-12))
            target_z_value = abs(float(real["beta_x"].iloc[j]) / max(float(real["se_x"].iloc[j]), 1e-12))
            finite = np.isfinite(mirror_z)
            if finite.any():
                scale = float(np.nanstd(mirror_z[finite]))
                scale = scale if np.isfinite(scale) and scale > 0 else 1.0
                term = np.full(len(idx_pool), 10.0)
                term[finite] = np.abs(mirror_z[finite] - target_z_value) / scale
                distance_terms.append(term)
        choice_probabilities = None
        if distance_terms:
            distances = np.nanmean(np.vstack(distance_terms), axis=0)
            order = np.argsort(distances, kind="mergesort")[: min(20, len(idx_pool))]
            idx_pool = idx_pool[order]
            ranked_distances = distances[order]
            weights = np.exp(-5.0 * np.clip(ranked_distances - np.nanmin(ranked_distances), 0.0, 50.0))
            if np.isfinite(weights).all() and weights.sum() > 0:
                choice_probabilities = weights / weights.sum()
        chosen_idx = rng.choice(idx_pool, size=r, replace=True, p=choice_probabilities)
        if has_exposure:
            beta_x[:, j] = pool_beta_x[chosen_idx]
            beta_x_residual[:, j] = pool_beta_x_residual[chosen_idx]
            se_x[:, j] = pool_se_x[chosen_idx]
        else:
            donor_pool = real_groups.get(real_blocks[j], np.array([j], dtype=int))
            donor_idx = rng.choice(donor_pool, size=r, replace=True)
            signs = np.where(rng.random(r) < 0.5, -1.0, 1.0)
            beta_x[:, j] = signs * real_abs_beta[donor_idx]
            beta_x_residual[:, j] = beta_x[:, j]
            se_x[:, j] = real_se[donor_idx]
        beta_y[:, j] = pool_beta_y[chosen_idx]
        se_y[:, j] = pool_se_y[chosen_idx]
        matched_maf.extend(pool_maf[chosen_idx].tolist())
        matched_eaf.extend(pool_eaf[chosen_idx].tolist())
        matched_se.extend(pool_se_y[chosen_idx].tolist())
        matched_ld.extend(pool_ld[chosen_idx].tolist())
        matched_z.extend(np.abs(pool_beta_x[chosen_idx] / np.maximum(pool_se_x[chosen_idx], 1e-12)).tolist() if has_exposure else [np.nan] * r)
        target_maf.extend([real_maf[j]] * r)
        target_eaf.extend([real_eaf[j]] * r)
        target_se.extend([float(real["se_y"].iloc[j])] * r)
        target_ld.extend([real_ld[j]] * r)
        target_z.extend([abs(float(real["beta_x"].iloc[j]) / max(float(real["se_x"].iloc[j]), 1e-12))] * r)
    reps = {"beta_x": beta_x, "beta_x_residual": beta_x_residual, "se_x": se_x, "beta_y": beta_y, "se_y": se_y}
    eaf_smd = _standardized_difference(np.asarray(target_eaf), np.asarray(matched_eaf))
    maf_smd = _standardized_difference(np.asarray(target_maf), np.asarray(matched_maf))
    se_smd = _standardized_difference(np.asarray(target_se), np.asarray(matched_se))
    ld_smd = _standardized_difference(np.asarray(target_ld), np.asarray(matched_ld))
    exposure_z_smd = _standardized_difference(np.asarray(target_z), np.asarray(matched_z)) if has_exposure else np.nan
    smds = [abs(v) for v in [maf_smd, se_smd, ld_smd, exposure_z_smd] if not np.isnan(v)]
    max_abs_smd = max(smds) if smds else np.nan
    matching_covariates = ["block", "maf", "se"]
    if np.isfinite(pool_ld).any() and np.isfinite(real_ld).any():
        matching_covariates.append("ld_score")
    if has_exposure:
        matching_covariates.append("exposure_z")
    if global_fallback_instruments > 0:
        status = "imbalanced_global_fallback"
    elif max_abs_smd > 0.10:
        status = "imbalanced"
    elif not has_exposure or "ld_score" not in matching_covariates:
        status = "ok_degraded_matching"
    else:
        status = "ok"
    diag = {
        "mirror_status": status,
        "mean_abs_eaf_diff": float(np.nanmean(np.abs(np.asarray(target_eaf) - np.asarray(matched_eaf)))),
        "mean_abs_maf_diff": float(np.nanmean(np.abs(np.asarray(target_maf) - np.asarray(matched_maf)))),
        "n_mirror_variants": int(r * n),
        "eaf_smd": eaf_smd,
        "maf_smd": maf_smd,
        "se_smd": se_smd,
        "ld_score_smd": ld_smd,
        "exposure_z_smd": exposure_z_smd,
        "max_abs_smd": max_abs_smd,
        "balance_threshold": 0.10,
        "global_fallback_instruments": int(global_fallback_instruments),
        "excluded_real_snps_from_pool": excluded_real_snps,
        "matching_covariates": ",".join(matching_covariates),
        "matching_frequency_metric": "minor_allele_frequency",
        "matching_frequency_source": matching_frequency_source,
        "real_frequency_fallback_instruments": fallback_count,
        "ld_structure_status": "marginal_ld_score_matched_pairwise_ld_not_verified",
        "mirror_exposure_source": "selected_full_exposure_with_error_residual" if has_residual else ("conditional_null_pool" if has_exposure else "within_block_selected_permutation"),
    }
    return reps, diag


def mirror_bias(replicates: list[pd.DataFrame], beta: float, psi: float) -> float:
    if not replicates:
        return 0.0
    vals = [score(rep, beta, psi) for rep in replicates if len(rep) > 0]
    if not vals:
        return 0.0
    return float(np.nanmean(vals))


def mirror_bias_arrays(replicates: dict[str, np.ndarray] | None, beta: float, psi: float) -> float:
    if replicates is None:
        return 0.0
    bx = replicates["beta_x"]
    bx_residual = replicates.get("beta_x_residual", bx)
    by = replicates["beta_y"]
    sx = replicates["se_x"]
    sy = replicates["se_y"]
    den = corrected_denominator(beta, sx, sy, psi)
    vals = np.sum(bx * (by - beta * bx_residual) / den, axis=1)
    return float(np.nanmean(vals))


def block_bootstrap_inference(beta: float, se: float, effective_blocks: float) -> dict[str, float | str]:
    """Return finite-block Student t inference for a block-bootstrap standard error."""
    if not np.isfinite(beta) or not np.isfinite(se) or se <= 0 or not np.isfinite(effective_blocks) or float(effective_blocks) < 2.0:
        return {
            "p": np.nan,
            "inference_distribution": "",
            "inference_degrees_of_freedom": np.nan,
            "inference_critical_value_95": np.nan,
        }
    degrees_of_freedom = float(effective_blocks) - 1.0
    statistic = float(beta) / float(se)
    return {
        "p": float(2.0 * stats.t.sf(abs(statistic), degrees_of_freedom)),
        "inference_distribution": "student_t_block_df",
        "inference_degrees_of_freedom": degrees_of_freedom,
        "inference_critical_value_95": float(stats.t.ppf(0.975, degrees_of_freedom)),
    }


def bmbc_estimate(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    mirror_repeats: int,
    block_bootstraps: int,
    psi: float,
    seed: int,
    compute_diagnostics: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(harmonized) < 1:
        return (
            _empty_bmbc("not_enough_instruments"),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    reps, diag = make_mirror_array_replicates(harmonized, mirror_pool, mirror_repeats, seed)
    rec, boot = bmbc_estimate_from_replicates(
        harmonized=harmonized,
        replicates=reps,
        mirror_diagnostics=diag,
        mirror_repeats=mirror_repeats,
        block_bootstraps=block_bootstraps,
        psi=psi,
        seed=seed,
        method="BMBC_MR",
    )
    if compute_diagnostics:
        influence = ld_block_influence(harmonized, mirror_pool, mirror_repeats=max(5, min(20, mirror_repeats)), psi=psi, seed=seed + 29)
        psi_curve = sample_overlap_curve(harmonized, mirror_pool, mirror_repeats=max(5, min(20, mirror_repeats)), psi_values=[-0.2, -0.1, 0.0, 0.1, 0.2], seed=seed + 41)
    else:
        influence = pd.DataFrame()
        psi_curve = pd.DataFrame()
    return rec, boot, influence, psi_curve


def bmbc_estimate_discovery_conditional(
    harmonized: pd.DataFrame,
    mirror_repeats: int,
    block_bootstraps: int,
    psi: float,
    p_threshold: float,
    winner_correlation: float,
    clump_r2: float,
    seed: int,
    ld_corr: np.ndarray | None = None,
    sampler_sweeps: int = 16,
    effect_prior_variance_z: float | None = None,
    effect_prior_source: str | None = None,
    ld_correlation_source: str | None = None,
    reference_ld_status: str | None = None,
    clump_r2_matrix: np.ndarray | None = None,
    clump_ld_metric: str | None = None,
    candidate_frame: pd.DataFrame | None = None,
    candidate_ld_corr: np.ndarray | None = None,
    effect_prior_family: str | None = None,
    effect_prior_location_z: float | None = None,
    effect_prior_component_variance_z: float | None = None,
    effect_prior_covariance_model: str = "independent_effects",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit BMBC-MR with the preferred full-candidate or degraded conditional mirror."""
    if len(harmonized) < 1:
        empty = _empty_bmbc("not_enough_instruments")
        empty["method"] = "BMBC_MR_discovery_conditional"
        return empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    reps, diag = discovery_conditional_mirror_replicates(
        harmonized=harmonized,
        repeats=mirror_repeats,
        p_threshold=p_threshold,
        winner_correlation=winner_correlation,
        psi=psi,
        seed=seed,
        clump_r2=clump_r2,
        ld_corr=ld_corr,
        sampler_sweeps=sampler_sweeps,
        effect_prior_variance_z=effect_prior_variance_z,
        effect_prior_source=effect_prior_source,
        ld_correlation_source=ld_correlation_source,
        reference_ld_status=reference_ld_status,
        clump_r2_matrix=clump_r2_matrix,
        clump_ld_metric=clump_ld_metric,
        candidate_frame=candidate_frame,
        candidate_ld_corr=candidate_ld_corr,
        effect_prior_family=effect_prior_family,
        effect_prior_location_z=effect_prior_location_z,
        effect_prior_component_variance_z=effect_prior_component_variance_z,
        effect_prior_covariance_model=effect_prior_covariance_model,
    )
    rec, boot = bmbc_estimate_from_replicates(
        harmonized=harmonized,
        replicates=reps,
        mirror_diagnostics=diag,
        mirror_repeats=mirror_repeats,
        block_bootstraps=block_bootstraps,
        psi=psi,
        seed=seed,
        method="BMBC_MR_discovery_conditional",
    )
    return rec, boot, pd.DataFrame(), pd.DataFrame()


def bmbc_estimate_from_replicates(
    harmonized: pd.DataFrame,
    replicates: dict[str, np.ndarray] | None,
    mirror_diagnostics: dict[str, Any],
    mirror_repeats: int,
    block_bootstraps: int,
    psi: float,
    seed: int,
    method: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Estimate BMBC-MR from a frozen mirror draw and its QC metadata."""
    ivw = ivw_corrected(harmonized, psi)

    def fun(beta: float) -> float:
        return score(harmonized, beta, psi) - mirror_bias_arrays(replicates, beta, psi)

    beta0 = ivw.get("beta", 0.0)
    if not np.isfinite(beta0):
        beta0 = 0.0
    beta, root_diagnostics = _solve_root_with_diagnostics(
        fun,
        float(beta0),
        grid_fun=lambda grid: _bmbc_score_grid(harmonized, replicates, grid, psi),
    )
    block_metrics = _effective_ld_block_metrics(harmonized, beta, psi)
    observed_blocks = int(block_metrics["observed_ld_blocks"])
    effective_blocks = float(block_metrics["effective_ld_blocks"])
    bootstrap_mirror_repeats = int(mirror_repeats)
    requested_bootstraps = int(block_bootstraps)
    mirror_status = str(mirror_diagnostics.get("mirror_status", "unknown"))
    mirror_valid = mirror_status in {"ok", "ok_degraded_matching"}
    if mirror_valid and observed_blocks >= 2 and effective_blocks >= 2.0 and requested_bootstraps >= 2:
        boot = _block_bootstrap(
            harmonized,
            replicates,
            n_boot=block_bootstraps,
            psi=psi,
            seed=seed + 17,
            beta0=beta,
        )
        valid_boot = boot[
            np.isfinite(pd.to_numeric(boot.get("beta"), errors="coerce"))
            & boot.get("root_status", pd.Series(index=boot.index, dtype=object)).astype(str).eq("unique_bracketed_root")
        ].copy()
        required_valid_bootstraps = max(2, math.ceil(0.80 * requested_bootstraps))
        if len(valid_boot) >= required_valid_bootstraps:
            bootstrap_status = "ok"
        else:
            bootstrap_status = "insufficient_unique_root_bootstraps"
    else:
        boot = pd.DataFrame()
        valid_boot = pd.DataFrame()
        required_valid_bootstraps = max(2, math.ceil(0.80 * max(requested_bootstraps, 0)))
        if not mirror_valid:
            bootstrap_status = f"not_run_mirror_qc_failed:{mirror_status}"
        elif observed_blocks < 2:
            bootstrap_status = "insufficient_ld_blocks"
        elif effective_blocks < 2.0:
            bootstrap_status = "insufficient_effective_ld_blocks"
        else:
            bootstrap_status = "insufficient_bootstrap_replicates"
    se = float(np.nanstd(valid_boot["beta"], ddof=1)) if bootstrap_status == "ok" and len(valid_boot) > 1 else np.nan
    root_valid = root_diagnostics.get("root_status") == "unique_bracketed_root"
    if mirror_status == "empty_pool":
        se = np.nan
        inference_status = "uncalibrated_empty_pool_no_bmbc_inference"
    elif not mirror_valid:
        se = np.nan
        inference_status = f"mirror_qc_failed:{mirror_status}"
    elif not root_valid:
        se = np.nan
        inference_status = f"invalid_root:{root_diagnostics.get('root_status', 'unknown')}"
    elif bootstrap_status == "ok" and np.isfinite(se) and se > 0:
        inference_status = "block_bootstrap"
    else:
        se = np.nan
        if bootstrap_status == "ok":
            bootstrap_status = "degenerate_bootstrap"
        inference_status = bootstrap_status
    finite_block_inference = block_bootstrap_inference(beta, se, effective_blocks)
    if inference_status != "block_bootstrap":
        finite_block_inference = {
            "p": np.nan,
            "inference_distribution": "",
            "inference_degrees_of_freedom": np.nan,
            "inference_critical_value_95": np.nan,
        }
    bias_at_ivw = mirror_bias_arrays(replicates, float(ivw.get("beta", np.nan)), psi) if np.isfinite(ivw.get("beta", np.nan)) else np.nan
    bootstrap_mean_unique_blocks = (
        float(pd.to_numeric(boot["bootstrap_unique_ld_blocks"], errors="coerce").mean())
        if not boot.empty and "bootstrap_unique_ld_blocks" in boot
        else np.nan
    )
    bootstrap_mean_effective_blocks = (
        float(pd.to_numeric(boot["bootstrap_effective_ld_blocks"], errors="coerce").mean())
        if not boot.empty and "bootstrap_effective_ld_blocks" in boot
        else np.nan
    )
    bootstrap_min_effective_blocks = (
        float(pd.to_numeric(boot["bootstrap_effective_ld_blocks"], errors="coerce").min())
        if not boot.empty and "bootstrap_effective_ld_blocks" in boot
        else np.nan
    )
    rec = {
        "method": method,
        "beta": beta,
        "se": se,
        "p": finite_block_inference["p"],
        "n_instruments": len(harmonized),
        "status": "uncalibrated_empty_pool" if mirror_status == "empty_pool" else mirror_status,
        "bmbc_layer": "selection_calibrated",
        "ivw_beta": ivw.get("beta", np.nan),
        "ivw_se": ivw.get("se", np.nan),
        "ivw_p": ivw.get("p", np.nan),
        "ivw_status": ivw.get("status", "unknown"),
        "ivw_inference_status": ivw.get("inference_status", "unknown"),
        "ivw_root_status": ivw.get("root_status", "unknown"),
        "ivw_root_candidate_count": ivw.get("root_candidate_count", np.nan),
        "ivw_root_score_at_solution": ivw.get("root_score_at_solution", np.nan),
        "ivw_root_derivative": ivw.get("root_derivative", np.nan),
        "selection_bias_score_at_ivw": bias_at_ivw,
        "beta_shift_vs_ivw": beta - float(ivw.get("beta", np.nan)) if np.isfinite(ivw.get("beta", np.nan)) else np.nan,
        "weak_instrument_inflation": weak_instrument_inflation(harmonized),
        "mirror_repeats": mirror_repeats,
        "block_bootstraps": block_bootstraps,
        "bootstrap_mirror_repeats": bootstrap_mirror_repeats,
        "bootstrap_score_backend": bootstrap_score_backend(),
        "bootstrap_root_grid_points": BOOTSTRAP_ROOT_GRID_POINTS,
        "bootstrap_root_refinement_iterations": BOOTSTRAP_ROOT_REFINEMENT_ITERATIONS,
        "bootstrap_root_refinement_backend": bootstrap_root_refinement_backend(replicates),
        "observed_ld_blocks": observed_blocks,
        "effective_ld_blocks": effective_blocks,
        "effective_ld_blocks_definition": "kish_effective_count_of_observed_block_information",
        "max_ld_block_information_share": block_metrics["max_ld_block_information_share"],
        "bootstrap_status": bootstrap_status,
        "bootstrap_valid_replicates": len(valid_boot),
        "bootstrap_required_valid_replicates": int(required_valid_bootstraps),
        "bootstrap_failure_rate": 1.0 - float(len(valid_boot)) / max(float(requested_bootstraps), 1.0),
        "bootstrap_mean_unique_ld_blocks": bootstrap_mean_unique_blocks,
        "bootstrap_mean_effective_ld_blocks": bootstrap_mean_effective_blocks,
        "bootstrap_min_effective_ld_blocks": bootstrap_min_effective_blocks,
        "inference_status": inference_status,
        "inference_distribution": finite_block_inference["inference_distribution"],
        "inference_degrees_of_freedom": finite_block_inference["inference_degrees_of_freedom"],
        "inference_critical_value_95": finite_block_inference["inference_critical_value_95"],
        **root_diagnostics,
        **mirror_diagnostics,
    }
    return rec, boot


def _bmbc_score_grid(
    harmonized: pd.DataFrame,
    replicates: dict[str, np.ndarray] | None,
    beta_grid: np.ndarray,
    psi: float,
    chunk_size: int = 32,
) -> np.ndarray:
    """Evaluate the calibrated score on a beta grid without Python scalar loops."""
    grid = np.asarray(beta_grid, dtype=float)
    gx = harmonized["beta_x"].to_numpy(float)
    gy = harmonized["beta_y"].to_numpy(float)
    sx = harmonized["se_x"].to_numpy(float)
    sy = harmonized["se_y"].to_numpy(float)
    values = np.empty(len(grid), dtype=float)
    if replicates is not None and bootstrap_score_backend() == "numba_fused_single_thread":
        mbx = np.asarray(replicates["beta_x"], dtype=float)
        mbx_residual = np.asarray(replicates.get("beta_x_residual", mbx), dtype=float)
        mby = np.asarray(replicates["beta_y"], dtype=float)
        msx = np.asarray(replicates["se_x"], dtype=float)
        msy = np.asarray(replicates["se_y"], dtype=float)
        return _bmbc_score_grid_numba(
            np.ascontiguousarray(gx),
            np.ascontiguousarray(gy),
            np.ascontiguousarray(sx),
            np.ascontiguousarray(sy),
            np.ascontiguousarray(mbx.T),
            np.ascontiguousarray(mbx_residual.T),
            np.ascontiguousarray(mby.T),
            np.ascontiguousarray(msx.T),
            np.ascontiguousarray(msy.T),
            np.ascontiguousarray(grid),
            float(psi),
            _mirror_standard_errors_are_replicate_constant(msx, msy),
        )
    for start in range(0, len(grid), int(chunk_size)):
        stop = min(start + int(chunk_size), len(grid))
        beta = grid[start:stop, None]
        den = np.maximum(sy[None, :] ** 2 + beta**2 * sx[None, :] ** 2 - 2.0 * beta * psi * sx[None, :] * sy[None, :], 1e-12)
        real_score = np.sum(gx[None, :] * (gy[None, :] - beta * gx[None, :]) / den, axis=1)
        if replicates is None:
            values[start:stop] = real_score
            continue
        mirror_beta = grid[start:stop, None, None]
        mbx = replicates["beta_x"]
        mbx_residual = replicates.get("beta_x_residual", mbx)
        mby = replicates["beta_y"]
        msx = replicates["se_x"]
        msy = replicates["se_y"]
        mirror_den = np.maximum(
            msy[None, :, :] ** 2
            + mirror_beta**2 * msx[None, :, :] ** 2
            - 2.0 * mirror_beta * psi * msx[None, :, :] * msy[None, :, :],
            1e-12,
        )
        mirror_score = np.mean(
            np.sum(mbx[None, :, :] * (mby[None, :, :] - mirror_beta * mbx_residual[None, :, :]) / mirror_den, axis=2),
            axis=1,
        )
        values[start:stop] = real_score - mirror_score
    return values


def _effective_ld_block_metrics(harmonized: pd.DataFrame, beta: float, psi: float) -> dict[str, float | int]:
    if harmonized.empty or "block" not in harmonized:
        return {
            "observed_ld_blocks": 0,
            "effective_ld_blocks": 0.0,
            "max_ld_block_information_share": np.nan,
        }
    blocks = harmonized["block"].astype(str)
    gx = harmonized["beta_x"].to_numpy(float)
    den = corrected_denominator(
        beta,
        harmonized["se_x"].to_numpy(float),
        harmonized["se_y"].to_numpy(float),
        psi,
    )
    information = np.maximum(gx**2 / den, 0.0)
    block_information = pd.Series(information).groupby(blocks.reset_index(drop=True), sort=False).sum().to_numpy(float)
    total = float(np.sum(block_information))
    if total <= 0 or not np.isfinite(total):
        effective = 0.0
        maximum_share = np.nan
    else:
        denominator = float(np.sum(block_information**2))
        effective = float(total**2 / denominator) if denominator > 0 else 0.0
        maximum_share = float(np.max(block_information) / total)
    return {
        "observed_ld_blocks": len(block_information),
        "effective_ld_blocks": effective,
        "max_ld_block_information_share": maximum_share,
    }


def sample_overlap_curve(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    mirror_repeats: int,
    psi_values: list[float],
    seed: int,
) -> pd.DataFrame:
    rows = []
    reps, mirror_diagnostics = make_mirror_array_replicates(harmonized, mirror_pool, mirror_repeats, seed)
    for psi in psi_values:
        ivw = ivw_corrected(harmonized, psi)
        def fun(beta: float, overlap: float = psi) -> float:
            return score(harmonized, beta, overlap) - mirror_bias_arrays(reps, beta, overlap)
        beta0 = ivw.get("beta", 0.0)
        if not np.isfinite(beta0):
            beta0 = 0.0
        beta, root_diagnostics = _solve_root_with_diagnostics(fun, float(beta0))
        rows.append(
            {
                "psi": psi,
                "beta": beta,
                "ivw_beta": ivw.get("beta", np.nan),
                "mirror_status": mirror_diagnostics.get("mirror_status", "unknown"),
                **root_diagnostics,
            }
        )
    return pd.DataFrame(rows)


def ld_block_influence(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    mirror_repeats: int,
    psi: float,
    seed: int,
) -> pd.DataFrame:
    if "block" not in harmonized or harmonized.empty:
        return pd.DataFrame()
    full_reps, full_mirror_diagnostics = make_mirror_array_replicates(harmonized, mirror_pool, mirror_repeats, seed)
    full_beta, full_root_diagnostics = _solve_root_with_diagnostics(
        lambda b: score(harmonized, b, psi) - mirror_bias_arrays(full_reps, b, psi),
        ivw_corrected(harmonized, psi).get("beta", 0.0),
    )
    rows = []
    for block in sorted(harmonized["block"].dropna().astype(str).unique()):
        sub = harmonized[harmonized["block"].astype(str) != block]
        if sub.empty:
            continue
        reps, mirror_diagnostics = make_mirror_array_replicates(sub, mirror_pool, mirror_repeats, stable_seed(seed, "leave_block", block))
        beta, root_diagnostics = _solve_root_with_diagnostics(
            lambda b, data=sub, draws=reps: score(data, b, psi) - mirror_bias_arrays(draws, b, psi),
            ivw_corrected(sub, psi).get("beta", full_beta),
        )
        rows.append(
            {
                "block": block,
                "beta_leave_one_block": beta,
                "delta_vs_full": beta - full_beta,
                "n_removed": int((harmonized["block"].astype(str) == block).sum()),
                "full_root_status": full_root_diagnostics.get("root_status", "unknown"),
                "full_mirror_status": full_mirror_diagnostics.get("mirror_status", "unknown"),
                "leave_one_root_status": root_diagnostics.get("root_status", "unknown"),
                "leave_one_mirror_status": mirror_diagnostics.get("mirror_status", "unknown"),
                "leave_one_root_candidates": root_diagnostics.get("root_candidate_count", np.nan),
            }
        )
    return pd.DataFrame(rows)


def weak_instrument_inflation(df: pd.DataFrame) -> float:
    f = (df["beta_x"].to_numpy(float) / np.maximum(df["se_x"].to_numpy(float), 1e-12)) ** 2
    if len(f) == 0:
        return np.nan
    mean_f = float(np.nanmean(f))
    return float(1.0 / max(mean_f, 1e-12))


def _block_bootstrap(
    harmonized: pd.DataFrame,
    replicates: dict[str, np.ndarray] | None,
    n_boot: int,
    psi: float,
    seed: int,
    beta0: float,
) -> pd.DataFrame:
    """Resample LD blocks while reusing the fitted mirror empirical law.

    A mirror row is a complete block-matched pseudo-dataset. Resampling those
    rows propagates mirror Monte Carlo uncertainty without rebuilding the
    matching model inside every bootstrap replicate. Block score curves are
    precomputed on the root grid, and bracketed roots are refined in batches
    with analytic score derivatives.
    """
    rng = np.random.default_rng(seed)
    if harmonized.empty or "block" not in harmonized or replicates is None:
        return pd.DataFrame()
    block_labels = harmonized["block"].astype(str).to_numpy()
    blocks, block_codes = np.unique(block_labels, return_inverse=True)
    n_blocks = len(blocks)
    n_mirror = int(replicates["beta_x"].shape[0])
    if n_blocks == 0 or n_mirror == 0:
        return pd.DataFrame()
    grid = np.linspace(
        float(beta0) - 5.0,
        float(beta0) + 5.0,
        BOOTSTRAP_ROOT_GRID_POINTS,
    )
    real_curves, mirror_curves = _block_score_curves(
        harmonized,
        replicates,
        grid,
        psi,
        block_codes,
        n_blocks,
    )
    information = _block_information(harmonized, beta0, psi, block_codes, n_blocks)
    block_sizes = np.bincount(block_codes, minlength=n_blocks)

    n_boot = int(n_boot)
    block_counts_matrix = np.empty((n_boot, n_blocks), dtype=float)
    mirror_weights_matrix = np.empty((n_boot, n_mirror), dtype=float)
    for i in range(n_boot):
        block_counts_matrix[i] = np.bincount(
            rng.integers(0, n_blocks, size=n_blocks),
            minlength=n_blocks,
        )
        mirror_counts = np.bincount(
            rng.integers(0, n_mirror, size=n_mirror),
            minlength=n_mirror,
        ).astype(float)
        mirror_weights_matrix[i] = mirror_counts / float(n_mirror)

    score_grids = _bootstrap_score_grids(
        real_curves,
        mirror_curves,
        block_counts_matrix,
        mirror_weights_matrix,
    )
    roots = _solve_bootstrap_roots(
        harmonized,
        replicates,
        beta0=float(beta0),
        grid=grid,
        score_grids=score_grids,
        psi=psi,
        block_codes=block_codes,
        block_counts=block_counts_matrix,
        mirror_weights=mirror_weights_matrix,
    )

    weighted_information = block_counts_matrix * information[None, :]
    information_total = weighted_information.sum(axis=1)
    information_square_sum = np.square(weighted_information).sum(axis=1)
    effective_blocks = np.divide(
        np.square(information_total),
        information_square_sum,
        out=np.zeros(n_boot, dtype=float),
        where=(information_total > 0) & (information_square_sum > 0),
    )
    maximum_share = np.divide(
        weighted_information.max(axis=1),
        information_total,
        out=np.full(n_boot, np.nan, dtype=float),
        where=information_total > 0,
    )
    mirror_weight_squares = np.square(mirror_weights_matrix).sum(axis=1)
    effective_mirror_repeats = np.divide(
        1.0,
        mirror_weight_squares,
        out=np.zeros(n_boot, dtype=float),
        where=mirror_weight_squares > 0,
    )

    roots.insert(0, "bootstrap", np.arange(1, n_boot + 1, dtype=int))
    roots.insert(2, "n", np.rint(block_counts_matrix @ block_sizes).astype(int))
    roots.insert(3, "bootstrap_unique_ld_blocks", np.sum(block_counts_matrix > 0, axis=1).astype(int))
    roots.insert(4, "bootstrap_effective_ld_blocks", effective_blocks)
    roots.insert(5, "bootstrap_max_block_information_share", maximum_share)
    roots.insert(6, "bootstrap_effective_mirror_repeats", effective_mirror_repeats)
    return roots


def _bootstrap_score_grids(
    real_curves: np.ndarray,
    mirror_curves: np.ndarray,
    block_counts: np.ndarray,
    mirror_weights: np.ndarray,
    bootstrap_chunk_size: int = 128,
) -> np.ndarray:
    """Evaluate all bootstrap score grids with block-wise matrix products."""
    n_boot = int(block_counts.shape[0])
    n_grid = int(real_curves.shape[1])
    n_blocks = int(real_curves.shape[0])
    score_grids = np.empty((n_boot, n_grid), dtype=float)
    for start in range(0, n_boot, int(bootstrap_chunk_size)):
        stop = min(start + int(bootstrap_chunk_size), n_boot)
        weights = mirror_weights[start:stop]
        counts = block_counts[start:stop]
        score_chunk = np.zeros((stop - start, n_grid), dtype=float)
        for block in range(n_blocks):
            mirror_mean = weights @ mirror_curves[block]
            score_chunk += counts[:, block, None] * (real_curves[block][None, :] - mirror_mean)
        score_grids[start:stop] = score_chunk
    return score_grids


def _solve_bootstrap_roots(
    harmonized: pd.DataFrame,
    replicates: dict[str, np.ndarray],
    beta0: float,
    grid: np.ndarray,
    score_grids: np.ndarray,
    psi: float,
    block_codes: np.ndarray,
    block_counts: np.ndarray,
    mirror_weights: np.ndarray,
    max_iterations: int = BOOTSTRAP_ROOT_REFINEMENT_ITERATIONS,
) -> pd.DataFrame:
    """Find all grid roots and refine the nearest bracket in vectorized batches."""
    n_boot = int(score_grids.shape[0])
    roots = np.full(n_boot, float(beta0), dtype=float)
    candidate_counts = np.zeros(n_boot, dtype=int)
    statuses = np.full(n_boot, "insufficient_finite_score_grid", dtype=object)
    left = np.full(n_boot, np.nan, dtype=float)
    right = np.full(n_boot, np.nan, dtype=float)
    left_score = np.full(n_boot, np.nan, dtype=float)
    right_score = np.full(n_boot, np.nan, dtype=float)
    previous_root = np.full(n_boot, np.nan, dtype=float)
    previous_score = np.full(n_boot, np.nan, dtype=float)
    bracketed = np.zeros(n_boot, dtype=bool)
    preaggregated_mirror = _prepare_constant_mirror_se_preaggregation(
        replicates,
        mirror_weights,
    )

    def evaluate(
        row_indices: np.ndarray,
        beta_values: np.ndarray,
        *,
        compute_derivative: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if preaggregated_mirror is not None:
            return _weighted_block_score_derivative_preaggregated(
                harmonized,
                beta_values,
                psi,
                block_codes,
                block_counts[row_indices],
                preaggregated_mirror["intercept"][row_indices],
                preaggregated_mirror["slope"][row_indices],
                preaggregated_mirror["se_x"],
                preaggregated_mirror["se_y"],
                compute_derivative=compute_derivative,
            )
        return _weighted_block_score_derivative_batch(
            harmonized,
            replicates,
            beta_values,
            psi,
            block_codes,
            block_counts[row_indices],
            mirror_weights[row_indices],
            compute_derivative=compute_derivative,
        )

    for row_index in range(n_boot):
        values = np.asarray(score_grids[row_index], dtype=float)
        finite_indices = np.flatnonzero(np.isfinite(values))
        if len(finite_indices) < 2:
            continue
        finite_grid = grid[finite_indices]
        finite_values = values[finite_indices]
        signs = np.sign(finite_values)
        exact_local = np.flatnonzero(signs == 0)
        crossings_local = np.flatnonzero(signs[:-1] * signs[1:] < 0)
        candidate_counts[row_index] = int(len(exact_local) + len(crossings_local))

        candidates: list[tuple[float, str, int]] = []
        candidates.extend((float(finite_grid[index]), "exact", int(index)) for index in exact_local)
        for index in crossings_local:
            x_left = float(finite_grid[index])
            x_right = float(finite_grid[index + 1])
            y_left = float(finite_values[index])
            y_right = float(finite_values[index + 1])
            interpolated = x_left - y_left * (x_right - x_left) / (y_right - y_left)
            candidates.append((float(interpolated), "bracket", int(index)))

        if not candidates:
            nearest = int(np.argmin(np.abs(finite_values)))
            roots[row_index] = float(finite_grid[nearest])
            statuses[row_index] = "no_bracket_minimum_absolute_score"
            continue

        selected_value, selected_kind, selected_index = min(candidates, key=lambda item: abs(item[0] - beta0))
        roots[row_index] = selected_value
        statuses[row_index] = (
            "unique_bracketed_root"
            if candidate_counts[row_index] == 1
            else "multiple_roots_nearest_initial"
        )
        if selected_kind == "bracket":
            bracketed[row_index] = True
            left[row_index] = float(finite_grid[selected_index])
            right[row_index] = float(finite_grid[selected_index + 1])
            left_score[row_index] = float(finite_values[selected_index])
            right_score[row_index] = float(finite_values[selected_index + 1])

    active = bracketed.copy()
    for iteration in range(int(max_iterations)):
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            break
        values, derivatives = evaluate(
            indices,
            roots[indices],
            compute_derivative=iteration == 0,
        )
        converged = np.isfinite(values) & (np.abs(values) <= 1e-7)
        active[indices[converged]] = False
        remaining = ~converged
        if not np.any(remaining):
            continue
        unresolved = indices[remaining]
        unresolved_values = values[remaining]
        same_as_left = np.sign(left_score[unresolved]) * np.sign(unresolved_values) <= 0
        right[unresolved[same_as_left]] = roots[unresolved[same_as_left]]
        right_score[unresolved[same_as_left]] = unresolved_values[same_as_left]
        left[unresolved[~same_as_left]] = roots[unresolved[~same_as_left]]
        left_score[unresolved[~same_as_left]] = unresolved_values[~same_as_left]

        if iteration == 0:
            assert derivatives is not None
            unresolved_derivatives = derivatives[remaining]
            proposal = roots[unresolved] - unresolved_values / unresolved_derivatives
            invalid = (
                ~np.isfinite(proposal)
                | ~np.isfinite(unresolved_derivatives)
                | (np.abs(unresolved_derivatives) < 1e-12)
                | (proposal <= left[unresolved])
                | (proposal >= right[unresolved])
            )
        else:
            secant_score_difference = unresolved_values - previous_score[unresolved]
            proposal = roots[unresolved] - unresolved_values * (
                roots[unresolved] - previous_root[unresolved]
            ) / secant_score_difference
            invalid = (
                ~np.isfinite(proposal)
                | ~np.isfinite(secant_score_difference)
                | (np.abs(secant_score_difference) < 1e-14)
                | (proposal <= left[unresolved])
                | (proposal >= right[unresolved])
                | (np.abs(proposal - roots[unresolved]) <= 1e-14 * (1.0 + np.abs(roots[unresolved])))
            )
            if np.any(invalid):
                fallback_indices = unresolved[invalid]
                bracket_score_difference = (
                    right_score[fallback_indices] - left_score[fallback_indices]
                )
                fallback = left[fallback_indices] - left_score[fallback_indices] * (
                    right[fallback_indices] - left[fallback_indices]
                ) / bracket_score_difference
                fallback_invalid = (
                    ~np.isfinite(fallback)
                    | ~np.isfinite(bracket_score_difference)
                    | (np.abs(bracket_score_difference) < 1e-14)
                    | (fallback <= left[fallback_indices])
                    | (fallback >= right[fallback_indices])
                )
                fallback[fallback_invalid] = 0.5 * (
                    left[fallback_indices[fallback_invalid]]
                    + right[fallback_indices[fallback_invalid]]
                )
                proposal[invalid] = fallback
                invalid[invalid] = False
        previous_root[unresolved] = roots[unresolved]
        previous_score[unresolved] = unresolved_values
        proposal[invalid] = 0.5 * (left[unresolved[invalid]] + right[unresolved[invalid]])
        roots[unresolved] = proposal

    all_indices = np.arange(n_boot, dtype=int)
    scores, derivatives = evaluate(
        all_indices,
        roots,
    )
    assert derivatives is not None
    estimated_beta_error = np.divide(
        np.abs(scores),
        np.abs(derivatives),
        out=np.full(n_boot, np.inf, dtype=float),
        where=np.isfinite(derivatives) & (np.abs(derivatives) > 0.0),
    )
    rescue = (
        bracketed
        & np.isin(
            statuses,
            ["unique_bracketed_root", "multiple_roots_nearest_initial"],
        )
        & (
            ~np.isfinite(scores)
            | (np.abs(scores) > 1e-7)
            | (estimated_beta_error > 5e-10)
        )
    )
    rescued_indices: list[int] = []
    for index in np.flatnonzero(rescue):
        try:
            roots[index] = float(
                optimize.brentq(
                    lambda value, row=index: float(
                        evaluate(
                            np.asarray([row], dtype=int),
                            np.asarray([value]),
                            compute_derivative=False,
                        )[0][0]
                    ),
                    float(left[index]),
                    float(right[index]),
                    maxiter=100,
                )
            )
            rescued_indices.append(int(index))
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    if rescued_indices:
        rescued = np.asarray(rescued_indices, dtype=int)
        rescued_scores, rescued_derivatives = evaluate(
            rescued,
            roots[rescued],
        )
        scores[rescued] = rescued_scores
        assert rescued_derivatives is not None
        derivatives[rescued] = rescued_derivatives
    flat = (statuses == "unique_bracketed_root") & (
        ~np.isfinite(derivatives) | (np.abs(derivatives) < 1e-8)
    )
    statuses[flat] = "flat_root"
    unresolved = (statuses == "unique_bracketed_root") & (
        ~np.isfinite(scores) | (np.abs(scores) > 1e-7)
    )
    statuses[unresolved] = "root_refinement_failed"
    return pd.DataFrame(
        {
            "beta": roots,
            "root_status": statuses,
            "root_candidate_count": candidate_counts,
            "root_score_at_solution": scores,
            "root_derivative": derivatives,
        }
    )


def _prepare_constant_mirror_se_preaggregation(
    replicates: dict[str, np.ndarray],
    mirror_weights: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Aggregate mirror numerators once when repeat-specific SEs are constant."""
    mbx = np.asarray(replicates["beta_x"], dtype=float)
    mbx_residual = np.asarray(replicates.get("beta_x_residual", mbx), dtype=float)
    mby = np.asarray(replicates["beta_y"], dtype=float)
    msx = np.asarray(replicates["se_x"], dtype=float)
    msy = np.asarray(replicates["se_y"], dtype=float)
    if (
        mbx.ndim != 2
        or mbx.shape != mbx_residual.shape
        or mbx.shape != mby.shape
        or mbx.shape != msx.shape
        or mbx.shape != msy.shape
        or mirror_weights.ndim != 2
        or mirror_weights.shape[1] != mbx.shape[0]
        or not _mirror_standard_errors_are_replicate_constant(msx, msy)
    ):
        return None
    weights = np.asarray(mirror_weights, dtype=float)
    return {
        "intercept": weights @ (mbx * mby),
        "slope": weights @ (mbx * mbx_residual),
        "se_x": msx[0].copy(),
        "se_y": msy[0].copy(),
    }


def _weighted_block_score_derivative_preaggregated(
    harmonized: pd.DataFrame,
    beta: np.ndarray,
    psi: float,
    block_codes: np.ndarray,
    block_counts: np.ndarray,
    mirror_intercept: np.ndarray,
    mirror_slope: np.ndarray,
    mirror_se_x: np.ndarray,
    mirror_se_y: np.ndarray,
    *,
    compute_derivative: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate bootstrap roots after exact mirror-numerator preaggregation."""
    beta = np.asarray(beta, dtype=float)
    gx = harmonized["beta_x"].to_numpy(float)
    gy = harmonized["beta_y"].to_numpy(float)
    sx = harmonized["se_x"].to_numpy(float)
    sy = harmonized["se_y"].to_numpy(float)
    instrument_weights = np.asarray(block_counts, dtype=float)[:, block_codes]
    values = beta[:, None]

    real_denominator = np.maximum(
        sy[None, :] ** 2
        + values**2 * sx[None, :] ** 2
        - 2.0 * values * psi * sx[None, :] * sy[None, :],
        1e-12,
    )
    real_numerator = gx[None, :] * (gy[None, :] - values * gx[None, :])
    real_component = real_numerator / real_denominator

    mirror_denominator = np.maximum(
        mirror_se_y[None, :] ** 2
        + values**2 * mirror_se_x[None, :] ** 2
        - 2.0 * values * psi * mirror_se_x[None, :] * mirror_se_y[None, :],
        1e-12,
    )
    mirror_numerator = mirror_intercept - values * mirror_slope
    mirror_component = mirror_numerator / mirror_denominator
    scores = np.sum(instrument_weights * (real_component - mirror_component), axis=1)
    if not compute_derivative:
        return scores, None

    real_denominator_derivative = (
        2.0 * values * sx[None, :] ** 2
        - 2.0 * psi * sx[None, :] * sy[None, :]
    )
    real_derivative = (
        -(gx[None, :] ** 2) / real_denominator
        - real_numerator * real_denominator_derivative / np.square(real_denominator)
    )
    mirror_denominator_derivative = (
        2.0 * values * mirror_se_x[None, :] ** 2
        - 2.0 * psi * mirror_se_x[None, :] * mirror_se_y[None, :]
    )
    mirror_derivative = (
        -mirror_slope / mirror_denominator
        - mirror_numerator * mirror_denominator_derivative / np.square(mirror_denominator)
    )
    derivatives = np.sum(
        instrument_weights * (real_derivative - mirror_derivative),
        axis=1,
    )
    return scores, derivatives


def _weighted_block_score_derivative_batch(
    harmonized: pd.DataFrame,
    replicates: dict[str, np.ndarray],
    beta: np.ndarray,
    psi: float,
    block_codes: np.ndarray,
    block_counts: np.ndarray,
    mirror_weights: np.ndarray,
    *,
    compute_derivative: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate bootstrap scores and optionally analytic derivatives."""
    beta = np.asarray(beta, dtype=float)
    gx = harmonized["beta_x"].to_numpy(float)
    gy = harmonized["beta_y"].to_numpy(float)
    sx = harmonized["se_x"].to_numpy(float)
    sy = harmonized["se_y"].to_numpy(float)
    mbx = np.asarray(replicates["beta_x"], dtype=float)
    mbx_residual = np.asarray(replicates.get("beta_x_residual", mbx), dtype=float)
    mby = np.asarray(replicates["beta_y"], dtype=float)
    msx = np.asarray(replicates["se_x"], dtype=float)
    msy = np.asarray(replicates["se_y"], dtype=float)
    if bootstrap_score_backend() == "numba_fused_single_thread":
        mirror_beta_x = np.ascontiguousarray(mbx.T)
        mirror_beta_x_residual = np.ascontiguousarray(mbx_residual.T)
        mirror_beta_y = np.ascontiguousarray(mby.T)
        mirror_se_x = np.ascontiguousarray(msx.T)
        mirror_se_y = np.ascontiguousarray(msy.T)
        numba_scores, numba_derivatives = _weighted_block_score_derivative_numba(
            np.ascontiguousarray(gx),
            np.ascontiguousarray(gy),
            np.ascontiguousarray(sx),
            np.ascontiguousarray(sy),
            np.ascontiguousarray(mirror_beta_x * mirror_beta_y),
            np.ascontiguousarray(mirror_beta_x * mirror_beta_x_residual),
            np.ascontiguousarray(mirror_se_x * mirror_se_x),
            np.ascontiguousarray(mirror_se_y * mirror_se_y),
            np.ascontiguousarray(mirror_se_x * mirror_se_y),
            np.ascontiguousarray(beta),
            float(psi),
            np.ascontiguousarray(block_codes, dtype=np.int64),
            np.ascontiguousarray(block_counts, dtype=float),
            np.ascontiguousarray(mirror_weights, dtype=float),
            bool(compute_derivative),
        )
        return numba_scores, numba_derivatives if compute_derivative else None
    scores = np.empty(len(beta), dtype=float)
    derivatives = np.empty(len(beta), dtype=float) if compute_derivative else None
    elements_per_bootstrap = max(int(mbx.size), 1)
    chunk_size = max(1, min(128, 2_000_000 // elements_per_bootstrap))

    for start in range(0, len(beta), chunk_size):
        stop = min(start + chunk_size, len(beta))
        values = beta[start:stop]
        instrument_weights = block_counts[start:stop, block_codes]

        real_beta = values[:, None]
        real_den = np.maximum(
            sy[None, :] ** 2
            + real_beta**2 * sx[None, :] ** 2
            - 2.0 * real_beta * psi * sx[None, :] * sy[None, :],
            1e-12,
        )
        real_num = gx[None, :] * (gy[None, :] - real_beta * gx[None, :])
        real_component = real_num / real_den
        if compute_derivative:
            real_den_derivative = (
                2.0 * real_beta * sx[None, :] ** 2
                - 2.0 * psi * sx[None, :] * sy[None, :]
            )
            real_derivative = (
                -(gx[None, :] ** 2) / real_den
                - real_num * real_den_derivative / np.square(real_den)
            )

        mirror_beta = values[:, None, None]
        mirror_den = np.maximum(
            msy[None, :, :] ** 2
            + mirror_beta**2 * msx[None, :, :] ** 2
            - 2.0 * mirror_beta * psi * msx[None, :, :] * msy[None, :, :],
            1e-12,
        )
        mirror_num = mbx[None, :, :] * (mby[None, :, :] - mirror_beta * mbx_residual[None, :, :])
        mirror_component = mirror_num / mirror_den
        weights = mirror_weights[start:stop]
        mirror_mean = np.einsum("cr,crn->cn", weights, mirror_component, optimize=True)
        scores[start:stop] = np.sum(instrument_weights * (real_component - mirror_mean), axis=1)
        if compute_derivative:
            mirror_den_derivative = (
                2.0 * mirror_beta * msx[None, :, :] ** 2
                - 2.0 * psi * msx[None, :, :] * msy[None, :, :]
            )
            mirror_derivative = (
                -(mbx[None, :, :] * mbx_residual[None, :, :]) / mirror_den
                - mirror_num * mirror_den_derivative / np.square(mirror_den)
            )
            mirror_mean_derivative = np.einsum(
                "cr,crn->cn",
                weights,
                mirror_derivative,
                optimize=True,
            )
            assert derivatives is not None
            derivatives[start:stop] = np.sum(
                instrument_weights * (real_derivative - mirror_mean_derivative),
                axis=1,
            )
    return scores, derivatives


def _block_information(
    harmonized: pd.DataFrame,
    beta: float,
    psi: float,
    block_codes: np.ndarray,
    n_blocks: int,
) -> np.ndarray:
    gx = harmonized["beta_x"].to_numpy(float)
    den = corrected_denominator(
        beta,
        harmonized["se_x"].to_numpy(float),
        harmonized["se_y"].to_numpy(float),
        psi,
    )
    return np.bincount(block_codes, weights=np.maximum(gx**2 / den, 0.0), minlength=n_blocks)


def _block_score_curves(
    harmonized: pd.DataFrame,
    replicates: dict[str, np.ndarray],
    beta_grid: np.ndarray,
    psi: float,
    block_codes: np.ndarray,
    n_blocks: int,
    chunk_size: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(beta_grid, dtype=float)
    gx = harmonized["beta_x"].to_numpy(float)
    gy = harmonized["beta_y"].to_numpy(float)
    sx = harmonized["se_x"].to_numpy(float)
    sy = harmonized["se_y"].to_numpy(float)
    mbx = replicates["beta_x"]
    mbx_residual = replicates.get("beta_x_residual", mbx)
    mby = replicates["beta_y"]
    msx = replicates["se_x"]
    msy = replicates["se_y"]
    n_mirror = int(mbx.shape[0])
    if bootstrap_score_backend() == "numba_fused_single_thread":
        return _block_score_curves_numba(
            np.ascontiguousarray(gx),
            np.ascontiguousarray(gy),
            np.ascontiguousarray(sx),
            np.ascontiguousarray(sy),
            np.ascontiguousarray(mbx.T),
            np.ascontiguousarray(mbx_residual.T),
            np.ascontiguousarray(mby.T),
            np.ascontiguousarray(msx.T),
            np.ascontiguousarray(msy.T),
            np.ascontiguousarray(grid),
            float(psi),
            np.ascontiguousarray(block_codes, dtype=np.int64),
            int(n_blocks),
            _mirror_standard_errors_are_replicate_constant(msx, msy),
        )
    real_curves = np.zeros((n_blocks, len(grid)), dtype=float)
    mirror_curves = np.zeros((n_blocks, n_mirror, len(grid)), dtype=float)
    block_indices = [np.flatnonzero(block_codes == block) for block in range(n_blocks)]
    for start in range(0, len(grid), int(chunk_size)):
        stop = min(start + int(chunk_size), len(grid))
        beta = grid[start:stop, None]
        den = np.maximum(sy[None, :] ** 2 + beta**2 * sx[None, :] ** 2 - 2.0 * beta * psi * sx[None, :] * sy[None, :], 1e-12)
        real_components = gx[None, :] * (gy[None, :] - beta * gx[None, :]) / den
        mirror_beta = grid[start:stop, None, None]
        mirror_den = np.maximum(
            msy[None, :, :] ** 2
            + mirror_beta**2 * msx[None, :, :] ** 2
            - 2.0 * mirror_beta * psi * msx[None, :, :] * msy[None, :, :],
            1e-12,
        )
        mirror_components = mbx[None, :, :] * (mby[None, :, :] - mirror_beta * mbx_residual[None, :, :]) / mirror_den
        for block, indices in enumerate(block_indices):
            real_curves[block, start:stop] = np.sum(real_components[:, indices], axis=1)
            mirror_curves[block, :, start:stop] = np.sum(mirror_components[:, :, indices], axis=2).T
    return real_curves, mirror_curves


def _weighted_block_score(
    harmonized: pd.DataFrame,
    replicates: dict[str, np.ndarray],
    beta: float,
    psi: float,
    block_codes: np.ndarray,
    n_blocks: int,
    block_counts: np.ndarray,
    mirror_weights: np.ndarray,
) -> float:
    gx = harmonized["beta_x"].to_numpy(float)
    gy = harmonized["beta_y"].to_numpy(float)
    sx = harmonized["se_x"].to_numpy(float)
    sy = harmonized["se_y"].to_numpy(float)
    real_components = gx * (gy - beta * gx) / corrected_denominator(beta, sx, sy, psi)
    real_by_block = np.bincount(block_codes, weights=real_components, minlength=n_blocks)
    mbx = replicates["beta_x"]
    mbx_residual = replicates.get("beta_x_residual", mbx)
    mby = replicates["beta_y"]
    msx = replicates["se_x"]
    msy = replicates["se_y"]
    mirror_components = mbx * (mby - beta * mbx_residual) / corrected_denominator(beta, msx, msy, psi)
    mirror_mean_components = np.einsum("r,rn->n", mirror_weights, mirror_components, optimize=True)
    mirror_by_block = np.bincount(block_codes, weights=mirror_mean_components, minlength=n_blocks)
    return float(np.dot(block_counts, real_by_block - mirror_by_block))


def _standardized_difference(real: np.ndarray, mirror: np.ndarray) -> float:
    real = np.asarray(real, dtype=float)
    mirror = np.asarray(mirror, dtype=float)
    real = real[np.isfinite(real)]
    mirror = mirror[np.isfinite(mirror)]
    if len(real) == 0 or len(mirror) == 0:
        return np.nan
    mean_difference = float(np.mean(mirror) - np.mean(real))
    pooled = math.sqrt((float(np.var(real)) + float(np.var(mirror))) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0:
        if np.isclose(mean_difference, 0.0, rtol=1e-12, atol=1e-12):
            return 0.0
        return float(np.copysign(np.inf, mean_difference))
    return float(mean_difference / pooled)


def _minor_allele_frequency(value: Any) -> float:
    try:
        frequency = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(frequency) or frequency < 0.0 or frequency > 1.0:
        return np.nan
    return float(min(frequency, 1.0 - frequency))


def _empty_bmbc(reason: str) -> dict[str, Any]:
    return {"method": "BMBC_MR", "beta": np.nan, "se": np.nan, "p": np.nan, "n_instruments": 0, "status": reason}
