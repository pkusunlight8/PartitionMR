from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats


def ld_score_from_corr(corr: np.ndarray) -> np.ndarray:
    arr = np.asarray(corr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("corr must be a square matrix")
    return np.sum(arr * arr, axis=1) - 1.0


def standardized_mean_difference(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    mean_difference = float(np.nanmean(y) - np.nanmean(x))
    pooled = np.sqrt((np.nanvar(x) + np.nanvar(y)) / 2.0)
    if pooled <= 0 or not np.isfinite(pooled):
        if np.isclose(mean_difference, 0.0, rtol=1e-12, atol=1e-12):
            return 0.0
        return float(np.copysign(np.inf, mean_difference))
    return float(mean_difference / pooled)


def add_ld_score_fallback(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ld_score" in out:
        out["ld_score"] = pd.to_numeric(out["ld_score"], errors="coerce")
    else:
        out["ld_score"] = np.nan
    return out


def select_ld_aware_mirrors(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    seed: int,
    candidates_per_instrument: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select one conditionally matched mirror per real instrument.

    Matching uses block first, then a standardized distance over MAF, outcome
    SE, and LD score. If annotations are absent, the function degrades to the
    available numeric covariates and records that in the QC fields.
    """
    if harmonized.empty or mirror_pool.empty:
        return pd.DataFrame(), {"ld_mirror_status": "empty_pool", "max_abs_smd": np.nan, "n_mirrors": 0}
    rng = np.random.default_rng(seed)
    real = add_ld_score_fallback(harmonized).reset_index(drop=True)
    pool = add_ld_score_fallback(mirror_pool).reset_index(drop=True)
    real_snps = set(real["SNP"].dropna().astype(str)) if "SNP" in real else set()
    excluded_real_snps = int(pool["SNP"].astype(str).isin(real_snps).sum()) if "SNP" in pool else 0
    if "SNP" in pool and real_snps:
        pool = pool[~pool["SNP"].astype(str).isin(real_snps)].reset_index(drop=True)
    if pool.empty:
        return pd.DataFrame(), {
            "ld_mirror_status": "empty_pool_after_real_snp_exclusion",
            "max_abs_smd": np.nan,
            "n_mirrors": 0,
            "excluded_real_snps_from_pool": excluded_real_snps,
        }
    real["block"] = real["block"].astype(str)
    pool["block"] = pool["block"].astype(str)
    rows = []
    used_global = 0
    used_mirrors: set[str] = set()
    for _, inst in real.iterrows():
        bucket = pool[pool["block"] == inst["block"]].copy()
        if bucket.empty:
            bucket = pool.copy()
            used_global += 1
        if "SNP" in bucket and used_mirrors:
            unused = bucket[~bucket["SNP"].astype(str).isin(used_mirrors)]
            if not unused.empty:
                bucket = unused
        bucket = _distance_rank(bucket, inst)
        keep = bucket.head(min(candidates_per_instrument, len(bucket)))
        distances = pd.to_numeric(keep["_distance"], errors="coerce").fillna(keep["_distance"].max()).to_numpy(float)
        weights = np.exp(-np.clip(distances - np.nanmin(distances), 0.0, 50.0))
        weights = weights / weights.sum() if np.isfinite(weights).all() and weights.sum() > 0 else None
        chosen = keep.iloc[int(rng.choice(np.arange(len(keep)), p=weights))]
        used_mirrors.add(str(chosen.get("SNP")))
        mirror_record = chosen.drop(labels=["_distance"], errors="ignore").to_dict()
        mirror_record.update(
            {
                "real_snp": inst.get("SNP"),
                "mirror_snp": chosen.get("SNP"),
                "block": chosen.get("block"),
                "real_eaf": inst.get("eaf_x", inst.get("eaf")),
                "mirror_eaf": chosen.get("eaf"),
                "real_maf": _minor_allele_frequency(inst.get("eaf_x", inst.get("eaf"))),
                "mirror_maf": _minor_allele_frequency(chosen.get("eaf")),
                "real_se": inst.get("se_y", inst.get("se")),
                "mirror_se": chosen.get("se"),
                "real_ld_score": inst.get("ld_score"),
                "mirror_ld_score": chosen.get("ld_score"),
                "distance": chosen.get("_distance"),
            }
        )
        rows.append(mirror_record)
    selected = pd.DataFrame(rows)
    qc = mirror_balance_qc(selected)
    matching_covariates = ["block", "maf", "se"]
    has_reference_ld = bool(pd.to_numeric(real["ld_score"], errors="coerce").notna().any() and pd.to_numeric(pool["ld_score"], errors="coerce").notna().any())
    has_complete_maf = bool(
        pd.to_numeric(selected.get("real_maf"), errors="coerce").notna().all()
        and pd.to_numeric(selected.get("mirror_maf"), errors="coerce").notna().all()
    )
    if has_reference_ld:
        matching_covariates.append("ld_score")
    if not has_complete_maf:
        match_status = "imbalanced_missing_maf"
    elif qc.get("max_abs_smd", np.inf) > 0.10 or used_global > 0:
        match_status = "imbalanced"
    elif has_reference_ld:
        match_status = "ok"
    else:
        match_status = "ok_degraded_matching"
    qc.update(
        {
            "ld_mirror_status": match_status,
            "n_mirrors": len(selected),
            "global_fallback_matches": int(used_global),
            "excluded_real_snps_from_pool": excluded_real_snps,
            "balance_threshold": 0.10,
            "matching_covariates": ",".join(matching_covariates),
            "matching_frequency_metric": "minor_allele_frequency",
            "ld_structure_status": "marginal_ld_score_matched_pairwise_ld_not_verified",
        }
    )
    return selected, qc


def mirror_balance_qc(selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {"eaf_smd": np.nan, "maf_smd": np.nan, "se_smd": np.nan, "ld_score_smd": np.nan, "max_abs_smd": np.nan}
    if {"real_maf", "mirror_maf"}.issubset(selected.columns):
        real_maf = pd.to_numeric(selected["real_maf"], errors="coerce")
        mirror_maf = pd.to_numeric(selected["mirror_maf"], errors="coerce")
    else:
        real_maf = pd.to_numeric(selected["real_eaf"], errors="coerce").map(_minor_allele_frequency)
        mirror_maf = pd.to_numeric(selected["mirror_eaf"], errors="coerce").map(_minor_allele_frequency)
    eaf_smd = standardized_mean_difference(selected["real_eaf"], selected["mirror_eaf"])
    maf_smd = standardized_mean_difference(real_maf, mirror_maf)
    se_smd = standardized_mean_difference(selected["real_se"], selected["mirror_se"])
    ld_smd = standardized_mean_difference(selected["real_ld_score"], selected["mirror_ld_score"])
    vals = [abs(v) for v in [maf_smd, se_smd, ld_smd] if not np.isnan(v)]
    return {
        "eaf_smd": eaf_smd,
        "maf_smd": maf_smd,
        "se_smd": se_smd,
        "ld_score_smd": ld_smd,
        "max_abs_smd": float(max(vals)) if vals else np.nan,
    }


def _distance_rank(bucket: pd.DataFrame, inst: pd.Series) -> pd.DataFrame:
    out = bucket.copy()
    terms = []
    for pool_col, real_key in [("se", "se_y"), ("ld_score", "ld_score")]:
        if pool_col not in out:
            continue
        target = inst.get(real_key, inst.get(pool_col))
        vals = pd.to_numeric(out[pool_col], errors="coerce").to_numpy(float)
        if not np.isfinite(target) or not np.isfinite(vals).any():
            continue
        scale = np.nanstd(vals)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        terms.append(np.abs(vals - float(target)) / scale)
    if "eaf" in out:
        target_maf = _minor_allele_frequency(inst.get("eaf_x", inst.get("eaf")))
        values = pd.to_numeric(out["eaf"], errors="coerce").map(_minor_allele_frequency).to_numpy(float)
        if np.isfinite(target_maf) and np.isfinite(values).any():
            scale = np.nanstd(values)
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            terms.append(np.abs(values - target_maf) / scale)
    if terms:
        out["_distance"] = np.nanmean(np.vstack(terms), axis=0)
    else:
        out["_distance"] = 0.0
    return out.sort_values("_distance", kind="mergesort")


def _minor_allele_frequency(value: object) -> float:
    try:
        frequency = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(frequency) or frequency < 0.0 or frequency > 1.0:
        return np.nan
    return float(min(frequency, 1.0 - frequency))


def stabilize_correlation(corr: np.ndarray, eigenvalue_floor: float = 1e-8) -> np.ndarray:
    """Return a symmetric positive-definite correlation matrix."""
    arr = np.asarray(corr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("corr must be a square matrix")
    if not np.isfinite(arr).all():
        raise ValueError("corr contains non-finite values")
    arr = 0.5 * (arr + arr.T)
    diagonal = np.sqrt(np.clip(np.diag(arr), eigenvalue_floor, None))
    arr = arr / diagonal[:, None] / diagonal[None, :]
    values, vectors = np.linalg.eigh(arr)
    values = np.clip(values, eigenvalue_floor, None)
    arr = (vectors * values) @ vectors.T
    diagonal = np.sqrt(np.clip(np.diag(arr), eigenvalue_floor, None))
    arr = arr / diagonal[:, None] / diagonal[None, :]
    np.fill_diagonal(arr, 1.0)
    return np.clip(arr, -1.0, 1.0)


def correlation_from_ld_metadata(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Construct the block-diagonal AR(1) correlation used by simulations.

    Applied analyses should pass a reference-panel matrix. Identity is returned
    only when no pairwise LD metadata exist, and the caller must retain that
    degraded status in its QC output.
    """
    n = len(frame)
    corr = np.eye(n, dtype=float)
    if n == 0:
        return corr, "empty"
    group_column = "ld_group" if "ld_group" in frame else ("block" if "block" in frame else None)
    if group_column is None or "ld_order" not in frame or "ld_rho" not in frame:
        return corr, "identity_fallback_missing_pairwise_ld_metadata"
    groups = frame[group_column].astype(str).to_numpy()
    order = pd.to_numeric(frame["ld_order"], errors="coerce").to_numpy(float)
    rho_values = pd.to_numeric(frame["ld_rho"], errors="coerce").to_numpy(float)
    if not np.isfinite(order).all() or not np.isfinite(rho_values).all():
        return corr, "identity_fallback_invalid_pairwise_ld_metadata"
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        rho = float(np.median(np.abs(rho_values[idx])))
        distance = np.abs(order[idx, None] - order[idx][None, :])
        corr[np.ix_(idx, idx)] = np.asarray(rho**distance, dtype=float)
    return stabilize_correlation(corr), "simulation_block_ar1_metadata"


def pairwise_ld_qc(
    corr: np.ndarray,
    clump_r2: float,
    tolerance: float = 1e-10,
    clump_r2_matrix: np.ndarray | None = None,
    clump_ld_metric: str | None = None,
) -> dict[str, Any]:
    """Summarize residual pairwise LD after clumping."""
    arr = stabilize_correlation(corr)
    n = arr.shape[0]
    metric = str(clump_ld_metric).strip() if clump_ld_metric is not None else ""
    if not metric:
        metric = (
            "squared_unphased_correlation"
            if clump_r2_matrix is None
            else "provided_r_squared_matrix"
        )
    if n < 2:
        return {
            "pairwise_ld_status": "verified_single_instrument",
            "n_pairwise_ld_pairs": 0,
            "max_pairwise_r2": 0.0,
            "median_pairwise_r2": 0.0,
            "p95_pairwise_r2": 0.0,
            "max_clump_pairwise_r2": 0.0,
            "median_clump_pairwise_r2": 0.0,
            "p95_clump_pairwise_r2": 0.0,
            "pairwise_r2_exceedances": 0,
            "pairwise_r2_exceedance_rate": 0.0,
            "unphased_pairwise_r2_exceedances": 0,
            "clump_ld_metric": metric,
            "clump_r2_threshold": float(clump_r2),
        }
    triangle = np.triu_indices(n, k=1)
    values = np.square(arr[triangle])
    if clump_r2_matrix is None:
        clump_values = values
    else:
        clump_matrix = np.asarray(clump_r2_matrix, dtype=float)
        if clump_matrix.shape != (n, n) or not np.isfinite(clump_matrix).all():
            raise ValueError("clump_r2_matrix must be a finite square matrix matching corr")
        if (
            np.max(np.abs(clump_matrix - clump_matrix.T)) > 1e-8
            or np.any(clump_matrix < -tolerance)
            or np.any(clump_matrix > 1.0 + tolerance)
            or np.max(np.abs(np.diag(clump_matrix) - 1.0)) > 1e-8
        ):
            raise ValueError("clump_r2_matrix is not a valid symmetric r-squared matrix")
        clump_values = np.clip(clump_matrix[triangle], 0.0, 1.0)
    exceedances = int(np.sum(clump_values > float(clump_r2) + float(tolerance)))
    unphased_exceedances = int(np.sum(values > float(clump_r2) + float(tolerance)))
    return {
        "pairwise_ld_status": "verified_pass" if exceedances == 0 else "verified_fail",
        "n_pairwise_ld_pairs": len(values),
        "max_pairwise_r2": float(np.max(values)),
        "median_pairwise_r2": float(np.median(values)),
        "p95_pairwise_r2": float(np.quantile(values, 0.95)),
        "max_clump_pairwise_r2": float(np.max(clump_values)),
        "median_clump_pairwise_r2": float(np.median(clump_values)),
        "p95_clump_pairwise_r2": float(np.quantile(clump_values, 0.95)),
        "pairwise_r2_exceedances": exceedances,
        "pairwise_r2_exceedance_rate": float(exceedances / len(clump_values)),
        "unphased_pairwise_r2_exceedances": unphased_exceedances,
        "clump_ld_metric": metric,
        "clump_r2_threshold": float(clump_r2),
    }


def clump_indices_from_corr(p_values: np.ndarray, corr: np.ndarray, clump_r2: float) -> np.ndarray:
    """Apply deterministic greedy clumping to a supplied LD matrix."""
    p_values = np.asarray(p_values, dtype=float)
    arr = stabilize_correlation(corr)
    if arr.shape != (len(p_values), len(p_values)):
        raise ValueError("corr shape does not match p_values")
    return clump_indices_from_r2(p_values, np.square(arr), clump_r2)


def clump_indices_from_r2(p_values: np.ndarray, r2_matrix: np.ndarray, clump_r2: float) -> np.ndarray:
    """Apply deterministic greedy clumping to a supplied r-squared matrix."""
    p_values = np.asarray(p_values, dtype=float)
    values = np.asarray(r2_matrix, dtype=float)
    if values.shape != (len(p_values), len(p_values)) or not np.isfinite(values).all():
        raise ValueError("r2_matrix shape does not match p_values or contains non-finite values")
    order = np.argsort(np.where(np.isfinite(p_values), p_values, np.inf), kind="mergesort")
    kept: list[int] = []
    for index in order:
        if not np.isfinite(p_values[index]):
            continue
        if kept and np.any(values[index, kept] > float(clump_r2) + 1e-10):
            continue
        kept.append(int(index))
    return np.asarray(kept, dtype=int)


def estimate_symmetric_two_normal_prior_from_candidate_z(
    candidate_discovery_z: np.ndarray,
) -> tuple[float, float, str]:
    """Fit an equal-weight symmetric two-normal effect prior to all candidates.

    The fitted likelihood is ``0.5 N(-mu, sigma_g^2 + 1) +
    0.5 N(mu, sigma_g^2 + 1)``. The unit term is discovery sampling error,
    leaving ``sigma_g^2`` as the within-component effect variance. This model
    accommodates the signed, non-zero instrument-effect modes used by the
    simulation without using selected variants or the known simulation truth.
    """
    values = np.asarray(candidate_discovery_z, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return np.nan, np.nan, "insufficient_full_candidate_z_for_symmetric_two_normal_mle"

    second_moment = float(np.mean(np.square(values)))
    fourth_moment = float(np.mean(np.power(values, 4)))
    location_fourth = max((3.0 * second_moment**2 - fourth_moment) / 2.0, 0.0)
    moment_location = float(location_fourth**0.25)
    moment_component_variance = max(second_moment - moment_location**2 - 1.0, 1e-6)
    upper_location = max(2.0, float(np.quantile(np.abs(values), 0.995)) * 1.5)
    upper_component_variance = max(4.0, second_moment * 4.0)

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        location = float(parameters[0])
        component_variance = float(np.exp(parameters[1]))
        observation_sd = float(np.sqrt(component_variance + 1.0))
        negative = stats.norm.logpdf(values, loc=-location, scale=observation_sd)
        positive = stats.norm.logpdf(values, loc=location, scale=observation_sd)
        return float(-np.sum(np.logaddexp(negative, positive) - np.log(2.0)))

    starts = [
        (moment_location, moment_component_variance),
        (float(np.median(np.abs(values))), max(float(np.var(values)) * 0.20, 1e-3)),
        (0.0, max(second_moment - 1.0, 1e-3)),
    ]
    fits = []
    bounds = [(0.0, upper_location), (float(np.log(1e-6)), float(np.log(upper_component_variance)))]
    for location, component_variance in starts:
        fit = optimize.minimize(
            negative_log_likelihood,
            x0=np.array(
                [
                    min(max(float(location), 0.0), upper_location),
                    np.log(min(max(float(component_variance), 1e-6), upper_component_variance)),
                ]
            ),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if fit.success and np.isfinite(fit.fun) and np.isfinite(fit.x).all():
            fits.append(fit)
    if not fits:
        return (
            moment_location,
            moment_component_variance,
            "full_candidate_symmetric_two_normal_moment_fallback",
        )
    best = min(fits, key=lambda fit: float(fit.fun))
    return (
        float(best.x[0]),
        float(np.exp(best.x[1])),
        "full_candidate_symmetric_two_normal_mle",
    )


def _stabilize_covariance(covariance: np.ndarray, eigenvalue_floor: float = 1e-10) -> np.ndarray:
    values = np.asarray(covariance, dtype=float)
    values = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    eigenvalues = np.clip(eigenvalues, eigenvalue_floor * scale, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _selected_summary_gaussian_posterior(
    observed_discovery_z: np.ndarray,
    observed_exposure_z: np.ndarray,
    corr: np.ndarray,
    prior_variance: float,
    winner_correlation: float,
    prior_covariance_model: str = "independent_effects",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the exact Gaussian posterior under independent effect priors.

    The sufficient observation is the mean of discovery and estimation Z
    statistics. Its error covariance is ``(1 + rho) / 2 * R``. The effect
    prior is ``N(0, tau^2 I)``, so the posterior covariance is not generally a
    scalar multiple of ``R``.
    """
    discovery_z = np.asarray(observed_discovery_z, dtype=float)
    exposure_z = np.asarray(observed_exposure_z, dtype=float)
    correlation = stabilize_correlation(corr)
    if discovery_z.shape != exposure_z.shape or correlation.shape != (
        len(discovery_z),
        len(discovery_z),
    ):
        raise ValueError("selected-summary posterior inputs have incompatible shapes")
    tau_squared = float(prior_variance)
    rho = float(winner_correlation)
    if not np.isfinite(tau_squared) or tau_squared < 0.0:
        raise ValueError("prior_variance must be finite and nonnegative")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("winner_correlation must be in [-1, 1]")
    if prior_covariance_model not in {"independent_effects", "ld_correlated_effects"}:
        raise ValueError(
            "prior_covariance_model must be independent_effects or ld_correlated_effects"
        )

    average_z = 0.5 * (discovery_z + exposure_z)
    if tau_squared <= 1e-12:
        return (
            np.zeros_like(average_z),
            np.zeros_like(correlation),
            "selected_summary_degenerate_zero_effect_prior",
        )
    if rho <= -1.0 + 1e-8:
        return (
            average_z,
            np.zeros_like(correlation),
            "selected_summary_exact_signal_from_antithetic_errors",
        )

    error_scale = 0.5 * (1.0 + rho)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    error_eigenvalues = error_scale * eigenvalues
    if prior_covariance_model == "independent_effects":
        prior_eigenvalues = np.full_like(eigenvalues, tau_squared)
        posterior_model = (
            "selected_summary_multivariate_gaussian_independent_effect_prior_ld_error"
        )
    else:
        prior_eigenvalues = tau_squared * eigenvalues
        posterior_model = (
            "selected_summary_multivariate_gaussian_ld_correlated_effect_prior_ld_error"
        )
    posterior_shrinkage = prior_eigenvalues / (
        prior_eigenvalues + error_eigenvalues
    )
    posterior_variance_eigenvalues = (
        prior_eigenvalues
        * error_eigenvalues
        / (prior_eigenvalues + error_eigenvalues)
    )
    posterior_mean = eigenvectors @ (
        posterior_shrinkage * (eigenvectors.T @ average_z)
    )
    posterior_covariance = (
        eigenvectors * posterior_variance_eigenvalues
    ) @ eigenvectors.T
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.T
    )
    return (
        posterior_mean,
        posterior_covariance,
        posterior_model,
    )


def _full_candidate_conditional_posterior_draws(
    real: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    repeats: int,
    p_threshold: float,
    winner_correlation: float,
    clump_r2: float,
    rng: np.random.Generator,
    effect_prior_location_z: float | None,
    effect_prior_component_variance_z: float | None,
    effect_prior_source: str | None,
    candidate_ld_corr: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Condition selected effects on all candidate discovery statistics by block."""
    required = {"SNP", "beta_x_discovery", "se_x", "p_x", "block"}
    missing = sorted(required.difference(candidate_frame.columns))
    if missing or "SNP" not in real:
        return None, None, {
            "mirror_status": "invalid_full_candidate_schema",
            "full_candidate_schema_missing": ",".join(missing or ["SNP"]),
        }
    candidates = candidate_frame.reset_index(drop=True).copy()
    candidate_snps = candidates["SNP"].astype(str)
    real_snps = real["SNP"].astype(str)
    if candidate_snps.duplicated().any() or real_snps.duplicated().any():
        return None, None, {"mirror_status": "duplicate_variant_ids_in_full_candidate_conditioning"}
    lookup = pd.Series(np.arange(len(candidates), dtype=int), index=candidate_snps)
    if not real_snps.isin(lookup.index).all():
        return None, None, {"mirror_status": "selected_variants_missing_from_full_candidates"}
    selected_candidate_index = lookup.loc[real_snps].to_numpy(int)

    candidate_se = pd.to_numeric(candidates["se_x"], errors="coerce").to_numpy(float)
    candidate_beta = pd.to_numeric(candidates["beta_x_discovery"], errors="coerce").to_numpy(float)
    candidate_p = pd.to_numeric(candidates["p_x"], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(candidate_se).all()
        or np.any(candidate_se <= 0.0)
        or not np.isfinite(candidate_beta).all()
        or not np.isfinite(candidate_p).all()
    ):
        return None, None, {"mirror_status": "invalid_full_candidate_statistics"}
    candidate_z, candidate_z_source = signed_z_from_beta_se_p(candidate_beta, candidate_se, candidate_p)
    if candidate_ld_corr is None:
        candidate_corr, candidate_corr_source = correlation_from_ld_metadata(candidates)
    else:
        candidate_corr = stabilize_correlation(candidate_ld_corr)
        if candidate_corr.shape != (len(candidates), len(candidates)):
            return None, None, {"mirror_status": "invalid_full_candidate_ld_matrix_shape"}
        candidate_corr_source = "provided_full_candidate_ld_matrix"

    thresholded_p = np.where(candidate_p <= float(p_threshold), candidate_p, np.inf)
    replay_index = clump_indices_from_corr(thresholded_p, candidate_corr, clump_r2)
    replay_snps = set(candidate_snps.iloc[replay_index])
    observed_snps = set(real_snps)
    replay_status = "exact_match" if replay_snps == observed_snps else "mismatch"
    if replay_status != "exact_match":
        return None, None, {
            "mirror_status": "full_candidate_selection_replay_mismatch",
            "candidate_selection_replay_status": replay_status,
            "candidate_selection_replay_missing": len(observed_snps.difference(replay_snps)),
            "candidate_selection_replay_unexpected": len(replay_snps.difference(observed_snps)),
        }

    if effect_prior_location_z is None or effect_prior_component_variance_z is None:
        prior_location, prior_component_variance, prior_source = estimate_symmetric_two_normal_prior_from_candidate_z(
            candidate_z
        )
    else:
        prior_location = float(effect_prior_location_z)
        prior_component_variance = float(effect_prior_component_variance_z)
        prior_source = str(effect_prior_source or "provided_full_candidate_symmetric_two_normal_prior")
    if (
        not np.isfinite(prior_location)
        or prior_location < 0.0
        or not np.isfinite(prior_component_variance)
        or prior_component_variance <= 0.0
    ):
        return None, None, {"mirror_status": "invalid_full_candidate_effect_prior"}

    observed_exposure_z = pd.to_numeric(real["beta_x"], errors="coerce").to_numpy(float) / pd.to_numeric(
        real["se_x"], errors="coerce"
    ).to_numpy(float)
    selected_corr = stabilize_correlation(candidate_corr[np.ix_(selected_candidate_index, selected_candidate_index)])
    posterior_draws = np.empty((int(repeats), len(real)), dtype=float)
    posterior_mean = np.empty(len(real), dtype=float)
    posterior_variance_diagonal = np.empty(len(real), dtype=float)
    real_position_by_candidate = {int(candidate_index): real_index for real_index, candidate_index in enumerate(selected_candidate_index)}
    candidate_blocks = candidates["block"].astype(str).to_numpy()
    rho = float(winner_correlation)

    for block in pd.unique(candidate_blocks):
        block_candidate_index = np.flatnonzero(candidate_blocks == block)
        block_selected_candidate = [index for index in block_candidate_index if int(index) in real_position_by_candidate]
        if not block_selected_candidate:
            continue
        block_lookup = {int(index): position for position, index in enumerate(block_candidate_index)}
        selected_local = np.asarray([block_lookup[int(index)] for index in block_selected_candidate], dtype=int)
        selected_real = np.asarray([real_position_by_candidate[int(index)] for index in block_selected_candidate], dtype=int)
        block_corr = stabilize_correlation(candidate_corr[np.ix_(block_candidate_index, block_candidate_index)])
        block_size = len(block_candidate_index)
        selected_size = len(selected_local)
        design = np.vstack([np.eye(block_size), np.eye(block_size)[selected_local]])
        observation_covariance = np.block(
            [
                [block_corr, rho * block_corr[:, selected_local]],
                [rho * block_corr[selected_local, :], block_corr[np.ix_(selected_local, selected_local)]],
            ]
        )
        observation_precision = np.linalg.pinv(_stabilize_covariance(observation_covariance), hermitian=True)
        prior_precision = 1.0 / max(prior_component_variance, 1e-10)
        posterior_precision = prior_precision * np.eye(block_size) + design.T @ observation_precision @ design
        posterior_covariance = _stabilize_covariance(np.linalg.pinv(posterior_precision, hermitian=True))
        block_z = candidate_z[block_candidate_index]
        component_sign = np.sign(block_z)
        component_sign[component_sign == 0.0] = 1.0
        prior_mean = prior_location * component_sign
        block_exposure_z = observed_exposure_z[selected_real]
        observations = np.concatenate([block_z, block_exposure_z])
        block_posterior_mean = posterior_covariance @ (
            prior_precision * prior_mean + design.T @ observation_precision @ observations
        )
        selected_posterior_mean = block_posterior_mean[selected_local]
        selected_posterior_covariance = _stabilize_covariance(
            posterior_covariance[np.ix_(selected_local, selected_local)]
        )
        block_draws = rng.multivariate_normal(
            selected_posterior_mean,
            selected_posterior_covariance,
            size=int(repeats),
            check_valid="raise",
            method="eigh",
        )
        posterior_draws[:, selected_real] = np.asarray(block_draws).reshape(int(repeats), selected_size)
        posterior_mean[selected_real] = selected_posterior_mean
        posterior_variance_diagonal[selected_real] = np.diag(selected_posterior_covariance)

    diagnostics = {
        "candidate_conditioning_scope": "all_preselection_candidates",
        "candidate_count_conditioned": len(candidates),
        "candidate_selected_count": len(real),
        "candidate_selection_replay_status": replay_status,
        "candidate_selection_replay_missing": 0,
        "candidate_selection_replay_unexpected": 0,
        "candidate_ld_correlation_source": candidate_corr_source,
        "candidate_discovery_z_source": candidate_z_source,
        "effect_prior_family": "symmetric_two_normal_fixed_component_map",
        "effect_prior_location_z": float(prior_location),
        "effect_prior_component_variance_z": float(prior_component_variance),
        "effect_prior_variance_z": float(prior_location**2 + prior_component_variance),
        "effect_prior_source": prior_source,
        "conditional_posterior_model": "full_candidate_ld_block_gaussian_given_symmetric_component_map",
        "posterior_variance_mean_z": float(np.mean(posterior_variance_diagonal)),
    }
    if "true_effect_z" in candidates:
        true_effect = pd.to_numeric(candidates["true_effect_z"], errors="coerce").to_numpy(float)
        valid_truth = np.isfinite(true_effect) & (true_effect != 0.0)
        candidate_component_sign = np.sign(candidate_z)
        diagnostics["candidate_component_sign_error_rate"] = (
            float(np.mean(candidate_component_sign[valid_truth] != np.sign(true_effect[valid_truth])))
            if valid_truth.any()
            else np.nan
        )
        selected_truth = true_effect[selected_candidate_index]
        selected_valid = np.isfinite(selected_truth) & (selected_truth != 0.0)
        diagnostics["selected_component_sign_error_rate"] = (
            float(
                np.mean(
                    np.sign(candidate_z[selected_candidate_index][selected_valid])
                    != np.sign(selected_truth[selected_valid])
                )
            )
            if selected_valid.any()
            else np.nan
        )
    else:
        diagnostics["candidate_component_sign_error_rate"] = np.nan
        diagnostics["selected_component_sign_error_rate"] = np.nan
    return posterior_draws, selected_corr, diagnostics


def discovery_conditional_mirror_replicates(
    harmonized: pd.DataFrame,
    repeats: int,
    p_threshold: float,
    winner_correlation: float,
    psi: float,
    seed: int,
    clump_r2: float,
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
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Generate discovery-side null mirrors from an empirical-Bayes residual law.

    With full preselection candidates, the preferred construction jointly
    conditions each LD block on all discovery Z scores and the selected
    estimation Z scores under a fitted symmetric two-normal effect prior. With
    selected summaries, the function uses a zero-mean Gaussian conditional
    model and records the conditioning scope.
    """
    if harmonized.empty:
        return None, {"mirror_status": "empty_harmonized_input", "n_mirror_variants": 0}
    if int(repeats) < 1:
        return None, {"mirror_status": "invalid_mirror_repeat_count", "n_mirror_variants": 0}
    if not 0.0 < float(p_threshold) < 1.0:
        return None, {"mirror_status": "invalid_discovery_p_threshold", "n_mirror_variants": 0}
    if not -1.0 <= float(winner_correlation) <= 1.0 or not -1.0 <= float(psi) <= 1.0:
        return None, {"mirror_status": "invalid_correlation_parameter", "n_mirror_variants": 0}
    if effect_prior_covariance_model not in {
        "independent_effects",
        "ld_correlated_effects",
    }:
        return None, {
            "mirror_status": "invalid_effect_prior_covariance_model",
            "n_mirror_variants": 0,
        }
    required = {"beta_x", "se_x", "se_y", "block"}
    missing = sorted(required.difference(harmonized.columns))
    if missing:
        return None, {
            "mirror_status": "invalid_real_schema",
            "mirror_schema_missing": ",".join(missing),
            "n_mirror_variants": 0,
        }

    real = harmonized.reset_index(drop=True).copy()
    se_x = pd.to_numeric(real["se_x"], errors="coerce").to_numpy(float)
    se_y = pd.to_numeric(real["se_y"], errors="coerce").to_numpy(float)
    if not np.isfinite(se_x).all() or not np.isfinite(se_y).all() or np.any(se_x <= 0) or np.any(se_y <= 0):
        return None, {"mirror_status": "invalid_standard_errors", "n_mirror_variants": 0}

    discovery_beta_column = "beta_x_discovery" if "beta_x_discovery" in real else "beta_x"
    discovery_beta = pd.to_numeric(real[discovery_beta_column], errors="coerce").to_numpy(float)
    discovery_p_column = "p_x_discovery" if "p_x_discovery" in real else "p_x"
    discovery_p = (
        pd.to_numeric(real[discovery_p_column], errors="coerce").to_numpy(float)
        if discovery_p_column in real
        else None
    )
    observed_discovery_z, discovery_z_source = signed_z_from_beta_se_p(
        discovery_beta,
        se_x,
        discovery_p,
    )
    observed_exposure_z = pd.to_numeric(real["beta_x"], errors="coerce").to_numpy(float) / se_x
    if not np.isfinite(observed_discovery_z).all() or not np.isfinite(observed_exposure_z).all():
        return None, {"mirror_status": "invalid_exposure_z", "n_mirror_variants": 0}
    threshold_z = float(stats.norm.isf(float(p_threshold) / 2.0))
    if discovery_p is not None:
        valid_discovery_p = np.isfinite(discovery_p) & (discovery_p >= 0.0) & (discovery_p <= 1.0)
        observed_selection_pass = np.where(
            valid_discovery_p,
            discovery_p <= float(p_threshold),
            np.abs(observed_discovery_z) >= threshold_z - 1e-10,
        )
    else:
        observed_selection_pass = np.abs(observed_discovery_z) >= threshold_z - 1e-10
    rng = np.random.default_rng(seed)
    rho = float(winner_correlation)
    candidate_diagnostics: dict[str, Any] = {}
    use_full_candidates = candidate_frame is not None and not candidate_frame.empty
    if use_full_candidates:
        posterior_signal, candidate_selected_corr, candidate_diagnostics = _full_candidate_conditional_posterior_draws(
            real=real,
            candidate_frame=candidate_frame,
            repeats=int(repeats),
            p_threshold=float(p_threshold),
            winner_correlation=rho,
            clump_r2=float(clump_r2),
            rng=rng,
            effect_prior_location_z=effect_prior_location_z,
            effect_prior_component_variance_z=effect_prior_component_variance_z,
            effect_prior_source=effect_prior_source,
            candidate_ld_corr=candidate_ld_corr,
        )
        if posterior_signal is None or candidate_selected_corr is None:
            return None, {
                "n_mirror_variants": 0,
                "candidate_conditioning_scope": "all_preselection_candidates_requested",
                **candidate_diagnostics,
            }
        corr = candidate_selected_corr
        corr_source = str(candidate_diagnostics.get("candidate_ld_correlation_source", "full_candidate_ld"))
        prior_location = float(candidate_diagnostics["effect_prior_location_z"])
        prior_component_variance = float(candidate_diagnostics["effect_prior_component_variance_z"])
        prior_variance = float(candidate_diagnostics["effect_prior_variance_z"])
        prior_source = str(candidate_diagnostics["effect_prior_source"])
        prior_family = str(candidate_diagnostics["effect_prior_family"])
        posterior_model = str(candidate_diagnostics["conditional_posterior_model"])
        posterior_mean = np.mean(posterior_signal, axis=0)
        posterior_variance = float(candidate_diagnostics.get("posterior_variance_mean_z", np.nan))
    else:
        if ld_corr is None:
            corr, corr_source = correlation_from_ld_metadata(real)
        else:
            corr = stabilize_correlation(ld_corr)
            if corr.shape != (len(real), len(real)):
                return None, {"mirror_status": "invalid_ld_matrix_shape", "n_mirror_variants": 0}
            corr_source = str(ld_correlation_source or "provided_reference_ld_matrix")
        if effect_prior_variance_z is None:
            prior_variance, prior_source = estimate_effect_prior_variance_from_selected_z(
                observed_discovery_z,
                threshold_z,
            )
        else:
            prior_variance = float(effect_prior_variance_z)
            prior_source = str(effect_prior_source or "provided_preselection_discovery_z_moment")
        if not np.isfinite(prior_variance) or prior_variance < 0.0:
            return None, {"mirror_status": "invalid_effect_prior_variance", "n_mirror_variants": 0}
        prior_location = 0.0
        prior_component_variance = prior_variance
        prior_family = str(effect_prior_family or "zero_mean_gaussian")
        posterior_mean, posterior_covariance, posterior_model = (
            _selected_summary_gaussian_posterior(
                observed_discovery_z,
                observed_exposure_z,
                corr,
                prior_variance,
                rho,
                effect_prior_covariance_model,
            )
        )
        posterior_variance = float(np.mean(np.diag(posterior_covariance)))
        if np.max(np.abs(posterior_covariance)) <= 1e-15:
            posterior_signal = np.broadcast_to(
                posterior_mean[None, :],
                (int(repeats), len(real)),
            ).copy()
        else:
            posterior_signal = rng.multivariate_normal(
                posterior_mean,
                posterior_covariance,
                size=int(repeats),
                check_valid="raise",
                method="eigh",
            )

    if use_full_candidates:
        effective_clump_r2_matrix = np.square(corr)
        effective_clump_ld_metric = clump_ld_metric or "known_simulation_squared_correlation_r2"
    elif clump_r2_matrix is None:
        effective_clump_r2_matrix = np.square(corr)
        effective_clump_ld_metric = clump_ld_metric or "squared_unphased_correlation"
    else:
        effective_clump_r2_matrix = np.asarray(clump_r2_matrix, dtype=float)
        effective_clump_ld_metric = clump_ld_metric or "provided_r_squared_matrix"
    pairwise_qc = pairwise_ld_qc(
        corr,
        clump_r2,
        clump_r2_matrix=effective_clump_r2_matrix,
        clump_ld_metric=effective_clump_ld_metric,
    )
    r = int(repeats)
    n = len(real)
    outcome_noise_z = rng.multivariate_normal(
        np.zeros(n),
        corr,
        size=r,
        check_valid="raise",
        method="eigh",
    )
    z_discovery = np.broadcast_to(observed_discovery_z[None, :], (r, n)).copy()
    z_exposure = np.broadcast_to(observed_exposure_z[None, :], (r, n)).copy()
    z_exposure_residual = z_exposure - posterior_signal
    z_outcome = (
        float(psi) * z_exposure_residual
        + np.sqrt(max(1.0 - float(psi) ** 2, 0.0)) * outcome_noise_z
    )
    discovery_p = 2.0 * stats.norm.sf(np.abs(z_discovery))
    selection_pass = np.broadcast_to(observed_selection_pass[None, :], (r, n))
    if pairwise_qc["pairwise_ld_status"] in {"verified_pass", "verified_single_instrument"}:
        retained_counts = np.full(r, n, dtype=int)
    else:
        retained_counts = np.asarray(
            [
                len(
                    clump_indices_from_r2(
                        discovery_p[i],
                        effective_clump_r2_matrix,
                        clump_r2,
                    )
                )
                for i in range(r)
            ],
            dtype=int,
        )
    selection_pass_rate = float(np.mean(selection_pass))
    same_pipeline_retention_rate = float(np.mean(retained_counts == n))
    reference_status = str(reference_ld_status or "not_separately_reported")
    degraded_ld = (
        (corr_source.startswith("identity_fallback") and n > 1)
        or "partial" in corr_source
        or "partial" in reference_status
    )
    degraded_prior = prior_source.startswith("selected_only") or not use_full_candidates
    if reference_status.startswith(("insufficient", "invalid")):
        mirror_status = "reference_ld_qc_failed"
    elif selection_pass_rate < 1.0 - 1e-12:
        mirror_status = "conditional_selection_failure"
    elif pairwise_qc["pairwise_ld_status"] == "verified_fail" or same_pipeline_retention_rate < 1.0:
        mirror_status = "imbalanced_pairwise_ld"
    elif degraded_ld or degraded_prior:
        mirror_status = "ok_degraded_matching"
    else:
        mirror_status = "ok"

    reps = {
        "beta_x": z_exposure * se_x[None, :],
        "beta_x_residual": z_exposure_residual * se_x[None, :],
        "se_x": np.broadcast_to(se_x[None, :], (r, n)).copy(),
        "beta_y": z_outcome * se_y[None, :],
        "se_y": np.broadcast_to(se_y[None, :], (r, n)).copy(),
        "z_discovery": z_discovery,
        "posterior_signal_z": posterior_signal,
        "outcome_noise_z": outcome_noise_z,
    }
    observed_abs_z = np.abs(observed_exposure_z)
    generated_abs_z = np.abs(z_exposure).ravel()
    diagnostics = {
        "mirror_status": mirror_status,
        "mirror_exposure_source": (
            "full_candidate_discovery_conditioned_exposure_error_residual"
            if use_full_candidates
            else "observed_selected_exposure_with_empirical_bayes_error_residual"
        ),
        "mirror_generation_assumption": (
            "symmetric_two_normal_effect_prior_plus_ld_correlated_gaussian_summary_errors_conditioned_on_all_candidate_discovery_and_selected_estimation_z"
            if use_full_candidates
            else "zero_mean_gaussian_effect_prior_plus_correlated_gaussian_summary_errors_conditional_on_selected_discovery_and_estimation_z"
        ),
        "n_mirror_variants": int(r * n),
        "mirror_repeats": r,
        "conditional_sampler_sweeps": 0,
        "conditional_posterior_model": posterior_model,
        "discovery_z_source": discovery_z_source,
        "effect_prior_variance_z": prior_variance,
        "effect_prior_sd_z": float(np.sqrt(prior_variance)),
        "effect_prior_family": prior_family,
        "effect_prior_location_z": float(prior_location),
        "effect_prior_component_variance_z": float(prior_component_variance),
        "effect_prior_source": prior_source,
        "effect_prior_covariance_model": (
            "symmetric_component_independent_effects"
            if use_full_candidates
            else effect_prior_covariance_model
        ),
        "posterior_variance_mean_z": float(posterior_variance),
        "posterior_signal_shrinkage": float(np.mean(np.divide(posterior_mean, observed_exposure_z, out=np.full(n, np.nan), where=np.abs(observed_exposure_z) > 1e-12))),
        "posterior_exposure_error_mean_abs_z": float(np.mean(np.abs(z_exposure_residual))),
        "posterior_exposure_error_sd_z": float(np.std(z_exposure_residual, ddof=1)) if z_exposure_residual.size > 1 else 0.0,
        "discovery_p_threshold": float(p_threshold),
        "discovery_z_threshold": threshold_z,
        "conditional_selection_pass_rate": selection_pass_rate,
        "same_pipeline_retention_rate": same_pipeline_retention_rate,
        "winner_reuse_correlation": float(winner_correlation),
        "sample_overlap_psi": float(psi),
        "ld_correlation_source": corr_source,
        "reference_ld_status": reference_status,
        "ld_structure_status": (
            "identity_fallback_not_reference_verified"
            if degraded_ld
            else f"pairwise_ld_{pairwise_qc['pairwise_ld_status']}"
        ),
        "observed_mean_abs_exposure_z": float(np.nanmean(observed_abs_z)),
        "conditional_mean_abs_exposure_z": float(np.nanmean(generated_abs_z)),
        "exposure_z_smd": standardized_mean_difference(
            np.repeat(observed_abs_z, r),
            generated_abs_z,
        ),
        "eaf_smd": 0.0,
        "maf_smd": 0.0,
        "se_smd": 0.0,
        "ld_score_smd": 0.0,
        "max_abs_smd": 0.0,
        "balance_threshold": 0.10,
        "matching_covariates": "block,maf,se,observed_exposure_z,full_pairwise_ld,selection_threshold",
        "matching_frequency_metric": "minor_allele_frequency",
        "global_fallback_instruments": 0,
        "candidate_conditioning_scope": "all_preselection_candidates" if use_full_candidates else "selected_summaries_only",
        **pairwise_qc,
        **candidate_diagnostics,
    }
    return reps, diagnostics


def conditional_mirror_replicates_for_psi(
    replicates: dict[str, np.ndarray] | None,
    psi: float,
) -> dict[str, np.ndarray] | None:
    """Re-evaluate conditional mirrors at ``psi`` using common latent draws.

    ``discovery_conditional_mirror_replicates`` retains the posterior exposure
    residual and the independent outcome-noise draw. Reusing both arrays makes
    a sample-overlap curve vary only ``psi`` rather than its Monte Carlo sample.
    """
    if replicates is None:
        return None
    if not -1.0 <= float(psi) <= 1.0:
        raise ValueError("psi must be in [-1, 1]")
    required = {"beta_x_residual", "se_x", "se_y", "outcome_noise_z"}
    missing = sorted(required.difference(replicates))
    if missing:
        raise ValueError(f"conditional mirror replicates lack arrays: {missing}")
    beta_x_residual = np.asarray(replicates["beta_x_residual"], dtype=float)
    se_x = np.asarray(replicates["se_x"], dtype=float)
    se_y = np.asarray(replicates["se_y"], dtype=float)
    outcome_noise_z = np.asarray(replicates["outcome_noise_z"], dtype=float)
    if not (
        beta_x_residual.shape == se_x.shape == se_y.shape == outcome_noise_z.shape
    ):
        raise ValueError("conditional mirror replicate arrays must have identical shapes")
    if (
        not np.isfinite(beta_x_residual).all()
        or not np.isfinite(se_x).all()
        or not np.isfinite(se_y).all()
        or not np.isfinite(outcome_noise_z).all()
        or np.any(se_x <= 0.0)
        or np.any(se_y <= 0.0)
    ):
        raise ValueError("conditional mirror replicate arrays must be finite with positive SEs")
    exposure_residual_z = beta_x_residual / se_x
    outcome_z = (
        float(psi) * exposure_residual_z
        + np.sqrt(max(1.0 - float(psi) ** 2, 0.0)) * outcome_noise_z
    )
    transformed = dict(replicates)
    transformed["beta_y"] = outcome_z * se_y
    return transformed


def signed_z_from_beta_se_p(
    beta: np.ndarray | pd.Series,
    se: np.ndarray | pd.Series,
    p_value: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, str]:
    """Recover signed Z scores, preferring authoritative GWAS P values.

    Public summary statistics often round beta and SE enough to move a selected
    variant just below its discovery Z threshold. A valid P value therefore
    supplies the magnitude, while beta supplies the direction. Invalid or
    absent P values fall back elementwise to beta / SE.
    """
    effects = np.asarray(beta, dtype=float)
    errors = np.asarray(se, dtype=float)
    ratio = np.divide(effects, errors, out=np.full(effects.shape, np.nan), where=errors > 0.0)
    if p_value is None:
        return ratio, "beta_over_se_fallback"
    probabilities = np.asarray(p_value, dtype=float)
    if probabilities.shape != effects.shape:
        raise ValueError("p-value shape does not match beta and se")
    valid = np.isfinite(probabilities) & (probabilities >= 0.0) & (probabilities <= 1.0)
    clipped = np.clip(probabilities, np.finfo(float).tiny, 1.0)
    magnitude = stats.norm.isf(clipped / 2.0)
    direction = np.sign(effects)
    direction = np.where(direction == 0.0, np.sign(ratio), direction)
    reconstructed = direction * magnitude
    usable = valid & np.isfinite(reconstructed)
    values = np.where(usable, reconstructed, ratio)
    source = "signed_p_value_with_beta_over_se_fallback" if not usable.all() else "signed_p_value"
    return values, source


def estimate_effect_prior_variance_from_selected_z(
    observed_discovery_z: np.ndarray,
    threshold_z: float,
) -> tuple[float, str]:
    """Estimate a Gaussian effect variance from selected Z scores only.

    This fallback is identifiable from tail excess but less stable than a
    preselection genome-wide moment and is therefore always labelled degraded.
    """
    values = np.asarray(observed_discovery_z, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) >= float(threshold_z) - 1e-10)]
    if len(values) < 2:
        return max(float(np.mean(values**2)) - 1.0, 0.0) if len(values) else 0.0, "selected_only_moment_degraded"

    sum_squares = float(np.sum(values**2))

    def negative_log_likelihood(log_sigma: float) -> float:
        sigma = float(np.exp(log_sigma))
        log_tail = float(np.log(2.0) + stats.norm.logsf(float(threshold_z) / sigma))
        return len(values) * np.log(sigma) + sum_squares / (2.0 * sigma**2) + len(values) * log_tail

    upper_sigma = max(10.0, float(np.max(np.abs(values))) * 4.0)
    fit = optimize.minimize_scalar(
        negative_log_likelihood,
        bounds=(0.0, float(np.log(upper_sigma))),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not fit.success or not np.isfinite(fit.x):
        return max(float(np.mean(values**2)) - 1.0, 0.0), "selected_only_moment_degraded"
    sigma = float(np.exp(fit.x))
    return max(sigma**2 - 1.0, 0.0), "selected_only_truncated_normal_mle_degraded"


def _sample_sign_truncated_gaussian(
    corr: np.ndarray,
    signs: np.ndarray,
    threshold: float,
    repeats: int,
    sweeps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized Gibbs sampler for a sign-constrained Gaussian tail."""
    corr = stabilize_correlation(corr)
    signs = np.asarray(signs, dtype=float)
    k = len(signs)
    if corr.shape != (k, k):
        raise ValueError("corr shape does not match signs")
    draws = np.empty((int(repeats), k), dtype=float)
    for j in range(k):
        draws[:, j] = _draw_truncated_normal(
            np.zeros(int(repeats)),
            np.ones(int(repeats)),
            float(signs[j]),
            float(threshold),
            rng,
        )
    if k == 1:
        return draws
    precision = np.linalg.pinv(corr, hermitian=True)
    for _ in range(int(sweeps)):
        for j in range(k):
            qjj = max(float(precision[j, j]), 1e-12)
            conditional_sd = np.full(int(repeats), np.sqrt(1.0 / qjj))
            conditional_mean = -(draws @ precision[j, :] - draws[:, j] * qjj) / qjj
            draws[:, j] = _draw_truncated_normal(
                conditional_mean,
                conditional_sd,
                float(signs[j]),
                float(threshold),
                rng,
            )
    return draws


def _draw_truncated_normal(
    mean: np.ndarray,
    sd: np.ndarray,
    sign: float,
    threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    eps = np.finfo(float).eps
    if sign >= 0:
        lower = stats.norm.cdf((float(threshold) - mean) / sd)
        u = lower + (1.0 - lower) * rng.random(len(mean))
    else:
        upper = stats.norm.cdf((-float(threshold) - mean) / sd)
        u = upper * rng.random(len(mean))
    u = np.clip(u, eps, 1.0 - eps)
    return mean + sd * stats.norm.ppf(u)
