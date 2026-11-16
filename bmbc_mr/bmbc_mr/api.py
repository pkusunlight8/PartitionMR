from __future__ import annotations

from typing import Any

import pandas as pd

from .estimators import all_baseline_methods
from .ld_mirror import select_ld_aware_mirrors
from .mirror import (
    bmbc_estimate,
    bmbc_estimate_discovery_conditional,
    make_mirror_array_replicates,
    mirror_bias_arrays,
)
from .pleiotropy import bmbc_layers_from_calibration


def fit_bmbc_mr(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    mirror_repeats: int = 500,
    block_bootstraps: int = 1000,
    psi: float = 0.0,
    seed: int = 1,
    include_layered: bool = True,
) -> pd.DataFrame:
    rows = all_baseline_methods(harmonized, psi=psi)
    rec, _, _, _ = bmbc_estimate(harmonized, mirror_pool, mirror_repeats, block_bootstraps, psi, seed)
    rows.append(rec)
    if include_layered:
        rows.extend(bmbc_layers_from_calibration(harmonized, rec))
    return pd.DataFrame(rows)


def make_mirror_pool(harmonized: pd.DataFrame, outcome_pool: pd.DataFrame, seed: int = 1) -> tuple[pd.DataFrame, dict[str, Any]]:
    return select_ld_aware_mirrors(harmonized, outcome_pool, seed=seed)


def fit_bmbc_mr_discovery_conditional(
    harmonized: pd.DataFrame,
    p_threshold: float,
    winner_correlation: float,
    clump_r2: float,
    mirror_repeats: int = 500,
    block_bootstraps: int = 1000,
    psi: float = 0.0,
    seed: int = 1,
    ld_corr: Any = None,
    effect_prior_variance_z: float | None = None,
    effect_prior_source: str | None = None,
    clump_r2_matrix: Any = None,
    clump_ld_metric: str | None = None,
    candidate_frame: pd.DataFrame | None = None,
    candidate_ld_corr: Any = None,
    effect_prior_family: str | None = None,
    effect_prior_location_z: float | None = None,
    effect_prior_component_variance_z: float | None = None,
    effect_prior_covariance_model: str = "independent_effects",
) -> pd.DataFrame:
    rec, _, _, _ = bmbc_estimate_discovery_conditional(
        harmonized=harmonized,
        mirror_repeats=mirror_repeats,
        block_bootstraps=block_bootstraps,
        psi=psi,
        p_threshold=p_threshold,
        winner_correlation=winner_correlation,
        clump_r2=clump_r2,
        seed=seed,
        ld_corr=ld_corr,
        effect_prior_variance_z=effect_prior_variance_z,
        effect_prior_source=effect_prior_source,
        clump_r2_matrix=clump_r2_matrix,
        clump_ld_metric=clump_ld_metric,
        candidate_frame=candidate_frame,
        candidate_ld_corr=candidate_ld_corr,
        effect_prior_family=effect_prior_family,
        effect_prior_location_z=effect_prior_location_z,
        effect_prior_component_variance_z=effect_prior_component_variance_z,
        effect_prior_covariance_model=effect_prior_covariance_model,
    )
    return pd.DataFrame([rec])


def estimate_mirror_bias(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    beta: float,
    mirror_repeats: int = 500,
    psi: float = 0.0,
    seed: int = 1,
) -> float:
    reps, _ = make_mirror_array_replicates(harmonized, mirror_pool, mirror_repeats, seed)
    return mirror_bias_arrays(reps, beta, psi)

