from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .estimators import egger, robust_huber
from .mirror import bmbc_estimate


def bmbc_layered_methods(
    harmonized: pd.DataFrame,
    mirror_pool: pd.DataFrame,
    mirror_repeats: int,
    block_bootstraps: int,
    psi: float,
    seed: int,
    compute_diagnostics: bool = True,
) -> list[dict[str, Any]]:
    """Return selection-only and pleiotropy-aware BMBC layers.

    BMBC-IVW is the canonical mirror-calibrated score estimator. BMBC-Egger and
    BMBC-RAPS-like-Huber keep the robust slope machinery separate by applying
    the mirror-estimated IVW shift to the corresponding robust baseline slope.
    The decomposition separates selection calibration from the robust slope.
    """
    bmbc, _, _, _ = bmbc_estimate(
        harmonized,
        mirror_pool,
        mirror_repeats=mirror_repeats,
        block_bootstraps=block_bootstraps,
        psi=psi,
        seed=seed,
        compute_diagnostics=compute_diagnostics,
    )
    return [_rename(bmbc, "BMBC_MR", "selection_calibrated")] + bmbc_layers_from_calibration(harmonized, bmbc)


def bmbc_layers_from_calibration(harmonized: pd.DataFrame, bmbc: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct layered slopes from one already-fitted mirror calibration."""
    shift = bmbc.get("beta_shift_vs_ivw", np.nan)
    rows = []
    eg = egger(harmonized)
    rows.append(_shift_method(eg, shift, "BMBC_Egger", bmbc))
    hb = robust_huber(harmonized)
    rows.append(_shift_method(hb, shift, "BMBC_RAPS_like_Huber", bmbc))
    return rows


def _rename(rec: dict[str, Any], method: str, layer: str) -> dict[str, Any]:
    out = dict(rec)
    out["method"] = method
    out["bmbc_layer"] = layer
    return out


def _shift_method(base: dict[str, Any], shift: float, method: str, bmbc: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out["uncalibrated_baseline_beta"] = out.get("beta", np.nan)
    out["uncalibrated_baseline_se"] = out.get("se", np.nan)
    out["uncalibrated_baseline_p"] = out.get("p", np.nan)
    out["method"] = method
    out["bmbc_layer"] = "selection_calibration_plus_pleiotropy_robust_slope"
    out["selection_shift_applied"] = shift
    out["mirror_status"] = bmbc.get("mirror_status", bmbc.get("status"))
    out["root_status"] = bmbc.get("root_status", "unknown")
    out["root_candidate_count"] = bmbc.get("root_candidate_count", np.nan)
    out["root_score_at_solution"] = bmbc.get("root_score_at_solution", np.nan)
    out["root_derivative"] = bmbc.get("root_derivative", np.nan)
    out["selection_bias_score_at_ivw"] = bmbc.get("selection_bias_score_at_ivw", np.nan)
    if np.isfinite(out.get("beta", np.nan)) and np.isfinite(shift):
        out["beta"] = float(out["beta"] + shift)
        out["se"] = np.nan
        out["p"] = np.nan
        out["inference_status"] = "not_estimable_for_exploratory_shift_approximation"
        out["status"] = "exploratory_shift_approximation"
    else:
        out["status"] = out.get("status", "not_estimable")
    return out
