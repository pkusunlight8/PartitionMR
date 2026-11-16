"""Run both BMBC-MR calibration paths on synthetic data."""
from pathlib import Path
import sys
import os
os.environ.setdefault("BMBC_USE_NUMBA", "0")
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bmbc_mr import make_mirror_pool, fit_bmbc_mr, fit_bmbc_mr_discovery_conditional

def main():
    here = Path(__file__).resolve().parent
    data = pd.read_csv(here / "synthetic_instruments.csv")
    pool = pd.read_csv(here / "synthetic_mirror_pool.csv")
    selected, qc = make_mirror_pool(data, pool, seed=7)
    assert len(selected) == len(data)
    fitted = fit_bmbc_mr(data, selected, mirror_repeats=50, block_bootstraps=80,
                         psi=0., seed=7, include_layered=False)
    result = fitted.loc[fitted.method == "BMBC_MR"].iloc[0]
    assert result.status == "ok" and result.inference_status == "block_bootstrap"
    assert all(np.isfinite(float(result[x])) for x in ("beta", "se", "p"))
    conditional = fit_bmbc_mr_discovery_conditional(
        data, p_threshold=5e-8, winner_correlation=.9, clump_r2=.01,
        mirror_repeats=50, block_bootstraps=80, psi=0., seed=7,
        ld_corr=np.eye(len(data)), effect_prior_variance_z=4.,
        effect_prior_source="synthetic example prior")
    cond = conditional.iloc[0]
    assert cond.status in {"ok", "ok_degraded_matching"} and cond.inference_status == "block_bootstrap"
    assert all(np.isfinite(float(cond[x])) for x in ("beta", "se", "p"))
    print(f"Mirror matching: n={len(selected)}, status={qc['ld_mirror_status']}")
    print(f"BMBC-MR: beta={result.beta:.8g}, status={result.status}, inference={result.inference_status}")
    print(f"Discovery conditional: beta={cond.beta:.8g}, status={cond.status}, inference={cond.inference_status}")
    return result, cond

if __name__ == "__main__":
    main()
