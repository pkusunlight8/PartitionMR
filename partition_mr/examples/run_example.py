"""Run full-candidate and selected-summary inference on synthetic data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partitionmr import ConditionalBlock, PosteriorConfig, conditional_test, summary_score

def main():
    here = Path(__file__).resolve().parent
    data = pd.read_csv(here / "synthetic_summary.csv")
    ld_frame = pd.read_csv(here / "synthetic_ld.csv", index_col=0)
    assert list(ld_frame.index) == list(data.SNP) == list(ld_frame.columns)
    ld = ld_frame.to_numpy(float)
    block = ConditionalBlock(z=(data.beta_x / data.se_x).to_numpy(), y=data.beta_y.to_numpy(),
                             ld=ld, sx=data.se_x.to_numpy(), sy=data.se_y.to_numpy(), name="synthetic")
    config = PosteriorConfig(mu=3., sigma=.75, mass_tolerance=1e-8)
    result = conditional_test([block], beta=0., config=config, threshold=2.5, clump_r2=.2)
    summary = summary_score(data.beta_x, data.se_x, data.beta_y, data.se_y, ld, prior_variance=4.)
    assert result["converged"] and result["posterior_converged"]
    assert 0 <= result["cdf_low"] <= result["cdf_high"] <= 1 and 0 <= result["p_upper"] <= 1
    assert np.isfinite(summary["beta"]) and 0 <= summary["p"] <= 1
    print(f"Partition MR: candidates={result['candidate_count']}, selected={result['selected_count']}, "
          f"CDF=[{result['cdf_low']:.8g}, {result['cdf_high']:.8g}], p_upper={result['p_upper']:.8g}")
    print(f"Summary score: beta={summary['beta']:.8g}, p={summary['p']:.8g}")
    return result, summary

if __name__ == "__main__":
    main()
