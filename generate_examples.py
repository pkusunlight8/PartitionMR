"""Regenerate only the small synthetic CSV inputs shipped with this package."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent

def main():
    rng = np.random.default_rng(20261116)
    n = 6
    bx = np.array([.20, -.16, .17, -.19, .15, .18])
    sx = np.full(n, .04)
    part = pd.DataFrame({"SNP": [f"synthetic_p{i+1}" for i in range(n)],
                         "ea": ["A", "C", "G", "T", "A", "C"],
                         "nea": ["G", "T", "A", "C", "G", "T"],
                         "beta_x": bx, "se_x": sx,
                         "beta_y": .25 * bx + rng.normal(0, .012, n),
                         "se_y": np.full(n, .04)})
    ld = .35 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    p = ROOT / "partition_mr" / "examples"
    part.to_csv(p / "synthetic_summary.csv", index=False, float_format="%.12g")
    pd.DataFrame(ld, index=part.SNP, columns=part.SNP).to_csv(p / "synthetic_ld.csv", float_format="%.12g")
    n = 12
    bx = rng.choice([-1., 1.], n) * np.repeat(rng.uniform(.125, .16, n // 2), 2)
    sx = np.full(n, .02)
    eaf = np.full(n, .3)
    real = pd.DataFrame({"SNP": [f"synthetic_b{i+1}" for i in range(n)],
        "block": [f"block_{i//2+1}" for i in range(n)], "beta_x": bx, "se_x": sx,
        "p_x": 2 * norm.sf(np.abs(bx / sx)), "beta_y": .22 * bx + rng.normal(0, .008, n),
        "se_y": np.full(n, .035), "eaf_x": eaf, "eaf_y": eaf, "ld_score": np.full(n, 1.1)})
    pool = pd.DataFrame({"SNP": [f"synthetic_m{i+1}" for i in range(n*4)],
        "block": [f"block_{i//8+1}" for i in range(n*4)],
        "eaf": np.full(n*4, .3),
        "beta_x": np.repeat(np.abs(bx[::2]), 8) * rng.choice([-1., 1.], n*4),
        "beta_x_residual": rng.normal(0, .02, n*4), "se_x": np.full(n*4, .02),
        "beta": rng.normal(0, .025, n*4),
        "se": np.full(n*4, .035), "ld_score": np.full(n*4, 1.1)})
    b = ROOT / "bmbc_mr" / "examples"
    real.to_csv(b / "synthetic_instruments.csv", index=False, float_format="%.12g")
    pool.to_csv(b / "synthetic_mirror_pool.csv", index=False, float_format="%.12g")

if __name__ == "__main__":
    main()
