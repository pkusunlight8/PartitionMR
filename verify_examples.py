"""Check synthetic examples, repeatability, exact finite mixtures and graph symmetry."""
from pathlib import Path
import importlib.util
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("BMBC_USE_NUMBA", "0")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    part = load("part_example", ROOT / "partition_mr/examples/run_example.py")
    r1, s1 = part.main()
    r2, s2 = part.main()
    assert (r1["cdf_low"], r1["cdf_high"], s1["beta"]) == (r2["cdf_low"], r2["cdf_high"], s2["beta"])
    from partitionmr import ConditionalBlock, PosteriorConfig, conditional_test, load_summary, align_summary
    from partition_mr import posterior, selection
    data = pd.read_csv(ROOT / "partition_mr/examples/synthetic_summary.csv")
    R = pd.read_csv(ROOT / "partition_mr/examples/synthetic_ld.csv", index_col=0).to_numpy(float)
    loaded = load_summary(ROOT / "partition_mr/examples/synthetic_summary.csv")
    aligned, alignment = align_summary(loaded, {
        "corr": R, "snps": loaded.SNP.to_numpy(),
        "a1": loaded.ea.to_numpy(), "a2": loaded.nea.to_numpy(),
    })
    np.testing.assert_allclose(aligned, R)
    assert alignment["n_variants"] == len(loaded) and alignment["allele_flips"] == 0
    z = (data.beta_x / data.se_x).to_numpy()
    y, sx, sy = (data[c].to_numpy() for c in ("beta_y", "se_x", "se_y"))
    selected = selection(z, R, 2.5, .2)[0]
    _, _, w, means, Q = posterior(z, R, 3., .75)
    t = z * sx / sy**2 * selected
    beta = .2
    direction = t * beta * sx
    observed = t @ y
    variance = direction @ Q @ direction + (t * sy) @ R @ (t * sy)
    exact = float(w[0] @ ndtr((observed - means[0] @ direction) / np.sqrt(variance)))
    block = ConditionalBlock(z, y, R, sx, sy, selected)
    for engine in ("barycentric", "lattice"):
        r = conditional_test([block], beta=beta, config=PosteriorConfig(mu=3., sigma=.75, mass_tolerance=1e-8), engine=engine)
        assert r["cdf_low"] - 1e-10 <= exact <= r["cdf_high"] + 1e-10
    print("Both probability engines enclose the exact 64-state CDF:", exact)
    bmbc = load("bmbc_example", ROOT / "bmbc_mr/examples/run_example.py")
    a, b = bmbc.main()
    a2, b2 = bmbc.main()
    for first, second in ((a, a2), (b, b2)):
        np.testing.assert_allclose(first[["beta", "se", "p"]].to_numpy(dtype=float),
                                   second[["beta", "se", "p"]].to_numpy(dtype=float), rtol=0, atol=1e-12)
    from bmbc_mr import estimate_mirror_bias, fit_bmbc_mr, make_mirror_pool
    instruments = pd.read_csv(ROOT / "bmbc_mr/examples/synthetic_instruments.csv")
    mirror_pool = pd.read_csv(ROOT / "bmbc_mr/examples/synthetic_mirror_pool.csv")
    mirrors, _ = make_mirror_pool(instruments, mirror_pool, seed=7)
    bias = estimate_mirror_bias(instruments, mirrors, beta=0., mirror_repeats=10, seed=7)
    layered = fit_bmbc_mr(instruments, mirrors, mirror_repeats=10, block_bootstraps=10, seed=7)
    assert np.isfinite(bias) and {"BMBC_MR", "BMBC_Egger", "BMBC_RAPS_like_Huber"}.issubset(set(layered.method))
    print("Input alignment, mirror bias, and layered interface: passed")
    if importlib.util.find_spec("torch"):
        import torch
        graph = load("graph_example", ROOT / "partition_mr/examples/run_graph.py")
        pred = graph.main()
        from partitionmr.graph import OddGraph
        torch.manual_seed(20261116)
        net = OddGraph()
        zt = torch.tensor(z[None, :], dtype=torch.float32)
        Rt = torch.tensor(R[None, :, :], dtype=torch.float32)
        mu, sigma = torch.tensor([3.]), torch.tensor([.75])
        signs = torch.tensor([1., -1., 1., -1., 1., -1.])
        order = torch.tensor([3, 0, 5, 2, 1, 4])
        with torch.no_grad():
            torch.testing.assert_close(net(zt*signs, Rt*signs[:,None]*signs[None,:], mu, sigma), pred*signs, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(net(zt[:,order], Rt[:,order][:,:,order], mu, sigma), pred[:,order], rtol=1e-5, atol=1e-6)
        print("Graph allele and permutation equivariance: passed")
    else:
        print("Graph check skipped: optional PyTorch is not installed")
    print("Synthetic example checks passed")

if __name__ == "__main__":
    main()
