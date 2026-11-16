"""Run a randomly initialized graph on synthetic inputs."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from partitionmr.graph import OddGraph

def main():
    here = Path(__file__).resolve().parent
    data = pd.read_csv(here / "synthetic_summary.csv")
    ld = pd.read_csv(here / "synthetic_ld.csv", index_col=0).to_numpy()
    torch.manual_seed(20261116)
    net = OddGraph()
    z = torch.tensor((data.beta_x / data.se_x).to_numpy()[None, :], dtype=torch.float32)
    with torch.no_grad():
        prediction = net(z, torch.tensor(ld[None, :, :], dtype=torch.float32),
                         torch.tensor([3.]), torch.tensor([.75]))
    assert prediction.shape == z.shape and torch.isfinite(prediction).all()
    print("Untrained graph output:", np.round(prediction.numpy(), 6).tolist())
    return prediction

if __name__ == "__main__":
    main()
