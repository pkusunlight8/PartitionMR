# Partition MR and BMBC-MR

This repository provides core Python implementations of **Partition Mendelian randomization (Partition MR)** and **Block-Mirror Bias-Calibrated Mendelian randomization (BMBC-MR)**. It includes runnable examples, small synthetic datasets, and a verification script.

## Getting started

Python 3.10 or later is required. Create a virtual environment and install the core dependencies:

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS or Linux, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then run:

```bash
python -m pip install -r requirements.txt
python partition_mr/examples/run_example.py
python bmbc_mr/examples/run_example.py
python verify_examples.py
```

The examples load the included CSV files, run each method, and print their results. The verification script checks reproducibility, compares both Partition MR probability engines with an exhaustive 64-state calculation, and checks BMBC-MR mirror calibration and discovery-conditional estimation.

For the optional graph-network example, install PyTorch:

```bash
python -m pip install torch
python partition_mr/examples/run_graph.py
```

The graph example initializes `OddGraph` with a fixed random seed and demonstrates a forward pass. When PyTorch is installed, the verification script also tests allele-sign and permutation equivariance. The example weights are randomly initialized.

Numba is an optional accelerator for BMBC-MR. `requirements-optional.txt` lists both Numba and PyTorch. The BMBC-MR example uses the NumPy backend by default; setting `BMBC_USE_NUMBA=1` before running it enables JIT compilation when Numba is installed.

## Repository layout

```text
partition_mr/
  partitionmr/       Public interface, input alignment, and optional graph model
  code/              Posterior and probability calculation engines
  examples/          Synthetic summary statistics, LD matrix, and examples
bmbc_mr/
  bmbc_mr/           Mirror matching, calibration, and estimators
  examples/          Synthetic instruments, mirror pool, and example
generate_examples.py Regenerate the synthetic CSV inputs with a fixed seed
verify_examples.py   Check the examples and numerical properties
```

`generate_examples.py` overwrites the synthetic CSV files in the two `examples/` directories. All included data are generated for demonstration.

## Partition MR

The Partition MR interface exposes `ConditionalBlock`, `PosteriorConfig`, `conditional_test`, `summary_score`, `load_summary`, `load_ld`, and `align_summary`. The first example constructs a candidate block and evaluates both the sign-mixture conditional test and the selected-summary Gaussian score.

The example summary table has one row per candidate variant. `SNP` identifies the variant; `ea` and `nea` specify the effect and other alleles; `beta_x` and `beta_y` are exposure and outcome associations; and `se_x` and `se_y` are their standard errors. The signed LD correlation matrix uses the same variant order and effect-allele orientation as the summary table. `ConditionalBlock` takes exposure Z scores, outcome effects, standard errors, and the LD matrix. `conditional_test` returns a CDF interval, a two-sided probability bound, and convergence information.

To import the interface from another script, add the `partition_mr` directory to Python's module search path, or run Python from that directory:

```python
from partitionmr import ConditionalBlock, PosteriorConfig, conditional_test, summary_score
```

## BMBC-MR

The BMBC-MR interface provides `make_mirror_pool`, `fit_bmbc_mr`, `fit_bmbc_mr_discovery_conditional`, and `estimate_mirror_bias`. The example constructs matched mirrors, fits the mirror-calibrated model, and runs discovery-conditional estimation.

The instrument table contains `SNP`, LD `block`, exposure `beta_x`, `se_x`, and `p_x`, outcome `beta_y` and `se_y`, effect-allele frequencies `eaf_x` and `eaf_y`, and matching annotation `ld_score`. The example mirror pool contains `SNP`, `block`, `eaf`, `beta`, `se`, `ld_score`, `beta_x`, `se_x`, and `beta_x_residual`. The output includes effect estimates, inference results, and diagnostic status fields. Discovery-conditional analysis accepts selection parameters, LD information, and an exposure-effect prior.

To import the interface from another script, add the `bmbc_mr` directory to Python's module search path, or run Python from that directory:

```python
from bmbc_mr import make_mirror_pool, fit_bmbc_mr, fit_bmbc_mr_discovery_conditional
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Taikang Yao and Ning Shen.
