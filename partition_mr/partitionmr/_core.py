"""Load the numerical engines included with the source distribution."""
from pathlib import Path
import sys

# The numerical engines are stored in the adjacent code directory.
source = Path(__file__).resolve().parents[1] / "code"
if (source / "adaptive_partition.py").is_file():
    sys.path.insert(0, str(source))

from adaptive_partition import adaptive_posterior, lattice_cdf_interval
from barycentric_convolution import barycentric_cdf_interval
from partition_mr import confidence_set, score_p, score_terms, selection
