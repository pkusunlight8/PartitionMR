"""Input adapters for harmonized variant summaries and signed LD matrices."""
from pathlib import Path
import numpy as np
import pandas as pd
from .api import _ld


def load_summary(path):
    """Read CSV or TSV with SNP, ea, nea, beta_x, se_x, beta_y and se_y."""
    path = Path(path)
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


def load_ld(path):
    """Read a NumPy archive containing corr, snps, a1 and a2 arrays."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"corr", "snps", "a1", "a2"}
        if not required.issubset(archive.files):
            raise ValueError("LD archive needs corr, snps, a1 and a2 arrays")
        return {name: archive[name] for name in required}


def align_summary(frame, reference):
    """Align signed LD to the summary effect allele without dropping variants.

    Exact and swapped allele pairs are supported. Inputs must already share
    a genome build and allele strand. Variant identifiers must be unique.
    """
    required = {"SNP", "ea", "nea", "beta_x", "se_x", "beta_y", "se_y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing summary columns: {sorted(required - set(frame.columns))}")
    if frame.empty or frame["SNP"].isna().any() or frame["SNP"].astype(str).duplicated().any():
        raise ValueError("Summary variants must be nonempty, identified and unique")
    if frame[["ea", "nea"]].isna().any().any():
        raise ValueError("Every summary variant needs effect and other alleles")
    snps = np.asarray(reference["snps"]).astype(str)
    a1, a2 = np.asarray(reference["a1"]).astype(str), np.asarray(reference["a2"]).astype(str)
    if (snps.ndim != 1 or a1.shape != snps.shape or a2.shape != snps.shape
            or len(set(snps)) != len(snps)):
        raise ValueError("Reference identifiers and alleles must be matching unique vectors")
    corr = np.asarray(reference["corr"], dtype=float)
    if corr.shape != (len(snps), len(snps)):
        raise ValueError("Reference correlation dimensions do not match the identifiers")
    lookup = {variant: i for i, variant in enumerate(snps)}
    missing = [variant for variant in frame["SNP"].astype(str) if variant not in lookup]
    if missing:
        raise ValueError(f"Reference is missing {len(missing)} variants: {missing[:5]}")
    indices = np.array([lookup[variant] for variant in frame["SNP"].astype(str)])
    signs = []
    for row, index in zip(frame.itertuples(index=False), indices):
        pair = (str(row.ea).upper(), str(row.nea).upper())
        ref = (a1[index].upper(), a2[index].upper())
        if pair[0] == pair[1] or ref[0] == ref[1]:
            raise ValueError(f"Identical effect and other alleles for {row.SNP}")
        if pair == ref:
            signs.append(1.)
        elif pair == ref[::-1]:
            signs.append(-1.)
        else:
            raise ValueError(f"Allele mismatch for {row.SNP}: {pair} versus {ref}")
    R = corr[np.ix_(indices, indices)] * np.outer(signs, signs)
    _ld(R, len(frame))
    return R, {"n_variants": len(frame), "allele_flips": int(np.sum(np.array(signs) < 0)),
               "minimum_ld_eigenvalue": float(np.linalg.eigvalsh(R).min()),
               "variant_order": frame["SNP"].astype(str).tolist()}
