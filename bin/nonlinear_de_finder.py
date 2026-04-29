#!/usr/bin/env python3

"""
Compute sigmoid elbow (d_e) from stress metrics and print values.

Supported methods:
  - mMDS
  - PCA
  - KPCA
  - TSNE
  - ISOMAP
  - UMAP
  - SE

Authors:
    Miko Stulajter

Version 2.0.0

Usage: nonlinear_de_finder.py <metrics_file>
"""

import argparse
import os
import re

import numpy as np
from scipy.optimize import curve_fit

COMPONENTS_TO_USE = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))


def r2_sigmoid(x, dy_dx, popt):
    y_pred = sigmoid(x, *popt)
    ss_res = np.sum((dy_dx - y_pred) ** 2)
    ss_tot = np.sum((dy_dx - np.mean(dy_dx)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def parse_stress_d_in(metrics_path):
    with open(metrics_path) as f:
        content = f.read()
    match = re.search(r"Stress:\s*([-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?)", content)
    return float(match.group(1)) if match else None


def split_series_filename(file_name):
    """
    Split filename into prefix + component number + suffix around `_c###_`.
    Example:
      C2H4O_UMAP_c096_weighted_error_metrics.csv
      -> ("C2H4O_UMAP", 96, "_weighted_error_metrics.csv")
    """
    match = re.match(r"^(?P<prefix>.+)_c(?P<comp>\d+)(?P<suffix>_.+)$", file_name)
    if not match:
        raise ValueError(
            "Input filename must contain component token like '_c096_' "
            "(e.g. C2H4O_METHOD_c096_weighted_error_metrics.csv)."
        )
    prefix = match.group("prefix")
    component = int(match.group("comp"))
    suffix = match.group("suffix")
    return prefix, component, suffix


def load_stress_series_from_seed_file(seed_file):
    """
    Load stress values for all matching component files in same directory.
    Matching pattern is inferred from seed file name.
    """
    seed_file = os.path.abspath(seed_file)
    if not os.path.isfile(seed_file):
        raise FileNotFoundError(f"Input file not found: {seed_file}")

    directory = os.path.dirname(seed_file)
    seed_name = os.path.basename(seed_file)
    prefix, _, suffix = split_series_filename(seed_name)
    pattern = re.compile(
        rf"^{re.escape(prefix)}_c(?P<comp>\d+){re.escape(suffix)}$"
    )

    stress_by_component = {}
    for name in sorted(os.listdir(directory)):
        full_path = os.path.join(directory, name)
        if not os.path.isfile(full_path):
            continue
        match = pattern.match(name)
        if not match:
            continue
        n_comp = int(match.group("comp"))
        val = parse_stress_d_in(full_path)
        if val is not None:
            stress_by_component[n_comp] = val

    return prefix, suffix, stress_by_component


def estimate_de(stress_vals, comp_vals):
    x = np.log2(comp_vals)
    y = np.log2(stress_vals)
    dy_dx = np.gradient(y, x)
    p0 = [1.0, np.mean(x), np.max(y) - np.min(y)]

    try:
        popt, _ = curve_fit(sigmoid, x, dy_dx, p0=p0, maxfev=20000)
    except RuntimeError:
        idx = np.argmax(np.abs(dy_dx))
        popt = [1.0, x[idx], dy_dx[idx]]

    d_e = float(np.exp2(popt[1]))
    r2 = r2_sigmoid(x, dy_dx, popt)
    return d_e, r2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate d_e from stress metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s RN_encoder_c064_weighted_error_metrics.csv
  %(prog)s RN_encoder_c096_weighted_error_metrics.csv
        """,
    )
    parser.add_argument(
        "metrics_file",
        help=(
            "Path to one *_error_metrics.csv file with a '_cNNN_' token; "
            "method-agnostic (mMDS/PCA/KPCA/TSNE/ISOMAP/UMAP/SE/etc.), used to infer full component series"
        ),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    series_name, series_suffix, stress_by_component = load_stress_series_from_seed_file(
        args.metrics_file
    )
    if not stress_by_component:
        raise SystemExit(
            f"No stress values parsed for inferred series: {series_name}*{series_suffix}"
        )

    comps = sorted(c for c in COMPONENTS_TO_USE if c in stress_by_component)
    if len(comps) < 2:
        raise SystemExit(
            f"Need at least 2 supported components from {COMPONENTS_TO_USE}; found {comps}"
        )

    stress_vals = [stress_by_component[c] for c in comps]
    d_e, r2 = estimate_de(stress_vals, comps)
    print(f"{series_name}: d_e={d_e:.4f}, r2={r2:.4f}, components={comps}")


if __name__ == "__main__":
    main()
