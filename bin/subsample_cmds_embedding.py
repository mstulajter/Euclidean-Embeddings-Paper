#!/usr/bin/env python3
"""
cMDS component runner: read the full cMDS result npz (e.g. c256) and the pipeline
input npz; take cmds_matrix[:, :k] for each requested k and run the rest of the
Embedding_pipeline output format (.npz, .csv) as normal.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: subsample_cmds_embedding.py --source-npz SOURCE_NPZ --source-components SOURCE_COMPONENTS [--components COMPONENTS] [--output-dir OUTPUT_DIR]
"""

import argparse
import os
import re
import numpy as np
import pandas as pd

import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from linear_embeddings import (
    combine_files,
    get_nodes_graphml,
)


def get_cmds_component(cmds_matrix, components):
    """Return first `components` columns of cMDS coordinate matrix."""
    return cmds_matrix[:, :components]


def load_labels_from_input_csv(csv_path, node_ids, source_coords, rtol=1e-6, atol=1e-8):
    df = pd.read_csv(csv_path)
    required_cols = {"node", "coordinates"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required_cols}: {csv_path}")

    labels = [str(x) for x in df["node"].tolist()]
    if len(labels) != len(node_ids):
        raise ValueError(
            f"Input CSV row count ({len(labels)}) != node_ids count ({len(node_ids)})"
        )

    csv_coords = []
    for coord_str in df["coordinates"].tolist():
        if not isinstance(coord_str, str):
            raise ValueError("Input CSV 'coordinates' must be colon-delimited strings")
        csv_coords.append([float(x) for x in coord_str.split(":")])
    csv_coords = np.asarray(csv_coords, dtype=np.float64)

    source_coords = np.asarray(source_coords, dtype=np.float64)
    if csv_coords.shape != source_coords.shape:
        raise ValueError(
            f"Input CSV coordinates shape {csv_coords.shape} != source coords shape {source_coords.shape}"
        )

    if not np.allclose(csv_coords, source_coords, rtol=rtol, atol=atol):
        max_abs_diff = float(np.max(np.abs(csv_coords - source_coords)))
        raise ValueError(
            f"Input CSV coordinates do not match source NPZ coords (max_abs_diff={max_abs_diff:.3e})"
        )

    return labels


def build_parser():
    parser = argparse.ArgumentParser(
        description="Derive cMDS component outputs from a full cMDS NPZ."
    )
    parser.add_argument(
        "--source-npz",
        required=True,
        help="Full cMDS result NPZ; must contain 'coords'",
    )
    parser.add_argument(
        "--input-npz",
        required=True,
        help="Input NPZ with 'og_matrix' and 'node_ids'",
    )
    parser.add_argument(
        "--components",
        "-c",
        type=int,
        nargs="+",
        required=True,
        help="Component counts",
    )
    parser.add_argument(
        "--graphml",
        type=str,
        default=None,
        help="Optional GraphML.bz2 for SMILES labels in CSV (fallback if --input-csv not provided)",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional source CSV to reuse node labels directly",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (default: same directory as --source-npz)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    with np.load(args.source_npz, allow_pickle=True) as data:
        cmds_matrix = np.asarray(data["coords"])
        source_node_ids = [str(x) for x in data["node_ids"]] if "node_ids" in data else None
    n_samples, n_full = cmds_matrix.shape

    with np.load(args.input_npz, allow_pickle=True) as data:
        D_in = np.asarray(data["og_matrix"])
        node_ids = [str(x) for x in data["node_ids"]]

    try:
        node_ids_sorted = sorted(node_ids, key=lambda x: float(x))
    except (ValueError, TypeError):
        node_ids_sorted = sorted(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    reorder_idx = [id_to_idx[nid] for nid in node_ids_sorted]
    D_in = np.asarray(D_in)[np.ix_(reorder_idx, reorder_idx)]
    node_ids = node_ids_sorted

    if source_node_ids is not None:
        source_nid_to_row = {nid: i for i, nid in enumerate(source_node_ids)}
        reorder_coords = [source_nid_to_row[nid] for nid in node_ids]
        cmds_matrix = np.asarray(cmds_matrix)[reorder_coords]

    if len(node_ids) != n_samples:
        raise SystemExit("Source coords rows must match input node_ids size.")

    if args.input_csv:
        node_labels = load_labels_from_input_csv(args.input_csv, node_ids, cmds_matrix)
    elif args.graphml:
        graphml_args = argparse.Namespace(original_bz2=args.graphml)
        node_labels = get_nodes_graphml(graphml_args, node_ids)
    else:
        node_labels = node_ids

    source_abspath = os.path.abspath(args.source_npz)
    source_dir = os.path.dirname(source_abspath)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else source_dir
    source_basename = os.path.basename(args.source_npz)
    basename = re.sub(r"_c\d+_.*\.npz$", "", source_basename)
    weight_suffix = "weight" if "weight" in source_basename and "unweighted" not in source_basename else "unweighted"

    invalid = [k for k in args.components if k < 1 or k > n_full]
    if invalid:
        raise SystemExit(f"Component counts must be in [1, {n_full}]. Invalid: {invalid}")

    for k in args.components:
        Y_k = get_cmds_component(cmds_matrix, k)

        cstr = str(k).zfill(3)
        out_stem = f"{basename}_c{cstr}_{weight_suffix}"
        out_npz = os.path.join(output_dir, out_stem + ".npz")
        out_csv = os.path.join(output_dir, out_stem + ".csv")

        np.savez_compressed(
            out_npz,
            coords=Y_k,
            node_ids=np.array(node_ids),
            method=np.array(["cMDS"]),
            components=np.array([k]),
        )
        combine_files(Y_k, node_labels, out_csv)

        print(f"Wrote c{k}: {out_npz}, {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
