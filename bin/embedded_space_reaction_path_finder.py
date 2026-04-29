#!/usr/bin/env python3

"""
Find reaction paths using embedding-derived edge weights.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: embedded_space_reaction_path_finder.py <npz> <graphml> -r <reactants> -p <products> [--delta DELTA] [--output OUTPUT]
"""

import argparse
import bz2
import os
import sys

import networkx as nx
import numpy as np

def _import_rpf():
    """Import path-finder helpers from local graphml_reaction_path_finder.py."""
    try:
        import graphml_reaction_path_finder as rpf  # type: ignore
        return rpf
    except ImportError as exc:
        raise RuntimeError(
            f"could not import graphml_reaction_path_finder.py from this repository: {exc}"
        ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find lowest-cost path (Dijkstra), then all paths within cutoff depth (BFS). "
            "Weights are embedding distances from NPZ coordinates."
        )
    )
    parser.add_argument(
        "npz",
        type=str,
        help="Path to NPZ file containing embedding coordinates",
    )
    parser.add_argument(
        "graphml",
        type=str,
        help="Path to GraphML(.bz2) file",
    )
    parser.add_argument(
        "-r",
        type=str,
        required=True,
        help="Reactants (SMILES)",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=True,
        help="Products (SMILES)",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=0,
        help="Cutoff depth = shortest path length + DELTA (default: 0)",
    )
    parser.add_argument(
        "--hide-paths",
        action="store_true",
        default=False,
        help="Hide Path1, Path2, etc. labels in output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print verbose output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output path. If omitted, saves in NPZ directory as generated filename. "
            "If a directory is provided, it must already exist."
        ),
    )
    return parser.parse_args()


def load_coords_and_mapping(npz_path: str) -> tuple[np.ndarray, dict[str, int]]:
    """Load coords from npz and build mapping GraphML_node_id -> embedding row index."""
    npz_path = os.path.abspath(npz_path)
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            coords = np.asarray(data["coords"])
            node_ids = data["node_ids"] if "node_ids" in data else None
    except Exception as exc:
        raise RuntimeError(f"failed to load npz '{npz_path}': {exc}") from exc

    n = coords.shape[0]
    id_to_index = {}

    if node_ids is not None:
        node_ids = list(node_ids)
        if len(node_ids) != n:
            raise ValueError(f"npz node_ids length {len(node_ids)} != coords rows {n}")
        for i, nid in enumerate(node_ids):
            key = str(nid) if not isinstance(nid, str) else nid
            id_to_index[key] = i
    else:
        import csv as csv_module
        csv_path = npz_path.rsplit(".", 1)[0] + "_node_features.csv"
        if not os.path.isfile(csv_path):
            raise ValueError(f"npz has no node_ids and no paired CSV at {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
        if not rows or "node" not in rows[0]:
            raise ValueError("CSV must have 'node' column")
        node_ids_csv = [r["node"] for r in rows]
        if len(node_ids_csv) != n:
            raise ValueError(f"CSV rows {len(node_ids_csv)} != coords rows {n}")
        for i, nid in enumerate(node_ids_csv):
            id_to_index[str(nid)] = i

    return coords, id_to_index


def load_graphml(graphml_path: str) -> nx.Graph:
    """Load GraphML (optionally bz2-compressed) into a NetworkX graph."""
    if graphml_path.endswith(".bz2"):
        with bz2.open(graphml_path, mode="rt", encoding="utf-8") as gfile:
            G = nx.read_graphml(gfile)
    else:
        with open(graphml_path, mode="r", encoding="utf-8") as gfile:
            G = nx.read_graphml(gfile)
    return G


def apply_embedding_weights_to_graph(G: nx.Graph, coords: np.ndarray, id_to_index: dict) -> None:
    """Set edge weight = Euclidean distance between endpoints in embedding."""
    missing = [node for node in G.nodes if node not in id_to_index]
    if missing:
        raise ValueError(
            f"{len(missing)} GraphML nodes missing from npz node_ids. First few: {missing[:5]}"
        )
    for u, v in G.edges():
        iu = id_to_index[u]
        iv = id_to_index[v]
        d = float(np.linalg.norm(coords[iu] - coords[iv]))
        G.edges[u, v]["weight"] = d


def main() -> None:
    try:
        args = parse_arguments()
        rpf = _import_rpf()
        npz_path = os.path.abspath(args.npz)
        graphml = os.path.abspath(args.graphml)
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        if not os.path.isfile(graphml):
            raise FileNotFoundError(f"GraphML file not found: {graphml}")
        rn = os.path.splitext(os.path.basename(npz_path))[0]

        coords, id_to_index = load_coords_and_mapping(npz_path)
        G = load_graphml(graphml)
        apply_embedding_weights_to_graph(G, coords, id_to_index)

        node_labels = {node: data.get("smiles", node) for node, data in G.nodes(data=True)}
        rule_key = rpf.check_rule_key(G) or None
        weight_key = "weight"

        reactant_node = rpf.find_node_for_label(node_labels, args.r)
        product_node = rpf.find_node_for_label(node_labels, args.p)

        outname = rpf.make_output_path(rn, args.delta)
        if args.output is None:
            outpath_full = os.path.join(os.path.dirname(npz_path), outname)
        else:
            output_arg = os.path.abspath(args.output)
            if os.path.isdir(output_arg):
                outpath_full = os.path.join(output_arg, outname)
            else:
                parent_dir = os.path.dirname(output_arg) or "."
                if not os.path.isdir(parent_dir):
                    raise FileNotFoundError(f"Output directory does not exist: {parent_dir}")
                outpath_full = output_arg

        with open(outpath_full, "w", encoding="utf-8") as outfile:
            real_stdout = sys.stdout
            sys.stdout = outfile
            try:
                _run(args, G, node_labels, reactant_node, product_node, weight_key, rule_key, rpf, outpath_full)
            finally:
                sys.stdout = real_stdout

        print(f"Output saved to {os.path.abspath(outpath_full)}", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


def _run(
    args: argparse.Namespace,
    G: nx.Graph,
    node_labels: dict,
    reactant_node: str,
    product_node: str,
    weight_key: str,
    rule_key: str | None,
    rpf,
    outpath: str,
) -> None:
    print(f"# Output file: {outpath}\n")
    print("# Weights = embedding (npz) Euclidean distances\n")
    print(f"Reactant: {args.r} (node: {reactant_node})")
    print(f"Product: {args.p} (node: {product_node})")

    try:
        path_nodes = nx.dijkstra_path(G, reactant_node, product_node, weight=weight_key)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print("No path found between reactant and product.")
        return

    path_length_steps = len(path_nodes) - 1
    path_cost = rpf.compute_path_cost(path_nodes, G, weight_key)
    print(f"\nDijkstra shortest path: {path_length_steps} steps, cost = {path_cost:.9f}")
    path_str = " --> ".join(node_labels[n] for n in path_nodes)
    print(f"  Path: {path_str}\n")

    cutoff = path_length_steps + args.delta
    print(f"BFS cutoff depth: {path_length_steps} + {args.delta} = {cutoff} steps\n")

    G_cropped = rpf.crop_graph_to_max_length(G, reactant_node, product_node, cutoff)
    if G_cropped.number_of_nodes() < G.number_of_nodes():
        print(f"Graph optimization: {G.number_of_nodes()} -> {G_cropped.number_of_nodes()} nodes\n")

    path_costs = rpf.bfs_paths_up_to_length(
        G_cropped,
        reactant_node,
        product_node,
        cutoff,
        weight_key,
        node_labels,
        verbose=args.verbose,
    )
    print(f"BFS found {len(path_costs)} path(s) with length <= {cutoff}\n")

    if not path_costs:
        print("No paths found within cutoff.")
        return

    analysis_results = rpf.analyze_paths(G, path_costs, rule_key, weight_key)
    rpf.print_results(analysis_results, hide_paths=args.hide_paths)


if __name__ == "__main__":
    main()
