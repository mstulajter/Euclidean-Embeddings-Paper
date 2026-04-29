#!/usr/bin/env python3

"""
Nonlinear embedding pipeline.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: nonlinear_embeddings.py <graphml_file> [--distance-npz DISTANCE_NPZ] [--save-distance-npz [SAVE_DISTANCE_NPZ]] [options]
"""

import argparse
import bz2
import os
import xml.etree.ElementTree as ET

import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import Isomap, MDS, SpectralEmbedding, TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


def preprocess_matrix(D, node_ids):
    n = len(node_ids)
    H = np.eye(n) - np.ones((n, n)) / n
    A = -0.5 * (D**2)
    return H @ A @ H


def symmetrize(D):
    return (D + D.T) / 2


def double_centering(D):
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean


def double_centering_symmetrization(D):
    centered = double_centering(D)
    return (centered + centered.T) / 2


def mmds(matrix, components):
    model = MDS(n_components=components, dissimilarity="precomputed", metric=True, n_init=4)
    return model.fit_transform(matrix)


def pca_embed(matrix, components):
    matrix = StandardScaler().fit_transform(matrix)
    return PCA(n_components=components).fit_transform(matrix)


def kpca_embed(matrix, components, kernel, gamma=None, degree=None, coef0=None):
    options = {"kernel": kernel}
    if gamma is not None:
        options["gamma"] = gamma
    if degree is not None:
        options["degree"] = degree
    if coef0 is not None:
        options["coef0"] = coef0
    model = KernelPCA(n_components=components, **options, max_iter=100000)
    return model.fit_transform(matrix)


def tsne_embed(matrix, components, perplexity=None, early_exaggeration=None, learning_rate=None, max_iter=None, exact=False):
    options = {"n_components": components, "init": "random", "metric": "precomputed"}
    if perplexity is not None:
        options["perplexity"] = perplexity
    if early_exaggeration is not None:
        options["early_exaggeration"] = early_exaggeration
    if learning_rate is not None:
        options["learning_rate"] = learning_rate
    if max_iter is not None:
        options["max_iter"] = max_iter
    if exact or components >= 4:
        options["method"] = "exact"
    return TSNE(**options).fit_transform(matrix)


def isomap_embed(matrix, components, n_neighbors=None, radius=None):
    if n_neighbors is not None:
        model = Isomap(n_components=components, n_neighbors=n_neighbors, radius=None, metric="precomputed")
    elif radius is not None:
        model = Isomap(n_components=components, n_neighbors=None, radius=radius, metric="precomputed")
    else:
        model = Isomap(n_components=components, metric="precomputed")
    return model.fit_transform(matrix)


def umap_embed(matrix, components, n_neighbors=None, min_dist=None, spread=None, target_metric=None):
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("UMAP is not installed in this environment.") from exc

    options = {"n_components": components, "metric": "precomputed"}
    if n_neighbors is not None:
        options["n_neighbors"] = n_neighbors
    if min_dist is not None:
        options["min_dist"] = min_dist
    if spread is not None:
        options["spread"] = spread
    if target_metric is not None:
        options["target_metric"] = target_metric
    return umap.UMAP(**options).fit_transform(matrix)


def se_embed(matrix, components, n_neighbors=None):
    k = n_neighbors if n_neighbors is not None else 10
    affinity = kneighbors_graph(matrix, n_neighbors=k, mode="connectivity", include_self=False)
    model = SpectralEmbedding(n_components=components, affinity="precomputed")
    return model.fit_transform(affinity)


def combine_files(coord_matrix, labels, out_file):
    formatted = [":".join(map(str, coords)) for coords in coord_matrix]
    pd.DataFrame({"node": labels, "coordinates": formatted}).to_csv(out_file, index=False)


def get_nodes_graphml(graphml_file, node_ids):
    with bz2.open(graphml_file, "rb") as f:
        root = ET.fromstring(f.read())
    graph = root.find("{http://graphml.graphdrawing.org/xmlns}graph")
    lookup = {
        node.attrib["id"]: node.find("{http://graphml.graphdrawing.org/xmlns}data").text
        for node in graph.findall("{http://graphml.graphdrawing.org/xmlns}node")
    }
    return [lookup[nid] for nid in node_ids]


def sort_node_ids(node_ids):
    try:
        return sorted(node_ids, key=lambda x: float(x))
    except (ValueError, TypeError):
        return sorted(node_ids)


def sort_node_ids_and_matrix(node_ids, D):
    node_ids_sorted = sort_node_ids(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    reorder_idx = [id_to_idx[nid] for nid in node_ids_sorted]
    D_reordered = np.asarray(D)[np.ix_(reorder_idx, reorder_idx)]
    return node_ids_sorted, D_reordered


def compute_distance_matrix_from_graphml(graphml_file, weight_key="weight", cores=1):
    nk.setNumberOfThreads(int(cores))
    with bz2.open(graphml_file, "rb") as f:
        graph_nx = nx.read_graphml(f)

    sorted_nodes = sort_node_ids(list(graph_nx.nodes()))
    sub = graph_nx.subgraph(sorted_nodes)
    ordered_graph = nx.Graph()
    ordered_graph.add_nodes_from(sorted_nodes)
    ordered_graph.add_edges_from(sub.edges(data=True))
    graph_nk = nk.nxadapter.nx2nk(ordered_graph, weightAttr=weight_key)

    all_nodes = list(range(graph_nk.numberOfNodes()))
    spsp = nk.distance.SPSP(graph_nk, all_nodes).run()
    n = len(all_nodes)
    D = np.zeros((n, n), dtype=np.float64)
    for i, source in enumerate(all_nodes):
        for j, target in enumerate(all_nodes[i + 1 :], start=i + 1):
            d = spsp.getDistance(source, target)
            D[i, j] = d
            D[j, i] = d
    return D, [str(x) for x in sorted_nodes]


def load_distance_matrix_npz(path):
    with np.load(path, allow_pickle=True) as data:
        if "og_matrix" not in data or "node_ids" not in data:
            raise ValueError(f"{path} must contain 'og_matrix' and 'node_ids'")
        D = np.asarray(data["og_matrix"])
        node_ids = [str(x) for x in data["node_ids"]]
    return sort_node_ids_and_matrix(node_ids, D)


def save_distance_matrix_npz(path, D, node_ids):
    np.savez_compressed(path, og_matrix=np.asarray(D), node_ids=np.array(node_ids))
    print(f"Saved distance cache: {path}")


def default_distance_cache_path(graphml_file, weight_key):
    rn = os.path.basename(graphml_file).replace(".graphml.bz2", "").replace(".graphml", "")
    mode = "unweighted" if weight_key is None else "weighted"
    return f"{rn}_{mode}_distance.npz"


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate nonlinear embeddings from GraphML.bz2.",
    )
    p.add_argument(
        "graphml_file",
        type=str,
        help="Input GraphML.bz2 file",
    )
    p.add_argument(
        "-w",
        "--weight",
        dest="weight",
        type=str,
        default="weight",
        help="Graph edge weight key",
    )
    p.add_argument(
        "--no-weights",
        dest="weight",
        action="store_const",
        const=None,
        help="Disable weighted distances",
    )
    p.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of threads for distance computation",
    )
    p.add_argument(
        "--distance-npz",
        type=str,
        default=None,
        help="Optional precomputed distance NPZ cache",
    )
    p.add_argument(
        "--save-distance-npz",
        type=str,
        nargs="?",
        const="__AUTO__",
        default=None,
        help="Output path for distance NPZ cache (default: <RN>_<weighted|unweighted>_distance.npz in launch directory)",
    )
    p.add_argument(
        "-c",
        "--components",
        type=int,
        default=2,
        help="Embedding dimensions",
    )
    p.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["mMDS", "PCA", "KPCA", "TSNE", "ISOMAP", "UMAP", "SE"],
        required=True,
        help="Embedding method",
    )
    p.add_argument(
        "--preprocess-pre",
        dest="preprocess",
        action="store_const",
        const="pre",
    )
    p.add_argument(
        "--preprocess-pdc",
        dest="preprocess",
        action="store_const",
        const="pdc",
    )
    p.add_argument(
        "--preprocess-dcs",
        dest="preprocess",
        action="store_const",
        const="dcs",
    )
    p.add_argument(
        "--preprocess-sym",
        dest="preprocess",
        action="store_const",
        const="sym",
    )
    p.add_argument(
        "--kpca-kernel",
        dest="kpca_kernel",
        type=str,
        default="precomputed",
    )
    p.add_argument(
        "--kpca-gamma",
        dest="kpca_gamma",
        type=float,
        default=None,
    )
    p.add_argument(
        "--kpca-degree",
        dest="kpca_degree",
        type=float,
        default=None,
    )
    p.add_argument(
        "--kpca-coef0",
        dest="kpca_coef0",
        type=float,
        default=None,
    )
    p.add_argument(
        "--tsne-perplexity",
        dest="tsne_perplexity",
        type=float,
        default=None,
    )
    p.add_argument(
        "--tsne-early-exaggeration",
        dest="tsne_early_exaggeration",
        type=float,
        default=None,
    )
    p.add_argument(
        "--tsne-learning-rate",
        dest="tsne_learning_rate",
        type=float,
        default=None,
    )
    p.add_argument(
        "--tsne-max-iter",
        dest="tsne_max_iter",
        type=int,
        default=None,
    )
    p.add_argument(
        "--tsne-exact",
        dest="tsne_exact",
        action="store_true",
    )
    p.add_argument(
        "--isomap-neighbors",
        dest="isomap_neighbors",
        type=int,
        default=None,
    )
    p.add_argument(
        "--isomap-radius",
        dest="isomap_radius",
        type=float,
        default=None,
    )
    p.add_argument(
        "--umap-neighbors",
        dest="umap_neighbors",
        type=int,
        default=None,
    )
    p.add_argument(
        "--umap-min-dist",
        dest="umap_min_dist",
        type=float,
        default=None,
    )
    p.add_argument(
        "--umap-spread",
        dest="umap_spread",
        type=float,
        default=None,
    )
    p.add_argument(
        "--umap-target-metric",
        dest="umap_target_metric",
        type=str,
        default=None,
    )
    p.add_argument(
        "--se-neighbors",
        dest="se_neighbors",
        type=int,
        default=None,
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default=None,
    )
    p.add_argument(
        "--no-save-npz",
        dest="save_npz",
        action="store_false",
    )
    p.add_argument(
        "--no-save-csv",
        dest="save_csv",
        action="store_false",
    )
    p.set_defaults(save_npz=True, save_csv=True)
    args = p.parse_args()

    if args.components <= 0:
        p.error("--components must be > 0")
    if args.preprocess is None:
        args.preprocess = "none"
    return args


def run_method(args, D_mod):
    if args.method == "mMDS":
        return mmds(D_mod, args.components)
    if args.method == "PCA":
        return pca_embed(D_mod, args.components)
    if args.method == "KPCA":
        return kpca_embed(
            D_mod,
            args.components,
            args.kpca_kernel,
            args.kpca_gamma,
            args.kpca_degree,
            args.kpca_coef0,
        )
    if args.method == "TSNE":
        return tsne_embed(
            D_mod,
            args.components,
            args.tsne_perplexity,
            args.tsne_early_exaggeration,
            args.tsne_learning_rate,
            args.tsne_max_iter,
            args.tsne_exact,
        )
    if args.method == "ISOMAP":
        return isomap_embed(
            D_mod,
            args.components,
            args.isomap_neighbors,
            args.isomap_radius,
        )
    if args.method == "UMAP":
        return umap_embed(
            D_mod,
            args.components,
            args.umap_neighbors,
            args.umap_min_dist,
            args.umap_spread,
            args.umap_target_metric,
        )
    return se_embed(D_mod, args.components, args.se_neighbors)


def main():
    args = parse_args()

    if args.distance_npz:
        D_in, node_ids = load_distance_matrix_npz(args.distance_npz)
        print(f"Loaded distance cache: {args.distance_npz}")
    else:
        D_in, node_ids = compute_distance_matrix_from_graphml(
            args.graphml_file,
            args.weight,
            args.cores,
        )
        if args.save_distance_npz:
            cache_path = (
                default_distance_cache_path(args.graphml_file, args.weight)
                if args.save_distance_npz == "__AUTO__"
                else args.save_distance_npz
            )
            save_distance_matrix_npz(cache_path, D_in, node_ids)

    if args.preprocess == "pre":
        D_mod = preprocess_matrix(D_in, node_ids)
    elif args.preprocess == "pdc":
        D_mod = double_centering(D_in)
    elif args.preprocess == "dcs":
        D_mod = double_centering_symmetrization(D_in)
    elif args.preprocess == "sym":
        D_mod = symmetrize(D_in)
    else:
        D_mod = D_in.copy()

    Y = run_method(args, D_mod)

    base = args.output_prefix or os.path.splitext(os.path.basename(args.graphml_file))[0].replace(".graphml", "")
    base = base.replace("_weighted", "").replace("_unweighted", "")
    preprocess_suffix = f"_{args.preprocess}" if args.preprocess != "none" else ""
    weight_suffix = "unweighted" if args.weight is None else "weighted"
    out = f"{base}_{args.method}{preprocess_suffix}_c{str(args.components).zfill(3)}_{weight_suffix}"

    if args.save_npz:
        np.savez_compressed(
            out + ".npz",
            coords=Y,
            node_ids=np.array(node_ids),
            method=np.array([args.method]),
            components=np.array([args.components]),
        )
        print(f"Saved {out}.npz")

    if args.save_csv:
        labels = get_nodes_graphml(args.graphml_file, node_ids)
        combine_files(Y, labels, out + ".csv")
        print(f"Saved {out}.csv")


if __name__ == "__main__":
    main()
