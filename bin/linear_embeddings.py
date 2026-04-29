#!/usr/bin/env python3

"""
Linear embedding pipeline (embedding only).

Supported methods:
  - cMDS
  - LMDS

Authors:
    Miko Stulajter

Version 2.0.0

Usage: linear_embeddings.py <graphml_file> [--distance-npz DISTANCE_NPZ] [--save-distance-npz [SAVE_DISTANCE_NPZ]] [options]
"""

import argparse
import bz2
import os
import xml.etree.ElementTree as ET

import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

def preprocess_matrix(D, node_ids):
    n = len(node_ids)
    H = np.eye(n) - np.ones((n, n)) / n
    A = -0.5 * (D ** 2)
    return H @ A @ H

def symmetrize(D):
    return (D + D.T) / 2

def double_centering(D):
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean

def double_centering_symmetrization(D):
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    D_centered = D - row_mean - col_mean + grand_mean
    return (D_centered + D_centered.T) / 2


def double_center_lmds(dist_matrix):
    dist_sq = dist_matrix**2
    row_means = dist_sq.mean(axis=1, keepdims=True)
    col_means = dist_sq.mean(axis=0, keepdims=True)
    total_mean = dist_sq.mean()
    return -0.5 * (dist_sq - row_means - col_means + total_mean)


def double_center_cross(landmark_dist, cross_dist):
    landmark_sq = landmark_dist**2
    cross_sq = cross_dist**2
    col_means = cross_sq.mean(axis=0, keepdims=True)
    row_means = landmark_sq.mean(axis=1, keepdims=True)
    return -0.5 * (cross_sq - col_means - row_means)


def select_landmarks(dist_matrix, num_landmarks, method='random', seed=None):
    n_nodes = dist_matrix.shape[0]
    all_indices = np.arange(n_nodes)
    
    if seed is not None:
        np.random.seed(seed)
    
    if method == 'random':
        landmark_indices = np.random.choice(all_indices, size=num_landmarks, replace=False)
    else:  # maxmin
        landmark_indices = [np.random.randint(0, n_nodes)]
        for _ in range(num_landmarks - 1):
            min_dists = np.min(dist_matrix[:, landmark_indices], axis=1)
            landmark_indices.append(np.argmax(min_dists))
        landmark_indices = np.array(landmark_indices)
    
    non_landmark_indices = np.setdiff1d(all_indices, landmark_indices)
    return landmark_indices, non_landmark_indices


def compute_embedding_coordinates(landmark_kernel_matrix, landmark_to_non_landmark_dist, landmark_dist_matrix, n_components):
    eigenvalues, eigenvectors = np.linalg.eigh(landmark_kernel_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    
    # Limit n_components to the number of available eigenvectors/positive eigenvalues
    max_components = min(n_components, len(eigenvalues))
    positive_eigenvals = eigenvalues[sorted_idx] > 1e-10
    num_positive = np.sum(positive_eigenvals)
    actual_components = min(max_components, num_positive) if num_positive > 0 else max_components
    
    eigenvalues = eigenvalues[sorted_idx][:actual_components]
    eigenvectors = eigenvectors[:, sorted_idx][:, :actual_components]
    sqrt_eigenvals = np.sqrt(np.maximum(eigenvalues, 0))
    
    landmark_coords = eigenvectors @ np.diag(sqrt_eigenvals)
    
    if landmark_to_non_landmark_dist.size > 0:
        B = double_center_cross(landmark_dist_matrix, landmark_to_non_landmark_dist)
        inv_sqrt = np.where(sqrt_eigenvals > 1e-10, 1.0 / sqrt_eigenvals, 0.0)
        non_landmark_coords = B.T @ eigenvectors @ np.diag(inv_sqrt)
    else:
        non_landmark_coords = np.array([]).reshape(0, actual_components)
    
    return landmark_coords, non_landmark_coords


def lmds(matrix, components, landmark_ratio=0.2, landmark_selection='random', seed=None):
    n_nodes = matrix.shape[0]
    dist_np = np.array(matrix)
    
    # Select landmarks
    num_landmarks = max(int(np.round(n_nodes * landmark_ratio)), min(components + 1, n_nodes))
    num_landmarks = min(num_landmarks, n_nodes)
    
    if num_landmarks >= n_nodes:
        landmark_indices = np.arange(n_nodes)
        non_landmark_indices = np.array([], dtype=np.int64)
    else:
        landmark_indices, non_landmark_indices = select_landmarks(
            dist_np, num_landmarks, landmark_selection, seed
        )
    
    # Compute distance matrices
    landmark_dist = dist_np[np.ix_(landmark_indices, landmark_indices)]
    landmark_kernel = double_center_lmds(landmark_dist)
    
    if len(non_landmark_indices) > 0:
        landmark_to_non_landmark_dist = dist_np[np.ix_(landmark_indices, non_landmark_indices)]
    else:
        landmark_to_non_landmark_dist = np.array([]).reshape(len(landmark_indices), 0)
    
    # Compute embeddings
    landmark_coords, non_landmark_coords = compute_embedding_coordinates(
        landmark_kernel, landmark_to_non_landmark_dist, landmark_dist, components
    )
    
    # Get actual number of components returned
    actual_components = landmark_coords.shape[1]
    
    # Combine in correct order
    all_embeddings = np.zeros((n_nodes, components))
    if actual_components <= components:
        all_embeddings[landmark_indices, :actual_components] = landmark_coords
        if len(non_landmark_indices) > 0:
            all_embeddings[non_landmark_indices, :actual_components] = non_landmark_coords
    else:
        all_embeddings[landmark_indices] = landmark_coords[:, :components]
        if len(non_landmark_indices) > 0:
            all_embeddings[non_landmark_indices] = non_landmark_coords[:, :components]
    
    return all_embeddings


def cmds(matrix, components):
    arr = np.asarray(matrix, dtype=np.float64, order='C')
    n = arr.shape[0]
    k = min(components, n)
    k_actual = min(k, n - 1) if n > 1 else 1
    eig_vals, eig_vecs = eigsh(arr, k=k_actual, which='LA', maxiter=5000)
    eig_vals = np.maximum(eig_vals, 0.0)
    eig_vecs = eig_vecs[:, ::-1]
    eig_vals = eig_vals[::-1]
    sqrt_vals = np.sqrt(np.maximum(eig_vals, 0.0))
    return np.real(eig_vecs * sqrt_vals)


def combine_files(coord_matrix, nodes, out_file):
    if len(coord_matrix) != len(nodes):
        raise Exception('The number of nodes does not match the number of coords')
    formatted_coords_file = [":".join(map(str, coords)) for coords in coord_matrix]
    combined_data = pd.DataFrame({
        'node': nodes,
        'coordinates': formatted_coords_file
    })
    combined_data.to_csv(out_file, index=False)
    return combined_data


def get_nodes_graphml(args, node_ids):
    with bz2.open(args.original_bz2, 'rb') as f:
        root = ET.fromstring(f.read())
    graph = root.find('{http://graphml.graphdrawing.org/xmlns}graph')

    # one pass: build dict
    lookup = {node.attrib['id']: node.find('{http://graphml.graphdrawing.org/xmlns}data').text
              for node in graph.findall('{http://graphml.graphdrawing.org/xmlns}node')}

    # directly return in node_ids order
    return [lookup[nid] for nid in node_ids]


def process_task(args, D_mod, method):
    try:
        if method == 'cMDS':
            Y = cmds(D_mod, args.components)
        elif method == 'LMDS':
            landmark_ratio = args.lmds_ratio
            landmark_selection = args.lmds_selection
            seed = args.lmds_seed
            method += f"_R-{landmark_ratio}_S-{landmark_selection}"
            Y = lmds(D_mod, args.components, landmark_ratio, landmark_selection, seed)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return Y, method
    except Exception as e:
        print(f'Error: {e}')
        return None, None


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate linear embeddings from GraphML.bz2."
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
        help="Optional precomputed distance NPZ with 'og_matrix' and 'node_ids'",
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
        choices=["cMDS", "LMDS"],
        required=True,
        help="Embedding method. Supported: cMDS, LMDS",
    )
    p.add_argument(
        "--preprocess-pre",
        dest="preprocess",
        action="store_const",
        const="pre",
        help="Use cMDS preprocessing matrix (H * -0.5D^2 * H)",
    )
    p.add_argument(
        "--preprocess-pdc",
        dest="preprocess",
        action="store_const",
        const="pdc",
        help="Use double-centering",
    )
    p.add_argument(
        "--preprocess-dcs",
        dest="preprocess",
        action="store_const",
        const="dcs",
        help="Use double-centering + symmetrization",
    )
    p.add_argument(
        "--preprocess-sym",
        dest="preprocess",
        action="store_const",
        const="sym",
        help="Use symmetrization",
    )
    p.add_argument(
        "--lmds-ratio",
        dest="lmds_ratio",
        type=float,
        default=0.3,
        help="Landmark ratio for LMDS",
    )
    p.add_argument(
        "--lmds-selection",
        dest="lmds_selection",
        type=str,
        choices=["random", "maxmin"],
        default="random",
        help="Landmark selection strategy for LMDS",
    )
    p.add_argument(
        "--lmds-seed",
        dest="lmds_seed",
        type=int,
        default=None,
        help="Optional LMDS random seed",
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
        help="Disable NPZ output",
    )
    p.add_argument(
        "--no-save-csv",
        dest="save_csv",
        action="store_false",
        help="Disable CSV output",
    )
    p.set_defaults(save_npz=True, save_csv=True)
    args = p.parse_args()

    if args.components <= 0:
        p.error("--components must be > 0")

    if args.preprocess is None:
        args.preprocess = "none"

    return args


def sort_node_ids_and_matrix(node_ids, D):
    try:
        node_ids_sorted = sorted(node_ids, key=lambda x: float(x))
    except (ValueError, TypeError):
        node_ids_sorted = sorted(node_ids)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    reorder_idx = [id_to_idx[nid] for nid in node_ids_sorted]
    D_reordered = np.asarray(D)[np.ix_(reorder_idx, reorder_idx)]
    return node_ids_sorted, D_reordered


def sort_node_ids(node_ids):
    try:
        return sorted(node_ids, key=lambda x: float(x))
    except (ValueError, TypeError):
        return sorted(node_ids)


def read_graphml_bz2_nx(graphml_file):
    with bz2.open(graphml_file, "rb") as f:
        return nx.read_graphml(f)


def compute_distance_matrix_from_graphml(graphml_file, weight_key="weight", cores=1):
    nk.setNumberOfThreads(int(cores))
    graph_nx = read_graphml_bz2_nx(graphml_file)
    sorted_nodes = sort_node_ids(list(graph_nx.nodes()))

    sub = graph_nx.subgraph(sorted_nodes)
    ordered_graph = nx.Graph()
    ordered_graph.add_nodes_from(sorted_nodes)
    ordered_graph.add_edges_from(sub.edges(data=True))
    graph_nk = nk.nxadapter.nx2nk(ordered_graph, weightAttr=weight_key)

    all_nodes = list(range(graph_nk.numberOfNodes()))
    spsp = nk.distance.SPSP(graph_nk, all_nodes).run()
    n = len(all_nodes)
    distance_matrix = np.zeros((n, n), dtype=np.float64)
    for i, source in enumerate(all_nodes):
        for j, target in enumerate(all_nodes[i + 1 :], start=i + 1):
            dist = spsp.getDistance(source, target)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix, [str(x) for x in sorted_nodes]


def load_distance_matrix_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        if "og_matrix" not in data:
            raise ValueError(f"'og_matrix' not found in {npz_path}")
        if "node_ids" not in data:
            raise ValueError(f"'node_ids' not found in {npz_path}")
        distance_matrix = np.asarray(data["og_matrix"])
        node_ids = [str(x) for x in data["node_ids"]]
    return sort_node_ids_and_matrix(node_ids, distance_matrix)


def save_distance_matrix_npz(npz_path, distance_matrix, node_ids):
    np.savez_compressed(
        npz_path,
        og_matrix=np.asarray(distance_matrix),
        node_ids=np.array(node_ids),
    )
    print(f"Saved distance cache: {npz_path}")


def default_distance_cache_path(graphml_file, weight_key):
    rn = os.path.basename(graphml_file).replace(".graphml.bz2", "").replace(".graphml", "")
    mode = "unweighted" if weight_key is None else "weighted"
    return f"{rn}_{mode}_distance.npz"


def main():
    args = parse_args()
    if args.distance_npz:
        D_in, node_ids = load_distance_matrix_npz(args.distance_npz)
        print(f"Loaded distance cache: {args.distance_npz}")
    else:
        D_in, node_ids = compute_distance_matrix_from_graphml(
            args.graphml_file,
            weight_key=args.weight,
            cores=args.cores,
        )
        if args.save_distance_npz:
            cache_path = (
                default_distance_cache_path(args.graphml_file, args.weight)
                if args.save_distance_npz == "__AUTO__"
                else args.save_distance_npz
            )
            save_distance_matrix_npz(cache_path, D_in, node_ids)

    D_base = D_in

    method = args.method
    
    if method == "cMDS" and args.preprocess == "none":
        args.preprocess = "pre"

    if args.preprocess == "pre":
        D_mod = preprocess_matrix(D_base, node_ids)
    elif args.preprocess == "pdc":
        D_mod = double_centering(D_base)
    elif args.preprocess == "dcs":
        D_mod = double_centering_symmetrization(D_base)
    elif args.preprocess == "sym":
        D_mod = symmetrize(D_base)    
    else:
        D_mod = D_base.copy()

    Y, postfix = process_task(args, D_mod, method)
    
    if Y is None or postfix is None:
        raise SystemExit(f"Error: Embedding failed for method {method}")
    
    base = args.output_prefix or os.path.splitext(os.path.basename(args.graphml_file))[0].replace(".graphml", "")
    
    base = base.replace("_weighted", "").replace("_unweighted", "")
    preprocess_suffix = f"_{args.preprocess}" if args.preprocess != "none" else ""
    weight_suffix = "unweighted" if args.weight is None else "weighted"
    out = f"{base}_{postfix}{preprocess_suffix}_c{str(args.components).zfill(3)}_{weight_suffix}"

    if args.save_npz:
        np.savez_compressed(
            out + ".npz",
            coords=Y,
            node_ids=np.array(node_ids),
            method=np.array([method]),
            components=np.array([args.components]),
        )
        print(f"Saved {out}.npz")

    if args.save_csv:
        graphml_args = argparse.Namespace(original_bz2=args.graphml_file)
        labels = get_nodes_graphml(graphml_args, node_ids)
        combine_files(Y, labels, out + ".csv")
        print(f"Saved {out}.csv")

if __name__ == "__main__":
    main()


