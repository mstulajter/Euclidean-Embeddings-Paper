#!/usr/bin/env python3

"""
Path-based embedding quality metrics.

Computes Stress, R², Pearson, Spearman, and Kendall tau by percentile
bins over graph distances, comparing graph distances to embedding distances.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: error_metrics.py <graphml_file> [--csv CSV | --embedding-npz EMBEDDING_NPZ] [--npz NPZ] [--error-metrics {first,second,all}] [--output OUTPUT]
"""

import argparse
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import canberra
import bz2
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.metrics import pairwise_distances
import networkit as nk


def calculate_stress(original_distances, embedded_distances, eps=1e-12):
    """Calculate normalized stress between original and embedded distances."""
    mask = original_distances > 0
    if not mask.any():
        return 0.0
    num = np.sum((original_distances[mask] - embedded_distances[mask]) ** 2)
    denom = np.sum(np.maximum(original_distances[mask], eps) ** 2)
    return np.sqrt(num / denom)


def calculate_r2(original_distances, embedded_distances):
    """Calculate R-squared (coefficient of determination)."""
    mask = original_distances > 0
    if not mask.any():
        return 0.0
    
    x = original_distances[mask]
    y = embedded_distances[mask]
    
    ss_res = np.sum((x - y) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_correlations(original_distances, embedded_distances):
    """Calculate correlation metrics."""
    mask = original_distances > 0
    if not mask.any():
        return 0.0, 0.0, 0.0
    
    x = original_distances[mask]
    y = embedded_distances[mask]
    
    pearson_corr, _ = pearsonr(x, y)
    spearman_corr, _ = spearmanr(x, y)
    kendall_tau, _ = kendalltau(x, y)
    
    return pearson_corr, spearman_corr, kendall_tau


def calculate_l1_l2_mae_mse(original_distances, embedded_distances):
    """Calculate L1, L2, MAE, and MSE on distances where original > 0."""
    mask = original_distances > 0
    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0
    diff = original_distances[mask] - embedded_distances[mask]
    l1_norm = float(np.sum(np.abs(diff)))
    l2_norm = float(np.sqrt(np.sum(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    return l1_norm, l2_norm, mae, mse


def calculate_nmse(original_distances, embedded_distances, eps=1e-12):
    """Normalized MSE using original distances as denominator."""
    mask = original_distances > 0
    if not mask.any():
        return 0.0
    x = original_distances[mask]
    y = embedded_distances[mask]
    denom = np.maximum(x, eps)
    return float(np.mean(((x - y) / denom) ** 2))


def calculate_mape(original_distances, embedded_distances, eps=1e-12):
    """MAPE using original distances as denominator."""
    mask = original_distances > 0
    if not mask.any():
        return 0.0
    x = original_distances[mask]
    y = embedded_distances[mask]
    denom = np.maximum(x, eps)
    return float(np.mean(np.abs((x - y) / denom)))


def calculate_canberra_metrics(original_distances, embedded_distances):
    """Total and mean Canberra distance."""
    if original_distances.size == 0:
        return 0.0, 0.0
    total = float(canberra(original_distances, embedded_distances))
    return total, total / float(original_distances.size)


def _r2_identity_one(x, y):
    if x.size < 2:
        return np.nan
    ss_res = np.sum((x - y) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)
    return 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)


def _r2_pearson_one(x, y):
    if x.size < 2:
        return np.nan
    r, _ = pearsonr(x, y)
    return np.nan if np.isnan(r) else float(r ** 2)


def _r2_fit_one(x, y):
    if x.size < 2:
        return np.nan
    A = np.column_stack([np.ones_like(x), x])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)


def compute_quartile_metrics(x, y):
    """Compute quartile and cumulative metrics using quartiles from x."""
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    per_bins = [
        x < q25,
        (x >= q25) & (x < q50),
        (x >= q50) & (x < q75),
        x >= q75,
    ]
    cum_bins = [x < q25, x < q50, x < q75, np.ones(x.size, dtype=bool)]

    r2_identity_q = tuple(_r2_identity_one(x[b], y[b]) for b in per_bins)
    r2_identity_c = tuple(_r2_identity_one(x[b], y[b]) for b in cum_bins)
    r2_pearson_q = tuple(_r2_pearson_one(x[b], y[b]) for b in per_bins)
    r2_pearson_c = tuple(_r2_pearson_one(x[b], y[b]) for b in cum_bins)
    r2_fit_q = tuple(_r2_fit_one(x[b], y[b]) for b in per_bins)
    r2_fit_c = tuple(_r2_fit_one(x[b], y[b]) for b in cum_bins)

    spearman_q = []
    kendall_q = []
    for b in per_bins:
        if np.sum(b) < 2:
            spearman_q.append(np.nan)
            kendall_q.append(np.nan)
            continue
        _, sp, kd = calculate_correlations(x[b], y[b])
        spearman_q.append(sp)
        kendall_q.append(kd)

    return {
        "r2_identity_overall": _r2_identity_one(x, y),
        "r2_identity_q": r2_identity_q,
        "r2_identity_c": r2_identity_c,
        "r2_pearson_overall": _r2_pearson_one(x, y),
        "r2_pearson_q": r2_pearson_q,
        "r2_pearson_c": r2_pearson_c,
        "r2_fit_overall": _r2_fit_one(x, y),
        "r2_fit_q": r2_fit_q,
        "r2_fit_c": r2_fit_c,
        "spearman_q": tuple(spearman_q),
        "kendall_q": tuple(kendall_q),
    }


def sort_node_ids(node_ids):
    """Sort node IDs so matrix row/column i maps to node_ids[i]."""
    try:
        return sorted(node_ids, key=lambda x: float(x))
    except (ValueError, TypeError):
        return sorted(node_ids)


def load_smiles_map_from_graphml(graphml_file):
    """Extract SMILES mapping from GraphML file.
    
    Returns:
        Dictionary mapping node_id to SMILES string
    """
    smiles_map = {}
    with bz2.open(graphml_file, 'rb') as f:
        root = ET.fromstring(f.read())
    graph = root.find('{http://graphml.graphdrawing.org/xmlns}graph')
    
    for node in graph.findall('{http://graphml.graphdrawing.org/xmlns}node'):
        node_id = node.attrib['id']
        data_elem = node.find('{http://graphml.graphdrawing.org/xmlns}data')
        if data_elem is not None and data_elem.text:
            smiles_map[node_id] = data_elem.text
    
    return smiles_map


def load_embeddings_by_node_id(csv_file, node_id_to_smiles=None, target_node_ids=None):
    """Load embeddings from CSV file.
    
    Args:
        csv_file: Path to CSV file
        node_id_to_smiles: Optional mapping from node_id to SMILES (if CSV uses SMILES)
        target_node_ids: Optional list of target node_ids to match embeddings to (e.g., NPZ node_ids)
    
    Returns:
        Dictionary mapping node_id to embedding coordinates
    """
    pos = {}
    df = pd.read_csv(csv_file)
    if 'node' in df.columns and 'coordinates' in df.columns:
        # Create reverse mapping: SMILES -> node_id
        # If target_node_ids provided, prioritize mapping to those node_ids
        smiles_to_node_id = {}
        if node_id_to_smiles:
            if target_node_ids:
                # Create set for fast lookup
                target_node_ids_set = set(str(nid) for nid in target_node_ids)
                # Map SMILES to target node_ids (NPZ node_ids) if they exist
                for node_id in target_node_ids:
                    node_id_str = str(node_id)
                    smiles = node_id_to_smiles.get(node_id_str)
                    if smiles:
                        # If multiple node_ids map to same SMILES, keep the one in target_node_ids
                        if smiles not in smiles_to_node_id or smiles_to_node_id[smiles] not in target_node_ids_set:
                            smiles_to_node_id[smiles] = node_id_str
            else:
                # No target_node_ids, use all mappings from node_id_to_smiles
                smiles_to_node_id = {smiles: node_id for node_id, smiles in node_id_to_smiles.items()}
        
        for _, row in df.iterrows():
            node_key = str(row['node'])
            coords_str = row['coordinates']
            if pd.isna(coords_str):
                continue

            if isinstance(coords_str, str):
                raw_parts = [part.strip() for part in coords_str.split(':') if part.strip()]
                if not raw_parts:
                    continue
                coords = np.array([float(x) for x in raw_parts], dtype=np.float64)
            else:
                coords = np.array([float(coords_str)], dtype=np.float64)

            # If CSV has SMILES and we have mapping, convert to node_id
            if smiles_to_node_id and node_key in smiles_to_node_id:
                node_id = smiles_to_node_id[node_key]
                pos[node_id] = coords
            else:
                # Assume CSV node is already the node_id
                pos[node_key] = coords
    else:
        raise ValueError("CSV must have 'node' and 'coordinates' columns")
    return pos


def load_distance_matrix(npz_file):
    """Load distance matrix from NPZ file.
    
    Returns:
        Tuple of (distance_matrix, node_ids)
    """
    with np.load(npz_file, allow_pickle=True) as data:
        if 'og_matrix' in data:
            distance_matrix = data['og_matrix']
            if 'node_ids' in data:
                node_ids = [str(x) for x in data['node_ids']]
            else:
                node_ids = None
            return distance_matrix, node_ids
        raise ValueError("'og_matrix' not found in npz file")


def load_embedding_coords_from_npz(npz_file, target_node_ids):
    """Load embedding coords from NPZ and align rows to target_node_ids order."""
    with np.load(npz_file, allow_pickle=True) as data:
        if "coords" not in data:
            raise ValueError(f"'coords' not found in embedding npz: {npz_file}")
        if "node_ids" not in data:
            raise ValueError(f"'node_ids' not found in embedding npz: {npz_file}")
        coords = np.asarray(data["coords"])
        emb_node_ids = [str(x) for x in data["node_ids"]]

    if len(emb_node_ids) != coords.shape[0]:
        raise ValueError("Embedding NPZ node_ids length does not match coords rows")
    emb_idx = {nid: i for i, nid in enumerate(emb_node_ids)}
    missing = [nid for nid in target_node_ids if nid not in emb_idx]
    if missing:
        raise ValueError(f"Distance matrix node_ids missing in embedding NPZ: {len(missing)} missing")
    return coords[[emb_idx[nid] for nid in target_node_ids]]


def compute_distance_matrix_from_graphml(graphml_file, weight_key='weight', cores=1):
    """Compute distance matrix from GraphML using NetworkKit.
    Node order = sort_node_ids (same as Embedding_pipeline); row/col i = node_ids[i].
    Builds an ordered graph so nx2nk assigns index i to sorted_nodes[i].
    """
    nk.setNumberOfThreads(int(cores))

    with bz2.open(graphml_file, 'rb') as f:
        graph_nx = nx.read_graphml(f)

    sorted_nodes = sort_node_ids(list(graph_nx.nodes()))
    sub = graph_nx.subgraph(sorted_nodes)
    G_ordered = nx.Graph()
    G_ordered.add_nodes_from(sorted_nodes)
    G_ordered.add_edges_from(sub.edges(data=True))
    G_nk = nk.nxadapter.nx2nk(G_ordered, weightAttr=weight_key)

    all_nodes = list(range(G_nk.numberOfNodes()))
    spsp = nk.distance.SPSP(G_nk, all_nodes).run()
    num_nodes = len(all_nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i, source in enumerate(all_nodes):
        for j, target in enumerate(all_nodes[i+1:], start=i+1):
            path_length = spsp.getDistance(source, target)
            distance_matrix[i, j] = path_length
            distance_matrix[j, i] = path_length

    return distance_matrix, sorted_nodes


def build_parser():
    parser = argparse.ArgumentParser(description="Calculate path metrics for an embedding.")
    parser.add_argument(
        "graphml_file",
        type=str,
        help="Path to GraphML.bz2 file",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        help="Embedding CSV with 'node' and 'coordinates' columns",
    )
    parser.add_argument(
        "--embedding-npz",
        type=str,
        required=False,
        help="Embedding NPZ with required keys: 'coords' and 'node_ids'",
    )
    parser.add_argument(
        "--npz",
        type=str,
        required=False,
        help="Optional NPZ with 'og_matrix' and 'node_ids'; computed from GraphML when omitted",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV path (default: derived from inputs)",
    )
    parser.add_argument(
        "--error-metrics",
        type=str,
        choices=["first", "second", "all"],
        default="all",
        help="Which metric set to output: first (path percentile), second (full aggregate), or all",
    )
    parser.add_argument(
        "--print-terminal",
        action="store_true",
        help="Print progress/metrics to terminal (default: write to output file only)",
    )
    parser.add_argument(
        "-w",
        "--weight",
        dest="weight",
        nargs="?",
        const="weight",
        default="weight",
        type=str,
        help="Graph edge weight key",
    )
    parser.add_argument(
        "--no-weights",
        dest="weight",
        action="store_const",
        const=None,
        help="Disable weighted graph distances",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores for distance matrix computation",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.csv and not args.embedding_npz:
        parser.error("One of --csv or --embedding-npz is required")
    if args.csv and args.embedding_npz:
        parser.error("Use only one embedding source: --csv or --embedding-npz")

    include_first = args.error_metrics in ("first", "all")
    include_second = args.error_metrics in ("second", "all")
    log = print if args.print_terminal else (lambda *a, **k: None)

    weight_key = args.weight
    
    # Load or compute distance matrix
    if args.npz:
        log("Loading distance matrix from NPZ...", flush=True)
        distance_matrix, npz_node_ids = load_distance_matrix(args.npz)
        log(f"Distance matrix shape: {distance_matrix.shape}", flush=True)
        if npz_node_ids:
            log(f"NPZ node_ids: {len(npz_node_ids)} nodes", flush=True)
    else:
        log("NPZ not provided, computing distance matrix from GraphML...", flush=True)
        distance_matrix, npz_node_ids = compute_distance_matrix_from_graphml(
            args.graphml_file, weight_key, args.cores
        )
        log(f"Computed distance matrix shape: {distance_matrix.shape}", flush=True)
        log(f"Node IDs: {len(npz_node_ids)} nodes", flush=True)
    
    log("Loading graph...", flush=True)
    with bz2.open(args.graphml_file, mode="rt") as gfile:
        G = nx.parse_graphml(gfile.read())
    
    log(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    
    # Get SMILES mapping from GraphML
    log("Extracting SMILES mapping from GraphML...", flush=True)
    node_id_to_smiles = load_smiles_map_from_graphml(args.graphml_file)
    log(f"Found {len(node_id_to_smiles)} nodes with SMILES", flush=True)
    
    # Get node IDs - prefer NPZ node_ids if available, otherwise use graph nodes
    if npz_node_ids:
        node_ids = [str(nid) for nid in npz_node_ids]
        log(f"Using node_ids from NPZ: {len(node_ids)} nodes", flush=True)
        # Distance matrix is already in NPZ node_ids order, so we use that order
    else:
        node_ids = [str(n) for n in G.nodes()]
        log(f"Using node_ids from graph: {len(node_ids)} nodes", flush=True)
    
    # Ensure distance matrix matches node_ids
    if len(distance_matrix) != len(node_ids):
        log(f"Warning: Distance matrix size ({len(distance_matrix)}) doesn't match node_ids ({len(node_ids)})", flush=True)
        log("Using distance matrix size for node_ids", flush=True)
        node_ids = [str(i) for i in range(len(distance_matrix))]
    
    if args.embedding_npz:
        log("Loading embeddings from NPZ...", flush=True)
        embedding_coords = load_embedding_coords_from_npz(args.embedding_npz, node_ids)
        log(f"Embedding shape: {embedding_coords.shape}", flush=True)
    else:
        log("Loading embeddings from CSV...", flush=True)
        pos = load_embeddings_by_node_id(args.csv, node_id_to_smiles, node_ids)
        log(f"Total embeddings loaded: {len(pos)} nodes", flush=True)
        if not pos:
            raise ValueError(
                "No embeddings loaded from CSV. Verify that CSV 'node' values match graph node_ids or GraphML smiles."
            )

        log("Aligning embeddings to distance matrix node order...", flush=True)
        embedding_coords_ordered = []
        missing_count = 0
        for node_id in node_ids:
            if node_id in pos:
                embedding_coords_ordered.append(pos[node_id])
            else:
                smiles = node_id_to_smiles.get(node_id)
                if smiles and smiles in pos:
                    embedding_coords_ordered.append(pos[smiles])
                else:
                    missing_count += 1
                    embedding_coords_ordered.append(None)

        if missing_count > 0:
            log(f"Warning: {missing_count} nodes missing embeddings", flush=True)

        if all(coord is not None for coord in embedding_coords_ordered):
            embedding_coords = np.array(embedding_coords_ordered)
        else:
            dim = next(len(coord) for coord in embedding_coords_ordered if coord is not None)
            embedding_coords = np.full((len(node_ids), dim), np.nan)
            for i, coord in enumerate(embedding_coords_ordered):
                if coord is not None:
                    embedding_coords[i] = coord
    
    log("Computing embedding distance matrix...", flush=True)
    if np.any(np.isnan(embedding_coords)):
        # Only compute distances for valid (non-NaN) embeddings
        valid_mask = ~np.isnan(embedding_coords).any(axis=1)
        valid_indices = np.where(valid_mask)[0]
        valid_embedding_coords = embedding_coords[valid_mask]
        embedding_dist_matrix_valid = pairwise_distances(valid_embedding_coords)
        
        # Create full distance matrix with NaN for invalid pairs
        embedding_dist_matrix = np.full((len(node_ids), len(node_ids)), np.nan)
        for idx_i, i in enumerate(valid_indices):
            for idx_j, j in enumerate(valid_indices):
                if i <= j:  # Only fill upper triangle
                    embedding_dist_matrix[i, j] = embedding_dist_matrix_valid[idx_i, idx_j]
                    embedding_dist_matrix[j, i] = embedding_dist_matrix_valid[idx_i, idx_j]
    else:
        embedding_dist_matrix = pairwise_distances(embedding_coords)
    
    # Extract upper triangle for comparison (avoid double counting)
    triu_i, triu_j = np.triu_indices_from(distance_matrix, k=1)
    graph_distances = distance_matrix[triu_i, triu_j]
    embedding_distances = embedding_dist_matrix[triu_i, triu_j]
    
    # Filter out invalid distances (NaN, inf, or zero)
    valid_mask = (
        np.isfinite(graph_distances) & 
        np.isfinite(embedding_distances) & 
        (graph_distances > 0) & 
        (embedding_distances > 0)
    )
    
    graph_distances = graph_distances[valid_mask]
    embedding_distances = embedding_distances[valid_mask]
    log(f"Valid distance pairs: {len(graph_distances)}", flush=True)
    n_valid = len(graph_distances)
    results = []
    second_metrics_rows = []

    if include_second:
        # Full aggregate metrics (get_metrics.py style)
        overall_stress = calculate_stress(graph_distances, embedding_distances)
        overall_r2 = calculate_r2(graph_distances, embedding_distances)
        overall_pearson, overall_spearman, overall_kendall = calculate_correlations(
            graph_distances, embedding_distances
        )
        l1_norm, l2_norm, mae, mse = calculate_l1_l2_mae_mse(graph_distances, embedding_distances)
        nmse = calculate_nmse(graph_distances, embedding_distances)
        mape = calculate_mape(graph_distances, embedding_distances)
        canberra_total, canberra_mean = calculate_canberra_metrics(graph_distances, embedding_distances)
        quartile_metrics = compute_quartile_metrics(graph_distances, embedding_distances)

        log("\n=== Overall Metrics ===", flush=True)
        log(
            f"Stress={overall_stress:.6f}, R²(identity)={overall_r2:.6f}, "
            f"Pearson={overall_pearson:.4f}, Spearman={overall_spearman:.4f}, Kendall={overall_kendall:.4f}",
            flush=True,
        )
        log(
            f"L1={l1_norm:.4f}, L2={l2_norm:.4f}, MAE={mae:.6f}, MSE={mse:.6f}, "
            f"NMSE={nmse:.6f}, MAPE={mape:.6f}",
            flush=True,
        )
        log(f"Canberra={canberra_total:.4f}, Mean Canberra={canberra_mean:.6f}", flush=True)

        second_metrics_rows = [
            ("stress_overall", overall_stress),
            ("r2_identity_overall", quartile_metrics["r2_identity_overall"]),
            ("r2_pearson_overall", quartile_metrics["r2_pearson_overall"]),
            ("r2_fit_overall", quartile_metrics["r2_fit_overall"]),
            ("pearson_overall", overall_pearson),
            ("spearman_overall", overall_spearman),
            ("kendall_overall", overall_kendall),
            ("l1_norm", l1_norm),
            ("l2_norm", l2_norm),
            ("mae", mae),
            ("mse", mse),
            ("nmse", nmse),
            ("mape", mape),
            ("canberra", canberra_total),
            ("mean_canberra", canberra_mean),
        ]
        second_metrics_rows.extend(
            (f"spearman_q{i}", val) for i, val in enumerate(quartile_metrics["spearman_q"], start=1)
        )
        second_metrics_rows.extend(
            (f"kendall_q{i}", val) for i, val in enumerate(quartile_metrics["kendall_q"], start=1)
        )
        second_metrics_rows.extend(
            (f"r2_identity_q{i}", val) for i, val in enumerate(quartile_metrics["r2_identity_q"], start=1)
        )
        second_metrics_rows.extend(
            (f"r2_identity_cum{i}", val) for i, val in enumerate(quartile_metrics["r2_identity_c"], start=1)
        )
        second_metrics_rows.extend(
            (f"r2_pearson_q{i}", val) for i, val in enumerate(quartile_metrics["r2_pearson_q"], start=1)
        )
        second_metrics_rows.extend(
            (f"r2_pearson_cum{i}", val) for i, val in enumerate(quartile_metrics["r2_pearson_c"], start=1)
        )
        second_metrics_rows.extend(
            (f"r2_fit_q{i}", val) for i, val in enumerate(quartile_metrics["r2_fit_q"], start=1)
        )
        second_metrics_rows.extend(
            (f"r2_fit_cum{i}", val) for i, val in enumerate(quartile_metrics["r2_fit_c"], start=1)
        )

    if include_first:
        # Sort by graph distance for percentile analysis
        sort_idx = np.lexsort((embedding_distances, graph_distances))
        graph_distances_sorted = graph_distances[sort_idx]
        embedding_distances_sorted = embedding_distances[sort_idx]

        assert np.all(np.diff(graph_distances_sorted) >= 0), "Graph distances should be sorted in ascending order"

        # Calculate metrics for different percentile ranges (0-10, 10-20, ..., 90-100)
        n = len(graph_distances_sorted)
        percentile_ranges = []
        for i in range(10):
            start_pct = i * 10
            end_pct = (i + 1) * 10
            start_idx = int(n * start_pct / 100)
            end_idx = int(n * end_pct / 100)
            if end_idx > start_idx:
                percentile_ranges.append((f"{start_pct}-{end_pct}%", start_idx, end_idx))

        log("\n=== Path Metrics by Percentile Range ===", flush=True)
        for subset_name, start_idx, end_idx in percentile_ranges:
            orig = graph_distances_sorted[start_idx:end_idx]
            emb = embedding_distances_sorted[start_idx:end_idx]

            stress = calculate_stress(orig, emb)
            r2 = calculate_r2(orig, emb)
            pearson, spearman, kendall = calculate_correlations(orig, emb)

            results.append({
                'percentile_range': subset_name,
                'n_paths': len(orig),
                'stress': stress,
                'r2': r2,
                'pearson': pearson,
                'spearman': spearman,
                'kendall': kendall
            })

            log(
                f"{subset_name}: {len(orig)} paths, Stress={stress:.6f}, "
                f"R²={r2:.6f}, Pearson={pearson:.4f}, Spearman={spearman:.4f}, Kendall={kendall:.4f}",
                flush=True,
            )
    
    # Save results as CSV
    if args.output:
        output_file = args.output
    else:
        # Default output: use embedding input name + "_error_metrics.csv"
        if args.csv:
            input_base = args.csv.rsplit(".", 1)[0]
        elif args.embedding_npz:
            input_base = args.embedding_npz.rsplit(".", 1)[0]
        else:
            input_base = "embedding"
        output_file = f"{input_base}_error_metrics.csv"
    
    def _fmt4(v):
        return "nan" if np.isnan(v) else f"{v:.4f}"
    def _fmt6(v):
        return "nan" if np.isnan(v) else f"{v:.6f}"

    with open(output_file, 'w') as f:
        stats_lines = []
        if include_second:
            embedding_file_label = args.embedding_npz if args.embedding_npz else args.csv
            distance_file_label = args.npz if args.npz else f"{args.graphml_file} (computed)"
            r2_id_q1, r2_id_q2, r2_id_q3, r2_id_q4 = quartile_metrics["r2_identity_q"]
            r2_id_c25, r2_id_c50, r2_id_c75, r2_id_c100 = quartile_metrics["r2_identity_c"]
            r2_p_q1, r2_p_q2, r2_p_q3, r2_p_q4 = quartile_metrics["r2_pearson_q"]
            r2_p_c25, r2_p_c50, r2_p_c75, r2_p_c100 = quartile_metrics["r2_pearson_c"]
            r2_fit_q = quartile_metrics["r2_fit_q"]
            r2_fit_c = quartile_metrics["r2_fit_c"]
            spearman_q1, spearman_q2, spearman_q3, spearman_q4 = quartile_metrics["spearman_q"]
            kendall_q1, kendall_q2, kendall_q3, kendall_q4 = quartile_metrics["kendall_q"]
            stats_lines = [
                "\n--- Embedding Quality Metrics (new) ---",
                f"Embedding file: {embedding_file_label}",
                f"Distance matrix file: {distance_file_label}",
                f"Embedding shape: {embedding_coords.shape}",
                "",
                "Fidelity to original distances (D_in):",
                f"Stress: {overall_stress:.6f}",
                f"L1 norm: {l1_norm:.4f}, L2 norm: {l2_norm:.4f}",
                f"MAE: {mae:.6f}, MSE: {mse:.6f}",
                f"Pearson: {overall_pearson:.4f}, Spearman: {overall_spearman:.4f}, Kendall tau: {overall_kendall:.4f}",
                f"Canberra: {canberra_total:.4f}, Mean Canberra: {canberra_mean:.6f}",
                f"NMSE: {nmse:.6f}, MAPE: {mape:.6f}",
                "",
                "Correlation per quartile (0–25, 25–50, 50–75, 75–100% of orig distances):",
                f"Spearman: {_fmt4(spearman_q1)}, {_fmt4(spearman_q2)}, {_fmt4(spearman_q3)}, {_fmt4(spearman_q4)}",
                f"Kendall:  {_fmt4(kendall_q1)}, {_fmt4(kendall_q2)}, {_fmt4(kendall_q3)}, {_fmt4(kendall_q4)}",
                "",
                "R² (Identity: y ≈ x, baseline mean(x)):",
                f"  overall: {_fmt6(quartile_metrics['r2_identity_overall'])}; quartiles: {_fmt4(r2_id_q1)}, {_fmt4(r2_id_q2)}, {_fmt4(r2_id_q3)}, {_fmt4(r2_id_q4)}; cumulative: {_fmt4(r2_id_c25)}, {_fmt4(r2_id_c50)}, {_fmt4(r2_id_c75)}, {_fmt4(r2_id_c100)}",
                "",
                "R² (Pearson r²):",
                f"  overall: {_fmt6(quartile_metrics['r2_pearson_overall'])}; quartiles: {_fmt4(r2_p_q1)}, {_fmt4(r2_p_q2)}, {_fmt4(r2_p_q3)}, {_fmt4(r2_p_q4)}; cumulative: {_fmt4(r2_p_c25)}, {_fmt4(r2_p_c50)}, {_fmt4(r2_p_c75)}, {_fmt4(r2_p_c100)}",
                "",
                "R² (Linear-fit y ≈ a + b x):",
                f"  overall: {_fmt6(quartile_metrics['r2_fit_overall'])}; quartiles: {_fmt4(r2_fit_q[0])}, {_fmt4(r2_fit_q[1])}, {_fmt4(r2_fit_q[2])}, {_fmt4(r2_fit_q[3])}; cumulative: {_fmt4(r2_fit_c[0])}, {_fmt4(r2_fit_c[1])}, {_fmt4(r2_fit_c[2])}, {_fmt4(r2_fit_c[3])}",
                "",
                "---------------------------------",
            ]
            f.write("\n".join(stats_lines))

        if include_first:
            if include_second:
                f.write("\n\n")
            f.write(f"# Total valid paths: {n_valid}\n")
            f.write("percentile_range,n_paths,stress,r2,pearson,spearman,kendall\n")
            for result in results:
                f.write(
                    f"{result['percentile_range']},{result['n_paths']},{result['stress']:.6f},"
                    f"{result['r2']:.6f},{result['pearson']:.4f},{result['spearman']:.4f},"
                    f"{result['kendall']:.4f}\n"
                )


if __name__ == "__main__":
    main()
