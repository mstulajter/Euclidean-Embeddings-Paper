#!/usr/bin/env python3

"""
Generate embeddings from GraphML.bz2 using LMDS and a distance-preserving encoder.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: encoder_embeddings.py <config.toml>
"""

import os
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import copy
import random
import bz2
import argparse
import gc
import time
import tomllib
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import networkit as nk
import networkx as nx

# --- Config ---

def load_config(config_file):
    """Load config from TOML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, 'rb') as f:
        return tomllib.load(f)


def get_default_config():
    """Return default config dict."""
    return {
        'input': {
            'input_file': None,
            'weight_attr': "weight",
            'cores': 1,
            'node_features_file': None,
            'initialization_method': 'landmark_mds',
            'landmark_ratio': 0.3,
            'landmark_selection': 'random',
            'max_memory_gb': 1800,
        },
        'model': {
            'embedding_dim': 10,
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1,
            'num_features': 30,
            'batch_size': 1000,
        },
        'training': {
            'epochs': 100000,
            'learning_rate': 0.1,
            'sample_ratio': 0.3,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'patience': 1000,
            'weight_decay': 1e-5,
            'eps': 1e-8,
            'max_norm': 1.0,
            'eta_min': 0.001,
            'pair_batch_size': 1_000_000,
        },
        'system': {
            'device': 'auto',
            'seed': None,
            'save': True,
            'output_name': None,
            'save_node_features': True,
            'save_npz': True,
        },
    }


def merge_configs(default_config, file_config=None):
    """Merge defaults with file config; file overrides."""
    if file_config is None:
        return default_config
    merged = copy.deepcopy(default_config)
    for section in merged:
        if section in file_config:
            merged[section].update(file_config[section])
    return merged

# --- Output ---

def generate_output_name(input_file, output_name=None):
    """Generate output name from input file stem, with collision-safe suffix."""
    if output_name:
        return output_name

    base = os.path.basename(str(input_file))
    if base.endswith(".graphml.bz2"):
        stem = base[:-12]
    else:
        stem = os.path.splitext(base)[0]
    stem = stem or "encoder"

    candidate = stem
    idx = 1
    while (
        os.path.exists(f"{candidate}.npz")
        or os.path.exists(f"{candidate}.csv")
        or os.path.exists(f"{candidate}_config.toml")
        or os.path.exists(f"{candidate}_node_features.csv")
    ):
        idx += 1
        candidate = f"{stem}_{idx}"
    return candidate


def save_config_toml(config, output_file):
    """Write config to TOML file."""
    def format_value(value):
        if isinstance(value, str):
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, list):
            if not value:
                return '[]'
            items = [f'"{v}"' if isinstance(v, str) else str(v) for v in value]
            return '[' + ', '.join(items) + ']'
        return str(value)
    
    with open(output_file, 'w') as f:
        f.write("# Configuration used for embedding generation\n\n")
        for section, values in config.items():
            if section.startswith('_') or not isinstance(values, dict):
                continue
            f.write(f"[{section}]\n")
            for key, value in values.items():
                if value is None:
                    f.write(f"# {key} = null\n")
                else:
                    f.write(f"{key} = {format_value(value)}\n")
            f.write("\n")

# --- Random ---

def get_seeded_generator(seed, device='cpu', offset=0):
    """Return a seeded PyTorch generator for the given device."""
    if seed is None:
        return None
    
    effective_seed = seed + offset if offset > 0 else seed
    
    try:
        if device.startswith('cuda') and torch.cuda.is_available():
            generator = torch.Generator(device=device)
        elif device.startswith('xpu') and hasattr(torch, 'xpu') and torch.xpu.is_available():
            generator = torch.Generator(device=device)
        elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator(device='cpu')
        generator.manual_seed(effective_seed)
        return generator
    except Exception:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(effective_seed)
        return generator


def set_random_seeds(seed=None, device=None):
    """Set Python, NumPy, PyTorch seeds for reproducibility."""
    if seed is None or seed == "":
        return
    
    if isinstance(seed, str):
        try:
            seed = int(seed)
        except ValueError:
            print(f"Warning: Invalid seed value '{seed}', skipping seed setting", flush=True)
            return
    
    print(f"Setting random seed to {seed}", flush=True)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    
    if device is not None:
        if device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif device.startswith('xpu') and hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
        elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize_seed(seed):
    """Normalize seed config value to int or None."""
    if seed in (None, ""):
        return None
    if isinstance(seed, str):
        seed = seed.strip()
        if not seed:
            return None
        return int(seed)
    return int(seed)


def resolve_output_name(config):
    """Resolve output name from config, or auto-generate one."""
    output_name_config = config['system'].get('output_name')
    if isinstance(output_name_config, str) and output_name_config.strip():
        return output_name_config.strip()
    return generate_output_name(config['input'].get('input_file'), None)


def build_parser():
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from GraphML.bz2 files using TOML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s encoder_config.toml

Example example_encoder_config.toml:
  [input]
  input_file = "Networks/RN.graphml.bz2"
  weight_attr = "weight"
  cores = 1
  initialization_method = "landmark_mds"
  landmark_ratio = 0.3
  landmark_selection = "maxmin"
  max_memory_gb = 64

  [model]
  embedding_dim = 5
  hidden_dims = [128, 64]
  dropout = 0.2
  num_features = 20

  [training]
  epochs = 5000
  learning_rate = 0.1
  sample_ratio = 0.3
  val_ratio = 0.15
  test_ratio = 0.15
  patience = 100
  weight_decay = 1e-4
  pair_batch_size = 1000000

  [system]
  device = "auto"
  seed = 42
  save = true
  output_name = "RN_encoder"
  save_node_features = true
  save_npz = true
        """
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to TOML configuration file",
    )
    return parser

# --- Graph ---

def read_graphml_bz2_nx(input_file):
    """Read GraphML.bz2 into NetworkX graph."""
    with bz2.open(input_file) as file_tmp:
        return nx.read_graphml(file_tmp)


def get_nk_graph_and_mapping(graph_nx, weightAttr=None):
    """Convert NetworkX to Networkit and return (G_nk, index->node_id mapping)."""
    sorted_nodes = graph_nx.nodes()
    mapping = {i: node for i, node in enumerate(sorted_nodes)}
    subgraph_nx = graph_nx.subgraph(sorted_nodes)
    G_nk = nk.nxadapter.nx2nk(subgraph_nx, weightAttr=weightAttr)
    return G_nk, mapping


def get_nodes_graphml(graphml_file, node_ids):
    """Extract node labels from GraphML for given node_ids.
    Uses iterparse to avoid loading the entire file into memory (avoids OverflowError
    for very large GraphML files).
    """
    ns = '{http://graphml.graphdrawing.org/xmlns}'
    node_ids_set = set(node_ids)
    lookup = {}

    with bz2.open(graphml_file, 'rt', encoding='utf-8') as f:
        for event, elem in ET.iterparse(f, events=('end',)):
            if elem.tag == f'{ns}node':
                nid = elem.get('id')
                if nid in node_ids_set:
                    data_elem = elem.find(f'{ns}data')
                    lookup[nid] = data_elem.text if data_elem is not None else None
                elem.clear()
                if len(lookup) == len(node_ids_set):
                    break

    return [lookup[nid] for nid in node_ids]

# --- Distances (compact upper triangle) ---

def _compact_size(n):
    """Number of upper-triangle entries: n*(n-1)//2."""
    return n * (n - 1) // 2


def expand_compact_to_full(compact, n):
    """Expand upper-triangle compact to full n×n symmetric matrix."""
    out = np.zeros((n, n), dtype=compact.dtype)
    i, j = np.triu_indices(n, k=1)
    out[i, j] = compact
    out[j, i] = compact
    return out


def get_distances_for_pairs(compact, n, i_indices, j_indices, device=None):
    """Look up distances for pairs (i[k], j[k]) from compact upper triangle."""
    compact_np = np.asarray(compact)
    i_np = np.asarray(i_indices, dtype=np.intp)
    j_np = np.asarray(j_indices, dtype=np.intp)
    if i_np.size == 0:
        out = np.array([], dtype=compact_np.dtype)
    else:
        i_small = np.minimum(i_np, j_np)
        j_large = np.maximum(i_np, j_np)
        idx_flat = np.where(i_small < j_large, i_small * (2 * n - i_small - 1) // 2 + (j_large - i_small - 1), -1)
        out = np.where(idx_flat >= 0, compact_np[idx_flat], 0.0)
    if device is not None and hasattr(i_indices, 'device'):
        return torch.tensor(out, dtype=torch.float32, device=device)
    return out


def get_distance_matrix_compact(G):
    """(compact 1D upper triangle, n). Saves ~half memory."""
    compact, n = get_distance_matrix_sampled_compact(G, list(range(G.numberOfNodes())))
    return compact, n


def get_distance_matrix_full(G, row_chunk_size=None, progress_interval=10):
    """Full n×n distance matrix by running SPSP in row chunks."""
    n = G.numberOfNodes()
    all_nodes = list(range(n))
    if row_chunk_size is None or row_chunk_size <= 0 or row_chunk_size >= n:
        row_chunk_size = n
    print(f"  [SPSP] Allocating full distance matrix ({n}×{n})...", flush=True)
    full_dist = np.zeros((n, n), dtype=np.float64)
    print(f"  [SPSP] Allocation done. Filling in row chunks (chunk_size={row_chunk_size})...", flush=True)
    next_pct = progress_interval if progress_interval else 101
    for start in range(0, n, row_chunk_size):
        end = min(start + row_chunk_size, n)
        sources = list(range(start, end))
        spsp = nk.distance.SPSP(G, sources)
        spsp.setTargets(all_nodes)
        spsp.run()
        chunk = np.asarray(spsp.getDistances(asarray=True), dtype=np.float64)
        full_dist[start:end, :] = chunk
        del chunk
        if progress_interval and progress_interval > 0:
            pct = 100 * (end / n)
            if pct >= next_pct or end == n:
                print(f"  Full distance matrix: {end}/{n} rows ({pct:.1f}%)", flush=True)
                next_pct = min(100, (int(pct) // progress_interval + 1) * progress_interval)
    print(f"  [SPSP] Full distance matrix filled.", flush=True)
    return full_dist


def get_distance_matrix_sampled_compact(G, sampled_nodes):
    """(compact upper triangle for sampled nodes, len(sampled_nodes))."""
    spsp = nk.distance.SPSP(G, sampled_nodes)
    spsp.setTargets(sampled_nodes)
    spsp.run()
    num_nodes = len(sampled_nodes)
    dists = np.asarray(spsp.getDistances(asarray=True), dtype=np.float64)
    i, j = np.triu_indices(num_nodes, k=1)
    compact = dists[i, j].copy()
    return compact, num_nodes


def get_graph_and_nk(input_file, cores=1, w=None):
    """Load GraphML.bz2, return (graph_nx, G_nk, mapping, node_ids). No distance matrix built."""
    nk.setNumberOfThreads(int(cores))
    graph_nx = read_graphml_bz2_nx(input_file)
    G_nk, mapping = get_nk_graph_and_mapping(graph_nx, w)
    node_ids = [mapping[i] for i in range(len(mapping))]
    return graph_nx, G_nk, mapping, node_ids


# --- Model ---

class DistancePreservingEncoder(nn.Module):
    """Encoder trained with distance-preserving (normalized stress) loss."""

    def __init__(self, config):
        super(DistancePreservingEncoder, self).__init__()
        model_config = config.get('model') or {}
        input_dim = model_config.get('num_features')
        embedding_dim = model_config.get('embedding_dim')
        hidden_dims = model_config.get('hidden_dims')
        dropout = model_config.get('dropout')

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if hasattr(self, 'encoder') and len(self.encoder) > 0:
            final_layer = self.encoder[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.normal_(final_layer.weight, mean=0, std=0.01)
                if final_layer.bias is not None:
                    nn.init.constant_(final_layer.bias, 0)

    def forward(self, x):
        return self.encoder(x)

    def encode(self, x):
        return self.encoder(x)

def normalized_stress_loss(original_distances, embedded_distances, eps=1e-12):
    """Normalized stress between original and embedded pair distances."""
    if len(original_distances) != len(embedded_distances):
        raise ValueError(
            f"Distance tensors must have equal length: "
            f"original={len(original_distances)}, embedded={len(embedded_distances)}"
        )
    mask = (original_distances > 0).float()
    orig_safe = original_distances + eps
    emb_safe = embedded_distances + eps
    diff_squared = (orig_safe - emb_safe) ** 2
    orig_squared = orig_safe ** 2
    numerator = torch.sum(diff_squared * mask)
    denominator = torch.sum(orig_squared * mask)
    if denominator == 0:
        return torch.tensor(0.0, device=original_distances.device)
    return torch.sqrt(numerator / denominator)

def create_distance_matrix(embeddings):
    """Pairwise L2 distances (n×n)."""
    return torch.cdist(embeddings, embeddings, p=2)


# --- Sampling ---

def sample_distance_pairs_without_matrix(G_nk, n, config, device):
    """Sample pair indices by selecting sample_ratio percentage of nodes, then all pairs from those nodes."""
    training_config = config.get('training')
    sample_ratio = float(training_config.get('sample_ratio'))
    if n <= 1:
        raise ValueError(f"Need at least 2 points to form pairs, got {n}")
    seed = config.get('_seed')
    
    # Select sample_ratio percentage of nodes
    n_selected = max(2, int(n * sample_ratio))
    n_selected = min(n_selected, n)
    rng = np.random.default_rng(seed)
    selected_nodes = rng.choice(n, size=n_selected, replace=False)
    selected_nodes = np.sort(selected_nodes)  # Sort for consistent ordering
    
    # Generate all pairs from selected nodes
    i_list, j_list = [], []
    for idx, i in enumerate(selected_nodes):
        for j in selected_nodes[idx + 1:]:
            i_list.append(i)
            j_list.append(j)
    
    all_i = torch.tensor(i_list, dtype=torch.long, device=device)
    all_j = torch.tensor(j_list, dtype=torch.long, device=device)
    sampled_pairs = len(all_i)
    print(f"\nDistance Pair Sampling (no full matrix): selected {n_selected}/{n} nodes ({sample_ratio*100:.1f}%), sampled {sampled_pairs} pairs", flush=True)
    return all_i, all_j


def get_pair_distances_from_graph(G_nk, train_i, train_j, val_i, val_j, test_i, test_j, device):
    """SPSP over unique nodes in pairs; return (train_d, val_d, test_d) 1D tensors."""
    def unique_nodes_from_pairs(i_t, j_t):
        if i_t is None or len(i_t) == 0:
            return np.array([], dtype=np.int64)
        return np.unique(torch.cat([i_t.cpu(), j_t.cpu()]).numpy())
    uniq = np.unique(np.concatenate([
        unique_nodes_from_pairs(train_i, train_j),
        unique_nodes_from_pairs(val_i, val_j),
        unique_nodes_from_pairs(test_i, test_j)
    ]))
    if len(uniq) == 0:
        return (
            torch.tensor([], dtype=torch.float32, device=device),
            torch.tensor([], dtype=torch.float32, device=device),
            torch.tensor([], dtype=torch.float32, device=device),
        )
    print(f"  Computing pair distances for {len(uniq)} unique nodes only (one-time SPSP)...", flush=True)
    k = len(uniq)
    compact_small, _ = get_distance_matrix_sampled_compact(G_nk, uniq.tolist())
    # Map original node indices to indices in uniq array
    node_to_idx = {node: idx for idx, node in enumerate(uniq)}
    def map_indices(i_arr, j_arr):
        i_mapped = np.array([node_to_idx.get(i, -1) for i in i_arr], dtype=np.int64)
        j_mapped = np.array([node_to_idx.get(j, -1) for j in j_arr], dtype=np.int64)
        return i_mapped, j_mapped
    
    train_i_mapped, train_j_mapped = map_indices(train_i.cpu().numpy(), train_j.cpu().numpy())
    train_d = torch.tensor(
        get_distances_for_pairs(compact_small, k, train_i_mapped, train_j_mapped),
        dtype=torch.float32, device=device
    )
    val_d = None
    if val_i is not None and len(val_i) > 0:
        val_i_mapped, val_j_mapped = map_indices(val_i.cpu().numpy(), val_j.cpu().numpy())
        val_d = torch.tensor(
            get_distances_for_pairs(compact_small, k, val_i_mapped, val_j_mapped),
            dtype=torch.float32, device=device
        )
    test_d = None
    if test_i is not None and len(test_i) > 0:
        test_i_mapped, test_j_mapped = map_indices(test_i.cpu().numpy(), test_j.cpu().numpy())
        test_d = torch.tensor(
            get_distances_for_pairs(compact_small, k, test_i_mapped, test_j_mapped),
            dtype=torch.float32, device=device
        )
    return train_d, val_d, test_d


def sample_distance_pairs(distance_matrix_or_compact, config, n=None, device=None):
    """Sample (i, j) for training; accepts full n×n or compact 1D + n."""
    training_config = config.get('training')
    system_config = config.get('system')
    if device is None:
        device = system_config.get('device')
    
    sample_ratio = float(training_config.get('sample_ratio'))

    arr = np.asarray(distance_matrix_or_compact)
    use_compact = arr.ndim == 1
    if use_compact:
        if n is None:
            raise ValueError("n must be provided when using compact distance storage")
        compact = arr
        if len(compact) != _compact_size(n):
            raise ValueError(f"Compact length {len(compact)} != n*(n-1)/2 = {_compact_size(n)} for n={n}")
    else:
        if not isinstance(distance_matrix_or_compact, torch.Tensor):
            distance_matrix = torch.FloatTensor(distance_matrix_or_compact).to(device)
        else:
            distance_matrix = distance_matrix_or_compact.to(device)
        n = distance_matrix.shape[0]

    seed = config.get('_seed')
    total_pairs = n * (n - 1) // 2
    
    if n <= 1:
        raise ValueError(f"Need at least 2 points to form pairs, got {n}")
    
    # Select sample_ratio percentage of nodes, then all pairs from those nodes
    n_selected = max(2, int(n * sample_ratio))
    n_selected = min(n_selected, n)
    rng = np.random.default_rng(seed)
    selected_nodes = rng.choice(n, size=n_selected, replace=False)
    selected_nodes = np.sort(selected_nodes)  # Sort for consistent ordering
    
    # Generate all pairs from selected nodes
    i_list, j_list = [], []
    for idx, i in enumerate(selected_nodes):
        for j in selected_nodes[idx + 1:]:
            i_list.append(i)
            j_list.append(j)
    
    i_arr = np.array(i_list, dtype=np.int64)
    j_arr = np.array(j_list, dtype=np.int64)
    selected_i = torch.tensor(i_arr, dtype=torch.long, device=device)
    selected_j = torch.tensor(j_arr, dtype=torch.long, device=device)
    # Keep distances on CPU for stats to avoid GPU OOM when sample_size is large
    if use_compact:
        dist_np = get_distances_for_pairs(compact, n, i_arr, j_arr)
    else:
        dist_np = distance_matrix[selected_i, selected_j].cpu().numpy()
    stats = {
        'min': float(np.min(dist_np)),
        'max': float(np.max(dist_np)),
        'mean': float(np.mean(dist_np)),
        'median': float(np.median(dist_np))
    } if len(dist_np) > 0 else {}
    sampled_pairs = len(selected_i)
    print(f"\nDistance Pair Sampling Results:", flush=True)
    print(f"  Selected {n_selected}/{n} nodes ({sample_ratio*100:.1f}%)", flush=True)
    print(f"  Total pairs: {total_pairs}, Sampled pairs: {sampled_pairs}", flush=True)
    if stats:
        print(f"  Distance stats: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, median={stats['median']:.2f}", flush=True)
    print(flush=True)
    return selected_i, selected_j

# --- Data ---

def load_node_features_from_csv(csv_file, node_labels):
    """Load features from CSV; rows match node_labels order."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Node features file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    required_cols = {'node', 'coordinates'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain {required_cols}. Found: {df.columns.tolist()}")

    node_to_features = {
        str(row['node']): [float(x) for x in str(row['coordinates']).split(':')]
        for _, row in df.iterrows()
    }

    features_list = [node_to_features.get(str(label)) for label in node_labels]
    missing = [i for i, f in enumerate(features_list) if f is None]
    
    if missing:
        raise ValueError(f"Missing features for {len(missing)} nodes. First few: {[node_labels[i] for i in missing[:5]]}")
    
    return np.array(features_list, dtype=np.float32)

# --- Node features (LMDS) ---

MIN_CROSS_CHUNK_SIZE = 1000
MEMORY_SAFETY_FACTOR = 0.75


def choose_lmds_memory_strategy(max_memory_gb, n_nodes, landmark_ratio, num_features):
    """Choose use_full_distance_matrix, full_matrix_row_chunk_size, use_chunked_cross_distances, cross_dist_chunk_size from max_memory_gb and n_nodes.

    Never uses fewer landmarks than requested (raises if that would be needed).
    cross_dist_chunk_size is never below MIN_CROSS_CHUNK_SIZE (raises if that would be needed).
    """
    if max_memory_gb is None or max_memory_gb <= 0:
        return None
    M = MEMORY_SAFETY_FACTOR * max_memory_gb * (1024**3)
    C = M / 8
    num_landmarks = max(int(np.round(n_nodes * landmark_ratio)), min(num_features + 1, n_nodes))
    num_landmarks = min(num_landmarks, n_nodes)

    if landmark_ratio >= 1.0 or num_landmarks >= n_nodes:
        if n_nodes * n_nodes > C:
            need_gb = (n_nodes * n_nodes * 8) / (1024**3)
            raise ValueError(
                f"Insufficient memory for full distance matrix: {n_nodes}×{n_nodes} needs ~{need_gb:.1f} GB "
                f"(max_memory_gb={max_memory_gb}, safety={MEMORY_SAFETY_FACTOR}). Increase max_memory_gb or use landmark_ratio < 1."
            )
        row_chunk = int((C - n_nodes * n_nodes) / n_nodes)
        row_chunk = max(1, min(50000, row_chunk))
        return (True, row_chunk, False, None)

    k = num_landmarks
    full_cross_elements = k * (2 * n_nodes - k)
    if full_cross_elements <= C:
        return (False, None, False, None)

    # Peak in chunked path (symmetric/packed): landmark_dist packed (0.5×k²) + full kernel for eigh (k²) + cross_chunk + B_chunk = 1.5×k² + 2×k×chunk
    if 3 * k * k // 2 >= C:
        need_gb = (3 * k * k // 2 + 2 * k * MIN_CROSS_CHUNK_SIZE) * 8 / (1024**3)
        raise ValueError(
            f"Insufficient memory for {100*landmark_ratio:.0f}% landmarks ({k}): "
            f"1.5×k² alone needs ~{1.5*k*k*8/(1024**3):.1f} GB (max_memory_gb={max_memory_gb}). Increase max_memory_gb or decrease landmark_ratio."
        )
    chunk_max = (C - 3 * k * k // 2) / (2 * k)
    if chunk_max < MIN_CROSS_CHUNK_SIZE:
        need_gb = (3 * k * k // 2 + 2 * k * MIN_CROSS_CHUNK_SIZE) * 8 / (1024**3)
        raise ValueError(
            f"Insufficient memory for {100*landmark_ratio:.0f}% landmarks ({k}) with cross_dist_chunk_size >= {MIN_CROSS_CHUNK_SIZE}: "
            f"would need ~{need_gb:.1f} GB (max_memory_gb={max_memory_gb}). Increase max_memory_gb or decrease landmark_ratio."
        )
    cross_chunk = max(MIN_CROSS_CHUNK_SIZE, min(50000, int(chunk_max)))
    return (False, None, True, cross_chunk)


def _packed_upper_idx(k, i, j):
    """Flat index in row-major upper triangle (i <= j)."""
    return i * (2 * k - i - 1) // 2 + j


def double_center(dist_matrix):
    dist_sq = dist_matrix**2
    row_means = dist_sq.mean(axis=1, keepdims=True)
    col_means = dist_sq.mean(axis=0, keepdims=True)
    total_mean = dist_sq.mean()
    return -0.5 * (dist_sq - row_means - col_means + total_mean)


def double_center_packed(packed_dist, k, row_sums_sq):
    """Double-center from packed upper triangle. row_sums_sq[i] = sum of D[i,:]**2 (passed from filler). Returns packed upper triangle of kernel."""
    row_means_sq = (row_sums_sq / k).astype(np.float64)
    total_mean_sq = row_sums_sq.sum() / (k * k)
    packed_kernel = np.empty(k * (k + 1) // 2, dtype=np.float64)
    for i in range(k):
        start = _packed_upper_idx(k, i, i)
        size = k - i
        packed_kernel[start:start + size] = -0.5 * (
            packed_dist[start:start + size] ** 2
            - row_means_sq[i]
            - row_means_sq[i:]
            + total_mean_sq
        )
    return packed_kernel


def _packed_to_full_upper(packed, k):
    """Expand packed upper triangle to full symmetric (k,k) matrix. Vectorized per row."""
    full = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        start = _packed_upper_idx(k, i, i)
        size = k - i
        full[i, i : i + size] = packed[start : start + size]
        full[i : i + size, i] = packed[start : start + size]
    return full


def double_center_inplace(D):
    """Double-center in place: D is overwritten with the Gram matrix (saves memory for full n×n)."""
    np.square(D, out=D)
    row_means = D.mean(axis=1, keepdims=True)
    col_means = D.mean(axis=0, keepdims=True)
    total_mean = D.mean()
    D -= row_means
    D -= col_means
    D += total_mean
    D *= -0.5


def classical_mds_full_matrix(D, n_components):
    """Classical MDS on full n×n distance matrix. Modifies D in place; returns (n, n_components) float64."""
    print(f"  [MDS] double_center_inplace...", flush=True)
    double_center_inplace(D)
    print(f"  [MDS] eigh...", flush=True)
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    max_components = min(n_components, len(eigenvalues))
    positive = eigenvalues[sorted_idx] > 1e-10
    num_positive = np.sum(positive)
    actual_components = min(max_components, num_positive) if num_positive > 0 else max_components
    eigenvalues = eigenvalues[sorted_idx][:actual_components]
    eigenvectors = eigenvectors[:, sorted_idx][:, :actual_components]
    sqrt_ev = np.sqrt(np.maximum(eigenvalues, 0))
    coords = eigenvectors @ np.diag(sqrt_ev)
    if actual_components < n_components:
        padded = np.zeros((D.shape[0], n_components), dtype=np.float64)
        padded[:, :actual_components] = coords
        return padded
    return coords


def double_center_cross(landmark_dist, cross_dist):
    landmark_sq = landmark_dist**2
    cross_sq = cross_dist**2
    col_means = cross_sq.mean(axis=0, keepdims=True)
    row_means = landmark_sq.mean(axis=1, keepdims=True)
    return -0.5 * (cross_sq - col_means - row_means)


def double_center_cross_packed(packed_dist, k, row_sums_sq, cross_dist):
    """B = double_center_cross without forming full landmark_sq; uses row_sums_sq/k as row means of D**2."""
    row_means_sq = (row_sums_sq / k).reshape(-1, 1)
    cross_sq = np.square(cross_dist, dtype=np.float64)
    col_means = cross_sq.mean(axis=0, keepdims=True)
    return -0.5 * (cross_sq - col_means - row_means_sq)


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


def select_landmarks_from_graph(G_nk, num_landmarks, method='random', seed=None):
    """Landmarks without full matrix; method 'random' or 'maxmin'."""
    n_nodes = G_nk.numberOfNodes()
    if seed is not None:
        np.random.seed(seed)
    if method == 'random':
        landmark_indices = np.random.choice(n_nodes, size=num_landmarks, replace=False)
        non_landmark_indices = np.setdiff1d(np.arange(n_nodes), landmark_indices)
        return np.array(landmark_indices), non_landmark_indices
    dist_min = np.full(n_nodes, np.inf)
    start_node = np.random.randint(0, n_nodes)
    dijkstra = nk.distance.Dijkstra(G_nk, start_node).run()
    for target in range(n_nodes):
        dist_min[target] = dijkstra.distance(target)
    selected = [start_node]
    for _ in range(num_landmarks - 1):
        next_node = int(np.argmax(dist_min))
        if np.isinf(dist_min[next_node]):
            break
        selected.append(next_node)
        dijkstra = nk.distance.Dijkstra(G_nk, next_node).run()
        for target in range(n_nodes):
            d = dijkstra.distance(target)
            if d < dist_min[target]:
                dist_min[target] = d
    landmark_indices = np.array(selected)
    non_landmark_indices = np.setdiff1d(np.arange(n_nodes), landmark_indices)
    return landmark_indices, non_landmark_indices


def get_landmark_distances(G_nk, landmark_indices, non_landmark_indices, progress_interval=None):
    """Compute landmark-to-landmark and landmark-to-non-landmark distances iteratively.

    Processes one landmark at a time for the Dijkstra (shortest-path) work; the main
    memory cost is the pre-allocated matrices: landmark_dist (k×k) and
    landmark_to_non_landmark_dist (k×n_non). For large k and n_non this can be hundreds
    of GB and cause OOM. Use use_chunked_cross_distances=true or set max_memory_gb in
    config so the chunked path is used instead.
    """
    k = len(landmark_indices)
    n_non = len(non_landmark_indices)

    landmark_dist = np.zeros((k, k), dtype=np.float64)
    landmark_to_non_landmark_dist = np.zeros((k, n_non), dtype=np.float64) if n_non > 0 else np.array([]).reshape(k, 0)

    if progress_interval is None and k > 1000:
        progress_interval = max(1, k // 20)

    GC_INTERVAL = 50  # force GC every N Dijkstra runs to limit memory buildup
    t0 = time.time()
    for i in range(k):
        dijkstra = nk.distance.Dijkstra(G_nk, int(landmark_indices[i])).run()
        for j in range(k):
            landmark_dist[i, j] = dijkstra.distance(int(landmark_indices[j]))
        for j, nidx in enumerate(non_landmark_indices):
            landmark_to_non_landmark_dist[i, j] = dijkstra.distance(int(nidx))
        del dijkstra
        if (i + 1) % GC_INTERVAL == 0:
            gc.collect()
        if progress_interval and (i + 1) % progress_interval == 0:
            elapsed = time.time() - t0
            print(f"  Landmark distances: {i + 1}/{k} ({100 * (i + 1) / k:.1f}%) [{elapsed:.1f}s]", flush=True)
    if k > 0 and (not progress_interval or k % progress_interval != 0):
        elapsed = time.time() - t0
        print(f"  Landmark distances: {k}/{k} (100.0%) [{elapsed:.1f}s]", flush=True)
    gc.collect()
    return landmark_dist, landmark_to_non_landmark_dist


def get_landmark_distances_landmarks_only(G_nk, landmark_indices, progress_interval=None, packed=True, use_spsp_batches=True):
    """Compute only landmark-to-landmark distances. If packed=True, return (packed_upper, row_sums_sq) to save memory.
    When use_spsp_batches=True and k is large, uses batched SPSP (sources=batch of landmarks, targets=all landmarks)
    instead of k single-source Dijkstras to avoid memory growth from many Dijkstra runs."""
    k = len(landmark_indices)
    n_packed = k * (k + 1) // 2
    if progress_interval is None and k > 1000:
        progress_interval = max(1, k // 20)
    landmark_list = [int(x) for x in landmark_indices]
    # Batched SPSP: one (batch_size × k) result per batch instead of k Dijkstra runs (avoids OOM from per-run state).
    SPSP_BATCH_MAX_GB = 1.0
    spsp_batch_size = min(k, max(1, int(SPSP_BATCH_MAX_GB * (1024**3) / (k * 8)))) if k > 0 else k
    use_batched = use_spsp_batches and k > 1000 and spsp_batch_size < k
    if packed:
        packed_dist = np.zeros(n_packed, dtype=np.float64)
        row_sums_sq = np.zeros(k, dtype=np.float64)
        if use_batched:
            print(f"  [LMDS] Landmark–landmark via batched SPSP (batch_size={spsp_batch_size})...", flush=True)
            t0 = time.time()
            next_pct = 5.0
            for batch_start in range(0, k, spsp_batch_size):
                batch_end = min(batch_start + spsp_batch_size, k)
                sources = landmark_list[batch_start:batch_end]
                spsp = nk.distance.SPSP(G_nk, sources)
                spsp.setTargets(landmark_list)
                spsp.run()
                chunk = np.asarray(spsp.getDistances(asarray=True), dtype=np.float64)
                del spsp
                # chunk shape: (batch_size, k)
                for local_i in range(chunk.shape[0]):
                    i = batch_start + local_i
                    row = chunk[local_i, :]
                    row_sums_sq[i] = np.sum(row * row)
                    for j in range(i, k):
                        idx = _packed_upper_idx(k, i, j)
                        packed_dist[idx] = row[j]
                del chunk
                gc.collect()
                pct = 100.0 * batch_end / k
                if pct >= next_pct or batch_end == k:
                    elapsed = time.time() - t0
                    print(f"  Landmark–landmark distances: {batch_end}/{k} ({pct:.1f}%) [{elapsed:.1f}s]", flush=True)
                    next_pct = min(100.0, (int(pct) // 5 + 1) * 5.0)
            if k > 0:
                elapsed = time.time() - t0
                print(f"  Landmark–landmark distances: {k}/{k} (100.0%) [{elapsed:.1f}s]", flush=True)
        else:
            GC_INTERVAL = 50
            t0 = time.time()
            for i in range(k):
                dijkstra = nk.distance.Dijkstra(G_nk, int(landmark_indices[i])).run()
                row_sum = 0.0
                for j in range(i, k):
                    d = dijkstra.distance(int(landmark_indices[j]))
                    idx = _packed_upper_idx(k, i, j)
                    packed_dist[idx] = d
                    row_sum += d * d
                for j in range(i):
                    d = dijkstra.distance(int(landmark_indices[j]))
                    row_sum += d * d
                row_sums_sq[i] = row_sum
                del dijkstra
                if (i + 1) % GC_INTERVAL == 0:
                    gc.collect()
                if progress_interval and (i + 1) % progress_interval == 0:
                    elapsed = time.time() - t0
                    print(f"  Landmark–landmark distances: {i + 1}/{k} ({100 * (i + 1) / k:.1f}%) [{elapsed:.1f}s]", flush=True)
            if k > 0 and (not progress_interval or k % progress_interval != 0):
                elapsed = time.time() - t0
                print(f"  Landmark–landmark distances: {k}/{k} (100.0%) [{elapsed:.1f}s]", flush=True)
            gc.collect()
        return packed_dist, row_sums_sq
    landmark_dist = np.zeros((k, k), dtype=np.float64)
    if use_batched:
        print(f"  [LMDS] Landmark–landmark via batched SPSP (batch_size={spsp_batch_size})...", flush=True)
        t0 = time.time()
        next_pct = 5.0
        for batch_start in range(0, k, spsp_batch_size):
            batch_end = min(batch_start + spsp_batch_size, k)
            sources = landmark_list[batch_start:batch_end]
            spsp = nk.distance.SPSP(G_nk, sources)
            spsp.setTargets(landmark_list)
            spsp.run()
            chunk = np.asarray(spsp.getDistances(asarray=True), dtype=np.float64)
            del spsp
            landmark_dist[batch_start:batch_end, :] = chunk
            del chunk
            gc.collect()
            pct = 100.0 * batch_end / k
            if pct >= next_pct or batch_end == k:
                elapsed = time.time() - t0
                print(f"  Landmark–landmark distances: {batch_end}/{k} ({pct:.1f}%) [{elapsed:.1f}s]", flush=True)
                next_pct = min(100.0, (int(pct) // 5 + 1) * 5.0)
        if k > 0:
            elapsed = time.time() - t0
            print(f"  Landmark–landmark distances: {k}/{k} (100.0%) [{elapsed:.1f}s]", flush=True)
    else:
        GC_INTERVAL = 50
        t0 = time.time()
        for i in range(k):
            dijkstra = nk.distance.Dijkstra(G_nk, int(landmark_indices[i])).run()
            for j in range(k):
                landmark_dist[i, j] = dijkstra.distance(int(landmark_indices[j]))
            del dijkstra
            if (i + 1) % GC_INTERVAL == 0:
                gc.collect()
            if progress_interval and (i + 1) % progress_interval == 0:
                elapsed = time.time() - t0
                print(f"  Landmark–landmark distances: {i + 1}/{k} ({100 * (i + 1) / k:.1f}%) [{elapsed:.1f}s]", flush=True)
        if k > 0 and (not progress_interval or k % progress_interval != 0):
            elapsed = time.time() - t0
            print(f"  Landmark–landmark distances: {k}/{k} (100.0%) [{elapsed:.1f}s]", flush=True)
        gc.collect()
    return landmark_dist


def get_cross_dist_chunk(G_nk, landmark_indices, non_landmark_indices, start, end):
    """Compute (k × (end-start)) distances from landmarks to non_landmark_indices[start:end]. Runs k Dijkstras."""
    k = len(landmark_indices)
    chunk_indices = non_landmark_indices[start:end]
    chunk_size = len(chunk_indices)
    out = np.zeros((k, chunk_size), dtype=np.float64)
    GC_INTERVAL = 50
    for i in range(k):
        dijkstra = nk.distance.Dijkstra(G_nk, int(landmark_indices[i])).run()
        for j, nidx in enumerate(chunk_indices):
            out[i, j] = dijkstra.distance(int(nidx))
        del dijkstra
        if (i + 1) % GC_INTERVAL == 0:
            gc.collect()
    gc.collect()
    return out


def lmds_from_landmark_matrices(landmark_dist, landmark_to_non_landmark_dist,
                                 landmark_indices, non_landmark_indices, n_nodes, components):
    landmark_kernel = double_center(landmark_dist)
    landmark_coords, non_landmark_coords = compute_embedding_coordinates(
        landmark_kernel, landmark_to_non_landmark_dist, landmark_dist, components
    )
    actual_components = landmark_coords.shape[1]
    all_embeddings = np.zeros((n_nodes, components), dtype=np.float64)
    if actual_components <= components:
        all_embeddings[landmark_indices, :actual_components] = landmark_coords
        if len(non_landmark_indices) > 0:
            all_embeddings[non_landmark_indices, :actual_components] = non_landmark_coords
    else:
        all_embeddings[landmark_indices] = landmark_coords[:, :components]
        if len(non_landmark_indices) > 0:
            all_embeddings[non_landmark_indices] = non_landmark_coords[:, :components]
    return all_embeddings


def lmds_from_landmark_matrices_chunked(G_nk, landmark_dist, landmark_indices, non_landmark_indices,
                                        n_nodes, components, cross_chunk_size=50000,
                                        progress_pct_interval=10, row_sums_sq=None):
    """LMDS without full (k × n_non). If row_sums_sq is not None, landmark_dist is packed upper triangle (1D) and we use symmetric optimizations (peak 1.5*k² + 2*k*chunk)."""
    k = len(landmark_indices)
    n_non = len(non_landmark_indices)
    use_packed = row_sums_sq is not None
    if use_packed:
        packed_dist = landmark_dist
        packed_kernel = double_center_packed(packed_dist, k, row_sums_sq)
        full_kernel = _packed_to_full_upper(packed_kernel, k)
        del packed_kernel
        eigenvalues, eigenvectors_full = np.linalg.eigh(full_kernel)
        del full_kernel
    else:
        landmark_kernel = double_center(landmark_dist)
        eigenvalues, eigenvectors_full = np.linalg.eigh(landmark_kernel)
        del landmark_kernel
    sorted_idx = np.argsort(eigenvalues)[::-1]
    max_components = min(components, len(eigenvalues))
    positive = eigenvalues[sorted_idx] > 1e-10
    num_positive = np.sum(positive)
    actual_components = min(max_components, num_positive) if num_positive > 0 else max_components
    eigenvalues = eigenvalues[sorted_idx][:actual_components]
    eigenvectors = np.ascontiguousarray(eigenvectors_full[:, sorted_idx][:, :actual_components])
    del eigenvectors_full
    sqrt_eigenvals = np.sqrt(np.maximum(eigenvalues, 0))
    inv_sqrt = np.where(sqrt_eigenvals > 1e-10, 1.0 / sqrt_eigenvals, 0.0)
    landmark_coords = eigenvectors @ np.diag(sqrt_eigenvals)

    all_embeddings = np.zeros((n_nodes, components), dtype=np.float64)
    all_embeddings[landmark_indices, :actual_components] = landmark_coords[:, :actual_components]
    if actual_components < components:
        pass  # rest stays zero

    if n_non == 0:
        return all_embeddings

    non_landmark_coords = np.zeros((n_non, actual_components), dtype=np.float64)
    chunk_size = min(cross_chunk_size, n_non)
    n_chunks = (n_non + chunk_size - 1) // chunk_size
    t0 = time.time()
    next_pct = progress_pct_interval
    for chunk_idx, start in enumerate(range(0, n_non, chunk_size), start=1):
        end = min(start + chunk_size, n_non)
        cross_chunk = get_cross_dist_chunk(G_nk, landmark_indices, non_landmark_indices, start, end)
        if use_packed:
            B_chunk = double_center_cross_packed(packed_dist, k, row_sums_sq, cross_chunk)
        else:
            B_chunk = double_center_cross(landmark_dist, cross_chunk)
        del cross_chunk
        non_landmark_coords[start:end, :] = (B_chunk.T @ eigenvectors @ np.diag(inv_sqrt))[:, :actual_components]
        del B_chunk
        if progress_pct_interval and progress_pct_interval > 0:
            pct = 100 * (end / n_non)
            if pct >= next_pct or end == n_non:
                elapsed = time.time() - t0
                print(
                    f"  Non-landmark coordinates: chunk {chunk_idx}/{n_chunks} — {end}/{n_non} ({pct:.1f}%) "
                    f"[{elapsed:.1f}s]",
                    flush=True,
                )
                next_pct = min(100, (int(pct) // progress_pct_interval + 1) * progress_pct_interval)

    if actual_components <= components:
        all_embeddings[non_landmark_indices, :actual_components] = non_landmark_coords
    else:
        all_embeddings[non_landmark_indices] = non_landmark_coords[:, :components]
    return all_embeddings


def create_node_features_landmark_mds_from_graph(G_nk, num_features, landmark_ratio, landmark_selection, seed, device, dtype=torch.float32, max_memory_gb=None, use_full_distance_matrix=False, full_matrix_row_chunk_size=20000, use_chunked_cross_distances=False, cross_dist_chunk_size=50000):
    """LMDS from graph. When max_memory_gb is set, the four strategy options are chosen automatically (never fewer landmarks; cross_chunk >= MIN_CROSS_CHUNK_SIZE)."""
    if isinstance(device, str):
        device = torch.device(device)
    n_nodes = G_nk.numberOfNodes()
    num_landmarks = max(int(np.round(n_nodes * landmark_ratio)), min(num_features + 1, n_nodes))
    num_landmarks = min(num_landmarks, n_nodes)

    strategy = choose_lmds_memory_strategy(max_memory_gb, n_nodes, landmark_ratio, num_features)
    if strategy is not None:
        use_full_distance_matrix, full_matrix_row_chunk_size, use_chunked_cross_distances, cross_dist_chunk_size = strategy
        if full_matrix_row_chunk_size is not None:
            print(f"  [LMDS] Strategy: full_matrix_row_chunk_size={full_matrix_row_chunk_size}", flush=True)
        if use_chunked_cross_distances and cross_dist_chunk_size is not None:
            print(f"  [LMDS] Strategy: use_chunked_cross_distances=True, cross_dist_chunk_size={cross_dist_chunk_size}", flush=True)

    if use_full_distance_matrix or num_landmarks >= n_nodes:
        chunk = full_matrix_row_chunk_size or 50000
        print(f"  Using full distance matrix via chunked SPSP ({n_nodes}×{n_nodes}, row_chunk={chunk}), then classical MDS", flush=True)
        full_dist = get_distance_matrix_full(G_nk, row_chunk_size=chunk)
        print(f"  [MDS] Starting classical MDS (double-center + eigh)...", flush=True)
        all_embeddings = classical_mds_full_matrix(full_dist, num_features)
        print(f"  [MDS] Classical MDS done.", flush=True)
        del full_dist
    elif use_chunked_cross_distances:
        cross_chunk = cross_dist_chunk_size or 50000
        if max_memory_gb is not None and max_memory_gb > 0:
            C = MEMORY_SAFETY_FACTOR * max_memory_gb * (1024**3) / 8
            rad = 4 * cross_chunk * cross_chunk + 6 * C
            max_k = int((-2 * cross_chunk + np.sqrt(rad)) / 3) if rad >= 0 else 0
            if num_landmarks > max_k:
                raise ValueError(
                    f"Requested {num_landmarks} landmarks would exceed memory (max_k={max_k} for cross_dist_chunk_size={cross_chunk}, max_memory_gb={max_memory_gb}). "
                    "Increase max_memory_gb or decrease landmark_ratio."
                )

        landmark_indices, non_landmark_indices = select_landmarks_from_graph(
            G_nk, num_landmarks, landmark_selection, seed
        )
        print(f"  [LMDS] Chunked cross: computing landmark–landmark only (packed upper triangle, {num_landmarks}×{num_landmarks})...", flush=True)
        packed_dist, row_sums_sq = get_landmark_distances_landmarks_only(G_nk, landmark_indices, packed=True)
        print(f"  [LMDS] Landmark–landmark done. Computing non-landmark coordinates in column chunks (chunk={cross_chunk})...", flush=True)
        all_embeddings = lmds_from_landmark_matrices_chunked(
            G_nk, packed_dist, landmark_indices, non_landmark_indices,
            n_nodes, num_features, cross_chunk_size=cross_chunk, row_sums_sq=row_sums_sq
        )
    else:
        # Full cross matrix (k × n_non) can OOM: ~8 * num_landmarks * (n_nodes - num_landmarks) bytes.
        # Auto-switch to chunked path if that would exceed threshold (avoids OOM without requiring config).
        n_non = n_nodes - num_landmarks
        full_cross_gb = (num_landmarks * n_non * 8) / (1024**3)
        FULL_CROSS_AUTO_CHUNK_GB = 100.0
        if full_cross_gb > FULL_CROSS_AUTO_CHUNK_GB and not use_chunked_cross_distances:
            use_chunked_cross_distances = True
            cross_dist_chunk_size = cross_dist_chunk_size or 50000
            print(
                f"  [LMDS] Full landmark×non-landmark matrix would need ~{full_cross_gb:.0f} GB; "
                f"using chunked cross distances (chunk={cross_dist_chunk_size}) to avoid OOM.",
                flush=True,
            )
        if use_chunked_cross_distances:
            cross_chunk = cross_dist_chunk_size or 50000
            if max_memory_gb is not None and max_memory_gb > 0:
                C = MEMORY_SAFETY_FACTOR * max_memory_gb * (1024**3) / 8
                rad = 4 * cross_chunk * cross_chunk + 6 * C
                max_k = int((-2 * cross_chunk + np.sqrt(rad)) / 3) if rad >= 0 else 0
                if num_landmarks > max_k:
                    raise ValueError(
                        f"Requested {num_landmarks} landmarks would exceed memory (max_k={max_k} for cross_dist_chunk_size={cross_chunk}, max_memory_gb={max_memory_gb}). "
                        "Increase max_memory_gb or decrease landmark_ratio."
                    )
            landmark_indices, non_landmark_indices = select_landmarks_from_graph(
                G_nk, num_landmarks, landmark_selection, seed
            )
            print(f"  [LMDS] Chunked cross: computing landmark–landmark only (packed upper triangle, {num_landmarks}×{num_landmarks})...", flush=True)
            packed_dist, row_sums_sq = get_landmark_distances_landmarks_only(G_nk, landmark_indices, packed=True)
            print(f"  [LMDS] Landmark–landmark done. Computing non-landmark coordinates in column chunks (chunk={cross_chunk})...", flush=True)
            all_embeddings = lmds_from_landmark_matrices_chunked(
                G_nk, packed_dist, landmark_indices, non_landmark_indices,
                n_nodes, num_features, cross_chunk_size=cross_chunk, row_sums_sq=row_sums_sq
            )
        else:
            if max_memory_gb is not None and max_memory_gb > 0:
                C = MEMORY_SAFETY_FACTOR * max_memory_gb * (1024**3) / 8
                radicand = n_nodes * n_nodes - C
                if radicand <= 0:
                    max_k = max(2, int(C / (2 * n_nodes)))
                else:
                    max_k = int(n_nodes - np.sqrt(radicand))
                max_k = max(max_k, min(num_features + 1, n_nodes))
                if num_landmarks > max_k:
                    raise ValueError(
                        f"Requested {num_landmarks} landmarks would exceed memory (max_k={max_k} for full cross, max_memory_gb={max_memory_gb}). "
                        "Set use_chunked_cross_distances=true in config or increase max_memory_gb or decrease landmark_ratio."
                    )

            landmark_indices, non_landmark_indices = select_landmarks_from_graph(
                G_nk, num_landmarks, landmark_selection, seed
            )

            print(f"  [LMDS] Computing landmark distances ({num_landmarks} landmarks)...", flush=True)
            landmark_dist, landmark_to_non_landmark_dist = get_landmark_distances(
                G_nk, landmark_indices, non_landmark_indices
            )
            print(f"  [LMDS] Landmark distances done. Computing coordinates...", flush=True)

            all_embeddings = lmds_from_landmark_matrices(
                landmark_dist, landmark_to_non_landmark_dist,
                landmark_indices, non_landmark_indices, n_nodes, num_features
            )

    actual_components = all_embeddings.shape[1]
    if actual_components < num_features:
        print(f"  Warning: Requested {num_features} components but only {actual_components} available. Padding with zeros.", flush=True)
        padded = np.zeros((n_nodes, num_features), dtype=np.float64)
        padded[:, :actual_components] = all_embeddings
        all_embeddings = padded

    # Avoid duplicating ~(n × num_features) in RAM: use from_numpy when staying on CPU (shares memory).
    if device.type == "cpu":
        if dtype == torch.float32:
            all_embeddings = all_embeddings.astype(np.float32)  # one conversion; float64 can be freed
        return torch.from_numpy(all_embeddings)
    return torch.tensor(all_embeddings, device=device, dtype=dtype)


def compute_embedding_coordinates(landmark_kernel_matrix, landmark_to_non_landmark_dist, landmark_dist_matrix, n_components, non_landmark_chunk_size=50000):
    """Compute LMDS coordinates; non-landmark coords are computed in column chunks to avoid a full (k x n_non) B matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(landmark_kernel_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]

    max_components = min(n_components, len(eigenvalues))
    positive_eigenvals = eigenvalues[sorted_idx] > 1e-10
    num_positive = np.sum(positive_eigenvals)
    actual_components = min(max_components, num_positive) if num_positive > 0 else max_components
    
    eigenvalues = eigenvalues[sorted_idx][:actual_components]
    eigenvectors = eigenvectors[:, sorted_idx][:, :actual_components]
    sqrt_eigenvals = np.sqrt(np.maximum(eigenvalues, 0))
    
    landmark_coords = eigenvectors @ np.diag(sqrt_eigenvals)
    
    if landmark_to_non_landmark_dist.size > 0:
        n_non = landmark_to_non_landmark_dist.shape[1]
        inv_sqrt = np.where(sqrt_eigenvals > 1e-10, 1.0 / sqrt_eigenvals, 0.0)
        non_landmark_coords = np.zeros((n_non, actual_components), dtype=np.float64)
        chunk_size = min(non_landmark_chunk_size, n_non)
        print(
            f"Computing non-landmark coordinates in column chunks (chunk={chunk_size})...",
            flush=True,
        )
        n_chunks = (n_non + chunk_size - 1) // chunk_size
        t0 = time.time()
        for chunk_idx, start in enumerate(range(0, n_non, chunk_size), start=1):
            end = min(start + chunk_size, n_non)
            B_chunk = double_center_cross(landmark_dist_matrix, landmark_to_non_landmark_dist[:, start:end])
            non_landmark_coords[start:end, :] = (B_chunk.T @ eigenvectors @ np.diag(inv_sqrt))
            pct = 100 * (end / n_non)
            elapsed = time.time() - t0
            print(
                f"  Non-landmark coordinates: chunk {chunk_idx}/{n_chunks} — {end}/{n_non} ({pct:.1f}%) "
                f"[{elapsed:.1f}s]",
                flush=True,
            )
    else:
        non_landmark_coords = np.array([]).reshape(0, actual_components)
    
    return landmark_coords, non_landmark_coords


def create_node_features_landmark_mds(distance_matrix, num_features, landmark_ratio, landmark_selection, seed):
    """LMDS from full distance matrix."""
    n_nodes = distance_matrix.shape[0]
    device = distance_matrix.device
    dtype = distance_matrix.dtype
    dist_np = distance_matrix.detach().cpu().numpy()

    num_landmarks = max(int(np.round(n_nodes * landmark_ratio)), min(num_features + 1, n_nodes))
    num_landmarks = min(num_landmarks, n_nodes)
    
    if num_landmarks >= n_nodes:
        landmark_indices = np.arange(n_nodes)
        non_landmark_indices = np.array([], dtype=np.int64)
    else:
        landmark_indices, non_landmark_indices = select_landmarks(
            dist_np, num_landmarks,             landmark_selection, seed
        )

    landmark_dist = dist_np[np.ix_(landmark_indices, landmark_indices)]
    landmark_kernel = double_center(landmark_dist)
    
    if len(non_landmark_indices) > 0:
        landmark_to_non_landmark_dist = dist_np[np.ix_(landmark_indices, non_landmark_indices)]
    else:
        landmark_to_non_landmark_dist = np.array([]).reshape(len(landmark_indices), 0)

    landmark_coords, non_landmark_coords = compute_embedding_coordinates(
        landmark_kernel, landmark_to_non_landmark_dist, landmark_dist, num_features
    )

    actual_components = landmark_coords.shape[1]

    all_embeddings = np.zeros((n_nodes, num_features))
    if actual_components <= num_features:
        all_embeddings[landmark_indices, :actual_components] = landmark_coords
        if len(non_landmark_indices) > 0:
            all_embeddings[non_landmark_indices, :actual_components] = non_landmark_coords
    else:
        all_embeddings[landmark_indices] = landmark_coords[:, :num_features]
        if len(non_landmark_indices) > 0:
            all_embeddings[non_landmark_indices] = non_landmark_coords[:, :num_features]
    
    if actual_components < num_features:
        print(f"  Warning: Requested {num_features} components but only {actual_components} available. Padding with zeros.", flush=True)
    
    return torch.tensor(all_embeddings, device=device, dtype=dtype)

# --- Training ---

def train_encoder(model, distance_matrix, train_i, train_j, config, val_i=None, val_j=None,
                  input_data=None, train_orig_dists=None, val_orig_dists=None):
    """Train encoder with distance preservation; supports 1D pair distances or full matrix."""
    training_config = config.get('training')
    model_config = config.get('model')
    system_config = config.get('system')

    epochs = int(training_config.get('epochs'))
    lr = float(training_config.get('learning_rate'))
    patience = int(training_config.get('patience'))
    weight_decay = float(training_config.get('weight_decay'))
    eps = float(training_config.get('eps'))
    max_norm = float(training_config.get('max_norm'))
    eta_min = float(training_config.get('eta_min'))
    num_features = int(model_config.get('num_features'))
    pair_batch_size = training_config.get('pair_batch_size')
    pair_batch_size = int(pair_batch_size) if pair_batch_size not in (None, "", 0) else 0
    device = system_config.get('device')

    model = model.to(device)
    use_1d_dists = train_orig_dists is not None
    n_train = len(train_i)
    use_batched_pairs = pair_batch_size > 0 and n_train > pair_batch_size
    use_validation = val_i is not None and val_j is not None and (val_orig_dists is not None or not use_1d_dists)

    if use_batched_pairs:
        train_i_cpu = train_i.cpu()
        train_j_cpu = train_j.cpu()
        train_orig_dists_cpu = train_orig_dists.cpu() if use_1d_dists else None
        if not use_1d_dists:
            if distance_matrix is None:
                raise ValueError("Either distance_matrix or train_orig_dists must be provided")
            distance_matrix = distance_matrix.to(device)
        if use_validation:
            val_i_cpu = val_i.cpu()
            val_j_cpu = val_j.cpu()
            val_orig_dists_cpu = val_orig_dists.cpu() if use_1d_dists and val_orig_dists is not None else None
    else:
        if use_1d_dists:
            train_orig_dists = train_orig_dists.to(device)
            if val_orig_dists is not None:
                val_orig_dists = val_orig_dists.to(device)
        else:
            if distance_matrix is None:
                raise ValueError("Either distance_matrix or train_orig_dists must be provided")
            distance_matrix = distance_matrix.to(device)
        train_i = train_i.to(device)
        train_j = train_j.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*eta_min)

    if input_data is None:
        if distance_matrix is None:
            raise ValueError("input_data or distance_matrix required for feature generation")
        # Generate features using Landmark MDS (PCA was removed)
        landmark_ratio = float(config['input'].get('landmark_ratio', 0.3))
        landmark_selection = config['input'].get('landmark_selection', 'random')
        seed = config.get('_seed')
        input_data = create_node_features_landmark_mds(
            distance_matrix, num_features=num_features,
            landmark_ratio=landmark_ratio,
            landmark_selection=landmark_selection,
            seed=seed
        )
    else:
        input_data = input_data.to(device)

    model.train()
    train_losses = []
    val_losses = [] if use_validation else None
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        if use_batched_pairs:
            # True mini-batch: forward only on nodes in this batch, backward + step per batch
            total_train_loss_val = 0.0
            for start in range(0, n_train, pair_batch_size):
                end = min(start + pair_batch_size, n_train)
                batch_i = train_i_cpu[start:end].to(device)
                batch_j = train_j_cpu[start:end].to(device)
                if use_1d_dists:
                    batch_orig = train_orig_dists_cpu[start:end].to(device)
                else:
                    batch_orig = distance_matrix[batch_i, batch_j]
                # Unique nodes in this batch -> forward only on these nodes
                combined = torch.cat([batch_i, batch_j])
                batch_nodes, inverse = torch.unique(combined, return_inverse=True)
                batch_i_local = inverse[:batch_i.size(0)]
                batch_j_local = inverse[batch_i.size(0):]
                batch_features = input_data[batch_nodes]
                batch_embeddings = model(batch_features)
                batch_emb_d = torch.norm(
                    batch_embeddings[batch_i_local] - batch_embeddings[batch_j_local], dim=1, p=2
                )
                batch_loss = normalized_stress_loss(batch_orig, batch_emb_d)
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
                total_train_loss_val += batch_loss.item() * (end - start) / n_train
            train_loss = torch.tensor(total_train_loss_val, device=device)
        else:
            embeddings = model(input_data)
            if use_1d_dists:
                train_orig_d = train_orig_dists
            else:
                train_orig_d = distance_matrix[train_i, train_j]
            train_emb_dists = torch.norm(embeddings[train_i] - embeddings[train_j], dim=1, p=2)
            train_loss = normalized_stress_loss(train_orig_d, train_emb_dists)
            optimizer.zero_grad()
            train_loss.backward()
        if not use_batched_pairs:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        scheduler.step()
        train_losses.append(train_loss.item())

        if use_validation:
            model.eval()
            with torch.no_grad():
                if use_batched_pairs:
                    # Full forward for validation (no backward)
                    embeddings = model(input_data)
                    n_val = len(val_i_cpu)
                    total_val_loss = 0.0
                    for start in range(0, n_val, pair_batch_size):
                        end = min(start + pair_batch_size, n_val)
                        batch_i = val_i_cpu[start:end].to(device)
                        batch_j = val_j_cpu[start:end].to(device)
                        if use_1d_dists and val_orig_dists_cpu is not None:
                            batch_orig = val_orig_dists_cpu[start:end].to(device)
                        else:
                            batch_orig = distance_matrix[batch_i, batch_j]
                        batch_emb_d = torch.norm(embeddings[batch_i] - embeddings[batch_j], dim=1, p=2)
                        w = (end - start) / n_val
                        total_val_loss = total_val_loss + normalized_stress_loss(batch_orig, batch_emb_d).item() * w
                    val_losses.append(total_val_loss)
                    current_loss = total_val_loss
                else:
                    if use_1d_dists:
                        val_orig_d = val_orig_dists
                    else:
                        val_orig_d = distance_matrix[val_i, val_j]
                    val_loss = normalized_stress_loss(
                        val_orig_d,
                        torch.norm(embeddings[val_i] - embeddings[val_j], dim=1, p=2)
                    )
                    val_losses.append(val_loss.item())
                    current_loss = val_loss.item()
            model.train()
        else:
            current_loss = train_loss.item()

        if epoch % 1000 == 0:
            msg = f"Epoch {epoch}: Train Loss = {train_loss.item():.6f}"
            if use_validation:
                msg += f", Val Loss = {current_loss:.6f}"
            print(msg, flush=True)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience > 0 and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})", flush=True)
            print(f"Best model found at epoch {best_epoch} with loss {best_loss:.6f}", flush=True)
            print(f"Current loss: {current_loss:.6f} (no improvement for {patience} epochs)", flush=True)
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None and patience_counter < patience:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

# --- Save / device ---

def save_results(embeddings, node_labels, npz_file, metrics, csv_file=None):
    """Write CSV (and optionally NPZ) with embeddings and metrics."""
    if csv_file is None:
        if npz_file is None:
            raise ValueError("Either csv_file or npz_file must be provided")
        csv_file = npz_file.rsplit('.', 1)[0] + '.csv'
    
    formatted_coords = [":".join(map(str, coords)) for coords in embeddings]
    df = pd.DataFrame({
        'node': node_labels,
        'coordinates': formatted_coords
    })
    df.to_csv(csv_file, index=False)
    print(f"Saved embeddings to {csv_file}", flush=True)
    
    if npz_file is not None:
        np.savez_compressed(npz_file, coords=embeddings, **metrics)
        print(f"Saved metrics to {npz_file}", flush=True)

def setup_device(device_str):
    """Resolve 'auto' to a GPU (cuda/xpu/mps) or CPU. Falls back to CPU when no GPU available."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device (Nvidia GPU)", flush=True)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
            print(f"Using XPU device (Intel GPU)", flush=True)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS device (Apple Silicon GPU)", flush=True)
        else:
            device = "cpu"
            print("Using CPU (no GPU available)", flush=True)
    else:
        device = device_str
        if device.lower() == "cpu" or device.lower() == "mkl":
            device = "cpu"
            print("Using CPU device", flush=True)
        elif device.lower() == "gpu":
            # PyTorch uses "cuda" for NVIDIA GPU, not "gpu"
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but CUDA not available. Encoder requires a GPU.")
            device = "cuda"
            print(f"Using CUDA device (Nvidia GPU)", flush=True)
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available. Encoder requires a GPU.")
            device = "cuda"
            print(f"Using CUDA device (Nvidia GPU)", flush=True)
        elif device.startswith("xpu"):
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                raise RuntimeError("XPU requested but not available. Encoder requires a GPU.")
            device = "xpu"
            print(f"Using XPU device (Intel GPU)", flush=True)
        elif device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("MPS requested but not available. Encoder requires a GPU.")
            print("Using MPS device (Apple Silicon GPU)", flush=True)

    print(f"Using device: {device}", flush=True)

    if device == "cpu":
        slurm_cpus = None  # prefer SLURM_* if set

        if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
            slurm_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            print(f"Detected SLURM: Using SLURM_JOB_CPUS_PER_NODE={slurm_cpus}", flush=True)
        elif 'SLURM_CPUS_ON_NODE' in os.environ:
            slurm_cpus = int(os.environ['SLURM_CPUS_ON_NODE'])
            print(f"Detected SLURM: Using SLURM_CPUS_ON_NODE={slurm_cpus}", flush=True)
        elif 'SLURM_CPUS_PER_TASK' in os.environ:
            slurm_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
            print(f"Detected SLURM: Using SLURM_CPUS_PER_TASK={slurm_cpus}", flush=True)

        if slurm_cpus and slurm_cpus > 0:
            omp_threads = os.environ.get('OMP_NUM_THREADS')
            if omp_threads and int(omp_threads) == 1:
                torch.set_num_threads(slurm_cpus)
                print(f"Set PyTorch CPU threads to {slurm_cpus} (from SLURM allocation)", flush=True)
            elif not omp_threads:
                torch.set_num_threads(slurm_cpus)
                print(f"Set PyTorch CPU threads to {slurm_cpus} (from SLURM allocation)", flush=True)
            else:
                print(f"Using OMP_NUM_THREADS={omp_threads} (SLURM allocated {slurm_cpus} CPUs)", flush=True)

        num_threads = torch.get_num_threads()
        num_interop_threads = torch.get_num_interop_threads()
        print(f"PyTorch CPU threads: {num_threads} (intra-op), {num_interop_threads} (inter-op)", flush=True)

    return device


def prepare_node_features(config, distance_tensor, node_labels, output_name, device):
    """Load CSV, random, or LMDS features; return (tensor, input_dim)."""
    node_features_file = config['input'].get('node_features_file')
    if node_features_file == "":
        node_features_file = None
    
    initialization_method = config['input'].get('initialization_method')
    num_features = int(config['model'].get('num_features'))
    save = config['system']['save']
    save_node_features = config['system'].get('save_node_features')
    
    if node_features_file is not None:
        print(f"Loading node features from: {node_features_file}", flush=True)
        features_array = load_node_features_from_csv(node_features_file, node_labels)
        sample_features = torch.tensor(features_array, device=device, dtype=torch.float32)
        input_dim = sample_features.shape[1]
        print(f"Loaded node features from CSV: {input_dim} features per node", flush=True)
        print(f"Features loaded for {len(node_labels)} nodes", flush=True)
    elif initialization_method == 'random':
        print(f"Generating random node features (dimension: {num_features})", flush=True)
        n_nodes = distance_tensor.shape[0]
        seed = config.get('_seed')
        generator = get_seeded_generator(seed, device, offset=1) if seed is not None else None
        if generator is not None:
            sample_features = torch.randn(n_nodes, num_features, device=device, dtype=torch.float32, generator=generator)
        else:
            sample_features = torch.randn(n_nodes, num_features, device=device, dtype=torch.float32)
        input_dim = num_features
        print(f"Generated random features for {n_nodes} nodes", flush=True)
    else:  # landmark_mds (from full distance matrix)
        print(f"Generating node features using Landmark MDS (dimension: {num_features})", flush=True)
        seed = config.get('_seed')
        landmark_selection = config['input'].get('landmark_selection', 'maxmin')
        print(f"  Landmark selection method: {landmark_selection}", flush=True)
        sample_features = create_node_features_landmark_mds(
            distance_tensor,
            num_features=num_features,
            landmark_ratio=float(config['input'].get('landmark_ratio')),
            landmark_selection=landmark_selection,
            seed=seed
        )
        input_dim = sample_features.shape[1]
        print(f"Node features dimension: {input_dim} (Landmark MDS)", flush=True)
    
    if save and save_node_features:
        features_np = sample_features.cpu().numpy()
        formatted_features = [":".join(map(str, features)) for features in features_np]
        features_df = pd.DataFrame({
            'node': node_labels,
            'coordinates': formatted_features
        })
        features_csv = f"{output_name}_node_features.csv"
        features_df.to_csv(features_csv, index=False)
        print(f"Saved node features to {features_csv}", flush=True)
    
    return sample_features, input_dim


def split_pairs(all_i, all_j, val_ratio, test_ratio, device, seed=None):
    """Train/val/test split. Shuffles on CPU to avoid GPU OOM with large pair counts."""
    n_samples = len(all_i)
    train_ratio = 1.0 - val_ratio - test_ratio

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio) if val_ratio > 0.0 else train_end

    # Move to CPU for shuffling to avoid GPU OOM with millions of pairs
    all_i_cpu = all_i.cpu()
    all_j_cpu = all_j.cpu()
    
    # Shuffle on CPU using numpy (more memory efficient for huge arrays)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    all_i_shuffled = all_i_cpu[perm]
    all_j_shuffled = all_j_cpu[perm]

    # Split on CPU, then move only the splits to GPU
    train_i = all_i_shuffled[:train_end].to(device)
    train_j = all_j_shuffled[:train_end].to(device)

    val_i = val_j = None
    if val_ratio > 0.0 and train_end < val_end:
        val_i = all_i_shuffled[train_end:val_end].to(device)
        val_j = all_j_shuffled[train_end:val_end].to(device)

    test_i = test_j = None
    if test_ratio > 0.0 and val_end < n_samples:
        test_i = all_i_shuffled[val_end:].to(device)
        test_j = all_j_shuffled[val_end:].to(device)

    print(f"  Split {n_samples} sampled pairs: train={len(train_i)}, val={len(val_i) if val_i is not None else 0}, test={len(test_i) if test_i is not None else 0}", flush=True)
    return train_i, train_j, val_i, val_j, test_i, test_j


def compute_test_loss(model, distance_tensor, test_i, test_j, sample_features, test_orig_dists=None, pair_batch_size=0):
    """Normalized stress on test pairs; uses test_orig_dists if provided. Batched when pair_batch_size > 0."""
    if test_i is None or test_j is None or len(test_i) == 0:
        return None
    model.eval()
    device = next(model.parameters()).device
    n_test = len(test_i)
    use_batch = pair_batch_size > 0 and n_test > pair_batch_size

    with torch.no_grad():
        embeddings = model(sample_features)
        if use_batch:
            test_i_cpu = test_i.cpu()
            test_j_cpu = test_j.cpu()
            test_orig_cpu = test_orig_dists.cpu() if test_orig_dists is not None else None
            total_loss = 0.0
            for start in range(0, n_test, pair_batch_size):
                end = min(start + pair_batch_size, n_test)
                batch_i = test_i_cpu[start:end].to(device)
                batch_j = test_j_cpu[start:end].to(device)
                if test_orig_cpu is not None:
                    batch_orig = test_orig_cpu[start:end].to(device)
                else:
                    batch_orig = distance_tensor[batch_i, batch_j]
                batch_emb_d = torch.norm(embeddings[batch_i] - embeddings[batch_j], dim=1, p=2)
                w = (end - start) / n_test
                total_loss += normalized_stress_loss(batch_orig, batch_emb_d).item() * w
            return total_loss
        if test_orig_dists is not None:
            test_orig_d = test_orig_dists
        else:
            test_orig_d = distance_tensor[test_i, test_j]
        test_emb_dists = torch.norm(embeddings[test_i] - embeddings[test_j], dim=1, p=2)
        return normalized_stress_loss(test_orig_d, test_emb_dists).item()


def save_all_results(config, output_name, embeddings, node_labels, save_npz, node_ids=None):
    """Write CSV, optional NPZ, and config TOML (embeddings only). If node_ids given, save in npz for alignment in get_metrics."""
    csv_file = f"{output_name}.csv"
    npz_file = f"{output_name}.npz"
    toml_file = f"{output_name}_config.toml"
    
    save_results(embeddings, node_labels, None, {}, csv_file)
    
    if save_npz:
        if node_ids is not None:
            np.savez_compressed(npz_file, coords=embeddings, node_ids=np.array(node_ids, dtype=object))
        else:
            np.savez_compressed(npz_file, coords=embeddings)
        print(f"Saved embeddings to {npz_file}", flush=True)
    
    save_config_toml(config, toml_file)
    print(f"Saved configuration to {toml_file}", flush=True)


def main():
    """Load config, build graph/features, train, evaluate, save."""
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}", flush=True)
    config = merge_configs(get_default_config(), load_config(args.config))

    seed = normalize_seed(config['system'].get('seed'))

    input_file = config['input']['input_file']
    if input_file is None:
        raise ValueError("'input_file' must be specified in [input] section of TOML file")

    output_name = resolve_output_name(config)
    print(f"Using output name: {output_name}", flush=True)

    device = setup_device(config['system']['device'])
    config['system']['device'] = device

    if seed is not None:
        set_random_seeds(seed, device=device)
        config['_seed'] = seed
    else:
        config['_seed'] = None

    timings = {}
    time_after_read_start = None

    print("\n=== Step 1: Loading graph (and optionally distance matrix) from GraphML.bz2 ===", flush=True)
    print(f"Loading graph from: {input_file}", flush=True)
    start_time = time.time()
    weight_attr = config['input']['weight_attr']
    weight_attr_value = weight_attr.strip() if (weight_attr and isinstance(weight_attr, str) and weight_attr.strip()) else None
    graph_nx, G_nk, mapping, node_ids = get_graph_and_nk(input_file, cores=config['input']['cores'], w=weight_attr_value)
    n_nodes = G_nk.numberOfNodes()
    use_random_sampling = True  # always random sampling
    use_no_full_matrix = (config['input'].get('initialization_method') == 'landmark_mds' and use_random_sampling)
    if use_no_full_matrix:
        distance_compact = None
        distance_tensor = None
        print(f"Graph loaded: {n_nodes} nodes (full distance matrix not built)", flush=True)
    else:
        distance_compact, _ = get_distance_matrix_compact(G_nk)
        distance_tensor = None
        print(f"Generated distance matrix (compact symmetric): {len(distance_compact)} entries (n*(n-1)/2)", flush=True)
    timings["load_graph"] = time.time() - start_time
    time_after_read_start = time.time()
    print(f"Number of nodes: {len(node_ids)}", flush=True)
    print("Step 1 done.", flush=True)

    print("\n=== Step 2: Extracting node labels ===", flush=True)
    start_time = time.time()
    node_labels = get_nodes_graphml(input_file, node_ids)
    timings["extract_node_labels"] = time.time() - start_time
    print(f"Extracted {len(node_labels)} node labels", flush=True)
    print("Step 2 done.", flush=True)

    test_ratio = float(config['training']['test_ratio'])
    if test_ratio > 0.0:
        print(f"\n=== Test set: {test_ratio*100:.1f}% of pairs will be reserved for testing ===", flush=True)

    print("\n=== Step 3: Preparing data for embedding ===", flush=True)
    start_time = time.time()
    initialization_method = config['input'].get('initialization_method')
    node_features_file = config['input'].get('node_features_file')
    if isinstance(node_features_file, str) and not node_features_file.strip():
        node_features_file = None
    config['input']['node_features_file'] = node_features_file

    if node_features_file is not None:
        # If a node features CSV is explicitly provided, always load it
        # (does not require any distance matrix, so distance_tensor can be None).
        sample_features, input_dim = prepare_node_features(
            config, None, node_labels, output_name, device
        )
    elif initialization_method == 'landmark_mds':
        num_features = int(config['model'].get('num_features'))
        landmark_ratio = float(config['input'].get('landmark_ratio'))
        landmark_selection = config['input'].get('landmark_selection', 'maxmin')
        seed = config.get('_seed')
        print(f"Generating node features using Landmark MDS from graph (space-efficient LMDS)", flush=True)
        print(f"  Landmark selection method: {landmark_selection}", flush=True)
        sample_features = create_node_features_landmark_mds_from_graph(
            G_nk, num_features=num_features, landmark_ratio=landmark_ratio,
            landmark_selection=landmark_selection, seed=seed, device=device,
            max_memory_gb=config['input'].get('max_memory_gb'),
            use_full_distance_matrix=config['input'].get('use_full_distance_matrix', False),
            full_matrix_row_chunk_size=config['input'].get('full_matrix_row_chunk_size', 20000),
            use_chunked_cross_distances=config['input'].get('use_chunked_cross_distances', False),
            cross_dist_chunk_size=config['input'].get('cross_dist_chunk_size', 50000)
        )
        input_dim = sample_features.shape[1]
        print(f"Node features dimension: {input_dim} (Landmark MDS from graph)", flush=True)
        if config['system']['save'] and config['system'].get('save_node_features'):
            features_np = sample_features.cpu().numpy()
            formatted_features = [":".join(map(str, features)) for features in features_np]
            features_df = pd.DataFrame({'node': node_labels, 'coordinates': formatted_features})
            features_csv = f"{output_name}_node_features.csv"
            features_df.to_csv(features_csv, index=False)
            print(f"Saved node features to {features_csv}", flush=True)
    else:
        if distance_compact is not None:
            full_for_pca = expand_compact_to_full(distance_compact, n_nodes)
            distance_tensor = torch.FloatTensor(full_for_pca).to(device)
        sample_features, input_dim = prepare_node_features(config, distance_tensor, node_labels, output_name, device)
    timings["node_features"] = time.time() - start_time
    print("Step 3 (node features) done.", flush=True)

    print("\n=== Step 4: Creating encoder model ===", flush=True)
    start_time = time.time()
    model = DistancePreservingEncoder(config)
    timings["create_model"] = time.time() - start_time
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters", flush=True)

    print("\n=== Sampling distance pairs ===", flush=True)
    start_time = time.time()
    if use_no_full_matrix:
        all_i, all_j = sample_distance_pairs_without_matrix(G_nk, n_nodes, config, device)
    else:
        all_i, all_j = sample_distance_pairs(distance_compact, config, n=n_nodes, device=device)
    seed_int = config.get('_seed')
    train_i, train_j, val_i, val_j, test_i, test_j = split_pairs(
        all_i, all_j, float(config['training']['val_ratio']), test_ratio, device, seed=seed_int
    )
    train_dists = val_dists = test_dists = None
    if use_no_full_matrix:
        print("Computing graph distances for sampled pairs (one-time)...", flush=True)
        train_dists, val_dists, test_dists = get_pair_distances_from_graph(
            G_nk, train_i, train_j, val_i, val_j, test_i, test_j, device
        )
        print("Pair distances computed.", flush=True)
    else:
        train_dists = torch.tensor(
            get_distances_for_pairs(distance_compact, n_nodes, train_i.cpu().numpy(), train_j.cpu().numpy()),
            dtype=torch.float32, device=device
        )
        if val_i is not None and len(val_i) > 0:
            val_dists = torch.tensor(
                get_distances_for_pairs(distance_compact, n_nodes, val_i.cpu().numpy(), val_j.cpu().numpy()),
                dtype=torch.float32, device=device
            )
        else:
            val_dists = None
        if test_i is not None and len(test_i) > 0:
            test_dists = torch.tensor(
                get_distances_for_pairs(distance_compact, n_nodes, test_i.cpu().numpy(), test_j.cpu().numpy()),
                dtype=torch.float32, device=device
            )
        else:
            test_dists = None
    timings["sample_and_pair_distances"] = time.time() - start_time

    print("\n=== Step 5: Training encoder ===", flush=True)
    start_time = time.time()
    model, train_losses, val_losses = train_encoder(
        model, None, train_i, train_j, config, val_i=val_i, val_j=val_j, input_data=sample_features,
        train_orig_dists=train_dists, val_orig_dists=val_dists
    )
    timings["training"] = time.time() - start_time

    pair_batch_size = config['training'].get('pair_batch_size')
    pair_batch_size = int(pair_batch_size) if pair_batch_size not in (None, "", 0) else 0
    start_time = time.time()
    test_loss = compute_test_loss(
        model, None, test_i, test_j, sample_features,
        test_orig_dists=test_dists,
        pair_batch_size=pair_batch_size
    )
    timings["test_eval"] = time.time() - start_time
    print("\n=== Final Distance Losses ===", flush=True)
    if len(train_losses) > 0:
        print(f"Training Loss: {train_losses[-1]:.6f}", flush=True)
    if val_losses is not None and len(val_losses) > 0:
        print(f"Validation Loss: {val_losses[-1]:.6f}", flush=True)
    if test_loss is not None:
        print(f"Test Loss: {test_loss:.6f}", flush=True)

    print("\n=== Step 6: Encode and save embedding ===", flush=True)
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(sample_features.to(device))
    embeddings_np = embeddings.cpu().numpy()
    timings["encode"] = time.time() - start_time

    if config['system']['save']:
        save_all_results(config, output_name, embeddings_np, node_labels, config['system'].get('save_npz', True), node_ids=node_ids)

    # After-read-to-embed: from right after load_graph through to having embeddings
    timings["after_read_to_embed"] = time.time() - time_after_read_start
    main_keys = ["extract_node_labels", "node_features", "create_model", "sample_and_pair_distances", "training", "test_eval", "encode"]
    timings["total"] = timings["load_graph"] + sum(timings.get(k, 0) for k in main_keys)

    # Print timing summary
    print("\n" + "=" * 60, flush=True)
    print("TIMING SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Number of nodes: {n_nodes}", flush=True)
    print(f"Embedding shape: {embeddings_np.shape}", flush=True)
    print("\nTimings (seconds):", flush=True)
    print(f"  Load graph:              {timings['load_graph']:>12.4f}", flush=True)
    print(f"  Extract node labels:     {timings['extract_node_labels']:>12.4f}", flush=True)
    print(f"  Node features (LMDS):    {timings['node_features']:>12.4f}", flush=True)
    print(f"  Create model:            {timings['create_model']:>12.4f}", flush=True)
    print(f"  Sample + pair distances: {timings['sample_and_pair_distances']:>12.4f}", flush=True)
    print(f"  Training:                {timings['training']:>12.4f}", flush=True)
    print(f"  Test evaluation:         {timings['test_eval']:>12.4f}", flush=True)
    print(f"  Encode:                  {timings['encode']:>12.4f}", flush=True)
    print(f"  {'-' * 40}", flush=True)
    print(f"  After load → embed:      {timings['after_read_to_embed']:>12.4f}  (comparable across methods)", flush=True)
    print(f"  TOTAL:                   {timings['total']:>12.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
