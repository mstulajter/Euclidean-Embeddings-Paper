#!/usr/bin/env python3

"""
Find reaction paths: Dijkstra for lowest-cost path, then BFS up to cutoff depth
(lowest path length + DELTA).

Authors:
    Miko Stulajter

Version 2.0.0

Usage: reaction_path_finder.py [-h] -gfile GRAPHML_FILE -r REACTANTS -p PRODUCTS [--delta DELTA]
"""

import argparse
import bz2
import itertools
import os
import random
import re
import sys
from collections import defaultdict, deque
import networkx as nx
try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import rdMolDescriptors

    RDLogger.DisableLog("rdApp.warning")
except Exception:
    Chem = None
    RDLogger = None
    rdMolDescriptors = None


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a molecular formula string (e.g. 'C3H3N3') into element counts."""
    counts: dict[str, int] = defaultdict(int)
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
        elem, num = match.group(1), match.group(2)
        counts[elem] += int(num) if num else 1
    return dict(counts)


def _merge_formulas(formulas: list[dict[str, int]]) -> str:
    """Merge formula dicts and return a single formula string (C first, then H, then alphabetical)."""
    total: dict[str, int] = defaultdict(int)
    for f in formulas:
        for elem, count in f.items():
            total[elem] += count
    order = ["C", "H"] + sorted(k for k in total if k not in ("C", "H"))
    parts = []
    for elem in order:
        if elem in total and total[elem] > 0:
            parts.append(f"{elem}{total[elem]}" if total[elem] > 1 else elem)
    return "".join(parts)


def molecular_formula_from_smiles(smiles: str) -> str:
    """Return combined molecular formula for (multi-mol) SMILES, or 'run' on failure."""
    if Chem is None or rdMolDescriptors is None:
        return "run"
    parts = [p.strip() for p in smiles.split(".")]
    formulas = []
    for part in parts:
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return "run"
        formulas.append(_parse_formula(rdMolDescriptors.CalcMolFormula(mol)))
    if not formulas:
        return "run"
    return _merge_formulas(formulas)


def make_output_path(rn: str, delta: int) -> str:
    """Return filename: {RN}_reaction_path_delta{num}_{8 digit unique tag}.txt"""
    tag = random.randint(10_000_000, 99_999_999)
    return f"{rn}_reaction_path_delta{delta}_{tag}.txt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find lowest-cost path (Dijkstra), then all paths within cutoff depth (BFS)."
    )
    parser.add_argument("-gfile", help="GraphML file (plain or .bz2)", type=str, required=True)
    parser.add_argument("-r", help="Reactants (SMILES)", type=str, required=True)
    parser.add_argument("-p", help="Products (SMILES)", type=str, required=True)
    parser.add_argument(
        "--delta",
        help="Cutoff depth = shortest path length + DELTA (default: 0)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hide-paths",
        help="Hide Path1, Path2, etc. labels in output",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Print verbose output",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def canonicalize_smiles(smiles: str) -> str | None:
    if Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_multimol_smiles(smiles: str) -> str | None:
    parts = smiles.split('.')
    canonical_parts = []
    for part in parts:
        canonical = canonicalize_smiles(part.strip())
        if canonical is None:
            return None
        canonical_parts.append(canonical)
    canonical_parts.sort()
    return '.'.join(canonical_parts)


def are_equivalent_smiles(smiles1: str, smiles2: str) -> bool:
    canon1 = canonicalize_multimol_smiles(smiles1)
    canon2 = canonicalize_multimol_smiles(smiles2)
    if canon1 is None or canon2 is None:
        return False
    return canon1 == canon2


def find_node_for_label(node_labels: dict, label: str) -> str:
    if label in node_labels.values():
        for node, lbl in node_labels.items():
            if lbl == label:
                return node
    label_permutations = {'.'.join(p) for p in itertools.permutations(label.split('.'))}
    for perm in label_permutations:
        node = next((node for node, lbl in node_labels.items() if lbl == perm), None)
        if node is not None:
            return node
    if canonicalize_multimol_smiles(label):
        for node, lbl in node_labels.items():
            if are_equivalent_smiles(label, lbl):
                return node
    print(f"ERROR: Label '{label}' not found in the graph.", file=sys.stderr)
    sys.exit(1)


def get_edge_weight(edge_data: dict, weight_key: str | None) -> float:
    if weight_key:
        weight = edge_data.get(weight_key)
        if weight is not None:
            try:
                weight_val = float(weight)
                if weight_val == weight_val and weight_val != float('inf') and weight_val != float('-inf'):
                    return weight_val
            except (ValueError, TypeError):
                pass
    return 1.0


def compute_path_cost(path_nodes: list, G: nx.Graph, weight_key: str | None) -> float:
    total_cost = 0.0
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge_data = G.edges[u, v]
        total_cost += get_edge_weight(edge_data, weight_key)
    return total_cost


def check_weight_key(G: nx.Graph) -> str | None:
    if G.number_of_edges() == 0:
        return None
    for _, _, edge_data in G.edges(data=True):
        if 'weight' in edge_data:
            try:
                float(edge_data['weight'])
                return 'weight'
            except (ValueError, TypeError):
                pass
    return None


def check_rule_key(G: nx.Graph) -> str | None:
    if G.number_of_edges() == 0:
        return None
    for _, _, edge_data in G.edges(data=True):
        if 'rule' in edge_data:
            return 'rule'
    return None


def crop_graph_to_max_length(
    G: nx.Graph,
    source: str,
    target: str,
    max_length: int,
) -> nx.Graph:
    source_nodes = set()
    if source in G:
        source_distances = nx.single_source_shortest_path_length(G, source, cutoff=max_length)
        source_nodes = set(source_distances.keys())
    target_nodes = set()
    if target in G:
        target_distances = nx.single_source_shortest_path_length(G, target, cutoff=max_length)
        target_nodes = set(target_distances.keys())
    relevant_nodes = source_nodes | target_nodes
    return G.subgraph(relevant_nodes).copy()


def bfs_paths_up_to_length(
    G: nx.Graph,
    source: str,
    target: str,
    max_length: int,
    weight_key: str | None,
    node_labels: dict,
    verbose: bool = False,
) -> list[tuple[float, str, list]]:
    """
    Enumerate all simple paths from source to target with number of edges <= max_length
    using BFS. Returns list of (cost, path_string, path_nodes) sorted by cost.
    """
    path_costs: list[tuple[float, str, list]] = []
    seen_paths: set[tuple] = set()  # (tuple of node ids) for deduplication
    queue: deque[tuple[str, tuple]] = deque([(source, (source,))])
    paths_found = 0

    while queue:
        node, path_tuple = queue.popleft()
        path_list = list(path_tuple)
        steps = len(path_list) - 1

        if node == target:
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                cost = compute_path_cost(path_list, G, weight_key)
                path_smiles = [node_labels[n] for n in path_list]
                path_str = " --> ".join(path_smiles)
                path_costs.append((cost, path_str, path_list))
                paths_found += 1
                if verbose and paths_found % 100 == 0:
                    print(f"  BFS found {paths_found} paths so far...", flush=True)
            continue

        if steps >= max_length:
            continue

        path_set = set(path_list)
        for v in G.neighbors(node):
            if v in path_set:
                continue
            queue.append((v, path_tuple + (v,)))

    path_costs.sort(key=lambda x: (x[0], x[1]))
    return path_costs


def extract_edge_info(
    G: nx.Graph,
    path_nodes: list,
    rule_key: str | None,
    weight_key: str | None,
) -> list[dict]:
    edges = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edge_data = G.edges[u, v]
        rule = edge_data.get(rule_key, '') if rule_key else ''
        weight = get_edge_weight(edge_data, weight_key)
        edges.append({'rule': rule, 'weight': weight})
    return edges


def analyze_paths(
    G: nx.Graph,
    path_costs: list[tuple[float, str, list]],
    rule_key: str | None,
    weight_key: str | None,
) -> list[dict] | None:
    if not path_costs:
        return None
    cost_groups = defaultdict(list)
    for cost, path_str, path_nodes in path_costs:
        cost_groups[cost].append((path_str, path_nodes))
    results = []
    for cost in sorted(cost_groups.keys()):
        paths_in_group = cost_groups[cost]
        all_edges = []
        path_costs_list = []
        path_lengths_list = []
        for path_str, path_nodes in paths_in_group:
            edges = extract_edge_info(G, path_nodes, rule_key, weight_key)
            all_edges.append((path_str, edges))
            path_costs_list.append(cost)
            path_lengths_list.append(len(edges))
        results.append({
            'cost': cost,
            'paths': all_edges,
            'path_costs': path_costs_list,
            'path_lengths': path_lengths_list,
        })
    return results


def format_multiplicity(path_costs: list[float]) -> str:
    if not path_costs:
        return "None"
    cost_counts = defaultdict(int)
    for cost in path_costs:
        cost_counts[cost] += 1
    parts = [f'{cost:.9f} ({count}x)' for cost, count in sorted(cost_counts.items(), reverse=True)]
    return ', '.join(parts)


def format_length_multiplicity(path_lengths: list[int]) -> str:
    if not path_lengths:
        return "None"
    length_counts = defaultdict(int)
    for length in path_lengths:
        length_counts[length] += 1
    parts = [f'Length {length} ({count}x)' for length, count in sorted(length_counts.items(), reverse=True)]
    return ', '.join(parts)


def print_results(analysis_results: list[dict] | None, hide_paths: bool = False) -> None:
    if not analysis_results:
        print('No paths found.')
        return
    for rank, result in enumerate(analysis_results, 1):
        cost = result['cost']
        paths = result['paths']
        path_costs = result['path_costs']
        path_lengths = result['path_lengths']
        print(f'{rank}: Cost: {cost:.9f}')
        if not hide_paths:
            for i, (path_str, _) in enumerate(paths, 1):
                print(f'    Path{i}: {path_str}')
        print('  Edge steps:')
        
        order_list = []
        path_str, edges = paths[0]
        for edge in edges:
            rule = edge['rule']
            weight = edge['weight']
            if rule and weight != 0.0:
                order_list.append(rule)
        for rule in order_list[:-1]:
            print(f'    {rule} --> ')
        print(f'    {order_list[-1]}')
        print(f'  Multiplicity:\n    {format_multiplicity(path_costs)}')
        print(f'  Path Length Multiplicity:\n    {format_length_multiplicity(path_lengths)}\n')


def main() -> None:
    args = parse_arguments()

    rn = molecular_formula_from_smiles(args.r)
    outpath = make_output_path(rn, args.delta)
    with open(outpath, "w", encoding="utf-8") as outfile:
        real_stdout = sys.stdout
        sys.stdout = outfile
        try:
            _run(args, outpath, rn)
        finally:
            sys.stdout = real_stdout
    print(f"Output saved to {os.path.abspath(outpath)}", file=sys.stderr)


def _run(args: argparse.Namespace, outpath: str, rn: str) -> None:
    print(f"# Output file: {outpath}\n")
    # Load graph
    gpath = args.gfile
    if gpath.endswith('.bz2'):
        with bz2.open(gpath, mode='rt', encoding='utf-8') as gfile:
            G = nx.read_graphml(gfile)
    else:
        with open(gpath, encoding='utf-8') as gfile:
            G = nx.read_graphml(gfile)

    node_labels = {node: data.get('smiles', node) for node, data in G.nodes(data=True)}
    rule_key = check_rule_key(G)
    if not rule_key:
        print('WARNING: No rule key found in graph. Using empty rules.')
        rule_key = None
    weight_key = check_weight_key(G)

    reactant_node = find_node_for_label(node_labels, args.r)
    product_node = find_node_for_label(node_labels, args.p)

    print(f'Reactant: {args.r} (node: {reactant_node})')
    print(f'Product: {args.p} (node: {product_node})')

    # 1) Dijkstra: lowest-cost path
    try:
        path_nodes = nx.dijkstra_path(G, reactant_node, product_node, weight=weight_key)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print('No path found between reactant and product.')
        return

    path_length_steps = len(path_nodes) - 1
    path_cost = compute_path_cost(path_nodes, G, weight_key)
    print(f'\nDijkstra shortest path: {path_length_steps} steps, cost = {path_cost:.9f}')
    path_str = ' --> '.join(node_labels[n] for n in path_nodes)
    print(f'  Path: {path_str}\n')

    cutoff = path_length_steps + args.delta
    print(f'BFS cutoff depth: {path_length_steps} + {args.delta} = {cutoff} steps\n')

    # 2) BFS: all paths with length <= cutoff
    G_cropped = crop_graph_to_max_length(G, reactant_node, product_node, cutoff)
    if G_cropped.number_of_nodes() < G.number_of_nodes():
        print(f'Graph optimization: {G.number_of_nodes()} -> {G_cropped.number_of_nodes()} nodes\n')

    path_costs = bfs_paths_up_to_length(
        G_cropped,
        reactant_node,
        product_node,
        cutoff,
        weight_key,
        node_labels,
        verbose=args.verbose,
    )
    print(f'BFS found {len(path_costs)} path(s) with length <= {cutoff}\n')

    if not path_costs:
        print('No paths found within cutoff.')
        return

    analysis_results = analyze_paths(G, path_costs, rule_key, weight_key)
    print_results(analysis_results, hide_paths=args.hide_paths)


if __name__ == '__main__':
    main()
