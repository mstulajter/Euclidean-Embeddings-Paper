#!/usr/bin/env python3
"""
Network Analysis Script

This script analyzes a network graph and calculates:
- Fractal dimension (via Compact Box Burning)
- Average edge weight (only for weighted graphs)
- Edge weight standard deviation (only for weighted graphs)

Authors:
    Miko Stulajter

Version 1.0.0

Usage: get_stats.py [-h] -gfile FILE [-cores CORES] [-w [WEIGHT]] [--no-weights] [-o OUTPUT] [--no-progress]
"""

import networkit as nk
import numpy as np
import bz2
import argparse
import os
import logging
import sys
from random import choice
import statsmodels.api as sm
import networkx as nx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for network analysis.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Analyze a network and calculate network fractal dimension and other statistics.')
    
    parser.add_argument(
        '-gfile',
        help="Network file as a '.graphml.bz2' file.",
        dest='gfile',
        type=str,
        required=True
    )

    parser.add_argument(
        '-cores',
        help='Number of cores running on (Default is 1).',
        dest='cores',
        default=1,
        type=int,
        required=False
    )

    parser.add_argument(
        '-w',
        help="Weight attribute name to use. (Default: 'weight'). Use --no-weights to disable.",
        dest='w',
        nargs='?',
        const='weight',
        default='weight',
        type=str,
        required=False
    )

    parser.add_argument(
        '--no-weights',
        help="Disable weight usage (overrides -w default).",
        dest='no_weights',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '-d',
        '--diameter',
        help="Provide graph diameter directly (skips calculation). If not provided, diameter will be calculated.",
        dest='diameter',
        type=int,
        required=False,
        default=None
    )

    parser.add_argument(
        '-o',
        '--output',
        help="Output file to write results to. If not specified, results are printed to stdout.",
        dest='output',
        type=str,
        required=False
    )

    parser.add_argument(
        '--no-progress',
        help="Disable progress bars (tqdm).",
        dest='no_progress',
        action='store_true',
        required=False
    )

    args = parser.parse_args()
    
    # If --no-weights is specified, override the default
    if args.no_weights:
        args.w = None
    
    return args


def main():
    """
    Main function to analyze network and calculate network fractal dimension and other statistics.
    """
    ### ~~~~~~ Argument parsing
    args = parse_arguments()

    ### ~~~~~~ Read graph from compressed GraphML file
    filename = os.path.basename(args.gfile)
    nk.setNumberOfThreads(int(args.cores))
    
    if args.w:
        # Load weighted graph from NetworkX format
        with bz2.open(args.gfile) as file_tmp:   
            graph = nx.read_graphml(file_tmp)
        H = nk.nxadapter.nx2nk(graph, weightAttr=args.w)
    else:
        # Load unweighted graph
        gmlReader = nk.graphio.GraphMLReader()
        with bz2.open(args.gfile) as file_tmp:
            H = gmlReader.read(file_tmp)
        H = nk.graphtools.toUndirected(H)
        H.removeMultiEdges()
        edge_list = list(H.iterEdges())
        H.removeAllEdges()
        for e1, e2 in edge_list:
            H.addEdge(e1, e2)

    ### ~~~~~~~~ Basic graph properties
    num_nodes = H.numberOfNodes()
    
    ### ~~~~~~~~ Diameter (needed for Compact Box Burning algorithm)
    if args.diameter is not None:
        # Use provided diameter
        diameter = args.diameter
        logging.info("Using provided diameter: %d", diameter)
    else:
        # Calculate diameter
        logging.info("Calculating graph diameter...")
        diam = nk.distance.Diameter(H, algo=1)
        diam.run()
        diameter = diam.getDiameter()[0]
        logging.info("Calculated diameter: %d", diameter)

    ### ~~~~~~~~ Compact Box Burning (CBB) algorithm for fractal dimension
    boxes_list = np.empty((diameter+1), dtype=float)
    boxes_list[0] = num_nodes

    # Parallelize all iterations (indx from 1 to diameter-1)
    if diameter > 1:
        # Use a position pool to track available positions
        position_pool = list(range(int(args.cores)))
        position_lock = threading.Lock()
        position_map = {}  # Map task indx to position
        
        def compute_boxes(indx):
            # Get a position from the pool
            with position_lock:
                if position_pool:
                    position = position_pool.pop(0)
                    position_map[indx] = position
                else:
                    # Fallback: use modulo if pool is empty (shouldn't happen)
                    position = (indx - 1) % int(args.cores)
            
            try:
                boxes = CBB_Only(H, indx, num_nodes, position=position, leave=False, disable=args.no_progress)
                return indx, boxes
            finally:
                # Return position to pool when done
                with position_lock:
                    if indx in position_map:
                        position_pool.append(position_map[indx])
                        del position_map[indx]
        
        with ThreadPoolExecutor(max_workers=int(args.cores)) as executor:
            # Submit all tasks
            future_to_indx = {executor.submit(compute_boxes, indx): indx 
                            for indx in range(1, diameter)}
            
            # Collect results with progress bar (positioned at the top)
            with tqdm(total=diameter-1, desc="Calculating CBB", unit="distance", 
                     position=int(args.cores), leave=True, disable=args.no_progress) as pbar:
                for future in as_completed(future_to_indx):
                    indx, boxes = future.result()
                    boxes_list[indx] = boxes
                    pbar.update(1)

    boxes_list[diameter] = 1
    box_length = np.arange(1, diameter + 2, dtype=float)

    ### ~~~~~~ Calculate Fractal Dimension using linear regression
    x = np.log(box_length.reshape((-1, 1)))
    y = np.log(np.array(boxes_list))
    x = sm.add_constant(x)
    CBB_model = sm.OLS(y, x).fit()
    frac_dim = np.abs(CBB_model.params[1])

    ### ~~~~~~ Calculate Edge Weight Statistics (if weighted graph)
    avg_edge_weight = None
    edge_weight_std = None
    if args.w and H.isWeighted():
        # Collect edge weights efficiently
        edge_weights = []
        num_edges = H.numberOfEdges()
        for u, v in tqdm(H.iterEdges(), total=num_edges, desc="Collecting edge weights", unit="edge", disable=args.no_progress):
            w = H.weight(u, v)
            if w is not None:
                edge_weights.append(w)
        if edge_weights:
            edge_weights = np.array(edge_weights)
            avg_edge_weight = np.mean(edge_weights)
            edge_weight_std = np.std(edge_weights, ddof=0)

    ### ~~~~~~ Prepare and output only requested statistics
    output_lines = []
    output_lines.append("Fractal dimension : %.5f" % frac_dim)
    if avg_edge_weight is not None:
        output_lines.append("Average edge weight : %.5f" % avg_edge_weight)
        output_lines.append("Edge weight std dev : %.5f" % edge_weight_std)

    output_text = "\n".join(output_lines) + "\n"

    # Print to stdout (machine- and human-readable)
    print(output_text, end='')

    # Write to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)


### ~~~~~~ Compact Box Burning (CBB) Algorithm
def CBB_Only(H, el, num_nodes, batch_size=10000, position=None, leave=True, disable=False):
    """
    Compact Box Burning algorithm to count boxes needed to cover the graph.
    
    Args:
        H: Networkit graph object
        el: Edge length threshold for box covering
        num_nodes: Total number of nodes in the graph
        batch_size: Number of nodes to process in each batch (for memory efficiency)
        position: Position for tqdm progress bar (for parallel execution)
        leave: Whether to leave progress bar after completion
        disable: Whether to disable the progress bar
        
    Returns:
        int: Number of boxes needed to cover the graph
    """
    boxes = 0
    nodes_list = list(H.iterNodes())
    uncovered_nodes = set(nodes_list)
    initial_nodes = len(uncovered_nodes)
    
    pbar = tqdm(total=initial_nodes, desc=f"Distance {el}", unit="node",
               position=position, leave=leave, ncols=100, disable=disable)
    
    while uncovered_nodes:
        candidate_set = uncovered_nodes.copy()
        while candidate_set:
            # Convert to list only once per iteration
            candidate_list = list(candidate_set)
            p = choice(candidate_list)
            candidate_set.discard(p)
            
            # Compute shortest paths from p to all nodes
            spsp = nk.distance.SPSP(H, [p])
            spsp.run()
            
            if len(candidate_set) > batch_size:
                # Process in batches for memory efficiency
                k_fa = set()
                candidate_list_batch = list(candidate_set)
                for i in range(0, len(candidate_list_batch), batch_size):
                    batch = candidate_list_batch[i:i+batch_size]
                    # Check distances for this batch
                    for k in batch:
                        if spsp.getDistance(p, k) > el:
                            k_fa.add(k)
            else:
                # For smaller sets, get all distances at once (faster)
                dist_from_p = np.array(spsp.getDistances()[0])
                k_fa = {k for k in candidate_set if dist_from_p[k] > el}
            
            candidate_set -= k_fa
            uncovered_nodes.discard(p)
            pbar.update(1)
        boxes += 1
    
    pbar.close()
    return boxes


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    try:
        main()
    except Exception:
        logging.exception("Fatal error while running get_stats")
        sys.exit(1)
