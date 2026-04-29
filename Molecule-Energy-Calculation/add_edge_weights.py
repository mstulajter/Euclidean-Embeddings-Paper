#!/usr/bin/env python3

"""
Add weights to edges in GraphML files based on node energies.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: add_edge_weights.py [-h] [-En ENERGY_KEYWORD] [-o OUTPUT_FILE] input_file input_energy_file
"""

import bz2
import argparse
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import re

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for input files and options.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Add Weights to Edges Based on Nodes"
    )

    parser.add_argument(
        "input_file",
        help="Path to the input GraphML.bz2 file",
        type=str
    )

    parser.add_argument(
        "input_energy_file",
        help="Path to the energy input CSV file",
        type=str
    )

    parser.add_argument(
        "-En",
        help="Keyword for the energy column (default: 'energy')",
        dest='Energy_Keyword',
        type=str,
        default='energy',
        required=False
    )

    parser.add_argument(
        "-o",
        help="Path to save the modified GraphML file",
        dest='output_file',
        type=str,
        required=False
    )

    return parser.parse_args()


def canonicalize_multimol_smiles(smiles: str) -> str | None:
    """
    Canonicalize a multi-molecule SMILES (molecules separated by '.').
    
    Args:
        smiles: Multi-molecule SMILES string
        
    Returns:
        Canonical multi-molecule SMILES (sorted) or None if invalid
    """
    parts = smiles.split('.')
    canonical_parts = []
    
    for part in parts:
        mol = Chem.MolFromSmiles(part.strip())
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        canonical_parts.append(canonical)
    
    # Sort to ensure consistent ordering
    canonical_parts.sort()
    return '.'.join(canonical_parts)


def parse_energy_csv(args):
    """
    Parse energy data from CSV file.
    
    Args:
        args: Parsed command-line arguments containing input_energy_file and Energy_Keyword
        
    Returns:
        Dictionary mapping canonical SMILES to energy values (handles both single and multi-molecule SMILES)
    """
    df = pd.read_csv(args.input_energy_file)
    energy_dict = {}
    for _, row in df.iterrows():
        smiles = row['smiles']
        # Canonicalize (handles both single and multi-molecule SMILES)
        canonical_smiles = canonicalize_multimol_smiles(smiles)
        if canonical_smiles:
            energy_dict[canonical_smiles] = row[args.Energy_Keyword]
            # Also store original if different (for backward compatibility)
            if canonical_smiles != smiles:
                energy_dict[smiles] = row[args.Energy_Keyword]
    return energy_dict


def parse_nodes_and_edges(input_file, energy_dict):
    """
    Parse nodes and edges from GraphML file manually.
    Calculate node energies and edge weights.
    
    Args:
        input_file: Path to input GraphML.bz2 file
        energy_dict: Dictionary mapping SMILES to energy values
        
    Returns:
        Tuple of (node_energies dict, edge_weights dict, valid_node_ids set, valid_edge_ids set)
    """
    node_energies = {}
    edge_weights = {}
    valid_node_ids = set()
    valid_edge_ids = set()
    missing_mol = set()
    
    with bz2.open(input_file, 'rb') as f:
        current_node_id = None
        current_node_smiles = None
        in_node = False
        in_data_d0 = False
        collecting_smiles = False
        
        for line_bytes in f:
            line = line_bytes.decode('utf-8', errors='replace')
            
            # Check for node opening tag
            node_match = re.search(r'<node\s+id="([^"]+)"', line)
            if node_match:
                current_node_id = node_match.group(1)
                current_node_smiles = None
                in_node = True
                in_data_d0 = False
                collecting_smiles = False
                continue
            
            # Check for data key="d0" (node SMILES)
            if in_node and '<data key="d0">' in line:
                data_match = re.search(r'<data key="d0">(.*?)</data>', line)
                if data_match:
                    current_node_smiles = data_match.group(1)
                    # Unescape XML entities
                    current_node_smiles = current_node_smiles.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                elif '<data key="d0">' in line and '</data>' not in line:
                    # Multi-line data
                    data_start = line.find('<data key="d0">') + len('<data key="d0">')
                    current_node_smiles = line[data_start:]
                    collecting_smiles = True
                continue
            
            # Continue reading multi-line SMILES data
            if collecting_smiles and in_node:
                if '</data>' in line:
                    data_end = line.find('</data>')
                    current_node_smiles += line[:data_end]
                    current_node_smiles = current_node_smiles.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                    collecting_smiles = False
                else:
                    current_node_smiles += line.strip()
                continue
            
            # Check for node closing tag
            if '</node>' in line and in_node:
                if current_node_id and current_node_smiles:
                    # Calculate node energy
                    node_energy = 0
                    all_molecules_found = True
                    # For each molecule in the node, look up its energy
                    # Try both exact match and canonicalized version
                    for molecule in current_node_smiles.split('.'):
                        mol_energy = energy_dict.get(molecule)
                        
                        # If not found, try canonicalized version
                        if mol_energy is None:
                            canonical_mol = canonicalize_multimol_smiles(molecule)
                            if canonical_mol:
                                mol_energy = energy_dict.get(canonical_mol)
                        
                        if mol_energy is None:
                            all_molecules_found = False
                            missing_mol.add(molecule)
                            break
                        node_energy += mol_energy
                    
                    if all_molecules_found:
                        node_energies[current_node_id] = node_energy
                        valid_node_ids.add(current_node_id)
                
                in_node = False
                current_node_id = None
                current_node_smiles = None
                continue
    
    # Second pass: parse edges and calculate weights
    with bz2.open(input_file, 'rb') as f:
        for line_bytes in f:
            line = line_bytes.decode('utf-8', errors='replace')
            
            # Check for edge opening tag
            edge_match = re.search(r'<edge\s+source="([^"]+)"\s+target="([^"]+)"', line)
            if edge_match:
                source = edge_match.group(1)
                target = edge_match.group(2)
                
                source_energy = node_energies.get(source)
                target_energy = node_energies.get(target)
                
                if source_energy is not None and target_energy is not None:
                    # Calculate edge weight
                    alpha = 1.
                    kappa = 5.
                    energy_diff = target_energy - source_energy
                    weight = np.sqrt(
                        energy_diff**2 + alpha**2 *
                        ((1 - np.exp(-kappa * energy_diff**2)) /
                         (1 + np.exp(-kappa * energy_diff**2)))
                    )
                    edge_weights[f'{source}-{target}'] = weight
                    valid_edge_ids.add((source, target))
    
    # Count total nodes in file for reporting
    total_nodes = 0
    with bz2.open(input_file, 'rb') as f:
        for line_bytes in f:
            line = line_bytes.decode('utf-8', errors='replace')
            if re.search(r'<node\s+id=', line):
                total_nodes += 1
    
    removed_nodes = total_nodes - len(valid_node_ids)
    print(f'Removed {removed_nodes} nodes out of {total_nodes}')
    print(f'Missing {len(missing_mol)} energies of molecules')
    
    return node_energies, edge_weights, valid_node_ids, valid_edge_ids


def write_graphml_with_weights(input_file, output_file, edge_weights, valid_node_ids, valid_edge_ids):
    """
    Write GraphML file with weights added to edges, skipping invalid nodes/edges.
    
    Args:
        input_file: Path to input GraphML.bz2 file
        output_file: Path to output file
        edge_weights: Dictionary mapping edge_id (source-target) to weight
        valid_node_ids: Set of valid node IDs to keep
        valid_edge_ids: Set of valid edge (source, target) tuples to keep
    """
    write_count = 0
    d2_key_written = False
    
    with bz2.open(input_file, 'rb') as in_f, bz2.open(output_file, 'wb') as out_f:
        current_node_id = None
        current_edge_source = None
        current_edge_target = None
        in_node = False
        in_edge = False
        collecting_node = False
        collecting_edge = False
        node_lines = []
        edge_lines = []
        
        for line_bytes in in_f:
            line = line_bytes.decode('utf-8', errors='replace')
            original_line_bytes = line_bytes
            
            # Write XML header and graphml opening tag
            if line.strip().startswith('<?xml') or line.strip().startswith('<graphml'):
                out_f.write(original_line_bytes)
                write_count += 1
                continue
            
            # Add d2 key definition after other keys
            if not d2_key_written and '<key' in line and 'd1' in line:
                # Write the d1 key line
                out_f.write(original_line_bytes)
                # Add d2 key definition
                d2_key_line = '  <key id="d2" for="edge" attr.name="weight" attr.type="float" />\n'
                out_f.write(d2_key_line.encode('utf-8'))
                d2_key_written = True
                write_count += 2
                continue
            
            # Check for node opening tag
            node_match = re.search(r'<node\s+id="([^"]+)"', line)
            if node_match:
                current_node_id = node_match.group(1)
                in_node = True
                collecting_node = current_node_id in valid_node_ids
                node_lines = [original_line_bytes] if collecting_node else []
                continue
            
            # Collect node lines if valid
            if collecting_node:
                node_lines.append(original_line_bytes)
                if '</node>' in line:
                    # Write the complete node
                    for node_line in node_lines:
                        out_f.write(node_line)
                    write_count += 1
                    collecting_node = False
                    in_node = False
                    current_node_id = None
                    node_lines = []
                continue
            
            # Skip invalid nodes
            if in_node and '</node>' in line:
                in_node = False
                current_node_id = None
                continue
            
            # Check for edge opening tag
            edge_match = re.search(r'<edge\s+source="([^"]+)"\s+target="([^"]+)"', line)
            if edge_match:
                current_edge_source = edge_match.group(1)
                current_edge_target = edge_match.group(2)
                edge_key = (current_edge_source, current_edge_target)
                in_edge = True
                collecting_edge = edge_key in valid_edge_ids
                edge_lines = [original_line_bytes] if collecting_edge else []
                continue
            
            # Collect edge lines if valid
            if collecting_edge:
                if '</edge>' in line:
                    # Write the edge lines (without the closing tag)
                    for edge_line in edge_lines:
                        out_f.write(edge_line)
                    
                    # Add weight data before closing tag
                    edge_id = f'{current_edge_source}-{current_edge_target}'
                    weight = edge_weights.get(edge_id)
                    if weight is not None:
                        weight_line = f'      <data key="d2">{weight:.9f}</data>\n'
                        out_f.write(weight_line.encode('utf-8'))
                    
                    # Write closing tag
                    out_f.write(original_line_bytes)
                    write_count += 1
                    collecting_edge = False
                    in_edge = False
                    current_edge_source = None
                    current_edge_target = None
                    edge_lines = []
                else:
                    # Collect edge lines (but not the closing tag)
                    edge_lines.append(original_line_bytes)
                continue
            
            # Skip invalid edges
            if in_edge and '</edge>' in line:
                in_edge = False
                current_edge_source = None
                current_edge_target = None
                continue
            
            # Write other lines as-is
            out_f.write(original_line_bytes)
            write_count += 1
            if write_count % 5000 == 0:
                out_f.flush()
        
        out_f.flush()




def main():
    """
    Main execution function.
    """
    args = parse_arguments()
    
    # Parse energy data from CSV
    energy_dict = parse_energy_csv(args)
    
    # Parse nodes and edges, calculate energies and weights
    node_energies, edge_weights, valid_node_ids, valid_edge_ids = parse_nodes_and_edges(
        args.input_file, energy_dict
    )
    
    # Count edges removed
    total_edges = len(edge_weights) + (sum(1 for _ in bz2.open(args.input_file, 'rb')) if True else 0)
    # Quick count of edges in file
    edge_count = 0
    with bz2.open(args.input_file, 'rb') as f:
        for line_bytes in f:
            line = line_bytes.decode('utf-8', errors='replace')
            if re.search(r'<edge\s+source=', line):
                edge_count += 1
    removed_edges = edge_count - len(edge_weights)
    print(f'Removed {removed_edges} edges, remaining {len(edge_weights)}')
    
    # Set default output filename if not provided
    if not args.output_file:
        if 'AME.graphml.bz2' in args.input_file:
            args.output_file = args.input_file.replace('AME.graphml.bz2', 'NW_KARC_AME.graphml.bz2')
        else:
            args.output_file = args.input_file.replace('.graphml.bz2', '_NW_KARC.graphml.bz2')
        args.output_file = args.output_file.split('/')[-1]
    
    # Write GraphML with weights
    write_graphml_with_weights(
        args.input_file, args.output_file, edge_weights, valid_node_ids, valid_edge_ids
    )


if __name__ == "__main__":
    main()

