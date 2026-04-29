#!/usr/bin/env python3

"""
Process a GraphML file to ensure all edges have their inverse.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: make_reversable.py [-h] [-o OUTPUT_FILE] input_file
"""

import bz2
import argparse
import networkit as nk
import shutil
from io import BytesIO
import sys
import os
import re


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for input files and options.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process a GraphML file to ensure all edges have their inverse."
    )

    parser.add_argument(
        "input_file",
        help="Path to the input GraphML.bz2 file",
        type=str
    )

    parser.add_argument(
        "-o",
        help="Path to save the modified GraphML file",
        dest='output_file',
        type=str,
        required=False
    )

    return parser.parse_args()


def read_graphml_bz2(args):
    """
    Read and decompress the GraphML.bz2 file.
    
    Args:
        args: Parsed command-line arguments containing input_file
        
    Returns:
        Decompressed file data as bytes
    """
    with bz2.open(args.input_file, 'rb') as f:
        data = f.read()
    return data


def parse_and_write_incremental(input_file, output_file):
    """
    Parse the GraphML file manually and write it with all edges having their inverses.
    As we read edges, we match them with their inverses and write them in pairs.
    This minimizes memory usage by only keeping unmatched edges in memory.
    
    Args:
        input_file: Path to input GraphML.bz2 file
        output_file: Path to output file
    """

    unmatched_edges = {}
    
    write_count = 0
    
    def flip_transformation(transformation):
        """Flip a transformation string (swap parts around >> or &gt;&gt;)."""
        split_symbol = '>>' if '>>' in transformation else '&gt;&gt;'
        if split_symbol in transformation:
            trans1, trans2 = transformation.split(split_symbol, 1)
            return f'{trans1}{split_symbol}{trans2}'
        return transformation
    
    def format_transformation_for_xml(transformation, use_escaped_gt):
        """Format transformation for XML, using same escaping format as original."""
        if use_escaped_gt:
            # Use &gt;&gt; format
            return transformation.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        else:
            # Use >> format (only escape & and <, not >)
            return transformation.replace('&', '&amp;').replace('<', '&lt;')
    
    with bz2.open(input_file, 'rb') as in_f, bz2.open(output_file, 'wb') as out_f:
        current_edge_lines = []
        current_source = None
        current_target = None
        current_transformation = None
        in_edge = False
        in_data = False
        collecting_edge = False
        
        for line_bytes in in_f:
            line = line_bytes.decode('utf-8', errors='replace')
            original_line_bytes = line_bytes
            
            # Check for edge opening tag
            edge_match = re.search(r'<edge\s+source="([^"]+)"\s+target="([^"]+)"', line)
            if edge_match:
                current_source = edge_match.group(1)
                current_target = edge_match.group(2)
                in_edge = True
                in_data = False
                current_transformation = None
                current_uses_escaped_gt = False
                current_edge_lines = [original_line_bytes]
                collecting_edge = True
                continue
            
            # If we're collecting an edge, add lines to it
            if collecting_edge:
                current_edge_lines.append(original_line_bytes)
                
                # Check for data element with key="d1" (edge transformation)
                if '<data key="d1">' in line:
                    # Detect if original uses &gt;&gt; or >>
                    uses_escaped_gt = '&gt;&gt;' in line
                    
                    data_match = re.search(r'<data key="d1">(.*?)</data>', line)
                    if data_match:
                        current_transformation = data_match.group(1)
                        # Unescape XML entities for processing
                        current_transformation = current_transformation.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                        current_uses_escaped_gt = uses_escaped_gt
                    elif '<data key="d1">' in line and '</data>' not in line:
                        # Multi-line data - start collecting
                        data_start = line.find('<data key="d1">') + len('<data key="d1">')
                        current_transformation = line[data_start:]
                        current_uses_escaped_gt = uses_escaped_gt
                        in_data = True
                
                # Continue reading multi-line data
                if in_data:
                    if '</data>' in line:
                        data_end = line.find('</data>')
                        current_transformation += line[:data_end]
                        current_transformation = current_transformation.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                        in_data = False
                    elif not current_transformation:
                        # Shouldn't happen, but handle it
                        pass
                    else:
                        current_transformation += line.strip()
                
                # Check for edge closing tag
                if '</edge>' in line:
                    collecting_edge = False
                    in_edge = False
                    
                    # Check if we have the inverse of this edge
                    inverse_key = (current_target, current_source)
                    if inverse_key in unmatched_edges:
                        # Found the pair! Write both edges together
                        inverse_lines, inverse_transformation, inverse_uses_escaped_gt = unmatched_edges.pop(inverse_key)
                        
                        # Write the original edge (the one we just finished reading)
                        for edge_line in current_edge_lines:
                            out_f.write(edge_line)
                        
                        # Write the inverse edge
                        for edge_line in inverse_lines:
                            out_f.write(edge_line)
                        
                        write_count += 2
                    else:
                        # No inverse found yet, store this edge
                        unmatched_edges[(current_source, current_target)] = (
                            current_edge_lines, 
                            current_transformation, 
                            current_uses_escaped_gt
                        )
                    
                    # Reset for next edge
                    current_edge_lines = []
                    current_source = None
                    current_target = None
                    current_transformation = None
                    current_uses_escaped_gt = False
                    in_data = False
                    continue
                
                # Continue collecting edge lines
                continue
            
            # Not in an edge - check if this is the closing graph tag
            if line.strip() == '</graph>' or line.strip().endswith('</graph>'):
                # Write all unmatched edges (these need their inverses added)
                for (source, target), (edge_lines, transformation, uses_escaped_gt) in unmatched_edges.items():
                    # Write the original edge
                    for edge_line in edge_lines:
                        out_f.write(edge_line)
                    
                    flipped_transformation = flip_transformation(transformation) if transformation else ''
                    transformation_formatted = format_transformation_for_xml(flipped_transformation, uses_escaped_gt)
                    edge_line = f'    <edge source="{target}" target="{source}">\n'.encode('utf-8')
                    data_line = f'      <data key="d1">{transformation_formatted}</data>\n'.encode('utf-8')
                    edge_close = '    </edge>\n'.encode('utf-8')
                    out_f.write(edge_line)
                    out_f.write(data_line)
                    out_f.write(edge_close)
                    write_count += 2
                
                # Write the closing graph tag
                out_f.write(original_line_bytes)
                write_count += 1
                if write_count % 5000 == 0:
                    out_f.flush()
                continue
            
            # Regular line - write as-is
            out_f.write(original_line_bytes)
            write_count += 1
            if write_count % 5000 == 0:
                out_f.flush()
        
        out_f.flush()


def check_num_edges(data):
    """
    Check the number of missing inverse edges in the graph.
    
    Args:
        data: GraphML file data as bytes
        
    Returns:
        Number of missing inverse edges
    """
    data_bytes = BytesIO(data)
    gmlReader = nk.graphio.GraphMLReader()
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    G = gmlReader.read(data_bytes)
    sys.stdout = old_stdout
    total_num_edges = G.numberOfEdges()
    G = nk.graphtools.toUndirected(G)
    G.removeMultiEdges()
    undir_num_edges = len(list(G.iterEdges()))
    return undir_num_edges * 2 - total_num_edges


def main():
    """
    Main execution function.
    """
    args = parse_arguments()
    
    # Read data for checking number of edges
    data = read_graphml_bz2(args)
    
    if not args.output_file:
        args.output_file = args.input_file.replace('.graphml.bz2', '_AME.graphml.bz2')
    
    missing_edges = check_num_edges(data)
    if missing_edges == 0:
        shutil.copyfile(args.input_file, args.output_file)
    else:
        # Parse and write manually, matching edges with their inverses as we go
        parse_and_write_incremental(args.input_file, args.output_file)

    input_name = args.input_file.replace('.graphml.bz2', '')
    print(f'{input_name}\t:\t{missing_edges} Missing Edges')


if __name__ == "__main__":
    main()

