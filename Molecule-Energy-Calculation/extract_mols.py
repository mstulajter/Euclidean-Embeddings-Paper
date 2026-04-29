#!/usr/bin/env python3

"""
Extract Molecules from GraphML.bz2 file

Authors:
    Miko Stulajter

Version 2.0.0

Usage: extract_mols.py -i <input.graphml.bz2> -o <output.csv>
"""

import argparse
import bz2
import csv
import io
import xml.etree.ElementTree as ET


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for input and output files.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract Molecules from GraphML.bz2 file"
    )
    parser.add_argument(
        "-i",
        help="Path to the input GraphML.bz2 file",
        dest='input_file',
        type=str,
        required=True
    )
    parser.add_argument(
        "-o",
        help="Path to save the output CSV file",
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


def find_molecules_incremental(data):
    """
    Extract unique molecules from all nodes in the graph using incremental parsing.
    This avoids loading the entire XML into memory at once.
    
    Args:
        data: XML data as bytes
        
    Returns:
        Set of unique molecule SMILES strings
    """
    molecules_set = set()
    graphml_ns = '{http://graphml.graphdrawing.org/xmlns}'
    
    # Use BytesIO to create a file-like object from bytes
    file_obj = io.BytesIO(data)
    
    # Use iterparse for incremental parsing
    context = ET.iterparse(file_obj, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)
    
    for event, elem in context:
        if event == 'end' and elem.tag == f'{graphml_ns}node':
            # Find the data element within this node
            data_elem = elem.find(f'{graphml_ns}data')
            if data_elem is not None and data_elem.text:
                molecules = data_elem.text
                for molecule in molecules.split('.'):
                    molecules_set.add(molecule)
            # Clear the element to free memory as we process
            elem.clear()
    
    return molecules_set


def write_molecules(molecules_set, args):
    """
    Write the extracted molecules to a CSV file with 'smiles' header.
    
    Args:
        molecules_set: Set of molecule SMILES strings
        args: Parsed command-line arguments containing output_file
    """
    with open(args.output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles'])
        for molecule in molecules_set:
            writer.writerow([molecule])


def main():
    """
    Main execution function.
    """
    args = parse_arguments()
    data = read_graphml_bz2(args)
    molecules_set = find_molecules_incremental(data)
    
    # If [H+] is present, ensure O and [OH3+] are also included
    if '[H+]' in molecules_set:
        if 'O' not in molecules_set:
            molecules_set.add('O')
        if '[OH3+]' not in molecules_set:
            molecules_set.add('[OH3+]')
    
    # Set default output filename if not provided
    if not args.output_file:
        args.output_file = args.input_file.replace('.graphml.bz2', '_MOLS.csv')
    
    write_molecules(molecules_set, args)


if __name__ == "__main__":
    main()
