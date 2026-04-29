#!/usr/bin/env python3

"""
Find equivalence classes for molecules using bond opening/closing rules.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: find_equivalent_mols.py -i <input.csv> -o <output.csv> -r <rules.yml>
"""

import argparse
import csv
from collections import Counter, deque
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel

# Charge patterns for detecting charged atoms in SMILES
_charge_patterns = {
    '[*+1]': Counter([1]),
    '[*+2]': Counter([2]),
    '[*+3]': Counter([3]),
    '[*-1]': Counter([-1]),
    '[*-2]': Counter([-2]),
    '[*-3]': Counter([-3])
}

# Charge exceptions (charge-separated pairs that should be treated differently)
_charge_exceptions = {'[C-1]#[O+1]': Counter([1, -1])}


class Charge(object):
    """
    Matching facility for charged and charge-separated compounds using SMILES strings.
    """

    def __init__(self):
        """
        Initialize charge patterns and exceptions with compiled SMARTS patterns.
        """
        self._patterns = _charge_patterns
        # Compile pybel representations of SMARTS patterns for efficient matching
        self._pysmarts = {s: pybel.Smarts(s) for s in self._patterns}
        self._exceptions = _charge_exceptions
        # Compile pybel representations of exception SMARTS patterns
        self._pyexceptions = {s: pybel.Smarts(s) for s in self._exceptions}

    def count(self, smiles):
        """
        Count numbers of charged atoms, removing charge exceptions.
        
        Args:
            smiles: SMILES string (may contain multiple molecules separated by '.')
            
        Returns:
            Counter object with net charges
        """
        charges = Counter()
        for s in smiles.split('.'):
            pymol = pybel.readstring('smi', s)
            # Count charged atoms using patterns
            for p, c in self._patterns.items():
                matches = len(self._pysmarts[p].findall(pymol))
                for m in range(matches):
                    charges += c
            # Remove charge exceptions (charge-separated pairs)
            for e, c in self._exceptions.items():
                matches = len(self._pyexceptions[e].findall(pymol))
                for m in range(matches):
                    charges -= c
        return charges

    def abs_charges(self, smiles):
        """
        Compute sum of absolute charges in SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Sum of absolute values of all charges
        """
        charges = self.count(smiles)
        return sum(abs(c) for c in charges.elements())


class Bonds(object):
    """
    Utility class for determining number of bonds (including multiplicity) from SMILES string.
    """

    def __init__(self):
        """
        Initialize the Bonds utility class.
        """
        pass

    def num_bonds(self, smiles):
        """
        Return total number of bonds (including bond order/multiplicity).
        
        Args:
            smiles: SMILES string (may contain multiple molecules separated by '.')
            
        Returns:
            Total number of bonds (sum of bond orders)
        """
        nb = 0
        for s in smiles.split('.'):
            pymol = pybel.readstring('smi', s)
            pymol.addh()
            # Sum bond orders for all bonds
            for b in range(pymol.OBMol.NumBonds()):
                nb += pymol.OBMol.GetBondById(b).GetBondOrder()
        return nb


def abs_charges(smiles):
    """
    Return sum of absolute charges in SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Sum of absolute values of all charges
    """
    ch = Charge()
    return ch.abs_charges(str(smiles))


def num_bonds(smiles):
    """
    Return number of bonds (including multiplicity) from SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Total number of bonds (sum of bond orders)
    """
    bonds = Bonds()
    return bonds.num_bonds(str(smiles))


def apply_reaction(smiles, reaction_smarts, forward=True):
    """
    Apply a reaction SMARTS pattern to a SMILES string.
    
    Args:
        smiles: Input SMILES string
        reaction_smarts: Reaction SMARTS pattern (format: reactant>>product)
        forward: If True, apply forward reaction; if False, apply reverse
        
    Returns:
        List of product SMILES strings (canonical, unique)
    """
    try:
        # Handle reverse reactions by swapping reactant and product
        if not forward:
            parts = reaction_smarts.split('>>')
            if len(parts) == 2:
                reaction_smarts = '>>'.join([parts[1], parts[0]])
            else:
                return []
        
        # Parse reaction SMARTS
        rxn = AllChem.ReactionFromSmarts(reaction_smarts)
        if rxn is None:
            return []
        
        # Parse reactant molecule (try without sanitization first)
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return []
        
        # Apply reaction to generate products
        products = rxn.RunReactants((mol,))
        
        # Convert products to canonical SMILES
        result = []
        for product_tuple in products:
            for product in product_tuple:
                if product is None:
                    continue
                try:
                    # Try sanitizing first for better results
                    Chem.SanitizeMol(product)
                    product_smiles = Chem.MolToSmiles(product, canonical=True)
                    if product_smiles:
                        result.append(product_smiles)
                except Exception:
                    # If sanitization fails, try without it
                    try:
                        product_smiles = Chem.MolToSmiles(product, canonical=True)
                        if product_smiles:
                            result.append(product_smiles)
                    except Exception:
                        pass
        
        return list(set(result))
    except Exception:
        return []


def find_equivalent_molecules(start_smiles, rules, zero_energy_types, max_depth=10):
    """
    Find all molecules equivalent to start_smiles via zero-energy transformations.
    
    Args:
        start_smiles: Starting SMILES string
        rules: Dictionary mapping reaction SMARTS to rule types
        zero_energy_types: List of rule types considered zero-energy (e.g., 'bond opening', 'bond closing')
        max_depth: Maximum search depth (default: 10)
        
    Returns:
        Set of equivalent SMILES strings
    """
    seen = set()
    queue = deque([(start_smiles, 0)])
    seen.add(start_smiles)
    equivalent = set([start_smiles])
    
    # Filter to only zero-energy rules
    zero_energy_rules = {
        smarts: rule_type for smarts, rule_type in rules.items()
        if rule_type in zero_energy_types
    }
    
    # Breadth-first search through equivalent molecules
    while queue:
        current_smiles, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
        
        # Try all zero-energy transformations
        for reaction_smarts, rule_type in zero_energy_rules.items():
            # Try both forward and reverse reactions
            for forward in [True, False]:
                products = apply_reaction(current_smiles, reaction_smarts, forward=forward)
                for product in products:
                    if product not in seen:
                        seen.add(product)
                        queue.append((product, depth + 1))
                        equivalent.add(product)
    
    return equivalent


def find_equivalence_classes(molecules, rules, zero_energy_types, max_depth=10):
    """
    Find equivalence classes for a list of molecules.
    
    Groups molecules into equivalence classes based on zero-energy transformations.
    Selects a representative for each class based on:
    - Number of bonds (prefer more bonds)
    - Absolute charge (prefer lower charge)
    - Presence in input set (prefer molecules from input)
    - Alphabetical order (for tie-breaking)
    
    Args:
        molecules: List of SMILES strings
        rules: Dictionary mapping reaction SMARTS to rule types
        zero_energy_types: List of rule types considered zero-energy
        max_depth: Maximum search depth for finding equivalents
        
    Returns:
        Dictionary mapping each input molecule to its equivalence class representative(s)
    """
    molecules_set = set(molecules)
    molecule_to_class = {}
    processed = set()
    
    for smiles in molecules:
        if smiles in processed:
            continue
        
        # Find all equivalent molecules (including those not in input set)
        all_equivalent_set = find_equivalent_molecules(smiles, rules, zero_energy_types, max_depth)
        all_equivalent_set.add(smiles)
        
        # Filter to only molecules in our input set (for tracking processed)
        equivalent_set = {s for s in all_equivalent_set if s in molecules_set}
        
        # Mark all input molecules in this equivalence class as processed
        processed.update(equivalent_set)
        
        # Find representative from ALL equivalent molecules (not just input set)
        # This allows picking the best representative even if it's not in the input
        if all_equivalent_set:
            representatives = []
            for s in all_equivalent_set:
                try:
                    charge = abs_charges(s)
                    bonds = num_bonds(s)
                    # Prefer molecules in the input set (1 if in input, 0 if not)
                    in_input = 1 if s in molecules_set else 0
                    representatives.append((s, bonds, charge, in_input))
                except Exception:
                    # If we can't compute properties, use the molecule as-is
                    in_input = 1 if s in molecules_set else 0
                    representatives.append((s, 0, float('inf'), in_input))
            
            # Sort by: -bonds (descending), charge (ascending), -in_input (descending), smiles (alphabetical)
            # This ensures deterministic selection: prefer more bonds, lower charge, molecules in input set, then alphabetical
            representatives.sort(key=lambda x: (-x[1], x[2], -x[3], x[0]))
            
            # Find all tied representatives (same bonds, charge, and in_input status)
            if representatives:
                best_bonds = representatives[0][1]
                best_charge = representatives[0][2]
                best_in_input = representatives[0][3]
                
                # Get all representatives with the same best score
                tied_representatives = [
                    r[0] for r in representatives
                    if r[1] == best_bonds and r[2] == best_charge and r[3] == best_in_input
                ]
                
                # If there's a tie, use a list; otherwise use a single string
                if len(tied_representatives) > 1:
                    representative = tied_representatives
                else:
                    representative = tied_representatives[0]
            else:
                # Fallback: use the first molecule in the set
                representative = list(all_equivalent_set)[0] if all_equivalent_set else smiles
            
            # Map all input molecules in this equivalence class to the representative(s)
            for s in equivalent_set:
                molecule_to_class[s] = representative
    
    # Ensure all molecules are in the map (fallback for molecules not processed)
    for smiles in molecules:
        if smiles not in molecule_to_class:
            molecule_to_class[smiles] = smiles
    
    return molecule_to_class


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description='Find equivalence classes for molecules using bond opening/closing rules'
    )
    
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='Input CSV file with molecules'
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Output CSV file (mapping file with smiles,equivalent columns)'
    )
    parser.add_argument(
        '-r',
        '--rules',
        required=True,
        help='Rules YAML file'
    )
    parser.add_argument(
        '-c',
        '--column',
        default='smiles',
        help='Column name containing SMILES (default: smiles)'
    )
    parser.add_argument(
        '-d',
        '--max-depth',
        type=int,
        default=10,
        help='Maximum depth for equivalence search (default: 10)'
    )
    parser.add_argument(
        '-u',
        '--unique',
        default=None,
        help='Also write unique molecules (equivalence class representatives) to this file'
    )
    
    args = parser.parse_args()
    
    # Load reaction rules from YAML file
    with open(args.rules, 'r') as fd:
        rules_data = yaml.safe_load(fd)
        rules = rules_data.get('rules', {})
    
    # Define zero-energy transformation types
    zero_energy_types = ['bond opening', 'bond closing']
    
    # Read molecules from CSV file
    molecules = []
    with open(args.input, 'r') as fd:
        reader = csv.DictReader(fd)
        if args.column not in reader.fieldnames:
            raise ValueError(f"Column '{args.column}' not found in {args.input}")
        
        for row in reader:
            smiles = row[args.column].strip()
            if smiles:
                molecules.append(smiles)
    
    print(f'Loaded {len(molecules)} molecules')
    print(f'Found {len([r for r in rules.values() if r in zero_energy_types])} zero-energy rules')
    
    # Find equivalence classes
    equiv_map = find_equivalence_classes(molecules, rules, zero_energy_types, args.max_depth)
    
    # Write mapping file (molecule -> representative)
    with open(args.output, 'w', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow([args.column, 'equivalent'])
        for smiles in molecules:
            equivalent = equiv_map.get(smiles, smiles)
            # Convert list to pipe-separated string, or keep as string
            if isinstance(equivalent, list):
                equivalent_str = '|'.join(sorted(equivalent))
            else:
                equivalent_str = equivalent
            writer.writerow([smiles, equivalent_str])
    print(f'Mapping file written to {args.output}')
    
    # If -u is provided, also write unique representatives file
    if args.unique:
        with open(args.unique, 'w', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(['smiles'])
            unique_representatives = set()
            # Expand lists to individual molecules
            for representative in equiv_map.values():
                if isinstance(representative, list):
                    unique_representatives.update(representative)
                else:
                    unique_representatives.add(representative)
            # Write all unique representatives (including all tied ones)
            for representative in sorted(unique_representatives):
                writer.writerow([representative])
        unique_count = len(unique_representatives)
        print(f'Wrote {unique_count} unique molecules to {args.unique}')
    
    # Count unique equivalence classes (treating lists as single classes)
    unique_equivalents = len(set(
        tuple(sorted(rep)) if isinstance(rep, list) else rep
        for rep in equiv_map.values()
    ))
    print(f'Found {unique_equivalents} unique equivalence classes')


if __name__ == '__main__':
    main()
