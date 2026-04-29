#!/usr/bin/env python3

"""
Merge energy data with equivalence mappings.

Authors:
    Miko Stulajter

Version 2.0.0

Usage: merge_energies.py <energy_file> <equivalence_file> <output_file>
"""

import argparse
import csv
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')


def canonicalize_smiles(smiles: str) -> str | None:
    """
    Convert SMILES to canonical form using RDKit.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES string or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


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
        canonical = canonicalize_smiles(part.strip())
        if canonical is None:
            return None
        canonical_parts.append(canonical)
    
    # Sort to ensure consistent ordering
    canonical_parts.sort()
    return '.'.join(canonical_parts)


def load_equivalence_mapping(equiv_file):
    """
    Load equivalence mapping from CSV file.
    
    Creates a bidirectional mapping: for each SMILES, stores all equivalent forms
    (including itself and canonicalized versions).
    
    Args:
        equiv_file: Path to CSV file with equivalence mappings
        
    Returns:
        tuple: (real_to_equiv, all_real_smiles, smiles_to_all_equivs)
            - real_to_equiv: dictionary mapping real smiles to their equivalent form(s)
              Values can be strings or lists of strings
            - all_real_smiles: set of all unique real smiles (from first column)
            - smiles_to_all_equivs: dictionary mapping any SMILES to set of all equivalent forms
    """
    real_to_equiv = {}
    all_real_smiles = set()
    smiles_to_all_equivs = {}  # Maps any SMILES to set of all equivalent forms
    
    with open(equiv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            real_smiles = row['smiles'].strip()
            equiv_smiles_str = row['equivalent'].strip()
            
            # Check if it's a pipe-separated list (tied representatives)
            if '|' in equiv_smiles_str:
                equiv_smiles = equiv_smiles_str.split('|')
                equiv_smiles = [s.strip() for s in equiv_smiles if s.strip()]
            else:
                equiv_smiles = equiv_smiles_str
            
            real_to_equiv[real_smiles] = equiv_smiles
            all_real_smiles.add(real_smiles)
            
            # Build equivalence sets: all forms that are equivalent to each other
            equiv_set = {real_smiles}
            if isinstance(equiv_smiles, list):
                equiv_set.update(equiv_smiles)
            elif equiv_smiles:
                equiv_set.add(equiv_smiles)
            
            # Also add canonicalized versions
            for smiles in equiv_set:
                canonical = canonicalize_multimol_smiles(smiles)
                if canonical:
                    equiv_set.add(canonical)
            
            # Map each form in the set to the full set
            for smiles in equiv_set:
                if smiles not in smiles_to_all_equivs:
                    smiles_to_all_equivs[smiles] = set()
                smiles_to_all_equivs[smiles].update(equiv_set)
    
    return real_to_equiv, all_real_smiles, smiles_to_all_equivs


def load_energy_data(energy_file):
    """
    Load energy data from CSV file.
    
    Args:
        energy_file: Path to CSV file with smiles and energy columns
    
    Returns:
        Tuple of:
        - Dictionary mapping smiles to energy values (as strings)
        - Dictionary mapping canonical smiles to list of original smiles (for lookup)
    """
    energy_data = {}
    canonical_to_originals = {}  # Maps canonical SMILES to list of original SMILES
    
    with open(energy_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row['smiles'].strip()
            energy = row['energy'].strip() if row['energy'] else ''
            energy_data[smiles] = energy
            
            # Also create canonical mapping for equivalent matching
            canonical = canonicalize_multimol_smiles(smiles)
            if canonical:
                if canonical not in canonical_to_originals:
                    canonical_to_originals[canonical] = []
                canonical_to_originals[canonical].append(smiles)
    
    return energy_data, canonical_to_originals


def normalize_equiv_forms(equiv_smiles):
    """
    Normalize equivalent forms to a canonical representation (sorted tuple).
    
    Args:
        equiv_smiles: String or list of equivalent SMILES forms
        
    Returns:
        Tuple of sorted equivalent forms (for use as dictionary key)
    """
    if isinstance(equiv_smiles, list):
        return tuple(sorted(equiv_smiles))
    elif equiv_smiles:
        return (equiv_smiles,)
    else:
        return tuple()


def get_best_energy_from_forms(forms, energy_data, smiles_to_all_equivs=None, canonical_to_originals=None):
    """
    Get the best (lowest) energy from a list of equivalent forms.
    
    Uses equivalence mapping and canonicalization to match different SMILES representations
    of the same molecule).
    
    Args:
        forms: List of SMILES strings to check for energies
        energy_data: Dictionary mapping SMILES to energy values
        smiles_to_all_equivs: Optional dictionary mapping SMILES to set of all equivalent forms
        canonical_to_originals: Optional dictionary mapping canonical SMILES to list of original SMILES
    
    Returns:
        Best (lowest) energy value found, or None if no valid energy found
    """
    best_energy = None
    best_energy_val = float('inf')
    
    # Collect all forms to check (original forms + all equivalents)
    all_forms_to_check = set(forms)
    
    # Add equivalent forms from equivalence mapping
    if smiles_to_all_equivs:
        for form in forms:
            if form in smiles_to_all_equivs:
                all_forms_to_check.update(smiles_to_all_equivs[form])
    
    # Also add canonicalized versions
    for form in list(all_forms_to_check):
        canonical = canonicalize_multimol_smiles(form)
        if canonical:
            all_forms_to_check.add(canonical)
            # If we have canonical mapping, add all originals that match
            if canonical_to_originals and canonical in canonical_to_originals:
                all_forms_to_check.update(canonical_to_originals[canonical])
    
    # Check all collected forms for energies
    for form in all_forms_to_check:
        if form in energy_data and energy_data[form]:
            try:
                energy_val = float(energy_data[form])
                if energy_val < best_energy_val:
                    best_energy_val = energy_val
                    best_energy = energy_data[form]
            except (ValueError, TypeError):
                raise ValueError(f"Invalid energy value for {form}: {energy_data[form]}")
    
    return best_energy


def merge_energies(energy_file, equiv_file, output_file):
    """
    Merge energy data with equivalence mappings.
    
    Calculates [H+] energy as the difference between [OH3+] and O if available.
    
    Args:
        energy_file: Path to CSV file with smiles and energy columns
        equiv_file: Path to CSV file with smiles (real) and equivalent columns
        output_file: Path to output CSV file with smiles and energy columns
    """
    real_to_equiv, all_real_smiles, smiles_to_all_equivs = load_equivalence_mapping(equiv_file)
    
    energy_data, canonical_to_originals = load_energy_data(energy_file)
    
    # Group real smiles by their normalized equivalent forms
    # This ensures real smiles with the same equivalent forms get the same energy
    equiv_groups = {}
    for real_smiles in all_real_smiles:
        equiv_smiles = real_to_equiv.get(real_smiles, '')
        normalized_equiv = normalize_equiv_forms(equiv_smiles)
        
        if normalized_equiv not in equiv_groups:
            equiv_groups[normalized_equiv] = []
        equiv_groups[normalized_equiv].append(real_smiles)
    
    # For each group, find the best energy from all equivalent forms
    group_energies = {}
    for normalized_equiv, real_smiles_list in equiv_groups.items():
        all_forms = list(normalized_equiv)
        for real_smiles in real_smiles_list:
            if real_smiles in energy_data and energy_data[real_smiles]:
                all_forms.append(real_smiles)
        
        # Get the best energy from all these forms (using equivalence mapping and canonicalization)
        best_energy = get_best_energy_from_forms(all_forms, energy_data, smiles_to_all_equivs, canonical_to_originals)
        group_energies[normalized_equiv] = best_energy if best_energy else ''
    
    # Process all unique real smiles
    results = []
    for real_smiles in sorted(all_real_smiles):
        equiv_smiles = real_to_equiv.get(real_smiles, '')
        normalized_equiv = normalize_equiv_forms(equiv_smiles)
        
        # Use the group energy (which is the best from all equivalent forms)
        energy = group_energies.get(normalized_equiv, '')
        
        results.append({
            'smiles': real_smiles,
            'energy': energy
        })
    
    # Calculate H+ energy as difference between [OH3+] and O (only if [H+] is present)
    h_plus_smiles = '[H+]'
    o_smiles = 'O'
    oh3_plus_smiles = '[OH3+]'
    
    # Find H+ entry in results
    h_plus_idx = None
    for i, result in enumerate(results):
        if result['smiles'] == h_plus_smiles:
            h_plus_idx = i
            break
    
    # Only calculate if [H+] is needed (present in results)
    if h_plus_idx is not None:
        # Get energies for O and [OH3+]
        o_energy = None
        oh3_plus_energy = None
        
        for result in results:
            if result['smiles'] == o_smiles and result['energy']:
                try:
                    o_energy = float(result['energy'])
                except (ValueError, TypeError):
                    pass
            elif result['smiles'] == oh3_plus_smiles and result['energy']:
                try:
                    oh3_plus_energy = float(result['energy'])
                except (ValueError, TypeError):
                    pass
        
        # Calculate H+ energy if we have both O and [OH3+] energies
        if o_energy is not None and oh3_plus_energy is not None:
            h_plus_energy = oh3_plus_energy - o_energy
            results[h_plus_idx]['energy'] = str(h_plus_energy)
    
    # Filter out entries without energy
    results_with_energy = []
    missing_count = 0
    for result in results:
        if result['energy'] and result['energy'].strip():
            results_with_energy.append(result)
        else:
            missing_count += 1
    
    # Write output (only entries with energy)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['smiles', 'energy'])
        writer.writeheader()
        writer.writerows(results_with_energy)
    
    print(f'Processed {len(results)} molecules')
    print(f'Missing energy: {missing_count} molecules')
    print(f'Output written to {output_file} with {len(results_with_energy)} molecules')


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description='Merge energy data with equivalence mappings. '
                    'Outputs smiles and energy using real smiles (not equivalent forms).'
    )
    parser.add_argument(
        'energy_file',
        type=Path,
        help='CSV file with smiles and energy columns'
    )
    parser.add_argument(
        'equivalence_file',
        type=Path,
        help='CSV file with smiles (real) and equivalent columns'
    )
    parser.add_argument(
        'output_file',
        type=Path,
        help='Output CSV file with smiles and energy columns'
    )
    
    args = parser.parse_args()
    
    if not args.energy_file.exists():
        parser.error(f"Energy file not found: {args.energy_file}")
    
    if not args.equivalence_file.exists():
        parser.error(f"Equivalence file not found: {args.equivalence_file}")
    
    merge_energies(args.energy_file, args.equivalence_file, args.output_file)


if __name__ == '__main__':
    main()
