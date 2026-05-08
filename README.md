Python code and generation scripts for the paper ‘Euclidean Embedding of Reaction Networks for Dimensionality Reduction of Chemical Spaces’. This includes scripts for generating reaction networks with colibri2, post-processing them to add karc edge weights, and constructing and evaluating Euclidean embeddings of reaction-network graphs stored in GraphML format (typically *.graphml.bz2).

The repository includes:
- linear embedding workflows (`cMDS`, `LMDS`)
- nonlinear embedding workflows (for example `MDS`, `PCA`, `KernelPCA`, `t-SNE`, `Isomap`, `SpectralEmbedding`, `UMAP`)
- distance-preserving encoder-based embedding workflow driven by TOML config
- metric/error analysis utilities
- reaction path finder utilities in graph and embedded spaces
- helper scripts for molecule/energy preprocessing

## Repository Layout

- `bin/linear_embeddings.py` - linear embedding pipeline
- `bin/nonlinear_embeddings.py` - nonlinear embedding pipeline
- `bin/encoder_embeddings.py` - configurable encoder training + embedding export
- `bin/error_metrics.py` - computes embedding quality/path metrics
- `bin/graphml_reaction_path_finder.py` - path finding directly on GraphML graph
- `bin/embedded_space_reaction_path_finder.py` - path finding in embedding space
- `bin/nonlinear_de_finder.py` - parameter fitting helper
- `bin/subsample_cmds_embedding.py` - subsampling/transform utility for cMDS embeddings
- `Molecule-Energy-Calculation/` - scripts to extract molecules, detect equivalents, merge energies, and add edge weights
- `Networks/` - reaction networks (RNs) from the paper, as `*.graphml.bz2`.
    - **Unweighted** variants have no karc edge weights
    - **weighted** variants have karc edge weights

## Requirements

See `requirements.txt`.

Core dependencies include:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `networkx`
- `networkit`
- `torch`
- `rdkit`
- `openbabel`
- `PyYAML`
- `umap-learn` (optional for UMAP mode)
- `colibri2`

## Full Workflow (Colibri2 -> Weighted Graph -> Embeddings -> Metrics)

This section turns the draft workflow into a runnable template.

### 0) Set variables

```bash
# Colibri2 service ports
PORT_SQL=1090
PORT_REDIS=1091

# Inputs/config
SET_FILE="/absolute/path/to/Molecule-Energy-Calculation/settings.toml"
RULES_FILE="/absolute/path/to/Molecule-Energy-Calculation/rules.toml"
EQUIV_RULES="/absolute/path/to/Molecule-Energy-Calculation/equivalence_rules.yml"
INITIAL_FLASK="C=C.O"

# Reaction network short name and reactant/product targets
RN="C2H6O"
R="Reactants"
P="Products"
```

### 1) Build/export reaction network with Colibri2

```bash
./Molecule-Energy-Calculation/initialize3 "$PORT_SQL" "$PORT_REDIS" "$SET_FILE" "$INITIAL_FLASK"
./Molecule-Energy-Calculation/network3 "$PORT_SQL" "$PORT_REDIS" "$SET_FILE" "$RULES_FILE"
./Molecule-Energy-Calculation/export3 "$PORT_SQL" "$SET_FILE" "${RN}.graphml.bz2"
```

### 2) Molecule extraction, equivalence, and energy merge

```bash
# Extract molecules from graph
python Molecule-Energy-Calculation/extract_mols.py -i "${RN}.graphml.bz2" -o "${RN}_molecules.csv"

# Find equivalence classes and unique representatives
python Molecule-Energy-Calculation/find_equivalent_mols.py \
  -i "${RN}_molecules.csv" \
  -r "$EQUIV_RULES" \
  -o "${RN}_molecules_equivalent.csv" \
  -u "${RN}_molecules_unique.csv"

# Submit, compute, and collect energies for unique molecules
./Molecule-Energy-Calculation/submit3 "$PORT_SQL" "$PORT_REDIS" "$SET_FILE" "${RN}_molecules_unique.csv"
./Molecule-Energy-Calculation/compute3 "$PORT_SQL" "$PORT_REDIS" "$SET_FILE"
./Molecule-Energy-Calculation/collect3 "$PORT_SQL" "$SET_FILE" "${RN}_molecules_unique_energies.csv"

# Merge energies with equivalence mapping
python Molecule-Energy-Calculation/merge_energies.py \
  "${RN}_molecules_unique_energies.csv" \
  "${RN}_molecules_equivalent.csv" \
  "${RN}_molecules_merged_energies.csv"
```

### 3) Create reversible weighted graph

```bash
python Molecule-Energy-Calculation/make_reversable.py \
  -o "${RN}_graph_reversable.graphml.bz2" \
  "${RN}.graphml.bz2"

python Molecule-Energy-Calculation/add_edge_weights.py \
  "${RN}_graph_reversable.graphml.bz2" \
  "${RN}_molecules_merged_energies.csv" \
  -o "${RN}_graph_weighted.graphml.bz2"
```

### 4) Graph-based reaction path search

```bash
python bin/graphml_reaction_path_finder.py \
  -gfile "${RN}_graph_weighted.graphml.bz2" \
  -r "$R" \
  -p "$P" \
  --delta 0 \
  > "${RN}_paths.txt"
```

### 5) Embed RNs into Euclidean Space

```bash
# Linear example
python bin/linear_embeddings.py "${RN}_graph_weighted.graphml.bz2"

# Nonlinear example
python bin/nonlinear_embeddings.py "${RN}_graph_weighted.graphml.bz2"

# Encoder example (edit config first)
python bin/encoder_embeddings.py example_encoder_config.toml
```

### 6) Evaluate embedding quality

```bash
# CSV embedding output example
python bin/error_metrics.py "${RN}_graph_weighted.graphml.bz2" \
  --csv "${RN}_LMDS_embedding.csv" \
  --error-metrics all \
  --output "${RN}_weighted_error_metrics.csv"
```

### 7) Nonlinear optimal embedding dimension estimation

```bash
python bin/nonlinear_de_finder.py "${RN}_mMDS_c096_weighted_error_metrics.csv"
```

### 8) Embedded-space reaction path search

```bash
python bin/embedded_space_reaction_path_finder.py \
  "${RN}_encoder_embedding.npz" \
  "${RN}_graph_weighted.graphml.bz2" \
  -r "$R" \
  -p "$P" \
  --delta 0 \
  --output "${RN}_embedded_paths.txt"
```

## Input/Output Notes

- Main input format is GraphML (compressed and uncompressed variants are used in scripts).
- Several scripts support cached distance matrices in `NPZ`.
- Embedding outputs are typically exported as CSV and/or NPZ.

## Reproducibility

- `bin/encoder_embeddings.py` supports explicit seed configuration.
- For deterministic CUDA behavior it sets `CUBLAS_WORKSPACE_CONFIG` when needed.
