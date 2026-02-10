## Atom Selection Syntax

Boltz provides a unified region selection syntax used across all CVs. The same formats work for `region1`/`region2` parameters (geometric CVs), `groups`/`rmsd_groups` parameters (group-based CVs), and legacy `atom1`/`atom2` parameters.

### Selection String Formats

| Format | Example | Description |
|--------|---------|-------------|
| `chain` | `A` | All atoms in chain |
| `chain:start-end` | `A:1-50` | All atoms in residue range |
| `chain:resid` | `A:15` | All atoms in single residue |
| `chain:resid:atom` | `A:15:CA` | Single specific atom |
| `chain:start-end:atom` | `A:1-50:CA` | Specific atom type across residue range |
| `chain::atom` | `A::CA` | Specific atom type across whole chain |

### Region Selection (Geometric CVs)

For `distance`, `angle`, `dihedral`, and `min_distance` CVs, use `region1`, `region2`, etc. When a region contains multiple atoms, the **center of mass (COM)** is used.

```yaml
# Distance between CA atoms of two residues
- steer:
    collective_variable: distance
    region1: "A:10:CA"
    region2: "A:50:CA"
    target: 30.0

# Distance between COMs of two domains
- steer:
    collective_variable: distance
    region1: "A:1-50"       # COM of N-terminal domain
    region2: "A:100-150"    # COM of C-terminal domain
    target: 40.0

# Distance between CA backbones of two regions
- steer:
    collective_variable: distance
    region1: "A:1-50:CA"    # COM of N-terminal CA trace
    region2: "A:100-150:CA" # COM of C-terminal CA trace
    target: 35.0

# Distance between whole chains
- steer:
    collective_variable: distance
    region1: "A"            # COM of chain A
    region2: "B"            # COM of chain B
    target: 25.0

# Minimum distance between two regions
- opt:
    collective_variable: min_distance
    region1: "A:1-50"
    region2: "B:1-50"
    target: 5.0
    strength: -1.0          # Minimize distance
```

### Selection algebra

For `rg`, `asphericity`, `pair_rmsd_grouped`, and other group-based CVs, use `groups` (and `rmsd_groups` for alignment). Multiple selections can be combined in a list.

```yaml
# Radius of gyration of CA backbone only
- opt:
    collective_variable: rg
    groups: ["A::CA"]       # All CA atoms in chain A
    target: 15.0

# Rg of specific domain backbone
- opt:
    collective_variable: rg
    groups: ["A:1-100:CA"]  # CA atoms of residues 1-100
    target: 12.0

# Asphericity of multiple chains
- opt:
    collective_variable: asphericity
    groups: ["A", "B"]      # All atoms in chains A and B
    strength: -1.0          # Minimize (more spherical)

# RMSD on ligand, aligned by protein backbone
- opt:
    collective_variable: pair_rmsd_grouped
    groups: ["B"]           # Compute RMSD on chain B (ligand)
    rmsd_groups: ["A::CA"]  # Align using CA atoms of chain A
    target: 2.0
    strength: 1.0           # Maximize diversity

# Combine multiple selections
- opt:
    collective_variable: rg
    groups: ["A:1-50:CA", "A:100-150:CA"]  # Two domain backbones
    target: 20.0
```

### Examples by Use Case

**Inter-domain distance (using COMs):**
```yaml
region1: "A:1-75"
region2: "A:100-175"
```

**Backbone-only metrics:**
```yaml
groups: ["A::CA"]           # CA trace
groups: ["A::N", "A::CA"]   # N and CA atoms
```

**Domain backbone:**
```yaml
groups: ["A:1-50:CA"]       # First 50 residues CA
```

**Multi-chain complex:**
```yaml
groups: ["A", "B", "C"]     # All atoms in 3 chains
```