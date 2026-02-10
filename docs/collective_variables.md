## Collective Variables

Metadiffusion supports collective variables (CVs) for steering and exploration. **All CVs can be used in both steer and explore configurations.** Most CVs support region selection via `region1`/`region2` or `groups` parameters.

---

### Structural CVs

#### `rg` - Radius of Gyration
Measures the root-mean-square distance of atoms from the center of mass: `Rg = sqrt(mean(||r_i - com||²))`. Indicates overall protein compactness.
- **Units**: Å
- **Parameters**: `groups` (optional chain selection)

```yaml
# Distance between domain COMs
- steer:
    collective_variable: rg
    target: 40.0
    strength: 4.0
    groups: "A" # only target chain A
    warmup: 0.1
    cutoff: 0.8
```

> [!TIP]
>This is the classic coordinate-based Rg formula. It differs from the SAXS-derived Rg (`loss_type: rg`) which computes Rg from the P(r) distribution.

#### `distance` - Distance Between Points/Regions
Euclidean distance between two points. When using `region1`/`region2`, computes distance between centers of mass, providing gradients to all atoms in each region.
- **Units**: Å
- **Parameters**: `region1`, `region2` (region specs) or `atom1_idx`, `atom2_idx` (indices)

```yaml
# Distance between domain COMs
- steer:
    collective_variable: distance
    region1: "A:1-50"
    region2: "A:100-150"
    target: 40.0
    strength: 2.0
```

> [!TIP]
> Steering singular atoms may not be effective due to the biases only applied to those atoms, whereas connected topology remains unchanged. Steering whole domains/regions are typically more effective.

#### `min_distance` - minimum distance
Soft minimum distance between two atom groups. Uses differentiable soft-min approximation. Useful for ensuring separation or contact between domains/chains.
- **Units**: Å
- **Parameters**: `region1`, `region2` (required), `softmin_beta` (default: 10.0)

```yaml
# Keep two domains separated
- opt:
    collective_variable: min_distance
    region1: "A:1-50"
    region2: "A:100-150"
    strength: 2.0         # Maximize minimum distance
```

> [!TIP]
> Steering singular atoms may not be effective due to the biases only applied to those atoms, whereas connected topology remains unchanged. Steering whole domains/regions are typically more effective.

#### `max_diameter` - Maximum Diameter
The largest pairwise distance between any two atoms. Represents the maximum extent of the structure. Uses soft-max approximation for differentiability.
- **Units**: Å
- **Parameters**: `groups` (optional)
- **Typical values**: 30-200 Å

> [!TIP]
> Steering singular atoms may not be effective due to the biases only applied to those atoms, whereas connected topology remains unchanged. Steering whole domains/regions are typically more effective.

#### `asphericity` - Shape Asphericity
Measures deviation from spherical shape using gyration tensor eigenvalues: A = (λ₁-λ₂)² + (λ₂-λ₃)² + (λ₁-λ₃)². Zero for perfect sphere, increases for elongated or flattened shapes.
- **Units**: dimensionless
- **Parameters**: `groups` (optional)
- **Typical values**: 0 (sphere) to >100 (elongated)

---

### Angle/Dihedral CVs

#### `angle` - Bond Angle
Angle formed by three points. When using `region1`/`region2`/`region3`, the center of mass of each region is used, providing dense gradients to all atoms in each region.
- **Units**: radians
- **Parameters**: `region1`, `region2` (vertex), `region3` for region mode; `atom1_idx`, `atom2_idx`, `atom3_idx` for single-atom mode
- **Optional**: `max_hops` (default: 0), `decay` (default: 0.5) - gradient propagation through bonds (disabled by default for speed)  - EXPERIMENTAL!

```yaml
# Angle between three domain COMs (recommended)
metadiffusion:
  - total_bias_clip: 8.0
  - steer:
      collective_variable: angle
      region1: "A:1-30:CA"   # N-terminal CA atoms
      region2: "A:35-45:CA"  # Hinge region (vertex)
      region3: "A:50-80:CA"  # C-terminal CA atoms
      target: 1.57           # 90 degrees in radians
      strength: 4.0
```

#### `dihedral` - Dihedral/Torsion Angle
Torsion angle defined by four points. When using `region1`-`region4`, the center of mass of each region is used. **Note**: Use `explore` instead of `opt` for dihedral steering due to angle wrapping at ±180°.
- **Units**: radians
- **Parameters**: `region1`, `region2`, `region3`, `region4` for region mode; `atom1_idx`-`atom4_idx` for single-atom mode
- **Optional**: `max_hops` (default: 0), `decay` (default: 0.5) - gradient propagation through bonds (disabled by default for speed) - EXPERIMENTAL!

```yaml
# Dihedral with explore (recommended for dihedral)
metadiffusion:
  - total_bias_clip: 8.0
  - explore:
      type: repulsion
      collective_variable: dihedral
      region1: "A:1-20"
      region2: "A:25-40"
      region3: "A:45-60"
      region4: "A:65-80"
      strength: 256.0
      sigma: 0.5
```

```yaml
# Use explore for dihedral_enhanced
metadiffusion:
  - total_bias_clip: 8.0
  - explore:
      explore_type: repulsion
      collective_variable: dihedral_enhanced
      atom1_idx: 0
      atom2_idx: 1
      atom3_idx: 2
      atom4_idx: 3
      strength: 256.0
      sigma: 0.5
```

---

### RMSD CVs

#### `rmsd` - Root Mean Square Deviation
RMSD to a reference structure after optimal superposition (Kabsch algorithm). Measures structural similarity to a known conformation.
- **Units**: Å
- **Parameters**: `reference_structure` (required), `groups` (optional)

#### `pair_rmsd` - Pairwise RMSD
Mean RMSD between all pairs of samples in the ensemble. Measures conformational diversity - higher values indicate more diverse structures.
- **Units**: Å
- **Parameters**: `groups` (optional)

> [!IMPORTANT]  
> `pair_rmsd` computes the pair ij RMSDs with the Kabsch alignment. As a result, you MUST specify multiple samples during diffusion for pairs to exist. This can be done by specifying `--diffusion_samples <num_samples>`

#### `pair_rmsd_grouped` - Grouped Pairwise RMSD
Pairwise RMSD with separate alignment and RMSD groups. Aligns structures using one group (e.g., protein backbone) then computes RMSD on a different group (e.g., ligand). Useful for measuring ligand pose diversity after protein alignment.
- **Units**: Å
- **Parameters**: `groups` (atoms to compute RMSD on), `rmsd_groups` (atoms to align by)

```yaml
- opt:
    collective_variable: pair_rmsd_grouped
    groups: [B]           # Compute RMSD on ligand
    rmsd_groups: [A]      # Align by protein
    strength: 2.0         # Maximize ligand diversity
```

> [!IMPORTANT]  
> `pair_rmsd_grouped` computes the pair ij RMSDs with the Kabsch alignment. As a result, you MUST specify multiple samples during diffusion for pairs to exist. This can be done by specifying `--diffusion_samples <num_samples>`

---

### Contact CVs

#### `sasa` - Solvent Accessible Surface Area
Differentiable approximation of solvent accessible surface area (SASA). Measures the surface area of the molecule accessible to solvent molecules (conceptually, the area traced by rolling a probe sphere over the van der Waals surface). Higher SASA indicates more exposed/unfolded conformations; lower SASA indicates more compact/buried structures.

Two methods are available:
- **LCPO** (default): Analytical Linear Combination of Pairwise Overlaps approximation. More accurate, based on pairwise atomic overlaps.
- **Coordination**: Fast neighbor-counting approximation. Atoms with more neighbors are considered more buried.

- **Units**: Å²
- **Parameters**:
  - `probe_radius` (default: 1.4 Å for water)
  - `sasa_method` ("lcpo" or "coordination", default: "lcpo")

**Example - Steer to target SASA:**
```yaml
metadiffusion:
  - steer:
      collective_variable: sasa
      target: 2000         # Target SASA in Å²
      strength: 1.0
      probe_radius: 1.4
```

---

### Secondary Structure CVs

#### `helix_content` - Helical Content
Fraction of residues in helical conformation based on i→i+4 CA distances (~6.3Å in helix). Quick estimate of alpha-helix content.
- **Units**: 0-1 (fraction)
- **Parameters**: `groups` (optional)

#### `sheet_content` - Sheet Content
Fraction of residues in extended/sheet conformation based on i→i+2 CA distances (~6.7Å in sheet). Quick estimate of beta-sheet content.
- **Units**: 0-1 (fraction)
- **Parameters**: `groups` (optional)
