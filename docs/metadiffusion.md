# Metadiffusion inputs

Metadiffusion enables enhanced sampling during Boltz-2 structure prediction by applying exploration potentials and steering forces based on collective variables (CVs).

> [!IMPORTANT]  
> Metadiffusion is **NOT COMPATIBLE** with Boltz-1

## Overview

Metadiffusion extends Boltz's diffusion process with:

- **Optimise (`opt`)**: Push any CV toward lower or higher values
- **Steering (`steer`)**: Harmonic restraints that guide structures toward target CV values
- **Exploration (`explore`)**: Metadynamics (hills) or repulsion potentials for enhanced sampling
- **SAXS (`saxs`)**: Small-angle X-ray scattering P(r) profile matching
- **Chemical Shift (`chemical_shift`)**: NMR CA/CB chemical shift matching using CheShift algorithm

All configurations are specified in the `metadiffusion` section of a default Boltz-2 input YAML file.

> [!NOTE]  
> Examples used in the paper are in examples/metadiffusion

> [!NOTE]
> Some CVs, such as `pair_rmsd`, require multiple samples. You MUST specify `--diffusion_samples <num_samples>` when running Boltz-2 for this behaviour to work.

---

## Quick Start

This is a quick start to generate diverse protein conformations, steer to a specified Rg for a chain, and explore SASA. For typical use, only one of the three metadiffusion parameters might be specified. This example serves only to demonstrate that they can be chained in a modular fashion.

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MALLKANKDLISAGLKEFSVLLNQQVFNDPLVSEEDMVTVVEDWMNFYINYYRQQVTGEPQERDKALQELRQELNTLANPFLAKYRDFLKS

  - protein:
      id: B
      sequence: GSHMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR

  - ligand:
      id: C
      ccd: HEM

metadiffusion:
  # Steer chain A to Rg of 10
  - steer:
      collective_variable: rg
      target: 10
      strength: 1.0
      groups: ["A"]
      ensemble: true
      warmup: 0.4
      cutoff: 0.7

  # Maximise structural diversity between samples
  - opt:
      collective_variable: pair_rmsd
      strength: 1.0
      cutoff: 0.8

  # Explore the SASA of chain A
  - explore:
      type: repulsion
      collective_variable: sasa
      strength: 0.2
      groups: ["A"]
      sigma: 5
      warmup: 0.1
      cutoff: 0.8
```

You can save the above example as `input.yaml` and run it with `boltz predict --use_msa_server --diffusion_samples 16 input.yaml`

> [!IMPORTANT]
> Different proteins respond to biases differently owing to the Boltz-2 prior. You can increase/decrease the strength, and/or remove highly flexible N or C-termini.

---

## Configuration

Metadiffusion is designed as an add-on to Boltz-2's default YAML.

The `metadiffusion` section is a list of configuration blocks. Each block specifies one of: `opt`, `steer`, `explore`, `saxs`, or `chemical_shift`.

For collective variables and selection algebra, please see collective_variables.md and selection_algebra.md respectively.

### Optimise (opt)

Pushes the system toward lower (or higher) values of any collective variable.

```yaml
- opt:
    collective_variable: <cv_name>    # Required: CV to optimize
    strength: <float>                 # Positive=maximise, negative=minimise (default: 1.0)
    guidance_interval: <int>          # Apply gradient every N steps (default: 1)
    warmup: <float>                   # Start after this diffusion fraction (default: 0.0)
    cutoff: <float>                   # Stop after this diffusion fraction (default: 0.75)
    # For energy CV only:
    # For CVs requiring region/atom selection:
    groups: [<region_specs>]          # Region specs for group-based CVs (rg, asphericity, etc.)
    rmsd_groups: [<region_specs>]     # Alignment regions for pair_rmsd_grouped CV
    region1: <region_spec>            # For distance/angle/dihedral CVs (preferred)
    region2: <region_spec>            # Supports: "A", "A:1-50", "A:5:CA", "A:1-50:CA", "A::CA"
    reference_structure: <path>       # For RMSD/native_contacts CVs
```

**Example: Minimise SASA**
```yaml
- opt:
    collective_variable: sasa
    strength: -5.0
```

**Example: Maximise diversity (pair_rmsd)**
```yaml
- opt:
    collective_variable: pair_rmsd
    strength: 1.0
```

> [!TIP]
> Some CVs, such as pair_rmsd, require multiple samples (since they directly compare the ij pairs to calculate RMSD). You must specify `--diffusion_samples <number>` when you run Boltz.

### Steering (steer)

Applies a harmonic potential to guide a CV toward a target value.

```yaml
- steer:
    collective_variable: <cv_name>    # Required: CV to steer (see Valid CV Names)
    target: <float>                   # Target value (required unless using target_from_saxs)
    target_from_saxs: <path>          # Extract target from SAXS P(r) file (Rg CV only)
    auto_rg_scale: <float>            # Scale factor for Rg from SAXS (default: 1.0)
    strength: <float>                 # Optional: Force constant (default: 1.0)
    guidance_interval: <int>          # Optional: Apply every N steps (default: 1)
    warmup: <float>                   # Optional: Start steering after this fraction (default: 0.0)
    cutoff: <float>                   # Optional: Stop steering after this fraction (default: 0.75)
    bias_clip: <float>                # Optional: Max per-atom displacement from this potential (Å)
    ensemble: <bool>                  # Optional: Per-ensemble loss (default: false)
    # For geometric CVs (distance, angle, dihedral):
    region1: <region_spec>            # First region (see Atom Selection Syntax)
    region2: <region_spec>            # Second region
    region3: <region_spec>            # Third region (for angle)
    region4: <region_spec>            # Fourth region (for dihedral)
    reference_structure: <path>       # For RMSD/native_contacts CVs
```

#### Per-sample vs. per-ensemble

The `ensemble` parameter controls how the steering loss is computed:

| Mode | `ensemble` | Description |
|------|------------|-------------|
| **Per-sample** | `false` (default) | Each sample steered independently toward target |
| **Per-ensemble** | `true` | CV averaged across samples, then loss computed from mean, requires `--diffusion_samples <num_samples>` to be specified to be effective |

**Example: Steer distance between two regions (domain COMs)**
```yaml
- steer:
    collective_variable: distance
    region1: "A:50-90:CA"       # Tip of lobe 1
    region2: "A:280-320:CA"     # Tip of lobe 2
    target: 20.0                # Close together (closed state)
    ensemble: true
    strength: 4.0
    bias_clip: 1.0
```

**Example: Steer hinge angle (protein open/closed)**
```yaml
- steer:
    collective_variable: angle
    region1: "A:1-140"          # Lobe 1 (N-terminal domain)
    region2: "A:150-200"        # Hinge region (vertex/pivot)
    region3: "A:210-375"        # Lobe 2 (C-terminal domain)
    target: 1.57                # 90 degrees in radians
    ensemble: false
    strength: 5.0
    bias_clip: 1.0
```

**Example: Steer Rg toward experimental SAXS value**
```yaml
- steer:
    collective_variable: rg
    target_from_saxs: experimental.out   # Extract Rg from P(r)
    auto_rg_scale: 0.8                   # Target 80% of experimental Rg
    warmup: 0.1
    cutoff: 0.8
    strength: 1.0
```

### Exploration (`explore`)

Applies exploration potentials for enhanced conformational sampling.

**Required fields:**
- `collective_variable`: Must be one of the 26 valid CV names
- `type`: Must be `"hills"` or `"repulsion"`


#### Repulsion

Pushes samples apart in CV space for structural diversity. Uses pairwise RMSD between samples to encourage diverse conformations.

```yaml
- explore:
    type: repulsion
    collective_variable: pair_rmsd    # Required: use pair_rmsd CV
    strength: <float>                 # Repulsion strength (default: 256.0)
    sigma: <float>                    # Repulsion width in Å (default: 5.0)
    warmup: <float>                   # Fraction of steps before activation (default: 0.2)
    cutoff: <float>                   # Diffusion time cutoff (default: 0.75)
    groups: [<region_specs>]          # Region specs for group-based CVs (rg, asphericity, etc.)
```

#### Hills (metadynamics-like)

Deposits Gaussian hills to escape free energy minima. This implements metadynamics-like adaptive exploration.

```yaml
- explore:
    type: hills
    collective_variable: <cv_name>    # Any CV (rg, distance, angle, energy, etc.)
    hill_height: <float>              # Height of Gaussian hills (default: 0.5)
    sigma: <float>                    # Width of Gaussian hills (default: 2.0)
    hill_interval: <int>              # Steps between hill deposits (default: 5)
    well_tempered: <bool>             # Use well-tempered scaling (default: false)
    bias_factor: <float>              # Well-tempered bias factor (default: 10.0)
    max_hills: <int>                  # Maximum hills to store (default: 1000)
    warmup: <float>                   # Start after this diffusion fraction (default: 0.2)
    cutoff: <float>                   # Stop after this diffusion fraction (default: 0.75)
    bias_clip: <float>                # Max per-atom displacement from this potential (Å)
    # For geometric CVs (distance, angle, dihedral):
    region1: <region_spec>            # First region (see Atom Selection Syntax)
    region2: <region_spec>            # Second region
    region3: <region_spec>            # Third region (for angle/dihedral)
    region4: <region_spec>            # Fourth region (for dihedral)
```

**Example with distance CV:**
```yaml
- explore:
    type: hills
    collective_variable: distance
    region1: "A:50-90:CA"       # First domain COM
    region2: "A:280-320:CA"     # Second domain COM
    hill_height: 0.5
    sigma: 5.0                  # 5 Angstrom width
    well_tempered: true
    bias_factor: 10.0
```

### SAXS

Matches experimental small-angle X-ray scattering P(r) profiles.

```yaml
- saxs:
    pr_file: <path>                   # Required: Path to GNOM .out file
    loss_type: <string>               # Loss function (default: "mse")
    strength: <float>                 # Force constant (default: 1.0)
    guidance_interval: <int>          # Apply every N steps (default: 1)
    warmup: <float>                   # Start steering after this fraction (default: 0.0)
    cutoff: <float>                   # Stop steering after this fraction (default: 0.9)
    sigma_bin: <float>                # P(r) binning smoothness (default: 0.5)
    # Uniform resampling (recommended for W1/W2):
    bins: <int>                       # Resample to N uniform bins (default: None = use file)
    bins_range: [r_min, r_max]        # Explicit range in Angstroms (default: from file)
    # W2-specific:
    # Rg loss-specific:
    rg_scale: <float>                 # Scale experimental Rg (default: 1.0)
```

**Uniform Resampling:**

GNOM output files have variable bin counts (50-200+) depending on the experiment. For Cramer/W1/W2 losses, gradient can vanish with too many bins. Use `bins` to resample P(r) to a uniform grid:

```yaml
- saxs:
    pr_file: experimental.out
    loss_type: cramer
    bins: 64                          # Resample to 64 uniform bins
    strength: 32
```

**Supported loss types:**

| Loss Type | Description | Best For |
|-----------|-------------|----------|
| `cramer` | Cramer distance (CDF-based) | Robust shape matching |
| `mse` | Mean Squared Error | Point-by-point matching |
| `rg` | Rg penalty from P(r) | Per-ensemble Rg matching |

**Composing Losses:**

Combine multiple loss functions by adding separate SAXS entries:

```yaml
metadiffusion:
  # Cramer for global shape matching
  - saxs:
      pr_file: experimental.out
      loss_type: cramer
      strength: 32

  # Rg penalty for correct overall size
  - saxs:
      pr_file: experimental.out
      loss_type: mse
      strength: 1.0
      rg_scale: 1.0
```

### Chemical Shift

Matches experimental NMR chemical shift data for CA and CB nuclei using the CheShift-2 algorithm. CheShift-2 predicts chemical shifts from backbone and sidechain torsion angles (phi, psi, chi1) with chi2 averaging for residues with rotamers.

> **Important:** Only CA and CB chemical shifts are supported. Attempts to use unsupported nuclei will raise `NotImplementedError`.

```yaml
- chemical_shift:
    ca_shifts: <path>                 # Path to CA shift file (PLUMED format)
    cb_shifts: <path>                 # Path to CB shift file (PLUMED format)
    strength: <float>                 # Force constant (default: 1.0)
    loss_type: <string>               # "chi_ccc" or "mse" (default: "chi")
    guidance_interval: <int>          # Apply every N steps (default: 1)
    warmup: <float>                   # Start steering after this fraction (default: 0.0)
    cutoff: <float>                   # Stop steering after this fraction (default: 0.9)
    bias_clip: <float>                # Optional: limit per-atom gradient magnitude
```

**Loss types:**

| Loss Type | Description |
|-----------|-------------|
| `ccc` | Concordance correlation coefficient |
| `chi` | Chi-squared with sigma=1.0 ppm for both CA and CB |
| `mse` | Mean Squared Error |

**Example: Steer toward experimental CA and CB shifts**
```yaml
metadiffusion:
  - chemical_shift:
      ca_shifts: CAshifts.dat
      cb_shifts: CBshifts.dat
      strength: 4.0
      loss_type: ccc
      warmup: 0.1
      cutoff: 0.9
```

**Example: CA shifts only**
```yaml
metadiffusion:
  - chemical_shift:
      ca_shifts: CAshifts.dat
      loss_type: ccc
      strength: 4.0
```
---

## Auxiliary sampling parameters

These parameters control the diffusion sampling process and can help stabilise metadiffusion runs with strong steering forces.

### noise_scale

Controls the stochastic noise injection during diffusion sampling. This is a **top-level YAML option**.

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVLSPADKTN...

# Top-level Boltz options
noise_scale: 0.2  # Reduced stochasticity
denoise_clip: 256  # Limit denoising displacement

metadiffusion:
  - opt:
      collective_variable: pair_rmsd
      strength: 2.0
```

### denoise_clip

Limits the maximum per-atom displacement in each denoising step. This prevents structure explosion when using strong steering forces.

This is a **top-level** parameter like `noise_scale`, not inside the `metadiffusion:` list.

```yaml
noise_scale: 0.2
denoise_clip: 256

metadiffusion:
  - opt:
      collective_variable: rg
      strength: 2.0  # Positive = maximize Rg
      warmup: 0.1
      cutoff: 0.7
```

### bias_clip

Limits the maximum per-atom displacement from guidance potentials. Unlike `denoise_clip`, this is specified **per-potential** for fine-grained control.

```yaml
metadiffusion:
  - opt:
      collective_variable: pair_rmsd
      strength: 1.0  # Positive = maximise diversity
      bias_clip: 0.5  # Limit max displacement to 1Å per biasing iteration

  - opt:
      collective_variable: rg
      strength: -1.0  # Negative = minimise Rg (compact)
      bias_clip: 1.0  # Limit max displacement to 1Å per biasing iteration
      warmup: 0.1
      cutoff: 0.8
```

### total_bias_clip

Limits the maximum per-atom displacement from **all potentials combined**. This is a top-level parameter that acts as a global safety limit after individual `bias_clip` limits have been applied.

```yaml
metadiffusion:
  - total_bias_clip: 1.0  # Global limit on combined guidance (Å)

  - opt:
      collective_variable: pair_rmsd
      strength: 1.0

  - saxs:
      pr_file: data.out
      strength: 10.0
```

### guidance_mode

Controls when and how guidance gradients are computed and applied during the diffusion loop.

```yaml
metadiffusion:
  - guidance_mode: combine  # Default: compute both pre and post, add them
  - steer:
      collective_variable: rg
      target: 25.0
      strength: 1.0
      warmup: 0.1
      cutoff: 0.8
```

**Available modes:**

| Mode | Description |
|------|-------------|
| `combine` | **(default)** Compute gradients on both noisy and denoised coordinates, add displacement vectors, apply to x₀ |
| `post` | Compute gradient on denoised x₀ only, apply to x₀ |
| `pre` | Compute gradient on noisy coordinates only, apply before model |
