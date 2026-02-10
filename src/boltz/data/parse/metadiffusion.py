"""
Parser for metadiffusion YAML configuration section.

This module parses the 'metadiffusion' section of Boltz YAML input files
and converts it into configuration dataclasses for potential instantiation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import warnings


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float, handling string inputs."""
    if value is None:
        return default
    if isinstance(value, str):
        return float(value)
    return float(value)


def _to_int(value: Any, default: int = 0) -> int:
    """Convert value to int, handling string inputs."""
    if value is None:
        return default
    if isinstance(value, str):
        return int(float(value))  # Handle "2.0" style strings
    return int(value)


def _warn_region_atom_conflict(config: Dict[str, Any], context: str = "config") -> None:
    """Warn if both regionN and atomN are specified (atomN is silently ignored)."""
    for i in range(1, 5):
        region_key = f"region{i}"
        atom_key = f"atom{i}"
        if config.get(region_key) is not None and config.get(atom_key) is not None:
            warnings.warn(
                f"Both '{region_key}' and '{atom_key}' specified in {context}. "
                f"'{atom_key}' will be ignored ('{region_key}' takes precedence).",
                UserWarning,
            )


@dataclass
class ScalingConfig:
    """Configuration for CV-based gradient scaling.

    Scales the gradient by per-atom weights computed from a scaling CV's gradient magnitude.
    Atoms that contribute more to the scaling CV (higher |dCV/dr|) get more/less steering.

    Multiple scaling CVs can be specified as an array - their weights are multiplied together.

    Example (single scaling CV):
        scaling:
          - collective_variable: rg
            strength: 1.0

    Example (multiple scaling CVs - weights multiplied):
        scaling:
          - collective_variable: rg
            strength: 1.0
          - collective_variable: asphericity
            strength: 0.5

    Example (with warmup/cutoff):
        scaling:
          - collective_variable: rg
            strength: 1.0
            warmup: 0.2   # Start scaling after 20% of diffusion
            cutoff: 0.8   # Stop scaling after 80% of diffusion

    Weight computation:
        1. For each scaling CV, compute gradient magnitude per atom: |dCV/dr|
        2. Convert to weights (invert if strength < 0)
        3. Multiply weights from all scaling CVs together
        4. Normalize per sample so mean(weights) = 1
        5. Apply to steering gradient with magnitude preservation
    """
    collective_variable: str  # CV type from registry (rg, distance, etc.)
    strength: float = 1.0  # + = high grad → high weight, - = inverted
    warmup: float = 0.0  # Start scaling after this fraction of steps (0.0 = from start)
    cutoff: float = 1.0  # Stop scaling after this fraction of steps (1.0 = until end)
    # CV-specific parameters (for CVs that need them)
    groups: Optional[List[str]] = None  # For group-based CVs
    # Legacy atom selection (still supported)
    atom1: Optional[str] = None  # For distance CV
    atom2: Optional[str] = None  # For distance CV
    atom3: Optional[str] = None  # For angle/dihedral CVs
    atom4: Optional[str] = None  # For dihedral CVs
    # New region selection (preferred) - supports "A", "A:1-50", "A:5:CA", "A:1-50:CA", "A::CA"
    region1: Optional[str] = None  # For distance/angle/dihedral CVs
    region2: Optional[str] = None  # For distance/angle/dihedral CVs
    region3: Optional[str] = None  # For angle/dihedral CVs
    region4: Optional[str] = None  # For dihedral CVs
    reference_structure: Optional[str] = None  # For RMSD CV


@dataclass
class ProjectionConfig:
    """Configuration for CV-based gradient projection.

    Projects the gradient onto the direction of the projection CV's gradient.
    Unlike scaling (which redistributes magnitude), projection constrains the
    gradient to act only along directions that change the projection CV.

    Mathematical operation (per atom):
        ê_i = normalize(dCV/dr_i)     # Unit vector in CV gradient direction
        g_projected_i = (g_i · ê_i) × ê_i   # Project onto CV gradient direction
        g_final = (1 - strength) × g_original + strength × g_projected  # Blend
        ||g_final|| = ||g_original||  # Magnitude preserved per sample

    Example (projection only):
        projection:
          - collective_variable: rg
            strength: 1.0

    Example (both scaling AND projection):
        scaling:
          - collective_variable: rg
            strength: 1.0
        projection:
          - collective_variable: distance
            strength: 0.8
        modifier_order: scale_first  # or 'project_first'

    Strength:
        - 0.0: No projection (original gradient unchanged)
        - 1.0: Full projection (gradient fully aligned with CV direction)
        - 0.5: 50% original + 50% projected (blended)

    Direction:
        Controls the sign of the projected gradient along the CV axis:
        - "preserve": Keep the sign from the original steering gradient (default)
        - "toward": Force gradient toward lower CV values
        - "away": Force gradient toward higher CV values

    Zero CV gradient handling:
        Atoms with zero/undefined CV gradient direction receive zero steering.
    Example (with warmup/cutoff):
        projection:
          - collective_variable: rg
            strength: 1.0
            warmup: 0.2   # Start projection after 20% of diffusion
            cutoff: 0.8   # Stop projection after 80% of diffusion
    """
    collective_variable: str  # CV type from registry (rg, distance, etc.)
    strength: float = 1.0  # 0=no projection, 1=full projection
    direction: str = "preserve"  # "preserve", "toward", or "away"
    zero_threshold: float = 1e-8  # Threshold for considering CV gradient as zero
    warmup: float = 0.0  # Start projection after this fraction of steps (0.0 = from start)
    cutoff: float = 1.0  # Stop projection after this fraction of steps (1.0 = until end)
    # CV-specific parameters (for CVs that need them)
    groups: Optional[List[str]] = None  # For group-based CVs
    # Legacy atom selection (still supported)
    atom1: Optional[str] = None  # For distance CV
    atom2: Optional[str] = None  # For distance CV
    atom3: Optional[str] = None  # For angle/dihedral CVs
    atom4: Optional[str] = None  # For dihedral CVs
    # New region selection (preferred) - supports "A", "A:1-50", "A:5:CA", "A:1-50:CA", "A::CA"
    region1: Optional[str] = None  # For distance/angle/dihedral CVs
    region2: Optional[str] = None  # For distance/angle/dihedral CVs
    region3: Optional[str] = None  # For angle/dihedral CVs
    region4: Optional[str] = None  # For dihedral CVs
    reference_structure: Optional[str] = None  # For RMSD CV


@dataclass
class SAXSConfig:
    """Configuration for a single SAXS P(r) loss component.

    Multiple SAXS entries can be specified to compose losses. Each entry
    applies one loss type with its own strength parameter.

    Loss types:
        - 'w1': Wasserstein-1 (Earth Mover's Distance) - fast CDF matching
        - 'w2': Wasserstein-2 (Sinkhorn) - geometric optimal transport
        - 'mse': Mean Squared Error - point-by-point matching
        - 'chi2': Chi-squared - error-weighted matching
        - 'mae': Mean Absolute Error - robust to outliers
        - 'kl': KL divergence - information-theoretic
        - 'rg': Rg penalty - per-ensemble Rg matching from P(r)

    Example composing W2 + MSE:
        metadiffusion:
          - saxs:
              pr_file: data.out
              loss_type: w2
              strength: 256
          - saxs:
              pr_file: data.out
              loss_type: mse
              strength: 65536
    """
    pr_file: str
    loss_type: str = "mse"  # 'w1', 'w2', 'mse', 'chi2', 'mae', 'kl', 'rg'
    strength: float = 1.0  # Force constant for this loss component
    guidance_interval: int = 1  # Apply gradient every N steps
    warmup: float = 0.0  # Start steering after this fraction of steps
    cutoff: float = 0.9  # Stop steering after this fraction of steps
    sigma_bin: float = 0.5  # Gaussian smoothing for P(r) binning
    units: str = "auto"  # 'nm', 'angstrom', or 'auto' (detect from max r value)

    # W2 Sinkhorn parameters (only used for w2 loss_type)
    w2_epsilon: float = 0.1  # Entropic regularization
    w2_num_iter: int = 100  # Sinkhorn iterations

    # Rg scaling (only used for rg loss_type)
    # Target Rg = rg_scale * Rg_experimental
    rg_scale: float = 1.0

    # Use only representative atoms for P(r) computation (much faster for large systems)
    # When True: Uses CA for proteins, C3' for nucleic acids (~50x faster)
    # When False: Uses all atoms (default, more accurate but slow for large systems)
    use_rep_atoms: bool = False

    # Resampling options for uniform binning
    # bins: Number of uniform bins to resample P(r) to (None = use original grid)
    # Recommended: 48-64 for W1/W2 losses to improve gradient quality
    bins: Optional[int] = None
    # bins_range: Optional (r_min, r_max) range in Angstroms (None = use file range)
    bins_range: Optional[Tuple[float, float]] = None

    # Optional CV-based gradient scaling (list of scaling CVs, weights multiplied)
    scaling: Optional[List["ScalingConfig"]] = None

    # Optional CV-based gradient projection (list of projection CVs)
    projection: Optional[List["ProjectionConfig"]] = None

    # Order of applying scaling and projection when both are specified
    # 'scale_first': Apply scaling, then projection
    # 'project_first': Apply projection, then scaling
    modifier_order: str = "scale_first"

    # Per-potential bias tempering: max per-atom displacement from this potential (Å)
    # None means no tempering for this potential
    bias_tempering: Optional[float] = None


@dataclass
class OptConfig:
    """Configuration for CV optimization (minimize or maximize).

    Pushes the system toward lower (minimize) or higher (maximize) CV values.
    The sign of `strength` determines the direction:
        - strength > 0: Maximize CV (push toward higher values)
        - strength < 0: Minimize CV (push toward lower values)
        - |strength|: Controls gradient magnitude

    Example (maximize Rg):
        - opt:
            collective_variable: rg
            strength: 1.0

    Example (maximize diversity with pair_rmsd):
        - opt:
            collective_variable: pair_rmsd
            strength: 0.5
    """
    collective_variable: str  # Any valid CV from VALID_CVS

    # Optimization parameters
    strength: float = 1.0  # Positive=maximize, negative=minimize
    guidance_interval: int = 1
    warmup: float = 0.0
    cutoff: float = 0.75
    log_gradient: bool = False  # Apply log compression for numerical stability

    # CV-specific parameters
    groups: Optional[List[str]] = None  # For group-based CVs
    rmsd_groups: Optional[List[str]] = None  # For pair_rmsd_grouped: atoms to align by
    # Region selection - unified format for all geometric CVs
    # Supports: "A:5:CA" (single atom), "A:1-20" (residue range), "A" (whole chain)
    # Single atom → point, Multiple atoms → center of mass (COM)
    region1: Optional[str] = None  # For distance (2), angle (3), dihedral (4)
    region2: Optional[str] = None
    region3: Optional[str] = None  # For angle/dihedral
    region4: Optional[str] = None  # For dihedral
    reference_structure: Optional[str] = None  # For RMSD, native_contacts

    # Contact parameters
    contact_cutoff: float = 4.5
    selection: str = "all"

    # SASA CV specific parameters
    probe_radius: float = 1.4  # Solvent probe radius in Å (default 1.4 for water)
    sasa_method: str = "lcpo"  # "lcpo" (analytical) or "coordination" (fast)

    # Optional CV-based gradient scaling (list of scaling CVs, weights multiplied)
    scaling: Optional[List["ScalingConfig"]] = None

    # Optional CV-based gradient projection (list of projection CVs)
    projection: Optional[List["ProjectionConfig"]] = None

    # Order of applying scaling and projection when both are specified
    modifier_order: str = "scale_first"

    # Per-potential bias tempering: max per-atom displacement from this potential (Å)
    # None means no tempering for this potential
    bias_tempering: Optional[float] = None


@dataclass
class CVConfig:
    """Configuration for a single collective variable."""
    cv_type: str
    groups: Optional[List[str]] = None
    # Atom selection for distance-like CVs
    atom1: Optional[str] = None  # format: chain:resid:atomname
    atom2: Optional[str] = None
    atom3: Optional[str] = None  # For angle/dihedral CVs
    atom4: Optional[str] = None  # For dihedral CVs
    # Reference structure for RMSD/native_contacts
    reference_structure: Optional[str] = None
    # Contact parameters
    contact_cutoff: float = 4.5
    selection: str = "all"  # hydrophobic / polar / all / custom
    # Per-CV sigma (for multi-CV metadynamics)
    sigma: Optional[float] = None


@dataclass
class ExploreConfig:
    """Configuration for an explore potential (hills/repulsion for conformational exploration)."""
    explore_type: str  # hills / repulsion
    collective_variable: Optional[str] = None
    collective_variables: Optional[List[CVConfig]] = None  # for multi-CV
    # CV-specific config (embedded)
    cv_config: Optional[CVConfig] = None
    # Optional name for output file naming
    name: Optional[str] = None
    # Common parameters
    groups: Optional[List[str]] = None
    warmup: float = 0.0
    cutoff: float = 0.75
    # Bias parameters
    strength: float = 256.0
    sigma: float = 5.0
    # Well-tempered hills (metadynamics-like)
    well_tempered: bool = False
    bias_factor: float = 10.0
    hill_interval: int = 5
    kT: float = 2.5
    # Hills specific
    hill_height: float = 0.5
    max_hills: int = 1000
    guidance_interval: int = 1
    # Optional harmonic target
    target_rmsd: Optional[float] = None
    # Reference structure (for rmsd, native_contacts)
    reference_structure: Optional[str] = None
    contact_cutoff: float = 4.5
    # Region selection - unified format
    region1: Optional[str] = None
    region2: Optional[str] = None
    region3: Optional[str] = None
    region4: Optional[str] = None
    selection: str = "all"
    # SASA CV specific parameters
    probe_radius: float = 1.4  # Solvent probe radius in Å (default 1.4 for water)
    sasa_method: str = "lcpo"  # "lcpo" (analytical) or "coordination" (fast)

    # Optional CV-based gradient scaling (list of scaling CVs, weights multiplied)
    scaling: Optional[List["ScalingConfig"]] = None

    # Optional CV-based gradient projection (list of projection CVs)
    projection: Optional[List["ProjectionConfig"]] = None

    # Order of applying scaling and projection when both are specified
    modifier_order: str = "scale_first"

    # Per-potential bias tempering: max per-atom displacement from this potential (Å)
    # None means no tempering for this potential
    bias_tempering: Optional[float] = None


@dataclass
class SteeringConfig:
    """Configuration for target-based steering.

    Target can be specified as:
        - A numeric value: target: 25.0
        - Extracted from SAXS P(r): target_from_saxs: data.out

    When using target_from_saxs with collective_variable: rg, the Rg is
    computed from the experimental P(r) curve. Use auto_rg_scale to adjust
    the target (e.g., auto_rg_scale: 0.9 means target = 0.9 * Rg_experimental).

    Loss modes:
        - ensemble: false (default) - Per-sample loss, each sample steered independently
        - ensemble: true - Per-ensemble loss, CV averaged across samples before loss
    """
    collective_variable: str
    target: Optional[float] = None  # Explicit target value
    target_from_saxs: Optional[str] = None  # Path to SAXS P(r) file to extract target
    auto_rg_scale: float = 1.0  # Scale factor for Rg extracted from SAXS (target = auto_rg_scale * Rg_exp)
    strength: float = 1.0  # Force constant
    guidance_interval: int = 1
    warmup: float = 0.0  # Start steering after this fraction of steps
    cutoff: float = 0.75  # Stop steering after this fraction of steps
    gaussian_noise_scale: float = 0.0

    # Loss computation mode
    ensemble: bool = False  # True = per-ensemble loss (CV averaged first), False = per-sample loss

    # CV-specific config
    groups: Optional[List[str]] = None
    reference_structure: Optional[str] = None
    contact_cutoff: float = 4.5
    # Region selection - unified format
    region1: Optional[str] = None
    region2: Optional[str] = None
    region3: Optional[str] = None
    region4: Optional[str] = None
    # SASA CV specific parameters
    probe_radius: float = 1.4  # Solvent probe radius in Å (default 1.4 for water)
    sasa_method: str = "lcpo"  # "lcpo" (analytical) or "coordination" (fast)

    # Optional CV-based gradient scaling (list of scaling CVs, weights multiplied)
    scaling: Optional[List["ScalingConfig"]] = None

    # Optional CV-based gradient projection (list of projection CVs)
    projection: Optional[List["ProjectionConfig"]] = None

    # Order of applying scaling and projection when both are specified
    modifier_order: str = "scale_first"

    # Per-potential bias tempering: max per-atom displacement from this potential (Å)
    # None means no tempering for this potential
    bias_tempering: Optional[float] = None


@dataclass
class ChemicalShiftConfig:
    """Configuration for NMR chemical shift steering using CheShift algorithm.

    IMPORTANT: Only CA and CB chemical shifts are supported.

    The CheShift algorithm predicts CA/CB shifts from backbone and sidechain
    torsion angles (phi, psi, chi1, chi2) using pre-computed quantum chemistry
    lookup tables.

    Shift files are in PLUMED format: lines of "residue_num shift_value"
    with 0.0 indicating missing/unknown values.

    DSS Reference Offset:
        The CheShift database returns DFT-reference chemical shifts. Experimental
        NMR data uses DSS (4,4-dimethyl-4-silapentane-1-sulfonic acid) as reference.
        The ca_dss_offset and cb_dss_offset parameters convert DFT to DSS reference:
            predicted_DSS = predicted_DFT + dss_offset

        Default values (2.0 for CA, 2.5 for CB) were calibrated on ubiquitin (1UBQ).
        For best accuracy, calibrate these offsets on your specific system by
        minimizing the mean error between predicted and experimental shifts.

    Example YAML:
        metadiffusion:
          - chemical_shift:
              ca_shifts: CAshifts.dat
              cb_shifts: CBshifts.dat
              strength: 1.0
              loss_type: chi
              ca_dss_offset: 2.0  # DSS reference offset for CA (ppm)
              cb_dss_offset: 2.5  # DSS reference offset for CB (ppm)
    """
    # Shift files (PLUMED format: residue_num shift_value per line)
    # ONLY CA and CB are supported
    ca_shifts: Optional[str] = None   # CA carbon shifts
    cb_shifts: Optional[str] = None   # CB carbon shifts

    # Reference offset handling:
    # - auto_offset=True (recommended): Automatically estimate offset from data
    #   offset = mean(experimental) - mean(predicted), applied before loss computation
    # - auto_offset=False: Use fixed ca_dss_offset/cb_dss_offset values
    auto_offset: bool = True  # Automatically estimate offset from data (recommended)
    ca_dss_offset: float = 0.0  # CA: fixed DFT->DSS offset (ppm), used if auto_offset=False
    cb_dss_offset: float = 0.0  # CB: fixed DFT->DSS offset (ppm), used if auto_offset=False

    # Steering parameters
    strength: float = 1.0  # Force constant
    loss_type: str = "chi"  # "chi" (chi-squared), "mse", or "corr" (correlation)
    guidance_interval: int = 1  # Apply gradient every N steps
    warmup: float = 0.0  # Start steering after this fraction of steps
    cutoff: float = 0.9  # Stop steering after this fraction of steps

    # Optional CV-based gradient scaling (list of scaling CVs, weights multiplied)
    scaling: Optional[List["ScalingConfig"]] = None

    # Optional CV-based gradient projection (list of projection CVs)
    projection: Optional[List["ProjectionConfig"]] = None

    # Order of applying scaling and projection when both are specified
    modifier_order: str = "scale_first"

    # Per-potential bias tempering: max per-atom displacement from this potential (Å)
    bias_tempering: Optional[float] = None


@dataclass
class MetadiffusionConfig:
    """Complete metadiffusion configuration."""
    saxs: List[SAXSConfig] = field(default_factory=list)  # Multiple SAXS loss components
    opt: List[OptConfig] = field(default_factory=list)  # CV optimization
    explore: List[ExploreConfig] = field(default_factory=list)  # Hills/repulsion exploration
    steering: List[SteeringConfig] = field(default_factory=list)
    chemical_shift: List[ChemicalShiftConfig] = field(default_factory=list)  # NMR chemical shift steering
    # Denoising tempering: max per-atom displacement per denoising step (Å)
    # Preserves relative magnitudes by uniform scaling across all atoms
    # None = disabled (default), set to e.g. 0.5 to prevent phase transition/explosion
    denoise_tempering: Optional[float] = None
    # Total bias tempering: max per-atom displacement from ALL potentials combined (Å)
    # Applied after individual per-potential bias_tempering limits
    # None = disabled (default), set to e.g. 1.0 to limit total guidance displacement
    total_bias_clip: Optional[float] = None
    # NOTE: bias_tempering is now per-potential (in SAXSConfig, OptConfig, SteeringConfig, ExploreConfig)

    # Noise scale for stochastic sampling (controls eps noise injection)
    # None = use Boltz default (1.003 for Boltz2, 0.901 for Boltz1)
    # Set to 0.0 for deterministic sampling
    # This is a top-level YAML option (not under metadiffusion)
    noise_scale: Optional[float] = None

    # Guidance mode: controls when guidance is applied during diffusion
    # - "combine" (default): compute both pre and post gradients, add displacement vectors, apply to x_0
    # - "post": apply guidance to x_0 prediction AFTER denoising only
    # - "pre": apply guidance to noisy coords BEFORE denoising only
    guidance_mode: str = "combine"


# Backward compatibility alias
BiasConfig = ExploreConfig


# Valid CV names (implemented)
VALID_CVS = {
    # Structural
    "rg",               # Radius of gyration
    "distance",         # Distance between two atoms/regions (auto-COM for multi-atom)
    "min_distance",     # Minimum pairwise distance between two groups
    "max_diameter",     # Maximum pairwise distance
    "asphericity",      # Shape anisotropy (sum of squared eigenvalue differences)
    # Angle/Dihedral
    "angle",            # Angle between three atoms/regions (auto-COM for multi-atom)
    "dihedral",         # Dihedral between four atoms/regions (auto-COM for multi-atom)
    # RMSD / Fluctuation
    "rmsd",             # RMSD to reference structure
    "pair_rmsd",        # Pairwise RMSD between samples (for diversity)
    #"pair_rmsd_norm_rg", # Pairwise RMSD normalized by Rg (size-invariant diversity)
    "pair_rmsd_grouped", # Pairwise RMSD with separate align/RMSD groups
    "sasa",             # Solvent accessible surface area
    # Content
    "helix_content",    # Helix content fraction
    "sheet_content",    # Sheet content fraction

    # Experimental
    #"dipole_moment",    # Dipole moment magnitude
    #"rmsf",             # Root mean square fluctuation (per-atom flexibility)
    # Contacts
    #"native_contacts",  # Fraction of native contacts (Q)
    #"coordination",     # Contact count
    #"hbond_count",      # Hydrogen bond count
    #"salt_bridges",     # Salt bridge count
    #"contact_order",    # Relative contact order
    #"local_contacts",   # Local contact count
    # "inter_chain",    # DEPRECATED - use distance with region1="A" region2="B"
    # "inter_domain",   # DEPRECATED - use distance with region1="A:1-50" region2="A:100-150"
    # "hinge_angle",    # DEPRECATED - use angle with region1/region2/region3
    #"alpharmsd",        # Alpha helix similarity
    #"antibetarmsd",     # Antiparallel beta similarity
    #"parabetarmsd",     # Parallel beta similarity
    #"acylindricity",    # Gyration tensor acylindricity
    #"shape_gyration",   # Shape parameter from gyration tensor
    #"angle_enhanced",   # Angle with gradient propagation to bonded neighbors
    #"dihedral_enhanced", # Dihedral with gradient propagation to bonded neighbors
    # Note: distance_region, angle_region, dihedral_region are DEPRECATED
    # - use distance/angle/dihedral with region1/region2/etc. instead (auto-converts)
}

# Valid loss types for SAXS P(r) steering
# Each loss type is applied independently; compose by using multiple saxs: entries
VALID_SAXS_LOSS_TYPES = {
    #"w1",     # Wasserstein-1 (Earth Mover's Distance) - fast CDF matching
    #"w2",     # Wasserstein-2 (Sinkhorn optimal transport) - geometric matching
    "cramer", # Cramér distance (squared W1) - error-proportional gradients
    "mse",    # Mean Squared Error - point-by-point matching
    #"chi2",   # Chi-squared - error-weighted matching
    #"mae",    # Mean Absolute Error - robust to outliers
    #"kl",     # KL divergence - information-theoretic
    "rg",     # Rg penalty - per-ensemble Rg matching from P(r)
}

# Valid explore types for metadiffusion
VALID_EXPLORE_TYPES = {
    "hills",      # Gaussian hills deposition (metadynamics-like)
    "repulsion",  # Gaussian repulsion between samples (decays with distance)
    "variance",   # Maximize variance of CV values across samples
}

# Backward compatibility alias
VALID_BIAS_TYPES = VALID_EXPLORE_TYPES

def parse_single_scaling_config(config: Dict[str, Any]) -> ScalingConfig:
    """Parse a single gradient scaling configuration from YAML dict."""
    cv = config.get("collective_variable")
    if cv is None:
        raise ValueError("Scaling config requires 'collective_variable' field specifying which CV to use for weighting")

    if cv not in VALID_CVS:
        raise ValueError(f"Invalid scaling collective_variable: {cv}. Valid CVs: {VALID_CVS}")

    # Get strength (support both 'strength' and legacy 'scaling_strength')
    if "scaling_strength" in config and "strength" not in config:
        warnings.warn(
            "The 'scaling_strength' parameter is deprecated. Use 'strength' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    strength = config.get("strength", config.get("scaling_strength", 1.0))
    if isinstance(strength, str):
        strength = float(strength)

    # Validate reference_structure file exists if specified
    reference_structure = config.get("reference_structure")
    if reference_structure is not None:
        reference_structure_path = Path(reference_structure)
        if not reference_structure_path.exists():
            raise FileNotFoundError(f"reference_structure file not found: {reference_structure}")

    # Parse warmup and cutoff
    warmup = _to_float(config.get("warmup"), 0.0)
    cutoff = _to_float(config.get("cutoff"), 1.0)

    return ScalingConfig(
        collective_variable=cv,
        strength=strength,
        warmup=warmup,
        cutoff=cutoff,
        groups=config.get("groups"),
        # Legacy atom selection
        atom1=config.get("atom1"),
        atom2=config.get("atom2"),
        atom3=config.get("atom3"),
        atom4=config.get("atom4"),
        # New region selection (preferred)
        region1=config.get("region1"),
        region2=config.get("region2"),
        region3=config.get("region3"),
        region4=config.get("region4"),
        reference_structure=reference_structure,
    )


def parse_scaling_config(config: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[ScalingConfig]:
    """Parse gradient scaling configuration from YAML.

    Accepts either a single dict or a list of dicts for backwards compatibility.

    Example YAML (single):
        scaling:
          - collective_variable: rg
            strength: 1.0

    Example YAML (multiple - weights multiplied):
        scaling:
          - collective_variable: rg
            strength: 1.0
          - collective_variable: asphericity
            strength: 0.5

    Returns:
        List of ScalingConfig objects
    """
    if isinstance(config, list):
        return [parse_single_scaling_config(c) for c in config]
    else:
        # Single dict - wrap in list for backwards compatibility
        return [parse_single_scaling_config(config)]


def parse_single_projection_config(config: Dict[str, Any]) -> ProjectionConfig:
    """Parse a single gradient projection configuration from YAML dict."""
    cv = config.get("collective_variable")
    if cv is None:
        raise ValueError("Projection config requires 'collective_variable' field specifying which CV to use for projection direction")

    if cv not in VALID_CVS:
        raise ValueError(f"Invalid projection collective_variable: {cv}. Valid CVs: {VALID_CVS}")

    # Get strength (support both 'strength' and legacy 'projection_strength')
    if "projection_strength" in config and "strength" not in config:
        warnings.warn(
            "The 'projection_strength' parameter is deprecated. Use 'strength' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    strength = config.get("strength", config.get("projection_strength", 1.0))
    if isinstance(strength, str):
        strength = float(strength)

    # Validate strength is in [0, 1]
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f"projection strength must be between 0.0 and 1.0, got: {strength}")

    # Ensure zero_threshold is a float
    zero_threshold = config.get("zero_threshold", 1e-8)
    if isinstance(zero_threshold, str):
        zero_threshold = float(zero_threshold)

    # Parse and validate direction
    direction = config.get("direction", "preserve")
    valid_directions = {"preserve", "toward", "away"}
    if direction not in valid_directions:
        raise ValueError(f"Invalid projection direction: {direction}. Valid: {valid_directions}")

    # Validate reference_structure file exists if specified
    reference_structure = config.get("reference_structure")
    if reference_structure is not None:
        reference_structure_path = Path(reference_structure)
        if not reference_structure_path.exists():
            raise FileNotFoundError(f"reference_structure file not found: {reference_structure}")

    # Parse warmup and cutoff
    warmup = _to_float(config.get("warmup"), 0.0)
    cutoff = _to_float(config.get("cutoff"), 1.0)

    return ProjectionConfig(
        collective_variable=cv,
        strength=strength,
        direction=direction,
        zero_threshold=zero_threshold,
        warmup=warmup,
        cutoff=cutoff,
        groups=config.get("groups"),
        # Legacy atom selection
        atom1=config.get("atom1"),
        atom2=config.get("atom2"),
        atom3=config.get("atom3"),
        atom4=config.get("atom4"),
        # New region selection (preferred)
        region1=config.get("region1"),
        region2=config.get("region2"),
        region3=config.get("region3"),
        region4=config.get("region4"),
        reference_structure=reference_structure,
    )


def parse_projection_config(config: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[ProjectionConfig]:
    """Parse gradient projection configuration from YAML.

    Accepts either a single dict or a list of dicts for backwards compatibility.

    Example YAML (single):
        projection:
          - collective_variable: rg
            strength: 1.0

    Example YAML (multiple - applied sequentially):
        projection:
          - collective_variable: rg
            strength: 1.0
          - collective_variable: distance
            strength: 0.5

    Returns:
        List of ProjectionConfig objects
    """
    if isinstance(config, list):
        return [parse_single_projection_config(c) for c in config]
    else:
        # Single dict - wrap in list for backwards compatibility
        return [parse_single_projection_config(config)]


# Valid modifier orders
VALID_MODIFIER_ORDERS = {"scale_first", "project_first"}


def parse_saxs_config(config: Dict[str, Any]) -> SAXSConfig:
    """Parse SAXS configuration from YAML dict."""
    if "guidance_weight" in config:
        raise ValueError(
            "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
        )

    if "pr_file" not in config:
        raise ValueError("SAXS config requires 'pr_file' field")

    pr_file = config["pr_file"]
    pr_path = Path(pr_file)
    if not pr_path.exists():
        raise FileNotFoundError(f"SAXS pr_file not found: {pr_file}")

    loss_type = config.get("loss_type", "mse").lower()
    if loss_type not in VALID_SAXS_LOSS_TYPES:
        raise ValueError(f"Invalid SAXS loss_type: {loss_type}. Valid types: {VALID_SAXS_LOSS_TYPES}")

    # Parse optional scaling config
    scaling = None
    if "scaling" in config:
        scaling = parse_scaling_config(config["scaling"])

    # Parse optional projection config
    projection = None
    if "projection" in config:
        projection = parse_projection_config(config["projection"])

    # Parse modifier_order
    modifier_order = config.get("modifier_order", "scale_first")
    if modifier_order not in VALID_MODIFIER_ORDERS:
        raise ValueError(f"Invalid modifier_order: {modifier_order}. Valid options: {VALID_MODIFIER_ORDERS}")

    # Parse units for r_grid (nm, angstrom, or auto-detect)
    units = config.get("units", "auto").lower()
    valid_units = {"nm", "angstrom", "auto"}
    if units not in valid_units:
        raise ValueError(f"Invalid units: {units}. Valid options: {valid_units}")

    # Parse per-potential bias_tempering
    bias_tempering = None
    if "bias_clip" in config:
        val = config["bias_clip"]
        bias_tempering = float(val) if val is not None else None

    # Parse bins for uniform resampling
    bins = None
    if "bins" in config:
        bins = _to_int(config["bins"], None)
        if bins is not None and bins < 2:
            raise ValueError(f"bins must be >= 2, got: {bins}")

    # Parse bins_range for resampling range
    bins_range = None
    if "bins_range" in config:
        br = config["bins_range"]
        if isinstance(br, (list, tuple)) and len(br) == 2:
            bins_range = (float(br[0]), float(br[1]))
            if bins_range[0] >= bins_range[1]:
                raise ValueError(f"bins_range[0] must be < bins_range[1], got: {bins_range}")
        else:
            raise ValueError(f"bins_range must be [r_min, r_max], got: {br}")

    # Parse sigma_bin with validation
    sigma_bin = _to_float(config.get("sigma_bin"), 0.5)
    if sigma_bin <= 0:
        raise ValueError(f"sigma_bin must be positive, got {sigma_bin}")

    return SAXSConfig(
        pr_file=pr_file,
        loss_type=loss_type,
        strength=_to_float(config.get("strength"), 1.0),
        guidance_interval=_to_int(config.get("guidance_interval"), 1),
        warmup=_to_float(config.get("warmup"), 0.0),
        cutoff=_to_float(config.get("cutoff"), 0.9),
        sigma_bin=sigma_bin,
        units=units,
        w2_epsilon=_to_float(config.get("w2_epsilon"), 0.1),
        w2_num_iter=_to_int(config.get("w2_num_iter"), 100),
        rg_scale=_to_float(config.get("rg_scale"), 1.0),
        use_rep_atoms=config.get("use_rep_atoms", False),
        bins=bins,
        bins_range=bins_range,
        scaling=scaling,
        projection=projection,
        modifier_order=modifier_order,
        bias_tempering=bias_tempering,
    )


def parse_opt_config(config: Dict[str, Any]) -> OptConfig:
    """Parse optimization configuration from YAML dict.

    Args:
        config: YAML dict with optimization parameters

    Returns:
        OptConfig object

    Raises:
        ValueError: If required fields missing or invalid values
        FileNotFoundError: If reference_structure doesn't exist
    """
    if "guidance_weight" in config:
        raise ValueError(
            "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
        )

    # Warn if both region and atom specified
    _warn_region_atom_conflict(config, "opt config")

    # Validate collective_variable
    cv = config.get("collective_variable")
    if cv is None:
        raise ValueError("Opt config requires 'collective_variable' field")

    if cv not in VALID_CVS:
        raise ValueError(f"Invalid collective_variable: {cv}. Valid CVs: {VALID_CVS}")

    # Validate reference_structure file exists if specified
    reference_structure = config.get("reference_structure")
    if reference_structure is not None:
        reference_structure_path = Path(reference_structure)
        if not reference_structure_path.exists():
            raise FileNotFoundError(f"reference_structure file not found: {reference_structure}")

    # Parse optional scaling config
    scaling = None
    if "scaling" in config:
        scaling = parse_scaling_config(config["scaling"])

    # Parse optional projection config
    projection = None
    if "projection" in config:
        projection = parse_projection_config(config["projection"])

    # Parse modifier_order
    modifier_order = config.get("modifier_order", "scale_first")
    if modifier_order not in VALID_MODIFIER_ORDERS:
        raise ValueError(f"Invalid modifier_order: {modifier_order}. Valid options: {VALID_MODIFIER_ORDERS}")

    # Parse per-potential bias_tempering
    bias_tempering = None
    if "bias_clip" in config:
        val = config["bias_clip"]
        bias_tempering = float(val) if val is not None else None

    return OptConfig(
        collective_variable=cv,
        strength=_to_float(config.get("strength"), 1.0),
        guidance_interval=_to_int(config.get("guidance_interval"), 1),
        warmup=_to_float(config.get("warmup"), 0.0),
        cutoff=_to_float(config.get("cutoff"), 0.75),
        log_gradient=config.get("log_gradient", False),
        groups=config.get("groups"),
        rmsd_groups=config.get("rmsd_groups"),
        # Region selection - prefer region1-4, fall back to atom1-4 for backward compatibility
        region1=config.get("region1") or config.get("atom1"),
        region2=config.get("region2") or config.get("atom2"),
        region3=config.get("region3") or config.get("atom3"),
        region4=config.get("region4") or config.get("atom4"),
        reference_structure=reference_structure,
        contact_cutoff=_to_float(config.get("contact_cutoff"), 4.5),
        selection=config.get("selection", "all"),
        # SASA CV specific
        probe_radius=_to_float(config.get("probe_radius"), 1.4),
        sasa_method=config.get("sasa_method", "lcpo"),
        scaling=scaling,
        projection=projection,
        modifier_order=modifier_order,
        bias_tempering=bias_tempering,
    )


def parse_cv_config(cv_dict: Dict[str, Any]) -> CVConfig:
    """Parse a CV configuration dict (for multi-CV)."""
    cv_type = cv_dict.get("cv")
    if cv_type is None:
        raise ValueError("CV config requires 'cv' field")

    if cv_type not in VALID_CVS:
        raise ValueError(f"Invalid CV type: {cv_type}. Valid CVs: {VALID_CVS}")

    # Parse optional sigma
    sigma = cv_dict.get("sigma")
    if sigma is not None:
        sigma = _to_float(sigma)

    return CVConfig(
        cv_type=cv_type,
        groups=cv_dict.get("groups"),
        atom1=cv_dict.get("atom1"),
        atom2=cv_dict.get("atom2"),
        reference_structure=cv_dict.get("reference_structure"),
        contact_cutoff=_to_float(cv_dict.get("contact_cutoff"), 4.5),
        selection=cv_dict.get("selection", "all"),
        sigma=sigma,
    )


def parse_explore_config(config: Dict[str, Any]) -> ExploreConfig:
    """Parse explore configuration from YAML dict."""
    if "guidance_weight" in config:
        raise ValueError(
            "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
        )

    # Warn if both region and atom specified
    _warn_region_atom_conflict(config, "explore config")

    # Support both "type" and "explore_type" (explore_type for backward compatibility)
    explore_type = config.get("type", config.get("explore_type", "hills")).lower()
    # Normalize "metadynamics" to "hills" (backward compatibility)
    if explore_type == "metadynamics":
        explore_type = "hills"
    if explore_type not in VALID_EXPLORE_TYPES:
        raise ValueError(f"Invalid explore type: {explore_type}. Valid types: {VALID_EXPLORE_TYPES}")

    # Handle single CV or multi-CV
    collective_variable = config.get("collective_variable")
    collective_variables = None

    if "collective_variables" in config:
        # Multi-CV hills bias
        cv_list = config["collective_variables"]
        collective_variables = [parse_cv_config(cv) for cv in cv_list]
    elif collective_variable:
        if collective_variable not in VALID_CVS:
            raise ValueError(f"Invalid CV: {collective_variable}. Valid CVs: {VALID_CVS}")
    else:
        raise ValueError("Bias config requires 'collective_variable' or 'collective_variables'")

    # Validate hills-only parameters
    hills_only_params = ["well_tempered", "bias_factor", "hill_height", "max_hills", "hill_interval", "kT"]
    if explore_type in ("repulsion", "variance"):
        for param in hills_only_params:
            if param in config:
                raise ValueError(
                    f"Parameter '{param}' is only valid for 'hills' explore type, not '{explore_type}'. "
                    f"Remove '{param}' from your {explore_type} explore configuration."
                )

    # Validate reference_structure file exists if specified
    reference_structure = config.get("reference_structure")
    if reference_structure is not None:
        reference_structure_path = Path(reference_structure)
        if not reference_structure_path.exists():
            raise FileNotFoundError(f"reference_structure file not found: {reference_structure}")

    # Parse optional scaling config
    scaling = None
    if "scaling" in config:
        scaling = parse_scaling_config(config["scaling"])

    # Parse optional projection config
    projection = None
    if "projection" in config:
        projection = parse_projection_config(config["projection"])

    # Parse modifier_order
    modifier_order = config.get("modifier_order", "scale_first")
    if modifier_order not in VALID_MODIFIER_ORDERS:
        raise ValueError(f"Invalid modifier_order: {modifier_order}. Valid options: {VALID_MODIFIER_ORDERS}")

    # Parse target_rmsd (optional float)
    target_rmsd = config.get("target_rmsd")
    if target_rmsd is not None:
        target_rmsd = _to_float(target_rmsd)

    # Parse per-potential bias_tempering
    bias_tempering = None
    if "bias_clip" in config:
        val = config["bias_clip"]
        bias_tempering = float(val) if val is not None else None

    return ExploreConfig(
        explore_type=explore_type,
        collective_variable=collective_variable,
        collective_variables=collective_variables,
        name=config.get("name"),
        groups=config.get("groups"),
        warmup=_to_float(config.get("warmup"), 0.2),
        cutoff=_to_float(config.get("cutoff"), 0.75),
        strength=_to_float(config.get("strength"), 256.0),
        sigma=_to_float(config.get("sigma"), 5.0),
        well_tempered=config.get("well_tempered", False),
        bias_factor=_to_float(config.get("bias_factor"), 10.0),
        hill_interval=_to_int(config.get("hill_interval"), 5),
        kT=_to_float(config.get("kT"), 2.5),
        hill_height=_to_float(config.get("hill_height"), 0.5),
        max_hills=_to_int(config.get("max_hills"), 1000),
        guidance_interval=_to_int(config.get("guidance_interval"), 1),
        target_rmsd=target_rmsd,
        reference_structure=config.get("reference_structure"),
        contact_cutoff=_to_float(config.get("contact_cutoff"), 4.5),
        # Region selection - prefer region1-4, fall back to atom1-4 for backward compatibility
        region1=config.get("region1") or config.get("atom1"),
        region2=config.get("region2") or config.get("atom2"),
        region3=config.get("region3") or config.get("atom3"),
        region4=config.get("region4") or config.get("atom4"),
        selection=config.get("selection", "all"),
        # SASA CV specific
        probe_radius=_to_float(config.get("probe_radius"), 1.4),
        sasa_method=config.get("sasa_method", "lcpo"),
        scaling=scaling,
        projection=projection,
        modifier_order=modifier_order,
        bias_tempering=bias_tempering,
    )


# Backward compatibility alias
parse_bias_config = parse_explore_config


def parse_steering_config(config: Dict[str, Any]) -> SteeringConfig:
    """Parse steering configuration from YAML dict."""
    if "guidance_weight" in config:
        raise ValueError(
            "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
        )

    # Warn if both region and atom specified
    _warn_region_atom_conflict(config, "steering config")

    cv = config.get("collective_variable")
    if cv is None:
        raise ValueError("Steering config requires 'collective_variable' field")

    if cv not in VALID_CVS:
        raise ValueError(f"Invalid CV: {cv}. Valid CVs: {VALID_CVS}")

    target = config.get("target")
    target_from_saxs = config.get("target_from_saxs")

    # Must have either target or target_from_saxs
    if target is None and target_from_saxs is None:
        raise ValueError("Steering config requires either 'target' or 'target_from_saxs' field")

    # target_from_saxs only valid for rg CV
    if target_from_saxs is not None and cv != "rg":
        raise ValueError(f"target_from_saxs is only valid for collective_variable: rg, got: {cv}")

    # Validate target_from_saxs file exists
    if target_from_saxs is not None:
        target_from_saxs_path = Path(target_from_saxs)
        if not target_from_saxs_path.exists():
            raise FileNotFoundError(f"target_from_saxs file not found: {target_from_saxs}")

    # Validate reference_structure file exists if specified
    reference_structure = config.get("reference_structure")
    if reference_structure is not None:
        reference_structure_path = Path(reference_structure)
        if not reference_structure_path.exists():
            raise FileNotFoundError(f"reference_structure file not found: {reference_structure}")

    # Parse optional scaling config
    scaling = None
    if "scaling" in config:
        scaling = parse_scaling_config(config["scaling"])

    # Parse optional projection config
    projection = None
    if "projection" in config:
        projection = parse_projection_config(config["projection"])

    # Parse modifier_order
    modifier_order = config.get("modifier_order", "scale_first")
    if modifier_order not in VALID_MODIFIER_ORDERS:
        raise ValueError(f"Invalid modifier_order: {modifier_order}. Valid options: {VALID_MODIFIER_ORDERS}")

    # Parse per-potential bias_tempering
    bias_tempering = None
    if "bias_clip" in config:
        val = config["bias_clip"]
        bias_tempering = float(val) if val is not None else None

    return SteeringConfig(
        collective_variable=cv,
        target=_to_float(target) if target is not None else None,
        target_from_saxs=target_from_saxs,
        auto_rg_scale=_to_float(config.get("auto_rg_scale"), 1.0),
        strength=_to_float(config.get("strength"), 1.0),
        guidance_interval=_to_int(config.get("guidance_interval"), 1),
        warmup=_to_float(config.get("warmup"), 0.0),
        cutoff=_to_float(config.get("cutoff"), 0.75),
        gaussian_noise_scale=_to_float(config.get("gaussian_noise_scale"), 0.0),
        ensemble=config.get("ensemble", False),
        groups=config.get("groups"),
        reference_structure=config.get("reference_structure"),
        contact_cutoff=_to_float(config.get("contact_cutoff"), 4.5),
        # Region selection - prefer region1-4, fall back to atom1-4 for backward compatibility
        region1=config.get("region1") or config.get("atom1"),
        region2=config.get("region2") or config.get("atom2"),
        region3=config.get("region3") or config.get("atom3"),
        region4=config.get("region4") or config.get("atom4"),
        # SASA CV specific
        probe_radius=_to_float(config.get("probe_radius"), 1.4),
        sasa_method=config.get("sasa_method", "lcpo"),
        scaling=scaling,
        projection=projection,
        modifier_order=modifier_order,
        bias_tempering=bias_tempering,
    )


# Valid loss types for chemical shift
VALID_CHEMICAL_SHIFT_LOSS_TYPES = {
    "chi",      # Chi-squared with nucleus-specific sigma
    "mse",      # Mean squared error
    "corr",     # Negative correlation (maximize R)
    "ccc",      # Concordance correlation coefficient (maximize R + match offset/scale)
    "chi_ccc",  # Combined chi-squared + CCC (per-residue + global correlation)
}


def parse_chemical_shift_config(config: Dict[str, Any], yaml_dir: Optional[Path] = None) -> ChemicalShiftConfig:
    """Parse chemical shift configuration from YAML dict.

    Args:
        config: YAML dict with chemical shift parameters
        yaml_dir: Directory containing the YAML file (for resolving relative paths)

    Returns:
        ChemicalShiftConfig object

    Raises:
        ValueError: If required fields missing or invalid values
        FileNotFoundError: If shift files don't exist
    """
    if "guidance_weight" in config:
        raise ValueError(
            "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
        )

    # Validate loss_type
    loss_type = config.get("loss_type", "chi").lower()
    if loss_type not in VALID_CHEMICAL_SHIFT_LOSS_TYPES:
        raise ValueError(
            f"Invalid chemical shift loss_type: {loss_type}. "
            f"Valid types: {VALID_CHEMICAL_SHIFT_LOSS_TYPES}"
        )

    # Resolve shift file paths relative to YAML directory
    # Only CA and CB shifts are supported (other nuclei deprecated)
    shift_keys = ['ca_shifts', 'cb_shifts']
    shift_files = {}
    has_any_shifts = False

    for key in shift_keys:
        if key in config and config[key]:
            filepath = config[key]
            # Resolve relative paths
            if yaml_dir and not Path(filepath).is_absolute():
                filepath = str(yaml_dir / filepath)

            shift_path = Path(filepath)
            if not shift_path.exists():
                raise FileNotFoundError(f"Chemical shift file not found: {filepath}")
            shift_files[key] = filepath
            has_any_shifts = True
        else:
            shift_files[key] = None

    # Must have at least one shift file
    if not has_any_shifts:
        raise ValueError(
            "Chemical shift config requires at least one shift file "
            "(ca_shifts and/or cb_shifts)"
        )

    # Parse optional scaling config
    scaling = None
    if "scaling" in config:
        scaling = parse_scaling_config(config["scaling"])

    # Parse optional projection config
    projection = None
    if "projection" in config:
        projection = parse_projection_config(config["projection"])

    # Parse modifier_order
    modifier_order = config.get("modifier_order", "scale_first")
    if modifier_order not in VALID_MODIFIER_ORDERS:
        raise ValueError(f"Invalid modifier_order: {modifier_order}. Valid options: {VALID_MODIFIER_ORDERS}")

    # Parse per-potential bias_tempering
    bias_tempering = None
    if "bias_clip" in config:
        val = config["bias_clip"]
        bias_tempering = float(val) if val is not None else None

    return ChemicalShiftConfig(
        ca_shifts=shift_files['ca_shifts'],
        cb_shifts=shift_files['cb_shifts'],
        auto_offset=config.get("auto_offset", True),
        ca_dss_offset=_to_float(config.get("ca_dss_offset"), 0.0),
        cb_dss_offset=_to_float(config.get("cb_dss_offset"), 0.0),
        strength=_to_float(config.get("strength"), 1.0),
        loss_type=loss_type,
        guidance_interval=_to_int(config.get("guidance_interval"), 1),
        warmup=_to_float(config.get("warmup"), 0.0),
        cutoff=_to_float(config.get("cutoff"), 0.9),
        scaling=scaling,
        projection=projection,
        modifier_order=modifier_order,
        bias_tempering=bias_tempering,
    )


def parse_metadiffusion_section(schema: Dict[str, Any]) -> Optional[MetadiffusionConfig]:
    """
    Parse the metadiffusion section from a Boltz YAML schema.

    Also parses top-level noise_scale and denoise_tempering if present.

    Args:
        schema: The full YAML schema dict

    Returns:
        MetadiffusionConfig if metadiffusion section or top-level params exist, None otherwise
    """
    # Parse top-level noise_scale (standalone Boltz option)
    noise_scale = None
    if "noise_scale" in schema:
        value = schema["noise_scale"]
        noise_scale = float(value) if value is not None else None

    # Parse top-level denoise_tempering (standalone Boltz option)
    denoise_tempering = None
    if "denoise_clip" in schema:
        value = schema["denoise_clip"]
        denoise_tempering = float(value) if value is not None else None

    if "metadiffusion" not in schema:
        # Return config with just top-level params if present
        if noise_scale is not None or denoise_tempering is not None:
            config = MetadiffusionConfig()
            config.noise_scale = noise_scale
            config.denoise_tempering = denoise_tempering
            return config
        return None

    metadiffusion_list = schema["metadiffusion"]
    if not isinstance(metadiffusion_list, list):
        # Detect common mistake: mixing mapping keys with list items
        # This happens when user writes:
        #   metadiffusion:
        #     total_bias_clip: 1.0   <-- no dash (mapping key)
        #     - opt:                       <-- has dash (list item)
        if isinstance(metadiffusion_list, dict):
            keys = list(metadiffusion_list.keys())
            # Check for top-level params that should be list items
            top_level_params = {
                "total_bias_clip", "guidance_mode",
                "noise_scale", "denoise_clip", "guidance_before_denoising"
            }
            # Note: noise_scale and denoise_clip should be at YAML top-level, not here
            found_top_level = [k for k in keys if k in top_level_params]
            if found_top_level:
                raise ValueError(
                    f"Malformed 'metadiffusion' section: found top-level parameters without '-' prefix: {found_top_level}\n\n"
                    f"All entries under 'metadiffusion:' must be list items (start with '-').\n\n"
                    f"INCORRECT:\n"
                    f"  metadiffusion:\n"
                    f"    total_bias_clip: 1.0   # Missing dash!\n"
                    f"    - opt:\n"
                    f"        collective_variable: rg\n\n"
                    f"CORRECT:\n"
                    f"  metadiffusion:\n"
                    f"    - total_bias_clip: 1.0  # Has dash\n"
                    f"    - opt:\n"
                    f"        collective_variable: rg"
                )
        raise ValueError(
            f"'metadiffusion' section must be a list of entries, got {type(metadiffusion_list).__name__}.\n\n"
            f"Each entry must start with '-'. Example:\n"
            f"  metadiffusion:\n"
            f"    - total_bias_clip: 1.0\n"
            f"    - opt:\n"
            f"        collective_variable: rg"
        )

    config = MetadiffusionConfig()
    config.noise_scale = noise_scale  # Apply top-level noise_scale
    config.denoise_tempering = denoise_tempering  # Apply top-level denoise_tempering

    for entry in metadiffusion_list:
        if not isinstance(entry, dict):
            raise ValueError(f"Metadiffusion entry must be a dict, got: {type(entry)}")

        # Determine entry type by its key
        if "saxs" in entry:
            saxs_config = parse_saxs_config(entry["saxs"])
            if saxs_config is not None:
                config.saxs.append(saxs_config)

        elif "opt" in entry:
            config.opt.append(parse_opt_config(entry["opt"]))

        elif "explore" in entry:
            config.explore.append(parse_explore_config(entry["explore"]))

        elif "bias" in entry:
            warnings.warn(
                "The 'bias' entry is deprecated. Use 'explore' instead.",
                DeprecationWarning
            )
            config.explore.append(parse_explore_config(entry["bias"]))

        elif "steering" in entry:
            config.steering.append(parse_steering_config(entry["steering"]))

        elif "steer" in entry:
            # Alias for steering
            config.steering.append(parse_steering_config(entry["steer"]))

        elif "chemical_shift" in entry:
            config.chemical_shift.append(parse_chemical_shift_config(entry["chemical_shift"]))

        elif "denoise_clip" in entry:
            # DEPRECATED: denoise_clip should be at top level like noise_scale
            warnings.warn(
                "'denoise_clip' inside metadiffusion list is deprecated. "
                "Move it to the top level of your YAML file, like 'noise_scale'.\n"
                "Example:\n"
                "  noise_scale: 0.2\n"
                "  denoise_clip: 256\n"
                "  metadiffusion:\n"
                "    - opt:\n"
                "        collective_variable: rg",
                DeprecationWarning,
                stacklevel=2,
            )
            value = entry["denoise_clip"]
            if value is None:
                config.denoise_tempering = None  # Disabled
            else:
                config.denoise_tempering = float(value)

        elif "total_bias_clip" in entry:
            # Total bias tempering: max per-atom displacement from ALL potentials combined (Å)
            value = entry["total_bias_clip"]
            if value is None:
                config.total_bias_clip = None  # Disabled
            else:
                config.total_bias_clip = float(value)

        elif "bias_clip" in entry:
            # DEPRECATED: bias_clip is now per-potential
            warnings.warn(
                "Global 'bias_clip' in metadiffusion is deprecated. "
                "Set bias_clip individually in each opt/steer/explore/saxs config.",
                DeprecationWarning,
                stacklevel=2,
            )

        elif "guidance_mode" in entry:
            # Guidance mode: "pre", "post", or "combine"
            mode = str(entry["guidance_mode"]).lower()
            if mode not in ("pre", "post", "combine"):
                raise ValueError(f"Invalid guidance_mode '{mode}'. Must be 'pre', 'post', or 'combine'.")
            config.guidance_mode = mode

        elif "guidance_before_denoising" in entry:
            # DEPRECATED: backward compatibility - convert bool to mode
            warnings.warn(
                "'guidance_before_denoising' is deprecated. Use 'guidance_mode: pre/post/combine' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            config.guidance_mode = "pre" if bool(entry["guidance_before_denoising"]) else "post"

        else:
            # Check if it's a direct entry (not nested under a key)
            if "pr_file" in entry:
                # Direct SAXS config
                saxs_config = parse_saxs_config(entry)
                if saxs_config is not None:
                    config.saxs.append(saxs_config)
            elif "type" in entry and ("collective_variable" in entry or "collective_variables" in entry):
                # Direct explore config (hills/repulsion)
                config.explore.append(parse_explore_config(entry))
            elif ("target" in entry or "target_from_saxs" in entry) and "collective_variable" in entry:
                # Direct steering config
                config.steering.append(parse_steering_config(entry))
            else:
                # Provide helpful error for unknown entries
                entry_keys = list(entry.keys())
                raise ValueError(
                    f"Unknown metadiffusion entry with keys: {entry_keys}\n\n"
                    f"Valid entry types are:\n"
                    f"  - opt:                  # Optimize a collective variable\n"
                    f"  - steer:                # Steer toward a target value\n"
                    f"  - explore:              # Exploration (hills/repulsion)\n"
                    f"  - saxs:                 # SAXS fitting\n"
                    f"  - chemical_shift:       # NMR chemical shift\n"
                    f"  - total_bias_clip: # Global gradient limit\n"
                    f"  - denoise_clip:         # Denoising step limit\n"
                    f"  - guidance_mode:        # pre/post/combine\n\n"
                    f"Make sure each entry starts with '-' (is a list item)."
                )

    return config


def debug_print_config(config: MetadiffusionConfig, enabled: bool = False) -> None:
    """Print debug information about parsed MetadiffusionConfig.

    Args:
        config: Parsed MetadiffusionConfig object
        enabled: Whether debug output is enabled
    """
    if not enabled:
        return

    print("[DEBUG] Parsed MetadiffusionConfig:")
    print(f"  noise_scale: {config.noise_scale}")
    print(f"  denoise_tempering: {config.denoise_tempering}")
    print(f"  total_bias_clip: {config.total_bias_clip}")
    print(f"  guidance_mode: {config.guidance_mode}")

    print(f"  explore configs: {len(config.explore)}")
    for i, e in enumerate(config.explore):
        print(f"    [{i}] type={e.explore_type}, cv={e.collective_variable}, strength={e.strength}, sigma={e.sigma}")
        if e.region1 or e.region2 or e.region3 or e.region4:
            print(f"        regions: {e.region1}, {e.region2}, {e.region3}, {e.region4}")

    print(f"  steering configs: {len(config.steering)}")
    for i, s in enumerate(config.steering):
        print(f"    [{i}] cv={s.collective_variable}, target={s.target}, strength={s.strength}, ensemble={s.ensemble}")
        if s.region1 or s.region2 or s.region3 or s.region4:
            print(f"        regions: {s.region1}, {s.region2}, {s.region3}, {s.region4}")

    print(f"  opt configs: {len(config.opt)}")
    for i, o in enumerate(config.opt):
        print(f"    [{i}] cv={o.collective_variable}, strength={o.strength}, method={o.method}")
        if o.region1 or o.region2 or o.region3 or o.region4:
            print(f"        regions: {o.region1}, {o.region2}, {o.region3}, {o.region4}")

    print(f"  saxs configs: {len(config.saxs)}")
    for i, sx in enumerate(config.saxs):
        print(f"    [{i}] pr_file={sx.pr_file}, strength={sx.strength}")

    print(f"  chemical_shift configs: {len(config.chemical_shift)}")
    for i, cs in enumerate(config.chemical_shift):
        print(f"    [{i}] ca_shifts={cs.ca_shifts}, cb_shifts={cs.cb_shifts}, strength={cs.strength}")



def single_scaling_config_to_dict(scaling: ScalingConfig) -> Dict[str, Any]:
    """Convert a single ScalingConfig to a dict representation."""
    result = {
        "collective_variable": scaling.collective_variable,
        "strength": scaling.strength,
    }
    if scaling.groups:
        result["groups"] = scaling.groups
    if scaling.atom1:
        result["atom1"] = scaling.atom1
    if scaling.atom2:
        result["atom2"] = scaling.atom2
    if scaling.reference_structure:
        result["reference_structure"] = scaling.reference_structure
    return result


def scaling_config_to_dict(scaling: List[ScalingConfig]) -> List[Dict[str, Any]]:
    """Convert a list of ScalingConfig to a list of dicts."""
    return [single_scaling_config_to_dict(s) for s in scaling]


def single_projection_config_to_dict(projection: ProjectionConfig) -> Dict[str, Any]:
    """Convert a single ProjectionConfig to a dict representation."""
    result = {
        "collective_variable": projection.collective_variable,
        "strength": projection.strength,
    }
    if projection.zero_threshold != 1e-8:
        result["zero_threshold"] = projection.zero_threshold
    if projection.groups:
        result["groups"] = projection.groups
    if projection.atom1:
        result["atom1"] = projection.atom1
    if projection.atom2:
        result["atom2"] = projection.atom2
    if projection.reference_structure:
        result["reference_structure"] = projection.reference_structure
    return result


def projection_config_to_dict(projection: List[ProjectionConfig]) -> List[Dict[str, Any]]:
    """Convert a list of ProjectionConfig to a list of dicts."""
    return [single_projection_config_to_dict(p) for p in projection]


def opt_config_to_dict(opt: OptConfig) -> Dict[str, Any]:
    """Convert an OptConfig to a dict representation."""
    result = {
        "collective_variable": opt.collective_variable,
        "strength": opt.strength,
        "guidance_interval": opt.guidance_interval,
        "warmup": opt.warmup,
        "cutoff": opt.cutoff,
        "log_gradient": opt.log_gradient,
    }
    # CV-specific params
    if opt.groups:
        result["groups"] = opt.groups
    if opt.rmsd_groups:
        result["rmsd_groups"] = opt.rmsd_groups
    # Region selection
    if opt.region1:
        result["region1"] = opt.region1
    if opt.region2:
        result["region2"] = opt.region2
    if opt.region3:
        result["region3"] = opt.region3
    if opt.region4:
        result["region4"] = opt.region4
    if opt.reference_structure:
        result["reference_structure"] = opt.reference_structure
    if opt.contact_cutoff != 4.5:
        result["contact_cutoff"] = opt.contact_cutoff
    if opt.selection != "all":
        result["selection"] = opt.selection
    # Gradient modification
    if opt.scaling:
        result["scaling"] = scaling_config_to_dict(opt.scaling)
    if opt.projection:
        result["projection"] = projection_config_to_dict(opt.projection)
    if opt.modifier_order != "scale_first":
        result["modifier_order"] = opt.modifier_order
    # Per-potential bias tempering
    if opt.bias_tempering is not None:
        result["bias_clip"] = opt.bias_tempering
    return result


def chemical_shift_config_to_dict(cs: ChemicalShiftConfig) -> Dict[str, Any]:
    """Convert a ChemicalShiftConfig to a dict representation."""
    result = {
        "strength": cs.strength,
        "loss_type": cs.loss_type,
        "guidance_interval": cs.guidance_interval,
        "warmup": cs.warmup,
        "cutoff": cs.cutoff,
    }
    # Shift files (only CA and CB supported)
    if cs.ca_shifts:
        result["ca_shifts"] = cs.ca_shifts
    if cs.cb_shifts:
        result["cb_shifts"] = cs.cb_shifts
    # Gradient modification
    if cs.scaling:
        result["scaling"] = scaling_config_to_dict(cs.scaling)
    if cs.projection:
        result["projection"] = projection_config_to_dict(cs.projection)
    if cs.modifier_order != "scale_first":
        result["modifier_order"] = cs.modifier_order
    # Per-potential bias tempering
    if cs.bias_tempering is not None:
        result["bias_clip"] = cs.bias_tempering
    return result


def metadiffusion_config_to_dict(config: MetadiffusionConfig) -> Dict[str, Any]:
    """
    Convert a MetadiffusionConfig back to a dict representation.

    Useful for debugging and serialization.
    """
    result = {}

    if config.saxs:
        saxs_list = []
        for s in config.saxs:
            saxs_dict = {
                "pr_file": s.pr_file,
                "loss_type": s.loss_type,
                "strength": s.strength,
                "guidance_interval": s.guidance_interval,
                "warmup": s.warmup,
                "cutoff": s.cutoff,
                "sigma_bin": s.sigma_bin,
                "units": s.units,
                "bins": s.bins,
                "bins_range": s.bins_range,
                "w2_epsilon": s.w2_epsilon,
                "w2_num_iter": s.w2_num_iter,
                "rg_scale": s.rg_scale,
                "use_rep_atoms": s.use_rep_atoms,
            }
            if s.scaling:
                saxs_dict["scaling"] = scaling_config_to_dict(s.scaling)
            if s.projection:
                saxs_dict["projection"] = projection_config_to_dict(s.projection)
            if s.modifier_order != "scale_first":
                saxs_dict["modifier_order"] = s.modifier_order
            if s.bias_tempering is not None:
                saxs_dict["bias_clip"] = s.bias_tempering
            saxs_list.append(saxs_dict)
        result["saxs"] = saxs_list

    if config.opt:
        result["opt"] = [opt_config_to_dict(o) for o in config.opt]

    explore_list = []
    for e in config.explore:
        # Base dict for all explore types
        explore_dict = {
            "type": e.explore_type,
            "warmup": e.warmup,
            "cutoff": e.cutoff,
            "strength": e.strength,
            "sigma": e.sigma,
            "guidance_interval": e.guidance_interval,
            "reference_structure": e.reference_structure,
            "contact_cutoff": e.contact_cutoff,
            "region1": e.region1,
            "region2": e.region2,
            "region3": e.region3,
            "region4": e.region4,
            "selection": e.selection,
        }
        # Hills-specific params (only for hills explore type)
        if e.explore_type == "hills":
            explore_dict["well_tempered"] = e.well_tempered
            explore_dict["bias_factor"] = e.bias_factor
            explore_dict["hill_interval"] = e.hill_interval
            explore_dict["kT"] = e.kT
            explore_dict["hill_height"] = e.hill_height
            explore_dict["max_hills"] = e.max_hills
            explore_dict["target_rmsd"] = e.target_rmsd
        # Only include name if set
        if e.name:
            explore_dict["name"] = e.name
        # Only include groups if set
        if e.groups:
            explore_dict["groups"] = e.groups
        # Include collective_variable for single-CV explores
        if e.collective_variable:
            explore_dict["collective_variable"] = e.collective_variable
        # Include collective_variables for multi-CV explores (only if not None)
        if e.collective_variables:
            explore_dict["collective_variables"] = [
                {
                    "cv": cv.cv_type,
                    "groups": cv.groups,
                    "atom1": cv.atom1,
                    "atom2": cv.atom2,
                    "reference_structure": cv.reference_structure,
                    "contact_cutoff": cv.contact_cutoff,
                    "selection": cv.selection,
                    "sigma": cv.sigma,
                }
                for cv in e.collective_variables
            ]
        if e.scaling:
            explore_dict["scaling"] = scaling_config_to_dict(e.scaling)
        if e.projection:
            explore_dict["projection"] = projection_config_to_dict(e.projection)
        if e.modifier_order != "scale_first":
            explore_dict["modifier_order"] = e.modifier_order
        if e.bias_tempering is not None:
            explore_dict["bias_clip"] = e.bias_tempering
        explore_list.append(explore_dict)
    result["explore"] = explore_list

    steering_list = []
    for s in config.steering:
        steering_dict = {
            "collective_variable": s.collective_variable,
            "target": s.target,
            "target_from_saxs": s.target_from_saxs,
            "auto_rg_scale": s.auto_rg_scale,
            "strength": s.strength,
            "guidance_interval": s.guidance_interval,
            "warmup": s.warmup,
            "cutoff": s.cutoff,
            "gaussian_noise_scale": s.gaussian_noise_scale,
            "ensemble": s.ensemble,
            "groups": s.groups,
            "reference_structure": s.reference_structure,
            "contact_cutoff": s.contact_cutoff,
            "region1": s.region1,
            "region2": s.region2,
            "region3": s.region3,
            "region4": s.region4,
        }
        if s.scaling:
            steering_dict["scaling"] = scaling_config_to_dict(s.scaling)
        if s.projection:
            steering_dict["projection"] = projection_config_to_dict(s.projection)
        if s.modifier_order != "scale_first":
            steering_dict["modifier_order"] = s.modifier_order
        if s.bias_tempering is not None:
            steering_dict["bias_clip"] = s.bias_tempering
        steering_list.append(steering_dict)
    result["steering"] = steering_list

    # Add chemical_shift configs
    if config.chemical_shift:
        result["chemical_shift"] = [chemical_shift_config_to_dict(cs) for cs in config.chemical_shift]

    # Add denoise_tempering (top-level metadiffusion parameter)
    if config.denoise_tempering is not None:
        result["denoise_clip"] = config.denoise_tempering

    # Add total_bias_clip (top-level limit on ALL potentials combined)
    if config.total_bias_clip is not None:
        result["total_bias_clip"] = config.total_bias_clip
    # NOTE: bias_tempering is now per-potential (included in each config dict above)

    # Add noise_scale (top-level YAML parameter, stored in config for convenience)
    if config.noise_scale is not None:
        result["noise_scale"] = config.noise_scale

    # Add guidance_mode (controls pre/post/combine guidance application)
    if config.guidance_mode != "combine":
        result["guidance_mode"] = config.guidance_mode

    return result