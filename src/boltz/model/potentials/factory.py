"""
Factory for creating potentials from YAML metadiffusion configuration.

This module provides factory functions to instantiate potentials from
MetadiffusionConfig objects parsed from YAML files.
"""

import warnings
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import torch

from boltz.model.potentials.potentials import (
    Potential,
    SAXSPrPotential,
)
from boltz.model.potentials.schedules import (
    ParameterSchedule,
    ExponentialInterpolation,
    PiecewiseStepFunction,
)
from boltz.data.parse.metadiffusion import (
    MetadiffusionConfig,
    SAXSConfig,
    OptConfig,
    ExploreConfig,
    SteeringConfig,
    CVConfig,
    ScalingConfig,
    ProjectionConfig,
    ChemicalShiftConfig,
)
from boltz.model.potentials.gradient_scaler import (
    GradientScaler,
    CompositeGradientScaler,
    GradientProjector,
    CompositeGradientProjector,
    GradientModifier,
)


def load_reference_structure(path: str) -> torch.Tensor:
    """
    Load reference structure coordinates from a PDB/CIF file.

    Args:
        path: Path to the reference structure file

    Returns:
        coords: Tensor of coordinates [N_atoms, 3]
    """
    from pathlib import Path as P
    import os

    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference structure not found: {path}")

    ext = P(path).suffix.lower()

    if ext == '.pdb':
        return _load_pdb_coords(path)
    elif ext == '.cif':
        return _load_cif_coords(path)
    else:
        raise ValueError(f"Unsupported reference structure format: {ext}. Use .pdb or .cif")


def _load_pdb_coords(path: str) -> torch.Tensor:
    """Load coordinates from PDB file."""
    coords = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue

    if not coords:
        raise ValueError(f"No coordinates found in PDB file: {path}")

    return torch.tensor(coords, dtype=torch.float32)


def _load_cif_coords(path: str) -> torch.Tensor:
    """Load coordinates from mmCIF file."""
    try:
        from gemmi import cif
    except ImportError:
        # Fallback: try simple parsing
        return _load_cif_coords_simple(path)

    doc = cif.read(path)
    block = doc.sole_block()

    coords = []
    x_col = block.find_values('_atom_site.Cartn_x')
    y_col = block.find_values('_atom_site.Cartn_y')
    z_col = block.find_values('_atom_site.Cartn_z')

    for x, y, z in zip(x_col, y_col, z_col):
        coords.append([float(x), float(y), float(z)])

    if not coords:
        raise ValueError(f"No coordinates found in CIF file: {path}")

    return torch.tensor(coords, dtype=torch.float32)


def _load_cif_coords_simple(path: str) -> torch.Tensor:
    """Simple CIF coordinate parser without gemmi."""
    coords = []
    in_atom_site = False
    x_idx = y_idx = z_idx = None
    headers = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('_atom_site.'):
                in_atom_site = True
                header = line.split('.')[1].split()[0]
                headers.append(header)
                if header == 'Cartn_x':
                    x_idx = len(headers) - 1
                elif header == 'Cartn_y':
                    y_idx = len(headers) - 1
                elif header == 'Cartn_z':
                    z_idx = len(headers) - 1
            elif in_atom_site and not line.startswith(('_', '#', 'loop_')):
                if line and x_idx is not None:
                    parts = line.split()
                    if len(parts) > max(x_idx, y_idx, z_idx):
                        try:
                            coords.append([
                                float(parts[x_idx]),
                                float(parts[y_idx]),
                                float(parts[z_idx])
                            ])
                        except (ValueError, IndexError):
                            continue

    if not coords:
        raise ValueError(f"No coordinates found in CIF file: {path}")

    return torch.tensor(coords, dtype=torch.float32)


def create_single_gradient_scaler(
    scaling_config: ScalingConfig,
    feats: Optional[Dict[str, Any]] = None,
) -> GradientScaler:
    """
    Create a single GradientScaler from a ScalingConfig.

    Args:
        scaling_config: Single ScalingConfig from YAML
        feats: Feature dictionary for CV creation

    Returns:
        GradientScaler instance
    """
    from boltz.model.potentials.collective_variables import create_cv_function

    cv_type = scaling_config.collective_variable
    cv_name = cv_type

    # Build CV kwargs dict for create_cv_function
    cv_kwargs = {}

    if scaling_config.groups:
        cv_kwargs['groups'] = scaling_config.groups
    # Legacy atom selection
    if scaling_config.atom1:
        cv_kwargs['atom1'] = scaling_config.atom1
    if scaling_config.atom2:
        cv_kwargs['atom2'] = scaling_config.atom2
    if scaling_config.atom3:
        cv_kwargs['atom3'] = scaling_config.atom3
    if scaling_config.atom4:
        cv_kwargs['atom4'] = scaling_config.atom4
    # New region selection (preferred) - supports "A", "A:1-50", "A:5:CA", "A:1-50:CA", "A::CA"
    if scaling_config.region1:
        cv_kwargs['region1'] = scaling_config.region1
    if scaling_config.region2:
        cv_kwargs['region2'] = scaling_config.region2
    if scaling_config.region3:
        cv_kwargs['region3'] = scaling_config.region3
    if scaling_config.region4:
        cv_kwargs['region4'] = scaling_config.region4
    if scaling_config.reference_structure:
        cv_kwargs['reference_structure'] = scaling_config.reference_structure
    # Pass feats for region resolution
    if feats:
        cv_kwargs['feats'] = feats

    # Create CV function
    cv_function = create_cv_function(cv_type, **cv_kwargs)

    return GradientScaler(
        cv_function=cv_function,
        strength=scaling_config.strength,
        cv_name=cv_name,
        warmup=scaling_config.warmup,
        cutoff=scaling_config.cutoff,
    )


def create_gradient_scaler(
    scaling_configs: List[ScalingConfig],
    feats: Optional[Dict[str, Any]] = None,
):
    """
    Create a gradient scaler from a list of ScalingConfigs.

    If a single config is provided, returns a GradientScaler.
    If multiple configs are provided, returns a CompositeGradientScaler
    that multiplies their weights together.

    Args:
        scaling_configs: List of ScalingConfig from YAML
        feats: Feature dictionary for CV creation

    Returns:
        GradientScaler or CompositeGradientScaler instance
    """
    if len(scaling_configs) == 1:
        return create_single_gradient_scaler(
            scaling_configs[0], feats=feats,
        )
    else:
        scalers = [
            create_single_gradient_scaler(cfg, feats=feats)
            for cfg in scaling_configs
        ]
        return CompositeGradientScaler(scalers)


def create_single_gradient_projector(
    projection_config: ProjectionConfig,
    feats: Optional[Dict[str, Any]] = None,
) -> GradientProjector:
    """
    Create a single GradientProjector from a ProjectionConfig.

    Args:
        projection_config: Single ProjectionConfig from YAML
        feats: Feature dictionary for CV creation

    Returns:
        GradientProjector instance
    """
    from boltz.model.potentials.collective_variables import create_cv_function

    cv_type = projection_config.collective_variable
    cv_name = cv_type

    # Build CV kwargs dict for create_cv_function
    cv_kwargs = {}

    if projection_config.groups:
        cv_kwargs['groups'] = projection_config.groups
    # Legacy atom selection
    if projection_config.atom1:
        cv_kwargs['atom1'] = projection_config.atom1
    if projection_config.atom2:
        cv_kwargs['atom2'] = projection_config.atom2
    if projection_config.atom3:
        cv_kwargs['atom3'] = projection_config.atom3
    if projection_config.atom4:
        cv_kwargs['atom4'] = projection_config.atom4
    # New region selection (preferred) - supports "A", "A:1-50", "A:5:CA", "A:1-50:CA", "A::CA"
    if projection_config.region1:
        cv_kwargs['region1'] = projection_config.region1
    if projection_config.region2:
        cv_kwargs['region2'] = projection_config.region2
    if projection_config.region3:
        cv_kwargs['region3'] = projection_config.region3
    if projection_config.region4:
        cv_kwargs['region4'] = projection_config.region4
    if projection_config.reference_structure:
        cv_kwargs['reference_structure'] = projection_config.reference_structure
    # Pass feats for region resolution
    if feats:
        cv_kwargs['feats'] = feats

    # Create CV function
    cv_function = create_cv_function(cv_type, **cv_kwargs)

    return GradientProjector(
        cv_function=cv_function,
        strength=projection_config.strength,
        direction=projection_config.direction,
        cv_name=cv_name,
        zero_threshold=projection_config.zero_threshold,
        warmup=projection_config.warmup,
        cutoff=projection_config.cutoff,
    )


def create_gradient_projector(
    projection_configs: List[ProjectionConfig],
    feats: Optional[Dict[str, Any]] = None,
):
    """
    Create a gradient projector from a list of ProjectionConfigs.

    If a single config is provided, returns a GradientProjector.
    If multiple configs are provided, returns a CompositeGradientProjector
    that applies them sequentially.

    Args:
        projection_configs: List of ProjectionConfig from YAML
        feats: Feature dictionary for CV creation

    Returns:
        GradientProjector or CompositeGradientProjector instance
    """
    if len(projection_configs) == 1:
        return create_single_gradient_projector(
            projection_configs[0], feats=feats,
        )
    else:
        projectors = [
            create_single_gradient_projector(cfg, feats=feats)
            for cfg in projection_configs
        ]
        return CompositeGradientProjector(projectors)


def create_gradient_modifier(
    scaling_configs: Optional[List[ScalingConfig]] = None,
    projection_configs: Optional[List[ProjectionConfig]] = None,
    modifier_order: str = "scale_first",
    feats: Optional[Dict[str, Any]] = None,
):
    """
    Create a gradient modifier from scaling and/or projection configs.

    This is the main factory function for combining scaling and projection.
    Returns None if neither scaling nor projection is specified.

    Args:
        scaling_configs: Optional list of ScalingConfig from YAML
        projection_configs: Optional list of ProjectionConfig from YAML
        modifier_order: 'scale_first' or 'project_first'
        feats: Feature dictionary for CV creation

    Returns:
        GradientModifier, GradientScaler, GradientProjector, or None
    """
    scaler = None
    projector = None

    if scaling_configs:
        scaler = create_gradient_scaler(
            scaling_configs, feats=feats,
        )

    if projection_configs:
        projector = create_gradient_projector(
            projection_configs, feats=feats,
        )

    # Return appropriate object based on what's configured
    if scaler is not None and projector is not None:
        return GradientModifier(
            scaler=scaler,
            projector=projector,
            order=modifier_order,
        )
    elif scaler is not None:
        return scaler
    elif projector is not None:
        return projector
    else:
        return None


def create_saxs_potential(
    config: SAXSConfig,
    saxs_data: Dict[str, Any],
    feats: Optional[Dict[str, Any]] = None,
) -> SAXSPrPotential:
    """
    Create a SAXSPrPotential from configuration.

    Args:
        config: SAXSConfig from YAML
        saxs_data: Parsed SAXS data dict with 'pr_exp' and 'r_grid'
        feats: Feature dictionary for gradient scaler creation

    Returns:
        SAXSPrPotential instance
    """
    bin_width = saxs_data['r_grid'][1] - saxs_data['r_grid'][0]
    sigma_bin = config.sigma_bin * bin_width

    potential = SAXSPrPotential(
        parameters={
            "guidance_interval": config.guidance_interval,
            "guidance_weight": 1.0,  # guidance_weight is redundant with strength, always 1.0
            "resampling_weight": 0.0,
            "pr_exp": saxs_data['pr_exp'],
            "r_grid": saxs_data['r_grid'],
            "k": config.strength,  # 'k' is internal param name, comes from config.strength
            "sigma_bin": sigma_bin,
            "loss_type": config.loss_type,
            "w2_epsilon": config.w2_epsilon,
            "w2_num_iter": config.w2_num_iter,
            "warmup": config.warmup,
            "cutoff": config.cutoff,
            "use_rep_atoms": config.use_rep_atoms,
            "rg_scale": config.rg_scale,
        }
    )

    # Attach gradient modifier if scaling and/or projection config is present
    modifier = create_gradient_modifier(
        scaling_configs=config.scaling,
        projection_configs=config.projection,
        modifier_order=config.modifier_order,
        feats=feats,
    )
    if modifier is not None:
        potential.gradient_scaler = modifier

    return potential


def create_opt_potential(
    config: OptConfig,
    feats: Optional[Dict[str, Any]] = None,
):
    """
    Create an optimization potential from configuration.

    Uses OptPotential for CV optimization.

    Args:
        config: OptConfig from YAML
        feats: Feature dictionary for CV/gradient scaler creation

    Returns:
        OptPotential instance
    """
    from boltz.model.potentials.collective_variables import create_cv_function
    from boltz.model.potentials.potentials import OptPotential

    cv_type = config.collective_variable

    # Determine if we're minimizing or maximizing
    # Positive strength = maximize CV (increase), negative strength = minimize CV (decrease)
    minimize = config.strength < 0
    k = abs(config.strength)

    # Build CV kwargs for create_cv_from_config
    cv_kwargs = {}
    if config.groups:
        cv_kwargs['groups'] = config.groups
    if config.rmsd_groups:
        cv_kwargs['rmsd_groups'] = config.rmsd_groups
    # Region selection - unified format for geometric CVs
    if config.region1:
        cv_kwargs['region1'] = config.region1
    if config.region2:
        cv_kwargs['region2'] = config.region2
    if config.region3:
        cv_kwargs['region3'] = config.region3
    if config.region4:
        cv_kwargs['region4'] = config.region4
    if config.reference_structure:
        cv_kwargs['reference_structure'] = config.reference_structure
    cv_kwargs['contact_cutoff'] = config.contact_cutoff
    cv_kwargs['selection'] = config.selection
    cv_kwargs['feats'] = feats
    # SASA CV specific
    cv_kwargs['probe_radius'] = config.probe_radius
    cv_kwargs['sasa_method'] = config.sasa_method

    cv_function = create_cv_from_config(cv_type, **cv_kwargs)

    potential = OptPotential(
        cv_function=cv_function,
        parameters={
            "guidance_interval": config.guidance_interval,
            "guidance_weight": 1.0,
            "resampling_weight": 0.0,
            "k": k,
            "minimize": minimize,
            "warmup": config.warmup,
            "cutoff": config.cutoff,
        }
    )

    # Filter out self-referential CVs from scaling/projection
    scaling_configs = None
    projection_configs = None

    if config.scaling is not None:
        non_self_scaling = [
            s for s in config.scaling if s.collective_variable != cv_type
        ]
        if len(non_self_scaling) < len(config.scaling):
            warnings.warn(
                f"Cannot scale {cv_type} gradient by {cv_type} CV (circular dependency). "
                f"Removing {cv_type} CV from scaling configs."
            )
        if non_self_scaling:
            scaling_configs = non_self_scaling

    if config.projection is not None:
        non_self_projection = [
            p for p in config.projection if p.collective_variable != cv_type
        ]
        if len(non_self_projection) < len(config.projection):
            warnings.warn(
                f"Cannot project {cv_type} gradient onto {cv_type} CV (circular dependency). "
                f"Removing {cv_type} CV from projection configs."
            )
        if non_self_projection:
            projection_configs = non_self_projection

    modifier = create_gradient_modifier(
        scaling_configs=scaling_configs,
        projection_configs=projection_configs,
        modifier_order=config.modifier_order,
        feats=feats,
    )
    if modifier is not None:
        potential.gradient_scaler = modifier

    return potential


def create_cv_from_config(
    cv_type: str,
    groups: Optional[List[str]] = None,
    rmsd_groups: Optional[List[str]] = None,
    region1: Optional[str] = None,
    region2: Optional[str] = None,
    region3: Optional[str] = None,
    region4: Optional[str] = None,
    reference_structure: Optional[str] = None,
    contact_cutoff: float = 4.5,
    selection: str = "all",
    feats: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    # SASA CV specific parameters
    probe_radius: float = 1.4,
    sasa_method: str = "lcpo",
):
    """
    Create a CV function from configuration.

    Args:
        cv_type: Type of collective variable
        groups: Chain IDs for group-based CVs (atoms to compute CV on)
        rmsd_groups: Chain IDs for alignment in pair_rmsd_grouped (atoms to align by)
        region1-4: Region specifications for geometric CVs
                   Format: "A:5:CA" (single atom), "A:1-20" (residue range), "A" (chain)
                   Multiple atoms use center of mass (COM)
        reference_structure: Path to reference structure for RMSD/native_contacts
        contact_cutoff: Cutoff distance for contacts
        selection: Selection mode for contacts (hydrophobic/polar/all)
        feats: Feature dictionary for atom/chain resolution

    Returns:
        cv_function: Callable (coords, feats, step) -> (cv_value, cv_gradient)
    """
    from boltz.model.potentials.collective_variables import create_cv_function

    kwargs = {}

    # Handle reference structure for RMSD/native_contacts
    if cv_type in ("rmsd", "native_contacts") and reference_structure:
        reference_coords = load_reference_structure(reference_structure)
        kwargs['reference_coords'] = reference_coords

    # Handle contact cutoff
    if cv_type in ("native_contacts", "coordination"):
        kwargs['contact_cutoff'] = contact_cutoff

    # Handle SASA CV parameters
    if cv_type == "sasa":
        kwargs['probe_radius'] = probe_radius
        kwargs['method'] = sasa_method

    # Handle deprecated CVs - convert to base equivalents
    if cv_type == "hinge_angle":
        warnings.warn(
            "The 'hinge_angle' CV is DEPRECATED. "
            "Use 'angle' with region1/region2/region3 instead.",
            DeprecationWarning,
            stacklevel=2
        )
        cv_type = "angle_region"

    if cv_type == "inter_chain":
        warnings.warn(
            "The 'inter_chain' CV is DEPRECATED. "
            "Use 'distance' with region1/region2 instead (e.g., region1: 'A', region2: 'B').",
            DeprecationWarning,
            stacklevel=2
        )
        cv_type = "distance_region"

    if cv_type == "inter_domain":
        warnings.warn(
            "The 'inter_domain' CV is DEPRECATED. "
            "Use 'distance' with region1/region2 instead.",
            DeprecationWarning,
            stacklevel=2
        )
        cv_type = "distance_region"

    # Explicit _region variants are deprecated - auto-conversion handles this
    if cv_type == "distance_region":
        warnings.warn(
            "The 'distance_region' CV is DEPRECATED. "
            "Use 'distance' with region1/region2 - it auto-converts to COM-based distance.",
            DeprecationWarning,
            stacklevel=2
        )

    if cv_type == "angle_region":
        warnings.warn(
            "The 'angle_region' CV is DEPRECATED. "
            "Use 'angle' with region1/region2/region3 - it auto-converts to COM-based angle.",
            DeprecationWarning,
            stacklevel=2
        )

    if cv_type == "dihedral_region":
        warnings.warn(
            "The 'dihedral_region' CV is DEPRECATED. "
            "Use 'dihedral' with region1/region2/region3/region4 - it auto-converts.",
            DeprecationWarning,
            stacklevel=2
        )

    # Handle region-based CVs (distance, angle, dihedral variants)
    # Auto-convert to region-based variants when regions are specified
    if cv_type in ("distance", "angle", "angle_enhanced", "dihedral", "dihedral_enhanced",
                   "distance_region", "min_distance", "angle_region", "dihedral_region"):
        if feats:
            from boltz.data.parse.atom_selection import build_chain_to_atom_mapping
            chain_mapping = build_chain_to_atom_mapping(feats)
            n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))

            if debug:
                print(f"[DEBUG] Resolving region specs for CV '{cv_type}':")
                print(f"  chain_mapping: {chain_mapping}")
                print(f"  n_atoms: {n_atoms}")

            # Convert region specs to masks
            if region1:
                kwargs['region1_mask'] = _resolve_region_to_mask(region1, feats, chain_mapping, n_atoms)
                if debug:
                    print(f"  region1 '{region1}' -> {kwargs['region1_mask'].sum().item()} atoms")
            if region2:
                kwargs['region2_mask'] = _resolve_region_to_mask(region2, feats, chain_mapping, n_atoms)
                if debug:
                    print(f"  region2 '{region2}' -> {kwargs['region2_mask'].sum().item()} atoms")
            if region3:
                kwargs['region3_mask'] = _resolve_region_to_mask(region3, feats, chain_mapping, n_atoms)
                if debug:
                    print(f"  region3 '{region3}' -> {kwargs['region3_mask'].sum().item()} atoms")
            if region4:
                kwargs['region4_mask'] = _resolve_region_to_mask(region4, feats, chain_mapping, n_atoms)
                if debug:
                    print(f"  region4 '{region4}' -> {kwargs['region4_mask'].sum().item()} atoms")

            # Auto-convert to region-based CV types when region masks are provided
            # This allows users to use "distance" with region specs and get COM-based distance
            if 'region1_mask' in kwargs and 'region2_mask' in kwargs:
                if cv_type == "distance":
                    cv_type = "distance_region"
                elif cv_type in ("angle", "angle_enhanced") and 'region3_mask' in kwargs:
                    if debug:
                        print(f"  Converting '{cv_type}' to 'angle_region'")
                    cv_type = "angle_region"
                elif cv_type in ("dihedral", "dihedral_enhanced") and 'region3_mask' in kwargs and 'region4_mask' in kwargs:
                    cv_type = "dihedral_region"

    # Handle chain masks for inter_chain CV
    if cv_type == "inter_chain" and groups and len(groups) >= 2 and feats:
        from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
        chain_mapping = build_chain_to_atom_mapping(feats)
        n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))

        chain1_mask = parse_group_selection_simple([groups[0]], n_atoms, chain_mapping, feats)
        chain2_mask = parse_group_selection_simple([groups[1]], n_atoms, chain_mapping, feats)

        kwargs['chain1_mask'] = chain1_mask
        kwargs['chain2_mask'] = chain2_mask

    # Handle atom mask for group-based CVs
    if cv_type in ("rg", "max_diameter", "coordination", "asphericity", "pair_rmsd", "pair_rmsd_norm_rg", "pair_rmsd_grouped", "rmsf") and groups and feats:
        from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
        chain_mapping = build_chain_to_atom_mapping(feats)
        n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))

        atom_mask = parse_group_selection_simple(groups, n_atoms, chain_mapping, feats)
        kwargs['atom_mask'] = atom_mask

    # Handle align_mask for pair_rmsd_grouped (separate alignment group)
    if cv_type == "pair_rmsd_grouped" and rmsd_groups and feats:
        from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
        chain_mapping = build_chain_to_atom_mapping(feats)
        n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))

        align_mask = parse_group_selection_simple(rmsd_groups, n_atoms, chain_mapping, feats)
        kwargs['align_mask'] = align_mask

    return create_cv_function(cv_type, **kwargs)


def _resolve_region_to_mask(
    spec: str,
    feats: Dict[str, Any],
    chain_mapping: Dict[str, Tuple[int, int]],
    n_atoms: int,
    debug: bool = False,
) -> torch.Tensor:
    """
    Resolve a region specification to an atom mask.

    Region spec formats:
        - "A:5:CA" - single atom (chain:resid:atomname)
        - "A:1-50:CA" - specific atom type across residue range (chain:start-end:atomname)
        - "A::CA" - specific atom type across whole chain (chain::atomname)
        - "A:1-20" - residue range, all atoms (chain:start-end)
        - "A" - whole chain

    Args:
        spec: Region specification string
        feats: Feature dictionary
        chain_mapping: Chain to atom range mapping
        n_atoms: Total number of atoms

    Returns:
        Boolean mask [n_atoms] with True for atoms in region
    """
    mask = torch.zeros(n_atoms, dtype=torch.bool)

    parts = spec.split(":")

    if len(parts) == 1:
        # Format: "A" - whole chain
        chain_id = parts[0]
        if chain_id in chain_mapping:
            start_idx, end_idx = chain_mapping[chain_id]
            mask[start_idx:end_idx] = True
        else:
            warnings.warn(f"Chain '{chain_id}' not found in chain_mapping. Region mask is empty.")
        return mask

    elif len(parts) == 2:
        # Format: "A:1-20" (residue range) or "A:5" (single residue)
        chain_id = parts[0]
        residue_spec = parts[1]

        if chain_id not in chain_mapping:
            warnings.warn(f"Chain '{chain_id}' not found in chain_mapping. Region mask is empty.")
            return mask

        start_idx, end_idx = chain_mapping[chain_id]

        # Parse residue range
        if "-" in residue_spec:
            try:
                res_start, res_end = residue_spec.split("-")
                res_start = int(res_start) - 1  # Convert to 0-indexed
                res_end = int(res_end) - 1  # Inclusive
            except ValueError:
                warnings.warn(f"Invalid residue range '{residue_spec}'. Region mask is empty.")
                return mask
        else:
            try:
                res_start = int(residue_spec) - 1  # Convert to 0-indexed
                res_end = res_start
            except ValueError:
                warnings.warn(f"Invalid residue '{residue_spec}'. Region mask is empty.")
                return mask

        # Get res_idx to find atoms in residue range
        res_idx_tensor = feats.get('res_idx') if feats.get('res_idx') is not None else feats.get('residue_index')
        res_idx_is_per_atom = False
        if res_idx_tensor is not None:
            import numpy as np
            if torch.is_tensor(res_idx_tensor):
                res_idx_array = res_idx_tensor.cpu().numpy()
            else:
                res_idx_array = list(res_idx_tensor)
            # Flatten if needed (e.g., [batch, n_atoms] -> [n_atoms])
            res_idx_array = np.asarray(res_idx_array).flatten()
            # Check if res_idx is per-atom (length >= n_atoms) or per-residue
            res_idx_is_per_atom = len(res_idx_array) >= n_atoms

        # Use res_idx only if it's per-atom mapping
        if res_idx_tensor is not None and res_idx_is_per_atom:
            for idx in range(start_idx, end_idx):
                if idx < len(res_idx_array):
                    if res_start <= res_idx_array[idx] <= res_end:
                        mask[idx] = True
        else:
            # Fallback: try atom_to_token first, then estimate atoms_per_res
            atom_to_token = feats.get('atom_to_token', None)
            if atom_to_token is not None:
                import numpy as np
                # atom_to_token can be:
                # - 1D array: direct mapping of atom idx -> token/residue idx
                # - 3D tensor [batch, n_atoms, n_tokens]: one-hot encoding
                if torch.is_tensor(atom_to_token):
                    att = atom_to_token.cpu()
                    if len(att.shape) == 3:
                        # 3D one-hot: [batch, n_atoms, n_tokens]
                        # Use argmax to get token index for each atom
                        # First squeeze batch dimension if present
                        if att.shape[0] == 1:
                            att = att.squeeze(0)  # [n_atoms, n_tokens]
                        # argmax along token dimension gives residue index for each atom
                        att_array = att.argmax(dim=-1).numpy()  # [n_atoms]
                    elif len(att.shape) == 2:
                        # 2D: might be [batch, n_atoms] or [n_atoms, n_tokens]
                        if att.shape[0] == 1:
                            att_array = att.squeeze(0).numpy()
                        else:
                            # Assume [n_atoms, n_tokens] one-hot
                            att_array = att.argmax(dim=-1).numpy()
                    else:
                        att_array = att.numpy().flatten()
                else:
                    att_array = np.asarray(list(atom_to_token)).flatten()

                att_array = np.asarray(att_array).flatten()

                # Get atom padding mask to exclude padding atoms
                atom_pad_mask = feats.get('atom_pad_mask', None)
                valid_atoms = None
                if atom_pad_mask is not None:
                    if torch.is_tensor(atom_pad_mask):
                        valid_atoms = atom_pad_mask.cpu().numpy().flatten().astype(bool)
                    else:
                        valid_atoms = np.asarray(atom_pad_mask).flatten().astype(bool)
                    # Squeeze batch dimension if present
                    if len(valid_atoms) > n_atoms:
                        valid_atoms = valid_atoms[:n_atoms]

                for idx in range(start_idx, end_idx):
                    if idx < len(att_array):
                        # Skip padding atoms if we have a validity mask
                        if valid_atoms is not None and idx < len(valid_atoms) and not valid_atoms[idx]:
                            continue
                        token_idx = int(att_array[idx])
                        if res_start <= token_idx <= res_end:
                            mask[idx] = True
            else:
                # Final fallback: estimate atoms_per_res from chain size
                warnings.warn("No res_idx or atom_to_token in feats. Using estimated atoms_per_res.")
                chain_atoms = end_idx - start_idx
                # Estimate number of residues from res_end (assuming 0-indexed)
                n_residues = res_end + 1  # Upper bound estimate
                if n_residues > 0:
                    atoms_per_res = max(1, chain_atoms // n_residues)
                else:
                    atoms_per_res = 5  # Conservative default for small molecules
                atom_start = start_idx + res_start * atoms_per_res
                atom_end = start_idx + (res_end + 1) * atoms_per_res
                atom_end = min(atom_end, end_idx)
                if atom_start < end_idx:
                    mask[atom_start:atom_end] = True

        return mask

    elif len(parts) == 3:
        chain_id = parts[0]
        residue_spec = parts[1]
        atom_name = parts[2]

        if residue_spec == "":
            # Format: "A::CA" - specific atom type across whole chain
            if chain_id not in chain_mapping:
                warnings.warn(f"Chain '{chain_id}' not found in chain_mapping. Region mask is empty.")
                return mask

            start_idx, end_idx = chain_mapping[chain_id]

            # Get res_idx to iterate through residues
            res_idx_tensor = feats.get('res_idx') if feats.get('res_idx') is not None else feats.get('residue_index')

            # Try to decode atom names from ref_atom_name_chars
            ref_atom_name_chars = feats.get('ref_atom_name_chars', None)
            if ref_atom_name_chars is not None:
                from boltz.model.potentials.collective_variables import decode_atom_name
                # Strip batch dimension if present: [batch, n_atoms, 4, 64] -> [n_atoms, 4, 64]
                if ref_atom_name_chars.dim() == 4:
                    ref_atom_name_chars = ref_atom_name_chars[0]

                # Match by decoded atom names
                for idx in range(start_idx, end_idx):
                    if idx < ref_atom_name_chars.shape[0]:
                        decoded_name = decode_atom_name(ref_atom_name_chars[idx])
                        if decoded_name == atom_name:
                            mask[idx] = True
                # If we found matches, return
                if mask.any():
                    return mask

            if res_idx_tensor is not None:
                import numpy as np
                if torch.is_tensor(res_idx_tensor):
                    res_idx_array = res_idx_tensor.cpu().numpy()
                else:
                    res_idx_array = list(res_idx_tensor)
                # Flatten if needed (e.g., [batch, n_atoms] -> [n_atoms])
                res_idx_array = np.asarray(res_idx_array).flatten()

                # Fallback: use positional matching with backbone order
                backbone_order = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'NZ', 'OG', 'OG1', 'SD', 'NE', 'NE1', 'NE2', 'OH', 'OD1', 'OD2', 'ND1', 'ND2', 'OE1', 'OE2', 'NH1', 'NH2', 'SG']
                atom_pos = backbone_order.index(atom_name) if atom_name in backbone_order else None

                # Find all atoms matching the name across whole chain
                current_res = None
                atoms_in_res = []
                for idx in range(start_idx, end_idx):
                    if idx >= len(res_idx_array):
                        break
                    res = res_idx_array[idx]

                    # New residue - reset counter
                    if res != current_res:
                        current_res = res
                        atoms_in_res = []

                    atoms_in_res.append(idx)
                    # Check if this is the target atom
                    if atom_pos is not None and len(atoms_in_res) - 1 == atom_pos:
                        mask[idx] = True
                    elif atom_pos is None and len(atoms_in_res) == 1:
                        # Unknown atom name, use first atom as fallback
                        mask[idx] = True
            else:
                # Fallback: try atom_to_token
                atom_to_token = feats.get('atom_to_token', None)
                if atom_to_token is not None:
                    import numpy as np
                    # Handle different atom_to_token formats
                    if torch.is_tensor(atom_to_token):
                        att = atom_to_token.cpu()
                        if len(att.shape) == 3:
                            # 3D one-hot: [batch, n_atoms, n_tokens]
                            if att.shape[0] == 1:
                                att = att.squeeze(0)
                            att_array = att.argmax(dim=-1).numpy()
                        elif len(att.shape) == 2:
                            if att.shape[0] == 1:
                                att_array = att.squeeze(0).numpy()
                            else:
                                att_array = att.argmax(dim=-1).numpy()
                        else:
                            att_array = att.numpy().flatten()
                    else:
                        att_array = np.asarray(list(atom_to_token)).flatten()

                    att_array = np.asarray(att_array).flatten()

                    backbone_order = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'NZ', 'OG', 'OG1', 'SD', 'NE', 'NE1', 'NE2', 'OH', 'OD1', 'OD2', 'ND1', 'ND2', 'OE1', 'OE2', 'NH1', 'NH2', 'SG']
                    atom_pos = backbone_order.index(atom_name) if atom_name in backbone_order else None

                    current_token = None
                    atoms_in_res = []
                    for idx in range(start_idx, end_idx):
                        if idx >= len(att_array):
                            break
                        token = int(att_array[idx])
                        if token != current_token:
                            current_token = token
                            atoms_in_res = []
                        atoms_in_res.append(idx)
                        if atom_pos is not None and len(atoms_in_res) - 1 == atom_pos:
                            mask[idx] = True
                        elif atom_pos is None and len(atoms_in_res) == 1:
                            mask[idx] = True
                else:
                    warnings.warn(f"No res_idx or atom_to_token in feats. Cannot resolve '{spec}'.")

            return mask

        elif "-" in residue_spec:
            # Format: "A:1-50:CA" - specific atom type across residue range
            if chain_id not in chain_mapping:
                warnings.warn(f"Chain '{chain_id}' not found in chain_mapping. Region mask is empty.")
                return mask

            try:
                res_start, res_end = residue_spec.split("-")
                res_start = int(res_start) - 1  # Convert to 0-indexed
                res_end = int(res_end) - 1  # Inclusive
            except ValueError:
                warnings.warn(f"Invalid residue range '{residue_spec}'. Region mask is empty.")
                return mask

            start_idx, end_idx = chain_mapping[chain_id]

            # Get atom-to-residue mapping (atom_to_token is atom-level, residue_index is token-level)
            import numpy as np
            atom_to_token = feats.get('atom_to_token', None)
            res_idx_array = None

            if atom_to_token is not None:
                # atom_to_token maps each atom to its token/residue
                if torch.is_tensor(atom_to_token):
                    att = atom_to_token.cpu()
                    if len(att.shape) == 3:
                        if att.shape[0] == 1:
                            att = att.squeeze(0)
                        res_idx_array = att.argmax(dim=-1).numpy()
                    elif len(att.shape) == 2:
                        if att.shape[0] == 1:
                            res_idx_array = att.squeeze(0).numpy()
                        else:
                            res_idx_array = att.argmax(dim=-1).numpy()
                    else:
                        res_idx_array = att.numpy().flatten()
                else:
                    res_idx_array = np.asarray(list(atom_to_token)).flatten()
                res_idx_array = np.asarray(res_idx_array).flatten()

            if res_idx_array is not None and len(res_idx_array) >= end_idx:
                # Try to decode atom names from ref_atom_name_chars first
                ref_atom_name_chars = feats.get('ref_atom_name_chars', None)
                if ref_atom_name_chars is not None:
                    from boltz.model.potentials.collective_variables import decode_atom_name
                    # Strip batch dimension if present: [batch, n_atoms, 4, 64] -> [n_atoms, 4, 64]
                    if ref_atom_name_chars.dim() == 4:
                        ref_atom_name_chars = ref_atom_name_chars[0]

                    # Match by decoded atom names in the residue range
                    for idx in range(start_idx, end_idx):
                        if idx >= len(res_idx_array):
                            break
                        res = res_idx_array[idx]
                        if res_start <= res <= res_end:
                            if idx < ref_atom_name_chars.shape[0]:
                                decoded_name = decode_atom_name(ref_atom_name_chars[idx])
                                if decoded_name == atom_name:
                                    mask[idx] = True
                    # If we found matches, return
                    if mask.any():
                        return mask

                # Fallback: Standard protein backbone atom order within a residue
                backbone_order = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'NZ', 'OG', 'OG1', 'SD', 'NE', 'NE1', 'NE2', 'OH', 'OD1', 'OD2', 'ND1', 'ND2', 'OE1', 'OE2', 'NH1', 'NH2', 'SG']
                atom_pos = backbone_order.index(atom_name) if atom_name in backbone_order else None

                # Find all atoms matching the name in the residue range
                current_res = None
                atoms_in_res = []
                for idx in range(start_idx, end_idx):
                    if idx >= len(res_idx_array):
                        break
                    res = res_idx_array[idx]

                    # New residue - check if we found the atom in previous residue
                    if res != current_res:
                        current_res = res
                        atoms_in_res = []

                    if res_start <= res <= res_end:
                        atoms_in_res.append(idx)
                        # Check if this is the target atom
                        if atom_pos is not None and len(atoms_in_res) - 1 == atom_pos:
                            mask[idx] = True
                        elif atom_pos is None:
                            # Atom name not in standard order, try first atom as fallback
                            if len(atoms_in_res) == 1:
                                mask[idx] = True
            else:
                # Fallback: try atom_to_token
                atom_to_token = feats.get('atom_to_token', None)
                if atom_to_token is not None:
                    import numpy as np
                    # Handle different atom_to_token formats
                    if torch.is_tensor(atom_to_token):
                        att = atom_to_token.cpu()
                        if len(att.shape) == 3:
                            # 3D one-hot: [batch, n_atoms, n_tokens]
                            if att.shape[0] == 1:
                                att = att.squeeze(0)
                            att_array = att.argmax(dim=-1).numpy()
                        elif len(att.shape) == 2:
                            if att.shape[0] == 1:
                                att_array = att.squeeze(0).numpy()
                            else:
                                att_array = att.argmax(dim=-1).numpy()
                        else:
                            att_array = att.numpy().flatten()
                    else:
                        att_array = np.asarray(list(atom_to_token)).flatten()

                    att_array = np.asarray(att_array).flatten()

                    backbone_order = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'NZ', 'OG', 'OG1', 'SD', 'NE', 'NE1', 'NE2', 'OH', 'OD1', 'OD2', 'ND1', 'ND2', 'OE1', 'OE2', 'NH1', 'NH2', 'SG']
                    atom_pos = backbone_order.index(atom_name) if atom_name in backbone_order else None

                    current_token = None
                    atoms_in_res = []
                    for idx in range(start_idx, end_idx):
                        if idx >= len(att_array):
                            break
                        token = int(att_array[idx])
                        if token != current_token:
                            current_token = token
                            atoms_in_res = []
                        if res_start <= token <= res_end:
                            atoms_in_res.append(idx)
                            if atom_pos is not None and len(atoms_in_res) - 1 == atom_pos:
                                mask[idx] = True
                            elif atom_pos is None and len(atoms_in_res) == 1:
                                mask[idx] = True
                else:
                    warnings.warn(f"No res_idx or atom_to_token in feats. Cannot resolve '{spec}'.")

            return mask
        else:
            # Format: "A:5:CA" - single atom
            atom_idx = _resolve_atom_spec(spec, feats, chain_mapping, debug=debug)
            if 0 <= atom_idx < n_atoms:
                mask[atom_idx] = True
            else:
                warnings.warn(f"Resolved atom index {atom_idx} out of range [0, {n_atoms}). Region mask is empty.")
            return mask

    else:
        warnings.warn(f"Invalid region spec format '{spec}'. Region mask is empty.")
        return mask


def _resolve_atom_spec(spec: str, feats: Dict[str, Any], chain_mapping: Dict[str, Tuple[int, int]], debug: bool = False) -> int:
    """
    Resolve an atom specification to an atom index.

    Args:
        spec: Atom specification (e.g., "A:15:CA" or "A:15")
        feats: Feature dictionary
        chain_mapping: Chain to atom range mapping
        debug: Whether to print debug information

    Returns:
        Atom index
    """
    import torch

    parts = spec.split(":")
    if len(parts) < 2:
        warnings.warn(f"Invalid atom spec '{spec}'. Using index 0.")
        return 0

    chain_id = parts[0]
    try:
        resid = int(parts[1])  # 1-indexed residue ID from user
    except ValueError:
        warnings.warn(f"Invalid residue ID in '{spec}'. Using index 0.")
        return 0

    atom_name = parts[2] if len(parts) >= 3 else "CA"

    if debug:
        print(f"    [DEBUG] _resolve_atom_spec('{spec}'): chain={chain_id}, resid={resid}, atom_name={atom_name}")

    # Method 1: Use atom_token_to_idx if available (preferred)
    atom_token_to_idx = feats.get('atom_token_to_idx', None)
    if atom_token_to_idx is not None:
        # This is a dict mapping (res_idx, atom_name) -> atom_idx
        # Note: res_idx is 0-indexed, resid from user is 1-indexed
        res_idx = resid - 1  # Convert to 0-indexed
        key = (res_idx, atom_name)
        if key in atom_token_to_idx:
            return atom_token_to_idx[key]

    # Method 2: Use chain_mapping + atom_to_token (per-atom residue mapping)
    if chain_id in chain_mapping:
        start_idx, end_idx = chain_mapping[chain_id]
        if debug:
            print(f"    [DEBUG] chain_mapping['{chain_id}'] = ({start_idx}, {end_idx})")

        import numpy as np
        target_res_idx = resid - 1  # Convert to 0-indexed

        # Standard protein backbone atom order within a residue
        backbone_order = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'NZ', 'OG', 'OG1', 'SD', 'NE', 'NE1', 'NE2', 'OH', 'OD1', 'OD2', 'ND1', 'ND2', 'OE1', 'OE2', 'NH1', 'NH2', 'SG']

        # Try atom_to_token first (this is per-atom mapping to residue/token)
        atom_to_token = feats.get('atom_to_token', None)
        res_idx_array = None

        if atom_to_token is not None:
            if torch.is_tensor(atom_to_token):
                att = atom_to_token.cpu()
                if len(att.shape) == 3:
                    # 3D one-hot: [batch, n_atoms, n_tokens]
                    if att.shape[0] == 1:
                        att = att.squeeze(0)
                    res_idx_array = att.argmax(dim=-1).numpy()
                elif len(att.shape) == 2:
                    if att.shape[0] == 1:
                        res_idx_array = att.squeeze(0).numpy()
                    else:
                        res_idx_array = att.argmax(dim=-1).numpy()
                else:
                    res_idx_array = att.numpy().flatten()
            else:
                res_idx_array = np.asarray(list(atom_to_token)).flatten()
            res_idx_array = np.asarray(res_idx_array).flatten()
            if debug:
                print(f"    [DEBUG] Using atom_to_token, shape: {len(res_idx_array)}")

        # Fallback: try res_idx (but this is often per-residue, not per-atom)
        if res_idx_array is None:
            res_idx_tensor = feats.get('res_idx') if feats.get('res_idx') is not None else feats.get('residue_index')
            if debug:
                print(f"    [DEBUG] res_idx_tensor: {type(res_idx_tensor)}, shape: {res_idx_tensor.shape if hasattr(res_idx_tensor, 'shape') else 'N/A'}")
            if res_idx_tensor is not None:
                if torch.is_tensor(res_idx_tensor):
                    res_idx_array = res_idx_tensor.cpu().numpy()
                else:
                    res_idx_array = list(res_idx_tensor)
                res_idx_array = np.asarray(res_idx_array).flatten()
                # Check if this is per-atom (covers atom range) or per-residue
                if len(res_idx_array) < end_idx:
                    if debug:
                        print(f"    [DEBUG] res_idx is per-residue ({len(res_idx_array)} < {end_idx}), skipping")
                    res_idx_array = None

        if res_idx_array is not None:
            # Find atoms in this residue
            # For ligands/non-polymer chains, use relative token indexing
            chain_tokens = res_idx_array[start_idx:end_idx]
            min_token_in_chain = int(min(chain_tokens)) if len(chain_tokens) > 0 else 0
            # Adjust target: for chains where tokens don't start at 0, use relative indexing
            adjusted_target = target_res_idx + min_token_in_chain

            if debug:
                unique_tokens = sorted(set(chain_tokens))
                print(f"    [DEBUG] Token indices in chain range [{start_idx}:{end_idx}]: unique={unique_tokens[:10]}... (target={target_res_idx}, adjusted={adjusted_target})")

            residue_atoms = []
            for idx in range(start_idx, end_idx):
                if idx < len(res_idx_array) and res_idx_array[idx] == adjusted_target:
                    residue_atoms.append(idx)

            # Try to decode atom names from ref_atom_name_chars
            ref_atom_name_chars = feats.get('ref_atom_name_chars', None)

            if residue_atoms:
                if debug:
                    print(f"    [DEBUG] Found {len(residue_atoms)} atoms in residue {target_res_idx}: {residue_atoms}")

                if ref_atom_name_chars is not None:
                    from boltz.model.potentials.collective_variables import decode_atom_name
                    # Strip batch dimension if present: [batch, n_atoms, 4, 64] -> [n_atoms, 4, 64]
                    if ref_atom_name_chars.dim() == 4:
                        ref_atom_name_chars = ref_atom_name_chars[0]

                    # Match by decoded atom names
                    if debug:
                        decoded_names = []
                        for idx in residue_atoms:
                            if idx < ref_atom_name_chars.shape[0]:
                                decoded_names.append((idx, decode_atom_name(ref_atom_name_chars[idx])))
                        print(f"    [DEBUG] Decoded atom names: {decoded_names}")

                    for idx in residue_atoms:
                        if idx < ref_atom_name_chars.shape[0]:
                            decoded_name = decode_atom_name(ref_atom_name_chars[idx])
                            if decoded_name == atom_name:
                                if debug:
                                    print(f"    [DEBUG] Matched atom '{atom_name}' at index {idx}")
                                return idx

                # Atom name not found in residue - for ligands, try chain-wide search before falling back
                # (fall through to chain-wide search below)

            # If residue not found or atom name not in residue, search entire chain for atom name
            # This is useful for ligands where all atoms might be in separate tokens
            if ref_atom_name_chars is not None:
                from boltz.model.potentials.collective_variables import decode_atom_name
                if ref_atom_name_chars.dim() == 4:
                    ref_atom_name_chars = ref_atom_name_chars[0]

                if debug:
                    print(f"    [DEBUG] Searching entire chain [{start_idx}:{end_idx}] for atom '{atom_name}'")

                for idx in range(start_idx, end_idx):
                    if idx < ref_atom_name_chars.shape[0]:
                        decoded_name = decode_atom_name(ref_atom_name_chars[idx])
                        if decoded_name == atom_name:
                            if debug:
                                print(f"    [DEBUG] Found atom '{atom_name}' at index {idx} via chain-wide search")
                            return idx

    # Fallback: use simple estimation
    if debug:
        print(f"    [DEBUG] Falling back to parse_atom_spec_simple for '{spec}'")
    from boltz.data.parse.atom_selection import parse_atom_spec_simple
    n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))

    try:
        result = parse_atom_spec_simple(spec, n_atoms, chain_mapping)
        if debug:
            print(f"    [DEBUG] parse_atom_spec_simple returned: {result}")
        return result
    except (ValueError, KeyError) as e:
        warnings.warn(f"Could not resolve atom spec '{spec}': {e}. Using index 0.")
        return 0


def create_explore_potential(
    config: ExploreConfig,
    feats: Optional[Dict[str, Any]] = None,
    debug: bool = False,
):
    """
    Create an explore potential (hills/repulsion) from configuration.

    Args:
        config: ExploreConfig from YAML
        feats: Feature dictionary for atom/chain resolution
        debug: Whether to print debug information

    Returns:
        MetadynamicsPotential instance
    """
    if debug:
        print(f"[DEBUG] Creating explore potential:")
        print(f"  type: {config.explore_type}")
        print(f"  cv: {config.collective_variable}")
        print(f"  strength (-> k): {config.strength}")
        print(f"  sigma (-> hill_sigma): {config.sigma}")
        print(f"  regions: {config.region1}, {config.region2}, {config.region3}, {config.region4}")
        print(f"  selection: {config.selection}")

    try:
        from boltz.model.potentials.metadynamics import MetadynamicsPotential
    except ImportError as e:
        raise RuntimeError("Metadynamics module not available.") from e

    # Create CV function
    cv_function = create_cv_from_config(
        cv_type=config.collective_variable,
        groups=config.groups,
        region1=config.region1,
        region2=config.region2,
        region3=config.region3,
        region4=config.region4,
        reference_structure=config.reference_structure,
        contact_cutoff=config.contact_cutoff,
        selection=config.selection,
        feats=feats,
        debug=debug,
        # SASA CV specific
        probe_radius=config.probe_radius,
        sasa_method=config.sasa_method,
    )

    # Generate default name if not provided
    explore_name = config.name
    if explore_name is None:
        explore_name = f"{config.explore_type}_{config.collective_variable}"

    potential = MetadynamicsPotential(
        cv_function=cv_function,
        parameters={
            "name": explore_name,
            "explore_type": config.explore_type,
            "cv_name": config.collective_variable,
            "guidance_interval": config.guidance_interval,
            "guidance_weight": 1.0,  # Deprecated - use strength instead
            "resampling_weight": 0.0,
            "hill_height": config.hill_height,
            "hill_sigma": config.sigma,
            "hill_interval": config.hill_interval,
            "well_tempered": config.well_tempered,
            "bias_factor": config.bias_factor,
            "kT": config.kT,
            "max_hills": config.max_hills,
            "k": config.strength,
            "warmup": config.warmup,
            "cutoff": config.cutoff,
            "bias_tempering": config.bias_clip,
        }
    )

    # Attach gradient modifier if scaling and/or projection config is present
    modifier = create_gradient_modifier(
        scaling_configs=config.scaling,
        projection_configs=config.projection,
        modifier_order=config.modifier_order,
        feats=feats,
    )
    if modifier is not None:
        potential.gradient_scaler = modifier

    return potential


# Backward compatibility alias
create_bias_potential = create_explore_potential


def create_steering_potential(
    config: SteeringConfig,
    feats: Optional[Dict[str, Any]] = None,
    debug: bool = False,
):
    """
    Create a steering potential from configuration.

    Uses harmonic potential to steer toward target CV value.

    Args:
        config: SteeringConfig from YAML
        feats: Feature dictionary
        debug: Whether to print debug information

    Returns:
        Potential instance
    """
    if debug:
        print(f"[DEBUG] Creating steering potential:")
        print(f"  cv: {config.collective_variable}")
        print(f"  target: {config.target}")
        print(f"  strength (-> k): {config.strength}")
        print(f"  ensemble: {config.ensemble}")
        print(f"  regions: {config.region1}, {config.region2}, {config.region3}, {config.region4}")
        print(f"  warmup: {config.warmup}, cutoff: {config.cutoff}")

    cv_type = config.collective_variable

    # All CVs (including rg) use HarmonicSteeringPotential for consistent
    # warmup/cutoff handling and unified code path.
    try:
        from boltz.model.potentials.metadynamics import MetadynamicsPotential
    except ImportError as e:
        raise RuntimeError("Metadynamics module not available.") from e

    # Create CV function
    cv_function = create_cv_from_config(
        cv_type=cv_type,
        groups=config.groups,
        region1=config.region1,
        region2=config.region2,
        region3=config.region3,
        region4=config.region4,
        reference_structure=config.reference_structure,
        contact_cutoff=config.contact_cutoff,
        selection=getattr(config, 'selection', 'all'),
        feats=feats,
        debug=debug,
        # SASA CV specific
        probe_radius=getattr(config, 'probe_radius', 1.4),
        sasa_method=getattr(config, 'sasa_method', 'lcpo'),
    )

    # Create a harmonic steering wrapper
    potential = HarmonicSteeringPotential(
        cv_function=cv_function,
        target=config.target,
        parameters={
            "guidance_interval": config.guidance_interval,
            "guidance_weight": 1.0,  # guidance_weight is redundant with strength, always 1.0
            "resampling_weight": 0.0,
            "k": config.strength,  # 'k' is internal param name, comes from config.strength
            "warmup": config.warmup,
            "cutoff": config.cutoff,
            "gaussian_noise_scale": config.gaussian_noise_scale,
            "ensemble": config.ensemble,  # Per-ensemble vs per-sample loss
        }
    )

    # Attach gradient modifier if scaling and/or projection config is present
    modifier = create_gradient_modifier(
        scaling_configs=config.scaling,
        projection_configs=config.projection,
        modifier_order=config.modifier_order,
        feats=feats,
    )
    if modifier is not None:
        potential.gradient_scaler = modifier

    return potential


class HarmonicSteeringPotential(Potential):
    """
    Harmonic steering potential for arbitrary CVs.

    Energy = 0.5 * k * (CV - target)^2

    Supports two loss modes:
        - Per-sample (ensemble=False): Each sample steered independently
        - Per-ensemble (ensemble=True): CV averaged across samples, single loss
    """

    def __init__(
        self,
        cv_function,
        target: float,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parameters)
        self.cv_function = cv_function
        self.target = target
        self._k = parameters.get('k', 1.0) if parameters else 1.0
        self._warmup = parameters.get('warmup', 0.0) if parameters else 0.0
        self._cutoff = parameters.get('cutoff', 0.75) if parameters else 0.75
        self._noise_scale = parameters.get('gaussian_noise_scale', 0.0) if parameters else 0.0
        self._ensemble = parameters.get('ensemble', False) if parameters else False

    def compute_variable(
        self,
        coords: torch.Tensor,
        index,
        ref_coords=None,
        ref_mask=None,
        compute_gradient: bool = False,
    ):
        """Compute collective variable value(s)."""
        if compute_gradient:
            return self.cv_function(coords, {}, 0)
        else:
            cv_values, _ = self.cv_function(coords, {}, 0)
            return cv_values

    def compute_function(
        self,
        cv_values: torch.Tensor,
        k: float = 1.0,
        negation_mask=None,
        compute_derivative: bool = False,
    ):
        """
        Compute harmonic energy from CV values.

        Energy = 0.5 * k * (CV - target)^2
        """
        delta = cv_values - self.target
        energy = 0.5 * k * delta ** 2

        if compute_derivative:
            dE_dCV = k * delta
            return energy, dE_dCV
        return energy

    def compute_gradient(
        self,
        coords: torch.Tensor,
        feats: dict,
        parameters: dict,
        step: int = 0,
    ) -> torch.Tensor:
        """Compute steering gradient.

        Per-sample mode (ensemble=False):
            Each sample has its own CV value and gets its own gradient.
            grad_i = k * (CV_i - target) * dCV_i/dr

        Per-ensemble mode (ensemble=True):
            CV values averaged first, then loss computed from mean.
            CV_mean = mean(CV_i), loss = k * (CV_mean - target)^2
            grad_i = k * (CV_mean - target) * dCV_i/dr / N
            (divided by N because each sample contributes 1/N to the mean)
        """
        current_step = parameters.get('_step_idx', 0)

        # Compute CV and gradient
        cv_values, cv_gradient = self.cv_function(coords, feats, current_step)
        # cv_values: [multiplicity]
        # cv_gradient: [multiplicity, N_atoms, 3]

        k = parameters.get('k', self._k)

        # Apply warmup and cutoff (0.0 = start of diffusion, 1.0 = end)
        progress = parameters.get('_relaxation', 0.0)
        if progress < self._warmup or progress > self._cutoff:
            k = 0.0

        ensemble_mode = parameters.get('ensemble', self._ensemble)

        if ensemble_mode:
            # Per-ensemble mode: steer ensemble mean toward target
            # All samples move coherently in the same direction
            multiplicity = cv_values.shape[0]
            cv_mean = cv_values.mean()
            delta = cv_mean - self.target

            # Use mean gradient direction for coherent ensemble movement
            # This ensures all samples move together toward the target
            mean_gradient = cv_gradient.mean(dim=0)  # [N_atoms, 3]

            # Normalize mean gradient to have same scale as individual gradients
            # (CV functions normalize to max norm 1.0, so we preserve that)
            grad_norm = mean_gradient.norm(dim=-1, keepdim=True).max()
            if grad_norm > 1e-8:
                mean_gradient = mean_gradient / grad_norm

            # Apply same gradient to all samples (broadcast)
            # k * delta gives similar strength to per-sample mode
            gradient = (k * delta) * mean_gradient.unsqueeze(0).expand(multiplicity, -1, -1)
        else:
            # Per-sample mode: each sample steered independently
            delta = cv_values - self.target
            force_magnitude = k * delta
            force_expanded = force_magnitude.unsqueeze(-1).unsqueeze(-1)
            gradient = force_expanded * cv_gradient

        # Add noise if requested
        if self._noise_scale > 0:
            noise = torch.randn_like(gradient) * self._noise_scale
            gradient = gradient + noise

        # Apply gradient scaling if scaler is set
        if self.gradient_scaler is not None:
            gradient = self.gradient_scaler.apply(gradient, coords, feats, step, progress)

        return gradient

    def compute_args(self, feats: dict, parameters: dict):
        """Prepare arguments for compute.

        Returns empty index so base compute() safely returns zeros.
        """
        k = parameters.get('k', self._k)
        return torch.empty(1, 0, dtype=torch.long), (k,), None, None, None


def create_chemical_shift_potential(
    config: ChemicalShiftConfig,
    feats: Optional[Dict[str, Any]] = None,
):
    """
    Create a ChemicalShiftPotential from configuration.

    Only CA and CB chemical shifts are supported using the CheShift algorithm.

    Args:
        config: ChemicalShiftConfig from YAML
        feats: Feature dictionary for atom/residue resolution

    Returns:
        ChemicalShiftPotential instance

    Raises:
        ValueError: If no valid shift files found
        NotImplementedError: If unsupported nuclei are requested
    """
    from boltz.model.potentials.chemical_shift import ChemicalShiftPotential, load_shift_file

    # Load experimental shifts from PLUMED-format files
    # ONLY CA and CB are supported
    exp_shifts = {}
    shift_files = {
        'CA': config.ca_shifts,
        'CB': config.cb_shifts,
    }

    for nucleus, filepath in shift_files.items():
        if filepath and Path(filepath).exists():
            exp_shifts[nucleus] = load_shift_file(filepath)

    if not exp_shifts:
        raise ValueError(
            "No valid shift files found. At least one shift file must be specified "
            "(ca_shifts or cb_shifts). Note: Only CA and CB are supported."
        )

    potential = ChemicalShiftPotential(
        parameters={
            'exp_shifts': exp_shifts,
            'loss_type': config.loss_type,
            'k': config.strength,
            'guidance_interval': config.guidance_interval,
            'warmup': config.warmup,
            'cutoff': config.cutoff,
            'bias_tempering': config.bias_tempering,
            'auto_offset': config.auto_offset,
            'ca_dss_offset': config.ca_dss_offset,
            'cb_dss_offset': config.cb_dss_offset,
        }
    )

    # Attach gradient modifier if scaling and/or projection config is present
    modifier = create_gradient_modifier(
        scaling_configs=config.scaling,
        projection_configs=config.projection,
        modifier_order=config.modifier_order,
        feats=feats,
    )
    if modifier is not None:
        potential.gradient_scaler = modifier

    return potential


def create_potentials_from_yaml(
    config: MetadiffusionConfig,
    feats: Optional[Dict[str, Any]] = None,
    saxs_data: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> List[Potential]:
    """
    Create all potentials from a MetadiffusionConfig.

    This is the main factory function that converts YAML configuration
    into instantiated potential objects.

    Args:
        config: MetadiffusionConfig from YAML parsing
        feats: Feature dictionary for atom/chain resolution
        saxs_data: Pre-parsed SAXS data (if SAXS steering enabled)
        debug: Whether to print debug information

    Returns:
        List of Potential instances
    """
    potentials = []

    # Create opt potentials (for CV optimization)
    for opt_config in config.opt:
        opt_potential = create_opt_potential(opt_config, feats=feats)
        potentials.append(opt_potential)

    # Create SAXS potentials (one for each SAXS config entry)
    if config.saxs:
        if saxs_data is None:
            raise ValueError(
                "SAXS steering configured but no SAXS data provided. "
                "Load SAXS data with parse_gnom_file() and pass as saxs_data parameter."
            )
        for saxs_config in config.saxs:
            potentials.append(create_saxs_potential(
                saxs_config, saxs_data, feats=feats,
            ))

    # Create explore potentials (hills/repulsion)
    for explore_cfg in config.explore:
        potentials.append(create_explore_potential(
            explore_cfg,
            feats=feats,
            debug=debug,
        ))

    # Create steering potentials
    for steering in config.steering:
        potentials.append(create_steering_potential(
            steering,
            feats=feats,
            debug=debug,
        ))

    # Create chemical shift potentials
    for cs_config in config.chemical_shift:
        potentials.append(create_chemical_shift_potential(cs_config, feats=feats))

    return potentials


def parse_gnom_file(filepath: str, units: str = "auto") -> Dict[str, Any]:
    """
    Parse a GNOM .out file to extract P(r) data.

    Args:
        filepath: Path to GNOM output file
        units: Input data units - "nm", "angstrom", or "auto" (auto-detect)
               Default is "auto" which detects based on max r value

    Returns:
        Dict with 'pr_exp', 'r_grid', and metadata
    """
    import numpy as np

    r_values = []
    pr_values = []
    errors = []

    in_pr_section = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # GNOM outputs P(r) in a section starting after specific header
            if 'R          P(R)' in line or 'Distance distribution' in line:
                in_pr_section = True
                continue

            if in_pr_section:
                # Skip header/separator lines
                if not line or line.startswith('#') or line.startswith('-'):
                    continue

                # End of P(r) section
                if line.startswith('Reciprocal space') or 'Total Estimate' in line:
                    break

                parts = line.split()
                if len(parts) >= 2:
                    try:
                        r = float(parts[0])
                        pr = float(parts[1])
                        r_values.append(r)
                        pr_values.append(pr)
                        if len(parts) >= 3:
                            errors.append(float(parts[2]))
                    except ValueError:
                        continue

    if not r_values:
        raise ValueError(f"Could not parse P(r) data from GNOM file: {filepath}")

    r_grid = np.array(r_values)
    pr_exp = np.array(pr_values)

    # Auto-detect units if needed
    if units == "auto":
        max_r = r_grid.max()
        # If max_r >= 50, file is likely already in Angstroms
        # (most proteins have Dmax < 500 Å = 50 nm)
        if max_r >= 50:
            units = "angstrom"
            print(f"Auto-detected SAXS units: angstrom (max_r={max_r:.1f} >= 50)")
        else:
            units = "nm"
            print(f"Auto-detected SAXS units: nm (max_r={max_r:.1f} < 50)")

    # Convert from nm to Angstroms if needed
    if units == "nm":
        r_grid = r_grid * 10.0  # nm -> Å

    # Normalize
    pr_exp = pr_exp / pr_exp.sum() if pr_exp.sum() > 0 else pr_exp

    return {
        'r_grid': torch.tensor(r_grid, dtype=torch.float32),
        'pr_exp': torch.tensor(pr_exp, dtype=torch.float32),
        'errors': torch.tensor(errors, dtype=torch.float32) if errors else None,
    }
