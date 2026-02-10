import multiprocessing
import os
import pickle
import platform
import shutil
import tarfile
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from rdkit import Chem
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.mol import load_canonicals
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml, parse_yaml_with_metadiffusion
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MOL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
# Legacy: MODEL_URL for backwards compatibility with tests
MODEL_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"

BOLTZ1_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
]

BOLTZ2_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
]

BOLTZ2_AFFINITY_URL_WITH_FALLBACK = [
    "https://model-gateway.boltz.bio/boltz2_aff.ckpt",
    "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
]


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    extra_mols_dir: Optional[Path] = None


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = False


@dataclass
class PairformerArgsV2:
    """Pairformer arguments."""

    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class Boltz2DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 1
    # Guidance mode: "combine" (default), "post", or "pre"
    # - combine: compute both pre and post gradients, add displacement vectors, apply to x_0
    # - post: apply guidance to x_0 prediction after denoising only
    # - pre: apply guidance to noisy coords before denoising only
    guidance_mode: str = "combine"

    # SAXS P(r) fitting parameters - multiple entries supported
    # Each entry is a dict with: pr_file, loss_type, strength, etc.
    saxs_pr_steering: bool = False  # Flag to enable SAXS output generation
    saxs_configs: Optional[List[Dict[str, Any]]] = None  # List of SAXS config dicts
    saxs_pr_data_cache: Optional[Dict[str, Dict[str, torch.Tensor]]] = None  # Cached P(r) data by file

    # Generic CV steering configs (from metadiffusion YAML steering section)
    # Each entry has: collective_variable, target, strength, etc.
    steering_configs: Optional[List[Dict[str, Any]]] = None

    # Generic CV explore configs (from metadiffusion YAML explore section)
    # Each entry has: collective_variable, type (hills/repulsion), strength, sigma, etc.
    explore_configs: Optional[List[Dict[str, Any]]] = None

    # Generic CV optimization configs (from metadiffusion YAML opt section)
    # Each entry has: collective_variable, strength (sign determines minimize/maximize), etc.
    opt_configs: Optional[List[Dict[str, Any]]] = None

    # Chemical shift steering configs (from metadiffusion YAML chemical_shift section)
    # Each entry has: ca_shifts, cb_shifts, strength, loss_type, etc.
    chemical_shift_configs: Optional[List[Dict[str, Any]]] = None

    # Denoising tempering: max per-atom displacement per denoising step (Å)
    # Preserves relative magnitudes by uniform scaling across all atoms
    # None = disabled (default), set to e.g. 0.5 to prevent phase transition/explosion
    denoise_tempering: Optional[float] = None
    # Total bias tempering: max per-atom displacement from ALL potentials combined (Å)
    # Applied after individual per-potential bias_tempering limits
    # None = disabled (default), set to e.g. 1.0 to limit total guidance displacement
    total_bias_tempering: Optional[float] = None
    # NOTE: bias_tempering is now per-potential (specified in each opt/steer/explore/saxs config)

    # Noise scale for stochastic sampling (controls eps noise injection)
    # None = use Boltz default (1.003 for Boltz2, 0.901 for Boltz1)
    # Set to 0.0 for deterministic sampling
    noise_scale: Optional[float] = None

    # Debug mode: print debug information during potential creation
    debug: bool = False

    # Per-record steering: maps record_id -> full steering_args dict
    # When records have different metadiffusion configs, each gets its own steering
    per_record_steering: Optional[Dict[str, Any]] = None

    # Metadynamics parameters (enhanced sampling with Gaussian hills)
    metadynamics: bool = False
    metadynamics_cv: str = "rg"  # CV type: "rg", "distance", "asphericity"
    metadynamics_hill_height: float = 0.5  # Base Gaussian hill height
    metadynamics_hill_sigma: float = 5.0  # Gaussian hill width in CV units
    metadynamics_hill_interval: int = 5  # Steps between hill deposits
    metadynamics_guidance_weight: float = 0.5  # Gradient descent weight
    metadynamics_guidance_interval: int = 1  # Apply bias gradient every N steps
    metadynamics_well_tempered: bool = False  # Use well-tempered metadynamics
    metadynamics_bias_factor: float = 10.0  # Bias factor gamma for well-tempered
    metadynamics_kT: float = 2.5  # Temperature kT for well-tempered
    metadynamics_max_hills: int = 1000  # Maximum hills to store


# Valid collective variable names for metadiffusion
VALID_CVS = {
    # Structural
    "rg", "distance", "min_distance", "max_diameter", "asphericity",
    # Angle/Dihedral
    "angle", "dihedral", "angle_enhanced", "dihedral_enhanced",
    # RMSD/Fluctuation
    "rmsd", "pair_rmsd", "pair_rmsd_norm_rg", "pair_rmsd_grouped", "rmsf",
    # Contacts
    "native_contacts", "coordination", "hbond_count", "salt_bridges",
    "contact_order", "local_contacts", "sasa",
    # Secondary structure
    "alpharmsd", "antibetarmsd", "parabetarmsd",
    # Shape
    "acylindricity", "shape_gyration",
    # Content
    "helix_content", "sheet_content",
    # Other
    "dipole_moment",
    # DEPRECATED - kept for backward compatibility, will be removed
    "distance_region", "angle_region", "dihedral_region",  # use distance/angle/dihedral with regions
    "inter_chain", "inter_domain",  # use distance with region1/region2
    "hinge_angle",  # use angle with region1/region2/region3
}

# Valid explore types
VALID_EXPLORE_TYPES = {"hills", "repulsion", "variance"}
VALID_BIAS_TYPES = VALID_EXPLORE_TYPES  # Backward compatibility alias


def apply_metadiffusion_to_steering_args(
    steering_args: BoltzSteeringParams,
    metadiff_dict: dict,
    base_path: Path,
) -> None:
    """Apply metadiffusion config from YAML to steering args.

    All metadiffusion settings are configured exclusively via YAML.

    Args:
        steering_args: BoltzSteeringParams to update
        metadiff_dict: Dict from metadiffusion JSON file
        base_path: Base path for resolving relative file paths
    """
    # Apply SAXS configs (multiple entries supported)
    saxs_list = metadiff_dict.get("saxs", [])
    if saxs_list:
        steering_args.saxs_configs = []
        steering_args.saxs_pr_data_cache = {}

        for saxs in saxs_list:
            if "guidance_weight" in saxs:
                raise ValueError(
                    "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
                )
            pr_file = saxs.get("pr_file")
            if not pr_file:
                warnings.warn(
                    "SAXS entry missing 'pr_file' field, skipping."
                )
                continue

            # Resolve relative path
            pr_path = Path(pr_file)
            if not pr_path.is_absolute():
                pr_path = base_path / pr_file

            if pr_path.exists():
                # Validate loss_type
                loss_type = saxs.get("loss_type", "mse")
                valid_loss_types = {"mse", "w2", "hybrid_w2_mse", "chi2", "rg", "w1", "mae", "kl", "cramer"}
                if loss_type not in valid_loss_types:
                    warnings.warn(
                        f"Invalid SAXS loss_type '{loss_type}'. "
                        f"Valid types: {sorted(valid_loss_types)}. Using 'mse'."
                    )
                    loss_type = "mse"

                # Parse bins for uniform resampling
                bins = saxs.get("bins")
                if bins is not None:
                    bins = int(bins)
                    if bins < 2:
                        warnings.warn(f"SAXS bins must be >= 2, got {bins}. Ignoring.")
                        bins = None

                # Parse bins_range
                bins_range = saxs.get("bins_range")
                if bins_range is not None:
                    if isinstance(bins_range, (list, tuple)) and len(bins_range) == 2:
                        bins_range = [float(bins_range[0]), float(bins_range[1])]
                        if bins_range[0] >= bins_range[1]:
                            warnings.warn(f"SAXS bins_range[0] must be < bins_range[1]. Ignoring.")
                            bins_range = None
                    else:
                        warnings.warn(f"SAXS bins_range must be [r_min, r_max]. Ignoring.")
                        bins_range = None

                # Store config with resolved path
                config = {
                    "pr_file": str(pr_path),
                    "loss_type": loss_type,
                    "strength": float(saxs.get("strength", 1.0)),
                    "guidance_interval": int(saxs.get("guidance_interval", 1)),
                    "warmup": float(saxs.get("warmup", 0.0)),
                    "cutoff": float(saxs.get("cutoff", 0.9)),
                    "sigma_bin": float(saxs.get("sigma_bin", 0.5)),
                    "units": saxs.get("units", "auto"),  # nm, angstrom, or auto
                    "w2_epsilon": float(saxs.get("w2_epsilon", 0.1)),
                    "w2_num_iter": int(saxs.get("w2_num_iter", 100)),
                    "rg_scale": float(saxs.get("rg_scale", 1.0)),
                    "use_rep_atoms": bool(saxs.get("use_rep_atoms", False)),
                    "bins": bins,
                    "bins_range": bins_range,
                    "bias_clip": saxs.get("bias_clip"),
                    "projection": saxs.get("projection"),
                    "scaling": saxs.get("scaling"),
                }
                steering_args.saxs_configs.append(config)
                # Enable SAXS output generation when at least one valid config exists
                steering_args.saxs_pr_steering = True
            else:
                warnings.warn(
                    f"SAXS PR file not found: {pr_path}. Skipping this SAXS entry."
                )

    # Apply steering configs - all CVs go through generic system
    steering_list = metadiff_dict.get("steering", [])
    for steer in steering_list:
        if "guidance_weight" in steer:
            raise ValueError(
                "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
            )
        cv = steer.get("collective_variable", "")

        # Validate CV name
        if not cv:
            warnings.warn(
                "Steering entry missing 'collective_variable' field, skipping."
            )
            continue
        if cv not in VALID_CVS:
            warnings.warn(
                f"Unknown collective_variable '{cv}' in steering. "
                f"Valid CVs: {sorted(VALID_CVS)}. Skipping."
            )
            continue

        # Handle target_from_saxs for Rg CV
        target = steer.get("target")
        target_from_saxs = steer.get("target_from_saxs")
        auto_rg_scale = steer.get("auto_rg_scale", 1.0)

        # Validate that target or target_from_saxs is provided
        if target is None and target_from_saxs is None:
            warnings.warn(
                f"Steering on '{cv}' missing both 'target' and 'target_from_saxs'. "
                "One must be provided. Skipping."
            )
            continue

        # Validate target_from_saxs is only used with rg CV
        if target_from_saxs is not None and cv != "rg":
            warnings.warn(
                f"'target_from_saxs' is only valid for collective_variable='rg', "
                f"got '{cv}'. Ignoring target_from_saxs."
            )
            target_from_saxs = None

        # Convert target to float if provided
        if target is not None:
            try:
                target = float(target)
            except (TypeError, ValueError):
                warnings.warn(
                    f"Invalid target value '{target}' for CV '{cv}'. "
                    "Must be a number. Skipping."
                )
                continue

        if target_from_saxs:
            # Resolve path
            saxs_path = Path(target_from_saxs)
            if not saxs_path.is_absolute():
                saxs_path = base_path / target_from_saxs
            if saxs_path.exists():
                target_from_saxs = str(saxs_path)
            else:
                warnings.warn(
                    f"SAXS file for target_from_saxs not found: {saxs_path}. "
                    "Ignoring target_from_saxs."
                )
                target_from_saxs = None

        # Resolve reference structure path if provided
        ref_struct = steer.get("reference_structure")
        if ref_struct:
            ref_path = Path(ref_struct)
            if not ref_path.is_absolute():
                ref_path = base_path / ref_struct
            if ref_path.exists():
                ref_struct = str(ref_path)
            else:
                warnings.warn(
                    f"Reference structure not found: {ref_path}. "
                    "Ignoring reference_structure."
                )
                ref_struct = None

        # Add to generic steering configs
        if steering_args.steering_configs is None:
            steering_args.steering_configs = []

        steering_args.steering_configs.append({
            "collective_variable": cv,
            "target": target,
            "target_from_saxs": target_from_saxs,
            "auto_rg_scale": auto_rg_scale,
            "strength": float(steer.get("strength", 1.0)),
            "guidance_interval": int(steer.get("guidance_interval", 1)),
            "warmup": float(steer.get("warmup", 0.0)),
            "cutoff": float(steer.get("cutoff", 0.75)),
            "contact_cutoff": float(steer.get("contact_cutoff", 4.5)),
            "ensemble": steer.get("ensemble", False),
            "reference_structure": ref_struct,
            "atom1": steer.get("atom1"),  # Raw string spec like "A:1:CA"
            "atom2": steer.get("atom2"),
            "atom3": steer.get("atom3"),
            "atom4": steer.get("atom4"),
            # Region specifications for angle/distance/dihedral CVs
            "region1": steer.get("region1"),
            "region2": steer.get("region2"),
            "region3": steer.get("region3"),
            "region4": steer.get("region4"),
            "groups": steer.get("groups"),
            "bias_clip": steer.get("bias_clip"),
            # SASA CV specific
            "probe_radius": float(steer.get("probe_radius", 1.4)),
            "sasa_method": steer.get("sasa_method", "lcpo"),
            "rmsd_groups": steer.get("rmsd_groups"),
            "selection": steer.get("selection", "all"),
        })

    # Apply explore configs (hills/repulsion)
    # Support both "explore" (new) and "biases" (deprecated) keys
    explore_list = metadiff_dict.get("explore", [])
    if not explore_list and "biases" in metadiff_dict:
        warnings.warn(
            "The 'biases' key in metadiffusion config is deprecated. Use 'explore' instead.",
            DeprecationWarning
        )
        explore_list = metadiff_dict.get("biases", [])
    for explore in explore_list:
        if "guidance_weight" in explore:
            raise ValueError(
                "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
            )
        # JSON uses "type" key from YAML serialization
        explore_type = explore.get("type", "") or explore.get("explore_type", "") or explore.get("bias_type", "")
        cv = explore.get("collective_variable", "")

        # Validate CV name
        if not cv:
            warnings.warn(
                "Explore entry missing 'collective_variable' field, skipping."
            )
            continue
        if cv not in VALID_CVS:
            warnings.warn(
                f"Unknown collective_variable '{cv}' in explore. "
                f"Valid CVs: {sorted(VALID_CVS)}. Skipping."
            )
            continue

        # Validate explore_type
        if explore_type not in VALID_EXPLORE_TYPES:
            warnings.warn(
                f"Invalid or missing explore type '{explore_type}' for CV '{cv}'. "
                f"Valid types: {sorted(VALID_EXPLORE_TYPES)}. Skipping."
            )
            continue

        # Add to explore_configs list
        if steering_args.explore_configs is None:
            steering_args.explore_configs = []
        steering_args.explore_configs.append({
            "collective_variable": cv,
            "type": explore_type,
            "strength": explore.get("strength", 256.0),
            "sigma": explore.get("sigma", 5.0),
            "guidance_weight": 1.0,  # Deprecated - use strength instead
            "guidance_interval": explore.get("guidance_interval", 1),
            "warmup": explore.get("warmup", 0.0),
            "cutoff": explore.get("cutoff", 0.75),
            "hill_height": explore.get("hill_height", 0.5),
            "hill_interval": explore.get("hill_interval", 5),
            "well_tempered": explore.get("well_tempered", False),
            "bias_factor": explore.get("bias_factor", 10.0),
            "kT": explore.get("kT", 2.5),
            "max_hills": explore.get("max_hills", 1000),
            "contact_cutoff": explore.get("contact_cutoff", 4.5),
            "reference_structure": explore.get("reference_structure"),
            "bias_clip": explore.get("bias_clip"),
            # Region specifications for angle/distance/dihedral CVs
            "region1": explore.get("region1"),
            "region2": explore.get("region2"),
            "region3": explore.get("region3"),
            "region4": explore.get("region4"),
            "groups": explore.get("groups"),
            "rmsd_groups": explore.get("rmsd_groups"),
        })

    # Apply opt configs (CV optimization - minimize or maximize)
    opt_list = metadiff_dict.get("opt", [])
#     print(f"DEBUG apply_metadiffusion: opt_list={opt_list}", flush=True)
    for opt in opt_list:
        if "guidance_weight" in opt:
            raise ValueError(
                "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
            )
        cv = opt.get("collective_variable", "")

        # Validate CV name
        if not cv:
            warnings.warn(
                "Opt entry missing 'collective_variable' field, skipping."
            )
            continue
        if cv not in VALID_CVS:
            warnings.warn(
                f"Unknown collective_variable '{cv}' in opt. "
                f"Valid CVs: {sorted(VALID_CVS)}. Skipping."
            )
            continue

        # Add to opt_configs list
        # Resolve reference structure path if provided
        ref_struct = opt.get("reference_structure")
        if ref_struct:
            ref_path = Path(ref_struct)
            if not ref_path.is_absolute():
                ref_path = base_path / ref_struct
            if ref_path.exists():
                ref_struct = str(ref_path)
            else:
                warnings.warn(
                    f"Reference structure not found: {ref_path}. "
                    "Ignoring reference_structure."
                )
                ref_struct = None

        if steering_args.opt_configs is None:
            steering_args.opt_configs = []
        steering_args.opt_configs.append({
            "collective_variable": cv,
            "strength": opt.get("strength", 1.0),
            "guidance_interval": opt.get("guidance_interval", 1),
            "warmup": opt.get("warmup", 0.0),
            "cutoff": opt.get("cutoff", 0.75),
            "log_gradient": opt.get("log_gradient", False),
            "contact_cutoff": opt.get("contact_cutoff", 4.5),
            "reference_structure": ref_struct,
            # Region selection - prefer region1-4, fall back to atom1-4 for backward compat
            "region1": opt.get("region1") or opt.get("atom1"),
            "region2": opt.get("region2") or opt.get("atom2"),
            "region3": opt.get("region3") or opt.get("atom3"),
            "region4": opt.get("region4") or opt.get("atom4"),
            "groups": opt.get("groups"),
            "bias_clip": opt.get("bias_clip"),
            # SASA CV specific
            "probe_radius": float(opt.get("probe_radius", 1.4)),
            "sasa_method": opt.get("sasa_method", "lcpo"),
            # RMSD specific
            "rmsd_groups": opt.get("rmsd_groups"),
        })

    # Apply chemical shift configs
    chemical_shift_list = metadiff_dict.get("chemical_shift", [])
    for cs in chemical_shift_list:
        if "guidance_weight" in cs:
            raise ValueError(
                "The 'guidance_weight' parameter is deprecated. Use 'strength' instead."
            )

        # Resolve shift file paths
        ca_shifts = cs.get("ca_shifts")
        cb_shifts = cs.get("cb_shifts")

        if ca_shifts:
            ca_path = Path(ca_shifts)
            if not ca_path.is_absolute():
                ca_path = base_path / ca_shifts
            if ca_path.exists():
                ca_shifts = str(ca_path)
            else:
                warnings.warn(f"CA shifts file not found: {ca_path}. Skipping.")
                ca_shifts = None

        if cb_shifts:
            cb_path = Path(cb_shifts)
            if not cb_path.is_absolute():
                cb_path = base_path / cb_shifts
            if cb_path.exists():
                cb_shifts = str(cb_path)
            else:
                warnings.warn(f"CB shifts file not found: {cb_path}. Skipping.")
                cb_shifts = None

        if not ca_shifts and not cb_shifts:
            warnings.warn("Chemical shift config has no valid shift files. Skipping.")
            continue

        if steering_args.chemical_shift_configs is None:
            steering_args.chemical_shift_configs = []

        steering_args.chemical_shift_configs.append({
            "ca_shifts": ca_shifts,
            "cb_shifts": cb_shifts,
            "strength": float(cs.get("strength", 1.0)),
            "loss_type": cs.get("loss_type", "chi"),
            "guidance_interval": int(cs.get("guidance_interval", 1)),
            "warmup": float(cs.get("warmup", 0.0)),
            "cutoff": float(cs.get("cutoff", 0.9)),
            "bias_clip": cs.get("bias_clip"),
        })

    # Parse denoise_clip (top-level metadiffusion parameter)
    if "denoise_clip" in metadiff_dict:
        value = metadiff_dict["denoise_clip"]
        steering_args.denoise_tempering = float(value) if value is not None else None

    # Parse total_bias_clip (top-level limit on ALL potentials combined)
    if "total_bias_clip" in metadiff_dict:
        value = metadiff_dict["total_bias_clip"]
        steering_args.total_bias_tempering = float(value) if value is not None else None

    # Parse noise_scale (top-level YAML parameter)
    if "noise_scale" in metadiff_dict:
        value = metadiff_dict["noise_scale"]
        steering_args.noise_scale = float(value) if value is not None else None

    # Parse guidance_mode (top-level metadiffusion parameter)
    # - "post" (default): apply guidance to x_0 prediction after denoising
    # - "pre": apply guidance to noisy coords before denoising
    # - "combine": compute both, add displacement vectors, apply to x_0
    if "guidance_mode" in metadiff_dict:
        steering_args.guidance_mode = str(metadiff_dict["guidance_mode"])
    elif "guidance_before_denoising" in metadiff_dict:
        # Backward compatibility
        steering_args.guidance_mode = "pre" if bool(metadiff_dict["guidance_before_denoising"]) else "post"

    # NOTE: bias_tempering is now per-potential (specified in each opt/steer/explore/saxs config)
    # It's parsed into each config's dict and passed to potentials


@rank_zero_only
def download_boltz1(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ1_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ1_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


@rank_zero_only
def download_boltz2(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    mols = cache / "mols"
    tar_mols = cache / "mols.tar"
    if not tar_mols.exists():
        click.echo(
            f"Downloading the CCD data to {tar_mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        urllib.request.urlretrieve(MOL_URL, str(tar_mols))  # noqa: S310
    if not mols.exists():
        click.echo(
            f"Extracting the CCD data to {mols}. "
            "This may take a bit of time. You may change the cache directory "
            "with the --cache flag."
        )
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(cache)  # noqa: S202

    # Download model
    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the Boltz-2 weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue

    # Download affinity model
    affinity_model = cache / "boltz2_aff.ckpt"
    if not affinity_model.exists():
        click.echo(
            f"Downloading the Boltz-2 affinity weights to {affinity_model}. You may "
            "change the cache directory with the --cache flag."
        )
        for i, url in enumerate(BOLTZ2_AFFINITY_URL_WITH_FALLBACK):
            try:
                urllib.request.urlretrieve(url, str(affinity_model))  # noqa: S310
                break
            except Exception as e:  # noqa: BLE001
                if i == len(BOLTZ2_AFFINITY_URL_WITH_FALLBACK) - 1:
                    msg = f"Failed to download model from all URLs. Last error: {e}"
                    raise RuntimeError(msg) from e
                continue


def get_cache_path() -> str:
    """Determine the cache path, prioritising the BOLTZ_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for boltz cache location.

    """
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            msg = f"BOLTZ_CACHE must be an absolute path, got: {env_cache}"
            raise ValueError(msg)
        return str(resolved_cache)

    return str(Path("~/.boltz").expanduser())


def check_inputs(data: Path) -> list[Path]:
    """Check the input data and output directory.

    Parameters
    ----------
    data : Path
        The input data.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        for d in data:
            if d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            if d.suffix.lower() not in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)
    else:
        data = [data]

    return data


def filter_inputs_structure(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Filter the manifest to only include missing predictions.

    Parameters
    ----------
    manifest : Manifest
        The manifest of the input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    # Check if existing predictions are found (only top-level prediction folders)
    pred_dir = outdir / "predictions"
    if pred_dir.exists():
        existing = {d.name for d in pred_dir.iterdir() if d.is_dir()}
    else:
        existing = set()

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        msg = (
            f"Found some existing predictions ({len(existing)}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = f"Found {len(existing)} existing predictions, will override."
        click.echo(msg)

    return manifest


def filter_inputs_affinity(
    manifest: Manifest,
    outdir: Path,
    override: bool = False,
) -> Manifest:
    """Check the input data and output directory for affinity.

    Parameters
    ----------
    manifest : Manifest
        The manifest.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    Manifest
        The manifest of the filtered input data.

    """
    click.echo("Checking input data for affinity.")

    # Get all affinity targets
    existing = {
        r.id
        for r in manifest.records
        if r.affinity
        and (outdir / "predictions" / r.id / f"affinity_{r.id}.json").exists()
    }

    # Remove them from the input data
    if existing and not override:
        manifest = Manifest([r for r in manifest.records if r.id not in existing])
        num_skipped = len(existing)
        msg = (
            f"Found some existing affinity predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "affinity predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing affinity predictions, will override."
        click.echo(msg)

    return manifest


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    msa_server_username : str, optional
        Username for basic authentication with MSA server.
    msa_server_password : str, optional
        Password for basic authentication with MSA server.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).

    """
    click.echo(f"Calling MSA server for target {target_id} with {len(data)} sequences")
    click.echo(f"MSA server URL: {msa_server_url}")
    click.echo(f"MSA pairing strategy: {msa_pairing_strategy}")
    
    # Construct auth headers if API key header/value is provided
    auth_headers = None
    if api_key_value:
        key = api_key_header if api_key_header else "X-API-Key"
        value = api_key_value
        auth_headers = {
            "Content-Type": "application/json",
            key: value
        }
        click.echo(f"Using API key authentication for MSA server (header: {key})")
    elif msa_server_username and msa_server_password:
        click.echo("Using basic authentication for MSA server")
    else:
        click.echo("No authentication provided for MSA server")
    
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            auth_headers=auth_headers,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


def process_input(  # noqa: C901, PLR0912, PLR0915, D103
    path: Path,
    ccd: dict,
    msa_dir: Path,
    mol_dir: Path,
    boltz2: bool,
    use_msa_server: bool,
    msa_server_url: str,
    msa_pairing_strategy: str,
    msa_server_username: Optional[str],
    msa_server_password: Optional[str],
    api_key_header: Optional[str],
    api_key_value: Optional[str],
    max_msa_seqs: int,
    processed_msa_dir: Path,
    processed_constraints_dir: Path,
    processed_templates_dir: Path,
    processed_mols_dir: Path,
    structure_dir: Path,
    records_dir: Path,
) -> None:
    try:
        # Parse data
        metadiffusion_config = None
        if path.suffix.lower() in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd, mol_dir, boltz2)
        elif path.suffix.lower() in (".yml", ".yaml"):
            target, metadiffusion_config = parse_yaml_with_metadiffusion(path, ccd, mol_dir, boltz2)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)  # noqa: TRY301
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)  # noqa: TRY301

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)  # noqa: TRY301

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                data=to_generate,
                target_id=target_id,
                msa_dir=msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
                msa_server_username=msa_server_username,
                msa_server_password=msa_server_password,
                api_key_header=api_key_header,
                api_key_value=api_key_value,
            )

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)  # noqa: TRY301

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                # Parse A3M
                if msa_path.suffix == ".a3m":
                    msa: MSA = parse_a3m(
                        msa_path,
                        taxonomy=None,
                        max_seqs=max_msa_seqs,
                    )
                elif msa_path.suffix == ".csv":
                    msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                else:
                    msg = f"MSA file {msa_path} not supported, only a3m or csv."
                    raise RuntimeError(msg)  # noqa: TRY301

                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Dump templates
        for template_id, template in target.templates.items():
            name = f"{target.record.id}_{template_id}.npz"
            template_path = processed_templates_dir / name
            template.dump(template_path)

        # Dump constraints
        constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
        target.residue_constraints.dump(constraints_path)

        # Dump extra molecules
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        with (processed_mols_dir / f"{target.record.id}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

        # Dump record
        record_path = records_dir / f"{target.record.id}.json"
        target.record.dump(record_path)

        # Dump metadiffusion config if present (in a separate subdirectory)
        if metadiffusion_config is not None:
            from boltz.data.parse.metadiffusion import metadiffusion_config_to_dict
            import json
            metadiff_dir = records_dir.parent / "metadiffusion"
            metadiff_dir.mkdir(parents=True, exist_ok=True)
            metadiff_path = metadiff_dir / f"{target.record.id}.json"
            with metadiff_path.open("w") as f:
                json.dump(metadiffusion_config_to_dict(metadiffusion_config), f, indent=2)

    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"Failed to process {path}. Skipping. Error: {e}.")  # noqa: T201


@rank_zero_only
def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    mol_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 8192,
    use_msa_server: bool = False,
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    boltz2: bool = False,
    preprocessing_threads: int = 1,
) -> Manifest:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 8192.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.
    msa_server_username : str, optional
        Username for basic authentication with MSA server, by default None.
    msa_server_password : str, optional
        Password for basic authentication with MSA server, by default None.
    api_key_header : str, optional
        Custom header key for API key authentication (default: X-API-Key).
    api_key_value : str, optional
        Custom header value for API key authentication (overrides --api_key if set).
    boltz2: bool, optional
        Whether to use Boltz2, by default False.
    preprocessing_threads: int, optional
        The number of threads to use for preprocessing, by default 1.

    Returns
    -------
    Manifest
        The manifest of the processed input data.

    """
    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_api_key = api_key_value is not None
    
    if has_basic_auth and has_api_key:
        raise ValueError(
            "Cannot use both basic authentication (--msa_server_username/--msa_server_password) "
            "and API key authentication (--api_key_header/--api_key_value). Please use only one authentication method."
        )

    # Check if records exist at output path
    records_dir = out_dir / "processed" / "records"
    if records_dir.exists():
        # Load existing records
        # Filter out old-style metadiffusion configs (for backwards compatibility)
        # A file is a metadiffusion config if:
        # 1. Its stem ends with "_metadiffusion" AND
        # 2. There exists a corresponding record file without the "_metadiffusion" suffix
        def is_metadiff_config(p: Path) -> bool:
            if not p.stem.endswith("_metadiffusion"):
                return False
            # Check if corresponding record exists
            base_stem = p.stem[:-len("_metadiffusion")]
            return (p.parent / f"{base_stem}.json").exists()

        existing = [
            Record.load(p) for p in records_dir.glob("*.json")
            if not is_metadiff_config(p)
        ]
        processed_ids = {record.id for record in existing}

        # Filter to missing only
        data = [d for d in data if d.stem not in processed_ids]

        # Nothing to do, update the manifest and return
        if data:
            click.echo(
                f"Found {len(existing)} existing processed inputs, skipping them."
            )
        else:
            click.echo("All inputs are already processed.")
            updated_manifest = Manifest(existing)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")

    # Create output directories
    msa_dir = out_dir / "msa"
    records_dir = out_dir / "processed" / "records"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    processed_templates_dir = out_dir / "processed" / "templates"
    processed_mols_dir = out_dir / "processed" / "mols"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    processed_constraints_dir.mkdir(parents=True, exist_ok=True)
    processed_templates_dir.mkdir(parents=True, exist_ok=True)
    processed_mols_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    if boltz2:
        ccd = load_canonicals(mol_dir)
    else:
        with ccd_path.open("rb") as file:
            ccd = pickle.load(file)  # noqa: S301

    # Create partial function
    process_input_partial = partial(
        process_input,
        ccd=ccd,
        msa_dir=msa_dir,
        mol_dir=mol_dir,
        boltz2=boltz2,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        max_msa_seqs=max_msa_seqs,
        processed_msa_dir=processed_msa_dir,
        processed_constraints_dir=processed_constraints_dir,
        processed_templates_dir=processed_templates_dir,
        processed_mols_dir=processed_mols_dir,
        structure_dir=structure_dir,
        records_dir=records_dir,
    )

    # Parse input data
    preprocessing_threads = min(preprocessing_threads, len(data))
    click.echo(f"Processing {len(data)} inputs with {preprocessing_threads} threads.")

    if preprocessing_threads > 1 and len(data) > 1:
        with Pool(preprocessing_threads) as pool:
            list(tqdm(pool.imap(process_input_partial, data), total=len(data)))
    else:
        for path in tqdm(data):
            process_input_partial(path)

    # Load all records and write manifest
    # Filter out old-style metadiffusion configs (for backwards compatibility)
    # A file is a metadiffusion config if:
    # 1. Its stem ends with "_metadiffusion" AND
    # 2. There exists a corresponding record file without the "_metadiffusion" suffix
    def is_metadiff_config(p: Path) -> bool:
        if not p.stem.endswith("_metadiffusion"):
            return False
        # Check if corresponding record exists
        base_stem = p.stem[:-len("_metadiffusion")]
        return (p.parent / f"{base_stem}.json").exists()

    record_files = [
        p for p in records_dir.glob("*.json")
        if not is_metadiff_config(p)
    ]
    records = [Record.load(p) for p in record_files]
    manifest = Manifest(records)
    manifest_path = out_dir / "processed" / "manifest.json"
    manifest.dump(manifest_path)


@click.group()
def cli() -> None:
    """Boltz."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help=(
        "The directory where to download the data and model. "
        "Default is ~/.boltz, or $BOLTZ_CACHE if set."
    ),
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--max_parallel_samples",
    type=int,
    help="The maximum number of samples to predict in parallel. Default is None.",
    default=5,
)
@click.option(
    "--step_scale",
    type=float,
    help=(
        "The step size is related to the temperature at "
        "which the diffusion process samples the distribution. "
        "The lower the higher the diversity among samples "
        "(recommended between 1 and 2). "
        "Default is 1.638 for Boltz-1 and 1.5 for Boltz-2. "
        "If not provided, the default step size will be used."
    ),
    default=None,
)
@click.option(
    "--diffusion_progress_bar",
    type=bool,
    is_flag=True,
    help="Show a progress bar during diffusion sampling steps. Default is False.",
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help=(
        "Pairing strategy to use. Used only if --use_msa_server is set. "
        "Options are 'greedy' and 'complete'"
    ),
    default="greedy",
)
@click.option(
    "--msa_server_username",
    type=str,
    help="MSA server username for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_USERNAME environment variable.",
    default=None,
)
@click.option(
    "--msa_server_password",
    type=str,
    help="MSA server password for basic auth. Used only if --use_msa_server is set. Can also be set via BOLTZ_MSA_PASSWORD environment variable.",
    default=None,
)
@click.option(
    "--api_key_header",
    type=str,
    help="Custom header key for API key authentication (default: X-API-Key).",
    default=None,
)
@click.option(
    "--api_key_value",
    type=str,
    help="Custom header value for API key authentication.",
    default=None,
)
@click.option(
    "--use_potentials",
    is_flag=True,
    help="Whether to use potentials for steering. Default is False.",
)
@click.option(
    "--model",
    default="boltz2",
    type=click.Choice(["boltz1", "boltz2"]),
    help="The model to use for prediction. Default is boltz2.",
)
@click.option(
    "--method",
    type=str,
    help="The method to use for prediction. Default is None.",
    default=None,
)
@click.option(
    "--preprocessing-threads",
    type=int,
    help="The number of threads to use for preprocessing. Default is 1.",
    default=multiprocessing.cpu_count(),
)
@click.option(
    "--affinity_mw_correction",
    is_flag=True,
    type=bool,
    help="Whether to add the Molecular Weight correction to the affinity value head.",
)
@click.option(
    "--sampling_steps_affinity",
    type=int,
    help="The number of sampling steps to use for affinity prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples_affinity",
    type=int,
    help="The number of diffusion samples to use for affinity prediction. Default is 5.",
    default=5,
)
@click.option(
    "--affinity_checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--max_msa_seqs",
    type=int,
    help="The maximum number of MSA sequences to use for prediction. Default is 8192.",
    default=8192,
)
@click.option(
    "--subsample_msa",
    is_flag=True,
    help="Whether to subsample the MSA. Default is True.",
)
@click.option(
    "--num_subsampled_msa",
    type=int,
    help="The number of MSA sequences to subsample. Default is 1024.",
    default=1024,
)
@click.option(
    "--no_kernels",
    is_flag=True,
    help="Whether to disable the kernels. Default False",
)
@click.option(
    "--write_embeddings",
    is_flag=True,
    help=" to dump the s and z embeddings into a npz file. Default is False.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output for YAML parsing and potential creation.",
)
def predict(  # noqa: C901, PLR0915, PLR0912
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    affinity_checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    sampling_steps_affinity: int = 200,
    diffusion_samples_affinity: int = 3,
    max_parallel_samples: Optional[int] = None,
    step_scale: Optional[float] = None,
    diffusion_progress_bar: bool = False,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    api_key_header: Optional[str] = None,
    api_key_value: Optional[str] = None,
    use_potentials: bool = False,
    model: Literal["boltz1", "boltz2"] = "boltz2",
    method: Optional[str] = None,
    affinity_mw_correction: Optional[bool] = False,
    preprocessing_threads: int = 1,
    max_msa_seqs: int = 8192,
    subsample_msa: bool = True,
    num_subsampled_msa: int = 1024,
    no_kernels: bool = False,
    write_embeddings: bool = False,
    debug: bool = False,
) -> None:
    """Run predictions with Boltz."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Supress some lightning warnings
    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set rdkit pickle logic
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        # Disable kernel tuning by default,
        # but do not modify envvar if already set by caller
        os.environ[key] = os.environ.get(key, "1")

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Get MSA server credentials from environment variables if not provided
    if use_msa_server:
        if msa_server_username is None:
            msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
        if msa_server_password is None:
            msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
        if api_key_value is None:
            api_key_value = os.environ.get("MSA_API_KEY_VALUE")
        
        click.echo(f"MSA server enabled: {msa_server_url}")
        if api_key_value:
            click.echo("MSA server authentication: using API key header")
        elif msa_server_username and msa_server_password:
            click.echo("MSA server authentication: using basic auth")
        else:
            click.echo("MSA server authentication: no credentials provided")

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    if model == "boltz1":
        download_boltz1(cache)
    elif model == "boltz2":
        download_boltz2(cache)
    else:
        msg = f"Model {model} not supported. Supported: boltz1, boltz2."
        raise ValueError(f"Model {model} not supported.")

    # Validate inputs
    data = check_inputs(data)

    # Check method
    if method is not None:
        if model == "boltz1":
            msg = "Method conditioning is not supported for Boltz-1."
            raise ValueError(msg)
        if method.lower() not in const.method_types_ids:
            method_names = list(const.method_types_ids.keys())
            msg = f"Method {method} not supported. Supported: {method_names}"
            raise ValueError(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        api_key_header=api_key_header,
        api_key_value=api_key_value,
        boltz2=model == "boltz2",
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=max_msa_seqs,
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Filter out existing predictions
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=out_dir,
        override=override,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        start_method = "fork" if platform.system() != "win32" and platform.system() != "Windows" else "spawn"
        strategy = DDPStrategy(start_method=start_method)
        if len(filtered_manifest.records) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions, taking the minimum."
            )
            click.echo(msg)
            if isinstance(devices, list):
                devices = devices[: max(1, len(filtered_manifest.records))]
            else:
                devices = max(1, min(len(filtered_manifest.records), devices))

    # Set up model parameters
    if model == "boltz2":
        diffusion_params = Boltz2DiffusionParams()
        step_scale = 1.5 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgsV2()
    else:
        diffusion_params = BoltzDiffusionParams()
        step_scale = 1.638 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgs()

    msa_args = MSAModuleArgs(
        subsample_msa=subsample_msa,
        num_subsampled_msa=num_subsampled_msa,
        use_paired_feature=model == "boltz2",
    )

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        boltz2=model == "boltz2",
        write_embeddings=write_embeddings,
    )

    # Set up trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32 if model == "boltz1" else "bf16-mixed",
    )

    if filtered_manifest.records:
        msg = f"Running structure prediction for {len(filtered_manifest.records)} input"
        msg += "s." if len(filtered_manifest.records) > 1 else "."
        click.echo(msg)

        # Create data module
        if model == "boltz2":
            data_module = Boltz2InferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                mol_dir=mol_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
                template_dir=processed.template_dir,
                extra_mols_dir=processed.extra_mols_dir,
                override_method=method,
            )
        else:
            data_module = BoltzInferenceDataModule(
                manifest=processed.manifest,
                target_dir=processed.targets_dir,
                msa_dir=processed.msa_dir,
                num_workers=num_workers,
                constraints_dir=processed.constraints_dir,
            )

        # Load model
        if checkpoint is None:
            if model == "boltz2":
                checkpoint = cache / "boltz2_conf.ckpt"
            else:
                checkpoint = cache / "boltz1_conf.ckpt"

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
            "diffusion_progress_bar": diffusion_progress_bar,
        }

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = use_potentials
        steering_args.physical_guidance_update = use_potentials
        steering_args.debug = debug

        # Load metadiffusion configs from YAML
        # All metadiffusion settings are configured via YAML only
        # This allows YAML-based steering without CLI flags
        # When --override is set, re-parse from original YAML to pick up changes
        metadiff_dir = out_dir / "processed" / "metadiffusion"
        records_dir = out_dir / "processed" / "records"

        # If override is set, re-parse metadiffusion from original YAML
        if override and data:
            from boltz.data.parse.yaml import parse_metadiffusion_from_yaml
            from boltz.data.parse.metadiffusion import metadiffusion_config_to_dict
            import json

            per_record_metadiff = {}
            base_path = Path(data[0]).parent if data else Path.cwd()
            metadiff_dir.mkdir(parents=True, exist_ok=True)

            for input_path in data:
                input_path = Path(input_path)
                if input_path.suffix.lower() not in (".yml", ".yaml"):
                    continue
                try:
                    metadiffusion_config = parse_metadiffusion_from_yaml(input_path)
                    if metadiffusion_config is not None:
                        if debug:
                            from boltz.data.parse.metadiffusion import debug_print_config
                            debug_print_config(metadiffusion_config, enabled=True)
                        metadiff_dict = metadiffusion_config_to_dict(metadiffusion_config)
                        # Record ID is the file stem (e.g. rg1.yaml -> rg1)
                        record_id = input_path.stem
                        per_record_metadiff[record_id] = metadiff_dict
                        # Update cached JSON so future runs without --override use new config
                        cache_path = metadiff_dir / f"{record_id}.json"
                        with cache_path.open("w") as f:
                            json.dump(metadiff_dict, f, indent=2)
                except Exception as e:
                    click.echo(f"Warning: Could not re-parse metadiffusion from {input_path}: {e}")

            if per_record_metadiff:
                # Apply first record's config as base (sets boolean flags for sampling)
                first_dict = next(iter(per_record_metadiff.values()))
                apply_metadiffusion_to_steering_args(
                    steering_args, first_dict, base_path=base_path,
                )

                # Check if configs differ between records
                configs_list = list(per_record_metadiff.values())
                all_identical = len(configs_list) <= 1 or all(
                    c == configs_list[0] for c in configs_list[1:]
                )

                if not all_identical:
                    per_record_steering = {}
                    for record_id, md_dict in per_record_metadiff.items():
                        rec_steering = BoltzSteeringParams()
                        rec_steering.fk_steering = use_potentials
                        rec_steering.physical_guidance_update = use_potentials
                        rec_steering.debug = debug
                        apply_metadiffusion_to_steering_args(
                            rec_steering, md_dict, base_path=base_path,
                        )
                        per_record_steering[record_id] = asdict(rec_steering)
                    steering_args.per_record_steering = per_record_steering
        else:
            # Load per-record metadiffusion configs from cached JSON
            per_record_metadiff = {}
            base_path = Path(data[0]).parent if data else Path.cwd()

            for record in manifest.records:
                # Try new location first
                metadiff_path = metadiff_dir / f"{record.id}.json"
                # Fall back to old location for backwards compatibility
                if not metadiff_path.exists():
                    metadiff_path = records_dir / f"{record.id}_metadiffusion.json"
                if metadiff_path.exists():
                    import json
                    with metadiff_path.open("r") as f:
                        metadiff_dict = json.load(f)
                    # Print debug info if enabled
                    if debug:
                        print(f"[DEBUG] Loaded metadiffusion config for {record.id}: {metadiff_path}")
                        print(f"[DEBUG] Config keys: {list(metadiff_dict.keys())}")
                        if metadiff_dict.get("steering"):
                            print(f"[DEBUG] Steering configs: {len(metadiff_dict.get('steering', []))}")
                        if metadiff_dict.get("explore"):
                            print(f"[DEBUG] Explore configs: {len(metadiff_dict.get('explore', []))}")
                    per_record_metadiff[record.id] = metadiff_dict

            if per_record_metadiff:
                # Apply first record's config as base (sets boolean flags for sampling)
                first_dict = next(iter(per_record_metadiff.values()))
                apply_metadiffusion_to_steering_args(
                    steering_args, first_dict, base_path=base_path,
                )

                # Check if configs differ between records
                configs_list = list(per_record_metadiff.values())
                all_identical = len(configs_list) <= 1 or all(
                    c == configs_list[0] for c in configs_list[1:]
                )

                if not all_identical:
                    # Build per-record steering dicts for records with different configs
                    per_record_steering = {}
                    for record_id, md_dict in per_record_metadiff.items():
                        rec_steering = BoltzSteeringParams()
                        rec_steering.fk_steering = use_potentials
                        rec_steering.physical_guidance_update = use_potentials
                        rec_steering.debug = debug
                        apply_metadiffusion_to_steering_args(
                            rec_steering, md_dict, base_path=base_path,
                        )
                        per_record_steering[record_id] = asdict(rec_steering)
                    steering_args.per_record_steering = per_record_steering

        # Load experimental P(r) data for SAXS configs and steering with target_from_saxs
        # Collect from base steering_args AND all per-record steering dicts
        all_saxs_configs = list(steering_args.saxs_configs or [])
        all_steering_configs = list(steering_args.steering_configs or [])
        if steering_args.per_record_steering:
            for rec_dict in steering_args.per_record_steering.values():
                all_saxs_configs.extend(rec_dict.get("saxs_configs") or [])
                all_steering_configs.extend(rec_dict.get("steering_configs") or [])

        # Collect all unique P(r) files to load with their settings
        # First config's settings win for each file
        pr_files_to_load = {}  # file -> {units, bins, bins_range}
        for config in all_saxs_configs:
            pr_file = config.get("pr_file")
            if pr_file and pr_file not in pr_files_to_load:
                pr_files_to_load[pr_file] = {
                    "units": config.get("units", "auto"),
                    "bins": config.get("bins"),
                    "bins_range": config.get("bins_range"),
                }

        # Also check for target_from_saxs in steering_configs (for Rg steering from SAXS)
        for config in all_steering_configs:
            saxs_file = config.get("target_from_saxs")
            if saxs_file and saxs_file not in pr_files_to_load:
                # Steering configs don't have bins settings, use defaults
                pr_files_to_load[saxs_file] = {
                    "units": "auto",
                    "bins": None,
                    "bins_range": None,
                }

        # Load all unique P(r) files
        if pr_files_to_load:
            from boltz.data.saxs import load_experimental_pr
            steering_args.saxs_pr_data_cache = {}

            for pr_file, settings in pr_files_to_load.items():
                pr_path = Path(pr_file)
                if pr_path.exists():
                    r_grid, pr_exp, pr_errors = load_experimental_pr(
                        pr_path,
                        normalize=True,
                        device=None,
                        units=settings["units"],
                        bins=settings["bins"],
                        bins_range=tuple(settings["bins_range"]) if settings["bins_range"] else None,
                    )
                    steering_args.saxs_pr_data_cache[pr_file] = {
                        'r_grid': r_grid,
                        'pr_exp': pr_exp,
                        'pr_errors': pr_errors
                    }
                    bins_info = f", resampled to {settings['bins']} bins" if settings["bins"] else ""
                    print(f"Loaded SAXS P(r) from: {pr_file}{bins_info}")

        # Propagate shared resources (e.g. SAXS data cache) to per-record steering dicts
        if steering_args.per_record_steering:
            for rec_dict in steering_args.per_record_steering.values():
                rec_dict["saxs_pr_data_cache"] = steering_args.saxs_pr_data_cache

        model_cls = Boltz2 if model == "boltz2" else Boltz1
        model_module = model_cls.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not no_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model_module.eval()

        # Compute structure predictions
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )

    # Check if affinity predictions are needed
    if any(r.affinity for r in manifest.records):
        # Print header
        click.echo("\nPredicting property: affinity\n")

        # Validate inputs
        manifest_filtered = filter_inputs_affinity(
            manifest=manifest,
            outdir=out_dir,
            override=override,
        )
        if not manifest_filtered.records:
            click.echo("Found existing affinity predictions for all inputs, skipping.")
            return

        msg = f"Running affinity prediction for {len(manifest_filtered.records)} input"
        msg += "s." if len(manifest_filtered.records) > 1 else "."
        click.echo(msg)

        pred_writer = BoltzAffinityWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
        )

        data_module = Boltz2InferenceDataModule(
            manifest=manifest_filtered,
            target_dir=out_dir / "predictions",
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method="other",
            affinity=True,
        )

        predict_affinity_args = {
            "recycling_steps": 5,
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        # Load affinity model
        if affinity_checkpoint is None:
            affinity_checkpoint = cache / "boltz2_aff.ckpt"

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        steering_args.contact_guidance_update = False
        
        model_module = Boltz2.load_from_checkpoint(
            affinity_checkpoint,
            strict=True,
            predict_args=predict_affinity_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
            affinity_mw_correction=affinity_mw_correction,
        )
        model_module.eval()

        trainer.callbacks[0] = pred_writer
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )


if __name__ == "__main__":
    cli()
