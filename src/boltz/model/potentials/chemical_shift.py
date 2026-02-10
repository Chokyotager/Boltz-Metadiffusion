"""
Differentiable NMR chemical shift potential using the CheShift algorithm.

This module implements chemical shift prediction for CA and CB atoms based
on backbone and sidechain torsion angles (phi, psi, chi1). For residues with
chi2 angles, predictions are averaged over all chi2 rotamers (excluding
invalid grid values).

The CheShift algorithm uses pre-computed lookup tables from quantum chemistry
calculations. This implementation uses differentiable trilinear interpolation
for gradient computation during structure steering.

IMPORTANT: Only CA and CB chemical shifts are supported. For other nuclei
(HA, HN, N, C), use specialized predictors or other methods.

Reference:
    Villegas et al. (2007) Proteins, 68, 789-799
"""

from typing import Dict, Optional, Tuple, Any, Union
import torch
from pathlib import Path

from boltz.model.potentials.potentials import Potential
from boltz.data.parse.cheshift_db import DifferentiableCheShift, compute_dihedral_torch
from boltz.data import const


# Nucleus-specific sigma values for chi-squared scoring (ppm)
# Sigma represents expected prediction uncertainty for each nucleus.
CHESHIFT_SIGMA = {
    'CA': 1.5,
    'CB': 1.5,
}

# Default DSS reference offsets (ppm) to convert CheShift DFT predictions to experimental frame
# CheShift database uses internal DFT reference; experimental data uses DSS.
# These can be overridden via YAML parameters ca_dss_offset and cb_dss_offset
DEFAULT_CA_DSS_OFFSET = 0.0
DEFAULT_CB_DSS_OFFSET = 0.0

# Supported nuclei
SUPPORTED_NUCLEI = {'CA', 'CB'}


def load_shift_file(filepath: str) -> Dict[int, float]:
    """Load experimental chemical shifts from file.

    Expected format (PLUMED-compatible):
        # Optional header comments
        1 52.3
        2 58.1
        ...

    Where column 1 is residue number (1-indexed), column 2 is shift value (ppm).
    Lines starting with # are ignored.

    Args:
        filepath: Path to shift file

    Returns:
        Dictionary mapping residue number to shift value
    """
    residue_shifts = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    resid = int(parts[0].lstrip('#'))
                    shift = float(parts[1])
                    if shift != 0.0:  # PLUMED uses 0.0 for missing values
                        residue_shifts[resid] = shift
                except ValueError:
                    continue

    return residue_shifts


class ChemicalShiftPotential(Potential):
    """Differentiable NMR chemical shift potential using CheShift algorithm.

    Computes predicted chemical shifts for CA and CB nuclei and penalizes
    deviation from experimental values. Uses CheShift lookup tables with
    chi2 averaging for residues with rotamers.

    Only CA and CB are supported. Attempts to use other nuclei will raise
    NotImplementedError.
    """

    def __init__(self, parameters: Dict[str, Any]):
        """Initialize chemical shift potential.

        Args:
            parameters: Dictionary containing:
                - exp_shifts: Dict[str, Dict[int, float]] - experimental shifts by nucleus
                - loss_type: 'chi' (chi-squared), 'mse', or 'corr' (correlation)
                - k: Force constant (strength)
                - guidance_interval, warmup, cutoff: Scheduling parameters

        Raises:
            NotImplementedError: If exp_shifts contains nuclei other than CA or CB
        """
        super().__init__(parameters)

        # Store experimental shifts and validate nuclei
        self._exp_shifts = parameters.get('exp_shifts', {})
        self._validate_nuclei()

        self._loss_type = parameters.get('loss_type', 'chi')

        # Reference offset handling
        self._auto_offset = parameters.get('auto_offset', True)
        self._ca_dss_offset = parameters.get('ca_dss_offset', DEFAULT_CA_DSS_OFFSET)
        self._cb_dss_offset = parameters.get('cb_dss_offset', DEFAULT_CB_DSS_OFFSET)

        # CheShift predictor (initialized lazily)
        self._cheshift_predictor: Optional[DifferentiableCheShift] = None

        # Cache for atom mapping
        self._atom_mapping_cache: Optional[Dict] = None

    def _validate_nuclei(self) -> None:
        """Validate that only CA and CB nuclei are requested.

        Raises:
            NotImplementedError: If unsupported nuclei are requested
        """
        unsupported = set(self._exp_shifts.keys()) - SUPPORTED_NUCLEI
        if unsupported:
            raise NotImplementedError(
                f"Only CA and CB chemical shifts are supported. "
                f"Unsupported nuclei requested: {unsupported}. "
                f"Please use only CA and CB shifts."
            )

    def _build_atom_mapping(self, feats: Dict[str, Any]) -> None:
        """Build mapping from (residue_idx, atom_name) to global atom index.

        Uses the model's atom_to_token and ref_atom_name_chars to find backbone atoms.
        """
        if self._atom_mapping_cache is not None:
            return

        self._atom_mapping_cache = {}

        # Get required tensors
        res_type = feats.get('res_type')
        if res_type is None:
            raise ValueError("res_type not found in feats")

        atom_to_token = feats.get('atom_to_token')  # [batch, n_atoms, n_tokens]
        ref_atom_name_chars = feats.get('ref_atom_name_chars')  # [batch, n_atoms, 4, 64]

        # Convert one-hot res_type to token IDs
        if hasattr(res_type, 'argmax'):
            token_ids = res_type.argmax(dim=-1)
        else:
            import numpy as np
            token_ids = np.argmax(res_type, axis=-1)

        # Handle tensor vs list
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        elif hasattr(token_ids, 'cpu'):
            token_ids = token_ids.cpu().numpy().tolist()

        # Squeeze batch dimension
        while isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        n_residues = len(token_ids)
        self._atom_mapping_cache['n_residues'] = n_residues

        # Store residue types
        for res_idx, tid in enumerate(token_ids):
            if 2 <= tid <= 21:  # Protein tokens
                aa_type = const.tokens[tid]
            else:
                aa_type = 'UNK'
            self._atom_mapping_cache[('res_type', res_idx)] = aa_type

        # If we have atom_to_token and ref_atom_name_chars, use them to find atom indices
        if atom_to_token is not None and ref_atom_name_chars is not None:
            # Work with batch index 0
            atom_to_token_0 = atom_to_token[0]  # [n_atoms, n_tokens]
            ref_atom_name_chars_0 = ref_atom_name_chars[0]  # [n_atoms, 4, 64]
            n_atoms = atom_to_token_0.shape[0]

            self._atom_mapping_cache['n_atoms'] = n_atoms

            # Decode atom names and build mapping
            for atom_idx in range(n_atoms):
                # Find which token this atom belongs to
                token_idx = atom_to_token_0[atom_idx].argmax().item()

                # Decode atom name from character encoding
                atom_name_chars = ref_atom_name_chars_0[atom_idx]  # [4, 64]
                atom_name = self._decode_atom_name(atom_name_chars)

                # Store mapping for backbone and sidechain atoms needed for dihedrals
                # Backbone: N, CA, C, O, CB
                # Sidechain for chi1: CG, CG1, OG, OG1, SG
                # Sidechain for chi2: CD, CD1, OD1, ND1, SD
                needed_atoms = {
                    'N', 'CA', 'C', 'O', 'CB',  # backbone
                    'CG', 'CG1', 'OG', 'OG1', 'SG',  # chi1
                    'CD', 'CD1', 'OD1', 'ND1', 'SD',  # chi2
                }
                if atom_name in needed_atoms:
                    self._atom_mapping_cache[(token_idx, atom_name)] = atom_idx

    def _decode_atom_name(self, atom_name_chars: torch.Tensor) -> str:
        """Decode atom name from character one-hot encoding.

        Args:
            atom_name_chars: [4, 64] one-hot encoded characters

        Returns:
            Decoded atom name string
        """
        # The encoding is ASCII - 32 (so printable chars start at 0):
        # ' ' (space) = 32 -> 0, 'A' = 65 -> 33, 'N' = 78 -> 46, etc.
        # To decode: char_idx + 32 gives ASCII code
        chars = []
        for i in range(4):
            char_idx = atom_name_chars[i].argmax().item()
            if char_idx > 0:  # 0 is padding (space character)
                ascii_code = char_idx + 32
                if 32 <= ascii_code <= 126:  # Printable ASCII range
                    chars.append(chr(ascii_code))
        return ''.join(chars).strip()

    def _get_atom_index(self, res_idx: int, atom_name: str) -> Optional[int]:
        """Get global atom index for a specific residue and atom name."""
        return self._atom_mapping_cache.get((res_idx, atom_name))

    def _get_residue_type(self, res_idx: int) -> str:
        """Get amino acid type for a residue."""
        return self._atom_mapping_cache.get(('res_type', res_idx), 'UNK')

    def _compute_shifts(
        self,
        coords: torch.Tensor,
        feats: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Compute predicted chemical shifts for CA and CB.

        Args:
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary

        Returns:
            shifts: Dict[nucleus, tensor[multiplicity, N_residues]]
        """
        # Build atom mapping on first call
        if self._atom_mapping_cache is None:
            self._build_atom_mapping(feats)

        n_residues = self._atom_mapping_cache['n_residues']
        mult = coords.shape[0]
        device = coords.device
        dtype = coords.dtype

        # Initialize CheShift predictor
        if self._cheshift_predictor is None:
            self._cheshift_predictor = DifferentiableCheShift()

        shifts = {}

        for nucleus in self._exp_shifts.keys():
            exp_shifts = self._exp_shifts.get(nucleus, {})

            # Collect shifts for all residues - use list to preserve gradients
            shift_list = []
            for res_idx in range(n_residues):
                # Only compute for residues with experimental data
                if (res_idx + 1) not in exp_shifts:  # PLUMED uses 1-indexed
                    # Use zero tensor that will be ignored in loss computation
                    shift = torch.zeros(mult, device=device, dtype=dtype)
                else:
                    shift = self._compute_residue_shift(
                        coords, nucleus, res_idx, device, dtype
                    )
                shift_list.append(shift)

            # Stack to [mult, n_residues] - preserves gradient graph
            shift_tensor = torch.stack(shift_list, dim=1)
            shifts[nucleus] = shift_tensor

        return shifts

    def _compute_residue_shift(
        self,
        coords: torch.Tensor,
        nucleus: str,
        res_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute chemical shift for one residue using CheShift.

        Args:
            coords: [multiplicity, N_atoms, 3]
            nucleus: 'CA' or 'CB'
            res_idx: Residue index (0-based)
            device, dtype: Tensor device and dtype

        Returns:
            shift: [multiplicity] predicted shift values
        """
        mult = coords.shape[0]
        n_residues = self._atom_mapping_cache['n_residues']
        aa_type = self._get_residue_type(res_idx)

        # Compute backbone dihedrals (phi, psi)
        phi = None
        psi = None

        # Compute phi: C(i-1) - N(i) - CA(i) - C(i)
        if res_idx > 0:
            c_prev = self._get_atom_index(res_idx - 1, 'C')
            n_curr = self._get_atom_index(res_idx, 'N')
            ca_curr = self._get_atom_index(res_idx, 'CA')
            c_curr = self._get_atom_index(res_idx, 'C')

            if all(idx is not None for idx in [c_prev, n_curr, ca_curr, c_curr]):
                phi = compute_dihedral_torch(
                    coords[:, c_prev, :],
                    coords[:, n_curr, :],
                    coords[:, ca_curr, :],
                    coords[:, c_curr, :],
                )

        # Compute psi: N(i) - CA(i) - C(i) - N(i+1)
        if res_idx < n_residues - 1:
            n_curr = self._get_atom_index(res_idx, 'N')
            ca_curr = self._get_atom_index(res_idx, 'CA')
            c_curr = self._get_atom_index(res_idx, 'C')
            n_next = self._get_atom_index(res_idx + 1, 'N')

            if all(idx is not None for idx in [n_curr, ca_curr, c_curr, n_next]):
                psi = compute_dihedral_torch(
                    coords[:, n_curr, :],
                    coords[:, ca_curr, :],
                    coords[:, c_curr, :],
                    coords[:, n_next, :],
                )

        # If we can't compute dihedrals, return zero (will not contribute)
        if phi is None or psi is None:
            return torch.zeros(mult, device=device, dtype=dtype)

        # Compute chi1 for residues with rotamers
        chi1 = None
        chi2 = None
        cg_atom = None
        if aa_type not in ('ALA', 'GLY'):
            n_curr = self._get_atom_index(res_idx, 'N')
            ca_curr = self._get_atom_index(res_idx, 'CA')
            cb_curr = self._get_atom_index(res_idx, 'CB')

            # Get CG or equivalent atom for chi1
            # Mapping: residue -> CG atom name
            cg_names = {
                'ARG': 'CG', 'ASN': 'CG', 'ASP': 'CG', 'GLN': 'CG', 'GLU': 'CG',
                'HIS': 'CG', 'ILE': 'CG1', 'LEU': 'CG', 'LYS': 'CG', 'MET': 'CG',
                'PHE': 'CG', 'PRO': 'CG', 'TRP': 'CG', 'TYR': 'CG', 'VAL': 'CG1',
                'SER': 'OG', 'THR': 'OG1', 'CYS': 'SG'
            }
            cg_name = cg_names.get(aa_type)
            if cg_name:
                cg_atom = self._get_atom_index(res_idx, cg_name)

            if all(idx is not None for idx in [n_curr, ca_curr, cb_curr, cg_atom]):
                chi1 = compute_dihedral_torch(
                    coords[:, n_curr, :],
                    coords[:, ca_curr, :],
                    coords[:, cb_curr, :],
                    coords[:, cg_atom, :],
                )

        # Compute chi2 for residues that have it (CA-CB-CG-CD)
        # Residues with chi2: ARG, ASN, ASP, GLN, GLU, HIS, ILE, LEU, LYS, MET, PHE, TRP, TYR
        chi2_residues = {'ARG', 'ASN', 'ASP', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'TRP', 'TYR'}
        if aa_type in chi2_residues and cg_atom is not None:
            ca_curr = self._get_atom_index(res_idx, 'CA')
            cb_curr = self._get_atom_index(res_idx, 'CB')
            # CD atom names vary by residue
            cd_names = {
                'ARG': 'CD', 'ASN': 'OD1', 'ASP': 'OD1', 'GLN': 'CD', 'GLU': 'CD',
                'HIS': 'ND1', 'ILE': 'CD1', 'LEU': 'CD1', 'LYS': 'CD', 'MET': 'SD',
                'PHE': 'CD1', 'TRP': 'CD1', 'TYR': 'CD1'
            }
            cd_name = cd_names.get(aa_type)
            if cd_name:
                cd_atom = self._get_atom_index(res_idx, cd_name)
                if all(idx is not None for idx in [ca_curr, cb_curr, cg_atom, cd_atom]):
                    chi2 = compute_dihedral_torch(
                        coords[:, ca_curr, :],
                        coords[:, cb_curr, :],
                        coords[:, cg_atom, :],
                        coords[:, cd_atom, :],
                    )

        # Use CheShift predictor with actual chi2 (or None for residues without chi2)
        ca_shift, cb_shift = self._cheshift_predictor.predict_shifts_torch(
            aa_type, phi, psi, chi1, chi2=chi2
        )

        # Return raw DFT values - offset is applied in compute_function
        # This allows auto_offset to estimate offset from all data points
        if nucleus == 'CA':
            return ca_shift
        else:  # CB
            return cb_shift

    def compute_variable(
        self,
        coords: torch.Tensor,
        index: Optional[torch.Tensor],
        ref_coords: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        compute_gradient: bool = False,
        feats: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict, Dict]]:
        """Compute predicted chemical shifts for CA and CB."""
        if feats is None:
            raise ValueError("feats required for chemical shift computation")

        # Build atom mapping on first call
        if self._atom_mapping_cache is None:
            self._build_atom_mapping(feats)

        if compute_gradient:
            coords = coords.detach().requires_grad_(True)

        shifts = self._compute_shifts(coords, feats)

        if not compute_gradient:
            return shifts

        # Compute gradients via autograd
        n_residues = self._atom_mapping_cache['n_residues']
        gradients = {}
        for nucleus, shift_tensor in shifts.items():
            shift_sum = shift_tensor.sum()
            grad = torch.autograd.grad(
                shift_sum, coords, create_graph=False, retain_graph=True
            )[0]
            gradients[nucleus] = grad.unsqueeze(1).expand(-1, n_residues, -1, -1)

        return shifts, gradients

    def compute_function(
        self,
        shifts_calc: Dict[str, torch.Tensor],
        k: float,
        negation_mask: Optional[torch.Tensor] = None,
        compute_derivative: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Compute loss from predicted vs experimental shifts.

        Args:
            shifts_calc: Dict mapping nucleus to predicted shifts [mult, N_residues]
            k: Force constant (strength)
            negation_mask: Not used
            compute_derivative: Whether to return derivative w.r.t. shifts

        Returns:
            If compute_derivative=False: total loss [scalar]
            If compute_derivative=True: (loss, dL/dshift)
        """
        first_tensor = next(iter(shifts_calc.values()))
        device = first_tensor.device
        dtype = first_tensor.dtype

        # Start with None so first loss assignment preserves gradient graph
        total_loss = None
        dL_dshift = {} if compute_derivative else None

        for nucleus, shift_calc in shifts_calc.items():
            exp_shifts = self._exp_shifts.get(nucleus, {})
            if not exp_shifts:
                continue

            mult, n_residues = shift_calc.shape
            device = shift_calc.device
            dtype = shift_calc.dtype

            # Build experimental shift tensor
            shift_exp = torch.zeros(n_residues, device=device, dtype=dtype)
            mask = torch.zeros(n_residues, device=device, dtype=torch.bool)
            for res_id, shift_val in exp_shifts.items():
                if 1 <= res_id <= n_residues:  # PLUMED uses 1-indexed
                    shift_exp[res_id - 1] = shift_val
                    mask[res_id - 1] = True

            if not mask.any():
                continue

            # Get masked values
            calc_masked = shift_calc[:, mask]  # [mult, n_masked]
            exp_masked = shift_exp[mask].unsqueeze(0)  # [1, n_masked]

            # Apply reference offset
            if self._auto_offset:
                # Auto-offset: estimate from data as mean(exp) - mean(calc)
                # This is the principled approach - offset derived from the data itself
                offset = exp_masked.mean() - calc_masked.mean()
            else:
                # Fixed offset from YAML parameters
                if nucleus == 'CA':
                    offset = self._ca_dss_offset
                else:
                    offset = self._cb_dss_offset

            # Apply offset to predictions before computing difference
            calc_adjusted = calc_masked + offset
            diff = calc_adjusted - exp_masked

            if self._loss_type == 'chi':
                # Chi-squared with nucleus-specific sigma
                sigma = CHESHIFT_SIGMA.get(nucleus, 1.0)
                loss = (diff / sigma) ** 2
                if compute_derivative:
                    dL = 2 * diff / (sigma ** 2)
                nucleus_loss = k * loss.sum()
            elif self._loss_type == 'corr':
                # Negative correlation loss (minimize to maximize correlation)
                # L = -corr(calc, exp) = -cov(calc,exp) / (std(calc) * std(exp))
                n = calc_masked.shape[1]
                calc_mean = calc_masked.mean(dim=1, keepdim=True)
                exp_mean = exp_masked.mean(dim=1, keepdim=True)
                calc_centered = calc_masked - calc_mean
                exp_centered = exp_masked - exp_mean

                calc_std = calc_centered.std(dim=1, keepdim=True) + 1e-8
                exp_std = exp_centered.std(dim=1, keepdim=True) + 1e-8

                cov = (calc_centered * exp_centered).mean(dim=1, keepdim=True)
                corr = cov / (calc_std * exp_std)

                # Loss is negative correlation (so minimizing maximizes correlation)
                loss = -corr
                nucleus_loss = k * loss.sum()

                if compute_derivative:
                    # d(-corr)/d(calc_i) = -d(corr)/d(calc_i)
                    # corr = cov / (std_calc * std_exp)
                    # d(corr)/d(calc_i) = (1/n) * (exp_centered / (std_calc * std_exp)
                    #                     - corr * calc_centered / (n * std_calc^2))
                    dL = -(exp_centered / (calc_std * exp_std)
                           - corr * calc_centered / (calc_std ** 2)) / n
                    dL = k * dL
            elif self._loss_type == 'ccc':
                # Concordance Correlation Coefficient loss
                # CCC = 2 * cov(x,y) / (var(x) + var(y) + (mean(x) - mean(y))^2)
                # Penalizes both poor correlation AND offset/scale differences
                n = calc_masked.shape[1]
                calc_mean = calc_masked.mean(dim=1, keepdim=True)
                exp_mean = exp_masked.mean(dim=1, keepdim=True)
                calc_centered = calc_masked - calc_mean
                exp_centered = exp_masked - exp_mean

                calc_var = (calc_centered ** 2).mean(dim=1, keepdim=True) + 1e-8
                exp_var = (exp_centered ** 2).mean(dim=1, keepdim=True) + 1e-8
                cov = (calc_centered * exp_centered).mean(dim=1, keepdim=True)

                mean_diff_sq = (calc_mean - exp_mean) ** 2
                denom = calc_var + exp_var + mean_diff_sq + 1e-8
                ccc = 2 * cov / denom

                # Loss is negative CCC (so minimizing maximizes CCC)
                loss = -ccc
                nucleus_loss = k * loss.sum()

                if compute_derivative:
                    # d(-CCC)/d(calc_i) = -(2/n) * ((y_i - μy) - CCC * (x_i - μy)) / denom
                    dL = -(2.0 / n) * (exp_centered - ccc * (calc_masked - exp_mean)) / denom
                    dL = k * dL
            elif self._loss_type == 'chi_ccc':
                # Combined Chi-squared + CCC loss
                # Best of both: per-residue fitting (chi) + global correlation (ccc)
                n = calc_masked.shape[1]
                sigma = CHESHIFT_SIGMA.get(nucleus, 1.0)

                # Chi-squared component (normalized by n to match CCC scale)
                chi_loss = ((diff / sigma) ** 2).mean(dim=1, keepdim=True)

                # CCC component
                calc_mean = calc_masked.mean(dim=1, keepdim=True)
                exp_mean = exp_masked.mean(dim=1, keepdim=True)
                calc_centered = calc_masked - calc_mean
                exp_centered = exp_masked - exp_mean

                calc_var = (calc_centered ** 2).mean(dim=1, keepdim=True) + 1e-8
                exp_var = (exp_centered ** 2).mean(dim=1, keepdim=True) + 1e-8
                cov = (calc_centered * exp_centered).mean(dim=1, keepdim=True)

                mean_diff_sq = (calc_mean - exp_mean) ** 2
                denom = calc_var + exp_var + mean_diff_sq + 1e-8
                ccc = 2 * cov / denom

                # Combined loss: chi (scaled to ~1) + (1 - ccc) which is in [0, 2]
                # Both components are roughly comparable in scale
                ccc_loss = 1 - ccc  # 0 when perfect, 2 when anti-correlated
                combined_loss = chi_loss + ccc_loss
                nucleus_loss = k * combined_loss.sum()

                if compute_derivative:
                    # Gradient is sum of chi gradient + ccc gradient
                    # d(chi)/d(calc_i) = 2 * diff_i / (n * sigma^2)
                    dL_chi = 2 * diff / (n * sigma ** 2)
                    # d(1-ccc)/d(calc_i) = -d(ccc)/d(calc_i)
                    dL_ccc = -(2.0 / n) * (exp_centered - ccc * (calc_masked - exp_mean)) / denom
                    dL = k * (dL_chi + dL_ccc)
            else:  # mse
                loss = diff ** 2
                if compute_derivative:
                    dL = 2 * diff
                nucleus_loss = k * loss.sum()
            if total_loss is None:
                total_loss = nucleus_loss
            else:
                total_loss = total_loss + nucleus_loss

            if compute_derivative:
                full_dL = torch.zeros_like(shift_calc)
                full_dL[:, mask] = k * dL
                dL_dshift[nucleus] = full_dL

        # If no shifts were computed, return zero tensor with gradients
        if total_loss is None:
            total_loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True).sum()

        if compute_derivative:
            return total_loss, dL_dshift
        return total_loss

    def compute_args(
        self,
        feats: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Tuple, None, None, None]:
        """Prepare arguments for compute_function."""
        k = parameters.get('k', 1.0)
        return None, (k,), None, None, None

    def compute_gradient(
        self,
        coords: torch.Tensor,
        feats: Dict[str, Any],
        parameters: Dict[str, Any],
        step: int = 0,
    ) -> torch.Tensor:
        """Compute gradient of chemical shift loss w.r.t. coordinates.

        Args:
            coords: [multiplicity, N_atoms, 3]
            feats: Feature dictionary
            parameters: Including k (strength), warmup, cutoff, etc.
            step: Current diffusion step

        Returns:
            gradient: [multiplicity, N_atoms, 3]
        """
        # Check scheduling
        progress = parameters.get('_relaxation', 0.0)
        warmup = parameters.get('warmup', 0.0)
        cutoff = parameters.get('cutoff', 0.9)
        guidance_interval = parameters.get('guidance_interval', 1)

        if progress < warmup or progress > cutoff:
            return torch.zeros_like(coords)

        if step % guidance_interval != 0:
            return torch.zeros_like(coords)

        k = parameters.get('k', 1.0)

        # Build atom mapping
        if self._atom_mapping_cache is None:
            self._build_atom_mapping(feats)

        # Exit inference mode and enable gradients (inference mode may be active during prediction)
        with torch.inference_mode(False):
            # Clone coords to create a new tensor not in inference mode, then enable gradients
            coords_grad = coords.detach().clone().requires_grad_(True)

            shifts = self._compute_shifts(coords_grad, feats)

            # Compute loss
            loss = self.compute_function(shifts, k, compute_derivative=False)

            # Backprop
            gradient = torch.autograd.grad(loss, coords_grad, create_graph=False)[0]

        # Normalize gradient to have unit max norm per sample (like CV functions)
        # This prevents gradient explosion and makes strength parameter meaningful
        grad_norms = gradient.norm(dim=-1, keepdim=True)  # [mult, n_atoms, 1]
        max_norm = grad_norms.max()
        if max_norm > 0:
            gradient = gradient / max_norm  # Now max gradient magnitude is 1.0

        # Apply gradient scaler if present
        if self.gradient_scaler is not None:
            gradient = self.gradient_scaler.apply(gradient, coords, feats, step, progress)

        return gradient
