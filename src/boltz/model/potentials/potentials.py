from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Set, List, Union
import warnings

import torch
import numpy as np
from boltz.data import const
from boltz.model.potentials.schedules import (
    ParameterSchedule,
    ExponentialInterpolation,
    PiecewiseStepFunction,
)
from boltz.model.loss.diffusionv2 import weighted_rigid_align

# Note: Metadynamics imports are done lazily inside get_potentials()
# to avoid circular import issues (they import Potential from this module)


class Potential(ABC):
    def __init__(
        self,
        parameters: Optional[
            Dict[str, Union[ParameterSchedule, float, int, bool]]
        ] = None,
    ):
        self.parameters = parameters
        self.gradient_scaler = None  # Optional CV-based gradient scaler

    def compute(self, coords, feats, parameters):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )

        if index.shape[1] == 0:
            return torch.zeros(coords.shape[:-2], device=coords.device)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_com_index = com_index[atom_pad_mask]
            unpad_coords = coords[..., atom_pad_mask, :]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=False,
        )
        energy = self.compute_function(
            value, *args, negation_mask=negation_mask, compute_derivative=False
        )

        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(neg_exp_energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            return (energy * softmax_energy).sum(dim=-1)

        return energy.sum(dim=tuple(range(1, energy.dim())))

    def compute_gradient(self, coords, feats, parameters, step: int = 0):
        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )
        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        if com_args is not None:
            com_index, atom_pad_mask = com_args
            unpad_coords = coords[..., atom_pad_mask, :]
            unpad_com_index = com_index[atom_pad_mask]
            coords = torch.zeros(
                (*unpad_coords.shape[:-2], unpad_com_index.max() + 1, 3),
                device=coords.device,
            ).scatter_reduce(
                -2,
                unpad_com_index.unsqueeze(-1).expand_as(unpad_coords),
                unpad_coords,
                "mean",
            )
            com_counts = torch.bincount(com_index[atom_pad_mask])
        else:
            com_index, atom_pad_mask = None, None

        if ref_args is not None:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = ref_args
            coords = coords[..., ref_atom_index, :]
        else:
            ref_coords, ref_mask, ref_atom_index, ref_token_index = (
                None,
                None,
                None,
                None,
            )

        if operator_args is not None:
            negation_mask, union_index = operator_args
        else:
            negation_mask, union_index = None, None

        value, grad_value = self.compute_variable(
            coords,
            index,
            ref_coords=ref_coords,
            ref_mask=ref_mask,
            compute_gradient=True,
        )
        energy, dEnergy = self.compute_function(
            value, 
            *args, negation_mask=negation_mask, compute_derivative=True
        )
        if union_index is not None:
            neg_exp_energy = torch.exp(-1 * parameters["union_lambda"] * energy)
            Z = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                neg_exp_energy,
                "sum",
            )
            softmax_energy = neg_exp_energy / Z[..., union_index]
            softmax_energy[Z[..., union_index] == 0] = 0
            f = torch.zeros(
                (*energy.shape[:-1], union_index.max() + 1), device=union_index.device
            ).scatter_reduce(
                -1,
                union_index.expand_as(energy),
                energy * softmax_energy,
                "sum",
            )
            dSoftmax = (
                dEnergy
                * softmax_energy
                * (1 + parameters["union_lambda"] * (energy - f[..., union_index]))
            )
            # Reshape dSoftmax to broadcast correctly
            dSoftmax_reshaped = dSoftmax
            while dSoftmax_reshaped.dim() < grad_value.dim() - 1:
                dSoftmax_reshaped = dSoftmax_reshaped.unsqueeze(-1)

            grad_value_flat = grad_value.flatten(start_dim=-3, end_dim=-2)

            # For distance-based potentials (contacts), grad_value has shape [batch, 2, N_pairs, 3]
            # After flattening: [batch, 2*N_pairs, 3]
            # dSoftmax has shape [batch, N_pairs], need to repeat for each atom in pair
            if grad_value.dim() == 4 and grad_value.shape[-3] == 2:
                # This is a distance potential with 2 atoms per pair
                # Repeat dSoftmax for each atom: [batch, N_pairs] -> [batch, 2*N_pairs]
                dSoftmax_reshaped = dSoftmax_reshaped.repeat_interleave(2, dim=-2)

            prod = dSoftmax_reshaped * grad_value_flat

            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),
                prod,
                "sum",
            )
        else:
            # Handle batch dimensions properly for both scalar and pairwise potentials
            # Scalar potentials (Rg): grad_value shape [batch, 1, N_atoms, 3], dEnergy shape [batch]
            # Pairwise potentials (CA distance): grad_value shape [batch, N_pairs, 2, 3], dEnergy shape [batch, N_pairs]

            # Check if this is a pairwise potential (distance-based)
            # Pairwise potentials: grad_value shape [batch, 2, N_pairs, 3] (2 atoms per pair)
            # Scalar potentials: grad_value shape [batch, 1, N_atoms, 3] (1 dummy dimension)
            # Check that dim -3 == 2 (exactly 2 atoms per pair)
            is_pairwise = grad_value.dim() == 4 and grad_value.shape[-3] == 2

            if is_pairwise:
                # For pairwise potentials (e.g., distance constraints)
                # grad_value shape: [batch, num_atoms_per_pair, N_pairs, 3]
                # dEnergy shape: [batch, N_pairs]
                # Need to repeat dEnergy for each atom and match the ordering
                num_atoms_per_pair = grad_value.shape[-3]  # dim -3, not -2

                # Expand dEnergy: [batch, N_pairs] -> [batch, num_atoms_per_pair, N_pairs]
                # First unsqueeze at dim -2 to add atoms dimension
                dEnergy_expanded = dEnergy.unsqueeze(-2).expand(
                    *dEnergy.shape[:-1], num_atoms_per_pair, dEnergy.shape[-1]
                )

                # Flatten to [batch, num_atoms_per_pair*N_pairs]
                dEnergy_flat = dEnergy_expanded.flatten(start_dim=-2, end_dim=-1)
                # Add dimension for coordinates: [batch, num_atoms_per_pair*N_pairs, 1]
                dEnergy_reshaped = dEnergy_flat.unsqueeze(-1)

                # Flatten grad_value: [batch, num_atoms_per_pair, N_pairs, 3] -> [batch, num_atoms_per_pair*N_pairs, 3]
                grad_value_flat = grad_value.flatten(start_dim=-3, end_dim=-2)
            else:
                # For scalar potentials, use original logic
                dEnergy_reshaped = dEnergy
                while dEnergy_reshaped.dim() < grad_value.dim() - 1:
                    dEnergy_reshaped = dEnergy_reshaped.unsqueeze(-1)
                grad_value_flat = grad_value.flatten(start_dim=-3, end_dim=-2)

            # Multiply: dEnergy_reshaped * grad_value_flat
            prod = dEnergy_reshaped * grad_value_flat

            if prod.dim() > 3:
                prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
            grad_atom = torch.zeros_like(coords).scatter_reduce(
                -2,
                index.flatten(start_dim=0, end_dim=1)
                .unsqueeze(-1)
                .expand((*coords.shape[:-2], -1, 3)),  # 9 x 516 x 3
                prod,
                "sum",
            )

        if com_index is not None:
            grad_atom = grad_atom[..., com_index, :]
        elif ref_token_index is not None:
            grad_atom = grad_atom[..., ref_token_index, :]

        # Apply gradient scaling if scaler is set
        if self.gradient_scaler is not None:
            progress = parameters.get('_relaxation', 0.0) if parameters else 0.0
            grad_atom = self.gradient_scaler.apply(grad_atom, coords, feats, step, progress)

        return grad_atom

    def compute_parameters(self, t):
        if self.parameters is None:
            return None
        parameters = {
            name: parameter
            if not isinstance(parameter, ParameterSchedule)
            else parameter.compute(t)
            for name, parameter in self.parameters.items()
        }
        # Store diffusion progress for warmup/cutoff checks
        # t=1.0 at start of diffusion, t=0.0 at end
        # _relaxation is inverted: 0.0 at start, 1.0 at end
        parameters['_relaxation'] = 1.0 - t
        return parameters

    @abstractmethod
    def compute_function(
        self, value, *args, negation_mask=None, compute_derivative=False
    ):
        raise NotImplementedError

    @abstractmethod
    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError

    @abstractmethod
    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError

    def get_reference_coords(self, feats, parameters):
        return None, None


class FlatBottomPotential(Potential):
    def compute_function(
        self,
        value,
        k,
        lower_bounds,
        upper_bounds,
        negation_mask=None,
        compute_derivative=False,
    ):
        if lower_bounds is None:
            lower_bounds = torch.full_like(value, float("-inf"))
        if upper_bounds is None:
            upper_bounds = torch.full_like(value, float("inf"))
        lower_bounds = lower_bounds.expand_as(value).clone()
        upper_bounds = upper_bounds.expand_as(value).clone()

        if negation_mask is not None:
            unbounded_below_mask = torch.isneginf(lower_bounds)
            unbounded_above_mask = torch.isposinf(upper_bounds)
            unbounded_mask = unbounded_below_mask + unbounded_above_mask
            assert torch.all(unbounded_mask + negation_mask)
            lower_bounds[~unbounded_above_mask * ~negation_mask] = upper_bounds[
                ~unbounded_above_mask * ~negation_mask
            ]
            upper_bounds[~unbounded_above_mask * ~negation_mask] = float("inf")
            upper_bounds[~unbounded_below_mask * ~negation_mask] = lower_bounds[
                ~unbounded_below_mask * ~negation_mask
            ]
            lower_bounds[~unbounded_below_mask * ~negation_mask] = float("-inf")

        neg_overflow_mask = value < lower_bounds
        pos_overflow_mask = value > upper_bounds

        energy = torch.zeros_like(value)
        energy[neg_overflow_mask] = (k * (lower_bounds - value))[neg_overflow_mask]
        energy[pos_overflow_mask] = (k * (value - upper_bounds))[pos_overflow_mask]
        if not compute_derivative:
            return energy

        dEnergy = torch.zeros_like(value)
        dEnergy[neg_overflow_mask] = (
            -1 * k.expand_as(neg_overflow_mask)[neg_overflow_mask]
        )
        dEnergy[pos_overflow_mask] = (
            1 * k.expand_as(pos_overflow_mask)[pos_overflow_mask]
        )

        return energy, dEnergy


class ReferencePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords, ref_mask, compute_gradient=False
    ):
        aligned_ref_coords = weighted_rigid_align(
            ref_coords.float(),
            coords[:, index].float(),
            ref_mask,
            ref_mask,
        )

        r = coords[:, index] - aligned_ref_coords
        r_norm = torch.linalg.norm(r, dim=-1)

        if not compute_gradient:
            return r_norm

        r_hat = r / r_norm.unsqueeze(-1)
        grad = (r_hat * ref_mask.unsqueeze(-1)).unsqueeze(1)
        return r_norm, grad


class DistancePotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
        # Add epsilon to avoid division by zero when atoms overlap
        r_ij_norm_safe = r_ij_norm + 1e-8
        r_hat_ij = r_ij / r_ij_norm_safe.unsqueeze(-1)

        if not compute_gradient:
            return r_ij_norm

        grad_i = r_hat_ij
        grad_j = -1 * r_hat_ij
        grad = torch.stack((grad_i, grad_j), dim=1)
        return r_ij_norm, grad


class DihedralPotential(Potential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
        r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

        n_ijk = torch.cross(r_ij, r_kj, dim=-1)
        n_jkl = torch.cross(r_kj, r_kl, dim=-1)

        r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
        n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
        n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)

        sign_phi = torch.sign(
            r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl, dim=-1).unsqueeze(-1)
        ).squeeze(-1, -2)
        phi = sign_phi * torch.arccos(
            torch.clamp(
                (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1, -2)
                / (n_ijk_norm * n_jkl_norm),
                -1 + 1e-8,
                1 - 1e-8,
            )
        )

        if not compute_gradient:
            return phi

        # Add epsilon to prevent division by zero for near-collinear atoms
        r_kj_norm_sq = r_kj_norm**2 + 1e-8
        n_ijk_norm_sq = n_ijk_norm**2 + 1e-8
        n_jkl_norm_sq = n_jkl_norm**2 + 1e-8

        a = (
            (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / r_kj_norm_sq
        ).unsqueeze(-1)
        b = (
            (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1, -2) / r_kj_norm_sq
        ).unsqueeze(-1)

        grad_i = n_ijk * (r_kj_norm / n_ijk_norm_sq).unsqueeze(-1)
        grad_l = -1 * n_jkl * (r_kj_norm / n_jkl_norm_sq).unsqueeze(-1)
        grad_j = (a - 1) * grad_i - b * grad_l
        grad_k = (b - 1) * grad_l - a * grad_i
        grad = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=1)
        return phi, grad


class AbsDihedralPotential(DihedralPotential):
    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        if not compute_gradient:
            phi = super().compute_variable(
                coords, index, compute_gradient=compute_gradient
            )
            phi = torch.abs(phi)
            return phi

        phi, grad = super().compute_variable(
            coords, index, compute_gradient=compute_gradient
        )
        grad[(phi < 0)[..., None, :, None].expand_as(grad)] *= -1
        phi = torch.abs(phi)

        return phi, grad


class PoseBustersPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["rdkit_bounds_index"][0]
        lower_bounds = feats["rdkit_lower_bounds"][0].clone()
        upper_bounds = feats["rdkit_upper_bounds"][0].clone()
        bond_mask = feats["rdkit_bounds_bond_mask"][0]
        angle_mask = feats["rdkit_bounds_angle_mask"][0]

        lower_bounds[bond_mask * ~angle_mask] *= 1.0 - parameters["bond_buffer"]
        upper_bounds[bond_mask * ~angle_mask] *= 1.0 + parameters["bond_buffer"]
        lower_bounds[~bond_mask * angle_mask] *= 1.0 - parameters["angle_buffer"]
        upper_bounds[~bond_mask * angle_mask] *= 1.0 + parameters["angle_buffer"]
        lower_bounds[bond_mask * angle_mask] *= 1.0 - min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        upper_bounds[bond_mask * angle_mask] *= 1.0 + min(
            parameters["bond_buffer"], parameters["angle_buffer"]
        )
        lower_bounds[~bond_mask * ~angle_mask] *= 1.0 - parameters["clash_buffer"]
        upper_bounds[~bond_mask * ~angle_mask] = float("inf")

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=pair_index.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=pair_index.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]
        bond_cutoffs = 0.35 + atom_vdw_radii[pair_index].mean(dim=0)
        lower_bounds[~bond_mask] = torch.max(lower_bounds[~bond_mask], bond_cutoffs[~bond_mask])
        upper_bounds[bond_mask] = torch.min(upper_bounds[bond_mask], bond_cutoffs[bond_mask])

        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class ConnectionsPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        pair_index = feats["connected_atom_index"][0]
        lower_bounds = None
        upper_bounds = torch.full(
            (pair_index.shape[1],), parameters["buffer"], device=pair_index.device
        )
        k = torch.ones_like(upper_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class VDWOverlapPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = (chain_sizes > 1)[atom_chain_id]

        vdw_radii = torch.zeros(
            const.num_elements, dtype=torch.float32, device=atom_chain_id.device
        )
        vdw_radii[1:119] = torch.tensor(
            const.vdw_radii, dtype=torch.float32, device=atom_chain_id.device
        )
        atom_vdw_radii = (
            feats["ref_element"].float() @ vdw_radii.unsqueeze(-1)
        ).squeeze(-1)[0]

        pair_index = torch.triu_indices(
            atom_chain_id.shape[0],
            atom_chain_id.shape[0],
            1,
            device=atom_chain_id.device,
        )

        pair_pad_mask = atom_pad_mask[pair_index].all(dim=0)
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]

        num_chains = atom_chain_id.max() + 1
        connected_chain_index = feats["connected_chain_index"][0]
        connected_chain_matrix = torch.eye(
            num_chains, device=atom_chain_id.device, dtype=torch.bool
        )
        connected_chain_matrix[connected_chain_index[0], connected_chain_index[1]] = (
            True
        )
        connected_chain_matrix[connected_chain_index[1], connected_chain_index[0]] = (
            True
        )
        connected_chain_mask = connected_chain_matrix[
            atom_chain_id[pair_index[0]], atom_chain_id[pair_index[1]]
        ]

        pair_index = pair_index[
            :, pair_pad_mask * pair_ion_mask * ~connected_chain_mask
        ]

        lower_bounds = atom_vdw_radii[pair_index].sum(dim=0) * (
            1.0 - parameters["buffer"]
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return pair_index, (k, lower_bounds, upper_bounds), None, None, None


class SymmetricChainCOMPotential(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        atom_chain_id = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["asym_id"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )[0]
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        chain_sizes = torch.bincount(atom_chain_id[atom_pad_mask])
        single_ion_mask = chain_sizes > 1

        pair_index = feats["symmetric_chain_index"][0]
        pair_ion_mask = single_ion_mask[pair_index[0]] * single_ion_mask[pair_index[1]]
        pair_index = pair_index[:, pair_ion_mask]
        lower_bounds = torch.full(
            (pair_index.shape[1],),
            parameters["buffer"],
            dtype=torch.float32,
            device=pair_index.device,
        )
        upper_bounds = None
        k = torch.ones_like(lower_bounds)

        return (
            pair_index,
            (k, lower_bounds, upper_bounds),
            (atom_chain_id, atom_pad_mask),
            None,
            None,
        )


class StereoBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        stereo_bond_index = feats["stereo_bond_index"][0]
        stereo_bond_orientations = feats["stereo_bond_orientations"][0].bool()

        lower_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        upper_bounds = torch.zeros(
            stereo_bond_orientations.shape, device=stereo_bond_orientations.device
        )
        lower_bounds[stereo_bond_orientations] = torch.pi - parameters["buffer"]
        upper_bounds[stereo_bond_orientations] = float("inf")
        lower_bounds[~stereo_bond_orientations] = float("-inf")
        upper_bounds[~stereo_bond_orientations] = parameters["buffer"]

        k = torch.ones_like(lower_bounds)

        return stereo_bond_index, (k, lower_bounds, upper_bounds), None, None, None


class ChiralAtomPotential(FlatBottomPotential, DihedralPotential):
    def compute_args(self, feats, parameters):
        chiral_atom_index = feats["chiral_atom_index"][0]
        chiral_atom_orientations = feats["chiral_atom_orientations"][0].bool()

        lower_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        upper_bounds = torch.zeros(
            chiral_atom_orientations.shape, device=chiral_atom_orientations.device
        )
        lower_bounds[chiral_atom_orientations] = parameters["buffer"]
        upper_bounds[chiral_atom_orientations] = float("inf")
        upper_bounds[~chiral_atom_orientations] = -1 * parameters["buffer"]
        lower_bounds[~chiral_atom_orientations] = float("-inf")

        k = torch.ones_like(lower_bounds)
        return chiral_atom_index, (k, lower_bounds, upper_bounds), None, None, None


class PlanarBondPotential(FlatBottomPotential, AbsDihedralPotential):
    def compute_args(self, feats, parameters):
        double_bond_index = feats["planar_bond_index"][0].T
        double_bond_improper_index = torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 0, 3],
            ],
            device=double_bond_index.device,
        ).T
        improper_index = (
            double_bond_index[:, double_bond_improper_index]
            .swapaxes(0, 1)
            .flatten(start_dim=1)
        )
        lower_bounds = None
        upper_bounds = torch.full(
            (improper_index.shape[1],),
            parameters["buffer"],
            device=improper_index.device,
        )
        k = torch.ones_like(upper_bounds)

        return improper_index, (k, lower_bounds, upper_bounds), None, None, None


class TemplateReferencePotential(FlatBottomPotential, ReferencePotential):
    def compute_args(self, feats, parameters):
        if "template_mask_cb" not in feats or "template_force" not in feats:
            return torch.empty([1, 0]), None, None, None, None

        template_mask = feats["template_mask_cb"][feats["template_force"]]
        if template_mask.shape[0] == 0:
            return torch.empty([1, 0]), None, None, None, None

        ref_coords = feats["template_cb"][feats["template_force"]].clone()
        ref_mask = feats["template_mask_cb"][feats["template_force"]].clone()
        ref_atom_index = (
            torch.bmm(
                feats["token_to_rep_atom"].float(),
                torch.arange(
                    feats["atom_pad_mask"].shape[1],
                    device=feats["atom_pad_mask"].device,
                    dtype=torch.float32,
                )[None, :, None],
            )
            .squeeze(-1)
            .long()
        )[0]
        ref_token_index = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["token_index"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()
        )[0]

        index = torch.arange(
            template_mask.shape[-1], dtype=torch.long, device=template_mask.device
        )[None]
        upper_bounds = torch.full(
            template_mask.shape, float("inf"), device=index.device, dtype=torch.float32
        )
        ref_idxs = torch.argwhere(template_mask).T
        upper_bounds[ref_idxs.unbind()] = feats["template_force_threshold"][
            feats["template_force"]
        ][ref_idxs[0]]

        lower_bounds = None
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            (ref_coords, ref_mask, ref_atom_index, ref_token_index),
            None,
        )


class ContactPotentital(FlatBottomPotential, DistancePotential):
    def compute_args(self, feats, parameters):
        index = feats["contact_pair_index"][0]
        union_index = feats["contact_union_index"][0]
        negation_mask = feats["contact_negation_mask"][0]
        lower_bounds = None
        upper_bounds = feats["contact_thresholds"][0].clone()
        k = torch.ones_like(upper_bounds)
        return (
            index,
            (k, lower_bounds, upper_bounds),
            None,
            None,
            (negation_mask, union_index),
        )


class RadiusOfGyrationPotential(Potential):
    """Potential for steering toward a target radius of gyration."""

    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False
    ):
        """
        Compute radius of gyration and optionally its gradient.

        Rg = sqrt(mean(||r_i - r_com||^2))

        Args:
            coords: Atom coordinates, shape [..., N_atoms, 3]
            index: Atom indices to include in Rg calculation, shape [N_selected]
            compute_gradient: Whether to compute gradients

        Returns:
            Rg value and optionally gradient w.r.t. coordinates
        """
        # Select atoms to include in Rg calculation
        # index has shape [1, N_selected], so we use index[0]
        selected_coords = coords[..., index[0], :]  # [..., N_selected, 3]

        # Compute center of mass (assuming equal masses)
        com = selected_coords.mean(dim=-2, keepdim=True)  # [..., 1, 3]

        # Compute squared distances from COM
        r_from_com = selected_coords - com  # [..., N_selected, 3]
        r_squared = (r_from_com ** 2).sum(dim=-1)  # [..., N_selected]

        # Radius of gyration
        mean_r_squared = r_squared.mean(dim=-1)  # [...]
        rg = torch.sqrt(mean_r_squared + 1e-8)  # Add small epsilon for numerical stability

        if not compute_gradient:
            return rg

        # Gradient of Rg w.r.t. atom positions
        # dRg/dr_i = (1/(N*Rg)) * (r_i - r_com - mean(r_j - r_com))
        N = selected_coords.shape[-2]
        mean_displacement = r_from_com.mean(dim=-2, keepdim=True)  # [..., 1, 3]

        grad_per_atom = (r_from_com - mean_displacement) / (N * rg.unsqueeze(-1).unsqueeze(-1))
        # Shape: [..., N_selected, 3]

        # Reshape to match expected output format: [..., 1, N_selected, 3]
        grad = grad_per_atom.unsqueeze(-3)

        return rg, grad

    def compute_function(
        self, value, k, target_rg, negation_mask=None, compute_derivative=False
    ):
        """
        Compute energy as quadratic penalty from target Rg.

        E = k * (Rg - target_Rg)^2

        Args:
            value: Current Rg values
            k: Force constant
            target_rg: Target radius of gyration
            compute_derivative: Whether to compute derivative

        Returns:
            Energy and optionally derivative w.r.t. Rg
        """
        diff = value - target_rg
        energy = k * (diff ** 2)

        if not compute_derivative:
            return energy

        dEnergy = 2 * k * diff
        return energy, dEnergy

    def compute_args(self, feats, parameters):
        """
        Extract relevant atoms and parameters for Rg calculation.

        Args:
            feats: Feature dictionary
            parameters: Dict with 'target_rg', 'k', and optional 'chain_ids'

        Returns:
            Tuple of (index, args, com_args, ref_args, operator_args)
        """
        atom_pad_mask = feats["atom_pad_mask"][0].bool()

        # Determine which atoms to include
        if "chain_ids" in parameters and parameters["chain_ids"] is not None:
            # Filter by specific chain IDs
            atom_chain_id = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["asym_id"].unsqueeze(-1).float()
                )
                .squeeze(-1)
                .long()
            )[0]

            chain_mask = torch.zeros_like(atom_chain_id, dtype=torch.bool)
            for chain_id in parameters["chain_ids"]:
                # Convert string chain labels (e.g., "A", "B") to integer asym_ids
                if isinstance(chain_id, str):
                    try:
                        chain_id_int = int(chain_id)
                    except ValueError:
                        chain_id_int = ord(chain_id.upper()) - ord('A')
                else:
                    chain_id_int = chain_id
                chain_mask |= (atom_chain_id == chain_id_int)

            selected_mask = atom_pad_mask & chain_mask
        else:
            # Use all non-padded atoms
            selected_mask = atom_pad_mask

        # Get indices of selected atoms
        index = torch.arange(
            atom_pad_mask.shape[0], dtype=torch.long, device=atom_pad_mask.device
        )[selected_mask]

        # Reshape to [1, N] to match expected 2D format in base class
        index = index.unsqueeze(0)

        # Prepare arguments for compute_function
        k = torch.tensor([parameters["k"]], device=atom_pad_mask.device)
        target_rg = torch.tensor([parameters["target_rg"]], device=atom_pad_mask.device)

        return index, (k, target_rg), None, None, None


class GenericCVPotential(Potential):
    """
    Generic potential for steering any collective variable toward a target.

    Works with any CV from the CV_REGISTRY in collective_variables.py.
    Supports both steering (harmonic restraint to target) and can be extended
    for other bias types.

    E = k * (CV - target)^2
    """

    def __init__(self, cv_function, parameters=None, debug=False):
        """
        Args:
            cv_function: Callable (coords, feats, step) -> (value, gradient)
                        Returns CV value [multiplicity] and gradient [multiplicity, N_atoms, 3]
            parameters: Dict with 'target', 'k', 'guidance_weight', 'guidance_interval'
            debug: Whether to print debug info during steering
        """
        super().__init__(parameters)
        self.cv_function = cv_function
        self.debug = debug
        self._step_count = 0

    def compute_variable(
        self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False, feats=None
    ):
        """
        Compute CV value using the wrapped CV function.

        Args:
            coords: Atom coordinates [multiplicity, N_atoms, 3]
            index: Not used (CV function handles atom selection)
            compute_gradient: Whether to compute gradient
            feats: Feature dictionary (passed to CV function)

        Returns:
            cv_value: [multiplicity] CV values
            gradient: [multiplicity, N_atoms, 3] if compute_gradient=True
        """
        if feats is None:
            feats = {}

        # Call the CV function
        cv_value, cv_gradient = self.cv_function(coords, feats, step=0)

        if not compute_gradient:
            return cv_value

        return cv_value, cv_gradient

    def compute_function(
        self, value, k, target, negation_mask=None, compute_derivative=False
    ):
        """
        Compute harmonic energy from target CV value.

        E = k * (CV - target)^2

        Args:
            value: Current CV values [multiplicity]
            k: Force constant
            target: Target CV value
            compute_derivative: Whether to compute derivative

        Returns:
            Energy and optionally derivative w.r.t. CV
        """
        diff = value - target
        energy = k * (diff ** 2)

        if not compute_derivative:
            return energy

        dEnergy = 2 * k * diff
        return energy, dEnergy

    def compute_args(self, feats, parameters):
        """
        Prepare arguments for CV computation.

        Returns:
            Tuple of (index, args, com_args, ref_args, operator_args)
        """
        device = feats["atom_pad_mask"].device

        # Index is not used by CV function, but we need a valid tensor
        atom_pad_mask = feats["atom_pad_mask"][0].bool()
        index = torch.arange(
            atom_pad_mask.shape[0], dtype=torch.long, device=device
        )[atom_pad_mask].unsqueeze(0)

        k = torch.tensor([parameters["k"]], device=device)
        target = torch.tensor([parameters["target"]], device=device)

        return index, (k, target), None, None, None

    def compute_gradient(self, coords, feats, parameters, step: int = 0):
        """
        Override base class to use CV function directly.

        The CV function returns gradients directly, so we don't need
        the complex index-based gradient computation from base class.
        """
        # Check warmup and cutoff (0.0 = start of diffusion, 1.0 = end)
        # Steering is active when: warmup <= progress <= cutoff
        progress = parameters.get('_relaxation', 0.0)
        warmup = parameters.get('warmup', 0.0)
        cutoff = parameters.get('cutoff', 0.75)
        if progress < warmup or progress > cutoff:
            return torch.zeros_like(coords)

        # Get CV value and gradient
        cv_value, cv_gradient = self.cv_function(coords, feats, step=step)
        # cv_value: [multiplicity]
        # cv_gradient: [multiplicity, N_atoms, 3]

        # Compute energy derivative w.r.t. CV
        target = parameters["target"]
        k = parameters["k"]

        # Debug: print CV value periodically
        self._step_count += 1
        if self.debug and self._step_count % 10 == 1:
            cv_name = parameters.get("cv_name", "CV")
            import math
            cv_deg = cv_value.mean().item() * 180 / math.pi if "angle" in cv_name.lower() or "dihedral" in cv_name.lower() else cv_value.mean().item()
            target_deg = target * 180 / math.pi if "angle" in cv_name.lower() or "dihedral" in cv_name.lower() else target
            diff_val = cv_value.mean().item() - target
            dE_dCV_val = 2 * k * diff_val
            grad_norm = cv_gradient.norm().item()
            print(f"[DEBUG] {cv_name} step {self._step_count}: CV={cv_deg:.1f}°, target={target_deg:.1f}°, diff={diff_val:.3f}, dE/dCV={dE_dCV_val:.3f}, |grad|={grad_norm:.3f}, progress={progress:.2f}", flush=True)

        # Check for ensemble mode
        ensemble_mode = parameters.get("ensemble", False)

        if ensemble_mode:
            # Per-ensemble mode: steer ensemble mean toward target
            # All samples move coherently in the same direction
            multiplicity = cv_value.shape[0]
            cv_mean = cv_value.mean()
            diff = cv_mean - target

            # Handle periodic boundary for dihedral angles (-π to +π)
            cv_name = parameters.get("cv_name", "")
            if "dihedral" in cv_name.lower():
                import math
                if diff > math.pi:
                    diff = diff - 2 * math.pi
                elif diff < -math.pi:
                    diff = diff + 2 * math.pi

            # Use mean gradient direction for coherent ensemble movement
            mean_gradient = cv_gradient.mean(dim=0)  # [N_atoms, 3]

            # Normalize mean gradient to have same scale as individual gradients
            grad_norm = mean_gradient.norm(dim=-1, keepdim=True).max()
            if grad_norm > 1e-8:
                mean_gradient = mean_gradient / grad_norm

            # Apply same gradient to all samples (broadcast)
            # 2 * k * diff gives similar strength to per-sample mode
            grad = (2 * k * diff) * mean_gradient.unsqueeze(0).expand(multiplicity, -1, -1)
        else:
            # Per-sample mode: each sample steered independently
            diff = cv_value - target
            # Handle periodic boundary for dihedral angles (-π to +π)
            cv_name = parameters.get("cv_name", "")
            if "dihedral" in cv_name.lower():
                import math
                diff = torch.where(diff > math.pi, diff - 2 * math.pi, diff)
                diff = torch.where(diff < -math.pi, diff + 2 * math.pi, diff)
            dE_dCV = 2 * k * diff  # [multiplicity]

            # Chain rule: dE/dr = dE/dCV * dCV/dr
            # dE_dCV: [multiplicity]
            # cv_gradient: [multiplicity, N_atoms, 3]
            grad = dE_dCV.unsqueeze(-1).unsqueeze(-1) * cv_gradient

        # Apply gradient scaling if scaler is set
        if self.gradient_scaler is not None:
            grad = self.gradient_scaler.apply(grad, coords, feats, step, progress)

        return grad


class RepulsionPotential(Potential):
    """
    Pairwise repulsion potential for pushing samples apart in CV space.

    For each pair of samples (i, j), applies Gaussian repulsion based on CV difference:
        E_ij = strength * exp(-(CV_i - CV_j)² / (2*sigma²))

    This creates repulsion when samples have similar CV values, pushing them
    to explore different regions of CV space.

    Mean energy for sample i: E_i = mean_{j≠i} E_ij
    Gradient: dE_i/dr = (dE_i/dCV_i) * (dCV_i/dr)

    Note: Sample-count invariant - gradient is averaged over pairs, so the same
    strength parameter works regardless of multiplicity.
    """

    def __init__(self, cv_function, parameters=None):
        """
        Args:
            cv_function: Callable (coords, feats, step) -> (value, gradient)
            parameters: Dict with 'strength', 'sigma', 'guidance_weight', etc.
        """
        super().__init__(parameters)
        self.cv_function = cv_function

    def compute_variable(self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False, feats=None):
        """Compute CV value using the wrapped CV function."""
        if feats is None:
            feats = {}
        cv_value, cv_gradient = self.cv_function(coords, feats, step=0)
        if not compute_gradient:
            return cv_value
        return cv_value, cv_gradient

    def compute_function(self, value, k, target, negation_mask=None, compute_derivative=False):
        """Compute pairwise Gaussian repulsion energy."""
        strength = k
        sigma = target if target > 0 else 0.5
        # For interface compatibility - actual computation is in compute_gradient
        gaussian = torch.exp(-value ** 2 / (2 * sigma ** 2))
        energy = strength * gaussian
        if not compute_derivative:
            return energy
        dEnergy = -strength * (value / (sigma ** 2)) * gaussian
        return energy, dEnergy

    def compute_args(self, feats, parameters):
        """Returns empty index so base compute() safely returns zeros."""
        return torch.empty(1, 0, dtype=torch.long), (0,), None, None, None

    def compute_gradient(self, coords, feats, parameters, step: int = 0):
        """
        Compute pairwise Gaussian repulsion gradient (sample-count invariant).

        For samples i and j with CV values s_i and s_j:
            E_ij = strength * exp(-(s_i - s_j)² / (2*sigma²))

        Mean energy for sample i:
            E_i = mean_{j≠i} E_ij = sum_{j≠i} E_ij / (N-1)

        Gradient w.r.t. s_i (averaged over pairs):
            dE_i/ds_i = mean_{j≠i} -strength * (s_i - s_j) / sigma² * exp(-(s_i - s_j)²/(2*sigma²))

        Chain rule:
            dE_i/dr = dE_i/ds_i * ds_i/dr

        Note: Gradient is sample-count invariant - same strength works regardless of multiplicity.
        """
        # Check warmup and cutoff
        progress = parameters.get('_relaxation', 0.0)
        warmup = parameters.get('warmup', 0.2)
        cutoff = parameters.get('cutoff', 0.75)
        if progress < warmup or progress > cutoff:
            return torch.zeros_like(coords)

        # Get CV value and gradient
        cv_value, cv_gradient = self.cv_function(coords, feats, step=step)
        # cv_value: [multiplicity]
        # cv_gradient: [multiplicity, N_atoms, 3]

        multiplicity = cv_value.shape[0]
        if multiplicity < 2:
            # Need at least 2 samples for pairwise repulsion
            return torch.zeros_like(coords)

        strength = parameters.get("strength", 1.0)
        sigma = parameters.get("sigma", 0.5)

        # Compute pairwise CV differences: diff[i,j] = CV_i - CV_j
        cv_i = cv_value.unsqueeze(1)  # [mult, 1]
        cv_j = cv_value.unsqueeze(0)  # [1, mult]
        diff = cv_i - cv_j  # [mult, mult]

        # Gaussian repulsion: E_ij = strength * exp(-diff²/(2*sigma²))
        gaussian = torch.exp(-diff ** 2 / (2 * sigma ** 2))  # [mult, mult]

        # dE_ij/ds_i = -strength * (s_i - s_j) / sigma² * gaussian
        dE_ij_ds_i = -strength * (diff / (sigma ** 2)) * gaussian  # [mult, mult]

        # Zero out diagonal (no self-repulsion)
        mask = 1.0 - torch.eye(multiplicity, device=cv_value.device, dtype=cv_value.dtype)
        dE_ij_ds_i = dE_ij_ds_i * mask

        # Mean over j to get dE_i/ds_i for each sample i (sample-count invariant)
        dE_ds = dE_ij_ds_i.sum(dim=1) / (multiplicity - 1)  # [mult]

        # Chain rule: dE/dr = dE/ds * ds/dr
        grad = dE_ds.unsqueeze(-1).unsqueeze(-1) * cv_gradient  # [mult, N_atoms, 3]

        # Apply gradient scaling if scaler is set
        if self.gradient_scaler is not None:
            grad = self.gradient_scaler.apply(grad, coords, feats, step, progress)

        return grad


class VariancePotential(Potential):
    """
    Variance maximization potential for pushing samples apart in CV space.

    Maximizes the variance of CV values across samples by pushing each sample
    away from the mean CV value:
        E = -strength * Var(CV) = -strength * mean((CV_i - mean)²)

    Gradient for sample i:
        dE/dCV_i = -strength * 2 * (CV_i - mean) / N

    This creates a force that pushes samples with CV > mean to higher values,
    and samples with CV < mean to lower values, maximizing spread.

    Unlike RepulsionPotential which uses Gaussian decay (weak at large distances),
    VariancePotential provides linear force proportional to distance from mean,
    giving consistent push regardless of current spread.
    """

    def __init__(self, cv_function, parameters=None):
        """
        Args:
            cv_function: Callable (coords, feats, step) -> (value, gradient)
            parameters: Dict with 'strength', 'guidance_weight', etc.
        """
        super().__init__(parameters)
        self.cv_function = cv_function

    def compute_variable(self, coords, index, ref_coords=None, ref_mask=None, compute_gradient=False, feats=None):
        """Compute CV value using the wrapped CV function."""
        if feats is None:
            feats = {}
        cv_value, cv_gradient = self.cv_function(coords, feats, step=0)
        if not compute_gradient:
            return cv_value
        return cv_value, cv_gradient

    def compute_function(self, value, k, target, negation_mask=None, compute_derivative=False):
        """Not used - actual computation is in compute_gradient."""
        # For interface compatibility
        if not compute_derivative:
            return value
        return value, torch.ones_like(value)

    def compute_args(self, feats, parameters):
        """Returns empty index so base compute() safely returns zeros."""
        return torch.empty(1, 0, dtype=torch.long), (0,), None, None, None

    def compute_gradient(self, coords, feats, parameters, step: int = 0):
        """
        Compute variance maximization gradient.

        For N samples with CV values s_1, ..., s_N:
            mean = (1/N) * sum_i s_i
            Var = (1/N) * sum_i (s_i - mean)²
            E = -strength * Var

        Gradient w.r.t. s_i:
            dE/ds_i = -strength * 2 * (s_i - mean) / N

        Chain rule:
            dE/dr = dE/ds * ds/dr
        """
        # Check warmup and cutoff
        progress = parameters.get('_relaxation', 0.0)
        warmup = parameters.get('warmup', 0.2)
        cutoff = parameters.get('cutoff', 0.75)
        if progress < warmup or progress > cutoff:
            return torch.zeros_like(coords)

        # Get CV value and gradient
        cv_value, cv_gradient = self.cv_function(coords, feats, step=step)
        # cv_value: [multiplicity]
        # cv_gradient: [multiplicity, N_atoms, 3]

        multiplicity = cv_value.shape[0]
        if multiplicity < 2:
            # Need at least 2 samples for variance
            return torch.zeros_like(coords)

        strength = parameters.get("strength", 1.0)

        # Compute mean CV value
        mean_cv = cv_value.mean()  # scalar

        # Gradient: dE/ds_i = -strength * 2 * (s_i - mean) / N
        # Negative because we want to MAXIMIZE variance (minimize -variance)
        dE_ds = -strength * 2.0 * (cv_value - mean_cv) / multiplicity  # [mult]

        # Chain rule: dE/dr = dE/ds * ds/dr
        grad = dE_ds.unsqueeze(-1).unsqueeze(-1) * cv_gradient  # [mult, N_atoms, 3]

        # Apply gradient scaling if scaler is set
        if self.gradient_scaler is not None:
            grad = self.gradient_scaler.apply(grad, coords, feats, step, progress)

        return grad


class OptPotential(Potential):
    """
    Generic optimization potential for any collective variable.

    Pushes the system toward lower (minimize) or higher (maximize) CV values.
    Unlike GenericCVPotential which steers toward a specific target value,
    OptPotential continuously pushes in one direction.

    Energy:
        - maximize: E = -k * CV (gradient points toward higher CV)
        - minimize: E = k * CV  (gradient points toward lower CV)

    Gradient:
        dE/dr = ±k * dCV/dr

    The sign of the `strength` parameter determines direction:
        - strength > 0: Maximize CV (push toward higher values)
        - strength < 0: Minimize CV (push toward lower values)

    Example uses:
        - collective_variable: energy, strength: -1.0 → minimize energy (stable)
        - collective_variable: rg, strength: 1.0 → maximize Rg (expanded)
        - collective_variable: pair_rmsd, strength: 1.0 → maximize diversity
    """

    def __init__(self, cv_function, parameters=None):
        """
        Args:
            cv_function: Callable (coords, feats, step) -> (value, gradient)
                        Returns CV value [multiplicity] and gradient [multiplicity, N_atoms, 3]
            parameters: Dict with 'k', 'minimize', 'guidance_interval', 'warmup', 'cutoff'
        """
        super().__init__(parameters)
        self.cv_function = cv_function

    def compute_variable(self, coords, index=None, ref_coords=None, ref_mask=None,
                         compute_gradient=False, feats=None):
        """Compute CV value using the wrapped CV function."""
        if feats is None:
            feats = {}
        cv_value, cv_gradient = self.cv_function(coords, feats, step=0)
        if not compute_gradient:
            return cv_value
        return cv_value, cv_gradient

    def compute_function(self, value, k, negation_mask=None, compute_derivative=False, **kwargs):
        """
        Compute linear energy: E = k * CV (for minimize) or E = -k * CV (for maximize).

        This is a simple linear potential used for optimization.
        """
        minimize = kwargs.get('minimize', True)
        if minimize:
            energy = k * value
        else:
            energy = -k * value
        if not compute_derivative:
            return energy
        dEnergy = k if minimize else -k
        return energy, dEnergy * torch.ones_like(value)

    def compute_args(self, feats, parameters):
        """Prepare arguments for compute.

        Returns empty index so base compute() safely returns zeros.
        """
        k = parameters.get('k', 1.0)
        return torch.empty(1, 0, dtype=torch.long), (k,), None, None, None

    def compute_gradient(self, coords, feats, parameters, step: int = 0):
        """
        Compute gradient to minimize or maximize the CV.

        Args:
            coords: Atom coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary
            parameters: Dict with optimization parameters
            step: Current diffusion step

        Returns:
            Gradient [multiplicity, N_atoms, 3]
        """
        # Check warmup and cutoff (0.0 = start of diffusion, 1.0 = end)
        progress = parameters.get('_relaxation', 0.0)
        warmup = parameters.get('warmup', 0.0)
        cutoff = parameters.get('cutoff', 0.75)
        if progress < warmup or progress > cutoff:
            return torch.zeros_like(coords)

        # Get CV value and gradient
        cv_value, cv_gradient = self.cv_function(coords, feats, step=step)
        # cv_value: [multiplicity]
        # cv_gradient: [multiplicity, N_atoms, 3]

        k = parameters.get("k", 1.0)
        minimize = parameters.get("minimize", True)

        # For minimize: gradient = +k * dCV/dr (pushes toward lower CV)
        #   The CV gradient points toward higher CV values.
        #   To minimize, we follow the gradient (move toward lower energy = higher CV gradient direction? No!)
        #   Actually: dE/dr = k * dCV/dr for E = k*CV. The optimizer moves against gradient.
        #   So the "force" applied is -dE/dr = -k * dCV/dr, which points toward lower CV.
        #   But in this codebase, compute_gradient returns dE/dr which gets subtracted from coords.
        #
        # For maximize: gradient = -k * dCV/dr (pushes toward higher CV)
        #   E = -k*CV, dE/dr = -k * dCV/dr
        #   Force = -dE/dr = k * dCV/dr, which points toward higher CV.
        #
        # So: minimize=True → return +k * dCV/dr
        #     minimize=False → return -k * dCV/dr

        if minimize:
            dE_dCV = k  # scalar
        else:
            dE_dCV = -k

        # Chain rule: dE/dr = dE/dCV * dCV/dr
        grad = dE_dCV * cv_gradient

        # Apply log compression if requested (for numerical stability)
        # log_gradient: sign(g) * log(1 + |g|) - compresses large gradients
        if parameters.get("log_gradient", False):
            grad_sign = torch.sign(grad)
            grad_abs = torch.abs(grad)
            grad = grad_sign * torch.log1p(grad_abs)

        # Apply gradient scaling if scaler is set
        if self.gradient_scaler is not None:
            grad = self.gradient_scaler.apply(grad, coords, feats, step, progress)

        return grad


# Helper functions for Wasserstein-2 optimal transport loss
def sinkhorn_distance(
    p: torch.Tensor,           # [N_bins] - calculated P(r)
    q: torch.Tensor,           # [N_bins] - experimental P(r)
    cost_matrix: torch.Tensor, # [N_bins, N_bins] - (r_i - r_j)²
    epsilon: float,            # Entropic regularization
    num_iter: int = 100,
    tol: float = 1e-6,
    return_plan: bool = False
) -> Union[torch.Tensor, tuple]:
    """
    Compute Wasserstein-2 distance using Sinkhorn algorithm.

    The Sinkhorn algorithm solves the entropic-regularized optimal transport:
        min_{T} Σ_ij C_ij T_ij + ε H(T)
        s.t. T @ 1 = p, T^T @ 1 = q, T ≥ 0

    where C is the cost matrix and H(T) is the entropy.

    Algorithm:
        K = exp(-C/ε)
        for iter in range(num_iter):
            u = p / (K @ v)
            v = q / (K.T @ u)
        T = diag(u) @ K @ diag(v)
        W2² = sum(T * C)

    Args:
        p: Source distribution (calculated P(r)), normalized
        q: Target distribution (experimental P(r)), normalized
        cost_matrix: Pairwise cost matrix (r_i - r_j)²
        epsilon: Entropic regularization parameter (smaller = more accurate, slower)
        num_iter: Maximum number of Sinkhorn iterations
        tol: Convergence tolerance
        return_plan: If True, return (loss, transport_plan), else just loss

    Returns:
        loss: Wasserstein-2² distance
        transport_plan: Optimal transport plan T (if return_plan=True)
    """
    # Kernel matrix
    K = torch.exp(-cost_matrix / epsilon)

    # Initialize dual variables
    u = torch.ones_like(p)
    v = torch.ones_like(q)

    # Sinkhorn iterations
    for i in range(num_iter):
        u_prev = u.clone()

        # Update dual variables
        u = p / (K @ v + 1e-10)
        v = q / (K.t() @ u + 1e-10)

        # Check convergence
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    # Compute transport plan
    transport_plan = torch.diag(u) @ K @ torch.diag(v)

    # Compute W2² distance
    loss = torch.sum(transport_plan * cost_matrix)

    if return_plan:
        return loss, transport_plan
    return loss


def compute_w2_gradient(
    pr_calc: torch.Tensor,         # [N_bins]
    pr_exp: torch.Tensor,          # [N_bins]
    transport_plan: torch.Tensor,  # [N_bins, N_bins]
    cost_matrix: torch.Tensor,     # [N_bins, N_bins]
    dr: float
) -> torch.Tensor:
    """
    Compute gradient of W2² distance w.r.t. calculated P(r).

    Uses implicit differentiation through the Sinkhorn solution.

    The gradient is:
        ∂W2²/∂p[i] = Σ_j C[i,j] * T[i,j] / p[i]

    where T is the optimal transport plan.

    Args:
        pr_calc: Calculated P(r), normalized [N_bins]
        pr_exp: Experimental P(r), normalized [N_bins]
        transport_plan: Optimal transport plan [N_bins, N_bins]
        cost_matrix: Cost matrix [N_bins, N_bins]
        dr: Bin width

    Returns:
        gradient: ∂W2²/∂p_calc [N_bins]
    """
    # Gradient formula: ∂W2²/∂p[i] = Σ_j C[i,j] * T[i,j] / p[i]
    gradient = (cost_matrix * transport_plan).sum(dim=1) / (pr_calc + 1e-10)

    # Account for normalization
    integral = pr_calc.sum() * dr
    gradient = gradient * dr / (integral + 1e-8)

    return gradient


def compute_peak_gradient(
    pr_calc: torch.Tensor,   # [N_bins]
    r_grid: torch.Tensor,    # [N_bins]
    r_peak_calc: torch.Tensor,
    r_peak_exp: torch.Tensor,
    temperature: float,
    dr: float
) -> torch.Tensor:
    """
    Compute gradient of peak location penalty using soft-argmax.

    The peak penalty is:
        L_peak = (r_peak_calc - r_peak_exp)²

    where r_peak_calc = Σ_i w_i * r_i with w_i = softmax(P(r)/T).

    Gradient:
        ∂L_peak/∂P(r)[i] = 2*(r_peak_calc - r_peak_exp) * ∂r_peak_calc/∂P(r)[i]
                         = 2*(r_peak_calc - r_peak_exp) * (r_i - r_peak_calc) * w_i / T

    Args:
        pr_calc: Calculated P(r) [N_bins]
        r_grid: Distance grid [N_bins]
        r_peak_calc: Calculated peak location (soft-argmax)
        r_peak_exp: Experimental peak location (soft-argmax)
        temperature: Temperature for soft-argmax
        dr: Bin width

    Returns:
        gradient: ∂L_peak/∂P(r) [N_bins]
    """
    # Soft-argmax weights
    weights = torch.nn.functional.softmax(pr_calc / temperature, dim=0)

    # Gradient of soft-argmax peak location
    # ∂r_peak/∂P(r)[i] = (r_i - r_peak) * w_i / T
    dr_peak_dpr = (r_grid - r_peak_calc) * weights / temperature

    # Gradient of squared loss
    gradient = 2 * (r_peak_calc - r_peak_exp) * dr_peak_dpr * dr

    return gradient


def compute_rg_gradient(
    pr_calc: torch.Tensor,   # [N_bins]
    r_grid: torch.Tensor,    # [N_bins]
    rg_calc: torch.Tensor,
    rg_exp: torch.Tensor,
    dr: float
) -> torch.Tensor:
    """
    Compute gradient of Rg penalty.

    The Rg penalty is:
        L_Rg = (Rg_calc - Rg_exp)²

    where Rg_calc = sqrt(Σ r² P(r) dr / Σ P(r) dr).

    For normalized P(r) (Σ P(r) dr = 1), this simplifies to:
        Rg = sqrt(Σ r² P(r) dr)

    Gradient:
        ∂Rg/∂P(r)[i] = r_i² / (2*Rg*integral_pr) - Rg*integral_r2pr / (2*integral_pr²)
                      = (r_i² - Rg²) / (2*Rg*integral_pr)

    For normalized P(r), this simplifies to:
        ∂Rg/∂P(r)[i] = (r_i² - Rg²) / (2*Rg)

    Args:
        pr_calc: Calculated P(r) [N_bins]
        r_grid: Distance grid [N_bins]
        rg_calc: Calculated Rg
        rg_exp: Experimental Rg
        dr: Bin width

    Returns:
        gradient: ∂L_Rg/∂P(r) [N_bins]
    """
    integral_pr = torch.sum(pr_calc * dr)

    # ∂Rg/∂P(r)[i] = (r²[i] - Rg²) / (2*Rg*integral_pr)
    grad_rg_pr = (r_grid ** 2 - rg_calc ** 2) / (2 * rg_calc * integral_pr + 1e-10)

    # ∂L/∂P(r) = 2*(Rg_calc - Rg_exp) * ∂Rg/∂P(r)
    gradient = 2 * (rg_calc - rg_exp) * grad_rg_pr * dr

    return gradient


class SAXSPrPotential(Potential):
    """
    Potential for SAXS P(r) curve fitting using ensemble averaging.

    Computes ensemble-averaged P(r) from all diffusion samples and fits to
    experimental data using Wasserstein-1 distance.

    Key differences from standard potentials:
        - Operates on ALL samples simultaneously (ensemble average)
        - Uses soft binning for differentiable histogram computation
        - Wasserstein-1 loss for robust P(r) matching
        - Single scalar loss per batch (not per-sample)
        - Gradients broadcast back to all samples

    Note: Each instance maintains its own cache since different P(r) files
        have different r_grid binning parameters. A class-level step tracker
        coordinates cache invalidation across instances.
    """

    # Class-level step tracker for coordinating cache invalidation
    _global_cache_step = -1

    def __init__(self, parameters=None):
        super().__init__(parameters)
        # Instance-level cache for P(r) computation
        # Each SAXS potential has its own cache since r_grid/sigma_bin may differ
        self._pr_cache = {}
        self._cache_step = -1  # Track which step the cache is valid for

    @classmethod
    def clear_cache(cls, step: int = -1):
        """Signal cache invalidation. Called at start of each diffusion step.

        This sets the global step counter. Each instance will check this
        and clear its own cache if the step has changed.
        """
        cls._global_cache_step = step

    def _check_cache_validity(self):
        """Check if instance cache is still valid, clear if step changed."""
        if self._cache_step != SAXSPrPotential._global_cache_step:
            self._pr_cache = {}
            self._cache_step = SAXSPrPotential._global_cache_step

    def get_cached_pr(self, cache_key):
        """Get cached P(r) and gradients if available."""
        self._check_cache_validity()
        return self._pr_cache.get(cache_key)

    def set_cached_pr(self, cache_key, pr_ensemble, grad_pr):
        """Cache P(r) and gradients."""
        self._check_cache_validity()
        self._pr_cache[cache_key] = (pr_ensemble, grad_pr)

    def compute_variable(
        self,
        coords,  # [multiplicity, N_atoms, 3]
        index,   # [1, N_CA] - indices of CA atoms (not used in atomistic mode)
        ref_coords=None,
        ref_mask=None,
        compute_gradient=False
    ):
        """
        Compute ensemble-averaged P(r) histogram from all-atom distances.

        Process:
            1. Use all atom coordinates: [multiplicity, N_atoms, 3]
            2. Compute pairwise distances within each sample
            3. Soft-bin distances into P(r) bins
            4. Average across multiplicity
            5. Normalize to match experimental integral

        Args:
            coords: All atom coordinates, shape [multiplicity, N_atoms, 3]
            index: CA atom indices (kept for API compatibility, not used)
            compute_gradient: Whether to compute gradients

        Returns:
            If compute_gradient=False:
                pr_ensemble: Ensemble-averaged P(r), shape [N_bins]
            If compute_gradient=True:
                (pr_ensemble, grad): Tuple of P(r) and gradient
                grad shape: [multiplicity, N_atoms, N_bins, 3]
        """
        # Check if we should use rep atoms only (much faster for large proteins)
        use_rep_atoms = getattr(self, '_use_rep_atoms', False)

        if use_rep_atoms and index is not None and index.shape[1] > 0:
            # Use CA atoms only - select from full coords using index
            # index has shape [1, N_CA], so we use index[0]
            atom_coords = coords[:, index[0], :]  # [multiplicity, N_CA, 3]
        else:
            # Use all atoms (atomistic mode)
            atom_coords = coords  # [multiplicity, N_atoms, 3]

        multiplicity, n_atoms, _ = atom_coords.shape

        # Use memory-efficient chunked computation for large systems
        # For all-atom mode, memory scales as O(n_atoms^2 * n_bins) which explodes quickly:
        #   - 500 atoms: ~250k pairs -> manageable
        #   - 1000 atoms: ~500k pairs -> borderline
        #   - 2000 atoms: ~2M pairs -> ~800MB for weights tensor alone
        #   - 3000 atoms: ~4.5M pairs -> OOM
        # Use chunked mode for > 500 atoms OR when multiplicity > 1 (multiple samples)
        # The weights tensor scales as O(multiplicity * n_pairs * n_bins)
        if n_atoms > 500 or (multiplicity > 1 and n_atoms > 300):
            result = self._compute_variable_chunked(coords, index, ref_coords, ref_mask, use_rep_atoms)
            # Chunked always returns (pr, grad) - extract just pr if gradient not requested
            if compute_gradient:
                return result
            else:
                return result[0]  # Just P(r)

        # Get bin parameters from stored experimental data
        # Move to same device as coords
        r_grid = self._r_grid.to(coords.device)        # [N_bins]
        sigma_bin = self._sigma_bin  # Gaussian width for soft binning

        n_bins = r_grid.shape[0]

        # Compute all pairwise atom-atom distances
        atom_i = atom_coords.unsqueeze(2)  # [multiplicity, N_atoms, 1, 3]
        atom_j = atom_coords.unsqueeze(1)  # [multiplicity, 1, N_atoms, 3]

        r_vec = atom_i - atom_j  # [multiplicity, N_atoms, N_atoms, 3]
        distances = torch.linalg.norm(r_vec, dim=-1)  # [multiplicity, N_atoms, N_atoms]

        # Extract upper triangle (avoid double counting and i=j)
        idx_i, idx_j = torch.triu_indices(n_atoms, n_atoms, offset=1, device=coords.device)
        distances_pairs = distances[:, idx_i, idx_j]  # [multiplicity, N_pairs]
        n_pairs = distances_pairs.shape[1]

        # Soft binning using Gaussian kernels
        dist_expanded = distances_pairs.unsqueeze(-1)  # [multiplicity, N_pairs, 1]
        r_grid_expanded = r_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N_bins]

        # Gaussian kernel: exp(-0.5 * ((d - r) / sigma)^2)
        diff = dist_expanded - r_grid_expanded  # [multiplicity, N_pairs, N_bins]
        weights = torch.exp(-0.5 * (diff / sigma_bin) ** 2)

        # Sum contributions from all pairs to get P(r) histogram per sample
        pr_per_sample = weights.sum(dim=1)  # [multiplicity, N_bins]

        # Ensemble average across samples
        pr_ensemble = pr_per_sample.mean(dim=0)  # [N_bins]

        # Normalize to match experimental integral
        integral = torch.trapz(pr_ensemble, r_grid)
        pr_ensemble_normalized = pr_ensemble / (integral + 1e-8)

        if not compute_gradient:
            return pr_ensemble_normalized

        # === Gradient Computation ===
        # Compute ∂(Gaussian weight)/∂d for each distance-bin pair
        dweights_dd = -(diff / (sigma_bin ** 2)) * weights  # [multiplicity, N_pairs, N_bins]

        # Account for ensemble averaging
        if getattr(self, '_per_sample_steering', False):
            # Per-sample steering: Weight gradients by deviation from ensemble
            # Samples that deviate more from ensemble → larger gradient (more steering)
            # Samples close to ensemble → smaller gradient (less steering)
            # This allows some samples to be compact, others extended

            # pr_per_sample: [multiplicity, N_bins]
            # pr_ensemble: [N_bins]
            pr_deviation = pr_per_sample - pr_ensemble.unsqueeze(0)  # [mult, N_bins]
            deviation_magnitude = pr_deviation.abs().sum(dim=-1, keepdim=True)  # [mult, 1]

            # Normalize: samples with larger deviation get larger gradients
            weights_per_sample = deviation_magnitude / (deviation_magnitude.mean() + 1e-8)  # [mult, 1]

            # Apply to gradient (broadcast over pairs and bins)
            dpr_ensemble_dd = dweights_dd * weights_per_sample.unsqueeze(1)  # [mult, N_pairs, N_bins]
        else:
            # Original behavior: same gradient to all samples
            # NOTE: We broadcast the full ensemble gradient to all samples (no division by multiplicity)
            # This ensures steering strength is independent of number of samples
            dpr_ensemble_dd = dweights_dd  # [multiplicity, N_pairs, N_bins]

        # Account for normalization (simplified: assume integral ≈ constant)
        dpr_norm_dd = dpr_ensemble_dd / (integral + 1e-8)

        # Compute ∂d_ij/∂coords for each atom
        r_vec_pairs = r_vec[:, idx_i, idx_j, :]  # [multiplicity, N_pairs, 3]
        dd_dr_i = r_vec_pairs / (distances_pairs.unsqueeze(-1) + 1e-8)  # [mult, N_pairs, 3]
        dd_dr_j = -dd_dr_i

        # Final gradient: ∂pr_norm/∂r_i = sum_k (∂pr_norm[k]/∂d_ij) * (∂d_ij/∂r_i)
        dd_dr_i_expanded = dd_dr_i.unsqueeze(2)  # [mult, N_pairs, 1, 3]
        dpr_norm_dd_expanded = dpr_norm_dd.unsqueeze(-1)  # [mult, N_pairs, N_bins, 1]

        dpr_dr_pairs_i = dpr_norm_dd_expanded * dd_dr_i_expanded  # [mult, N_pairs, N_bins, 3]
        dpr_dr_pairs_j = dpr_norm_dd_expanded * dd_dr_j.unsqueeze(2)  # [mult, N_pairs, N_bins, 3]

        # Scatter gradients back to atoms used in P(r) computation
        grad_atoms_local = torch.zeros(
            multiplicity, n_atoms, n_bins, 3,
            device=coords.device, dtype=coords.dtype
        )

        # Add contributions from pairs where atom is i (first in pair)
        grad_atoms_local.index_add_(
            1,  # dimension to scatter along (atoms)
            idx_i,  # indices
            dpr_dr_pairs_i  # values to add
        )

        # Add contributions from pairs where atom is j (second in pair)
        grad_atoms_local.index_add_(
            1,
            idx_j,
            dpr_dr_pairs_j
        )

        # If using CA-only mode, scatter back to full atom positions
        if use_rep_atoms and self._ca_indices is not None:
            n_all_atoms = self._n_atoms
            grad_atoms = torch.zeros(
                multiplicity, n_all_atoms, n_bins, 3,
                device=coords.device, dtype=coords.dtype
            )
            # Scatter CA gradients to their positions in full array
            grad_atoms[:, self._ca_indices, :, :] = grad_atoms_local
        else:
            grad_atoms = grad_atoms_local

        return pr_ensemble_normalized, grad_atoms

    def _compute_variable_chunked(
        self,
        coords,  # [multiplicity, N_atoms, 3]
        index,
        ref_coords=None,
        ref_mask=None,
        use_rep_atoms=False,
    ):
        """
        Memory-efficient version using atom chunking and vectorized sparse binning.

        Optimizations:
        1. Uses torch.cdist for efficient distance computation
        2. Processes atom pairs in chunks to limit peak memory
        3. Vectorized sparse binning: no Python loops over bin offsets
        4. Single pass: computes P(r) and gradients together
        """
        # Check if we should use representative atoms only
        if use_rep_atoms and index is not None and index.shape[1] > 0:
            atom_coords = coords[:, index[0], :]  # [multiplicity, N_rep, 3]
        else:
            atom_coords = coords  # [multiplicity, N_atoms, 3]

        multiplicity, n_atoms_used, _ = atom_coords.shape

        r_grid = self._r_grid.to(coords.device)
        sigma_bin = self._sigma_bin
        n_bins = r_grid.shape[0]
        bin_width = r_grid[1] - r_grid[0] if n_bins > 1 else sigma_bin
        r_min = r_grid[0].item()

        # Sparse binning: only consider bins within 4σ of each distance
        n_sigma = 4
        bins_per_distance = int(2 * n_sigma * sigma_bin / bin_width) + 2

        # Pre-compute offset tensor for vectorized binning
        offsets = torch.arange(-bins_per_distance // 2, bins_per_distance // 2 + 1,
                               device=coords.device, dtype=torch.long)  # [n_offsets]
        n_offsets = offsets.shape[0]

        # Atom chunk size
        chunk_size = min(500, n_atoms_used)

        # Single pass: compute P(r) and gradients together
        pr_per_sample = torch.zeros(multiplicity, n_bins, device=coords.device, dtype=coords.dtype)
        grad_atoms_local = torch.zeros(multiplicity, n_atoms_used, n_bins, 3,
                                       device=coords.device, dtype=coords.dtype)

        for m in range(multiplicity):
            sample_coords = atom_coords[m]  # [N_atoms_used, 3]

            # Process upper triangle in chunks
            for i_start in range(0, n_atoms_used, chunk_size):
                i_end = min(i_start + chunk_size, n_atoms_used)
                coords_i = sample_coords[i_start:i_end]  # [chunk_i, 3]
                n_i = i_end - i_start

                for j_start in range(i_start, n_atoms_used, chunk_size):
                    j_end = min(j_start + chunk_size, n_atoms_used)
                    coords_j = sample_coords[j_start:j_end]  # [chunk_j, 3]
                    n_j = j_end - j_start

                    # Compute distance vectors and distances
                    r_vec = coords_i.unsqueeze(1) - coords_j.unsqueeze(0)  # [chunk_i, chunk_j, 3]
                    distances_chunk = torch.linalg.norm(r_vec, dim=-1)  # [chunk_i, chunk_j]

                    # Handle diagonal blocks (only upper triangle)
                    if i_start == j_start:
                        mask = torch.triu(torch.ones(n_i, n_j, dtype=torch.bool, device=coords.device), diagonal=1)
                    elif i_start < j_start:
                        mask = torch.ones(n_i, n_j, dtype=torch.bool, device=coords.device)
                    else:
                        continue

                    if not mask.any():
                        continue

                    # Extract valid pairs
                    local_i, local_j = torch.where(mask)
                    n_pairs = local_i.shape[0]
                    distances_flat = distances_chunk[mask]  # [n_pairs]
                    r_vec_flat = r_vec[local_i, local_j]  # [n_pairs, 3]

                    # Convert to global indices
                    global_i = local_i + i_start
                    global_j = local_j + j_start

                    # Vectorized sparse binning: compute all bin contributions at once
                    # bin_indices: [n_pairs] - center bin for each distance
                    bin_indices = ((distances_flat - r_min) / bin_width).long()

                    # target_bins: [n_pairs, n_offsets] - all bins each distance contributes to
                    target_bins = bin_indices.unsqueeze(-1) + offsets.unsqueeze(0)  # [n_pairs, n_offsets]

                    # Validity mask: [n_pairs, n_offsets]
                    valid_mask = (target_bins >= 0) & (target_bins < n_bins)

                    # Get bin centers for all targets: [n_pairs, n_offsets]
                    # Clamp to valid range for indexing, then mask later
                    target_bins_clamped = target_bins.clamp(0, n_bins - 1)
                    bin_centers = r_grid[target_bins_clamped]  # [n_pairs, n_offsets]

                    # Compute Gaussian weights: [n_pairs, n_offsets]
                    diff = distances_flat.unsqueeze(-1) - bin_centers  # [n_pairs, n_offsets]
                    weights = torch.exp(-0.5 * (diff / sigma_bin) ** 2)
                    weights = weights * valid_mask  # Zero out invalid bins

                    # Accumulate P(r) histogram using scatter_add
                    # Flatten for scatter: [n_pairs * n_offsets]
                    flat_bins = target_bins_clamped.flatten()
                    flat_weights = weights.flatten()
                    pr_per_sample[m].index_add_(0, flat_bins, flat_weights)

                    # === Gradient computation ===
                    # Distance gradient: d(d_ij)/d(r_i) = (r_i - r_j) / d_ij
                    dd_dr_i = r_vec_flat / (distances_flat.unsqueeze(-1) + 1e-8)  # [n_pairs, 3]

                    # Gaussian weight derivative: [n_pairs, n_offsets]
                    dweights_dd = -(diff / (sigma_bin ** 2)) * weights

                    # Gradient contribution per (pair, offset): [n_pairs, n_offsets, 3]
                    # dP(r)/dr_i = dw/dd * dd/dr_i
                    grad_contrib = dweights_dd.unsqueeze(-1) * dd_dr_i.unsqueeze(1)  # [n_pairs, n_offsets, 3]

                    # Scatter gradients to atoms
                    # For atom i: flat index = atom_i * n_bins + bin_idx
                    # For atom j: same but with negative gradient

                    # Expand indices for all offsets: [n_pairs, n_offsets]
                    global_i_exp = global_i.unsqueeze(-1).expand(-1, n_offsets)
                    global_j_exp = global_j.unsqueeze(-1).expand(-1, n_offsets)

                    # Flat indices: [n_pairs * n_offsets]
                    flat_idx_i = (global_i_exp * n_bins + target_bins_clamped).flatten()
                    flat_idx_j = (global_j_exp * n_bins + target_bins_clamped).flatten()

                    # Flatten gradients: [n_pairs * n_offsets, 3]
                    grad_contrib_flat = grad_contrib.reshape(-1, 3)

                    # Scatter to gradient tensor
                    grad_flat = grad_atoms_local[m].view(-1, 3)
                    grad_flat.index_add_(0, flat_idx_i, grad_contrib_flat)
                    grad_flat.index_add_(0, flat_idx_j, -grad_contrib_flat)  # Negative for atom j

        # Ensemble average and normalize
        pr_ensemble = pr_per_sample.mean(dim=0)
        integral = torch.trapz(pr_ensemble, r_grid)
        pr_ensemble_normalized = pr_ensemble / (integral + 1e-8)

        # Normalize gradients by integral
        grad_atoms_local = grad_atoms_local / (integral + 1e-8)

        # If using rep atoms only, scatter back to full atom positions
        if use_rep_atoms and self._ca_indices is not None:
            n_all_atoms = self._n_atoms
            grad_atoms = torch.zeros(
                multiplicity, n_all_atoms, n_bins, 3,
                device=coords.device, dtype=coords.dtype
            )
            grad_atoms[:, self._ca_indices, :, :] = grad_atoms_local
        else:
            grad_atoms = grad_atoms_local

        return pr_ensemble_normalized, grad_atoms

    def compute_function(
        self,
        pr_calc,  # [N_bins] - calculated ensemble P(r)
        pr_exp,   # [N_bins] - experimental P(r)
        r_grid,   # [N_bins] - distance grid
        k,        # scalar - force constant
        negation_mask=None,
        compute_derivative=False,
        loss_type='w2_mse'  # 'w1', 'w2', 'mse', 'chi2', 'mae', 'kl', 'w2_mse'
    ):
        """
        Compute loss between calculated and experimental P(r).

        Available loss functions:
        - 'w1': Wasserstein-1 (Earth Mover's Distance) - fast, CDF matching
        - 'w2': Wasserstein-2 (Sinkhorn optimal transport) - geometric matching
        - 'mse': Mean Squared Error - point-by-point matching
        - 'chi2': Chi-squared - error-weighted matching
        - 'mae': Mean Absolute Error - robust to outliers
        - 'kl': KL divergence - information-theoretic
        - 'w2_mse': W2 + MSE combined (default, best for IDPs)

        Args:
            pr_calc: Calculated ensemble P(r), shape [N_bins]
            pr_exp: Experimental P(r), shape [N_bins]
            r_grid: Distance grid, shape [N_bins]
            k: Force constant (scalar)
            compute_derivative: Whether to compute derivative
            loss_type: Type of loss function to use

        Returns:
            If compute_derivative=False:
                energy: Scalar loss value
            If compute_derivative=True:
                (energy, dEnergy_dPr): Energy and derivative w.r.t. P(r)
                dEnergy_dPr shape: [N_bins]
        """
        dr = r_grid[1] - r_grid[0]

        # Backward compatibility aliases
        if loss_type == 'wasserstein':
            loss_type = 'w1'
        elif loss_type == 'wasserstein2':
            loss_type = 'w2'

        if loss_type == 'w1':
            # Wasserstein-1 distance: W1 = ∫|CDF_calc - CDF_exp| dr
            cdf_calc = torch.cumsum(pr_calc * dr, dim=0)
            cdf_exp = torch.cumsum(pr_exp * dr, dim=0)
            cdf_diff = cdf_calc - cdf_exp
            loss = torch.abs(cdf_diff).sum() * dr
            energy = k * loss

            if not compute_derivative:
                return energy

            # Gradient: dW1/dPr = dr × reverse_cumsum(sign(CDF_diff))
            sign_cdf_diff = torch.sign(cdf_diff)
            sign_cumsum_reverse = torch.flip(
                torch.cumsum(torch.flip(sign_cdf_diff, [0]), dim=0),
                [0]
            )
            dLoss_dPr = dr * sign_cumsum_reverse
            dEnergy_dPr = k * dLoss_dPr

        elif loss_type == 'cramer':
            # Cramér distance (squared W1): ∫(CDF_calc - CDF_exp)² dr
            # Unlike W1, gradient is proportional to error magnitude
            cdf_calc = torch.cumsum(pr_calc * dr, dim=0)
            cdf_exp = torch.cumsum(pr_exp * dr, dim=0)
            cdf_diff = cdf_calc - cdf_exp
            loss = (cdf_diff ** 2).sum() * dr
            energy = k * loss

            if not compute_derivative:
                return energy

            # Gradient: d(Cramér)/dPr[k] = 2 * dr² * Σ_{i≥k} cdf_diff[i]
            # This is the reverse cumsum of cdf_diff (not sign!)
            cdf_diff_cumsum_reverse = torch.flip(
                torch.cumsum(torch.flip(cdf_diff, [0]), dim=0),
                [0]
            )
            dLoss_dPr = 2.0 * dr * dr * cdf_diff_cumsum_reverse
            dEnergy_dPr = k * dLoss_dPr

        elif loss_type == 'mse':
            # Mean Squared Error: MSE = Σ(P_calc - P_exp)²
            diff = pr_calc - pr_exp
            loss = (diff ** 2).sum() * dr  # Weighted by dr for proper integration
            energy = k * loss

            if not compute_derivative:
                return energy

            # Gradient: dMSE/dPr = 2(P_calc - P_exp)
            dLoss_dPr = 2.0 * diff * dr
            dEnergy_dPr = k * dLoss_dPr

        elif loss_type == 'chi2':
            # Chi-squared: χ² = Σ[(P_calc - P_exp)² / P_exp]
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            diff = pr_calc - pr_exp
            loss = ((diff ** 2) / (pr_exp + eps)).sum() * dr
            energy = k * loss

            if not compute_derivative:
                return energy

            # Gradient: dχ²/dPr = 2(P_calc - P_exp) / P_exp
            dLoss_dPr = 2.0 * diff / (pr_exp + eps) * dr
            dEnergy_dPr = k * dLoss_dPr

        elif loss_type == 'mae':
            # Mean Absolute Error: MAE = Σ|P_calc - P_exp|
            diff = pr_calc - pr_exp
            loss = torch.abs(diff).sum() * dr
            energy = k * loss

            if not compute_derivative:
                return energy

            # Gradient: dMAE/dPr = sign(P_calc - P_exp)
            dLoss_dPr = torch.sign(diff) * dr
            dEnergy_dPr = k * dLoss_dPr

        elif loss_type == 'kl':
            # KL divergence: D_KL(P_calc || P_exp) = Σ P_calc log(P_calc/P_exp)
            # Add small epsilon to avoid log(0) and division by zero
            eps = 1e-8
            pr_calc_safe = pr_calc + eps
            pr_exp_safe = pr_exp + eps

            # KL divergence
            loss = (pr_calc_safe * torch.log(pr_calc_safe / pr_exp_safe)).sum() * dr
            energy = k * loss

            if not compute_derivative:
                return energy

            # Gradient: dKL/dP_calc = log(P_calc/P_exp) + 1
            dLoss_dPr = (torch.log(pr_calc_safe / pr_exp_safe) + 1.0) * dr
            dEnergy_dPr = k * dLoss_dPr

        elif loss_type == 'w2':
            # Wasserstein-2 distance via Sinkhorn algorithm
            # W2² = min_T Σ_ij C_ij T_ij where C_ij = (r_i - r_j)²
            dr = r_grid[1] - r_grid[0]

            # Cost matrix: C[i,j] = (r[i] - r[j])²
            cost_matrix = (r_grid.unsqueeze(0) - r_grid.unsqueeze(1)) ** 2

            # Entropic regularization parameters
            epsilon = self.parameters.get('w2_epsilon', 0.1)
            num_iter = self.parameters.get('w2_num_iter', 100)

            # Normalize distributions
            pr_calc_norm = pr_calc * dr / (pr_calc.sum() * dr + 1e-8)
            pr_exp_norm = pr_exp * dr / (pr_exp.sum() * dr + 1e-8)

            # Compute W2²
            loss, transport_plan = sinkhorn_distance(
                pr_calc_norm, pr_exp_norm, cost_matrix, epsilon,
                num_iter=num_iter, return_plan=True
            )

            energy = k * loss

            if not compute_derivative:
                return energy

            # Gradient via implicit differentiation
            dLoss_dPr = compute_w2_gradient(
                pr_calc_norm, pr_exp_norm, transport_plan, cost_matrix, dr
            )
            dEnergy_dPr = k * dLoss_dPr

        elif loss_type == 'rg':
            # Rg penalty only - per-ensemble Rg matching from P(r)
            # Computes Rg from P(r) and penalizes deviation from experimental Rg
            dr = r_grid[1] - r_grid[0]

            # Compute Rg from calculated P(r)
            rg_calc = torch.sqrt(
                torch.sum(r_grid ** 2 * pr_calc * dr) / (2.0 * pr_calc.sum() * dr + 1e-8)
            )

            # Compute Rg from experimental P(r)
            rg_exp = torch.sqrt(
                torch.sum(r_grid ** 2 * pr_exp * dr) / (2.0 * pr_exp.sum() * dr + 1e-8)
            )

            # Apply rg_scale to experimental Rg
            rg_scale = self.parameters.get('rg_scale', 1.0)
            rg_target = rg_exp * rg_scale

            # Squared deviation loss
            loss = (rg_calc - rg_target) ** 2
            energy = k * loss

            if not compute_derivative:
                return energy

            # Gradient of Rg loss w.r.t. P(r)
            dLoss_dPr = compute_rg_gradient(pr_calc, r_grid, rg_calc, rg_target, dr)
            dEnergy_dPr = k * dLoss_dPr

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        return energy, dEnergy_dPr

    def compute_args(self, feats, parameters):
        """
        Extract CA atoms and prepare experimental P(r) data.

        Args:
            feats: Feature dictionary containing atom information
            parameters: Dict with:
                - 'pr_exp': Experimental P(r) values [N_bins]
                - 'r_grid': Distance grid [N_bins]
                - 'k': Force constant
                - 'sigma_bin': Gaussian width for soft binning

        Returns:
            Tuple of (index, args, com_args, ref_args, operator_args)
        """
        atom_pad_mask = feats["atom_pad_mask"][0].bool()

        # Identify CA atoms
        ca_mask = self._identify_ca_atoms(feats)
        ca_mask = ca_mask & atom_pad_mask  # Intersect with valid atoms

        # Get indices of CA atoms
        ca_indices = torch.arange(
            atom_pad_mask.shape[0], dtype=torch.long, device=atom_pad_mask.device
        )[ca_mask]

        # Reshape to [1, N_CA]
        index = ca_indices.unsqueeze(0)

        # Store bin parameters for use in compute_variable
        # Move r_grid to the same device as the features
        self._r_grid = parameters['r_grid'].to(atom_pad_mask.device)
        self._sigma_bin = parameters['sigma_bin']
        self._per_sample_steering = parameters.get('per_sample_steering', False)
        self._use_rep_atoms = parameters.get('use_rep_atoms', False)
        self._ca_indices = index[0] if index is not None else None  # Store for gradient scattering
        self._n_atoms = atom_pad_mask.shape[0]  # Store total atom count for gradient output

        # Prepare arguments for compute_function
        # Move experimental data to the same device as features
        pr_exp = parameters['pr_exp'].to(atom_pad_mask.device)
        r_grid = parameters['r_grid'].to(atom_pad_mask.device)
        k = parameters['k']
        loss_type = parameters.get('loss_type', 'wasserstein')

        return index, (pr_exp, r_grid, k, loss_type), None, None, None

    def _identify_ca_atoms(self, feats):
        """
        Identify CA (alpha-carbon) atoms from features.

        Uses token_to_rep_atom to identify representative atoms (typically CA).

        Args:
            feats: Feature dictionary

        Returns:
            Boolean mask [N_atoms] indicating CA atoms
        """
        # Use token_to_rep_atom to get representative atoms (typically CA)
        token_to_rep = feats.get('token_to_rep_atom')
        if token_to_rep is not None:
            # Get indices of rep atoms
            rep_indices = token_to_rep[0].argmax(dim=0)  # [N_tokens]
            ca_mask = torch.zeros(feats['atom_pad_mask'].shape[1],
                                dtype=torch.bool,
                                device=feats['atom_pad_mask'].device)
            ca_mask[rep_indices] = True
            return ca_mask

        # Fallback: raise error if can't identify
        raise NotImplementedError(
            "CA atom identification requires token_to_rep_atom feature"
        )

    def compute_gradient(self, coords, feats, parameters, step: int = 0):
        """
        Override base class gradient computation for vector-valued CV.

        For P(r) potential with vector CV [N_bins]:
            grad_value: [multiplicity, N_atoms, N_bins, 3]
            dEnergy: [N_bins]
            result: sum_k dEnergy[k] * grad_value[:, :, k, :] → [multiplicity, N_atoms, 3]

        Uses class-level caching to share P(r) computation across multiple
        SAXS potentials with different loss types.
        """
        # Get step index from parameters (passed by diffusionv2) or use the step argument
        step_idx = parameters.get('_step_idx', step)

        # Clear cache at start of each new step (MUST be before early returns!)
        # This frees the cached grad_pr tensor (~27MB for 8 samples) from previous step
        SAXSPrPotential.clear_cache(step_idx)


        # Check warmup and cutoff (0.0 = start of diffusion, 1.0 = end)
        # SAXS steering is active when: warmup <= progress <= cutoff
        progress = parameters.get('_relaxation', 0.0)
        warmup = parameters.get('warmup', 0.0)
        cutoff = parameters.get('cutoff', 0.9)
        if progress < warmup or progress > cutoff:
            return torch.zeros_like(coords)

        index, args, com_args, ref_args, operator_args = self.compute_args(
            feats, parameters
        )

        if index.shape[1] == 0:
            return torch.zeros_like(coords)

        # Check cache for P(r) computation
        # The cache is cleared at the start of each diffusion step in diffusionv2.py
        # so we only need to distinguish between rep_atoms and all-atoms mode
        use_rep_atoms = getattr(self, '_use_rep_atoms', False)
        cache_key = use_rep_atoms

        cached = self.get_cached_pr(cache_key)
        if cached is not None:
            pr_ensemble, grad_pr = cached
        else:
            # Compute P(r) and gradients (atomistic mode - all atoms)
            pr_ensemble, grad_pr = self.compute_variable(
                coords, index, compute_gradient=True
            )
            # Cache for this potential instance
            self.set_cached_pr(cache_key, pr_ensemble, grad_pr)

        # pr_ensemble: [N_bins]
        # grad_pr: [multiplicity, N_atoms, N_bins, 3]

        # Compute energy and derivative w.r.t. P(r)
        pr_exp, r_grid, k, loss_type = args
        energy, dEnergy_dPr = self.compute_function(
            pr_ensemble, pr_exp, r_grid, k, compute_derivative=True, loss_type=loss_type
        )
        # dEnergy_dPr: [N_bins]

        # Contract over bins: grad = sum_k (dE/dPr[k]) * (dPr[k]/dr)
        # This is already in full atom space, no scattering needed
        grad_atom = torch.einsum('mnkd,k->mnd', grad_pr, dEnergy_dPr)

        # Apply gradient scaling if scaler is set
        if self.gradient_scaler is not None:
            grad_atom = self.gradient_scaler.apply(grad_atom, coords, feats, step, progress)

        # Normalize gradient (same pattern as CVs)
        grad_norms = grad_atom.norm(dim=-1, keepdim=True)
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            grad_atom = grad_atom / max_norm

        return grad_atom


def get_potentials(steering_args, boltz2=False, feats=None, debug=False):
    potentials = []
    if debug:
        print("[DEBUG] get_potentials called")
        print(f"  fk_steering: {steering_args.get('fk_steering')}")
        print(f"  physical_guidance_update: {steering_args.get('physical_guidance_update')}")
    if steering_args["fk_steering"] or steering_args["physical_guidance_update"]:
        potentials.extend(
            [
                SymmetricChainCOMPotential(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": 0.5
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.5,
                        "buffer": ExponentialInterpolation(
                            start=1.0, end=5.0, alpha=-2.0
                        ),
                    }
                ),
                VDWOverlapPotential(
                    parameters={
                        "guidance_interval": 5,
                        "guidance_weight": (
                            PiecewiseStepFunction(thresholds=[0.4], values=[0.125, 0.0])
                            if steering_args["physical_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": PiecewiseStepFunction(
                            thresholds=[0.6], values=[0.01, 0.0]
                        ),
                        "buffer": 0.225,
                    }
                ),
                ConnectionsPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.15
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 2.0,
                    }
                ),
                PoseBustersPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.01
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 0.1,
                        "bond_buffer": 0.125,
                        "angle_buffer": 0.125,
                        "clash_buffer": 0.10,
                    }
                ),
                ChiralAtomPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.1
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                StereoBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.52360,
                    }
                ),
                PlanarBondPotential(
                    parameters={
                        "guidance_interval": 1,
                        "guidance_weight": 0.05
                        if steering_args["physical_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                        "buffer": 0.26180,
                    }
                ),
            ]
        )
    if boltz2 and (
        steering_args["fk_steering"] or steering_args["contact_guidance_update"]
    ):
        potentials.extend(
            [
                ContactPotentital(
                    parameters={
                        "guidance_interval": 4,
                        "guidance_weight": (
                            PiecewiseStepFunction(
                                thresholds=[0.25, 0.75], values=[0.0, 0.5, 1.0]
                            )
                            if steering_args["contact_guidance_update"]
                            else 0.0
                        ),
                        "resampling_weight": 1.0,
                        "union_lambda": ExponentialInterpolation(
                            start=8.0, end=0.0, alpha=-2.0
                        ),
                    }
                ),
                TemplateReferencePotential(
                    parameters={
                        "guidance_interval": 2,
                        "guidance_weight": 0.1
                        if steering_args["contact_guidance_update"]
                        else 0.0,
                        "resampling_weight": 1.0,
                    }
                ),
            ]
        )

    # Legacy rg_steering support: convert to a steering_configs entry so it
    # goes through the modern HarmonicSteeringPotential path.
    if steering_args.get("rg_steering", False):
        rg_params = steering_args.get("rg_params") or {}
        target_rg = None
        target_from_saxs = rg_params.get("target_from_saxs")
        saxs_data_cache = steering_args.get("saxs_pr_data_cache") or {}

        if target_from_saxs and target_from_saxs in saxs_data_cache:
            saxs_data = saxs_data_cache[target_from_saxs]
            r_grid = saxs_data['r_grid']
            pr_exp = saxs_data['pr_exp']
            dr = r_grid[1] - r_grid[0]
            rg_exp = torch.sqrt(
                torch.sum(r_grid ** 2 * pr_exp * dr) / (2.0 * pr_exp.sum() * dr + 1e-8)
            ).item()
            auto_rg_scale = rg_params.get("auto_rg_scale", 1.0)
            target_rg = rg_exp * auto_rg_scale
        else:
            target_rg = rg_params.get("target_rg") or steering_args.get("rg_target", 20.0)

        # Inject as a steering_configs entry so it uses the modern code path
        legacy_rg_config = {
            "collective_variable": "rg",
            "target": target_rg,
            "strength": rg_params.get("k") or steering_args.get("rg_k", 1.0),
            "guidance_interval": rg_params.get("guidance_interval") or steering_args.get("rg_guidance_interval", 1),
            "warmup": 0.0,
            "cutoff": 0.75,
            "groups": rg_params.get("chain_ids") or steering_args.get("rg_chain_ids"),
        }
        existing = steering_args.get("steering_configs") or []
        existing.append(legacy_rg_config)
        steering_args["steering_configs"] = existing

    # Add SAXS P(r) potentials (multiple entries supported)
    saxs_configs = steering_args.get("saxs_configs") or []
    saxs_data_cache = steering_args.get("saxs_pr_data_cache") or {}

    # Import factory functions for gradient modifier creation (lazy to avoid circular import)
    from boltz.model.potentials.factory import create_gradient_modifier
    from boltz.data.parse.metadiffusion import parse_scaling_config, parse_projection_config

    for saxs_config in saxs_configs:
        pr_file = saxs_config.get("pr_file")
        if pr_file and pr_file in saxs_data_cache:
            saxs_data = saxs_data_cache[pr_file]
            bin_width = saxs_data['r_grid'][1] - saxs_data['r_grid'][0]
            sigma_bin = saxs_config.get("sigma_bin", 0.5) * bin_width

            potential = SAXSPrPotential(
                parameters={
                    "guidance_interval": saxs_config.get("guidance_interval", 1),
                    "guidance_weight": saxs_config.get("guidance_weight", 1.0),
                    "resampling_weight": 0.0,
                    "pr_exp": saxs_data['pr_exp'],
                    "r_grid": saxs_data['r_grid'],
                    "k": saxs_config.get("strength", 1.0),
                    "warmup": saxs_config.get("warmup", 0.0),
                    "cutoff": saxs_config.get("cutoff", 0.9),
                    "sigma_bin": sigma_bin,
                    "loss_type": saxs_config.get("loss_type", "mse"),
                    "w2_epsilon": saxs_config.get("w2_epsilon", 0.1),
                    "w2_num_iter": saxs_config.get("w2_num_iter", 100),
                    "rg_scale": saxs_config.get("rg_scale", 1.0),
                    "use_rep_atoms": saxs_config.get("use_rep_atoms", False),
                    "bias_tempering": saxs_config.get("bias_clip"),
                }
            )

            # Attach gradient modifier if scaling and/or projection config is present
            scaling_config = saxs_config.get("scaling")
            projection_config = saxs_config.get("projection")
            if scaling_config or projection_config:
                # Parse raw dicts into config objects
                scaling_configs = parse_scaling_config(scaling_config) if scaling_config else None
                projection_configs = parse_projection_config(projection_config) if projection_config else None
                modifier = create_gradient_modifier(
                    scaling_configs=scaling_configs,
                    projection_configs=projection_configs,
                    modifier_order=saxs_config.get("modifier_order", "scale_first"),
                    feats=feats,
                )
                if modifier is not None:
                    potential.gradient_scaler = modifier

            potentials.append(potential)

    # Add metadynamics potential if requested (lazy import to avoid circular dependency)
    if steering_args.get("metadynamics", False):
        try:
            from boltz.model.potentials.metadynamics import MetadynamicsPotential
            from boltz.model.potentials.collective_variables import create_cv_function
        except ImportError as e:
            raise RuntimeError(
                "Metadynamics requested but module not available. "
                "Check that collective_variables.py and metadynamics.py exist."
            ) from e

        cv_type = steering_args.get("metadynamics_cv", "rg")

        cv_function = create_cv_function(cv_type)

        potentials.append(
            MetadynamicsPotential(
                cv_function=cv_function,
                parameters={
                    "cv_name": cv_type,  # Track the CV name for export
                    "guidance_interval": steering_args.get("metadynamics_guidance_interval", 1),
                    "guidance_weight": steering_args.get("metadynamics_guidance_weight", 0.5),
                    "resampling_weight": 0.0,
                    "hill_height": steering_args.get("metadynamics_hill_height", 0.5),
                    "hill_sigma": steering_args.get("metadynamics_hill_sigma", 5.0),
                    "hill_interval": steering_args.get("metadynamics_hill_interval", 5),
                    "well_tempered": steering_args.get("metadynamics_well_tempered", False),
                    "bias_factor": steering_args.get("metadynamics_bias_factor", 10.0),
                    "kT": steering_args.get("metadynamics_kT", 2.5),
                    "max_hills": steering_args.get("metadynamics_max_hills", 1000),
                }
            )
        )

    # Add generic CV steering potentials (from metadiffusion YAML steering section)
    steering_configs = steering_args.get("steering_configs") or []
    saxs_data_cache = steering_args.get("saxs_pr_data_cache") or {}

    if debug and steering_configs:
        print(f"[DEBUG] Creating {len(steering_configs)} steering potentials")

    for steer_config in steering_configs:
        if debug:
            print(f"[DEBUG] Steering config: cv={steer_config.get('collective_variable')}, "
                  f"target={steer_config.get('target')}, strength={steer_config.get('strength')}, "
                  f"ensemble={steer_config.get('ensemble')}")
        cv_type = steer_config.get("collective_variable")
        target = steer_config.get("target")

        if cv_type is None:
            continue

        # Handle target_from_saxs for Rg CV
        if target is None and cv_type == "rg":
            target_from_saxs = steer_config.get("target_from_saxs")
            if target_from_saxs and target_from_saxs in saxs_data_cache:
                # Extract Rg from SAXS P(r) data
                saxs_data = saxs_data_cache[target_from_saxs]
                r_grid = saxs_data['r_grid']
                pr_exp = saxs_data['pr_exp']
                dr = r_grid[1] - r_grid[0]

                # Compute Rg from P(r): Rg² = ∫r²P(r)dr / (2∫P(r)dr)
                rg_exp = torch.sqrt(
                    torch.sum(r_grid ** 2 * pr_exp * dr) / (2.0 * pr_exp.sum() * dr + 1e-8)
                ).item()

                # Apply scale factor
                auto_rg_scale = steer_config.get("auto_rg_scale", 1.0)
                target = rg_exp * auto_rg_scale

        if target is None:
            continue

        try:
            from boltz.model.potentials.collective_variables import create_cv_function

            # Build kwargs for CV function
            cv_kwargs = {}
            if steer_config.get("reference_structure"):
                # Load reference coords if needed
                cv_kwargs["reference_coords"] = steer_config.get("reference_coords")
            if steer_config.get("contact_cutoff"):
                cv_kwargs["contact_cutoff"] = steer_config["contact_cutoff"]
            # Handle atom specs for distance/angle/dihedral CVs
            # These need to be resolved from string specs like "A:1:CA" to integer indices
            if feats is not None:
                from boltz.data.parse.atom_selection import parse_atom_spec_simple, build_chain_to_atom_mapping
                try:
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))

                    for atom_key in ["atom1", "atom2", "atom3", "atom4"]:
                        atom_spec = steer_config.get(atom_key)
                        if atom_spec is not None and isinstance(atom_spec, str):
                            atom_idx = parse_atom_spec_simple(atom_spec, n_atoms, chain_mapping)
                            cv_kwargs[f"{atom_key}_idx"] = atom_idx
                except Exception as e:
                    warnings.warn(f"Failed to resolve atom specs for {cv_type}: {e}")

            # Handle reference structure for rmsd/native_contacts CVs
            if cv_type in ("rmsd", "native_contacts") and steer_config.get("reference_structure"):
                try:
                    from boltz.model.potentials.factory import load_reference_structure
                    ref_path = steer_config["reference_structure"]
                    cv_kwargs["reference_coords"] = load_reference_structure(ref_path)
                except Exception as e:
                    warnings.warn(f"Failed to load reference structure for {cv_type}: {e}")

            # Helper function to get n_atoms from feats
            def _get_n_atoms(feats):
                n_atoms = feats.get('n_atoms', 0)
                if n_atoms == 0:
                    if 'atom_to_token' in feats:
                        att = feats['atom_to_token']
                        if hasattr(att, 'shape'):
                            # Could be [batch, n_atoms, n_tokens] or [batch, n_atoms] or [n_atoms]
                            if len(att.shape) == 3:
                                n_atoms = att.shape[1]  # [batch, n_atoms, n_tokens]
                            elif len(att.shape) == 2:
                                n_atoms = att.shape[1] if att.shape[0] == 1 else att.shape[0]
                            else:
                                n_atoms = att.shape[0]
                        else:
                            n_atoms = len(att)
                    elif 'atom_pad_mask' in feats:
                        apm = feats['atom_pad_mask']
                        if hasattr(apm, 'shape') and len(apm.shape) > 1:
                            n_atoms = apm.shape[-1]
                        else:
                            n_atoms = len(apm)
                    elif 'chain_id' in feats:
                        n_atoms = len(feats['chain_id'])
                return n_atoms

            # Handle groups for inter_chain/inter_domain CVs
            groups = steer_config.get("groups")
            if cv_type in ("inter_chain", "inter_domain") and groups and len(groups) >= 2 and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = _get_n_atoms(feats)

                    chain1_mask = parse_group_selection_simple([groups[0]], n_atoms, chain_mapping, feats)
                    chain2_mask = parse_group_selection_simple([groups[1]], n_atoms, chain_mapping, feats)

                    if cv_type == "inter_chain":
                        cv_kwargs['chain1_mask'] = chain1_mask
                        cv_kwargs['chain2_mask'] = chain2_mask
                    else:  # inter_domain
                        cv_kwargs['domain1_mask'] = chain1_mask
                        cv_kwargs['domain2_mask'] = chain2_mask
                except Exception as e:
                    warnings.warn(f"Failed to create chain/domain masks for {cv_type}: {e}")

            # Handle groups for hinge_angle CV (needs 3 groups)
            if cv_type == "hinge_angle" and groups and len(groups) >= 3 and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = _get_n_atoms(feats)

                    cv_kwargs['domain1_mask'] = parse_group_selection_simple([groups[0]], n_atoms, chain_mapping, feats)
                    cv_kwargs['hinge_mask'] = parse_group_selection_simple([groups[1]], n_atoms, chain_mapping, feats)
                    cv_kwargs['domain2_mask'] = parse_group_selection_simple([groups[2]], n_atoms, chain_mapping, feats)
                except Exception as e:
                    warnings.warn(f"Failed to create domain masks for {cv_type}: {e}")

            # Handle groups as atom_mask for CVs that use it (rg, sasa, rmsd, etc.)
            # This converts groups like ["A"] or ["A", "B"] into a combined atom mask
            MULTI_GROUP_CVS = {"inter_chain", "inter_domain", "hinge_angle"}
            if groups and cv_type not in MULTI_GROUP_CVS and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = _get_n_atoms(feats)
                    cv_kwargs['atom_mask'] = parse_group_selection_simple(groups, n_atoms, chain_mapping, feats)
                except Exception as e:
                    warnings.warn(f"Failed to create atom mask from groups for {cv_type}: {e}")

            # Handle rmsd_groups as align_mask for pair_rmsd_grouped
            rmsd_groups = steer_config.get("rmsd_groups")
            if cv_type == "pair_rmsd_grouped" and rmsd_groups and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = _get_n_atoms(feats)
                    cv_kwargs['align_mask'] = parse_group_selection_simple(rmsd_groups, n_atoms, chain_mapping, feats)
                except Exception as e:
                    warnings.warn(f"Failed to create align mask from rmsd_groups for {cv_type}: {e}")

            if not groups:
                # Fallback: use raw values if provided as integers
                if steer_config.get("atom1_idx") is not None:
                    cv_kwargs["atom1_idx"] = steer_config["atom1_idx"]
                if steer_config.get("atom2_idx") is not None:
                    cv_kwargs["atom2_idx"] = steer_config["atom2_idx"]
                if steer_config.get("atom3_idx") is not None:
                    cv_kwargs["atom3_idx"] = steer_config["atom3_idx"]
                if steer_config.get("atom4_idx") is not None:
                    cv_kwargs["atom4_idx"] = steer_config["atom4_idx"]

            # Handle region-based CVs (distance, angle, dihedral, etc.)
            region1 = steer_config.get("region1")
            region2 = steer_config.get("region2")
            region3 = steer_config.get("region3")
            region4 = steer_config.get("region4")

            if any([region1, region2, region3, region4]) and feats is not None:
                try:
                    from boltz.model.potentials.factory import _resolve_region_to_mask
                    from boltz.data.parse.atom_selection import build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = _get_n_atoms(feats)

                    if debug:
                        print(f"[DEBUG] Resolving region specs for CV '{cv_type}':")
                        print(f"  chain_mapping: {chain_mapping}")
                        print(f"  n_atoms: {n_atoms}")
                        # Debug: show what keys are in feats
                        print(f"  feats keys: {[k for k in feats.keys() if not k.startswith('_')][:20]}")
                        # Check asym_id and residue_index shapes
                        if 'asym_id' in feats:
                            asym = feats['asym_id']
                            print(f"  asym_id shape: {asym.shape if hasattr(asym, 'shape') else len(asym)}")
                            if hasattr(asym, 'shape') and len(asym.flatten()) < 300:
                                print(f"  asym_id values: {asym.flatten().tolist()[:50]}...")
                        if 'residue_index' in feats:
                            ri = feats['residue_index']
                            print(f"  residue_index shape: {ri.shape if hasattr(ri, 'shape') else len(ri)}")
                        # Show atom_to_token for first few atoms to understand layout
                        att = feats.get('atom_to_token')
                        if att is not None:
                            if hasattr(att, 'shape') and att.dim() == 3:
                                att_flat = att.squeeze(0).argmax(dim=-1)
                            elif hasattr(att, 'shape') and att.dim() == 2:
                                att_flat = att.argmax(dim=-1) if att.shape[-1] > att.shape[0] else att.squeeze(0)
                            else:
                                att_flat = att
                            print(f"  atom_to_token[0:10]: {att_flat[:10].tolist()}")
                            print(f"  atom_to_token[1350:1360]: {att_flat[1350:1360].tolist()}")

                    if region1:
                        cv_kwargs['region1_mask'] = _resolve_region_to_mask(region1, feats, chain_mapping, n_atoms, debug=debug)
                        if debug:
                            print(f"  region1 '{region1}' -> {cv_kwargs['region1_mask'].sum().item()} atoms")
                    if region2:
                        cv_kwargs['region2_mask'] = _resolve_region_to_mask(region2, feats, chain_mapping, n_atoms, debug=debug)
                        if debug:
                            print(f"  region2 '{region2}' -> {cv_kwargs['region2_mask'].sum().item()} atoms")
                    if region3:
                        cv_kwargs['region3_mask'] = _resolve_region_to_mask(region3, feats, chain_mapping, n_atoms, debug=debug)
                        if debug:
                            print(f"  region3 '{region3}' -> {cv_kwargs['region3_mask'].sum().item()} atoms")
                    if region4:
                        cv_kwargs['region4_mask'] = _resolve_region_to_mask(region4, feats, chain_mapping, n_atoms, debug=debug)
                        if debug:
                            print(f"  region4 '{region4}' -> {cv_kwargs['region4_mask'].sum().item()} atoms")

                    # Auto-convert to region-based CV types
                    # Special case: if all regions have exactly one atom and user wants
                    # angle_enhanced or dihedral_enhanced, keep those CVs with bond hops
                    # by extracting atom indices instead of using region-based CVs
                    if 'region1_mask' in cv_kwargs and 'region2_mask' in cv_kwargs:
                        r1_count = cv_kwargs['region1_mask'].sum().item()
                        r2_count = cv_kwargs['region2_mask'].sum().item()
                        r3_count = cv_kwargs.get('region3_mask', torch.tensor([0])).sum().item() if 'region3_mask' in cv_kwargs else 0
                        r4_count = cv_kwargs.get('region4_mask', torch.tensor([0])).sum().item() if 'region4_mask' in cv_kwargs else 0

                        # Check if single-atom regions with enhanced CV type
                        if cv_type == "angle_enhanced" and 'region3_mask' in cv_kwargs and r1_count == 1 and r2_count == 1 and r3_count == 1:
                            # Extract atom indices for angle_enhanced with bond hops
                            cv_kwargs['atom1_idx'] = torch.where(cv_kwargs['region1_mask'])[0].item()
                            cv_kwargs['atom2_idx'] = torch.where(cv_kwargs['region2_mask'])[0].item()
                            cv_kwargs['atom3_idx'] = torch.where(cv_kwargs['region3_mask'])[0].item()
                            if debug:
                                print(f"  Single-atom angle_enhanced: atom1_idx={cv_kwargs['atom1_idx']}, atom2_idx={cv_kwargs['atom2_idx']}, atom3_idx={cv_kwargs['atom3_idx']}")
                            # Remove region masks - angle_enhanced uses atom indices
                            del cv_kwargs['region1_mask']
                            del cv_kwargs['region2_mask']
                            del cv_kwargs['region3_mask']
                            # Keep cv_type as angle_enhanced
                        elif cv_type == "dihedral_enhanced" and 'region3_mask' in cv_kwargs and 'region4_mask' in cv_kwargs and r1_count == 1 and r2_count == 1 and r3_count == 1 and r4_count == 1:
                            # Extract atom indices for dihedral_enhanced with bond hops
                            cv_kwargs['atom1_idx'] = torch.where(cv_kwargs['region1_mask'])[0].item()
                            cv_kwargs['atom2_idx'] = torch.where(cv_kwargs['region2_mask'])[0].item()
                            cv_kwargs['atom3_idx'] = torch.where(cv_kwargs['region3_mask'])[0].item()
                            cv_kwargs['atom4_idx'] = torch.where(cv_kwargs['region4_mask'])[0].item()
                            # Remove region masks
                            del cv_kwargs['region1_mask']
                            del cv_kwargs['region2_mask']
                            del cv_kwargs['region3_mask']
                            del cv_kwargs['region4_mask']
                            # Keep cv_type as dihedral_enhanced
                        elif cv_type == "distance":
                            cv_type = "distance_region"
                        elif cv_type in ("angle", "angle_enhanced") and 'region3_mask' in cv_kwargs:
                            cv_type = "angle_region"
                        elif cv_type in ("dihedral", "dihedral_enhanced") and 'region3_mask' in cv_kwargs and 'region4_mask' in cv_kwargs:
                            cv_type = "dihedral_region"
                except Exception as e:
                    warnings.warn(f"Failed to resolve region specs for steering {cv_type}: {e}")

            # Handle max_hops and decay for enhanced CVs
            if steer_config.get("max_hops") is not None:
                cv_kwargs["max_hops"] = steer_config["max_hops"]
            if steer_config.get("decay") is not None:
                cv_kwargs["decay"] = steer_config["decay"]

            cv_function = create_cv_function(cv_type, **cv_kwargs)

            steer_potential = GenericCVPotential(
                cv_function=cv_function,
                parameters={
                    "guidance_interval": steer_config.get("guidance_interval", 1),
                    "guidance_weight": steer_config.get("guidance_weight", 1.0),
                    "resampling_weight": 0.0,
                    "target": target,
                    "k": steer_config.get("strength", 1.0),
                    "warmup": steer_config.get("warmup", 0.0),
                    "cutoff": steer_config.get("cutoff", 0.75),
                    "bias_tempering": steer_config.get("bias_clip"),
                    "cv_name": cv_type,
                },
                debug=debug,
            )

            # Attach gradient modifier if scaling and/or projection config is present
            scaling_config = steer_config.get("scaling")
            projection_config = steer_config.get("projection")
            if scaling_config or projection_config:
                scaling_configs = parse_scaling_config(scaling_config) if scaling_config else None
                projection_configs = parse_projection_config(projection_config) if projection_config else None
                modifier = create_gradient_modifier(
                    scaling_configs=scaling_configs,
                    projection_configs=projection_configs,
                    modifier_order=steer_config.get("modifier_order", "scale_first"),
                    feats=feats,
                )
                if modifier is not None:
                    steer_potential.gradient_scaler = modifier

            potentials.append(steer_potential)
        except Exception as e:
            warnings.warn(f"Failed to create CV potential for {cv_type}: {e}")

    # Add generic CV bias potentials (hills/repulsion from metadiffusion YAML explore section)
    # Support both "bias_configs" (legacy) and "explore_configs" (current)
    bias_configs = steering_args.get("bias_configs") or steering_args.get("explore_configs") or []

    if debug and bias_configs:
        print(f"[DEBUG] Creating {len(bias_configs)} explore potentials")

    for bias_config in bias_configs:
        if debug:
            print(f"[DEBUG] Explore config: type={bias_config.get('type', 'hills')}, "
                  f"cv={bias_config.get('collective_variable')}, "
                  f"strength={bias_config.get('strength')}, sigma={bias_config.get('sigma')}")
        cv_type = bias_config.get("collective_variable")
        bias_type = bias_config.get("type", "hills")

        if cv_type is None:
            continue

        try:
            from boltz.model.potentials.collective_variables import create_cv_function
            from boltz.model.potentials.metadynamics import MetadynamicsPotential

            # Build kwargs for CV function
            cv_kwargs = {}
            if bias_config.get("contact_cutoff"):
                cv_kwargs["contact_cutoff"] = bias_config["contact_cutoff"]

            # Handle atom specs for distance/angle/dihedral CVs (e.g., "A:1:CA")
            if feats is not None:
                from boltz.data.parse.atom_selection import parse_atom_spec_simple, build_chain_to_atom_mapping
                try:
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))

                    for atom_key in ["atom1", "atom2", "atom3", "atom4"]:
                        atom_spec = bias_config.get(atom_key)
                        if atom_spec is not None and isinstance(atom_spec, str):
                            atom_idx = parse_atom_spec_simple(atom_spec, n_atoms, chain_mapping)
                            cv_kwargs[f"{atom_key}_idx"] = atom_idx
                except Exception as e:
                    warnings.warn(f"Failed to resolve atom specs for explore {cv_type}: {e}")

            # Handle max_hops and decay for enhanced CVs
            if bias_config.get("max_hops") is not None:
                cv_kwargs["max_hops"] = bias_config["max_hops"]
            if bias_config.get("decay") is not None:
                cv_kwargs["decay"] = bias_config["decay"]

            # Handle reference structure for rmsd/native_contacts CVs
            if cv_type in ("rmsd", "native_contacts") and bias_config.get("reference_structure"):
                try:
                    from boltz.model.potentials.factory import load_reference_structure
                    ref_path = bias_config["reference_structure"]
                    cv_kwargs["reference_coords"] = load_reference_structure(ref_path)
                except Exception as e:
                    warnings.warn(f"Failed to load reference structure for explore {cv_type}: {e}")

            # Handle groups as atom_mask for CVs that use it (rg, sasa, rmsd, etc.)
            groups = bias_config.get("groups")
            MULTI_GROUP_CVS = {"inter_chain", "inter_domain", "hinge_angle"}
            if groups and cv_type not in MULTI_GROUP_CVS and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))
                    cv_kwargs['atom_mask'] = parse_group_selection_simple(groups, n_atoms, chain_mapping, feats)
                except Exception as e:
                    warnings.warn(f"Failed to create atom mask from groups for explore {cv_type}: {e}")

            # Handle rmsd_groups as align_mask for pair_rmsd_grouped
            rmsd_groups = bias_config.get("rmsd_groups")
            if cv_type == "pair_rmsd_grouped" and rmsd_groups and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = feats.get('n_atoms', len(feats.get('chain_id', [])))
                    cv_kwargs['align_mask'] = parse_group_selection_simple(rmsd_groups, n_atoms, chain_mapping, feats)
                except Exception as e:
                    warnings.warn(f"Failed to create align mask from rmsd_groups for explore {cv_type}: {e}")

            # Handle region-based CVs (distance, angle, dihedral, etc.)
            region1 = bias_config.get("region1")
            region2 = bias_config.get("region2")
            region3 = bias_config.get("region3")
            region4 = bias_config.get("region4")

            if any([region1, region2, region3, region4]) and feats is not None:
                try:
                    from boltz.model.potentials.factory import _resolve_region_to_mask
                    from boltz.data.parse.atom_selection import build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)

                    if debug:
                        print(f"[DEBUG] Resolving regions for explore CV '{cv_type}':")
                        print(f"  chain_mapping: {chain_mapping}")

                    # Get n_atoms
                    n_atoms = feats.get('n_atoms', 0)
                    if n_atoms == 0:
                        if 'atom_pad_mask' in feats:
                            apm = feats['atom_pad_mask']
                            if hasattr(apm, 'shape') and len(apm.shape) > 1:
                                n_atoms = apm.shape[-1]
                            else:
                                n_atoms = len(apm)
                        elif 'chain_id' in feats:
                            n_atoms = len(feats['chain_id'])

                    if region1:
                        cv_kwargs['region1_mask'] = _resolve_region_to_mask(region1, feats, chain_mapping, n_atoms, debug=debug)
                        if debug:
                            print(f"  region1 '{region1}' -> {cv_kwargs['region1_mask'].sum().item()} atoms")
                    if region2:
                        cv_kwargs['region2_mask'] = _resolve_region_to_mask(region2, feats, chain_mapping, n_atoms, debug=debug)
                        if debug:
                            print(f"  region2 '{region2}' -> {cv_kwargs['region2_mask'].sum().item()} atoms")
                    if region3:
                        cv_kwargs['region3_mask'] = _resolve_region_to_mask(region3, feats, chain_mapping, n_atoms, debug=debug)
                        if debug:
                            print(f"  region3 '{region3}' -> {cv_kwargs['region3_mask'].sum().item()} atoms")
                    if region4:
                        cv_kwargs['region4_mask'] = _resolve_region_to_mask(region4, feats, chain_mapping, n_atoms, debug=debug)
                        if debug:
                            print(f"  region4 '{region4}' -> {cv_kwargs['region4_mask'].sum().item()} atoms")

                    # Auto-convert to region-based CV types
                    # Special case: if all regions have exactly one atom and user wants
                    # angle_enhanced or dihedral_enhanced, keep those CVs with bond hops
                    if 'region1_mask' in cv_kwargs and 'region2_mask' in cv_kwargs:
                        r1_count = cv_kwargs['region1_mask'].sum().item()
                        r2_count = cv_kwargs['region2_mask'].sum().item()
                        r3_count = cv_kwargs.get('region3_mask', torch.tensor([0])).sum().item() if 'region3_mask' in cv_kwargs else 0
                        r4_count = cv_kwargs.get('region4_mask', torch.tensor([0])).sum().item() if 'region4_mask' in cv_kwargs else 0

                        # Check if single-atom regions with enhanced CV type
                        if cv_type == "angle_enhanced" and 'region3_mask' in cv_kwargs and r1_count == 1 and r2_count == 1 and r3_count == 1:
                            # Extract atom indices for angle_enhanced with bond hops
                            cv_kwargs['atom1_idx'] = torch.where(cv_kwargs['region1_mask'])[0].item()
                            cv_kwargs['atom2_idx'] = torch.where(cv_kwargs['region2_mask'])[0].item()
                            cv_kwargs['atom3_idx'] = torch.where(cv_kwargs['region3_mask'])[0].item()
                            if debug:
                                print(f"  Single-atom angle_enhanced: atom1={cv_kwargs['atom1_idx']}, atom2={cv_kwargs['atom2_idx']}, atom3={cv_kwargs['atom3_idx']}")
                            del cv_kwargs['region1_mask']
                            del cv_kwargs['region2_mask']
                            del cv_kwargs['region3_mask']
                        elif cv_type == "dihedral_enhanced" and 'region3_mask' in cv_kwargs and 'region4_mask' in cv_kwargs and r1_count == 1 and r2_count == 1 and r3_count == 1 and r4_count == 1:
                            # Extract atom indices for dihedral_enhanced with bond hops
                            cv_kwargs['atom1_idx'] = torch.where(cv_kwargs['region1_mask'])[0].item()
                            cv_kwargs['atom2_idx'] = torch.where(cv_kwargs['region2_mask'])[0].item()
                            cv_kwargs['atom3_idx'] = torch.where(cv_kwargs['region3_mask'])[0].item()
                            cv_kwargs['atom4_idx'] = torch.where(cv_kwargs['region4_mask'])[0].item()
                            del cv_kwargs['region1_mask']
                            del cv_kwargs['region2_mask']
                            del cv_kwargs['region3_mask']
                            del cv_kwargs['region4_mask']
                        elif cv_type == "distance":
                            cv_type = "distance_region"
                        elif cv_type in ("angle", "angle_enhanced") and 'region3_mask' in cv_kwargs:
                            cv_type = "angle_region"
                        elif cv_type in ("dihedral", "dihedral_enhanced") and 'region3_mask' in cv_kwargs and 'region4_mask' in cv_kwargs:
                            cv_type = "dihedral_region"
                except Exception as e:
                    warnings.warn(f"Failed to resolve region specs for explore {cv_type}: {e}")

            cv_function = create_cv_function(cv_type, **cv_kwargs)

            explore_potential = None
            if bias_type == "hills":
                explore_potential = MetadynamicsPotential(
                    cv_function=cv_function,
                    parameters={
                        "cv_name": cv_type,  # Track the CV name for export
                        "guidance_interval": bias_config.get("guidance_interval", 1),
                        "guidance_weight": bias_config.get("guidance_weight", 1.0),
                        "resampling_weight": 0.0,
                        "hill_height": bias_config.get("hill_height", 0.5),
                        "hill_sigma": bias_config.get("sigma", 5.0),
                        "hill_interval": bias_config.get("hill_interval", 5),
                        "well_tempered": bias_config.get("well_tempered", False),
                        "bias_factor": bias_config.get("bias_factor", 10.0),
                        "kT": bias_config.get("kT", 2.5),
                        "max_hills": bias_config.get("max_hills", 1000),
                        "warmup": bias_config.get("warmup", 0.2),
                        "cutoff": bias_config.get("cutoff", 0.75),
                        "bias_tempering": bias_config.get("bias_clip"),
                    }
                )
            elif bias_type == "repulsion":
                # Repulsion bias - steer samples apart in CV space
                # Use RepulsionPotential with Gaussian repulsion (bounded gradients)
                explore_potential = RepulsionPotential(
                    cv_function=cv_function,
                    parameters={
                        "guidance_interval": bias_config.get("guidance_interval", 1),
                        "guidance_weight": bias_config.get("guidance_weight", 1.0),
                        "resampling_weight": 0.0,
                        "strength": bias_config.get("strength", 256.0),
                        "sigma": bias_config.get("sigma", 5.0),
                        "warmup": bias_config.get("warmup", 0.2),
                        "cutoff": bias_config.get("cutoff", 0.75),
                        "bias_tempering": bias_config.get("bias_clip"),
                    }
                )
            elif bias_type == "variance":
                # Variance maximization - push samples away from mean CV value
                # Provides linear force proportional to distance from mean
                explore_potential = VariancePotential(
                    cv_function=cv_function,
                    parameters={
                        "guidance_interval": bias_config.get("guidance_interval", 1),
                        "guidance_weight": bias_config.get("guidance_weight", 1.0),
                        "resampling_weight": 0.0,
                        "strength": bias_config.get("strength", 1.0),
                        "warmup": bias_config.get("warmup", 0.2),
                        "cutoff": bias_config.get("cutoff", 0.75),
                        "bias_tempering": bias_config.get("bias_clip"),
                    }
                )

            if explore_potential is not None:
                # Attach gradient modifier if scaling and/or projection config is present
                scaling_config = bias_config.get("scaling")
                projection_config = bias_config.get("projection")
                if scaling_config or projection_config:
                    scaling_configs = parse_scaling_config(scaling_config) if scaling_config else None
                    projection_configs = parse_projection_config(projection_config) if projection_config else None
                    modifier = create_gradient_modifier(
                        scaling_configs=scaling_configs,
                        projection_configs=projection_configs,
                        modifier_order=bias_config.get("modifier_order", "scale_first"),
                        feats=feats,
                    )
                    if modifier is not None:
                        explore_potential.gradient_scaler = modifier

                potentials.append(explore_potential)
        except Exception as e:
            warnings.warn(f"Failed to create bias potential for {cv_type}: {e}")

    # Add generic CV optimization potentials (from metadiffusion YAML opt section)
    # These potentials push the CV toward lower (minimize) or higher (maximize) values
    opt_configs = steering_args.get("opt_configs") or []
#     print(f"DEBUG get_potentials: opt_configs={opt_configs}", flush=True)

    for opt_config in opt_configs:
        cv_type = opt_config.get("collective_variable")

        if cv_type is None:
            continue

        try:
            from boltz.model.potentials.collective_variables import create_cv_function

            # Build kwargs for CV function
            cv_kwargs = {}
            if opt_config.get("contact_cutoff"):
                cv_kwargs["contact_cutoff"] = opt_config["contact_cutoff"]

            # Handle reference structure for rmsd/native_contacts CVs
            if cv_type in ("rmsd", "native_contacts") and opt_config.get("reference_structure"):
                try:
                    from boltz.model.potentials.factory import load_reference_structure
                    ref_path = opt_config["reference_structure"]
                    cv_kwargs["reference_coords"] = load_reference_structure(ref_path)
                except Exception as e:
                    warnings.warn(f"Failed to load reference structure for opt {cv_type}: {e}")

            # Handle feats for atom mask
            if feats is not None:
                cv_kwargs["atom_mask"] = feats.get("atom_pad_mask")

            # Handle groups for inter_chain/inter_domain CVs
            groups = opt_config.get("groups")
            if cv_type in ("inter_chain", "inter_domain") and groups and len(groups) >= 2 and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)

                    # Get n_atoms
                    n_atoms = feats.get('n_atoms', 0)
                    if n_atoms == 0:
                        if 'atom_pad_mask' in feats:
                            apm = feats['atom_pad_mask']
                            if hasattr(apm, 'shape') and len(apm.shape) > 1:
                                n_atoms = apm.shape[-1]
                            else:
                                n_atoms = len(apm)
                        elif 'chain_id' in feats:
                            n_atoms = len(feats['chain_id'])

                    chain1_mask = parse_group_selection_simple([groups[0]], n_atoms, chain_mapping, feats)
                    chain2_mask = parse_group_selection_simple([groups[1]], n_atoms, chain_mapping, feats)

                    if cv_type == "inter_chain":
                        cv_kwargs['chain1_mask'] = chain1_mask
                        cv_kwargs['chain2_mask'] = chain2_mask
                    else:  # inter_domain
                        cv_kwargs['domain1_mask'] = chain1_mask
                        cv_kwargs['domain2_mask'] = chain2_mask
                except Exception as e:
                    warnings.warn(f"Failed to create chain/domain masks for opt {cv_type}: {e}")

            # Handle groups for hinge_angle CV (needs 3 groups) - DEPRECATED
            if cv_type == "hinge_angle" and groups and len(groups) >= 3 and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)

                    # Get n_atoms
                    n_atoms = feats.get('n_atoms', 0)
                    if n_atoms == 0:
                        if 'atom_pad_mask' in feats:
                            apm = feats['atom_pad_mask']
                            if hasattr(apm, 'shape') and len(apm.shape) > 1:
                                n_atoms = apm.shape[-1]
                            else:
                                n_atoms = len(apm)
                        elif 'chain_id' in feats:
                            n_atoms = len(feats['chain_id'])

                    cv_kwargs['domain1_mask'] = parse_group_selection_simple([groups[0]], n_atoms, chain_mapping, feats)
                    cv_kwargs['hinge_mask'] = parse_group_selection_simple([groups[1]], n_atoms, chain_mapping, feats)
                    cv_kwargs['domain2_mask'] = parse_group_selection_simple([groups[2]], n_atoms, chain_mapping, feats)
                except Exception as e:
                    warnings.warn(f"Failed to create domain masks for opt {cv_type}: {e}")

            # Handle groups as atom_mask for CVs that use it (rg, sasa, rmsd, etc.)
            MULTI_GROUP_CVS = {"inter_chain", "inter_domain", "hinge_angle"}
            if groups and cv_type not in MULTI_GROUP_CVS and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = feats.get('n_atoms', 0)
                    if n_atoms == 0:
                        if 'atom_pad_mask' in feats:
                            apm = feats['atom_pad_mask']
                            if hasattr(apm, 'shape') and len(apm.shape) > 1:
                                n_atoms = apm.shape[-1]
                            else:
                                n_atoms = len(apm)
                        elif 'chain_id' in feats:
                            n_atoms = len(feats['chain_id'])
                    cv_kwargs['atom_mask'] = parse_group_selection_simple(groups, n_atoms, chain_mapping, feats)
                except Exception as e:
                    warnings.warn(f"Failed to create atom mask from groups for opt {cv_type}: {e}")

            # Handle rmsd_groups as align_mask for pair_rmsd_grouped
            rmsd_groups = opt_config.get("rmsd_groups")
            if cv_type == "pair_rmsd_grouped" and rmsd_groups and feats is not None:
                try:
                    from boltz.data.parse.atom_selection import parse_group_selection_simple, build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)
                    n_atoms = feats.get('n_atoms', 0)
                    if n_atoms == 0:
                        if 'atom_pad_mask' in feats:
                            apm = feats['atom_pad_mask']
                            if hasattr(apm, 'shape') and len(apm.shape) > 1:
                                n_atoms = apm.shape[-1]
                            else:
                                n_atoms = len(apm)
                        elif 'chain_id' in feats:
                            n_atoms = len(feats['chain_id'])
                    cv_kwargs['align_mask'] = parse_group_selection_simple(rmsd_groups, n_atoms, chain_mapping, feats)
                except Exception as e:
                    warnings.warn(f"Failed to create align mask from rmsd_groups for opt {cv_type}: {e}")

            # Handle region-based CVs (distance, angle, dihedral, min_distance, etc.)
            region1 = opt_config.get("region1")
            region2 = opt_config.get("region2")
            region3 = opt_config.get("region3")
            region4 = opt_config.get("region4")

            if any([region1, region2, region3, region4]) and feats is not None:
                try:
                    from boltz.model.potentials.factory import _resolve_region_to_mask
                    from boltz.data.parse.atom_selection import build_chain_to_atom_mapping
                    chain_mapping = build_chain_to_atom_mapping(feats)

                    # Get n_atoms
                    n_atoms = feats.get('n_atoms', 0)
                    if n_atoms == 0:
                        if 'atom_pad_mask' in feats:
                            apm = feats['atom_pad_mask']
                            if hasattr(apm, 'shape') and len(apm.shape) > 1:
                                n_atoms = apm.shape[-1]
                            else:
                                n_atoms = len(apm)
                        elif 'chain_id' in feats:
                            n_atoms = len(feats['chain_id'])

                    if region1:
                        cv_kwargs['region1_mask'] = _resolve_region_to_mask(region1, feats, chain_mapping, n_atoms)
                    if region2:
                        cv_kwargs['region2_mask'] = _resolve_region_to_mask(region2, feats, chain_mapping, n_atoms)
                    if region3:
                        cv_kwargs['region3_mask'] = _resolve_region_to_mask(region3, feats, chain_mapping, n_atoms)
                    if region4:
                        cv_kwargs['region4_mask'] = _resolve_region_to_mask(region4, feats, chain_mapping, n_atoms)

                    # Auto-convert to region-based CV types
                    if 'region1_mask' in cv_kwargs and 'region2_mask' in cv_kwargs:
                        if cv_type == "distance":
                            cv_type = "distance_region"
                        elif cv_type in ("angle", "angle_enhanced") and 'region3_mask' in cv_kwargs:
                            cv_type = "angle_region"
                        elif cv_type in ("dihedral", "dihedral_enhanced") and 'region3_mask' in cv_kwargs and 'region4_mask' in cv_kwargs:
                            cv_type = "dihedral_region"
                except Exception as e:
                    warnings.warn(f"Failed to resolve region specs for opt {cv_type}: {e}")

            cv_function = create_cv_function(cv_type, **cv_kwargs)

            # Determine minimize/maximize from sign of strength
            # Positive strength = maximize CV (increase), negative = minimize CV (decrease)
            strength = opt_config.get("strength", 1.0)
            minimize = strength < 0
            k = abs(strength)

            opt_potential = OptPotential(
                cv_function=cv_function,
                parameters={
                    "guidance_interval": opt_config.get("guidance_interval", 1),
                    "guidance_weight": 1.0,
                    "resampling_weight": 0.0,
                    "k": k,
                    "minimize": minimize,
                    "warmup": opt_config.get("warmup", 0.0),
                    "cutoff": opt_config.get("cutoff", 0.75),
                    "log_gradient": opt_config.get("log_gradient", False),
                    "bias_tempering": opt_config.get("bias_clip"),
                }
            )

            # Attach gradient modifier if scaling and/or projection config is present
            scaling_config = opt_config.get("scaling")
            projection_config = opt_config.get("projection")
            if scaling_config or projection_config:
                scaling_configs = parse_scaling_config(scaling_config) if scaling_config else None
                projection_configs = parse_projection_config(projection_config) if projection_config else None
                modifier = create_gradient_modifier(
                    scaling_configs=scaling_configs,
                    projection_configs=projection_configs,
                    modifier_order=opt_config.get("modifier_order", "scale_first"),
                    feats=feats,
                )
                if modifier is not None:
                    opt_potential.gradient_scaler = modifier

            potentials.append(opt_potential)
#             print(f"DEBUG: Created OptPotential for {cv_type}, k={k}, minimize={minimize}, total potentials={len(potentials)}", flush=True)
        except Exception as e:
            warnings.warn(f"Failed to create opt potential for {cv_type}: {e}")

    # Add chemical shift potentials (from metadiffusion YAML chemical_shift section)
    chemical_shift_configs = steering_args.get("chemical_shift_configs") or []
    for cs_config in chemical_shift_configs:
        try:
            from boltz.model.potentials.chemical_shift import ChemicalShiftPotential, load_shift_file

            # Load experimental shifts
            exp_ca_shifts = None
            exp_cb_shifts = None

            ca_file = cs_config.get("ca_shifts")
            if ca_file:
                exp_ca_shifts = load_shift_file(ca_file)

            cb_file = cs_config.get("cb_shifts")
            if cb_file:
                exp_cb_shifts = load_shift_file(cb_file)

            if exp_ca_shifts is None and exp_cb_shifts is None:
                warnings.warn("Chemical shift config has no valid shift data. Skipping.")
                continue

            # Convert to exp_shifts dict format expected by ChemicalShiftPotential
            exp_shifts = {}
            if exp_ca_shifts is not None:
                exp_shifts['CA'] = exp_ca_shifts
            if exp_cb_shifts is not None:
                exp_shifts['CB'] = exp_cb_shifts

            # strength controls gradient magnitude via guidance_weight
            # k is kept at 1.0 since gradient is normalized then scaled by guidance_weight
            strength = cs_config.get("strength", 1.0)
            potential = ChemicalShiftPotential(
                parameters={
                    "guidance_interval": cs_config.get("guidance_interval", 1),
                    "guidance_weight": strength,
                    "resampling_weight": 0.0,
                    "k": 1.0,  # Loss uses k=1.0, strength applied via guidance_weight
                    "loss_type": cs_config.get("loss_type", "chi"),
                    "exp_shifts": exp_shifts,
                    "warmup": cs_config.get("warmup", 0.0),
                    "cutoff": cs_config.get("cutoff", 0.9),
                    "bias_tempering": cs_config.get("bias_clip"),
                }
            )

            # Attach gradient modifier if scaling and/or projection config is present
            scaling_config = cs_config.get("scaling")
            projection_config = cs_config.get("projection")
            if scaling_config or projection_config:
                scaling_configs = parse_scaling_config(scaling_config) if scaling_config else None
                projection_configs = parse_projection_config(projection_config) if projection_config else None
                modifier = create_gradient_modifier(
                    scaling_configs=scaling_configs,
                    projection_configs=projection_configs,
                    modifier_order=cs_config.get("modifier_order", "scale_first"),
                    feats=feats,
                )
                if modifier is not None:
                    potential.gradient_scaler = modifier

            potentials.append(potential)
        except Exception as e:
            warnings.warn(f"Failed to create chemical shift potential: {e}")

    return potentials
