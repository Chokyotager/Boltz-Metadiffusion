"""
General metadynamics potential for enhanced sampling during diffusion.

Implements standard and well-tempered metadynamics with pluggable
collective variables (CVs).
"""

import torch
from typing import Optional, Callable, List, Dict, Any, Tuple
from boltz.model.potentials.potentials import Potential


class MetadynamicsPotential(Potential):
    """
    General metadynamics potential for enhanced sampling.

    Deposits Gaussian hills on collective variables (CVs) to explore
    conformational space. Supports both standard and well-tempered metadynamics.

    The bias potential is:
        V(s) = sum_i h_i * exp(-0.5 * (s - s_i)^2 / sigma^2)

    where s is the current CV value, s_i are deposited hill centers,
    h_i are hill heights, and sigma is the hill width.

    For well-tempered metadynamics:
        h_i = h_0 * exp(-V(s_i) / (k_B * dT))

    where dT = T * (gamma - 1) and gamma is the bias factor.

    Attributes:
        cv_function: Callable that computes CV from coords
                     Signature: cv_function(coords, feats) -> (cv_values, cv_gradient)
        hills: List of deposited hills with {cv_center, height, sigma, step}
    """

    def __init__(
        self,
        cv_function: Callable[[torch.Tensor, dict], Tuple[torch.Tensor, torch.Tensor]],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            cv_function: Callable that computes CV and gradient
                         Signature: (coords, feats) -> (cv_values, cv_gradient)
                         cv_values: [multiplicity] or scalar
                         cv_gradient: [multiplicity, N_atoms, 3]
            parameters: Dict with metadynamics settings:
                - hill_height: Base height of Gaussian hills (default: 0.5)
                - hill_sigma: Width of Gaussian hills in CV units (default: 2.0)
                - hill_interval: Steps between hill deposits (default: 5)
                - well_tempered: Use well-tempered metadynamics (default: False)
                - bias_factor: Bias factor gamma for well-tempered (default: 10.0)
                - kT: Temperature kT for well-tempered (default: 2.5)
                - max_hills: Maximum number of hills to store (default: 1000)
        """
        super().__init__(parameters)
        self.cv_function = cv_function
        self.hills: List[Dict[str, Any]] = []
        # Track repulsion energies for each sample pair
        self.repulsion_history: List[Dict[str, Any]] = []

        # Extract parameters with defaults
        self._name = self.parameters.get('name')
        # Support both 'explore_type' (new) and 'bias_type' (deprecated)
        self._explore_type = self.parameters.get('explore_type') or self.parameters.get('bias_type', 'hills')
        self._cv_name = self.parameters.get('cv_name', 'unknown')
        self._hill_height = self.parameters.get('hill_height', 0.5)
        self._hill_sigma = self.parameters.get('hill_sigma', 2.0)
        self._hill_interval = self.parameters.get('hill_interval', 5)
        self._well_tempered = self.parameters.get('well_tempered', False)
        self._bias_factor = self.parameters.get('bias_factor', 10.0)
        # Validate bias_factor for well-tempered metadynamics
        if self._well_tempered and self._bias_factor <= 1.0:
            import warnings
            warnings.warn(
                f"bias_factor must be > 1.0 for well-tempered metadynamics, got {self._bias_factor}. "
                f"Falling back to standard metadynamics."
            )
            self._well_tempered = False
        self._kT = self.parameters.get('kT', 2.5)
        self._max_hills = self.parameters.get('max_hills', 1000)

    def reset_hills(self):
        """Clear all deposited hills and repulsion history."""
        self.hills = []
        self.repulsion_history = []

    def record_repulsion_energy(self, step: int, cv_values: torch.Tensor, energies: torch.Tensor):
        """
        Record repulsion energies for export.

        Args:
            step: Current diffusion step
            cv_values: CV values for each sample pair [N_pairs]
            energies: Repulsion energies for each pair [N_pairs]
        """
        self.repulsion_history.append({
            'step': step,
            'cv_values': cv_values.detach().cpu().tolist() if torch.is_tensor(cv_values) else cv_values,
            'energies': energies.detach().cpu().tolist() if torch.is_tensor(energies) else energies,
        })

    def export_data(self) -> Dict[str, Any]:
        """
        Export explore data for JSON serialization.

        Returns:
            Dict with explore metadata and history (hills or repulsion energies)
        """
        data = {
            'name': self._name,
            'explore_type': self._explore_type,
            'collective_variable': self._cv_name,
            'parameters': {
                'sigma': self._hill_sigma,
                'well_tempered': self._well_tempered,
                'bias_factor': self._bias_factor if self._well_tempered else None,
                'kT': self._kT if self._well_tempered else None,
            }
        }

        if self._explore_type == 'hills':
            data['parameters']['hill_height'] = self._hill_height
            data['parameters']['hill_interval'] = self._hill_interval
            data['parameters']['max_hills'] = self._max_hills
            data['hills'] = self.hills
        elif self._explore_type == 'repulsion':
            data['parameters']['strength'] = self.parameters.get('k', 256.0)
            data['repulsion_history'] = self.repulsion_history

        return data

    def compute_variable(
        self,
        coords: torch.Tensor,
        index: Optional[torch.Tensor] = None,
        ref_coords: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        compute_gradient: bool = False,
    ) -> torch.Tensor:
        """
        Compute collective variable value(s).

        Args:
            coords: [multiplicity, N_atoms, 3]
            index: Not used (CV function handles indexing)
            compute_gradient: If True, return (cv_values, cv_gradient)

        Returns:
            cv_values: [multiplicity] CV values
            cv_gradient (if compute_gradient): [multiplicity, N_atoms, 3]
        """
        if compute_gradient:
            return self.cv_function(coords, {})
        else:
            cv_values, _ = self.cv_function(coords, {})
            return cv_values

    def compute_function(
        self,
        cv_values: torch.Tensor,
        k: float,
        negation_mask: Optional[torch.Tensor] = None,
        compute_derivative: bool = False,
        loss_type: str = 'metadynamics',
    ) -> torch.Tensor:
        """
        Compute bias potential.

        For hills bias type:
            V(s) = sum_i h_i * exp(-0.5 * (s - s_i)^2 / sigma^2)

        For repulsion bias type:
            V(s) = k * exp(-0.5 * (s / sigma)^2)
            This pushes samples toward higher CV values (diversity).

        Args:
            cv_values: [multiplicity] current CV values
            k: Force constant (scales the bias)
            compute_derivative: If True, return (energy, dV_dCV)

        Returns:
            energy: [multiplicity] bias energies
            dV_dCV (if compute_derivative): [multiplicity] derivative w.r.t. CV
        """
        device = cv_values.device
        dtype = cv_values.dtype
        multiplicity = cv_values.shape[0] if cv_values.dim() > 0 else 1

        if cv_values.dim() == 0:
            cv_values = cv_values.unsqueeze(0)

        # Handle variance bias type: maximize variance of CV values across samples
        if self._explore_type == 'variance':
            # Variance = mean((CV_i - mean)²)
            # E = -k * Variance (negative because we maximize)
            cv_mean = cv_values.mean()
            deviation = cv_values - cv_mean  # [multiplicity]
            variance = (deviation ** 2).mean()

            energy = -k * variance * torch.ones(multiplicity, device=device, dtype=dtype)

            if compute_derivative:
                # dE/dCV_i = -k * 2 * (CV_i - mean) / N
                dV_dCV = -k * 2 * deviation / multiplicity
                return energy, dV_dCV
            return energy

        # Handle repulsion bias type: pairwise repulsion between samples
        if self._explore_type == 'repulsion':
            sigma = self._hill_sigma
            # Pairwise repulsion: samples with similar CV values repel each other
            # E_i = mean_{j != i} k * exp(-0.5 * (cv_i - cv_j)^2 / sigma^2)
            # This pushes samples to have diverse CV values
            # Note: Sample-count invariant (averaged over pairs)

            energy = torch.zeros(multiplicity, device=device, dtype=dtype)
            dV_dCV = torch.zeros(multiplicity, device=device, dtype=dtype) if compute_derivative else None

            for i in range(multiplicity):
                for j in range(multiplicity):
                    if i != j:
                        diff = cv_values[i] - cv_values[j]
                        gaussian = torch.exp(-0.5 * (diff / sigma) ** 2)
                        energy[i] += k * gaussian

                        if compute_derivative:
                            # dE_i/dCV_i = k * (-diff / sigma^2) * exp(...)
                            dV_dCV[i] += k * (-diff / (sigma ** 2)) * gaussian

            # Average over pairs (sample-count invariant)
            if multiplicity > 1:
                energy = energy / (multiplicity - 1)
                if compute_derivative:
                    dV_dCV = dV_dCV / (multiplicity - 1)

            if compute_derivative:
                return energy, dV_dCV
            return energy

        # Hills bias type: compute bias from deposited hills
        # No hills yet - return zero bias
        if len(self.hills) == 0:
            energy = torch.zeros(multiplicity, device=device, dtype=dtype)
            if compute_derivative:
                dV_dCV = torch.zeros(multiplicity, device=device, dtype=dtype)
                return k * energy, k * dV_dCV
            return k * energy

        # Compute bias from all hills
        energy = torch.zeros(multiplicity, device=device, dtype=dtype)
        dV_dCV = torch.zeros(multiplicity, device=device, dtype=dtype)

        for hill in self.hills:
            s_i = hill['cv_center']
            h_i = hill['height']
            sigma = hill['sigma']

            # Gaussian: exp(-0.5 * (s - s_i)^2 / sigma^2)
            diff = cv_values - s_i
            gaussian = torch.exp(-0.5 * (diff / sigma) ** 2)

            energy += h_i * gaussian

            if compute_derivative:
                # dV/ds = h_i * (-(s - s_i) / sigma^2) * exp(...)
                dV_dCV += h_i * (-diff / (sigma ** 2)) * gaussian

        if compute_derivative:
            return k * energy, k * dV_dCV
        return k * energy

    def compute_explore_at_cv(self, cv_value: float) -> float:
        """
        Compute current explore potential at a specific CV value.

        Used for well-tempered metadynamics hill height scaling.

        Args:
            cv_value: CV value to evaluate explore potential at

        Returns:
            explore_energy: Total explore potential at cv_value
        """
        if len(self.hills) == 0:
            return 0.0

        bias = 0.0
        for hill in self.hills:
            s_i = hill['cv_center']
            h_i = hill['height']
            sigma = hill['sigma']

            diff = cv_value - s_i
            gaussian = torch.exp(torch.tensor(-0.5 * (diff / sigma) ** 2))
            bias += h_i * gaussian.item()

        return bias

    def deposit_hill(self, cv_value: float, step_idx: int):
        """
        Deposit a new Gaussian hill at current CV value.

        For well-tempered metadynamics, height is scaled by:
            h = h_0 * exp(-V(cv_value) / (kT * (gamma - 1)))

        Args:
            cv_value: Current CV value (scalar)
            step_idx: Current diffusion step (for logging)
        """
        # Convert tensor to float if needed
        if torch.is_tensor(cv_value):
            cv_value = cv_value.item()

        # Compute hill height
        if self._well_tempered:
            current_bias = self.compute_explore_at_cv(cv_value)
            dT = self._kT * (self._bias_factor - 1)
            height = self._hill_height * torch.exp(torch.tensor(-current_bias / (dT + 1e-8))).item()
        else:
            height = self._hill_height

        # Add hill
        self.hills.append({
            'cv_center': cv_value,
            'height': height,
            'sigma': self._hill_sigma,
            'step': step_idx,
        })

        # Prune old hills if exceeding max
        if len(self.hills) > self._max_hills:
            self.hills = self.hills[-self._max_hills:]

    def compute_gradient(
        self,
        coords: torch.Tensor,
        feats: dict,
        parameters: dict,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Compute metadynamics gradient for steering.

        Uses chain rule: dV/dr = (dV/dCV) * (dCV/dr)

        Args:
            coords: [multiplicity, N_atoms, 3]
            feats: Feature dictionary
            parameters: Potential parameters (includes k, _step_idx, etc.)
            step: Current diffusion step (for gradient scaling)

        Returns:
            gradient: [multiplicity, N_atoms, 3]
        """
        # Check warmup and cutoff (0.0 = start of diffusion, 1.0 = end)
        progress = parameters.get('_relaxation', 0.0)
        warmup = self.parameters.get('warmup', 0.0)
        cutoff = self.parameters.get('cutoff', 0.75)
        if progress < warmup or progress > cutoff:
            return torch.zeros_like(coords)

        # Get force constant and step index
        k = parameters.get('k', 1.0)
        current_step = parameters.get('_step_idx', 0)

        # Compute CV and its gradient (pass current_step for async caching)
        cv_values, cv_gradient = self.cv_function(coords, feats, current_step)
        # cv_values: [multiplicity]
        # cv_gradient: [multiplicity, N_atoms, 3]

        # Compute bias energy and derivative w.r.t. CV
        energy, dV_dCV = self.compute_function(cv_values, k, compute_derivative=True)
        # energy: [multiplicity]
        # dV_dCV: [multiplicity]

        # For repulsion bias type, record the energy at each step
        if self._explore_type == 'repulsion':
            self.record_repulsion_energy(current_step, cv_values, energy)

        # Chain rule: dV/dr = dV_dCV * dCV/dr
        # dV_dCV: [multiplicity] -> [multiplicity, 1, 1]
        # cv_gradient: [multiplicity, N_atoms, 3]
        dV_dCV_expanded = dV_dCV.unsqueeze(-1).unsqueeze(-1)
        gradient = dV_dCV_expanded * cv_gradient

        # Apply gradient scaling if scaler is set
        if self.gradient_scaler is not None:
            progress = parameters.get('_relaxation', 0.0)
            gradient = self.gradient_scaler.apply(gradient, coords, feats, step, progress)

        return gradient

    def compute_args(self, feats: dict, parameters: dict):
        """
        Prepare arguments for compute_function.

        For metadynamics, we need the CV function and parameters.
        Returns empty index so base compute() safely returns zeros.
        """
        k = parameters.get('k', 1.0)
        return torch.empty(1, 0, dtype=torch.long), (k,), None, None, None

    def get_hill_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about deposited hills.

        Returns:
            Dict with hill statistics
        """
        if len(self.hills) == 0:
            return {
                'n_hills': 0,
                'total_height': 0.0,
                'cv_range': (0.0, 0.0),
            }

        cv_centers = [h['cv_center'] for h in self.hills]
        heights = [h['height'] for h in self.hills]

        return {
            'n_hills': len(self.hills),
            'total_height': sum(heights),
            'mean_height': sum(heights) / len(heights),
            'cv_range': (min(cv_centers), max(cv_centers)),
            'cv_mean': sum(cv_centers) / len(cv_centers),
        }
