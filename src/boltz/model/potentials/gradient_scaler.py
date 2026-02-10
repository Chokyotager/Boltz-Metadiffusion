"""
Gradient scaler and projector for CV-based per-coordinate gradient modification.

This module provides classes that modify gradients from steering, bias, SAXS, and
other potentials based on a CV's gradient:

- GradientScaler: Scales gradient magnitude per atom based on |dCV/dr|
- GradientProjector: Projects gradient onto CV gradient direction per atom
- GradientModifier: Combines scaling and projection with configurable order

Scaling Example:
    With Rg CV and strength=1.0:
    - Atoms with high |dRg/dr| get more steering weight
    - Atoms with low |dRg/dr| get less steering weight

Projection Example:
    With energy CV and strength=1.0:
    - Each atom's steering gradient is projected onto dE/dr direction
    - Steering only acts along the energy gradient direction
    - Atoms with zero dE/dr get zero steering (stable atoms don't move)
"""

import torch
from typing import Callable, Optional, Tuple


class GradientScaler:
    """
    Scales gradients based on per-atom CV gradient magnitudes.

    The scaler computes per-atom weights from the gradient magnitude of a scaling CV,
    then applies these weights to redistribute the steering gradient across atoms
    while preserving the per-sample total gradient magnitude.

    Weight computation:
        1. Compute scaling CV gradient: dCV/dr [mult, N_atoms, 3]
        2. Per-atom weight = |dCV/dr|_atom (gradient magnitude at each atom)
        3. If strength < 0: invert weights (1 / magnitude)
        4. Normalize per sample so mean(weights) = 1
        5. Apply strength to control sensitivity: weights^|strength|

    Magnitude preservation:
        For each sample, the total gradient magnitude is preserved:
        ||weighted_grad[i]|| = ||original_grad[i]||

    Attributes:
        cv_function: Callable that computes (cv_value, cv_gradient) from coords and feats
        strength: Controls direction and sensitivity of scaling
                  Positive: high |dCV/dr| → high weight
                  Negative: high |dCV/dr| → low weight (inverted)
        cv_name: Name of the scaling CV for logging/debugging
    """

    def __init__(
        self,
        cv_function: Callable[[torch.Tensor, dict, int], Tuple[torch.Tensor, torch.Tensor]],
        strength: float = 1.0,
        cv_name: str = "unknown",
        warmup: float = 0.0,
        cutoff: float = 1.0,
    ):
        """
        Initialize the gradient scaler.

        Args:
            cv_function: Function that takes (coords, feats, step) and returns
                        (cv_value, cv_gradient) where cv_gradient is [mult, N_atoms, 3]
            strength: Scaling sensitivity. Positive means high gradient → high weight.
                     Negative inverts the relationship. Magnitude controls sensitivity.
            cv_name: Name of the CV for logging purposes
            warmup: Start scaling after this fraction of diffusion (0.0 = from start)
            cutoff: Stop scaling after this fraction of diffusion (1.0 = until end)
        """
        self.cv_function = cv_function
        self.strength = strength
        self.cv_name = cv_name
        self.warmup = warmup
        self.cutoff = cutoff

        # Cache for avoiding redundant CV computation within same step
        self._cached_step: Optional[int] = None
        self._cached_weights: Optional[torch.Tensor] = None
        self._cached_coords_hash: Optional[int] = None

    def _coords_hash(self, coords: torch.Tensor) -> int:
        """Compute a hash for coordinate tensor to detect changes."""
        # Use shape and a sample of values for fast approximate hash
        return hash((coords.shape, float(coords.sum()), float(coords.std())))

    def compute_weights(
        self,
        coords: torch.Tensor,
        feats: dict,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Compute per-atom weights from scaling CV gradient magnitudes.

        Args:
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary for CV computation
            step: Current diffusion step (for caching)

        Returns:
            weights: Per-atom weights [multiplicity, N_atoms] with mean=1 per sample
        """
        coords_hash = self._coords_hash(coords)

        # Check cache
        if (self._cached_step == step and
            self._cached_weights is not None and
            self._cached_coords_hash == coords_hash):
            return self._cached_weights

        # Compute scaling CV and its gradient
        _, cv_gradient = self.cv_function(coords, feats, step)
        # cv_gradient: [multiplicity, N_atoms, 3]

        # Compute per-atom gradient magnitude
        grad_magnitude = torch.linalg.norm(cv_gradient, dim=-1)  # [mult, N_atoms]

        # Handle zero gradients
        grad_magnitude = grad_magnitude + 1e-10

        # Compute raw weights based on strength sign
        if self.strength >= 0:
            # High gradient magnitude → high weight
            raw_weights = grad_magnitude
        else:
            # Inverted: high gradient magnitude → low weight
            raw_weights = 1.0 / grad_magnitude

        # Normalize weights per sample so mean = 1
        mean_weights = raw_weights.mean(dim=-1, keepdim=True)
        weights = raw_weights / (mean_weights + 1e-10)

        # Apply strength to control sensitivity
        # weights^|strength| gives control over how extreme the weighting is
        abs_strength = abs(self.strength)
        if abs_strength != 1.0:
            weights = weights ** abs_strength
            # Re-normalize after power operation
            weights = weights / (weights.mean(dim=-1, keepdim=True) + 1e-10)

        # Cache results
        self._cached_step = step
        self._cached_weights = weights
        self._cached_coords_hash = coords_hash

        return weights

    def scale_gradient(
        self,
        gradient: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-atom weights to gradient with per-sample magnitude preservation.

        Args:
            gradient: Steering/bias gradient [multiplicity, N_atoms, 3]
            weights: Per-atom weights [multiplicity, N_atoms]

        Returns:
            scaled_gradient: Weighted gradient with preserved per-sample magnitude
        """
        # Expand weights for broadcasting: [mult, N_atoms] → [mult, N_atoms, 1]
        weights_expanded = weights.unsqueeze(-1)

        # Apply weights to gradient
        weighted_grad = weights_expanded * gradient

        # Preserve per-sample gradient magnitude
        # Compute original norm per sample
        orig_norm = torch.linalg.norm(
            gradient.reshape(gradient.shape[0], -1), dim=-1, keepdim=True
        )  # [mult, 1]

        # Compute weighted norm per sample
        weighted_norm = torch.linalg.norm(
            weighted_grad.reshape(weighted_grad.shape[0], -1), dim=-1, keepdim=True
        )  # [mult, 1]

        # Scale to preserve magnitude (avoid division by zero)
        scale_factor = orig_norm / (weighted_norm + 1e-10)

        # Reshape scale_factor for broadcasting: [mult, 1] → [mult, 1, 1]
        scale_factor = scale_factor.unsqueeze(-1)

        scaled_grad = weighted_grad * scale_factor

        return scaled_grad

    def apply(
        self,
        gradient: torch.Tensor,
        coords: torch.Tensor,
        feats: dict,
        step: int = 0,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """
        Convenience method to compute weights and scale gradient in one call.

        Args:
            gradient: Steering/bias gradient [multiplicity, N_atoms, 3]
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary for CV computation
            step: Current diffusion step
            progress: Diffusion progress (0.0 = start, 1.0 = end) for warmup/cutoff

        Returns:
            scaled_gradient: Weighted gradient with preserved per-sample magnitude
        """
        # Check warmup and cutoff
        if progress < self.warmup or progress > self.cutoff:
            return gradient  # Return unscaled gradient outside active window

        weights = self.compute_weights(coords, feats, step)
        return self.scale_gradient(gradient, weights)

    def clear_cache(self):
        """Clear the cached weights (call when starting new trajectory)."""
        self._cached_step = None
        self._cached_weights = None
        self._cached_coords_hash = None


class CompositeGradientScaler:
    """
    Combines multiple GradientScalers by multiplying their weights.

    When multiple scaling CVs are specified, their per-atom weights are
    multiplied together and then renormalized so mean(weights) = 1.

    Example:
        With energy and Rg scaling CVs:
        - energy_weights: [0.5, 1.5, 2.0, 0.8, 1.2]  (mean=1)
        - rg_weights:     [1.2, 0.8, 1.0, 1.5, 0.5]  (mean=1)
        - combined:       [0.6, 1.2, 2.0, 1.2, 0.6]  (renormalized to mean=1)

        Atoms that contribute highly to BOTH scaling CVs get the highest weights.
    """

    def __init__(self, scalers: list):
        """
        Initialize composite scaler from a list of GradientScalers.

        Args:
            scalers: List of GradientScaler instances
        """
        self.scalers = scalers
        self._cached_step = None
        self._cached_weights = None
        self._cached_coords_hash = None

    def _coords_hash(self, coords: torch.Tensor) -> int:
        """Compute a hash for coordinate tensor to detect changes."""
        return hash((coords.shape, float(coords.sum()), float(coords.std())))

    def compute_weights(
        self,
        coords: torch.Tensor,
        feats: dict,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Compute combined per-atom weights by multiplying individual scaler weights.

        Args:
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary for CV computation
            step: Current diffusion step (for caching)

        Returns:
            weights: Per-atom weights [multiplicity, N_atoms] with mean=1 per sample
        """
        coords_hash = self._coords_hash(coords)

        # Check cache
        if (self._cached_step == step and
            self._cached_weights is not None and
            self._cached_coords_hash == coords_hash):
            return self._cached_weights

        # Compute weights from each scaler and multiply
        combined_weights = None
        for scaler in self.scalers:
            weights = scaler.compute_weights(coords, feats, step)
            if combined_weights is None:
                combined_weights = weights.clone()
            else:
                combined_weights = combined_weights * weights

        # Renormalize so mean = 1 per sample
        if combined_weights is not None:
            mean_weights = combined_weights.mean(dim=-1, keepdim=True)
            combined_weights = combined_weights / (mean_weights + 1e-10)

        # Cache results
        self._cached_step = step
        self._cached_weights = combined_weights
        self._cached_coords_hash = coords_hash

        return combined_weights

    def scale_gradient(
        self,
        gradient: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply per-atom weights to gradient with per-sample magnitude preservation.

        Args:
            gradient: Steering/bias gradient [multiplicity, N_atoms, 3]
            weights: Per-atom weights [multiplicity, N_atoms]

        Returns:
            scaled_gradient: Weighted gradient with preserved per-sample magnitude
        """
        # Expand weights for broadcasting: [mult, N_atoms] → [mult, N_atoms, 1]
        weights_expanded = weights.unsqueeze(-1)

        # Apply weights to gradient
        weighted_grad = weights_expanded * gradient

        # Preserve per-sample gradient magnitude
        orig_norm = torch.linalg.norm(
            gradient.reshape(gradient.shape[0], -1), dim=-1, keepdim=True
        )
        weighted_norm = torch.linalg.norm(
            weighted_grad.reshape(weighted_grad.shape[0], -1), dim=-1, keepdim=True
        )

        scale_factor = orig_norm / (weighted_norm + 1e-10)
        scale_factor = scale_factor.unsqueeze(-1)

        return weighted_grad * scale_factor

    def apply(
        self,
        gradient: torch.Tensor,
        coords: torch.Tensor,
        feats: dict,
        step: int = 0,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """
        Convenience method to compute weights and scale gradient in one call.

        Args:
            gradient: Steering/bias gradient [multiplicity, N_atoms, 3]
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary for CV computation
            step: Current diffusion step
            progress: Diffusion progress (0.0 = start, 1.0 = end) for warmup/cutoff

        Returns:
            scaled_gradient: Weighted gradient with preserved per-sample magnitude
        """
        # For composite scalers, check if any scaler is active
        # Each individual scaler has its own warmup/cutoff
        active_scalers = [
            s for s in self.scalers
            if progress >= s.warmup and progress <= s.cutoff
        ]

        if not active_scalers:
            return gradient  # No active scalers, return unchanged

        # Compute weights only from active scalers
        combined_weights = None
        for scaler in active_scalers:
            weights = scaler.compute_weights(coords, feats, step)
            if combined_weights is None:
                combined_weights = weights.clone()
            else:
                combined_weights = combined_weights * weights

        # Renormalize so mean = 1 per sample
        if combined_weights is not None:
            mean_weights = combined_weights.mean(dim=-1, keepdim=True)
            combined_weights = combined_weights / (mean_weights + 1e-10)

        return self.scale_gradient(gradient, combined_weights)

    def clear_cache(self):
        """Clear the cached weights (call when starting new trajectory)."""
        self._cached_step = None
        self._cached_weights = None
        self._cached_coords_hash = None
        # Also clear caches in individual scalers
        for scaler in self.scalers:
            scaler.clear_cache()


class GradientProjector:
    """
    Projects gradients onto CV gradient direction per-atom.

    For each atom i, the gradient is projected onto the direction of dCV/dr_i.
    This constrains the steering to act only along the direction that changes
    the projection CV.

    Mathematical operation:
        ê_i = dCV/dr_i / ||dCV/dr_i||  (unit vector in CV gradient direction)
        g_projected_i = (g_i · ê_i) × ê_i  (projection onto CV gradient direction)
        g_final = (1 - strength) × g_original + strength × g_projected  (blend)
        ||g_final|| = ||g_original||  (magnitude preserved per sample)

    Example:
        With energy CV and strength=1.0:
        - Atom's steering gradient [1, 1, 0] with energy gradient [1, 0, 0]
        - Projected gradient: (1*1 + 1*0 + 0*0) × [1, 0, 0] = [1, 0, 0]
        - Steering only acts along the energy gradient direction

    Direction control:
        - "preserve": Keep sign from original gradient (default)
        - "toward": Force gradient toward lower CV values (e.g., toward stability)
        - "away": Force gradient toward higher CV values

    Attributes:
        cv_function: Callable that computes (cv_value, cv_gradient) from coords/feats
        strength: Blending factor (0=original, 1=fully projected)
        direction: "preserve", "toward", or "away" - controls sign of projected gradient
        cv_name: Name of the projection CV for logging/debugging
        zero_threshold: Below this gradient magnitude, direction is undefined
    """

    def __init__(
        self,
        cv_function: Callable[[torch.Tensor, dict, int], Tuple[torch.Tensor, torch.Tensor]],
        strength: float = 1.0,
        direction: str = "preserve",
        cv_name: str = "unknown",
        zero_threshold: float = 1e-8,
        warmup: float = 0.0,
        cutoff: float = 1.0,
    ):
        """
        Initialize the gradient projector.

        Args:
            cv_function: Function that takes (coords, feats, step) and returns
                        (cv_value, cv_gradient) where cv_gradient is [mult, N_atoms, 3]
            strength: Blending factor. 0.0 = original gradient, 1.0 = fully projected.
            direction: Controls sign of projected gradient:
                      "preserve" - keep sign from original gradient (default)
                      "toward" - force gradient toward lower CV values
                      "away" - force gradient toward higher CV values
            cv_name: Name of the CV for logging purposes
            zero_threshold: Threshold below which CV gradient direction is undefined
            warmup: Start projection after this fraction of diffusion (0.0 = from start)
            cutoff: Stop projection after this fraction of diffusion (1.0 = until end)
        """
        if direction not in {"preserve", "toward", "away"}:
            raise ValueError(f"Invalid direction: {direction}. Must be 'preserve', 'toward', or 'away'")
        self.cv_function = cv_function
        self.strength = strength
        self.direction = direction
        self.cv_name = cv_name
        self.zero_threshold = zero_threshold
        self.warmup = warmup
        self.cutoff = cutoff

        # Cache for avoiding redundant CV computation within same step
        self._cached_step: Optional[int] = None
        self._cached_directions: Optional[torch.Tensor] = None
        self._cached_valid_mask: Optional[torch.Tensor] = None
        self._cached_coords_hash: Optional[int] = None

    def _coords_hash(self, coords: torch.Tensor) -> int:
        """Compute a hash for coordinate tensor to detect changes."""
        return hash((coords.shape, float(coords.sum()), float(coords.std())))

    def compute_directions(
        self,
        coords: torch.Tensor,
        feats: dict,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-atom CV gradient directions (unit vectors).

        Args:
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary for CV computation
            step: Current diffusion step (for caching)

        Returns:
            directions: [mult, N_atoms, 3] unit vectors (or zero where undefined)
            valid_mask: [mult, N_atoms] bool mask where direction is defined
        """
        coords_hash = self._coords_hash(coords)

        # Check cache
        if (self._cached_step == step and
            self._cached_directions is not None and
            self._cached_coords_hash == coords_hash):
            return self._cached_directions, self._cached_valid_mask

        # Compute CV gradient
        _, cv_gradient = self.cv_function(coords, feats, step)
        # cv_gradient: [multiplicity, N_atoms, 3]

        # Compute magnitude per atom
        grad_magnitude = torch.linalg.norm(cv_gradient, dim=-1, keepdim=True)  # [mult, N_atoms, 1]

        # Create valid mask (where gradient is non-zero)
        valid_mask = (grad_magnitude.squeeze(-1) > self.zero_threshold)  # [mult, N_atoms]

        # Compute unit directions (safe division)
        directions = cv_gradient / (grad_magnitude + self.zero_threshold)  # [mult, N_atoms, 3]

        # Zero out directions where invalid
        directions = directions * valid_mask.unsqueeze(-1)

        # Cache results
        self._cached_step = step
        self._cached_directions = directions
        self._cached_valid_mask = valid_mask
        self._cached_coords_hash = coords_hash

        return directions, valid_mask

    def project_gradient(
        self,
        gradient: torch.Tensor,
        directions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project gradient onto directions with magnitude preservation.

        Args:
            gradient: [mult, N_atoms, 3] original steering gradient
            directions: [mult, N_atoms, 3] unit direction vectors (point toward higher CV)
            valid_mask: [mult, N_atoms] where projection is valid

        Returns:
            projected: [mult, N_atoms, 3] projected gradient (magnitude preserved)

        Note on direction:
            The `directions` tensor points toward higher CV values (e.g., higher energy).
            - "preserve": Keep sign from original gradient's projection
            - "toward": Force gradient toward LOWER CV values (negative direction)
            - "away": Force gradient toward HIGHER CV values (positive direction)
        """
        # Compute projection magnitude: (g · e)
        dot_product = (gradient * directions).sum(dim=-1, keepdim=True)  # [mult, N_atoms, 1]

        # Apply direction control
        if self.direction == "toward":
            # Force toward lower CV values (negative direction)
            # Use -|dot_product| so projected always points opposite to CV gradient
            dot_product = -torch.abs(dot_product)
        elif self.direction == "away":
            # Force toward higher CV values (positive direction)
            # Use |dot_product| so projected always points along CV gradient
            dot_product = torch.abs(dot_product)
        # else "preserve": keep original sign

        projected = dot_product * directions  # [mult, N_atoms, 3]

        # For invalid atoms (zero CV gradient), set projected to zero
        projected = projected * valid_mask.unsqueeze(-1)

        # Blend: g_final = (1-strength)*g_original + strength*g_projected
        blended = (1 - self.strength) * gradient + self.strength * projected

        # Preserve per-sample gradient magnitude
        orig_norm = torch.linalg.norm(
            gradient.reshape(gradient.shape[0], -1), dim=-1, keepdim=True
        )  # [mult, 1]
        blended_norm = torch.linalg.norm(
            blended.reshape(blended.shape[0], -1), dim=-1, keepdim=True
        )  # [mult, 1]

        # Scale to preserve magnitude (avoid division by zero)
        scale_factor = orig_norm / (blended_norm + 1e-10)
        scale_factor = scale_factor.unsqueeze(-1)  # [mult, 1, 1]

        return blended * scale_factor

    def apply(
        self,
        gradient: torch.Tensor,
        coords: torch.Tensor,
        feats: dict,
        step: int = 0,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """
        Convenience method to compute directions and project gradient in one call.

        Args:
            gradient: Steering/bias gradient [multiplicity, N_atoms, 3]
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary for CV computation
            step: Current diffusion step
            progress: Diffusion progress (0.0 = start, 1.0 = end) for warmup/cutoff

        Returns:
            projected_gradient: Projected gradient with preserved per-sample magnitude
        """
        # Check warmup and cutoff
        if progress < self.warmup or progress > self.cutoff:
            return gradient  # Return unprojected gradient outside active window

        directions, valid_mask = self.compute_directions(coords, feats, step)
        result = self.project_gradient(gradient, directions, valid_mask)
        return result

    def clear_cache(self):
        """Clear the cached directions (call when starting new trajectory)."""
        self._cached_step = None
        self._cached_directions = None
        self._cached_valid_mask = None
        self._cached_coords_hash = None


class CompositeGradientProjector:
    """
    Chains multiple GradientProjectors sequentially.

    When multiple projection CVs are specified, projections are applied one after
    another. Note that projection is NOT commutative - the order matters.

    Example:
        With energy and Rg projection CVs (in that order):
        1. First project onto energy gradient direction
        2. Then project result onto Rg gradient direction
        Final result is constrained to directions that affect both CVs.
    """

    def __init__(self, projectors: list):
        """
        Initialize composite projector from a list of GradientProjectors.

        Args:
            projectors: List of GradientProjector instances
        """
        self.projectors = projectors

    def apply(
        self,
        gradient: torch.Tensor,
        coords: torch.Tensor,
        feats: dict,
        step: int = 0,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply projections sequentially.

        Args:
            gradient: Steering/bias gradient [multiplicity, N_atoms, 3]
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary for CV computation
            step: Current diffusion step
            progress: Diffusion progress (0.0 = start, 1.0 = end) for warmup/cutoff

        Returns:
            projected_gradient: Gradient after all projections
        """
        result = gradient
        for projector in self.projectors:
            # Each projector checks its own warmup/cutoff internally
            result = projector.apply(result, coords, feats, step, progress)
        return result

    def clear_cache(self):
        """Clear all cached data."""
        for projector in self.projectors:
            projector.clear_cache()


class GradientModifier:
    """
    Combines gradient scaling and projection with configurable order.

    This class allows both scaling (magnitude redistribution) and projection
    (direction constraint) to be applied to the same gradient in a configurable
    order.

    Order options:
        - "scale_first": Apply scaling, then projection
        - "project_first": Apply projection, then scaling

    Example usage:
        modifier = GradientModifier(scaler, projector, order="scale_first")
        modified_grad = modifier.apply(gradient, coords, feats, step)

    Attributes:
        scaler: Optional GradientScaler or CompositeGradientScaler
        projector: Optional GradientProjector or CompositeGradientProjector
        order: "scale_first" or "project_first"
    """

    def __init__(
        self,
        scaler=None,
        projector=None,
        order: str = "scale_first",
    ):
        """
        Initialize the gradient modifier.

        Args:
            scaler: GradientScaler, CompositeGradientScaler, or None
            projector: GradientProjector, CompositeGradientProjector, or None
            order: "scale_first" or "project_first"
        """
        self.scaler = scaler
        self.projector = projector
        self.order = order

        if order not in ("scale_first", "project_first"):
            raise ValueError(f"Invalid order: {order}. Use 'scale_first' or 'project_first'")

    def apply(
        self,
        gradient: torch.Tensor,
        coords: torch.Tensor,
        feats: dict,
        step: int = 0,
        progress: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply scaling and/or projection in the configured order.

        Args:
            gradient: Steering/bias gradient [multiplicity, N_atoms, 3]
            coords: Atomic coordinates [multiplicity, N_atoms, 3]
            feats: Feature dictionary for CV computation
            step: Current diffusion step
            progress: Diffusion progress (0.0 = start, 1.0 = end) for warmup/cutoff

        Returns:
            modified_gradient: Gradient after scaling and/or projection
        """
        if self.order == "scale_first":
            if self.scaler is not None:
                gradient = self.scaler.apply(gradient, coords, feats, step, progress)
            if self.projector is not None:
                gradient = self.projector.apply(gradient, coords, feats, step, progress)
        else:  # project_first
            if self.projector is not None:
                gradient = self.projector.apply(gradient, coords, feats, step, progress)
            if self.scaler is not None:
                gradient = self.scaler.apply(gradient, coords, feats, step, progress)

        return gradient

    def clear_cache(self):
        """Clear all cached data."""
        if self.scaler is not None:
            self.scaler.clear_cache()
        if self.projector is not None:
            self.projector.clear_cache()
