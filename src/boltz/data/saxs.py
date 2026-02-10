"""SAXS P(r) data loading utilities for ensemble-averaged structure fitting."""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List, Union


def resample_pr(
    r_orig: np.ndarray,
    pr_orig: np.ndarray,
    bins: int,
    bins_range: Optional[Tuple[float, float]] = None,
    pr_errors: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Resample P(r) to a uniform grid using linear interpolation.

    This standardizes the bin count for consistent loss computation,
    particularly beneficial for W1/W2 optimal transport losses.

    Args:
        r_orig: Original distance grid, shape (N_orig,)
        pr_orig: Original P(r) values, shape (N_orig,)
        bins: Number of uniform bins for output
        bins_range: Optional (r_min, r_max) range in Angstroms.
                    If None, uses range from original data.
        pr_errors: Optional experimental errors, shape (N_orig,)

    Returns:
        Tuple of (r_new, pr_new, pr_errors_new)
        - r_new: Uniform distance grid, shape (bins,)
        - pr_new: Resampled P(r), shape (bins,)
        - pr_errors_new: Resampled errors if provided, else None
    """
    # Determine output range
    if bins_range is not None:
        r_min, r_max = bins_range
    else:
        r_min, r_max = r_orig[0], r_orig[-1]

    # Create uniform grid
    r_new = np.linspace(r_min, r_max, bins, dtype=np.float32)

    # Linear interpolation - values outside original range become 0
    # (P(r) should be 0 outside measured range)
    pr_new = np.interp(r_new, r_orig, pr_orig, left=0.0, right=0.0).astype(np.float32)

    # Interpolate errors if provided
    pr_errors_new = None
    if pr_errors is not None:
        pr_errors_new = np.interp(r_new, r_orig, pr_errors, left=0.0, right=0.0).astype(np.float32)

    return r_new, pr_new, pr_errors_new


def detect_gnom_units(filepath: Path) -> str:
    """
    Auto-detect whether GNOM file uses nm or Angstrom units.

    Heuristic: If max(r) < 50, assume nm (most proteins < 500 Å = 50 nm).
               If max(r) >= 50, assume Angstroms.

    Args:
        filepath: Path to GNOM .out file

    Returns:
        "nm" or "angstrom"
    """
    r_values = []
    in_pr_section = False

    with open(filepath, 'r') as f:
        for line in f:
            if 'Distance distribution' in line:
                in_pr_section = True
                continue

            if in_pr_section and line.strip().startswith('R'):
                continue

            if in_pr_section:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        r = float(parts[0])
                        r_values.append(r)
                    except ValueError:
                        break

    if not r_values:
        # Default to nm if we can't parse
        return "nm"

    max_r = max(r_values)
    # Threshold: 50 nm = 500 Å. Most proteins have Dmax well below this.
    # If max_r >= 50, file is likely already in Angstroms
    if max_r >= 50:
        return "angstrom"
    else:
        return "nm"


def parse_gnom_pr(filepath: Path, units_nm: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse GNOM output file to extract P(r) curve.

    Args:
        filepath: Path to GNOM .out file
        units_nm: If True, input data is in nanometers and will be converted to Angstroms

    Returns:
        Tuple of (r_grid, pr_values, pr_errors)
        - r_grid: Distance values in Angstroms, shape (N_bins,)
        - pr_values: P(r) values, shape (N_bins,)
        - pr_errors: Experimental errors, shape (N_bins,)
    """
    r_values = []
    pr_values = []
    pr_errors = []

    in_pr_section = False

    with open(filepath, 'r') as f:
        for line in f:
            # Detect start of P(r) section
            if 'Distance distribution' in line:
                in_pr_section = True
                continue

            # Skip header lines in P(r) section
            if in_pr_section and line.strip().startswith('R'):
                continue

            # Parse data lines
            if in_pr_section:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        r = float(parts[0])
                        pr = float(parts[1])
                        err = float(parts[2])
                        r_values.append(r)
                        pr_values.append(pr)
                        pr_errors.append(err)
                    except ValueError:
                        # End of data section
                        break

    if len(r_values) == 0:
        raise ValueError(f"No P(r) data found in {filepath}")

    r_array = np.array(r_values, dtype=np.float32)
    pr_array = np.array(pr_values, dtype=np.float32)
    err_array = np.array(pr_errors, dtype=np.float32)

    # Convert from nanometers to Angstroms if needed
    # When R → 10*R, we need P(r) → P(r)/10 to maintain ∫P(r)dr = 1
    if units_nm:
        r_array = r_array * 10.0  # nm → Å
        pr_array = pr_array / 10.0  # Adjust P(r) for new integration measure
        err_array = err_array / 10.0  # Scale errors too

    return r_array, pr_array, err_array


def normalize_pr(pr_values: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """
    Normalize P(r) curve: integral P(r) dr = 1.

    Uses trapezoidal rule for integration.

    Args:
        pr_values: P(r) values, shape (N_bins,)
        r_grid: Distance grid, shape (N_bins,)

    Returns:
        Normalized P(r) values
    """
    integral = np.trapz(pr_values, r_grid)
    if integral > 0:
        return pr_values / integral
    return pr_values


def compute_rg_from_pr(pr_values: np.ndarray, r_grid: np.ndarray) -> float:
    """
    Compute radius of gyration from P(r) distribution.

    Uses the formula: Rg = sqrt(∫ r² P(r) dr / (2 ∫ P(r) dr))

    For normalized P(r) where ∫ P(r) dr = 1, this simplifies to:
    Rg = sqrt((1/2) ∫ r² P(r) dr)

    Args:
        pr_values: P(r) values, shape (N_bins,)
        r_grid: Distance grid, shape (N_bins,)

    Returns:
        Rg in same units as r_grid (typically Angstroms)
    """
    # Normalize P(r) to ensure integral = 1
    integral_pr = np.trapz(pr_values, r_grid)
    if integral_pr <= 0:
        raise ValueError("P(r) integral must be positive")

    pr_normalized = pr_values / integral_pr

    # Compute Rg² = (1/2) ∫ r² P(r) dr
    integrand = r_grid**2 * pr_normalized
    rg_squared = 0.5 * np.trapz(integrand, r_grid)

    if rg_squared < 0:
        raise ValueError(f"Rg² computation gave negative value: {rg_squared}")

    rg = np.sqrt(rg_squared)
    return float(rg)


def load_experimental_pr(
    filepath: Path,
    normalize: bool = True,
    device: Optional[torch.device] = None,
    units: str = "auto",
    bins: Optional[int] = None,
    bins_range: Optional[Tuple[float, float]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load experimental P(r) data from GNOM file.

    Args:
        filepath: Path to GNOM .out file
        normalize: Whether to normalize integral to 1
        device: Torch device for tensors
        units: Input data units - "nm", "angstrom", or "auto" (auto-detect)
               Default is "auto" which detects based on max r value
        bins: Optional number of uniform bins to resample to.
              If None, uses original grid from file.
              Recommended: 48-64 for W1/W2 losses.
        bins_range: Optional (r_min, r_max) range in Angstroms for resampling.
                    If None, uses range from original data.

    Returns:
        Tuple of (r_grid, pr_exp, pr_errors) as torch tensors, all in Angstroms
    """
    # Resolve units
    if units == "auto":
        detected_units = detect_gnom_units(filepath)
        units_nm = (detected_units == "nm")
        print(f"Auto-detected SAXS units: {detected_units} (max_r threshold: 50)")
    else:
        units_nm = (units == "nm")

    r_grid, pr_values, pr_errors = parse_gnom_pr(filepath, units_nm=units_nm)

    # Resample to uniform grid if bins specified
    if bins is not None:
        r_grid, pr_values, pr_errors = resample_pr(
            r_grid, pr_values, bins, bins_range, pr_errors
        )
        print(f"Resampled P(r) to {bins} uniform bins, range: {r_grid[0]:.1f} - {r_grid[-1]:.1f} Å")

    if normalize:
        # Compute integral before normalization
        integral = np.trapz(pr_values, r_grid)
        if integral > 0:
            pr_values = pr_values / integral
            # Scale errors proportionally
            if pr_errors is not None:
                pr_errors = pr_errors / integral

    # Convert to torch tensors
    r_grid_t = torch.from_numpy(r_grid)
    pr_exp_t = torch.from_numpy(pr_values)
    pr_errors_t = torch.from_numpy(pr_errors) if pr_errors is not None else torch.zeros_like(pr_exp_t)

    if device is not None:
        r_grid_t = r_grid_t.to(device)
        pr_exp_t = pr_exp_t.to(device)
        pr_errors_t = pr_errors_t.to(device)

    return r_grid_t, pr_exp_t, pr_errors_t
