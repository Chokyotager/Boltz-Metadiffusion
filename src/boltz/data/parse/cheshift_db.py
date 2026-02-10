"""
CheShift database loader for CA/CB chemical shift prediction.

CheShift predicts CA and CB shifts based on backbone torsion angles (phi, psi)
and sidechain rotamer angles (chi1). It uses empirical lookup tables with
linear interpolation.

Reference:
    Martin et al. (2013) PNAS 110(42):16826-31
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


# Standard amino acids supported by CheShift
CHESHIFT_AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU',
    'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TYR', 'TRP', 'VAL'
]

# Amino acids without chi1 (use phi/psi only)
NO_CHI1_AA = {'ALA', 'GLY'}

# Amino acids with chi1 only (no chi2 filtering needed)
CHI1_ONLY_AA = {'SER', 'THR', 'VAL', 'CYS'}

# Chi1 rotamer bin centers for interpolation
CHI1_ROTAMERS = np.array([
    -180., -150., -120., -90., -60., -30., 0., 30., 60., 90., 120., 150., 180.
])


@dataclass
class CheShiftDatabase:
    """CheShift lookup tables for each amino acid."""
    # {aa_type: np.array of shape (N, 6) for [phi, psi, chi1_id, chi2_id, CA, CB]}
    data: Dict[str, np.ndarray]
    
    @classmethod
    def load(cls, db_path: Path) -> 'CheShiftDatabase':
        """Load CheShift database from directory."""
        data = {}
        cs_db_path = db_path / 'CS_DB'
        
        for aa in CHESHIFT_AMINO_ACIDS:
            db_file = cs_db_path / f'CS_db_{aa}'
            if db_file.exists():
                rows = []
                with open(db_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            values = [float(x) for x in line.split()]
                            rows.append(values)
                data[aa] = np.array(rows)
        
        return cls(data=data)
    
    def predict_shifts(
        self,
        aa_type: str,
        phi: float,
        psi: float,
        chi1: float = 0.0,
        chi2: float = 0.0,
    ) -> Tuple[float, float]:
        """Predict CA and CB shifts for given torsion angles.

        Args:
            aa_type: Three-letter amino acid code
            phi: Phi angle in degrees [-180, 180]
            psi: Psi angle in degrees [-180, 180]
            chi1: Chi1 angle in degrees [-180, 180] (ignored for ALA/GLY)
            chi2: Chi2 angle in degrees [-180, 180] (used for most residues)

        Returns:
            Tuple of (CA_shift, CB_shift) in ppm
        """
        from scipy.interpolate import griddata

        if aa_type not in self.data:
            return np.nan, np.nan

        db = self.data[aa_type]

        # Round angles to nearest 10 degrees for grid point lookup
        phi_round = int(round(phi / 10) * 10)
        psi_round = int(round(psi / 10) * 10)

        # Get phi range for interpolation
        if phi > phi_round:
            phi_range = [phi_round, phi_round + 10]
        else:
            if phi_round == -180:
                phi_round = 180
            phi_range = [phi_round - 10, phi_round]

        # Get psi range for interpolation
        if psi > psi_round:
            psi_range = [psi_round, psi_round + 10]
        else:
            if psi_round == -180:
                psi_round = 180
            psi_range = [psi_round - 10, psi_round]

        # Filter database for nearby grid points
        phi_list = []
        for phi_val in phi_range:
            y = int(phi_val * 0.1 + 19)
            if y > 37:
                y = -(37 - y)
            length = len(db) // 37
            end = length * y
            start = end - length
            for i in range(start, end):
                phi_list.append(db[i])

        if aa_type in NO_CHI1_AA:
            # ALA and GLY: use 2D interpolation (phi, psi)
            points = []
            values_ca = []
            values_cb = []

            for row in phi_list:
                for psi_val in psi_range:
                    if int(row[1]) == psi_val:
                        points.append((row[0], row[1]))
                        values_ca.append(row[4])
                        values_cb.append(row[5])

            if not points:
                return np.nan, np.nan

            points = np.array(points)
            ca = griddata(points, np.array(values_ca), (phi, psi), method='linear')
            cb = griddata(points, np.array(values_cb), (phi, psi), method='linear')

            return float(ca), float(cb)

        elif aa_type in CHI1_ONLY_AA:
            # SER, THR, VAL, CYS: use 3D interpolation (phi, psi, chi1) without chi2
            # Find two nearest chi1 rotamers
            idx = np.abs(CHI1_ROTAMERS - chi1).argmin()
            nearest_chi1_a = CHI1_ROTAMERS[idx]
            chi1_temp = np.delete(CHI1_ROTAMERS, idx)
            idx2 = np.abs(chi1_temp - chi1).argmin()
            nearest_chi1_b = chi1_temp[idx2]

            points = []
            values_ca = []
            values_cb = []

            for row in phi_list:
                for psi_val in psi_range:
                    if int(row[1]) == psi_val and (row[2] == nearest_chi1_a or row[2] == nearest_chi1_b):
                        points.append((row[0], row[1], row[2]))
                        values_ca.append(row[4])
                        values_cb.append(row[5])

            if not points:
                return np.nan, np.nan

            points = np.array(points)
            ca = griddata(points, np.array(values_ca), (phi, psi, chi1), method='linear')
            cb = griddata(points, np.array(values_cb), (phi, psi, chi1), method='linear')

            return float(ca), float(cb)

        else:
            # Other residues: use 3D interpolation (phi, psi, chi1) with chi2 filtering
            # Find two nearest chi1 rotamers
            idx = np.abs(CHI1_ROTAMERS - chi1).argmin()
            nearest_chi1_a = CHI1_ROTAMERS[idx]
            chi1_temp = np.delete(CHI1_ROTAMERS, idx)
            idx2 = np.abs(chi1_temp - chi1).argmin()
            nearest_chi1_b = chi1_temp[idx2]

            # Find nearest chi2 rotamer from database entries
            # Get unique chi2 values from first few entries
            chi2_rotamers = []
            for i in range(min(3, len(phi_list))):
                rotamer = phi_list[i][3]
                if rotamer < 0:
                    rotamer = rotamer + 360
                chi2_rotamers.append(rotamer)
            if 0. in chi2_rotamers:
                chi2_rotamers.append(360)
            chi2_rotamers = np.array(chi2_rotamers)

            # Convert chi2 to 0-360 range for comparison
            chi2_adj = chi2 + 360 if chi2 < 0 else chi2
            idx = np.abs(chi2_rotamers - chi2_adj).argmin()
            nearest_chi2 = chi2_rotamers[idx]
            # Convert back to -180 to 180 range
            if nearest_chi2 > 180:
                nearest_chi2 = nearest_chi2 - 360

            points = []
            values_ca = []
            values_cb = []

            for row in phi_list:
                for psi_val in psi_range:
                    if (int(row[1]) == psi_val and
                        row[3] == nearest_chi2 and
                        (row[2] == nearest_chi1_a or row[2] == nearest_chi1_b)):
                        points.append((row[0], row[1], row[2]))
                        values_ca.append(row[4])
                        values_cb.append(row[5])

            if not points:
                return np.nan, np.nan

            points = np.array(points)
            ca = griddata(points, np.array(values_ca), (phi, psi, chi1), method='linear')
            cb = griddata(points, np.array(values_cb), (phi, psi, chi1), method='linear')

            return float(ca), float(cb)


# Global database instance
_CHESHIFT_DB: Optional[CheShiftDatabase] = None


def get_cheshift_db(db_path: Optional[Path] = None) -> CheShiftDatabase:
    """Get or load the global CheShift database.
    
    Args:
        db_path: Path to CheShift database directory. If None, uses default.
        
    Returns:
        CheShiftDatabase instance
    """
    global _CHESHIFT_DB
    
    if _CHESHIFT_DB is None:
        if db_path is None:
            # Try default paths
            candidates = [
                Path('/data/b/cheshift/cheshift'),
                Path(__file__).parent.parent / 'cheshift',
            ]
            for candidate in candidates:
                if (candidate / 'CS_DB').exists():
                    db_path = candidate
                    break
            
            if db_path is None:
                raise RuntimeError("Could not find CheShift database")
        
        _CHESHIFT_DB = CheShiftDatabase.load(db_path)
    
    return _CHESHIFT_DB


class DifferentiableCheShift:
    """Differentiable CheShift predictor using PyTorch.

    Uses differentiable bilinear interpolation to match scipy griddata exactly
    while maintaining gradient flow.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the differentiable CheShift predictor.

        Args:
            db_path: Path to CheShift database
        """
        self.db = get_cheshift_db(db_path)
        self._build_grid_tables()

    def _build_grid_tables(self):
        """Build grid-based lookup tables for interpolation."""
        self.grids = {}

        for aa, data in self.db.data.items():
            # For ALA and GLY: 2D grid (phi, psi) -> (CA, CB)
            # For others: 3D grid (phi, psi, chi1) -> (CA, CB)

            if aa in NO_CHI1_AA:
                # Build 2D grid: phi from -180 to 180 (step 10), psi from -180 to 180 (step 10)
                # Grid is 37 x 37
                n_phi = 37
                n_psi = 37

                ca_grid = torch.zeros(n_phi, n_psi, dtype=torch.float32)
                cb_grid = torch.zeros(n_phi, n_psi, dtype=torch.float32)

                for row in data:
                    phi_val, psi_val = row[0], row[1]
                    ca_val, cb_val = row[4], row[5]

                    # Convert to grid indices
                    phi_idx = int((phi_val + 180) / 10) % 37
                    psi_idx = int((psi_val + 180) / 10) % 37

                    ca_grid[phi_idx, psi_idx] = ca_val
                    cb_grid[phi_idx, psi_idx] = cb_val

                self.grids[aa] = {
                    'ca': ca_grid,
                    'cb': cb_grid,
                    'has_chi1': False,
                }
            else:
                # Build 4D grid: phi (37) x psi (37) x chi1 (13) x chi2 (variable)
                # chi2 rotamer values vary by residue type
                n_phi = 37
                n_psi = 37
                n_chi1 = 13

                # Get unique chi2 values for this residue
                chi2_vals = sorted(set(row[3] for row in data))
                n_chi2 = len(chi2_vals)
                chi2_to_idx = {v: i for i, v in enumerate(chi2_vals)}

                ca_grid = torch.zeros(n_phi, n_psi, n_chi1, n_chi2, dtype=torch.float32)
                cb_grid = torch.zeros(n_phi, n_psi, n_chi1, n_chi2, dtype=torch.float32)

                for row in data:
                    phi_val, psi_val, chi1_val, chi2_val = row[0], row[1], row[2], row[3]
                    ca_val, cb_val = row[4], row[5]

                    # Convert to grid indices
                    phi_idx = int((phi_val + 180) / 10) % 37
                    psi_idx = int((psi_val + 180) / 10) % 37
                    chi1_idx = int((chi1_val + 180) / 30) % 13
                    chi2_idx = chi2_to_idx[chi2_val]

                    ca_grid[phi_idx, psi_idx, chi1_idx, chi2_idx] = ca_val
                    cb_grid[phi_idx, psi_idx, chi1_idx, chi2_idx] = cb_val

                self.grids[aa] = {
                    'ca': ca_grid,
                    'cb': cb_grid,
                    'has_chi1': True,
                    'chi2_vals': torch.tensor(chi2_vals, dtype=torch.float32),
                }

    def predict_shifts_torch(
        self,
        aa_type: str,
        phi: torch.Tensor,
        psi: torch.Tensor,
        chi1: Optional[torch.Tensor] = None,
        chi2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict CA and CB shifts using differentiable interpolation.

        Args:
            aa_type: Three-letter amino acid code
            phi: Phi angle(s) in degrees [-180, 180]
            psi: Psi angle(s) in degrees [-180, 180]
            chi1: Chi1 angle(s) in degrees (optional, required for most residues)
            chi2: Chi2 angle(s) in degrees (optional, selects nearest rotamer)

        Returns:
            Tuple of (CA_shift, CB_shift) tensors
        """
        if aa_type not in self.grids:
            nan = torch.tensor(float('nan'), device=phi.device, dtype=phi.dtype)
            return nan, nan

        grid_data = self.grids[aa_type]

        if not grid_data['has_chi1']:
            # Use differentiable triangular interpolation (matches scipy griddata)
            return self._triangular_interp(
                phi, psi,
                grid_data['ca'].to(phi.device),
                grid_data['cb'].to(phi.device),
            )
        else:
            # For residues with chi1/chi2, use 4D grid interpolation
            if chi1 is None:
                chi1 = torch.zeros_like(phi)
            return self._interp_with_chi2(
                phi, psi, chi1, chi2,
                grid_data['ca'].to(phi.device),
                grid_data['cb'].to(phi.device),
                grid_data['chi2_vals'].to(phi.device),
            )

    # Sentinel value threshold - values >= this are invalid
    SENTINEL = 900.0

    def _triangular_interp(
        self,
        phi: torch.Tensor,
        psi: torch.Tensor,
        ca_grid: torch.Tensor,
        cb_grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable triangular interpolation matching scipy griddata exactly.

        Grid is 37x37 covering phi,psi from -180 to 180 in steps of 10.
        Scipy griddata uses Delaunay triangulation which splits each grid cell
        into two triangles along the diagonal from (0,0) to (1,1).

        Returns 0.0 for invalid grid regions (sentinel values).
        """
        # Normalize to grid coordinates [0, 36]
        phi_norm = (phi + 180.0) / 10.0
        psi_norm = (psi + 180.0) / 10.0

        # Get integer indices and fractional parts (t, s in [0, 1])
        phi_idx = phi_norm.floor().long()
        psi_idx = psi_norm.floor().long()

        t = phi_norm - phi_idx.float()  # fractional phi
        s = psi_norm - psi_idx.float()  # fractional psi

        # Wrap indices for periodicity
        phi_idx0 = phi_idx % 37
        phi_idx1 = (phi_idx + 1) % 37
        psi_idx0 = psi_idx % 37
        psi_idx1 = (psi_idx + 1) % 37

        # Get the 4 corner values for CA
        ca_00 = ca_grid[phi_idx0, psi_idx0]  # (0, 0)
        ca_01 = ca_grid[phi_idx0, psi_idx1]  # (0, 1)
        ca_10 = ca_grid[phi_idx1, psi_idx0]  # (1, 0)
        ca_11 = ca_grid[phi_idx1, psi_idx1]  # (1, 1)

        # Get the 4 corner values for CB
        cb_00 = cb_grid[phi_idx0, psi_idx0]
        cb_01 = cb_grid[phi_idx0, psi_idx1]
        cb_10 = cb_grid[phi_idx1, psi_idx0]
        cb_11 = cb_grid[phi_idx1, psi_idx1]

        # Check for invalid values - sentinel (>=900) or negative (database errors)
        # Check CA and CB validity separately (GLY has no CB, so CB is always sentinel)
        ca_valid = ((ca_00 > 0) & (ca_00 < self.SENTINEL) &
                    (ca_01 > 0) & (ca_01 < self.SENTINEL) &
                    (ca_10 > 0) & (ca_10 < self.SENTINEL) &
                    (ca_11 > 0) & (ca_11 < self.SENTINEL))
        cb_valid = ((cb_00 > 0) & (cb_00 < self.SENTINEL) &
                    (cb_01 > 0) & (cb_01 < self.SENTINEL) &
                    (cb_10 > 0) & (cb_10 < self.SENTINEL) &
                    (cb_11 > 0) & (cb_11 < self.SENTINEL))

        # Triangular interpolation using barycentric coordinates
        # Delaunay triangulation splits cell into:
        #   Lower triangle (t > s): vertices (0,0), (1,0), (1,1)
        #   Upper triangle (t <= s): vertices (0,0), (0,1), (1,1)
        #
        # Barycentric weights for lower triangle (t > s):
        #   w0 = 1 - t, w1 = t - s, w2 = s
        #   result = w0 * v00 + w1 * v10 + w2 * v11
        #
        # Barycentric weights for upper triangle (t <= s):
        #   w0 = 1 - s, w1 = s - t, w2 = t
        #   result = w0 * v00 + w1 * v01 + w2 * v11

        # Compute both triangle results
        # Lower triangle: (0,0), (1,0), (1,1)
        ca_lower = (1 - t) * ca_00 + (t - s) * ca_10 + s * ca_11
        cb_lower = (1 - t) * cb_00 + (t - s) * cb_10 + s * cb_11

        # Upper triangle: (0,0), (0,1), (1,1)
        ca_upper = (1 - s) * ca_00 + (s - t) * ca_01 + t * ca_11
        cb_upper = (1 - s) * cb_00 + (s - t) * cb_01 + t * cb_11

        # Select based on which triangle the point is in (t > s means lower)
        lower_mask = (t > s).float()
        ca_shift = lower_mask * ca_lower + (1 - lower_mask) * ca_upper
        cb_shift = lower_mask * cb_lower + (1 - lower_mask) * cb_upper

        # Zero out invalid regions (CA and CB independently)
        ca_shift = ca_shift * ca_valid.float()
        cb_shift = cb_shift * cb_valid.float()

        return ca_shift, cb_shift

    def _interp_with_chi2(
        self,
        phi: torch.Tensor,
        psi: torch.Tensor,
        chi1: torch.Tensor,
        chi2: Optional[torch.Tensor],
        ca_grid: torch.Tensor,
        cb_grid: torch.Tensor,
        chi2_vals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable 3D tetrahedral interpolation.

        Grid is 37x37x13xN covering:
        - phi: -180 to 180 in steps of 10
        - psi: -180 to 180 in steps of 10
        - chi1: -180 to 180 in steps of 30
        - chi2: residue-specific rotamer values

        Algorithm:
        1. If chi2 specified: select nearest chi2 rotamer
           If chi2 not specified: compute for all chi2 rotamers and average
           (averaging approximates scipy's behavior which mixes chi2 through Delaunay)
        2. Find two nearest chi1 rotamers
        3. Perform 3D tetrahedral interpolation using Delaunay triangulation
        """
        n_chi2 = chi2_vals.shape[0]

        # Determine chi2 mode
        if chi2 is not None:
            chi2_adj = torch.where(chi2 < 0, chi2 + 360, chi2)
            chi2_vals_adj = torch.where(chi2_vals < 0, chi2_vals + 360, chi2_vals)
            chi2_expanded = chi2_adj.unsqueeze(-1) if chi2_adj.dim() == 0 else chi2_adj.unsqueeze(-1)
            chi2_dists = torch.abs(chi2_vals_adj - chi2_expanded)
            _, chi2_idx = chi2_dists.min(dim=-1)
            chi2_indices = [chi2_idx]
        else:
            # Average over all chi2 rotamers
            chi2_indices = [torch.full_like(phi, i, dtype=torch.long) if phi.dim() > 0
                           else torch.tensor(i, dtype=torch.long, device=phi.device)
                           for i in range(n_chi2)]

        # Chi1 rotamer values
        chi1_rotamers = torch.tensor(
            [-180., -150., -120., -90., -60., -30., 0., 30., 60., 90., 120., 150., 180.],
            device=chi1.device, dtype=chi1.dtype
        )

        # Find two nearest chi1 rotamers
        chi1_expanded = chi1.unsqueeze(-1) if chi1.dim() == 0 else chi1.unsqueeze(-1)
        chi1_dists = torch.abs(chi1_rotamers - chi1_expanded)

        min_dist1, idx1 = chi1_dists.min(dim=-1)
        ri0 = idx1 % 13

        chi1_dists_masked = chi1_dists.clone()
        if chi1.dim() == 0:
            chi1_dists_masked[idx1] = float('inf')
        else:
            chi1_dists_masked.scatter_(-1, idx1.unsqueeze(-1), float('inf'))
        _, idx2 = chi1_dists_masked.min(dim=-1)
        ri1 = idx2 % 13

        ri_min = torch.minimum(ri0, ri1)
        ri_max = torch.maximum(ri0, ri1)

        # Normalize coordinates
        t_full = (phi + 180.0) / 10.0
        s_full = (psi + 180.0) / 10.0

        ti = t_full.floor().long()
        si = s_full.floor().long()
        tf = t_full - ti.float()
        sf = s_full - si.float()

        chi1_min_val = chi1_rotamers[ri_min]
        chi1_max_val = chi1_rotamers[ri_max]
        chi1_range = chi1_max_val - chi1_min_val
        rf = torch.where(chi1_range > 0, (chi1 - chi1_min_val) / chi1_range, torch.zeros_like(chi1))

        ti0 = ti % 37
        ti1 = (ti + 1) % 37
        si0 = si % 37
        si1 = (si + 1) % 37

        t, s, r = tf, sf, rf

        # Compute for each chi2 and average (excluding sentinel values)
        ca_results = []
        cb_results = []
        ca_valid_masks = []
        cb_valid_masks = []

        for chi2_idx in chi2_indices:
            # Get 8 corner values for the cube
            ca_000 = ca_grid[ti0, si0, ri_min, chi2_idx]
            ca_001 = ca_grid[ti0, si0, ri_max, chi2_idx]
            ca_010 = ca_grid[ti0, si1, ri_min, chi2_idx]
            ca_011 = ca_grid[ti0, si1, ri_max, chi2_idx]
            ca_100 = ca_grid[ti1, si0, ri_min, chi2_idx]
            ca_101 = ca_grid[ti1, si0, ri_max, chi2_idx]
            ca_110 = ca_grid[ti1, si1, ri_min, chi2_idx]
            ca_111 = ca_grid[ti1, si1, ri_max, chi2_idx]

            cb_000 = cb_grid[ti0, si0, ri_min, chi2_idx]
            cb_001 = cb_grid[ti0, si0, ri_max, chi2_idx]
            cb_010 = cb_grid[ti0, si1, ri_min, chi2_idx]
            cb_011 = cb_grid[ti0, si1, ri_max, chi2_idx]
            cb_100 = cb_grid[ti1, si0, ri_min, chi2_idx]
            cb_101 = cb_grid[ti1, si0, ri_max, chi2_idx]
            cb_110 = cb_grid[ti1, si1, ri_min, chi2_idx]
            cb_111 = cb_grid[ti1, si1, ri_max, chi2_idx]

            # Check if all 8 corners are valid (not sentinel or negative)
            # Check CA and CB validity separately
            ca_valid = ((ca_000 > 0) & (ca_000 < self.SENTINEL) &
                        (ca_001 > 0) & (ca_001 < self.SENTINEL) &
                        (ca_010 > 0) & (ca_010 < self.SENTINEL) &
                        (ca_011 > 0) & (ca_011 < self.SENTINEL) &
                        (ca_100 > 0) & (ca_100 < self.SENTINEL) &
                        (ca_101 > 0) & (ca_101 < self.SENTINEL) &
                        (ca_110 > 0) & (ca_110 < self.SENTINEL) &
                        (ca_111 > 0) & (ca_111 < self.SENTINEL))
            cb_valid = ((cb_000 > 0) & (cb_000 < self.SENTINEL) &
                        (cb_001 > 0) & (cb_001 < self.SENTINEL) &
                        (cb_010 > 0) & (cb_010 < self.SENTINEL) &
                        (cb_011 > 0) & (cb_011 < self.SENTINEL) &
                        (cb_100 > 0) & (cb_100 < self.SENTINEL) &
                        (cb_101 > 0) & (cb_101 < self.SENTINEL) &
                        (cb_110 > 0) & (cb_110 < self.SENTINEL) &
                        (cb_111 > 0) & (cb_111 < self.SENTINEL))
            ca_valid_masks.append(ca_valid.float())
            cb_valid_masks.append(cb_valid.float())

            # 3D tetrahedral interpolation
            ca, cb = self._tet_interp_3d(
                t, s, r,
                ca_000, ca_001, ca_010, ca_011, ca_100, ca_101, ca_110, ca_111,
                cb_000, cb_001, cb_010, cb_011, cb_100, cb_101, cb_110, cb_111,
            )
            ca_results.append(ca)
            cb_results.append(cb)

        # Average over valid chi2 rotamers only (CA and CB independently)
        if len(ca_results) == 1:
            # Single chi2 - apply validity mask (zero out invalid)
            ca_shift = ca_results[0] * ca_valid_masks[0]
            cb_shift = cb_results[0] * cb_valid_masks[0]
        else:
            # Weighted average excluding invalid values
            ca_sum = torch.zeros_like(ca_results[0])
            cb_sum = torch.zeros_like(cb_results[0])
            ca_valid_count = torch.zeros_like(ca_results[0])
            cb_valid_count = torch.zeros_like(cb_results[0])

            for ca, cb, ca_mask, cb_mask in zip(ca_results, cb_results, ca_valid_masks, cb_valid_masks):
                ca_sum = ca_sum + ca * ca_mask
                cb_sum = cb_sum + cb * cb_mask
                ca_valid_count = ca_valid_count + ca_mask
                cb_valid_count = cb_valid_count + cb_mask

            # Avoid division by zero - use first valid result if none valid
            ca_valid_count = torch.clamp(ca_valid_count, min=1.0)
            cb_valid_count = torch.clamp(cb_valid_count, min=1.0)
            ca_shift = ca_sum / ca_valid_count
            cb_shift = cb_sum / cb_valid_count

        return ca_shift, cb_shift

    def _tet_interp_3d(
        self,
        t: torch.Tensor,
        s: torch.Tensor,
        r: torch.Tensor,
        ca_000: torch.Tensor, ca_001: torch.Tensor, ca_010: torch.Tensor, ca_011: torch.Tensor,
        ca_100: torch.Tensor, ca_101: torch.Tensor, ca_110: torch.Tensor, ca_111: torch.Tensor,
        cb_000: torch.Tensor, cb_001: torch.Tensor, cb_010: torch.Tensor, cb_011: torch.Tensor,
        cb_100: torch.Tensor, cb_101: torch.Tensor, cb_110: torch.Tensor, cb_111: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """3D tetrahedral interpolation using Delaunay triangulation.

        The cube is divided into 6 tetrahedra. Selection is based on:
        - s > r vs s <= r
        - t + s compared to 1
        - t + r compared to 1
        """
        def tet_interp(v0, v1, v2, v3, ca_vals, cb_vals, t, s, r):
            """Interpolate using barycentric coordinates."""
            T00, T01, T02 = v1[0] - v0[0], v2[0] - v0[0], v3[0] - v0[0]
            T10, T11, T12 = v1[1] - v0[1], v2[1] - v0[1], v3[1] - v0[1]
            T20, T21, T22 = v1[2] - v0[2], v2[2] - v0[2], v3[2] - v0[2]

            rhs0, rhs1, rhs2 = t - v0[0], s - v0[1], r - v0[2]

            det = (T00 * (T11 * T22 - T12 * T21) -
                   T01 * (T10 * T22 - T12 * T20) +
                   T02 * (T10 * T21 - T11 * T20))

            b1 = ((rhs0 * (T11 * T22 - T12 * T21) -
                   T01 * (rhs1 * T22 - T12 * rhs2) +
                   T02 * (rhs1 * T21 - T11 * rhs2)) / det)
            b2 = ((T00 * (rhs1 * T22 - T12 * rhs2) -
                   rhs0 * (T10 * T22 - T12 * T20) +
                   T02 * (T10 * rhs2 - rhs1 * T20)) / det)
            b3 = ((T00 * (T11 * rhs2 - rhs1 * T21) -
                   T01 * (T10 * rhs2 - rhs1 * T20) +
                   rhs0 * (T10 * T21 - T11 * T20)) / det)
            b0 = 1 - b1 - b2 - b3

            ca = b0 * ca_vals[0] + b1 * ca_vals[1] + b2 * ca_vals[2] + b3 * ca_vals[3]
            cb = b0 * cb_vals[0] + b1 * cb_vals[1] + b2 * cb_vals[2] + b3 * cb_vals[3]
            return ca, cb

        # Compute all 6 tetrahedra results
        ca_tet0, cb_tet0 = tet_interp((0, 1, 1), (0, 1, 0), (1, 0, 0), (0, 0, 0),
                                       [ca_011, ca_010, ca_100, ca_000],
                                       [cb_011, cb_010, cb_100, cb_000], t, s, r)
        ca_tet1, cb_tet1 = tet_interp((0, 1, 1), (0, 0, 1), (1, 0, 0), (0, 0, 0),
                                       [ca_011, ca_001, ca_100, ca_000],
                                       [cb_011, cb_001, cb_100, cb_000], t, s, r)
        ca_tet2, cb_tet2 = tet_interp((0, 1, 1), (1, 0, 1), (1, 1, 1), (1, 0, 0),
                                       [ca_011, ca_101, ca_111, ca_100],
                                       [cb_011, cb_101, cb_111, cb_100], t, s, r)
        ca_tet3, cb_tet3 = tet_interp((0, 1, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0),
                                       [ca_011, ca_110, ca_111, ca_100],
                                       [cb_011, cb_110, cb_111, cb_100], t, s, r)
        ca_tet4, cb_tet4 = tet_interp((0, 1, 1), (1, 1, 0), (0, 1, 0), (1, 0, 0),
                                       [ca_011, ca_110, ca_010, ca_100],
                                       [cb_011, cb_110, cb_010, cb_100], t, s, r)
        ca_tet5, cb_tet5 = tet_interp((0, 1, 1), (1, 0, 1), (0, 0, 1), (1, 0, 0),
                                       [ca_011, ca_101, ca_001, ca_100],
                                       [cb_011, cb_101, cb_001, cb_100], t, s, r)

        # Select tetrahedron based on conditions
        s_gt_r = (s > r).float()
        t_plus_s_gt_1 = (t + s > 1).float()
        t_plus_s_lt_1 = (t + s < 1).float()
        t_plus_r_gt_1 = (t + r > 1).float()

        w0 = s_gt_r * (1 - t_plus_s_gt_1)
        w1 = (1 - s_gt_r) * t_plus_s_lt_1 * (1 - t_plus_r_gt_1)
        w2 = (1 - s_gt_r) * (1 - t_plus_s_lt_1) * t_plus_r_gt_1
        w3 = s_gt_r * t_plus_s_gt_1 * t_plus_r_gt_1
        w4 = s_gt_r * t_plus_s_gt_1 * (1 - t_plus_r_gt_1)
        w5 = (1 - s_gt_r) * t_plus_s_lt_1 * t_plus_r_gt_1

        ca = w0 * ca_tet0 + w1 * ca_tet1 + w2 * ca_tet2 + w3 * ca_tet3 + w4 * ca_tet4 + w5 * ca_tet5
        cb = w0 * cb_tet0 + w1 * cb_tet1 + w2 * cb_tet2 + w3 * cb_tet3 + w4 * cb_tet4 + w5 * cb_tet5

        return ca, cb


def compute_dihedral_torch(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    p4: torch.Tensor,
) -> torch.Tensor:
    """Compute dihedral angle in degrees from 4 points.
    
    Args:
        p1, p2, p3, p4: Points as tensors of shape [..., 3]
        
    Returns:
        Dihedral angle in degrees, shape [...]
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    
    n1 = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
    n2 = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)
    
    b2_norm = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    m1 = torch.cross(n1, b2_norm, dim=-1)
    
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)

    # Negate to match IUPAC/PyMOL convention used by CheShift database
    return -torch.atan2(y, x) * 180.0 / np.pi
