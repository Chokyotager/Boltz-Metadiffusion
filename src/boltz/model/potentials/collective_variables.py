"""
Collective variables for metadynamics and enhanced sampling.

These functions compute collective variables (CVs) and their gradients
for use with metadynamics biasing during diffusion.
"""

import math
import torch
from typing import Optional, Tuple, Callable, Dict, List, Set


# =============================================================================
# Helper functions for gradient propagation
# =============================================================================

def build_bond_adjacency(feats: dict, n_atoms: int, device: torch.device) -> Dict[int, List[int]]:
    """
    Build adjacency dict from bond information in feats.

    Args:
        feats: Feature dictionary containing bond information
        n_atoms: Number of atoms
        device: Torch device

    Returns:
        Dict mapping atom_idx -> list of bonded neighbor indices
    """
    adjacency = {i: [] for i in range(n_atoms)}

    # Try ligand bonds first (explicit bond info)
    edge_index = feats.get('ligand_edge_index', None)
    bond_mask = feats.get('ligand_edge_bond_mask', None)

    if edge_index is not None and bond_mask is not None and edge_index.shape[1] > 0:
        bonded_edges = edge_index[:, bond_mask]
        for i in range(bonded_edges.shape[1]):
            a, b = bonded_edges[0, i].item(), bonded_edges[1, i].item()
            if a < n_atoms and b < n_atoms:
                # Only add if not already present (edge_index often has both directions)
                if b not in adjacency[a]:
                    adjacency[a].append(b)
        if sum(len(v) for v in adjacency.values()) > 0:
            return adjacency

    # Try explicit connections
    conn_idx = feats.get('connections_edge_index', None)
    if conn_idx is not None and conn_idx.shape[1] > 0:
        for i in range(conn_idx.shape[1]):
            a, b = conn_idx[0, i].item(), conn_idx[1, i].item()
            if a < n_atoms and b < n_atoms:
                # Only add if not already present
                if b not in adjacency[a]:
                    adjacency[a].append(b)
        if sum(len(v) for v in adjacency.values()) > 0:
            return adjacency

    # Fallback: infer bonds from atom names for proteins/nucleic acids
    adjacency = _infer_polymer_bonds(feats, n_atoms, device)

    return adjacency


def _infer_polymer_bonds(
    feats: dict,
    n_atoms: int,
    device: torch.device,
) -> Dict[int, List[int]]:
    """
    Infer bond connectivity for proteins and nucleic acids from atom names.

    For proteins: N-CA-C-O backbone + CA-CB sidechain + C(i)-N(i+1) peptide bonds
    For nucleic acids: Sugar-phosphate backbone + inter-nucleotide bonds

    Args:
        feats: Feature dictionary with ref_atom_name_chars and atom_to_token
        n_atoms: Number of atoms
        device: Torch device

    Returns:
        Dict mapping atom_idx -> list of bonded neighbor indices
    """
    adjacency = {i: [] for i in range(n_atoms)}

    ref_atom_name_chars = feats.get('ref_atom_name_chars', None)
    atom_to_token = feats.get('atom_to_token', None)

    if ref_atom_name_chars is None or atom_to_token is None:
        return adjacency

    # Strip batch dimensions if present
    if ref_atom_name_chars.dim() == 4:
        ref_atom_name_chars = ref_atom_name_chars[0]
    if atom_to_token.dim() == 3:
        atom_to_token = atom_to_token[0]

    # Check dimensions match
    if ref_atom_name_chars.shape[0] != n_atoms:
        return adjacency
    if atom_to_token.shape[0] != n_atoms:
        return adjacency

    # Group atoms by residue/token
    residue_atoms = {}  # token_idx -> {atom_name: atom_idx}
    for atom_idx in range(n_atoms):
        if atom_to_token.dim() == 2:
            token_idx = atom_to_token[atom_idx].argmax().item()
        else:
            token_idx = atom_to_token[atom_idx].item()

        atom_name = decode_atom_name(ref_atom_name_chars[atom_idx])
        if not atom_name:
            continue

        if token_idx not in residue_atoms:
            residue_atoms[token_idx] = {}
        residue_atoms[token_idx][atom_name] = atom_idx

    def add_bond(a1_idx, a2_idx):
        """Add bidirectional bond."""
        if a1_idx not in adjacency[a2_idx]:
            adjacency[a2_idx].append(a1_idx)
        if a2_idx not in adjacency[a1_idx]:
            adjacency[a1_idx].append(a2_idx)

    # Define standard bonds for proteins
    protein_intra_bonds = [
        ("N", "CA"), ("CA", "C"), ("C", "O"),  # Backbone
        ("CA", "CB"),  # Beta carbon
        # Common sidechain bonds from CB
        ("CB", "CG"), ("CB", "CG1"), ("CB", "CG2"), ("CB", "OG"), ("CB", "OG1"), ("CB", "SG"),
        ("CG", "CD"), ("CG", "CD1"), ("CG", "CD2"), ("CG", "OD1"), ("CG", "OD2"), ("CG", "ND1"), ("CG", "ND2"), ("CG", "SD"),
        ("CD", "CE"), ("CD", "NE"), ("CD", "OE1"), ("CD", "OE2"), ("CD", "NE2"),
        ("CE", "NZ"), ("CE", "SD"),
        ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
        ("CZ", "NH1"), ("CZ", "NH2"), ("CZ", "OH"),
        ("ND1", "CE1"), ("NE2", "CE1"),  # His ring
    ]

    # Define standard bonds for nucleic acids (DNA/RNA)
    nucleic_intra_bonds = [
        # Sugar-phosphate backbone
        ("P", "OP1"), ("P", "OP2"), ("P", "O5'"),
        ("O5'", "C5'"), ("C5'", "C4'"), ("C4'", "C3'"), ("C3'", "O3'"),
        # Sugar ring
        ("C4'", "O4'"), ("O4'", "C1'"), ("C1'", "C2'"), ("C2'", "C3'"),
        # RNA 2'-OH
        ("C2'", "O2'"),
        # Base attachment
        ("C1'", "N1"), ("C1'", "N9"),  # N1 for pyrimidines, N9 for purines
        # Pyrimidine base (C, T, U)
        ("N1", "C2"), ("C2", "O2"), ("C2", "N3"), ("N3", "C4"), ("C4", "N4"), ("C4", "O4"), ("C4", "C5"),
        ("C5", "C6"), ("C5", "C7"), ("C6", "N1"),  # C7 is methyl in T
        # Purine base (A, G)
        ("N9", "C4"), ("C4", "N3"), ("N3", "C2"), ("C2", "N1"), ("N1", "C6"), ("C6", "N6"), ("C6", "O6"),
        ("C6", "C5"), ("C5", "N7"), ("N7", "C8"), ("C8", "N9"), ("C5", "C4"),
        ("C2", "N2"),  # G has N2
    ]

    # Build intra-residue bonds
    for token_idx, atoms in residue_atoms.items():
        atom_names = set(atoms.keys())

        # Check if this looks like a nucleic acid (has sugar atoms)
        is_nucleic = any(name in atom_names for name in ["C1'", "C2'", "C3'", "C4'", "O4'", "P"])

        if is_nucleic:
            bonds_to_try = nucleic_intra_bonds
        else:
            bonds_to_try = protein_intra_bonds

        for a1_name, a2_name in bonds_to_try:
            if a1_name in atoms and a2_name in atoms:
                add_bond(atoms[a1_name], atoms[a2_name])

    # Build inter-residue bonds (peptide bonds for proteins, phosphodiester for nucleic acids)
    # Only connect residues that are truly adjacent (consecutive token indices)
    sorted_tokens = sorted(residue_atoms.keys())
    for i in range(len(sorted_tokens) - 1):
        t1, t2 = sorted_tokens[i], sorted_tokens[i + 1]

        # Skip if tokens are not consecutive (gap in sequence)
        if t2 != t1 + 1:
            continue

        atoms1, atoms2 = residue_atoms[t1], residue_atoms[t2]

        # Protein peptide bond: C(i) - N(i+1)
        if "C" in atoms1 and "N" in atoms2:
            add_bond(atoms1["C"], atoms2["N"])

        # Nucleic acid phosphodiester bond: O3'(i) - P(i+1)
        if "O3'" in atoms1 and "P" in atoms2:
            add_bond(atoms1["O3'"], atoms2["P"])

    return adjacency


def propagate_gradient_to_neighbors(
    gradient: torch.Tensor,
    source_atoms: List[int],
    adjacency: Dict[int, List[int]],
    max_hops: int = 10,
    decay: float = 0.5,
) -> torch.Tensor:
    """
    Propagate gradient from source atoms to bonded neighbors.

    Args:
        gradient: [multiplicity, n_atoms, 3] - original sparse gradient
        source_atoms: indices of atoms with non-zero gradient
        adjacency: bond adjacency dict
        max_hops: maximum bond distance for propagation
        decay: decay factor per hop (0.5 = half strength per bond)

    Returns:
        Enhanced gradient with propagated values
    """
    enhanced = gradient.clone()

    for source in source_atoms:
        source_grad = gradient[:, source, :]  # [mult, 3]
        if source_grad.norm() < 1e-8:
            continue

        # BFS to find neighbors within max_hops
        visited = {source: 0}  # atom -> hop distance
        queue = [(source, 0)]

        while queue:
            atom, dist = queue.pop(0)
            if dist >= max_hops:
                continue
            for neighbor in adjacency.get(atom, []):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        # Apply decayed gradient to neighbors
        for neighbor, hop_dist in visited.items():
            if neighbor == source:
                continue
            weight = decay ** hop_dist
            enhanced[:, neighbor, :] += weight * source_grad

    return enhanced


def propagate_gradient_with_barriers(
    gradient: torch.Tensor,
    source_atom: int,
    adjacency: Dict[int, List[int]],
    barrier_atoms: Set[int],
    max_hops: int = 10,
    decay: float = 0.5,
) -> torch.Tensor:
    """
    Propagate gradient from a single source atom, stopping at barrier atoms.

    This prevents gradient cancellation when propagating from opposite ends
    of an angle or dihedral - each side propagates independently without
    crossing through the vertex/central atoms.

    Args:
        gradient: [multiplicity, n_atoms, 3] - gradient tensor to enhance
        source_atom: index of atom to propagate from
        adjacency: bond adjacency dict
        barrier_atoms: set of atom indices that block propagation
        max_hops: maximum bond distance for propagation
        decay: decay factor per hop (0.5 = half strength per bond)

    Returns:
        Enhanced gradient with propagated values (modified in place)
    """
    source_grad = gradient[:, source_atom, :]  # [mult, 3]
    if source_grad.norm() < 1e-8:
        return gradient

    # BFS to find neighbors within max_hops, but don't cross barriers
    visited = {source_atom: 0}  # atom -> hop distance
    queue = [(source_atom, 0)]

    while queue:
        atom, dist = queue.pop(0)
        if dist >= max_hops:
            continue
        for neighbor in adjacency.get(atom, []):
            if neighbor not in visited:
                # Don't propagate through barrier atoms, but do visit them
                # (they get gradient but don't propagate further)
                visited[neighbor] = dist + 1
                if neighbor not in barrier_atoms:
                    queue.append((neighbor, dist + 1))

    # Apply decayed gradient to neighbors
    for neighbor, hop_dist in visited.items():
        if neighbor == source_atom:
            continue
        weight = decay ** hop_dist
        gradient[:, neighbor, :] += weight * source_grad

    return gradient


def propagate_rotational_gradient_with_barriers(
    coords: torch.Tensor,
    gradient: torch.Tensor,
    source_atom: int,
    pivot_atom: int,
    rotation_axis: torch.Tensor,
    adjacency: Dict[int, List[int]],
    barrier_atoms: Set[int],
    sign: float = 1.0,
    max_hops: int = 10,
    decay: float = 0.5,
) -> torch.Tensor:
    """
    Propagate rotational gradient from a source atom, stopping at barriers.

    Unlike simple gradient copying, this computes the correct rotational tangent
    for each propagated atom based on its position relative to the pivot.

    The tangent is: cross(rotation_axis, r - pivot)
    This gives the direction an atom would move for a rotation around the axis.

    Args:
        coords: [multiplicity, n_atoms, 3] - coordinates
        gradient: [multiplicity, n_atoms, 3] - gradient tensor to enhance
        source_atom: index of atom to propagate from
        pivot_atom: index of pivot point (vertex of angle)
        rotation_axis: [multiplicity, 3] - unit rotation axis
        adjacency: bond adjacency dict
        barrier_atoms: set of atom indices that block propagation
        sign: +1.0 or -1.0 to control rotation direction
        max_hops: maximum bond distance for propagation
        decay: decay factor per hop (0.5 = half strength per bond)

    Returns:
        Enhanced gradient with rotational tangents (modified in place)
    """
    source_grad = gradient[:, source_atom, :]  # [mult, 3]
    if source_grad.norm() < 1e-8:
        return gradient

    # BFS to find neighbors within max_hops, but don't cross barriers
    visited = {source_atom: 0}  # atom -> hop distance
    queue = [(source_atom, 0)]

    while queue:
        atom, dist = queue.pop(0)
        if dist >= max_hops:
            continue
        for neighbor in adjacency.get(atom, []):
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                if neighbor not in barrier_atoms:
                    queue.append((neighbor, dist + 1))

    # Apply rotational gradient to neighbors
    pivot_coord = coords[:, pivot_atom, :]  # [mult, 3]

    for neighbor, hop_dist in visited.items():
        if neighbor == source_atom:
            continue

        weight = decay ** hop_dist

        # Compute relative position from pivot
        r_rel = coords[:, neighbor, :] - pivot_coord  # [mult, 3]

        # Compute rotational tangent: cross(axis, r_rel)
        tangent = torch.cross(rotation_axis, r_rel, dim=-1)  # [mult, 3]

        # Normalize tangent magnitude relative to source gradient
        tangent_norm = tangent.norm(dim=-1, keepdim=True) + 1e-8
        source_norm = source_grad.norm(dim=-1, keepdim=True) + 1e-8
        tangent_scaled = tangent * (source_norm / tangent_norm)

        gradient[:, neighbor, :] += sign * weight * tangent_scaled

    return gradient


def compute_region_com(
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute center of mass for a region defined by a mask.

    Args:
        coords: [multiplicity, n_atoms, 3] coordinates
        mask: [n_atoms] boolean mask for atoms in region

    Returns:
        com: [multiplicity, 3] center of mass for each sample
    """
    mask_float = mask.float().to(coords.device)
    n_atoms_in_region = mask_float.sum() + 1e-8
    # Weighted sum: [mult, n_atoms, 3] * [n_atoms, 1] -> sum -> [mult, 3]
    com = (coords * mask_float.view(1, -1, 1)).sum(dim=1) / n_atoms_in_region
    return com


# =============================================================================
# Atom type identification helpers (for CVs that need specific atom types)
# =============================================================================

# Protein backbone atom names
BACKBONE_ATOM_NAMES = ["N", "CA", "C", "O"]

# Charged residue information (for salt bridges)
# Positive: Lys (NZ), Arg (NE, CZ, NH1, NH2), His (can be positive at low pH)
# Negative: Asp (OD1, OD2), Glu (OE1, OE2)
POSITIVE_RESIDUES = ["LYS", "ARG"]
NEGATIVE_RESIDUES = ["ASP", "GLU"]
POSITIVE_ATOMS = {
    "LYS": ["NZ"],
    "ARG": ["NE", "NH1", "NH2"],  # CZ is carbon, not charged
}
NEGATIVE_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}


def decode_atom_name(atom_name_chars: torch.Tensor) -> str:
    """
    Decode atom name from one-hot character tensor.

    Boltz encodes atom names as: ord(char) - 32, then one-hot with 64 classes.
    So 'N' -> 78 - 32 = 46, 'C' -> 67 - 32 = 35, 'A' -> 65 - 32 = 33, etc.

    Args:
        atom_name_chars: [4, num_chars] or [num_chars] one-hot encoded characters

    Returns:
        Atom name string (e.g., "CA", "N", "OD1")
    """
    if atom_name_chars.dim() == 1:
        # Single character, find argmax and add 32 back
        idx = atom_name_chars.argmax().item()
        return chr(idx + 32) if idx > 0 else ""

    # Multiple characters [4, num_chars] - one-hot for each position
    name = ""
    for i in range(atom_name_chars.shape[0]):
        idx = atom_name_chars[i].argmax().item()
        if idx > 0:  # 0 is typically padding
            # Add 32 back to convert from Boltz encoding to ASCII
            char = chr(idx + 32) if idx < 96 else ""  # 96 + 32 = 128 (max ASCII)
            if char.isalnum() or char in "'*+-":  # Allow more special chars for modified residues
                name += char
    return name.strip()


def get_backbone_atom_mask(
    feats: dict,
    n_atoms: int,
    atom_name: str = "CA",
) -> torch.Tensor:
    """
    Create a mask for specific backbone atoms (CA, N, C, O).

    This uses the ref_atom_name_chars feature from Boltz to identify atoms.
    If ref_atom_name_chars is not available, falls back to assuming one atom per residue.

    Args:
        feats: Feature dictionary from Boltz
        n_atoms: Total number of atoms
        atom_name: Which backbone atom to select ("CA", "N", "C", "O")

    Returns:
        mask: [n_atoms] boolean tensor, True for matching atoms
    """
    device = feats.get('atom_pad_mask', torch.zeros(1)).device

    # Try to get atom name information
    ref_atom_name_chars = feats.get('ref_atom_name_chars', None)

    if ref_atom_name_chars is not None:
        # Strip batch dimension if present
        # ref_atom_name_chars: [batch, n_atoms, 4, num_chars] -> [n_atoms, 4, num_chars]
        if ref_atom_name_chars.dim() == 4:
            ref_atom_name_chars = ref_atom_name_chars[0]

        if ref_atom_name_chars.shape[0] == n_atoms:
            # ref_atom_name_chars is [n_atoms, 4, num_chars] or [n_atoms, num_chars]
            mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)

            # Decode each atom name and check if it matches
            for i in range(n_atoms):
                decoded = decode_atom_name(ref_atom_name_chars[i])
                if decoded == atom_name:
                    mask[i] = True

            return mask

    # Fallback: use atom_to_token to identify one atom per residue
    # This assumes the first atom of each token/residue is the representative
    atom_to_token = feats.get('atom_to_token', None)
    if atom_to_token is not None:
        # atom_to_token can be [batch, n_atoms, n_tokens] or [n_atoms, n_tokens] or [n_atoms]
        # Strip batch dimension if present
        if atom_to_token.dim() == 3:
            atom_to_token = atom_to_token[0]  # [n_atoms, n_tokens]

        if atom_to_token.dim() == 2:
            token_idx = atom_to_token.argmax(dim=1)  # [n_atoms]
        else:
            token_idx = atom_to_token

        # Mark first atom of each token (crude approximation for CA)
        mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        seen_tokens = set()
        for i in range(min(n_atoms, len(token_idx))):
            t = token_idx[i].item()
            if t not in seen_tokens:
                seen_tokens.add(t)
                mask[i] = True
        return mask

    # Ultimate fallback: assume sequential atoms, pick every ~5th (avg residue size)
    # This is very crude but better than nothing
    mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
    mask[::5] = True  # Every 5th atom as crude CA approximation
    return mask


def get_charged_atom_masks(
    feats: dict,
    n_atoms: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masks for positively and negatively charged atoms.

    For salt bridges:
    - Positive: Lys NZ, Arg NE/NH1/NH2
    - Negative: Asp OD1/OD2, Glu OE1/OE2

    Args:
        feats: Feature dictionary from Boltz
        n_atoms: Total number of atoms

    Returns:
        positive_mask: [n_atoms] boolean tensor for positive atoms
        negative_mask: [n_atoms] boolean tensor for negative atoms
    """
    device = feats.get('atom_pad_mask', torch.zeros(1)).device

    positive_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
    negative_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)

    # Need both atom names and residue types
    ref_atom_name_chars = feats.get('ref_atom_name_chars', None)
    atom_to_token = feats.get('atom_to_token', None)
    res_type = feats.get('res_type', None)

    if ref_atom_name_chars is None or atom_to_token is None or res_type is None:
        # Can't determine charged atoms without this info
        # Return empty masks (will effectively disable the CV)
        return positive_mask, negative_mask

    # Import const for token IDs
    try:
        from boltz.data import const
        token_ids = const.token_ids
    except ImportError:
        return positive_mask, negative_mask

    # Strip batch dimensions if present
    # ref_atom_name_chars: [batch, n_atoms, 4, 64] -> [n_atoms, 4, 64]
    if ref_atom_name_chars.dim() == 4:
        ref_atom_name_chars = ref_atom_name_chars[0]

    # atom_to_token: [batch, n_atoms, n_tokens] -> [n_atoms, n_tokens]
    if atom_to_token.dim() == 3:
        atom_to_token = atom_to_token[0]

    # res_type: [batch, n_tokens] or [batch, n_tokens, num_types] -> strip batch
    if res_type.dim() == 3:
        res_type = res_type[0]
    elif res_type.dim() == 2 and res_type.shape[0] == 1:
        # Might be [1, n_tokens] batch dim
        res_type = res_type[0]

    # Get token index for each atom
    if atom_to_token.dim() == 2:
        token_idx = atom_to_token.argmax(dim=1)  # [n_atoms]
    else:
        token_idx = atom_to_token

    # res_type is [n_tokens, num_res_types] one-hot or [n_tokens] indices
    if res_type.dim() == 2:
        res_type_idx = res_type.argmax(dim=1)  # [n_tokens]
    else:
        res_type_idx = res_type

    # Check each atom
    for i in range(min(n_atoms, len(token_idx))):
        atom_name = decode_atom_name(ref_atom_name_chars[i])
        t_idx = token_idx[i].item()
        if t_idx >= len(res_type_idx):
            continue
        res_idx = res_type_idx[t_idx].item()

        # Check for positive residues
        for res_name in POSITIVE_RESIDUES:
            if res_name in token_ids and res_idx == token_ids[res_name]:
                if atom_name in POSITIVE_ATOMS.get(res_name, []):
                    positive_mask[i] = True
                break

        # Check for negative residues
        for res_name in NEGATIVE_RESIDUES:
            if res_name in token_ids and res_idx == token_ids[res_name]:
                if atom_name in NEGATIVE_ATOMS.get(res_name, []):
                    negative_mask[i] = True
                break

    return positive_mask, negative_mask


def get_backbone_donor_acceptor_masks(
    feats: dict,
    n_atoms: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masks for H-bond donors and acceptors using backbone heavy atoms.

    Since Boltz doesn't predict hydrogens, we use:
    - Donors: Backbone N atoms (the H is attached to N)
    - Acceptors: Backbone O atoms (carbonyl oxygen)

    Args:
        feats: Feature dictionary from Boltz
        n_atoms: Total number of atoms

    Returns:
        donor_mask: [n_atoms] boolean tensor for donor heavy atoms (N)
        acceptor_mask: [n_atoms] boolean tensor for acceptor atoms (O)
    """
    donor_mask = get_backbone_atom_mask(feats, n_atoms, atom_name="N")
    acceptor_mask = get_backbone_atom_mask(feats, n_atoms, atom_name="O")

    return donor_mask, acceptor_mask


def angle_region_cv(
    coords: torch.Tensor,
    feats: dict,
    region1_mask: torch.Tensor,
    region2_mask: torch.Tensor,
    region3_mask: torch.Tensor,
    max_hops: int = 10,
    decay: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute angle between three regions using their centers of mass.

    The angle is measured at region2 (vertex/hinge), between vectors to region1 and region3.
    Gradients are computed as ROTATIONAL motion around the hinge axis to enable
    proper rigid-body domain movement, then propagated through bonds.

    The gradient points in the direction that INCREASES the angle (opens the hinge).
    Steering will use (current - target) * gradient, so if current > target,
    it will subtract gradient to decrease the angle.

    Args:
        coords: [multiplicity, n_atoms, 3] coordinates
        feats: Feature dictionary
        region1_mask: [n_atoms] mask for first region (domain 1)
        region2_mask: [n_atoms] mask for vertex region (hinge)
        region3_mask: [n_atoms] mask for third region (domain 2)
        max_hops: maximum bond distance for gradient propagation
        decay: decay factor per hop (0.5 = half strength per bond)

    Returns:
        angle: [multiplicity] angle in radians
        gradient: [multiplicity, n_atoms, 3] dAngle/dr (rotational + propagated)
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Move masks to device
    region1_mask = region1_mask.to(device)
    region2_mask = region2_mask.to(device)
    region3_mask = region3_mask.to(device)

    # Compute COMs
    com1 = compute_region_com(coords, region1_mask)  # [mult, 3] domain 1
    com2 = compute_region_com(coords, region2_mask)  # [mult, 3] hinge (vertex)
    com3 = compute_region_com(coords, region3_mask)  # [mult, 3] domain 2

    # Vectors from hinge to domains
    v1 = com1 - com2  # [mult, 3] hinge -> domain1
    v2 = com3 - com2  # [mult, 3] hinge -> domain2

    # Compute angle
    v1_norm = torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8
    v2_norm = torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8
    v1_unit = v1 / v1_norm
    v2_unit = v2 / v2_norm

    cos_theta = torch.clamp((v1_unit * v2_unit).sum(dim=-1), -1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # [mult]

    # Rotation axis: perpendicular to the plane of v1 and v2
    # n = v1 × v2 (right-hand rule)
    rotation_axis = torch.cross(v1, v2, dim=-1)  # [mult, 3]
    axis_norm = torch.linalg.norm(rotation_axis, dim=-1, keepdim=True) + 1e-8
    rotation_axis = rotation_axis / axis_norm  # unit vector [mult, 3]

    # Initialize gradient
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)

    # For domain 1: gradient = -(rotation_axis × (r - com2))
    # cross(axis, r_rel) gives tangent for rotation that DECREASES angle (v1 toward v2)
    # We negate to get gradient that INCREASES angle (standard convention: gradient = ∂θ/∂r)
    region1_coords = coords[:, region1_mask, :]  # [mult, n1, 3]
    r1_rel = region1_coords - com2.unsqueeze(1)  # [mult, n1, 3] relative to hinge
    axis_expanded = rotation_axis.unsqueeze(1)  # [mult, 1, 3]
    tangent1 = torch.cross(axis_expanded.expand_as(r1_rel), r1_rel, dim=-1)  # [mult, n1, 3]
    gradient[:, region1_mask, :] = -tangent1  # Negated: direction that increases angle

    # For domain 2: opposite rotation to domain 1
    region3_coords = coords[:, region3_mask, :]  # [mult, n3, 3]
    r3_rel = region3_coords - com2.unsqueeze(1)  # [mult, n3, 3]
    tangent3 = torch.cross(axis_expanded.expand_as(r3_rel), r3_rel, dim=-1)  # [mult, n3, 3]
    gradient[:, region3_mask, :] = tangent3  # No negation: opposite to domain 1

    # Hinge region: zero gradient (pivot point)
    # Already zero from initialization

    # Propagate gradients through bonds to neighbors
    if max_hops > 0:
        adjacency = build_bond_adjacency(feats, n_atoms, device)
        # Source atoms are all atoms in region1 and region3
        source_atoms = torch.where(region1_mask | region3_mask)[0].tolist()
        gradient = propagate_gradient_to_neighbors(
            gradient, source_atoms, adjacency, max_hops, decay
        )

    # Normalize gradient so max magnitude is 1
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return theta, gradient


def dihedral_region_cv(
    coords: torch.Tensor,
    feats: dict,
    region1_mask: torch.Tensor,
    region2_mask: torch.Tensor,
    region3_mask: torch.Tensor,
    region4_mask: torch.Tensor,
    max_hops: int = 10,
    decay: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dihedral angle between four regions using their centers of mass.

    The dihedral is the angle between planes (COM1-COM2-COM3) and (COM2-COM3-COM4).
    Gradients are computed as ROTATIONAL motion around the central axis (COM2-COM3)
    to enable proper rigid-body domain movement, then propagated through bonds.

    The gradient points in the direction that INCREASES the dihedral angle
    (standard convention: gradient = ∂φ/∂r).

    Args:
        coords: [multiplicity, n_atoms, 3] coordinates
        feats: Feature dictionary
        region1_mask: [n_atoms] mask for first region
        region2_mask: [n_atoms] mask for second region
        region3_mask: [n_atoms] mask for third region
        region4_mask: [n_atoms] mask for fourth region
        max_hops: maximum bond distance for gradient propagation
        decay: decay factor per hop (0.5 = half strength per bond)

    Returns:
        dihedral: [multiplicity] dihedral angle in radians (-π to π)
        gradient: [multiplicity, n_atoms, 3] dDihedral/dr (rotational + propagated)
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Move masks to device
    region1_mask = region1_mask.to(device)
    region2_mask = region2_mask.to(device)
    region3_mask = region3_mask.to(device)
    region4_mask = region4_mask.to(device)

    # Compute COMs
    com1 = compute_region_com(coords, region1_mask)  # [mult, 3]
    com2 = compute_region_com(coords, region2_mask)  # [mult, 3]
    com3 = compute_region_com(coords, region3_mask)  # [mult, 3]
    com4 = compute_region_com(coords, region4_mask)  # [mult, 3]

    # Vectors along the chain
    b1 = com2 - com1  # [mult, 3]
    b2 = com3 - com2  # [mult, 3] - central bond (rotation axis)
    b3 = com4 - com3  # [mult, 3]

    # Normal vectors to planes
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    # Normalize plane normals
    n1_norm = torch.linalg.norm(n1, dim=-1, keepdim=True) + 1e-8
    n2_norm = torch.linalg.norm(n2, dim=-1, keepdim=True) + 1e-8
    n1_unit = n1 / n1_norm
    n2_unit = n2 / n2_norm

    # Central axis (rotation axis) - normalized
    b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-8
    axis = b2 / b2_norm  # [mult, 3]

    # m1 for sign determination (perpendicular to n1 in the plane perpendicular to b2)
    m1 = torch.cross(n1_unit, axis, dim=-1)

    # Dihedral angle using atan2 for proper quadrant
    x = (n1_unit * n2_unit).sum(dim=-1)
    y = (m1 * n2_unit).sum(dim=-1)
    phi = torch.atan2(y, x)  # [mult], range (-π, π)

    # Initialize gradient
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)

    # Rotation axis expanded for broadcasting
    axis_expanded = axis.unsqueeze(1)  # [mult, 1, 3]

    # Region 1: rotate around axis through COM2
    # tangent = cross(axis, r - COM2) gives CCW rotation when looking along +axis
    # CCW rotation of region1 around b2 increases the dihedral (moves n1 toward n2)
    region1_coords = coords[:, region1_mask, :]  # [mult, n1, 3]
    r1_rel = region1_coords - com2.unsqueeze(1)  # relative to COM2
    tangent1 = torch.cross(axis_expanded.expand_as(r1_rel), r1_rel, dim=-1)
    gradient[:, region1_mask, :] = tangent1  # CCW rotation increases dihedral

    # Region 4: rotate around axis through COM3
    # Opposite rotation direction to region1
    region4_coords = coords[:, region4_mask, :]  # [mult, n4, 3]
    r4_rel = region4_coords - com3.unsqueeze(1)  # relative to COM3
    tangent4 = torch.cross(axis_expanded.expand_as(r4_rel), r4_rel, dim=-1)
    gradient[:, region4_mask, :] = -tangent4  # CW rotation (opposite to region1)

    # Regions 2 and 3: zero gradient (pivot points)
    # Already zero from initialization

    # Propagate gradients through bonds to neighbors
    if max_hops > 0:
        adjacency = build_bond_adjacency(feats, n_atoms, device)
        # Source atoms are all atoms in region1 and region4
        source_atoms = torch.where(region1_mask | region4_mask)[0].tolist()
        gradient = propagate_gradient_to_neighbors(
            gradient, source_atoms, adjacency, max_hops, decay
        )

    # Normalize gradient so max magnitude is 1
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return phi, gradient


def distance_region_cv(
    coords: torch.Tensor,
    feats: dict,
    region1_mask: torch.Tensor,
    region2_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distance between two regions using their centers of mass.

    For single-atom regions, this is equivalent to the distance between atoms.
    For multi-atom regions, this computes the distance between COMs.

    Args:
        coords: [multiplicity, n_atoms, 3] coordinates
        feats: Feature dictionary
        region1_mask: [n_atoms] mask for first region
        region2_mask: [n_atoms] mask for second region

    Returns:
        distance: [multiplicity] distance in Angstroms
        gradient: [multiplicity, n_atoms, 3] dDistance/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device

    # Move masks to device
    region1_mask = region1_mask.to(device)
    region2_mask = region2_mask.to(device)

    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)

        # Compute COMs
        com1 = compute_region_com(coords_grad, region1_mask)  # [mult, 3]
        com2 = compute_region_com(coords_grad, region2_mask)  # [mult, 3]

        # Distance between COMs
        diff = com2 - com1
        distance = torch.linalg.norm(diff, dim=-1)  # [mult]

        # Compute gradient via autograd
        distance_sum = distance.sum()
        distance_sum.backward()
        gradient = coords_grad.grad.clone()

    # Scale gradients by region size to ensure rigid-body-like motion
    # Without this, small regions get strong per-atom gradients while
    # large regions get weak gradients, causing boundary breaks
    n1 = region1_mask.float().sum()
    n2 = region2_mask.float().sum()

    # Multiply each atom's gradient by the number of atoms in its region
    gradient[:, region1_mask, :] = gradient[:, region1_mask, :] * n1.view(1, 1, 1)
    gradient[:, region2_mask, :] = gradient[:, region2_mask, :] * n2.view(1, 1, 1)

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return distance, gradient


def min_distance_cv(
    coords: torch.Tensor,
    feats: dict,
    region1_mask: torch.Tensor,
    region2_mask: torch.Tensor,
    softmin_beta: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minimum distance between two groups of atoms.

    Uses a differentiable soft-minimum approximation:
        min_dist ≈ -1/β * log(Σ exp(-β * d_ij))

    This is useful for:
    - Ensuring two domains don't clash (maximize min_distance)
    - Measuring closest approach between regions
    - Contact formation (steer min_distance to target)

    Args:
        coords: [multiplicity, n_atoms, 3] coordinates
        feats: Feature dictionary
        region1_mask: [n_atoms] mask for first group
        region2_mask: [n_atoms] mask for second group
        softmin_beta: Temperature for soft-min (higher = sharper, default=10.0)
                      β=10 gives good gradient smoothness while being close to true min

    Returns:
        min_distance: [multiplicity] minimum distance in Angstroms
        gradient: [multiplicity, n_atoms, 3] dMinDist/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device

    # Move masks to device and ensure they're not inference tensors
    region1_mask = region1_mask.detach().clone().to(device)
    region2_mask = region2_mask.detach().clone().to(device)

    # Get indices of atoms in each group (outside inference mode)
    with torch.inference_mode(False):
        idx1 = torch.where(region1_mask)[0]  # [n1]
        idx2 = torch.where(region2_mask)[0]  # [n2]

    if len(idx1) == 0 or len(idx2) == 0:
        # Empty groups - return zeros
        return torch.zeros(multiplicity, device=device), torch.zeros_like(coords)

    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)

        # Extract coordinates for each group
        # coords1: [mult, n1, 3], coords2: [mult, n2, 3]
        coords1 = coords_grad[:, idx1, :]
        coords2 = coords_grad[:, idx2, :]

        # Compute pairwise distances: [mult, n1, n2]
        # diff[m, i, j] = coords1[m, i] - coords2[m, j]
        diff = coords1.unsqueeze(2) - coords2.unsqueeze(1)  # [mult, n1, n2, 3]
        pairwise_dist = torch.linalg.norm(diff, dim=-1)  # [mult, n1, n2]

        # Soft-minimum using log-sum-exp trick for numerical stability
        # softmin = -1/β * log(Σ exp(-β * d))
        neg_beta_dist = -softmin_beta * pairwise_dist  # [mult, n1, n2]
        # Flatten the n1*n2 dimension for logsumexp
        neg_beta_dist_flat = neg_beta_dist.view(multiplicity, -1)  # [mult, n1*n2]
        logsumexp = torch.logsumexp(neg_beta_dist_flat, dim=-1)  # [mult]
        min_dist = -logsumexp / softmin_beta  # [mult]

        # Compute gradient via autograd
        min_dist_sum = min_dist.sum()
        min_dist_sum.backward()
        gradient = coords_grad.grad.clone()

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return min_dist, gradient


# =============================================================================
# Shape-based CVs
# =============================================================================

def radius_of_gyration_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute radius of gyration as a collective variable.

    Rg = sqrt(mean(||r_i - r_com||^2))

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3] in Angstroms
        feats: Feature dictionary (for atom masks)
        atom_mask: Optional mask for which atoms to include [N_atoms]

    Returns:
        rg_values: [multiplicity] Rg values in Angstroms
        rg_gradient: [multiplicity, N_atoms, 3] dRg/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom mask from feats if not provided
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
    atom_mask = atom_mask.to(device)

    # Ensure atom_mask is 1D of shape [n_atoms]
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]
    if atom_mask.shape[0] != n_atoms:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)

    # Use torch.inference_mode(False) to enable gradient computation
    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)
        atom_mask_float = atom_mask.detach().clone().float()
        n_valid = atom_mask_float.sum() + 1e-8

        # Compute center of mass for each sample
        masked_coords = coords_grad * atom_mask_float.view(1, -1, 1)
        com = masked_coords.sum(dim=1, keepdim=True) / n_valid

        # Compute distances from COM
        centered = (coords_grad - com) * atom_mask_float.view(1, -1, 1)

        # Rg^2 = mean(||r_i - com||^2)
        dist_sq = (centered ** 2).sum(dim=-1)
        rg_sq = dist_sq.sum(dim=1) / n_valid
        rg = torch.sqrt(rg_sq + 1e-8)

        # Compute gradient using autograd
        rg_sum = rg.sum()
        try:
            gradient = torch.autograd.grad(rg_sum, coords_grad, create_graph=False)[0]
        except RuntimeError:
            gradient = torch.zeros_like(coords)

        gradient = gradient * atom_mask_float.view(1, -1, 1)

        # Normalize gradient for effective steering
        grad_norms = gradient.norm(dim=-1, keepdim=True)
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            gradient = gradient / max_norm

        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            gradient = torch.zeros_like(gradient)
        if torch.isnan(rg).any() or torch.isinf(rg).any():
            rg = torch.zeros_like(rg)

    return rg.detach(), gradient.detach()


def distance_cv(
    coords: torch.Tensor,
    feats: dict,
    atom1_idx: int = 0,
    atom2_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distance between two atoms as a CV.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (unused)
        atom1_idx: Index of first atom (default: 0, first atom)
        atom2_idx: Index of second atom (default: -1, last atom)

    Returns:
        distance: [multiplicity] distances in Angstroms
        gradient: [multiplicity, N_atoms, 3] dD/dr
    """
    if atom2_idx < 0:
        atom2_idx = coords.shape[1] + atom2_idx
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom positions
    r1 = coords[:, atom1_idx, :]  # [mult, 3]
    r2 = coords[:, atom2_idx, :]  # [mult, 3]

    # Distance
    delta = r2 - r1  # [mult, 3]
    dist = torch.linalg.norm(delta, dim=-1)  # [mult]

    # Gradient
    # dD/dr1 = -(r2 - r1) / D = -delta / D
    # dD/dr2 = (r2 - r1) / D = delta / D
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)

    unit_vec = delta / (dist.unsqueeze(-1) + 1e-8)  # [mult, 3]
    gradient[:, atom1_idx, :] = -unit_vec
    gradient[:, atom2_idx, :] = unit_vec

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return dist, gradient


def _compute_asphericity(coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """Helper to compute asphericity value only."""
    atom_mask_float = atom_mask.float()
    n_valid = atom_mask_float.sum() + 1e-8

    # COM
    masked_coords = coords * atom_mask_float.view(1, -1, 1)
    com = masked_coords.sum(dim=1, keepdim=True) / n_valid

    # Centered coordinates
    centered = (coords - com) * atom_mask_float.view(1, -1, 1)

    # Gyration tensor
    gyration = torch.einsum('mia,mib->mab', centered, centered) / n_valid

    # Eigenvalues (sorted ascending)
    eigenvalues = torch.linalg.eigvalsh(gyration)

    # Asphericity: (λ1 - λ2)² + (λ2 - λ3)² + (λ1 - λ3)²
    l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    return (l1 - l2)**2 + (l2 - l3)**2 + (l1 - l3)**2


def asphericity_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute asphericity as a collective variable.

    Asphericity measures deviation from spherical shape:
    A = (λ1 - λ2)^2 + (λ2 - λ3)^2 + (λ1 - λ3)^2

    where λ1, λ2, λ3 are eigenvalues of the gyration tensor.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom_mask: Optional mask for atoms

    Returns:
        asphericity: [multiplicity] asphericity values
        gradient: [multiplicity, N_atoms, 3] dA/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom mask from feats if not provided
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
        if atom_mask.dim() == 2:
            atom_mask = atom_mask[0]  # [N_atoms]

    atom_mask = atom_mask.to(device)

    # Use torch.inference_mode(False) to override inference mode (stronger than no_grad)
    with torch.inference_mode(False):
        # Clone coords and atom_mask to avoid inference tensor issues
        coords_grad = coords.detach().clone().requires_grad_(True)
        atom_mask_float = atom_mask.detach().clone().float()
        n_valid = atom_mask_float.sum() + 1e-8

        # COM
        masked_coords = coords_grad * atom_mask_float.view(1, -1, 1)
        com = masked_coords.sum(dim=1, keepdim=True) / n_valid

        # Centered coordinates
        centered = (coords_grad - com) * atom_mask_float.view(1, -1, 1)

        # Gyration tensor with regularization for numerical stability
        gyration = torch.einsum('mia,mib->mab', centered, centered) / n_valid

        # Add small regularization to diagonal to prevent ill-conditioned matrix
        # This ensures eigendecomposition doesn't fail on degenerate structures
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # [1, 3, 3]
        gyration = gyration + 1e-6 * eye

        # Eigenvalues (sorted ascending)
        try:
            eigenvalues = torch.linalg.eigvalsh(gyration)
        except torch._C._LinAlgError:
            # Fallback for ill-conditioned matrix: return zero asphericity and gradient
            return torch.zeros(multiplicity, device=device, dtype=dtype), torch.zeros_like(coords)

        # Asphericity: (λ1 - λ2)² + (λ2 - λ3)² + (λ1 - λ3)²
        l1, l2, l3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        asphericity = (l1 - l2)**2 + (l2 - l3)**2 + (l1 - l3)**2

        # Compute gradient using autograd
        # Sum asphericity over samples for a scalar loss to backprop
        asphericity_sum = asphericity.sum()
        try:
            gradient = torch.autograd.grad(asphericity_sum, coords_grad, create_graph=False)[0]
        except RuntimeError:
            # Fallback if gradient computation fails
            gradient = torch.zeros_like(coords)

        # Zero out gradient for masked atoms
        gradient = gradient * atom_mask_float.view(1, -1, 1)

        # Normalize gradient to have unit max norm per sample
        # This prevents gradient explosion and makes strength parameter meaningful
        grad_norms = gradient.norm(dim=-1, keepdim=True)  # [mult, n_atoms, 1]
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            gradient = gradient / max_norm  # Now max gradient magnitude is 1.0

        # Check for NaN/Inf and replace with zeros (robustness)
        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            gradient = torch.zeros_like(gradient)
        if torch.isnan(asphericity).any() or torch.isinf(asphericity).any():
            asphericity = torch.zeros_like(asphericity)

    return asphericity.detach(), gradient.detach()





def rmsd_cv(
    coords: torch.Tensor,
    feats: dict,
    reference_coords: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RMSD to a reference structure as a CV.

    Uses Kabsch algorithm for optimal alignment.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        reference_coords: Reference coordinates [N_atoms, 3]
        atom_mask: Optional mask for which atoms to include [N_atoms]

    Returns:
        rmsd: [multiplicity] RMSD values in Angstroms
        gradient: [multiplicity, N_atoms, 3] dRMSD/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Ensure reference is on same device
    reference_coords = reference_coords.to(device=device, dtype=dtype)
    n_ref_atoms = reference_coords.shape[0]

    # Handle size mismatch: pad reference or truncate to match
    if n_ref_atoms < n_atoms:
        # Pad reference with zeros
        padding = torch.zeros(n_atoms - n_ref_atoms, 3, device=device, dtype=dtype)
        reference_coords = torch.cat([reference_coords, padding], dim=0)
    elif n_ref_atoms > n_atoms:
        # Truncate reference (shouldn't normally happen)
        reference_coords = reference_coords[:n_atoms]

    # Get atom mask
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
    atom_mask = atom_mask.to(device)
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]
    if atom_mask.shape[0] != n_atoms:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)

    # Also mask out padded reference atoms (if we padded)
    if n_ref_atoms < n_atoms:
        ref_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        ref_mask[:n_ref_atoms] = True
        atom_mask = atom_mask.bool() & ref_mask

    n_valid = atom_mask.sum().float()

    # Mask coordinates
    mask_expand = atom_mask.unsqueeze(0).unsqueeze(-1)  # [1, N_atoms, 1]

    # Center both structures
    coords_masked = coords * mask_expand
    ref_masked = reference_coords.unsqueeze(0) * mask_expand

    coords_center = coords_masked.sum(dim=1, keepdim=True) / n_valid
    ref_center = ref_masked.sum(dim=1, keepdim=True) / n_valid

    coords_centered = (coords - coords_center) * mask_expand
    ref_centered = (reference_coords.unsqueeze(0) - ref_center) * mask_expand

    # Kabsch algorithm for optimal rotation
    # We want to align coords to ref, so find R such that coords @ R ≈ ref
    # H = coords^T @ ref (P^T @ Q where P=coords, Q=ref)
    H = torch.einsum('mni,mnj->mij', coords_centered, ref_centered)  # [mult, 3, 3]

    # SVD: H = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(H)

    # Optimal rotation: R = U @ V^T
    R = U @ Vt  # [mult, 3, 3]

    # Handle reflection case (det(R) = -1)
    d = torch.det(R)
    # For samples where det < 0, flip sign of last column of U
    mask = (d < 0).unsqueeze(-1).unsqueeze(-1)
    U_corrected = U.clone()
    U_corrected[:, :, -1] = torch.where(mask.squeeze(-1), -U[:, :, -1], U[:, :, -1])
    R = U_corrected @ Vt

    # Rotate coordinates to align with reference: coords_aligned = coords @ R
    coords_rotated = torch.einsum('mni,mij->mnj', coords_centered, R)  # [mult, N_atoms, 3]

    # Compute RMSD
    diff = (coords_rotated - ref_centered) * mask_expand
    rmsd_sq = (diff ** 2).sum(dim=(-2, -1)) / n_valid  # [mult]
    rmsd = torch.sqrt(rmsd_sq + 1e-8)

    # Gradient: d(RMSD)/d(coords) = d(RMSD)/d(coords_rotated) @ R^T
    # Since coords_rotated = coords @ R, we have d(coords_rotated)/d(coords) = R
    # This is an approximation ignoring the rotation gradient
    grad_rotated = diff / (rmsd.unsqueeze(-1).unsqueeze(-1) * n_valid + 1e-8)

    # Transform gradient back to original frame: gradient = grad_rotated @ R^T
    gradient = torch.einsum('mni,mji->mnj', grad_rotated, R)  # [mult, N_atoms, 3]
    gradient = gradient * mask_expand

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return rmsd, gradient


def _kabsch_rotation(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Compute optimal rotation matrix to align P onto Q using Kabsch algorithm.

    Args:
        P: Source coordinates [N_atoms, 3] (will be rotated)
        Q: Target coordinates [N_atoms, 3] (reference)

    Returns:
        R: Rotation matrix [3, 3] such that P @ R aligns with Q
    """
    # Covariance matrix H = P^T @ Q
    H = P.T @ Q  # [3, 3]

    # SVD: H = U @ S @ V^T
    U, S, Vh = torch.linalg.svd(H)

    # Optimal rotation: R = U @ V^T
    R = U @ Vh

    # Check for reflection (det(R) = -1)
    if torch.det(R) < 0:
        # Flip sign of last column of U
        U_fixed = U.clone()
        U_fixed[:, -1] = -U_fixed[:, -1]
        R = U_fixed @ Vh

    return R


def _kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Kabsch-aligned RMSD between two structures.

    Args:
        P: First structure [N_atoms, 3]
        Q: Second structure [N_atoms, 3]
        mask: Atom mask [N_atoms]

    Returns:
        rmsd: Scalar RMSD value
        P_aligned: P after optimal alignment [N_atoms, 3]
        Q_centered: Q after centering [N_atoms, 3]
        R: Rotation matrix [3, 3] used to align P to Q
    """
    n_valid = mask.sum().float()
    mask_3d = mask.unsqueeze(-1).float()  # [N_atoms, 1]

    # Center both structures
    P_masked = P * mask_3d
    Q_masked = Q * mask_3d
    P_com = P_masked.sum(dim=0, keepdim=True) / n_valid
    Q_com = Q_masked.sum(dim=0, keepdim=True) / n_valid
    P_centered = (P - P_com) * mask_3d
    Q_centered = (Q - Q_com) * mask_3d

    # Get optimal rotation: find R such that P @ R aligns with Q
    R = _kabsch_rotation(P_centered, Q_centered)

    # Align P to Q
    P_aligned = P_centered @ R

    # Compute RMSD
    diff = P_aligned - Q_centered
    rmsd = torch.sqrt((diff ** 2).sum() / n_valid + 1e-8)

    return rmsd, P_aligned, Q_centered, R


def pair_rmsd_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise RMSD between all samples in the batch using Kabsch alignment.

    This CV computes the mean pairwise RMSD between all pairs of samples,
    using proper Kabsch (SVD) alignment to find optimal rotation. This ensures
    RMSD measures conformational differences, not orientation or size differences.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom_mask: Optional mask for which atoms to include [N_atoms]

    Returns:
        mean_pair_rmsd: [multiplicity] mean pairwise RMSD for each sample
        gradient: [multiplicity, N_atoms, 3] d(mean_pair_rmsd)/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom mask
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))

    # Ensure atom_mask is 1D [n_atoms] regardless of input shape
    atom_mask = atom_mask.to(device)
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]  # Take first element until 1D

    mask_expand = atom_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, N_atoms, 1]
    n_valid = atom_mask.sum().float()

    if multiplicity < 2:
        # With single sample, pairwise RMSD is undefined - return zeros
        return (
            torch.zeros(multiplicity, device=device, dtype=dtype),
            torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)
        )

    # Compute pairwise Kabsch-aligned RMSD between all pairs of samples
    pair_rmsds = torch.zeros(multiplicity, multiplicity, device=device, dtype=dtype)
    # Store aligned differences and rotation matrices for gradient computation
    aligned_diffs = {}  # (i, j) -> (P_aligned - Q_centered)
    rotations = {}  # (i, j) -> R matrix used to align i to j

    for i in range(multiplicity):
        for j in range(i + 1, multiplicity):
            # Compute Kabsch-aligned RMSD
            rmsd_ij, P_aligned, Q_centered, R = _kabsch_rmsd(
                coords[i], coords[j], atom_mask
            )
            pair_rmsds[i, j] = rmsd_ij
            pair_rmsds[j, i] = rmsd_ij

            # Store aligned difference and rotation matrix
            aligned_diffs[(i, j)] = P_aligned - Q_centered  # [N_atoms, 3]
            rotations[(i, j)] = R  # [3, 3]

    # Mean pairwise RMSD for each sample (excluding self)
    mean_pair_rmsd = pair_rmsds.sum(dim=1) / (multiplicity - 1)  # [mult]

    # Gradient computation with Kabsch alignment
    # The gradient in the aligned frame is: diff / (rmsd * N)
    # Must transform back to original frame using R^T
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)

    for i in range(multiplicity):
        grad_i = torch.zeros(n_atoms, 3, device=device, dtype=dtype)
        for j in range(multiplicity):
            if i != j:
                rmsd_ij = pair_rmsds[i, j]
                if rmsd_ij > 1e-8:
                    if (i, j) in aligned_diffs:
                        # i was aligned to j: diff is in j's frame
                        # grad_aligned = diff / (rmsd * N)
                        # grad_original = grad_aligned @ R^T (transform back to i's frame)
                        diff = aligned_diffs[(i, j)]
                        R = rotations[(i, j)]
                        grad_aligned = diff / (rmsd_ij * n_valid)
                        grad_i += grad_aligned @ R.T
                    else:
                        # j was aligned to i: diff = P_j_aligned - Q_i_centered
                        # where P_j_aligned = P_j_centered @ R
                        # For d(rmsd)/dr_i, gradient is -(diff) / (rmsd * N)
                        # No rotation needed - i was the reference frame (Q), not rotated
                        diff = aligned_diffs[(j, i)]
                        grad_aligned = -diff / (rmsd_ij * n_valid)
                        grad_i += grad_aligned

        gradient[i] = grad_i / (multiplicity - 1)

    gradient = gradient * mask_expand

    # Normalize gradient to have unit max norm per sample
    # This prevents the gradient from overwhelming the diffusion process
    grad_norms = gradient.norm(dim=-1, keepdim=True)  # [mult, n_atoms, 1]
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm  # Now max gradient magnitude is 1.0

    return mean_pair_rmsd, gradient


def _kabsch_rmsd_norm_rg(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Kabsch-aligned RMSD between two structures after normalizing to unit Rg.

    This makes the RMSD size-invariant - it measures shape differences only.

    Args:
        P: First structure [N_atoms, 3]
        Q: Second structure [N_atoms, 3]
        mask: Atom mask [N_atoms]

    Returns:
        rmsd: Scalar RMSD value (size-invariant)
        P_aligned: P after normalization and alignment [N_atoms, 3]
        Q_normalized: Q after normalization [N_atoms, 3]
        R: Rotation matrix [3, 3] used to align P to Q
    """
    n_valid = mask.sum().float()
    mask_3d = mask.unsqueeze(-1).float()  # [N_atoms, 1]

    # Center both structures
    P_masked = P * mask_3d
    Q_masked = Q * mask_3d
    P_com = P_masked.sum(dim=0, keepdim=True) / n_valid
    Q_com = Q_masked.sum(dim=0, keepdim=True) / n_valid
    P_centered = (P - P_com) * mask_3d
    Q_centered = (Q - Q_com) * mask_3d

    # Normalize to unit Rg (size-invariant)
    P_rg = torch.sqrt((P_centered ** 2).sum() / n_valid + 1e-8)
    Q_rg = torch.sqrt((Q_centered ** 2).sum() / n_valid + 1e-8)
    P_normalized = P_centered / P_rg
    Q_normalized = Q_centered / Q_rg

    # Get optimal rotation: find R such that P_normalized @ R aligns with Q_normalized
    R = _kabsch_rotation(P_normalized, Q_normalized)

    # Align P to Q
    P_aligned = P_normalized @ R

    # Compute RMSD
    diff = P_aligned - Q_normalized
    rmsd = torch.sqrt((diff ** 2).sum() / n_valid + 1e-8)

    return rmsd, P_aligned, Q_normalized, R


def pair_rmsd_norm_rg_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise RMSD between samples after normalizing each to unit Rg.

    This CV is size-invariant - it measures conformational/shape differences only,
    not size differences. Useful for diversity maximization without encouraging
    structure expansion.

    Uses Kabsch alignment for rotation invariance.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom_mask: Optional mask for which atoms to include [N_atoms]

    Returns:
        mean_pair_rmsd: [multiplicity] mean pairwise RMSD for each sample
        gradient: [multiplicity, N_atoms, 3] d(mean_pair_rmsd)/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom mask
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))

    # Ensure atom_mask is 1D [n_atoms] regardless of input shape
    atom_mask = atom_mask.to(device)
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]

    mask_expand = atom_mask.unsqueeze(0).unsqueeze(-1).float()  # [1, N_atoms, 1]
    n_valid = atom_mask.sum().float()

    if multiplicity < 2:
        return (
            torch.zeros(multiplicity, device=device, dtype=dtype),
            torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)
        )

    # Compute pairwise Kabsch-aligned RMSD (with Rg normalization)
    pair_rmsds = torch.zeros(multiplicity, multiplicity, device=device, dtype=dtype)
    aligned_diffs = {}
    rotations = {}

    for i in range(multiplicity):
        for j in range(i + 1, multiplicity):
            rmsd_ij, P_aligned, Q_normalized, R = _kabsch_rmsd_norm_rg(
                coords[i], coords[j], atom_mask
            )
            pair_rmsds[i, j] = rmsd_ij
            pair_rmsds[j, i] = rmsd_ij
            aligned_diffs[(i, j)] = P_aligned - Q_normalized
            rotations[(i, j)] = R

    # Mean pairwise RMSD for each sample
    mean_pair_rmsd = pair_rmsds.sum(dim=1) / (multiplicity - 1)

    # Gradient computation with proper frame transformation
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)

    for i in range(multiplicity):
        grad_i = torch.zeros(n_atoms, 3, device=device, dtype=dtype)
        for j in range(multiplicity):
            if i != j:
                rmsd_ij = pair_rmsds[i, j]
                if rmsd_ij > 1e-8:
                    if (i, j) in aligned_diffs:
                        # i was aligned to j: transform gradient back to i's frame
                        diff = aligned_diffs[(i, j)]
                        R = rotations[(i, j)]
                        grad_aligned = diff / (rmsd_ij * n_valid)
                        grad_i += grad_aligned @ R.T
                    else:
                        # j was aligned to i: gradient is already in i's frame
                        diff = aligned_diffs[(j, i)]
                        grad_i += -diff / (rmsd_ij * n_valid)
        gradient[i] = grad_i / (multiplicity - 1)

    gradient = gradient * mask_expand

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return mean_pair_rmsd, gradient


def rmsf_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Root Mean Square Fluctuation (RMSF) as a collective variable.

    RMSF measures per-atom fluctuation around the ensemble mean position.
    Returns the mean RMSF across selected atoms.

    RMSF_i = sqrt(mean_over_samples(||r_i - <r_i>||²))
    CV = mean_over_atoms(RMSF_i)

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (for atom masks)
        atom_mask: Optional mask for which atoms to include [N_atoms]

    Returns:
        mean_rmsf: [multiplicity] mean RMSF value (same for all samples)
        gradient: [multiplicity, N_atoms, 3] d(mean_rmsf)/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    if multiplicity < 2:
        # Need at least 2 samples to compute fluctuations
        return (
            torch.zeros(multiplicity, device=device, dtype=dtype),
            torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype),
        )

    # Get atom mask
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
    atom_mask = atom_mask.to(device)
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]
    if atom_mask.shape[0] != n_atoms:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)

    atom_mask = atom_mask.to(dtype=dtype)
    n_selected = atom_mask.sum().item()

    if n_selected < 1:
        return (
            torch.zeros(multiplicity, device=device, dtype=dtype),
            torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype),
        )

    # Compute mean position across samples for each atom
    # [N_atoms, 3]
    mean_coords = coords.mean(dim=0)

    # Compute deviation from mean for each sample
    # [multiplicity, N_atoms, 3]
    deviations = coords - mean_coords.unsqueeze(0)

    # Apply atom mask
    deviations_masked = deviations * atom_mask.view(1, -1, 1)

    # Compute squared deviations per atom: ||r_i - <r_i>||²
    # [multiplicity, N_atoms]
    sq_deviations = (deviations_masked ** 2).sum(dim=-1)

    # Mean squared deviation across samples for each atom
    # [N_atoms]
    mean_sq_dev = sq_deviations.mean(dim=0)

    # RMSF per atom: sqrt(mean squared deviation)
    # [N_atoms]
    rmsf_per_atom = torch.sqrt(mean_sq_dev + 1e-8)

    # Apply mask and compute mean RMSF across selected atoms
    rmsf_masked = rmsf_per_atom * atom_mask
    mean_rmsf = rmsf_masked.sum() / n_selected

    # Return same value for all samples (RMSF is an ensemble property)
    mean_rmsf_per_sample = mean_rmsf.expand(multiplicity)

    # Gradient computation
    # d(mean_rmsf)/dr_i,s = d/dr_i,s [ mean_atoms( sqrt(mean_samples(||r - <r>||²)) ) ]
    #
    # For sample s and atom i:
    # d(RMSF_i)/dr_i,s = (r_i,s - <r_i>) / (RMSF_i * N_samples)
    #
    # d(mean_rmsf)/dr_i,s = (1/N_atoms) * (r_i,s - <r_i>) / (RMSF_i * N_samples)
    #
    # This gradient points away from the mean - following it increases RMSF

    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)

    for s in range(multiplicity):
        # Gradient for sample s
        # [N_atoms, 3]
        dev_s = deviations[s]  # r_i,s - <r_i>

        # Avoid division by zero for atoms with zero RMSF
        rmsf_safe = rmsf_per_atom.clamp(min=1e-8)

        # Gradient: (r - <r>) / (RMSF * N_samples * N_atoms)
        grad_s = dev_s / (rmsf_safe.view(-1, 1) * multiplicity * n_selected)

        # Apply atom mask
        grad_s = grad_s * atom_mask.view(-1, 1)

        gradient[s] = grad_s

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return mean_rmsf_per_sample, gradient


def pair_rmsd_grouped_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
    align_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise RMSD with separate alignment and RMSD groups.

    Aligns structures using one group of atoms (align_mask), then computes
    RMSD on a different group (atom_mask). Useful for measuring ligand pose
    diversity after aligning on protein atoms.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (unused, for API consistency)
        atom_mask: Mask for atoms to compute RMSD on [N_atoms] (e.g., ligand)
        align_mask: Mask for atoms to use for Kabsch alignment [N_atoms] (e.g., protein)

    Returns:
        mean_pair_rmsd: [multiplicity] mean pairwise RMSD for each sample
        gradient: [multiplicity, N_atoms, 3] d(mean_pair_rmsd)/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Default masks: all atoms
    if atom_mask is None:
        atom_mask = torch.ones(n_atoms, device=device, dtype=dtype)
    if align_mask is None:
        align_mask = torch.ones(n_atoms, device=device, dtype=dtype)

    atom_mask = atom_mask.to(device=device, dtype=dtype)
    align_mask = align_mask.to(device=device, dtype=dtype)

    n_rmsd_atoms = atom_mask.sum().item()
    n_align_atoms = align_mask.sum().item()

    if n_rmsd_atoms < 1:
        return (
            torch.zeros(multiplicity, device=device, dtype=dtype),
            torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype),
        )

    if n_align_atoms < 3:
        # Not enough atoms for alignment, fall back to regular pair_rmsd on atom_mask
        return pair_rmsd_cv(coords, feats, atom_mask)

    # Compute pairwise RMSD: align using align_mask, compute RMSD using atom_mask
    pair_rmsds = torch.zeros(multiplicity, multiplicity, device=device, dtype=dtype)
    aligned_diffs = {}  # (i, j) -> aligned difference for atom_mask atoms
    rotations = {}  # (i, j) -> R matrix

    for i in range(multiplicity):
        for j in range(i + 1, multiplicity):
            # Extract alignment atoms
            P_align = coords[i] * align_mask.view(-1, 1)  # [N, 3]
            Q_align = coords[j] * align_mask.view(-1, 1)

            # Center alignment atoms
            P_align_mean = (P_align.sum(dim=0) / n_align_atoms)
            Q_align_mean = (Q_align.sum(dim=0) / n_align_atoms)
            P_align_centered = (coords[i] - P_align_mean) * align_mask.view(-1, 1)
            Q_align_centered = (coords[j] - Q_align_mean) * align_mask.view(-1, 1)

            # Kabsch rotation using alignment atoms
            R = _kabsch_rotation(P_align_centered, Q_align_centered)

            # Apply rotation to ALL atoms of sample i (centered on align group)
            P_centered_full = coords[i] - P_align_mean  # Center on align group mean
            P_rotated_full = P_centered_full @ R  # Rotate all atoms

            # Center sample j on its align group mean
            Q_centered_full = coords[j] - Q_align_mean

            # Compute RMSD using only atom_mask atoms
            P_rmsd = P_rotated_full * atom_mask.view(-1, 1)
            Q_rmsd = Q_centered_full * atom_mask.view(-1, 1)
            diff = P_rmsd - Q_rmsd

            rmsd_ij = torch.sqrt((diff ** 2).sum() / n_rmsd_atoms + 1e-8)
            pair_rmsds[i, j] = rmsd_ij
            pair_rmsds[j, i] = rmsd_ij

            # Store for gradient computation
            aligned_diffs[(i, j)] = diff  # Already masked
            rotations[(i, j)] = R

    # Mean pairwise RMSD for each sample
    mean_pair_rmsd = pair_rmsds.sum(dim=1) / (multiplicity - 1)

    # Gradient computation
    # For atom_mask atoms only: d(RMSD)/dr = diff / (RMSD * N)
    # Need to transform back to original frame using R^T
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)

    for i in range(multiplicity):
        grad_i = torch.zeros(n_atoms, 3, device=device, dtype=dtype)
        for j in range(multiplicity):
            if i != j:
                rmsd_ij = pair_rmsds[i, j]
                if rmsd_ij > 1e-8:
                    if (i, j) in aligned_diffs:
                        # i was aligned to j
                        diff = aligned_diffs[(i, j)]
                        R = rotations[(i, j)]
                        # Gradient in aligned frame
                        grad_aligned = diff / (rmsd_ij * n_rmsd_atoms)
                        # Transform back to original frame
                        grad_i += grad_aligned @ R.T
                    else:
                        # j was aligned to i: (j, i) in aligned_diffs
                        diff = aligned_diffs[(j, i)]
                        # Gradient for i is negative of aligned diff (already in i's frame approx)
                        grad_i += -diff / (rmsd_ij * n_rmsd_atoms)
        gradient[i] = grad_i / (multiplicity - 1)

    # Apply atom_mask to gradient (only atom_mask atoms should have gradients)
    gradient = gradient * atom_mask.view(1, -1, 1)

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return mean_pair_rmsd, gradient


def native_contacts_cv(
    coords: torch.Tensor,
    feats: dict,
    reference_coords: torch.Tensor,
    contact_cutoff: float = 4.5,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute fraction of native contacts (Q) as a CV.

    Uses a soft switching function for differentiability.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        reference_coords: Reference coordinates [N_atoms, 3]
        contact_cutoff: Distance cutoff for defining contacts (Angstroms)
        atom_mask: Optional mask for which atoms to include

    Returns:
        Q: [multiplicity] fraction of native contacts (0-1)
        gradient: [multiplicity, N_atoms, 3] dQ/dr (normalized)
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    reference_coords = reference_coords.to(device=device, dtype=dtype)
    n_ref_atoms = reference_coords.shape[0]

    # Handle size mismatch: pad reference or truncate to match
    if n_ref_atoms < n_atoms:
        # Pad reference with zeros
        padding = torch.zeros(n_atoms - n_ref_atoms, 3, device=device, dtype=dtype)
        reference_coords = torch.cat([reference_coords, padding], dim=0)
    elif n_ref_atoms > n_atoms:
        # Truncate reference
        reference_coords = reference_coords[:n_atoms]

    # Get atom mask
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
    atom_mask = atom_mask.to(device)
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]
    if atom_mask.shape[0] != n_atoms:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
    atom_mask = atom_mask.bool()

    # Also mask out padded reference atoms (if we padded)
    if n_ref_atoms < n_atoms:
        ref_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        ref_mask[:n_ref_atoms] = True
        atom_mask = atom_mask & ref_mask

    # Compute pairwise distances in reference (no grad needed)
    ref_diff = reference_coords.unsqueeze(0) - reference_coords.unsqueeze(1)  # [N, N, 3]
    ref_dist = torch.linalg.norm(ref_diff, dim=-1)  # [N, N]

    # Use autograd for gradient computation
    with torch.inference_mode(False):
        # Identify native contacts (excluding self and nearby residues)
        # Assume residues are sequential in atom indices for simplicity
        seq_sep = torch.abs(torch.arange(n_atoms, device=device).unsqueeze(0) -
                            torch.arange(n_atoms, device=device).unsqueeze(1))

        # Clone ref_dist to avoid inference tensor issues
        ref_dist_grad = ref_dist.detach().clone()
        native_contact_mask = (ref_dist_grad < contact_cutoff) & (seq_sep > 3)  # [N, N]
        atom_mask_grad = atom_mask.detach().clone()
        native_contact_mask = native_contact_mask & atom_mask_grad.unsqueeze(0) & atom_mask_grad.unsqueeze(1)

        n_native = native_contact_mask.sum().float()
        if n_native < 1:
            # No native contacts found - return outside context manager
            return torch.zeros(multiplicity, device=device), torch.zeros(multiplicity, n_atoms, 3, device=device)

        coords_grad = coords.detach().clone().requires_grad_(True)

        # Compute current pairwise distances
        diff = coords_grad.unsqueeze(2) - coords_grad.unsqueeze(1)  # [mult, N, N, 3]
        dist = torch.linalg.norm(diff, dim=-1)  # [mult, N, N]

        # Soft switching function: s(r) = 1 / (1 + exp(beta * (r - r0)))
        # where r0 = contact_cutoff, beta controls steepness
        beta = 2.0  # Steepness parameter
        r0 = contact_cutoff * 1.5  # Slightly larger than cutoff for smooth transition

        switching = torch.sigmoid(-beta * (dist - r0))  # [mult, N, N]

        # Apply native contact mask
        switching_masked = switching * native_contact_mask.unsqueeze(0).float()

        # Q = sum of switching function / n_native
        Q = switching_masked.sum(dim=(-2, -1)) / n_native  # [mult]

        # Compute gradient via autograd
        Q_sum = Q.sum()
        gradient = torch.autograd.grad(Q_sum, coords_grad)[0]

    # Normalize gradient (CRITICAL for effective steering)
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return Q.detach(), gradient.detach()


def coordination_cv(
    coords: torch.Tensor,
    feats: dict,
    contact_cutoff: float = 6.0,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute coordination number (total contacts) as a CV.

    Uses a soft switching function for differentiability.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        contact_cutoff: Distance cutoff for contacts (Angstroms)
        atom_mask: Optional mask for which atoms to include

    Returns:
        coord_num: [multiplicity] coordination number
        gradient: [multiplicity, N_atoms, 3] dN/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom mask
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
    atom_mask = atom_mask.to(device)

    # Ensure atom_mask is 1D of shape [n_atoms]
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]
    if atom_mask.shape[0] != n_atoms:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
    atom_mask = atom_mask.bool()

    # Exclude nearby residues (sequence separation > 3)
    seq_sep = torch.abs(torch.arange(n_atoms, device=device).unsqueeze(0) -
                        torch.arange(n_atoms, device=device).unsqueeze(1))
    pair_mask = (seq_sep > 3) & atom_mask.unsqueeze(0) & atom_mask.unsqueeze(1)

    # Compute pairwise distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [mult, N, N, 3]
    dist = torch.linalg.norm(diff, dim=-1)  # [mult, N, N]

    # Soft switching function
    beta = 2.0
    r0 = contact_cutoff

    switching = torch.sigmoid(-beta * (dist - r0))  # [mult, N, N]
    switching_masked = switching * pair_mask.unsqueeze(0)

    # Coordination number = sum of switching function (divided by 2 to avoid double counting)
    coord_num = switching_masked.sum(dim=(-2, -1)) / 2  # [mult]

    # Gradient
    ds_dr = -beta * switching * (1 - switching)
    dist_safe = dist + 1e-8
    grad_dist = diff / dist_safe.unsqueeze(-1)

    ds_dr_masked = ds_dr * pair_mask.unsqueeze(0)

    grad_i = (ds_dr_masked.unsqueeze(-1) * grad_dist).sum(dim=2)
    grad_j = (ds_dr_masked.unsqueeze(-1) * (-grad_dist)).sum(dim=1)

    gradient = (grad_i + grad_j) / 2  # [mult, N, 3]

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return coord_num, gradient


def inter_chain_cv(
    coords: torch.Tensor,
    feats: dict,
    chain1_mask: torch.Tensor,
    chain2_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distance between centers of mass of two chains.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        chain1_mask: Boolean mask for chain 1 atoms [N_atoms]
        chain2_mask: Boolean mask for chain 2 atoms [N_atoms]

    Returns:
        distance: [multiplicity] COM distance in Angstroms
        gradient: [multiplicity, N_atoms, 3] dD/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Use torch.inference_mode(False) to enable gradient computation
    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)

        # Clone masks to avoid inference tensor issues
        chain1_mask_f = chain1_mask.detach().clone().to(device).float()
        chain2_mask_f = chain2_mask.detach().clone().to(device).float()

        n1 = chain1_mask_f.sum() + 1e-8
        n2 = chain2_mask_f.sum() + 1e-8

        # Compute COMs
        com1 = (coords_grad * chain1_mask_f.view(1, -1, 1)).sum(dim=1) / n1
        com2 = (coords_grad * chain2_mask_f.view(1, -1, 1)).sum(dim=1) / n2

        # Distance between COMs
        delta = com2 - com1
        dist = torch.linalg.norm(delta, dim=-1)

        # Compute gradient using autograd
        dist_sum = dist.sum()
        try:
            gradient = torch.autograd.grad(dist_sum, coords_grad, create_graph=False)[0]
        except RuntimeError:
            gradient = torch.zeros_like(coords)

        # Normalize gradient for effective steering
        grad_norms = gradient.norm(dim=-1, keepdim=True)
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            gradient = gradient / max_norm

        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            gradient = torch.zeros_like(gradient)
        if torch.isnan(dist).any() or torch.isinf(dist).any():
            dist = torch.zeros_like(dist)

    return dist.detach(), gradient.detach()


def max_diameter_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute maximum pairwise distance as a CV.

    Uses autograd for gradient computation.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom_mask: Optional mask for atoms

    Returns:
        max_dist: [multiplicity] maximum distance in Angstroms
        gradient: [multiplicity, N_atoms, 3] d(max_dist)/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom mask
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
    atom_mask = atom_mask.to(device)
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]
    if atom_mask.shape[0] != n_atoms:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)

    # Use torch.inference_mode(False) to enable gradient computation
    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)
        atom_mask_f = atom_mask.detach().clone().float()

        # Compute pairwise distances
        diff = coords_grad.unsqueeze(2) - coords_grad.unsqueeze(1)  # [mult, N, N, 3]
        dist = torch.linalg.norm(diff, dim=-1)  # [mult, N, N]

        # Mask out invalid pairs and self-distances
        pair_mask = atom_mask_f.unsqueeze(0) * atom_mask_f.unsqueeze(1)
        eye_mask = 1.0 - torch.eye(n_atoms, device=device, dtype=dtype)
        pair_mask = pair_mask * eye_mask

        # Set invalid distances to -inf for softmax
        dist_masked = dist * pair_mask - 1e10 * (1 - pair_mask)

        # Soft maximum using log-sum-exp trick with higher beta for sharper max
        beta = 10.0  # Higher temperature for sharper approximation
        dist_flat = dist_masked.view(multiplicity, -1)

        # Stable log-sum-exp
        max_val = dist_flat.max(dim=-1, keepdim=True)[0]
        softmax_dist = max_val.squeeze(-1) + torch.log(
            torch.exp(beta * (dist_flat - max_val)).sum(dim=-1) + 1e-8
        ) / beta

        # Compute gradient using autograd
        softmax_sum = softmax_dist.sum()
        try:
            gradient = torch.autograd.grad(softmax_sum, coords_grad, create_graph=False)[0]
        except RuntimeError:
            gradient = torch.zeros_like(coords)

        gradient = gradient * atom_mask_f.view(1, -1, 1)

        # Normalize gradient for effective steering
        grad_norms = gradient.norm(dim=-1, keepdim=True)
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            gradient = gradient / max_norm

        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            gradient = torch.zeros_like(gradient)
        if torch.isnan(softmax_dist).any() or torch.isinf(softmax_dist).any():
            softmax_dist = torch.zeros_like(softmax_dist)

    return softmax_dist.detach(), gradient.detach()


# =============================================================================
# Angle and Dihedral CVs
# =============================================================================

def angle_cv(
    coords: torch.Tensor,
    feats: dict,
    atom1_idx: int,
    atom2_idx: int,
    atom3_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute angle between three atoms as a CV.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom1_idx, atom2_idx, atom3_idx: Indices of the three atoms (angle at atom2)

    Returns:
        angle: [multiplicity] angles in radians
        gradient: [multiplicity, N_atoms, 3] dAngle/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    r1 = coords[:, atom1_idx, :]  # [mult, 3]
    r2 = coords[:, atom2_idx, :]  # [mult, 3]
    r3 = coords[:, atom3_idx, :]  # [mult, 3]

    # Vectors from central atom
    v1 = r1 - r2  # [mult, 3]
    v2 = r3 - r2  # [mult, 3]

    # Norms
    n1 = torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8  # [mult, 1]
    n2 = torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8  # [mult, 1]

    # Unit vectors
    u1 = v1 / n1
    u2 = v2 / n2

    # Cosine of angle
    cos_theta = (u1 * u2).sum(dim=-1)  # [mult]
    cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

    # Angle
    theta = torch.acos(cos_theta)  # [mult]

    # Gradient — clamp sin_theta away from zero to prevent explosion
    # near θ≈0 and θ≈π (collinear/anti-collinear atoms)
    sin_theta = torch.clamp(torch.sin(theta).abs(), min=1e-4)

    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)

    # dtheta/dr1 = -1/sin(theta) * (u2 - cos_theta * u1) / n1
    # dtheta/dr3 = -1/sin(theta) * (u1 - cos_theta * u2) / n2
    # dtheta/dr2 = -dtheta/dr1 - dtheta/dr3

    grad_r1 = -(u2 - cos_theta.unsqueeze(-1) * u1) / (sin_theta.unsqueeze(-1) * n1)
    grad_r3 = -(u1 - cos_theta.unsqueeze(-1) * u2) / (sin_theta.unsqueeze(-1) * n2)
    grad_r2 = -grad_r1 - grad_r3

    gradient[:, atom1_idx, :] = grad_r1
    gradient[:, atom2_idx, :] = grad_r2
    gradient[:, atom3_idx, :] = grad_r3

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return theta, gradient


def angle_enhanced_cv(
    coords: torch.Tensor,
    feats: dict,
    atom1_idx: int,
    atom2_idx: int,
    atom3_idx: int,
    max_hops: int = 10,
    decay: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute angle CV with ROTATIONAL gradients propagated to bonded neighbors.

    This addresses the gradient sparsity problem of angle_cv by propagating
    the gradient from the 3 core atoms to their bonded neighbors using
    proper rotational tangent vectors (not just copied gradients).

    Propagation uses barriers to prevent gradient cancellation:
    - atom1 propagates to its side, blocked by atom2 (vertex)
    - atom3 propagates to its side, blocked by atom2 (vertex)
    - atom2 (vertex) does not propagate (it's the pivot point)

    Rotational gradients ensure that propagated atoms move in the correct
    tangential direction for an angle change (like a door hinge).

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (must contain bond connectivity)
        atom1_idx, atom2_idx, atom3_idx: Indices of the three atoms (angle at atom2)
        max_hops: Maximum bond distance for propagation (default 10)
        decay: Decay factor per hop (default 0.5 = 50% per bond)

    Returns:
        angle: [multiplicity] angles in radians
        gradient: [multiplicity, N_atoms, 3] enhanced dAngle/dr with rotational tangents
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get base angle and gradient
    theta, gradient = angle_cv(coords, feats, atom1_idx, atom2_idx, atom3_idx)

    # Build adjacency
    adjacency = build_bond_adjacency(feats, n_atoms, coords.device)

    # Compute rotation axis for rotational gradients
    # Vectors from vertex (atom2) to atom1 and atom3
    r1 = coords[:, atom1_idx, :]  # [mult, 3]
    r2 = coords[:, atom2_idx, :]  # [mult, 3] - vertex (pivot)
    r3 = coords[:, atom3_idx, :]  # [mult, 3]

    v1 = r1 - r2  # vertex -> atom1
    v2 = r3 - r2  # vertex -> atom3

    # Rotation axis: perpendicular to angle plane (v1 × v2)
    rotation_axis = torch.cross(v1, v2, dim=-1)  # [mult, 3]
    axis_norm = rotation_axis.norm(dim=-1, keepdim=True) + 1e-8
    rotation_axis = rotation_axis / axis_norm  # unit vector

    # Propagate from atom1's side with rotational gradients
    # sign=-1.0 because cross(axis, r_rel) gives direction that decreases angle
    # We want gradient that increases angle (standard convention)
    propagate_rotational_gradient_with_barriers(
        coords, gradient, atom1_idx, atom2_idx, rotation_axis, adjacency,
        barrier_atoms={atom2_idx, atom3_idx},
        sign=-1.0, max_hops=max_hops, decay=decay
    )

    # Propagate from atom3's side with opposite rotation direction
    # sign=+1.0 for opposite direction (the two sides rotate opposite ways)
    propagate_rotational_gradient_with_barriers(
        coords, gradient, atom3_idx, atom2_idx, rotation_axis, adjacency,
        barrier_atoms={atom2_idx, atom1_idx},
        sign=+1.0, max_hops=max_hops, decay=decay
    )

    # Vertex (atom2) does NOT propagate - it's the pivot point
    # Its gradient stays localized to itself

    # Re-normalize
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return theta, gradient


def dihedral_cv(
    coords: torch.Tensor,
    feats: dict,
    atom1_idx: int,
    atom2_idx: int,
    atom3_idx: int,
    atom4_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dihedral angle between four atoms as a CV.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom1_idx, atom2_idx, atom3_idx, atom4_idx: Indices of the four atoms

    Returns:
        dihedral: [multiplicity] dihedral angles in radians [-pi, pi]
        gradient: [multiplicity, N_atoms, 3] dDihedral/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)

        r1 = coords_grad[:, atom1_idx, :]
        r2 = coords_grad[:, atom2_idx, :]
        r3 = coords_grad[:, atom3_idx, :]
        r4 = coords_grad[:, atom4_idx, :]

        # Bond vectors
        b1 = r2 - r1
        b2 = r3 - r2
        b3 = r4 - r3

        # Normal vectors to planes
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)

        # Normalize
        n1_norm = torch.linalg.norm(n1, dim=-1, keepdim=True) + 1e-8
        n2_norm = torch.linalg.norm(n2, dim=-1, keepdim=True) + 1e-8
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # b2 unit vector
        b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-8
        b2_u = b2 / b2_norm

        # m1 = n1 x b2_u
        m1 = torch.cross(n1, b2_u, dim=-1)

        # Dihedral angle
        x = (n1 * n2).sum(dim=-1)
        y = (m1 * n2).sum(dim=-1)
        phi = torch.atan2(y, x)  # [mult]

        # Compute gradient of phi directly
        # atan2 is differentiable everywhere except origin (x=y=0)
        # d(atan2)/dx = -y/(x²+y²), d(atan2)/dy = x/(x²+y²)
        # The branch cut at ±π doesn't affect gradient magnitude
        phi_sum = phi.sum()

        try:
            gradient = torch.autograd.grad(phi_sum, coords_grad, create_graph=False)[0]
        except RuntimeError:
            gradient = torch.zeros_like(coords)

        # Normalize gradient
        grad_norms = gradient.norm(dim=-1, keepdim=True)
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            gradient = gradient / max_norm

        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            gradient = torch.zeros_like(gradient)
        if torch.isnan(phi).any() or torch.isinf(phi).any():
            phi = torch.zeros_like(phi)

    return phi.detach(), gradient.detach()


def dihedral_enhanced_cv(
    coords: torch.Tensor,
    feats: dict,
    atom1_idx: int,
    atom2_idx: int,
    atom3_idx: int,
    atom4_idx: int,
    max_hops: int = 10,
    decay: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dihedral CV with ROTATIONAL gradients propagated to bonded neighbors.

    This addresses the gradient sparsity problem of dihedral_cv by propagating
    the gradient from the 4 core atoms to their bonded neighbors using
    proper rotational tangent vectors (not just copied gradients).

    Propagation uses barriers to prevent gradient cancellation:
    - atom1 propagates to its side, blocked by central bond (atom2, atom3)
    - atom4 propagates to its side, blocked by central bond (atom2, atom3)
    - atom2 and atom3 (central bond) do not propagate (rotation axis)

    Rotational gradients ensure that propagated atoms move in the correct
    tangential direction for a dihedral twist around the central bond.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (must contain bond connectivity)
        atom1_idx, atom2_idx, atom3_idx, atom4_idx: Indices of the four atoms
        max_hops: Maximum bond distance for propagation (default 10)
        decay: Decay factor per hop (default 0.5 = 50% per bond)

    Returns:
        dihedral: [multiplicity] dihedral angles in radians [-pi, pi]
        gradient: [multiplicity, N_atoms, 3] enhanced dDihedral/dr with rotational tangents
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device

    # Get base dihedral and gradient
    phi, gradient = dihedral_cv(coords, feats, atom1_idx, atom2_idx, atom3_idx, atom4_idx)

    # Build adjacency
    adjacency = build_bond_adjacency(feats, n_atoms, coords.device)

    # Rotation axis is the central bond direction (atom2 -> atom3)
    r2 = coords[:, atom2_idx, :]  # [mult, 3]
    r3 = coords[:, atom3_idx, :]  # [mult, 3]
    rotation_axis = r3 - r2  # central bond direction
    axis_norm = rotation_axis.norm(dim=-1, keepdim=True) + 1e-8
    rotation_axis = rotation_axis / axis_norm  # unit vector [mult, 3]

    # Central bond atoms are the barriers (rotation axis)
    central_atoms = {atom2_idx, atom3_idx}

    # Propagate from atom1's side with rotational gradients
    # Pivot is atom2 (where atom1 connects to the central bond)
    propagate_rotational_gradient_with_barriers(
        coords, gradient, atom1_idx, atom2_idx, rotation_axis, adjacency,
        barrier_atoms=central_atoms | {atom4_idx},
        sign=-1.0, max_hops=max_hops, decay=decay
    )

    # Propagate from atom4's side with opposite rotation direction
    # Pivot is atom3 (where atom4 connects to the central bond)
    propagate_rotational_gradient_with_barriers(
        coords, gradient, atom4_idx, atom3_idx, rotation_axis, adjacency,
        barrier_atoms=central_atoms | {atom1_idx},
        sign=+1.0, max_hops=max_hops, decay=decay
    )

    # Central bond atoms (atom2, atom3) do NOT propagate - they define the rotation axis

    # Re-normalize
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return phi, gradient


def _compute_dihedral_value(coords, i1, i2, i3, i4):
    """Helper to compute dihedral value only."""
    r1, r2, r3, r4 = coords[:, i1], coords[:, i2], coords[:, i3], coords[:, i4]
    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    n1 = n1 / (torch.linalg.norm(n1, dim=-1, keepdim=True) + 1e-8)
    n2 = n2 / (torch.linalg.norm(n2, dim=-1, keepdim=True) + 1e-8)
    b2_u = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-8)
    m1 = torch.cross(n1, b2_u, dim=-1)
    return torch.atan2((m1 * n2).sum(dim=-1), (n1 * n2).sum(dim=-1))


# =============================================================================
# Contact-based CVs
# =============================================================================

def hbond_count_cv(
    coords: torch.Tensor,
    feats: dict,
    donor_mask: Optional[torch.Tensor] = None,
    acceptor_mask: Optional[torch.Tensor] = None,
    distance_cutoff: float = 3.5,
    angle_cutoff: float = 2.094,  # 120 degrees in radians (unused - would need H positions)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate hydrogen bond count using distance-based soft criterion.

    Since Boltz doesn't predict hydrogens, this uses backbone heavy atoms:
    - Donors: Backbone N atoms (the amide H is attached to N)
    - Acceptors: Backbone O atoms (carbonyl oxygen)

    The N-O distance cutoff of 3.5 Å approximates the N-H...O=C hydrogen bond
    geometry where the actual H...O distance would be ~2.0 Å.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (must contain ref_atom_name_chars for proper detection)
        donor_mask: Optional mask for donor heavy atoms [N_atoms]. If None, uses backbone N.
        acceptor_mask: Optional mask for acceptor atoms [N_atoms]. If None, uses backbone O.
        distance_cutoff: N-O distance cutoff in Angstroms (default 3.5 Å for backbone H-bonds)

    Returns:
        hbond_count: [multiplicity] soft hydrogen bond count
        gradient: [multiplicity, N_atoms, 3] dCount/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Default: use backbone N (donors) and O (acceptors) for protein H-bonds
    if donor_mask is None or acceptor_mask is None:
        default_donor, default_acceptor = get_backbone_donor_acceptor_masks(feats, n_atoms)
        if donor_mask is None:
            donor_mask = default_donor
        if acceptor_mask is None:
            acceptor_mask = default_acceptor

    donor_mask = donor_mask.to(device)
    acceptor_mask = acceptor_mask.to(device)

    # Compute pairwise distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [mult, N, N, 3]
    dist = torch.linalg.norm(diff, dim=-1)  # [mult, N, N]

    # Create pair mask (donor-acceptor pairs only)
    pair_mask = donor_mask.unsqueeze(1) & acceptor_mask.unsqueeze(0)  # [N, N]
    # Exclude self-interactions
    pair_mask = pair_mask & ~torch.eye(n_atoms, dtype=torch.bool, device=device)

    # Switching function: 1/(1 + exp(k*(d - d0)))
    k = 5.0  # Steepness
    d0 = distance_cutoff
    switching = torch.sigmoid(-k * (dist - d0))  # [mult, N, N]

    # Mask and sum
    switching_masked = switching * pair_mask.unsqueeze(0).float()
    hbond_count = switching_masked.sum(dim=(1, 2))  # [mult]

    # Gradient
    dsw_dd = -k * switching * (1 - switching)  # [mult, N, N]
    dsw_dd_masked = dsw_dd * pair_mask.unsqueeze(0).float()

    dist_safe = dist + 1e-8
    dd_dr = diff / dist_safe.unsqueeze(-1)  # [mult, N, N, 3]

    grad_i = (dsw_dd_masked.unsqueeze(-1) * dd_dr).sum(dim=2)  # [mult, N, 3]
    grad_j = (dsw_dd_masked.unsqueeze(-1) * (-dd_dr)).sum(dim=1)  # [mult, N, 3]

    gradient = grad_i + grad_j

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return hbond_count, gradient


def salt_bridges_cv(
    coords: torch.Tensor,
    feats: dict,
    positive_mask: Optional[torch.Tensor] = None,
    negative_mask: Optional[torch.Tensor] = None,
    distance_cutoff: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Count salt bridges between positively and negatively charged residues.

    Salt bridges are electrostatic interactions between:
    - Positive: Lys NZ, Arg NE/NH1/NH2 (guanidinium)
    - Negative: Asp OD1/OD2, Glu OE1/OE2 (carboxylate)

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (must contain ref_atom_name_chars, atom_to_token, res_type)
        positive_mask: Mask for positively charged atoms [N_atoms]. If None, auto-detects.
        negative_mask: Mask for negatively charged atoms [N_atoms]. If None, auto-detects.
        distance_cutoff: Distance cutoff for salt bridge (default 4.0 Å)

    Returns:
        salt_bridge_count: [multiplicity] soft salt bridge count
        gradient: [multiplicity, N_atoms, 3] dCount/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Default: auto-detect charged atoms from residue and atom types
    if positive_mask is None or negative_mask is None:
        default_positive, default_negative = get_charged_atom_masks(feats, n_atoms)
        if positive_mask is None:
            positive_mask = default_positive
        if negative_mask is None:
            negative_mask = default_negative

    positive_mask = positive_mask.to(device)
    negative_mask = negative_mask.to(device)

    # Pairwise distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)

    # Pair mask (positive-negative pairs)
    pair_mask = positive_mask.unsqueeze(1) & negative_mask.unsqueeze(0)

    # Switching function
    k = 5.0
    switching = torch.sigmoid(-k * (dist - distance_cutoff))
    switching_masked = switching * pair_mask.unsqueeze(0).float()

    count = switching_masked.sum(dim=(1, 2))

    # Gradient
    dsw_dd = -k * switching * (1 - switching)
    dsw_dd_masked = dsw_dd * pair_mask.unsqueeze(0).float()

    dist_safe = dist + 1e-8
    dd_dr = diff / dist_safe.unsqueeze(-1)

    grad_i = (dsw_dd_masked.unsqueeze(-1) * dd_dr).sum(dim=2)
    grad_j = (dsw_dd_masked.unsqueeze(-1) * (-dd_dr)).sum(dim=1)
    gradient = grad_i + grad_j

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return count, gradient


def contact_order_cv(
    coords: torch.Tensor,
    feats: dict,
    contact_cutoff: float = 8.0,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute relative contact order.

    CO = (1/N_contacts) * sum_{i,j in contact} |i - j| / L

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        contact_cutoff: Distance cutoff for contacts
        atom_mask: Optional mask for atoms to include

    Returns:
        contact_order: [multiplicity] relative contact order
        gradient: [multiplicity, N_atoms, 3] dCO/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
    atom_mask = atom_mask.to(device)
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]
    if atom_mask.shape[0] != n_atoms:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
    atom_mask = atom_mask.bool()

    # Pairwise distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)

    # Sequence separation
    idx = torch.arange(n_atoms, device=device, dtype=dtype)
    seq_sep = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))  # [N, N]

    # Pair mask (exclude close in sequence, min separation 3)
    pair_mask = (seq_sep >= 3) & atom_mask.unsqueeze(1) & atom_mask.unsqueeze(0)

    # Soft contact function
    k = 2.0
    contact_prob = torch.sigmoid(-k * (dist - contact_cutoff))  # [mult, N, N]
    contact_prob_masked = contact_prob * pair_mask.unsqueeze(0).float()

    # Weighted sum of sequence separations
    n_contacts = contact_prob_masked.sum(dim=(1, 2)) + 1e-8
    weighted_sep = (contact_prob_masked * seq_sep.unsqueeze(0)).sum(dim=(1, 2))

    L = atom_mask.sum().float()
    contact_order = weighted_sep / (n_contacts * L)

    # Gradient (approximate - complex due to ratio)
    dcp_dd = -k * contact_prob * (1 - contact_prob)
    dcp_dd_masked = dcp_dd * pair_mask.unsqueeze(0).float()

    dist_safe = dist + 1e-8
    dd_dr = diff / dist_safe.unsqueeze(-1)

    # d(weighted_sep)/dr
    dws_dr_i = (dcp_dd_masked.unsqueeze(-1) * seq_sep.unsqueeze(0).unsqueeze(-1) * dd_dr).sum(dim=2)
    dws_dr_j = (dcp_dd_masked.unsqueeze(-1) * seq_sep.unsqueeze(0).unsqueeze(-1) * (-dd_dr)).sum(dim=1)

    # d(n_contacts)/dr
    dnc_dr_i = (dcp_dd_masked.unsqueeze(-1) * dd_dr).sum(dim=2)
    dnc_dr_j = (dcp_dd_masked.unsqueeze(-1) * (-dd_dr)).sum(dim=1)

    # Quotient rule: d(ws/nc)/dr = (nc * dws - ws * dnc) / nc^2
    gradient = ((n_contacts.view(-1, 1, 1) * (dws_dr_i + dws_dr_j) -
                 weighted_sep.view(-1, 1, 1) * (dnc_dr_i + dnc_dr_j)) /
                (n_contacts.view(-1, 1, 1) ** 2 * L))

    gradient = gradient * atom_mask.unsqueeze(0).unsqueeze(-1)

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return contact_order, gradient


def local_contacts_cv(
    coords: torch.Tensor,
    feats: dict,
    contact_cutoff: float = 8.0,
    sequence_separation: int = 3,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Count local contacts (within sequence separation window).

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        contact_cutoff: Distance cutoff for contacts
        sequence_separation: Maximum sequence separation for "local"
        atom_mask: Optional mask for atoms to include

    Returns:
        local_contacts: [multiplicity] local contact count
        gradient: [multiplicity, N_atoms, 3] dCount/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
    atom_mask = atom_mask.to(device)
    while atom_mask.dim() > 1:
        atom_mask = atom_mask[0]
    if atom_mask.shape[0] != n_atoms:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
    atom_mask = atom_mask.bool()

    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)

    idx = torch.arange(n_atoms, device=device)
    seq_sep = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))

    # Local: sequence separation <= threshold but > 0
    pair_mask = ((seq_sep > 0) & (seq_sep <= sequence_separation) &
                 atom_mask.unsqueeze(1) & atom_mask.unsqueeze(0))

    k = 2.0
    contact_prob = torch.sigmoid(-k * (dist - contact_cutoff))
    contact_prob_masked = contact_prob * pair_mask.unsqueeze(0).float()

    count = contact_prob_masked.sum(dim=(1, 2))

    # Gradient
    dcp_dd = -k * contact_prob * (1 - contact_prob)
    dcp_dd_masked = dcp_dd * pair_mask.unsqueeze(0).float()

    dist_safe = dist + 1e-8
    dd_dr = diff / dist_safe.unsqueeze(-1)

    grad_i = (dcp_dd_masked.unsqueeze(-1) * dd_dr).sum(dim=2)
    grad_j = (dcp_dd_masked.unsqueeze(-1) * (-dd_dr)).sum(dim=1)
    gradient = grad_i + grad_j

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return count, gradient


# =============================================================================
# Solvent Accessible Surface Area (SASA)
# =============================================================================

def get_vdw_radii_from_feats(
    feats: dict,
    n_atoms: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Get VDW radii for each atom from features.

    Uses the ref_element one-hot encoding to look up VDW radii from const.vdw_radii.

    Args:
        feats: Feature dictionary containing ref_element
        n_atoms: Number of atoms
        device: Torch device
        dtype: Torch dtype

    Returns:
        radii: [n_atoms] VDW radii in Angstroms
    """
    from boltz.data import const

    # Get VDW radii tensor
    vdw_radii_list = const.vdw_radii
    n_vdw = len(vdw_radii_list)

    # Get ref_element from feats (one-hot encoded)
    ref_element = feats.get('ref_element', None)

    if ref_element is not None:
        # Handle batch dimension: [batch, N_atoms, num_elements]
        if ref_element.dim() == 3:
            ref_element = ref_element[0]
        ref_element = ref_element.to(device=device, dtype=dtype)

        # Truncate atoms if needed
        if ref_element.shape[0] > n_atoms:
            ref_element = ref_element[:n_atoms]
        elif ref_element.shape[0] < n_atoms:
            # Pad if needed (shouldn't happen normally)
            return torch.ones(n_atoms, device=device, dtype=dtype) * 1.7

        # Handle mismatch between num_elements and vdw_radii length
        # Truncate ref_element to match vdw_radii length, or pad vdw_radii
        n_elem = ref_element.shape[1]
        if n_elem > n_vdw:
            # Only use the first n_vdw columns (elements beyond vdw_radii get 0 weight)
            ref_element_truncated = ref_element[:, :n_vdw]
            vdw_radii = torch.tensor(vdw_radii_list, device=device, dtype=dtype)
        elif n_elem < n_vdw:
            # Truncate vdw_radii to match
            ref_element_truncated = ref_element
            vdw_radii = torch.tensor(vdw_radii_list[:n_elem], device=device, dtype=dtype)
        else:
            ref_element_truncated = ref_element
            vdw_radii = torch.tensor(vdw_radii_list, device=device, dtype=dtype)

        # Matmul: [N_atoms, n_elem] @ [n_elem] -> [N_atoms]
        radii = ref_element_truncated @ vdw_radii

        # For atoms with no valid element (radii = 0), use carbon radius as fallback
        radii = torch.where(radii > 0.1, radii, torch.tensor(1.7, device=device, dtype=dtype))
    else:
        # Fallback: use carbon radius (1.7 Å) for all atoms
        radii = torch.ones(n_atoms, device=device, dtype=dtype) * 1.7

    return radii


def sasa_lcpo(
    coords: torch.Tensor,
    radii: torch.Tensor,
    probe_radius: float = 1.4,
    cutoff: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LCPO (Linear Combination of Pairwise Overlaps) based differentiable SASA.

    Based on Weiser et al. (1999) "Approximate Solvent Accessible Surface Area
    from Tetrahedrally Directed Neighbor Densities".

    This is a simplified version that computes:
    SASA_i = P1 * S_i + P2 * sum_j(A_ij)

    where:
    - S_i = 4*pi*(r_i + r_probe)^2 is the isolated atom surface area
    - A_ij is the overlap area between atoms i and j

    Args:
        coords: [multiplicity, N_atoms, 3] atomic coordinates
        radii: [N_atoms] VDW radii
        probe_radius: Solvent probe radius (default 1.4 Å for water)
        cutoff: Distance cutoff for neighbor calculations

    Returns:
        sasa_total: [multiplicity] total SASA in Å²
        sasa_per_atom: [multiplicity, N_atoms] per-atom SASA in Å²
    """
    mult, n_atoms, _ = coords.shape
    device, dtype = coords.device, coords.dtype

    # Extended radii (VDW + probe)
    r_ext = radii + probe_radius  # [N]

    # Pairwise distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [mult, N, N, 3]
    dist = torch.linalg.norm(diff, dim=-1)  # [mult, N, N]

    # Sum of extended radii (overlap threshold)
    r_sum = r_ext.unsqueeze(0) + r_ext.unsqueeze(1)  # [N, N]

    # Soft overlap indicator (differentiable version of d_ij < r_i + r_j)
    # Sigmoid steepness controls transition sharpness
    sigmoid_k = 5.0
    overlap = torch.sigmoid(sigmoid_k * (r_sum.unsqueeze(0) - dist))  # [mult, N, N]

    # Zero out self-interactions
    eye_mask = torch.eye(n_atoms, device=device, dtype=dtype).unsqueeze(0)
    overlap = overlap * (1 - eye_mask)

    # Apply distance cutoff for efficiency
    cutoff_mask = (dist < cutoff).float()
    overlap = overlap * cutoff_mask

    # LCPO parameters (simplified atom-type-independent values)
    # These are approximations that work reasonably well for proteins
    P1 = 1.0    # Isolated surface contribution
    P2 = -0.5   # Pairwise overlap reduction factor

    # Isolated surface area: S_i = 4*pi*(r_i + r_probe)^2
    S_i = 4 * math.pi * r_ext**2  # [N]

    # Pairwise overlap area approximation
    # A_ij = pi * r_ext_i * (r_sum_ij - d_ij) when overlapping
    # Clamped to ensure non-negative overlap depth
    overlap_depth = torch.clamp(r_sum.unsqueeze(0) - dist, min=0)  # [mult, N, N]
    A_ij = math.pi * r_ext.view(1, -1, 1) * overlap_depth * overlap  # [mult, N, N]

    # Per-atom SASA: SASA_i = P1 * S_i + P2 * sum_j(A_ij)
    sasa_per_atom = P1 * S_i.unsqueeze(0) + P2 * A_ij.sum(dim=2)  # [mult, N]

    # Physical constraint: SASA cannot be negative
    sasa_per_atom = torch.clamp(sasa_per_atom, min=0)

    # Total SASA
    sasa_total = sasa_per_atom.sum(dim=1)  # [mult]

    return sasa_total, sasa_per_atom


def sasa_coordination(
    coords: torch.Tensor,
    radii: torch.Tensor,
    probe_radius: float = 1.4,
    cutoff: float = 8.0,
    max_neighbors: float = 12.0,
) -> torch.Tensor:
    """
    Coordination-based SASA approximation (faster but less accurate).

    Approximates SASA by counting overlapping neighbors with soft switching.
    Atoms with more neighbors are more "buried" and have less SASA.

    Args:
        coords: [multiplicity, N_atoms, 3] atomic coordinates
        radii: [N_atoms] VDW radii
        probe_radius: Solvent probe radius
        cutoff: Distance cutoff for neighbor counting
        max_neighbors: Maximum expected neighbors (normalization)

    Returns:
        sasa_total: [multiplicity] total SASA in Å²
    """
    mult, n_atoms, _ = coords.shape
    device, dtype = coords.device, coords.dtype

    # Extended radii
    r_ext = radii + probe_radius  # [N]

    # Pairwise distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)  # [mult, N, N]

    # Overlap threshold: atoms overlap when d < r_i + r_j + buffer
    buffer = 0.5  # Small buffer for transition region
    r_sum = r_ext.unsqueeze(0) + r_ext.unsqueeze(1) + buffer

    # Soft neighbor count (burial)
    sigmoid_k = 3.0
    neighbor_contribution = torch.sigmoid(sigmoid_k * (r_sum.unsqueeze(0) - dist))

    # Zero out self
    eye_mask = torch.eye(n_atoms, device=device, dtype=dtype).unsqueeze(0)
    neighbor_contribution = neighbor_contribution * (1 - eye_mask)

    # Apply cutoff
    cutoff_mask = (dist < cutoff).float()
    neighbor_contribution = neighbor_contribution * cutoff_mask

    # Per-atom burial (normalized)
    burial = neighbor_contribution.sum(dim=2) / max_neighbors  # [mult, N]
    burial = torch.clamp(burial, max=1.0)

    # Per-atom SASA: max area * (1 - burial)
    S_i = 4 * math.pi * r_ext**2  # Isolated surface area
    sasa_per_atom = S_i.unsqueeze(0) * (1 - burial)  # [mult, N]

    # Total SASA
    sasa_total = sasa_per_atom.sum(dim=1)  # [mult]

    return sasa_total


def sasa_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
    probe_radius: float = 1.4,
    method: str = "lcpo",
    per_residue: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable Solvent Accessible Surface Area (SASA) collective variable.

    SASA measures the surface area of a biomolecule accessible to solvent.
    This is computed by conceptually rolling a probe sphere (representing
    water, radius 1.4 Å) over the van der Waals surface of the molecule.

    This implementation provides two methods:
    - "lcpo": LCPO analytical approximation (more accurate, based on pairwise overlaps)
    - "coordination": Soft neighbor counting (faster, more approximate)

    The gradient dSASA/d(coords) is computed via automatic differentiation,
    enabling use in metadynamics steering toward exposed or buried conformations.

    Args:
        coords: [multiplicity, N_atoms, 3] atomic coordinates in Angstroms
        feats: Feature dictionary from Boltz (contains ref_element for VDW radii)
        atom_mask: Optional [N_atoms] boolean mask for subset of atoms
        probe_radius: Solvent probe radius in Angstroms (default 1.4 for water)
        method: "lcpo" (analytical, recommended) or "coordination" (fast, approximate)
        per_residue: If True, aggregate per-residue SASA (uses atom_to_token)

    Returns:
        sasa: [multiplicity] total SASA in Å²
        gradient: [multiplicity, N_atoms, 3] normalized dSASA/dr

    Example YAML usage:
        metadiffusion:
          - explore:
              collective_variable: sasa
              strength: 0.1        # Bias toward higher SASA (unfolding)
              probe_radius: 1.4    # Water probe
              method: lcpo         # or "coordination" for speed

        # Or for steering to specific SASA:
          - steer:
              collective_variable: sasa
              target: 15000        # Target SASA in Å²
              strength: 0.5
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get VDW radii from features
    radii = get_vdw_radii_from_feats(feats, n_atoms, device, dtype)

    # Apply atom mask if provided
    if atom_mask is not None:
        atom_mask = atom_mask.to(device)
        while atom_mask.dim() > 1:
            atom_mask = atom_mask[0]
        if atom_mask.shape[0] != n_atoms:
            atom_mask = None

    # Create masked versions for computation
    if atom_mask is not None:
        atom_mask = atom_mask.bool()
        # For masked atoms, set radius to 0 (they won't contribute)
        radii = radii * atom_mask.float()

    # Compute SASA with gradient tracking
    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)

        if method == "lcpo":
            sasa, sasa_per_atom = sasa_lcpo(coords_grad, radii, probe_radius)
        elif method == "coordination":
            sasa = sasa_coordination(coords_grad, radii, probe_radius)
        else:
            raise ValueError(f"Unknown SASA method: {method}. Use 'lcpo' or 'coordination'.")

        # Backprop for gradient (sum over multiplicity for scalar loss)
        sasa_sum = sasa.sum()
        gradient = torch.autograd.grad(
            sasa_sum, coords_grad, create_graph=False, retain_graph=False
        )[0]

    # Normalize gradient (standard for CVs in this codebase)
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    # Handle NaN/Inf (shouldn't happen but safety check)
    if torch.isnan(gradient).any() or torch.isinf(gradient).any():
        gradient = torch.zeros_like(gradient)

    return sasa.detach(), gradient.detach()


# =============================================================================
# Multi-chain / Domain CVs
# =============================================================================

def inter_domain_cv(
    coords: torch.Tensor,
    feats: dict,
    domain1_mask: torch.Tensor,
    domain2_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute center-of-mass distance between two domains.

    This is identical to inter_chain_cv but named for domain usage.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        domain1_mask: Mask for first domain atoms [N_atoms]
        domain2_mask: Mask for second domain atoms [N_atoms]

    Returns:
        distance: [multiplicity] COM distance in Angstroms
        gradient: [multiplicity, N_atoms, 3] dDistance/dr
    """
    return inter_chain_cv(coords, feats, domain1_mask, domain2_mask)


def hinge_angle_cv(
    coords: torch.Tensor,
    feats: dict,
    domain1_mask: torch.Tensor,
    hinge_mask: torch.Tensor,
    domain2_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute hinge angle between two domains connected by a hinge region.

    The angle is computed between vectors from hinge COM to each domain COM.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        domain1_mask: Mask for first domain atoms [N_atoms]
        hinge_mask: Mask for hinge region atoms [N_atoms]
        domain2_mask: Mask for second domain atoms [N_atoms]

    Returns:
        angle: [multiplicity] hinge angle in radians
        gradient: [multiplicity, N_atoms, 3] dAngle/dr
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Use torch.inference_mode(False) to enable gradient computation
    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)

        # Clone masks to avoid inference tensor issues
        domain1_mask_f = domain1_mask.detach().clone().to(device).float()
        hinge_mask_f = hinge_mask.detach().clone().to(device).float()
        domain2_mask_f = domain2_mask.detach().clone().to(device).float()

        n1 = domain1_mask_f.sum() + 1e-8
        n_hinge = hinge_mask_f.sum() + 1e-8
        n2 = domain2_mask_f.sum() + 1e-8

        # COMs
        com1 = (coords_grad * domain1_mask_f.view(1, -1, 1)).sum(dim=1) / n1
        com_hinge = (coords_grad * hinge_mask_f.view(1, -1, 1)).sum(dim=1) / n_hinge
        com2 = (coords_grad * domain2_mask_f.view(1, -1, 1)).sum(dim=1) / n2

        # Vectors from hinge to domains
        v1 = com1 - com_hinge
        v2 = com2 - com_hinge

        # Angle between vectors
        v1_norm = torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8
        v2_norm = torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8
        u1 = v1 / v1_norm
        u2 = v2 / v2_norm

        cos_theta = (u1 * u2).sum(dim=-1)
        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)

        # Compute gradient using autograd
        theta_sum = theta.sum()
        try:
            gradient = torch.autograd.grad(theta_sum, coords_grad, create_graph=False)[0]
        except RuntimeError:
            gradient = torch.zeros_like(coords)

        # Normalize gradient for effective steering
        grad_norms = gradient.norm(dim=-1, keepdim=True)
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            gradient = gradient / max_norm

        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            gradient = torch.zeros_like(gradient)
        if torch.isnan(theta).any() or torch.isinf(theta).any():
            theta = torch.zeros_like(theta)

    return theta.detach(), gradient.detach()


def _compute_hinge_angle(coords, d1_mask, h_mask, d2_mask):
    """Helper to compute hinge angle value (kept for compatibility)."""
    n1 = d1_mask.sum() + 1e-8
    nh = h_mask.sum() + 1e-8
    n2 = d2_mask.sum() + 1e-8
    com1 = (coords * d1_mask.view(1, -1, 1)).sum(dim=1) / n1
    comh = (coords * h_mask.view(1, -1, 1)).sum(dim=1) / nh
    com2 = (coords * d2_mask.view(1, -1, 1)).sum(dim=1) / n2
    v1, v2 = com1 - comh, com2 - comh
    u1 = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + 1e-8)
    u2 = v2 / (torch.linalg.norm(v2, dim=-1, keepdim=True) + 1e-8)
    cos_t = torch.clamp((u1 * u2).sum(dim=-1), -1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_t)


# =============================================================================
# Secondary Structure CVs (RMSD to ideal structures)
# =============================================================================

# Ideal backbone dihedral angles for secondary structures
ALPHA_HELIX_PHI = -57.0 * 3.14159 / 180.0  # -57 degrees
ALPHA_HELIX_PSI = -47.0 * 3.14159 / 180.0  # -47 degrees
BETA_SHEET_PHI = -120.0 * 3.14159 / 180.0  # -120 degrees
BETA_SHEET_PSI = 120.0 * 3.14159 / 180.0   # 120 degrees


def alpharmsd_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute deviation from ideal alpha helix geometry using CA-CA distances.

    Ideal CA-CA distances in alpha helix:
    - i to i+1: 3.8 Å
    - i to i+2: 5.4 Å
    - i to i+3: 5.0 Å
    - i to i+4: 6.2 Å

    Note: This CV requires CA atoms only. If atom_mask is not provided,
    it will attempt to identify CA atoms from feats.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (should contain ref_atom_name_chars for CA detection)
        atom_mask: Optional mask for CA atoms [N_atoms]. If None, auto-detects CA atoms.

    Returns:
        alpha_score: [multiplicity] alpha helix similarity (higher = more helical)
        gradient: [multiplicity, N_atoms, 3]
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get CA atom mask - this CV only makes sense for CA atoms
    ca_mask = get_backbone_atom_mask(feats, n_atoms, atom_name="CA").to(device)
    if atom_mask is not None:
        # atom_mask may be a groups-based all-atom mask; intersect with CA detection
        atom_mask = atom_mask.to(device)
        while atom_mask.dim() > 1:
            atom_mask = atom_mask[0]
        if atom_mask.shape[0] == n_atoms:
            ca_mask = ca_mask & atom_mask.bool()
    atom_mask = ca_mask

    # Get indices of CA atoms
    ca_indices = torch.where(atom_mask)[0]
    n_ca = len(ca_indices)

    # Ideal CA-CA distances in alpha helix
    ideal_dist = {1: 3.8, 2: 5.4, 3: 5.0, 4: 6.2}

    total_score = torch.zeros(multiplicity, device=device, dtype=dtype)
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)
    n_pairs = 0

    for sep, d_ideal in ideal_dist.items():
        if sep >= n_ca:
            continue

        # Loop over CA atoms by sequence position
        for i in range(n_ca - sep):
            ca_i = ca_indices[i].item()
            ca_j = ca_indices[i + sep].item()

            r_i = coords[:, ca_i, :]
            r_j = coords[:, ca_j, :]
            diff = r_j - r_i
            dist = torch.linalg.norm(diff, dim=-1) + 1e-8

            # Score: exp(-((d - d_ideal)/sigma)^2)
            sigma = 0.5
            delta = dist - d_ideal
            score = torch.exp(-(delta / sigma) ** 2)
            total_score += score

            # Gradient
            dscore_dd = -2 * delta / (sigma ** 2) * score
            unit_vec = diff / dist.unsqueeze(-1)

            gradient[:, ca_i, :] -= dscore_dd.unsqueeze(-1) * unit_vec
            gradient[:, ca_j, :] += dscore_dd.unsqueeze(-1) * unit_vec

            n_pairs += 1

    if n_pairs > 0:
        total_score = total_score / n_pairs
        gradient = gradient / n_pairs

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return total_score, gradient


def antibetarmsd_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute deviation from ideal antiparallel beta sheet geometry.

    Uses extended CA-CA distances typical of beta sheets.
    Only considers CA atoms for the distance calculations.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom_mask: Optional mask for CA atoms (auto-detected if None)

    Returns:
        beta_score: [multiplicity] antiparallel beta similarity
        gradient: [multiplicity, N_atoms, 3]
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get CA atom mask - this CV only makes sense for CA atoms
    ca_mask = get_backbone_atom_mask(feats, n_atoms, atom_name="CA").to(device)
    if atom_mask is not None:
        # atom_mask may be a groups-based all-atom mask; intersect with CA detection
        atom_mask = atom_mask.to(device)
        while atom_mask.dim() > 1:
            atom_mask = atom_mask[0]
        if atom_mask.shape[0] == n_atoms:
            ca_mask = ca_mask & atom_mask.bool()
    atom_mask = ca_mask

    # Get indices of CA atoms
    ca_indices = torch.where(atom_mask)[0]
    n_ca = len(ca_indices)

    # Ideal CA-CA distances in beta sheet (more extended)
    ideal_dist = {1: 3.8, 2: 6.7}  # Extended configuration

    total_score = torch.zeros(multiplicity, device=device, dtype=dtype)
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)
    n_pairs = 0

    for sep, d_ideal in ideal_dist.items():
        if sep >= n_ca:
            continue

        for i in range(n_ca - sep):
            # Get actual atom indices for CA atoms
            ca_i = ca_indices[i].item()
            ca_j = ca_indices[i + sep].item()

            r_i = coords[:, ca_i, :]
            r_j = coords[:, ca_j, :]
            diff = r_j - r_i
            dist = torch.linalg.norm(diff, dim=-1) + 1e-8

            sigma = 0.5
            delta = dist - d_ideal
            score = torch.exp(-(delta / sigma) ** 2)
            total_score += score

            dscore_dd = -2 * delta / (sigma ** 2) * score
            unit_vec = diff / dist.unsqueeze(-1)

            gradient[:, ca_i, :] -= dscore_dd.unsqueeze(-1) * unit_vec
            gradient[:, ca_j, :] += dscore_dd.unsqueeze(-1) * unit_vec

            n_pairs += 1

    if n_pairs > 0:
        total_score = total_score / n_pairs
        gradient = gradient / n_pairs

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return total_score, gradient


def parabetarmsd_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute deviation from ideal parallel beta sheet geometry.

    Similar to antiparallel but with slightly different geometry.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom_mask: Optional mask for atoms

    Returns:
        beta_score: [multiplicity] parallel beta similarity
        gradient: [multiplicity, N_atoms, 3]
    """
    # Parallel beta has similar distances to antiparallel
    return antibetarmsd_cv(coords, feats, atom_mask)


# =============================================================================
# Shape CVs (Gyration Tensor)
# =============================================================================

def acylindricity_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute acylindricity from gyration tensor.

    Acylindricity c = lambda_y - lambda_x, where lambda are eigenvalues
    of the gyration tensor sorted as lambda_z >= lambda_y >= lambda_x.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom_mask: Optional mask for atoms

    Returns:
        acylindricity: [multiplicity]
        gradient: [multiplicity, N_atoms, 3]
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom mask from feats if not provided
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
        if atom_mask.dim() == 2:
            atom_mask = atom_mask[0]  # [N_atoms]

    atom_mask = atom_mask.to(device)

    # Use torch.inference_mode(False) to override inference mode
    with torch.inference_mode(False):
        # Clone coords and atom_mask to avoid inference tensor issues
        coords_grad = coords.detach().clone().requires_grad_(True)
        atom_mask_float = atom_mask.detach().clone().float()
        n_valid = atom_mask_float.sum() + 1e-8

        # COM
        masked_coords = coords_grad * atom_mask_float.view(1, -1, 1)
        com = masked_coords.sum(dim=1, keepdim=True) / n_valid

        # Centered coordinates
        centered = (coords_grad - com) * atom_mask_float.view(1, -1, 1)

        # Gyration tensor with regularization for numerical stability
        gyration = torch.einsum('mia,mib->mab', centered, centered) / n_valid

        # Add small regularization to diagonal
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        gyration = gyration + 1e-6 * eye

        # Eigenvalues (sorted ascending)
        try:
            eigenvalues = torch.linalg.eigvalsh(gyration)
        except torch._C._LinAlgError:
            return torch.zeros(multiplicity, device=device, dtype=dtype), torch.zeros_like(coords)

        # Acylindricity = lambda_y - lambda_x (middle - smallest)
        acyl = eigenvalues[:, 1] - eigenvalues[:, 0]

        # Compute gradient using autograd
        acyl_sum = acyl.sum()
        try:
            gradient = torch.autograd.grad(acyl_sum, coords_grad, create_graph=False)[0]
        except RuntimeError:
            gradient = torch.zeros_like(coords)

        # Zero out gradient for masked atoms
        gradient = gradient * atom_mask_float.view(1, -1, 1)

        # Normalize gradient to have unit max norm
        grad_norms = gradient.norm(dim=-1, keepdim=True)
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            gradient = gradient / max_norm

        # Check for NaN/Inf and replace with zeros
        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            gradient = torch.zeros_like(gradient)
        if torch.isnan(acyl).any() or torch.isinf(acyl).any():
            acyl = torch.zeros_like(acyl)

    return acyl.detach(), gradient.detach()


def shape_gyration_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute shape parameter from gyration tensor.

    Shape = 1 - 3 * (lambda_x*lambda_y + lambda_y*lambda_z + lambda_x*lambda_z) / (lambda_x + lambda_y + lambda_z)^2

    Ranges from 0 (sphere) to 1 (linear).

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        atom_mask: Optional mask for atoms

    Returns:
        shape: [multiplicity] shape parameter
        gradient: [multiplicity, N_atoms, 3]
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get atom mask from feats if not provided
    if atom_mask is None:
        atom_mask = feats.get('atom_pad_mask', torch.ones(1, n_atoms, dtype=torch.bool, device=device))
        if atom_mask.dim() == 2:
            atom_mask = atom_mask[0]

    atom_mask = atom_mask.to(device)

    # Use torch.inference_mode(False) to override inference mode
    with torch.inference_mode(False):
        coords_grad = coords.detach().clone().requires_grad_(True)
        atom_mask_float = atom_mask.detach().clone().float()
        n_valid = atom_mask_float.sum() + 1e-8

        masked_coords = coords_grad * atom_mask_float.view(1, -1, 1)
        com = masked_coords.sum(dim=1, keepdim=True) / n_valid
        centered = (coords_grad - com) * atom_mask_float.view(1, -1, 1)

        gyration = torch.einsum('mia,mib->mab', centered, centered) / n_valid

        # Add regularization
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        gyration = gyration + 1e-6 * eye

        try:
            eigenvalues = torch.linalg.eigvalsh(gyration)
        except torch._C._LinAlgError:
            return torch.zeros(multiplicity, device=device, dtype=dtype), torch.zeros_like(coords)

        lx, ly, lz = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        trace = lx + ly + lz + 1e-8
        cross_sum = lx * ly + ly * lz + lx * lz

        shape = 1 - 3 * cross_sum / (trace ** 2)

        # Compute gradient using autograd
        shape_sum = shape.sum()
        try:
            gradient = torch.autograd.grad(shape_sum, coords_grad, create_graph=False)[0]
        except RuntimeError:
            gradient = torch.zeros_like(coords)

        gradient = gradient * atom_mask_float.view(1, -1, 1)

        # Normalize gradient
        grad_norms = gradient.norm(dim=-1, keepdim=True)
        max_norm = grad_norms.max()
        if max_norm > 1e-8:
            gradient = gradient / max_norm

        if torch.isnan(gradient).any() or torch.isinf(gradient).any():
            gradient = torch.zeros_like(gradient)
        if torch.isnan(shape).any() or torch.isinf(shape).any():
            shape = torch.zeros_like(shape)

    return shape.detach(), gradient.detach()


# =============================================================================
# Secondary Structure Content CVs
# =============================================================================

def helix_content_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate helix content based on CA-CA distances.

    In an alpha helix, consecutive CA atoms (i to i+4) have a characteristic
    distance of ~6.3 Å. This CV measures how many such pairs match this pattern.

    Note: This CV requires CA atoms only. If atom_mask is not provided,
    it will attempt to identify CA atoms from feats. The loop iterates over
    CA atom indices, not all atom indices.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (should contain ref_atom_name_chars for CA detection)
        atom_mask: Optional mask for CA atoms [N_atoms]. If None, auto-detects CA atoms.

    Returns:
        helix_content: [multiplicity] fraction of helical structure
        gradient: [multiplicity, N_atoms, 3]
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get CA atom mask - this CV only makes sense for CA atoms
    ca_mask = get_backbone_atom_mask(feats, n_atoms, atom_name="CA").to(device)
    if atom_mask is not None:
        # atom_mask may be a groups-based all-atom mask; intersect with CA detection
        atom_mask = atom_mask.to(device)
        while atom_mask.dim() > 1:
            atom_mask = atom_mask[0]
        if atom_mask.shape[0] == n_atoms:
            ca_mask = ca_mask & atom_mask.bool()
    atom_mask = ca_mask

    # Get indices of CA atoms
    ca_indices = torch.where(atom_mask)[0]
    n_ca = len(ca_indices)

    # i->i+4 CA-CA distance in helix ~6.3 A
    ideal_dist = 6.3
    sigma = 1.0

    total_score = torch.zeros(multiplicity, device=device, dtype=dtype)
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)
    n_pairs = 0

    # Loop over CA atoms by their sequence position (not atom index)
    for i in range(n_ca - 4):
        ca_i = ca_indices[i].item()
        ca_j = ca_indices[i + 4].item()

        r_i = coords[:, ca_i, :]
        r_j = coords[:, ca_j, :]
        diff = r_j - r_i
        dist = torch.linalg.norm(diff, dim=-1) + 1e-8

        delta = dist - ideal_dist
        score = torch.exp(-(delta / sigma) ** 2)
        total_score += score

        dscore_dd = -2 * delta / (sigma ** 2) * score
        unit_vec = diff / dist.unsqueeze(-1)

        gradient[:, ca_i, :] -= dscore_dd.unsqueeze(-1) * unit_vec
        gradient[:, ca_j, :] += dscore_dd.unsqueeze(-1) * unit_vec

        n_pairs += 1

    n_possible = max(1, n_ca - 4)
    helix_content = total_score / max(1, n_possible)
    gradient = gradient / max(1, n_possible)

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return helix_content, gradient


def sheet_content_cv(
    coords: torch.Tensor,
    feats: dict,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate sheet content based on CA-CA distances.

    In a beta sheet, consecutive CA atoms (i to i+2) have a characteristic
    distance of ~6.7 Å (extended conformation). This CV measures how many
    such pairs match this pattern.

    Note: This CV requires CA atoms only. If atom_mask is not provided,
    it will attempt to identify CA atoms from feats.

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary (should contain ref_atom_name_chars for CA detection)
        atom_mask: Optional mask for CA atoms [N_atoms]. If None, auto-detects CA atoms.

    Returns:
        sheet_content: [multiplicity] fraction of sheet-like structure
        gradient: [multiplicity, N_atoms, 3]
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Get CA atom mask - this CV only makes sense for CA atoms
    ca_mask = get_backbone_atom_mask(feats, n_atoms, atom_name="CA").to(device)
    if atom_mask is not None:
        # atom_mask may be a groups-based all-atom mask; intersect with CA detection
        atom_mask = atom_mask.to(device)
        while atom_mask.dim() > 1:
            atom_mask = atom_mask[0]
        if atom_mask.shape[0] == n_atoms:
            ca_mask = ca_mask & atom_mask.bool()
    atom_mask = ca_mask

    # Get indices of CA atoms
    ca_indices = torch.where(atom_mask)[0]
    n_ca = len(ca_indices)

    # i->i+2 CA-CA distance in sheet ~6.7 A (extended)
    ideal_dist = 6.7
    sigma = 0.8

    total_score = torch.zeros(multiplicity, device=device, dtype=dtype)
    gradient = torch.zeros(multiplicity, n_atoms, 3, device=device, dtype=dtype)
    n_pairs = 0

    # Loop over CA atoms by their sequence position
    for i in range(n_ca - 2):
        ca_i = ca_indices[i].item()
        ca_j = ca_indices[i + 2].item()

        r_i = coords[:, ca_i, :]
        r_j = coords[:, ca_j, :]
        diff = r_j - r_i
        dist = torch.linalg.norm(diff, dim=-1) + 1e-8

        delta = dist - ideal_dist
        score = torch.exp(-(delta / sigma) ** 2)
        total_score += score

        dscore_dd = -2 * delta / (sigma ** 2) * score
        unit_vec = diff / dist.unsqueeze(-1)

        gradient[:, ca_i, :] -= dscore_dd.unsqueeze(-1) * unit_vec
        gradient[:, ca_j, :] += dscore_dd.unsqueeze(-1) * unit_vec

        n_pairs += 1

    n_possible = max(1, n_ca - 2)
    sheet_content = total_score / max(1, n_possible)
    gradient = gradient / max(1, n_possible)

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return sheet_content, gradient


def dipole_moment_cv(
    coords: torch.Tensor,
    feats: dict,
    charges: Optional[torch.Tensor] = None,
    atom_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute magnitude of dipole moment.

    mu = |sum_i q_i * r_i|

    Args:
        coords: Atom coordinates [multiplicity, N_atoms, 3]
        feats: Feature dictionary
        charges: Partial charges for each atom [N_atoms]. If None, uses formal
                 charges from charged residues (Lys/Arg = +1, Asp/Glu = -1).
        atom_mask: Optional mask for atoms

    Returns:
        dipole: [multiplicity] dipole moment magnitude
        gradient: [multiplicity, N_atoms, 3]
    """
    multiplicity, n_atoms, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    if atom_mask is None:
        atom_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
    atom_mask = atom_mask.to(device).float()

    if charges is None:
        # Use formal charges from charged residues (Lys/Arg = +1, Asp/Glu = -1)
        # This is physically meaningful unlike arbitrary alternating charges
        positive_mask, negative_mask = get_charged_atom_masks(feats, n_atoms)
        charges = torch.zeros(n_atoms, device=device, dtype=dtype)
        charges[positive_mask] = 1.0  # +1 for Lys NZ, Arg NE/NH1/NH2
        charges[negative_mask] = -1.0  # -1 for Asp OD1/OD2, Glu OE1/OE2

    charges = charges.to(device=device, dtype=dtype)

    # Dipole vector
    weighted_coords = coords * (charges * atom_mask).view(1, -1, 1)
    dipole_vec = weighted_coords.sum(dim=1)  # [mult, 3]

    # Magnitude
    dipole_mag = torch.linalg.norm(dipole_vec, dim=-1) + 1e-8  # [mult]

    # Gradient: d|mu|/dr_i = q_i * mu / |mu|
    unit_dipole = dipole_vec / dipole_mag.unsqueeze(-1)  # [mult, 3]

    gradient = (charges * atom_mask).view(1, -1, 1) * unit_dipole.unsqueeze(1)

    # Normalize gradient
    grad_norms = gradient.norm(dim=-1, keepdim=True)
    max_norm = grad_norms.max()
    if max_norm > 1e-8:
        gradient = gradient / max_norm

    return dipole_mag, gradient


# CV Registry
CV_REGISTRY = {
    # Structural
    "rg": radius_of_gyration_cv,
    "distance": distance_cv,
    "distance_region": distance_region_cv,
    "min_distance": min_distance_cv,
    "max_diameter": max_diameter_cv,
    "asphericity": asphericity_cv,
    # Angle/Dihedral
    "angle": angle_cv,
    "dihedral": dihedral_cv,
    "angle_enhanced": angle_enhanced_cv,
    "dihedral_enhanced": dihedral_enhanced_cv,
    "angle_region": angle_region_cv,
    "dihedral_region": dihedral_region_cv,
    # RMSD / Fluctuation
    "rmsd": rmsd_cv,
    "pair_rmsd": pair_rmsd_cv,
    "pair_rmsd_norm_rg": pair_rmsd_norm_rg_cv,
    "pair_rmsd_grouped": pair_rmsd_grouped_cv,
    "rmsf": rmsf_cv,
    # Contacts
    "native_contacts": native_contacts_cv,
    "coordination": coordination_cv,
    "hbond_count": hbond_count_cv,
    "salt_bridges": salt_bridges_cv,
    "contact_order": contact_order_cv,
    "local_contacts": local_contacts_cv,
    "sasa": sasa_cv,
    # Multi-chain/Domain - DEPRECATED, use distance/angle with region1/region2 instead
    "inter_chain": inter_chain_cv,  # DEPRECATED: use distance with region1="A" region2="B"
    "inter_domain": inter_domain_cv,  # DEPRECATED: use distance with region1/region2
    "hinge_angle": hinge_angle_cv,  # DEPRECATED: use angle with region1/region2/region3
    # Secondary structure
    "alpharmsd": alpharmsd_cv,
    "antibetarmsd": antibetarmsd_cv,
    "parabetarmsd": parabetarmsd_cv,
    # Shape
    "acylindricity": acylindricity_cv,
    "shape_gyration": shape_gyration_cv,
    # Content
    "helix_content": helix_content_cv,
    "sheet_content": sheet_content_cv,
    # Other
    "dipole_moment": dipole_moment_cv,
}


def create_cv_function(
    cv_type: str,
    **kwargs
) -> Callable:
    """
    Factory function to create a CV function.

    Args:
        cv_type: Type of CV (see CV_REGISTRY for supported types)
        **kwargs: Additional arguments for specific CV types:
            - atom1_idx, atom2_idx: For distance CV
            - reference_coords: For rmsd/native_contacts CVs
            - contact_cutoff: For native_contacts/coordination CVs
            - chain1_mask, chain2_mask: For inter_chain CV

    Returns:
        cv_function: Callable (coords, feats, current_step) -> (cv_value, cv_gradient)
    """
    if cv_type == "rg":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: radius_of_gyration_cv(coords, feats, atom_mask)

    elif cv_type == "distance":
        atom1_idx = kwargs.get('atom1_idx', 0)
        atom2_idx = kwargs.get('atom2_idx', -1)
        return lambda coords, feats, step=0: distance_cv(coords, feats, atom1_idx, atom2_idx)

    elif cv_type == "distance_region":
        region1_mask = kwargs.get('region1_mask')
        region2_mask = kwargs.get('region2_mask')
        if region1_mask is None or region2_mask is None:
            raise ValueError("region1_mask and region2_mask required for 'distance_region' CV")
        return lambda coords, feats, step=0: distance_region_cv(coords, feats, region1_mask, region2_mask)

    elif cv_type == "min_distance":
        region1_mask = kwargs.get('region1_mask')
        region2_mask = kwargs.get('region2_mask')
        if region1_mask is None or region2_mask is None:
            raise ValueError("region1_mask and region2_mask required for 'min_distance' CV")
        softmin_beta = kwargs.get('softmin_beta', 10.0)
        return lambda coords, feats, step=0: min_distance_cv(coords, feats, region1_mask, region2_mask, softmin_beta)

    elif cv_type == "asphericity":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: asphericity_cv(coords, feats, atom_mask)

    elif cv_type == "rmsd":
        reference_coords = kwargs.get('reference_coords')
        if reference_coords is None:
            raise ValueError("reference_coords required for 'rmsd' CV")
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: rmsd_cv(coords, feats, reference_coords, atom_mask)

    elif cv_type == "pair_rmsd":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: pair_rmsd_cv(coords, feats, atom_mask)

    elif cv_type == "pair_rmsd_norm_rg":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: pair_rmsd_norm_rg_cv(coords, feats, atom_mask)

    elif cv_type == "pair_rmsd_grouped":
        atom_mask = kwargs.get('atom_mask', None)
        align_mask = kwargs.get('align_mask', None)
        return lambda coords, feats, step=0: pair_rmsd_grouped_cv(coords, feats, atom_mask, align_mask)

    elif cv_type == "rmsf":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: rmsf_cv(coords, feats, atom_mask)

    elif cv_type == "native_contacts":
        reference_coords = kwargs.get('reference_coords')
        if reference_coords is None:
            raise ValueError("reference_coords required for 'native_contacts' CV")
        contact_cutoff = kwargs.get('contact_cutoff', 4.5)
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: native_contacts_cv(
            coords, feats, reference_coords, contact_cutoff, atom_mask
        )

    elif cv_type == "coordination":
        contact_cutoff = kwargs.get('contact_cutoff', 6.0)
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: coordination_cv(coords, feats, contact_cutoff, atom_mask)

    elif cv_type == "inter_chain":
        chain1_mask = kwargs.get('chain1_mask')
        chain2_mask = kwargs.get('chain2_mask')
        if chain1_mask is None or chain2_mask is None:
            raise ValueError("chain1_mask and chain2_mask required for 'inter_chain' CV")
        return lambda coords, feats, step=0: inter_chain_cv(coords, feats, chain1_mask, chain2_mask)

    elif cv_type == "max_diameter":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: max_diameter_cv(coords, feats, atom_mask)

    # Angle/Dihedral CVs
    elif cv_type == "angle":
        atom1_idx = kwargs.get('atom1_idx', 0)
        atom2_idx = kwargs.get('atom2_idx', 1)
        atom3_idx = kwargs.get('atom3_idx', 2)
        return lambda coords, feats, step=0: angle_cv(coords, feats, atom1_idx, atom2_idx, atom3_idx)

    elif cv_type == "dihedral":
        atom1_idx = kwargs.get('atom1_idx', 0)
        atom2_idx = kwargs.get('atom2_idx', 1)
        atom3_idx = kwargs.get('atom3_idx', 2)
        atom4_idx = kwargs.get('atom4_idx', 3)
        return lambda coords, feats, step=0: dihedral_cv(coords, feats, atom1_idx, atom2_idx, atom3_idx, atom4_idx)

    elif cv_type == "angle_enhanced":
        atom1_idx = kwargs.get('atom1_idx', 0)
        atom2_idx = kwargs.get('atom2_idx', 1)
        atom3_idx = kwargs.get('atom3_idx', 2)
        max_hops = kwargs.get('max_hops', 100)  # High default for broad coverage on large proteins
        decay = kwargs.get('decay', 0.8)  # 0.8 gives ~62 hop effective range, 42% coverage on 200 res
        return lambda coords, feats, step=0: angle_enhanced_cv(
            coords, feats, atom1_idx, atom2_idx, atom3_idx, max_hops, decay
        )

    elif cv_type == "dihedral_enhanced":
        atom1_idx = kwargs.get('atom1_idx', 0)
        atom2_idx = kwargs.get('atom2_idx', 1)
        atom3_idx = kwargs.get('atom3_idx', 2)
        atom4_idx = kwargs.get('atom4_idx', 3)
        max_hops = kwargs.get('max_hops', 100)  # High default for broad coverage on large proteins
        decay = kwargs.get('decay', 0.8)  # 0.8 gives ~62 hop effective range, 42% coverage on 200 res
        return lambda coords, feats, step=0: dihedral_enhanced_cv(
            coords, feats, atom1_idx, atom2_idx, atom3_idx, atom4_idx, max_hops, decay
        )

    # Region-based angle/dihedral CVs (using center of mass)
    elif cv_type == "angle_region":
        region1_mask = kwargs.get('region1_mask')
        region2_mask = kwargs.get('region2_mask')
        region3_mask = kwargs.get('region3_mask')
        if region1_mask is None or region2_mask is None or region3_mask is None:
            raise ValueError("region1_mask, region2_mask, region3_mask required for 'angle_region' CV")
        max_hops = kwargs.get('max_hops', 0)  # Disable propagation by default
        decay = kwargs.get('decay', 0.5)
        return lambda coords, feats, step=0: angle_region_cv(
            coords, feats, region1_mask, region2_mask, region3_mask, max_hops, decay
        )

    elif cv_type == "dihedral_region":
        region1_mask = kwargs.get('region1_mask')
        region2_mask = kwargs.get('region2_mask')
        region3_mask = kwargs.get('region3_mask')
        region4_mask = kwargs.get('region4_mask')
        if region1_mask is None or region2_mask is None or region3_mask is None or region4_mask is None:
            raise ValueError("region1_mask, region2_mask, region3_mask, region4_mask required for 'dihedral_region' CV")
        max_hops = kwargs.get('max_hops', 0)  # Disable propagation by default
        decay = kwargs.get('decay', 0.8)
        return lambda coords, feats, step=0: dihedral_region_cv(
            coords, feats, region1_mask, region2_mask, region3_mask, region4_mask, max_hops, decay
        )

    # Contact CVs
    elif cv_type == "hbond_count":
        donor_mask = kwargs.get('donor_mask', None)
        acceptor_mask = kwargs.get('acceptor_mask', None)
        distance_cutoff = kwargs.get('distance_cutoff', 3.5)
        return lambda coords, feats, step=0: hbond_count_cv(coords, feats, donor_mask, acceptor_mask, distance_cutoff)

    elif cv_type == "salt_bridges":
        positive_mask = kwargs.get('positive_mask', None)
        negative_mask = kwargs.get('negative_mask', None)
        distance_cutoff = kwargs.get('distance_cutoff', 4.0)
        return lambda coords, feats, step=0: salt_bridges_cv(coords, feats, positive_mask, negative_mask, distance_cutoff)

    elif cv_type == "contact_order":
        contact_cutoff = kwargs.get('contact_cutoff', 8.0)
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: contact_order_cv(coords, feats, contact_cutoff, atom_mask)

    elif cv_type == "local_contacts":
        contact_cutoff = kwargs.get('contact_cutoff', 8.0)
        sequence_separation = kwargs.get('sequence_separation', 3)
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: local_contacts_cv(coords, feats, contact_cutoff, sequence_separation, atom_mask)

    elif cv_type == "sasa":
        atom_mask = kwargs.get('atom_mask', None)
        probe_radius = kwargs.get('probe_radius', 1.4)
        method = kwargs.get('method', 'lcpo')
        return lambda coords, feats, step=0: sasa_cv(
            coords, feats, atom_mask, probe_radius, method
        )

    # Domain CVs
    elif cv_type == "inter_domain":
        domain1_mask = kwargs.get('domain1_mask')
        if domain1_mask is None:
            domain1_mask = kwargs.get('chain1_mask')
        domain2_mask = kwargs.get('domain2_mask')
        if domain2_mask is None:
            domain2_mask = kwargs.get('chain2_mask')
        if domain1_mask is None or domain2_mask is None:
            raise ValueError("domain1_mask and domain2_mask required for 'inter_domain' CV")
        return lambda coords, feats, step=0: inter_domain_cv(coords, feats, domain1_mask, domain2_mask)

    elif cv_type == "hinge_angle":
        # DEPRECATED: Use angle_region with region1/region2/region3 instead
        import warnings
        warnings.warn(
            "The 'hinge_angle' CV is DEPRECATED. Use 'angle' with region1/region2/region3 instead.",
            DeprecationWarning,
            stacklevel=2
        )
        domain1_mask = kwargs.get('domain1_mask') or kwargs.get('region1_mask')
        hinge_mask = kwargs.get('hinge_mask') or kwargs.get('region2_mask')
        domain2_mask = kwargs.get('domain2_mask') or kwargs.get('region3_mask')
        if domain1_mask is None or hinge_mask is None or domain2_mask is None:
            raise ValueError("hinge_angle requires domain1_mask/hinge_mask/domain2_mask (or use 'angle' with region1/region2/region3)")
        return lambda coords, feats, step=0: hinge_angle_cv(coords, feats, domain1_mask, hinge_mask, domain2_mask)

    # Secondary structure CVs
    elif cv_type == "alpharmsd":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: alpharmsd_cv(coords, feats, atom_mask)

    elif cv_type == "antibetarmsd":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: antibetarmsd_cv(coords, feats, atom_mask)

    elif cv_type == "parabetarmsd":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: parabetarmsd_cv(coords, feats, atom_mask)

    # Shape CVs
    elif cv_type == "acylindricity":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: acylindricity_cv(coords, feats, atom_mask)

    elif cv_type == "shape_gyration":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: shape_gyration_cv(coords, feats, atom_mask)

    # Content CVs
    elif cv_type == "helix_content":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: helix_content_cv(coords, feats, atom_mask)

    elif cv_type == "sheet_content":
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: sheet_content_cv(coords, feats, atom_mask)

    elif cv_type == "dipole_moment":
        charges = kwargs.get('charges', None)
        atom_mask = kwargs.get('atom_mask', None)
        return lambda coords, feats, step=0: dipole_moment_cv(coords, feats, charges, atom_mask)

    else:
        raise ValueError(f"Unknown CV type: {cv_type}. Available CVs: {list(CV_REGISTRY.keys())}")
