"""
Atom selection parser for metadiffusion configuration.

This module parses atom selection strings from YAML configuration
and converts them to atom indices for use in CV computation.

Supported formats:
- "A:15:CA" - chain:residue_id:atom_name
- "A:15" - chain:residue_id (uses CA by default)
- Groups: ["A", "B"] - list of chain IDs
"""

import torch
from typing import Optional, List, Tuple, Dict, Any, Union
import re


def parse_atom_spec(
    spec: str,
    feats: Dict[str, Any],
    default_atom: str = "CA",
) -> int:
    """
    Parse an atom specification string to an atom index.

    Args:
        spec: Atom specification in format "chain:resid:atomname" or "chain:resid"
        feats: Feature dictionary containing atom/residue/chain information
        default_atom: Default atom name if not specified (default: "CA")

    Returns:
        Atom index in the coordinate array

    Raises:
        ValueError: If the specification cannot be parsed or atom not found
    """
    parts = spec.split(":")

    if len(parts) == 2:
        chain_id, resid_str = parts
        atom_name = default_atom
    elif len(parts) == 3:
        chain_id, resid_str, atom_name = parts
    else:
        raise ValueError(f"Invalid atom spec format: {spec}. Expected 'chain:resid' or 'chain:resid:atom'")

    try:
        resid = int(resid_str)
    except ValueError:
        raise ValueError(f"Invalid residue ID in atom spec: {resid_str}")

    # Get chain and residue information from feats
    # The exact structure depends on how Boltz stores this information
    chain_ids = feats.get("chain_id", None)
    res_ids = feats.get("res_id", None)
    atom_names = feats.get("atom_name", None)

    if chain_ids is None or res_ids is None:
        raise ValueError("Feature dict missing chain_id or res_id information")

    # Convert to numpy/list for easier searching
    if torch.is_tensor(chain_ids):
        chain_ids = chain_ids.cpu().numpy()
    if torch.is_tensor(res_ids):
        res_ids = res_ids.cpu().numpy()

    # Find matching atom
    for idx in range(len(chain_ids)):
        if str(chain_ids[idx]) == chain_id and int(res_ids[idx]) == resid:
            # If we have atom names, check that too
            if atom_names is not None:
                if torch.is_tensor(atom_names):
                    atom_names_np = atom_names.cpu().numpy()
                else:
                    atom_names_np = atom_names

                # Check if this atom matches
                current_atom = atom_names_np[idx] if idx < len(atom_names_np) else None
                if current_atom is not None:
                    # Handle different atom name formats
                    current_atom_str = str(current_atom).strip()
                    if current_atom_str == atom_name:
                        return idx
            else:
                # No atom names available, return first match
                return idx

    raise ValueError(f"Atom not found: {spec}")


def parse_atom_spec_simple(
    spec: str,
    n_atoms: int,
    chain_to_atom_range: Dict[str, Tuple[int, int]],
    resid_to_atom_idx: Optional[Dict[Tuple[str, int], List[int]]] = None,
) -> int:
    """
    Simplified atom spec parser using chain-to-atom mapping.

    Args:
        spec: Atom specification string
        n_atoms: Total number of atoms
        chain_to_atom_range: Dict mapping chain ID to (start_idx, end_idx)
        resid_to_atom_idx: Optional dict mapping (chain_id, resid) to atom indices

    Returns:
        Atom index
    """
    parts = spec.split(":")

    if len(parts) < 2:
        raise ValueError(f"Invalid atom spec: {spec}")

    chain_id = parts[0]
    resid = int(parts[1])

    if chain_id not in chain_to_atom_range:
        raise ValueError(f"Unknown chain ID: {chain_id}")

    start_idx, end_idx = chain_to_atom_range[chain_id]

    if resid_to_atom_idx and (chain_id, resid) in resid_to_atom_idx:
        atom_indices = resid_to_atom_idx[(chain_id, resid)]
        if len(parts) == 3:
            # Specific atom requested - would need atom name mapping
            # For now, return first atom of residue (usually N)
            return atom_indices[0]
        else:
            # Return CA-like atom (middle of residue atoms)
            return atom_indices[len(atom_indices) // 2]

    # Fallback: estimate based on residue position
    # Assume ~average atoms per residue
    atoms_in_chain = end_idx - start_idx
    # This is a rough estimate - real implementation needs residue info
    atom_idx = start_idx + resid  # Simplified
    return min(max(atom_idx, start_idx), end_idx - 1)


def parse_group_selection(
    groups: List[str],
    feats: Dict[str, Any],
) -> torch.Tensor:
    """
    Parse chain group selection to an atom mask.

    Args:
        groups: List of chain IDs (e.g., ["A", "B"])
        feats: Feature dictionary

    Returns:
        Boolean mask tensor [N_atoms] for selected atoms
    """
    chain_ids = feats.get("chain_id", None)
    if chain_ids is None:
        raise ValueError("Feature dict missing chain_id information")

    if torch.is_tensor(chain_ids):
        chain_ids_np = chain_ids.cpu().numpy()
    else:
        chain_ids_np = chain_ids

    n_atoms = len(chain_ids_np)
    mask = torch.zeros(n_atoms, dtype=torch.bool)

    for idx, chain_id in enumerate(chain_ids_np):
        if str(chain_id) in groups:
            mask[idx] = True

    return mask


def parse_group_selection_simple(
    groups: List[str],
    n_atoms: int,
    chain_to_atom_range: Dict[str, Tuple[int, int]],
    feats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Group selection using chain-to-atom mapping with full region spec support.

    Supported formats:
    - "A" - select all atoms in chain A
    - "A:1-20" - select atoms in residues 1-20 of chain A
    - "A:5:CA" - select single atom (chain:resid:atomname)
    - "A:1-50:CA" - select specific atom type across residue range
    - "A::CA" - select specific atom type across whole chain

    Args:
        groups: List of region specifications
        n_atoms: Total number of atoms
        chain_to_atom_range: Dict mapping chain ID to (start_idx, end_idx)
        feats: Optional feature dict for residue-based selection

    Returns:
        Boolean mask tensor [N_atoms]
    """
    mask = torch.zeros(n_atoms, dtype=torch.bool)

    # Standard protein backbone atom order within a residue
    backbone_order = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2',
                      'CE', 'CE1', 'CE2', 'NZ', 'OG', 'OG1', 'SD', 'NE', 'NE1', 'NE2',
                      'OH', 'OD1', 'OD2', 'ND1', 'ND2', 'OE1', 'OE2', 'NH1', 'NH2', 'SG']

    for group_spec in groups:
        parts = group_spec.split(':')

        if len(parts) == 1:
            # Format: "A" - whole chain
            chain_id = parts[0]
            if chain_id in chain_to_atom_range:
                start_idx, end_idx = chain_to_atom_range[chain_id]
                mask[start_idx:end_idx] = True

        elif len(parts) == 2:
            # Format: "A:1-20" (residue range) or "A:5" (single residue)
            chain_id = parts[0]
            residue_spec = parts[1]

            if chain_id not in chain_to_atom_range:
                continue

            chain_start, chain_end = chain_to_atom_range[chain_id]

            # Parse residue range
            if "-" in residue_spec:
                try:
                    start_res, end_res = map(int, residue_spec.split('-'))
                except ValueError:
                    continue
            else:
                try:
                    start_res = int(residue_spec)
                    end_res = start_res
                except ValueError:
                    continue

            # Use residue information if available
            if feats is not None:
                res_idx_tensor = feats.get('res_idx', None)
                if res_idx_tensor is not None:
                    if torch.is_tensor(res_idx_tensor):
                        res_idx_array = res_idx_tensor.cpu().numpy()
                    else:
                        res_idx_array = list(res_idx_tensor)

                    # res_idx is 0-indexed, user input is 1-indexed
                    start_res_0 = start_res - 1
                    end_res_0 = end_res - 1

                    for idx in range(chain_start, chain_end):
                        if idx < len(res_idx_array):
                            if start_res_0 <= res_idx_array[idx] <= end_res_0:
                                mask[idx] = True
                    continue

                # Try alternative field names
                res_ids = feats.get("res_id", feats.get("residue_index", None))
                chain_ids = feats.get("chain_id", feats.get("asym_id", None))

                atom_to_token = feats.get("atom_to_token", None)
                if atom_to_token is not None and res_ids is not None and chain_ids is not None:
                    if torch.is_tensor(atom_to_token):
                        if atom_to_token.dim() == 3:
                            atom_to_token = atom_to_token.squeeze(0).argmax(dim=-1)
                        elif atom_to_token.dim() == 2:
                            if atom_to_token.shape[0] == 1:
                                atom_to_token = atom_to_token.squeeze(0)
                            elif atom_to_token.shape[-1] > atom_to_token.shape[0]:
                                atom_to_token = atom_to_token.argmax(dim=-1)
                        atom_to_token_np = atom_to_token.cpu().numpy()
                    else:
                        atom_to_token_np = list(atom_to_token)

                    if torch.is_tensor(res_ids):
                        if res_ids.dim() > 1:
                            res_ids = res_ids.squeeze(0)
                        res_ids_np = res_ids.cpu().numpy()
                    else:
                        res_ids_np = list(res_ids)
                    if torch.is_tensor(chain_ids):
                        if chain_ids.dim() > 1:
                            chain_ids = chain_ids.squeeze(0)
                        chain_ids_np = chain_ids.cpu().numpy()
                    else:
                        chain_ids_np = list(chain_ids)

                    def to_chain_letter(cid):
                        try:
                            cid_int = int(cid)
                            if 0 <= cid_int < 26:
                                return chr(ord('A') + cid_int)
                        except (ValueError, TypeError):
                            pass
                        return str(cid)

                    for atom_idx in range(n_atoms):
                        if atom_idx >= len(atom_to_token_np):
                            continue
                        tok_idx = atom_to_token_np[atom_idx]
                        if tok_idx >= len(res_ids_np) or tok_idx >= len(chain_ids_np):
                            continue

                        atom_chain = to_chain_letter(chain_ids_np[tok_idx])
                        if atom_chain == chain_id:
                            res_id = int(res_ids_np[tok_idx]) + 1
                            if start_res <= res_id <= end_res:
                                mask[atom_idx] = True
                    continue

                elif res_ids is not None and chain_ids is not None:
                    if torch.is_tensor(res_ids):
                        res_ids_np = res_ids.cpu().numpy()
                    else:
                        res_ids_np = res_ids
                    if torch.is_tensor(chain_ids):
                        chain_ids_np = chain_ids.cpu().numpy()
                    else:
                        chain_ids_np = chain_ids

                    for idx in range(n_atoms):
                        if idx >= len(res_ids_np) or idx >= len(chain_ids_np):
                            continue
                        if str(chain_ids_np[idx]) == chain_id:
                            res_id = int(res_ids_np[idx])
                            if start_res <= res_id <= end_res:
                                mask[idx] = True
                    continue

            # Fallback: estimate based on position in chain
            atoms_in_chain = chain_end - chain_start
            atoms_per_residue = 8

            rel_start = max(0, (start_res - 1) * atoms_per_residue)
            rel_end = min(atoms_in_chain, end_res * atoms_per_residue)

            abs_start = chain_start + rel_start
            abs_end = chain_start + rel_end

            mask[abs_start:abs_end] = True

        elif len(parts) == 3:
            # Format: "A:5:CA" (single atom), "A:1-50:CA" (range), or "A::CA" (whole chain)
            chain_id = parts[0]
            residue_spec = parts[1]
            atom_name = parts[2]

            if chain_id not in chain_to_atom_range:
                continue

            chain_start, chain_end = chain_to_atom_range[chain_id]

            # Determine atom position in standard order
            atom_pos = backbone_order.index(atom_name) if atom_name in backbone_order else None

            if residue_spec == "":
                # Format: "A::CA" - atom type across whole chain
                if feats is not None:
                    res_idx_tensor = feats.get('res_idx', None)
                    if res_idx_tensor is not None:
                        if torch.is_tensor(res_idx_tensor):
                            res_idx_array = res_idx_tensor.cpu().numpy()
                        else:
                            res_idx_array = list(res_idx_tensor)

                        # Track atoms within each residue to find the target atom
                        current_res = None
                        atoms_in_res = []

                        for idx in range(chain_start, chain_end):
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

            elif "-" in residue_spec:
                # Format: "A:1-50:CA" - atom type across residue range
                try:
                    start_res, end_res = map(int, residue_spec.split('-'))
                except ValueError:
                    continue

                start_res_0 = start_res - 1
                end_res_0 = end_res - 1

                if feats is not None:
                    res_idx_tensor = feats.get('res_idx', None)
                    if res_idx_tensor is not None:
                        if torch.is_tensor(res_idx_tensor):
                            res_idx_array = res_idx_tensor.cpu().numpy()
                        else:
                            res_idx_array = list(res_idx_tensor)

                        # Track atoms within each residue to find the target atom
                        current_res = None
                        atoms_in_res = []

                        for idx in range(chain_start, chain_end):
                            if idx >= len(res_idx_array):
                                break
                            res = res_idx_array[idx]

                            # New residue - reset counter
                            if res != current_res:
                                current_res = res
                                atoms_in_res = []

                            if start_res_0 <= res <= end_res_0:
                                atoms_in_res.append(idx)
                                # Check if this is the target atom
                                if atom_pos is not None and len(atoms_in_res) - 1 == atom_pos:
                                    mask[idx] = True
                                elif atom_pos is None and len(atoms_in_res) == 1:
                                    # Unknown atom name, use first atom as fallback
                                    mask[idx] = True
            else:
                # Format: "A:5:CA" - single atom
                try:
                    resid = int(residue_spec)
                except ValueError:
                    continue

                resid_0 = resid - 1

                if feats is not None:
                    res_idx_tensor = feats.get('res_idx', None)
                    if res_idx_tensor is not None:
                        if torch.is_tensor(res_idx_tensor):
                            res_idx_array = res_idx_tensor.cpu().numpy()
                        else:
                            res_idx_array = list(res_idx_tensor)

                        # Find atoms in this residue
                        residue_atoms = []
                        for idx in range(chain_start, chain_end):
                            if idx < len(res_idx_array) and res_idx_array[idx] == resid_0:
                                residue_atoms.append(idx)

                        if residue_atoms:
                            if atom_pos is not None and atom_pos < len(residue_atoms):
                                mask[residue_atoms[atom_pos]] = True
                            elif residue_atoms:
                                # Fallback to first atom
                                mask[residue_atoms[0]] = True

    return mask


def get_chain_com_indices(
    chain_id: str,
    feats: Dict[str, Any],
    atom_type: str = "CA",
) -> torch.Tensor:
    """
    Get atom indices for computing center of mass of a chain.

    Args:
        chain_id: Chain identifier
        feats: Feature dictionary
        atom_type: Type of atoms to use (default: "CA" for alpha carbons)

    Returns:
        Tensor of atom indices for the chain
    """
    chain_ids = feats.get("chain_id", None)
    atom_names = feats.get("atom_name", None)

    if chain_ids is None:
        raise ValueError("Feature dict missing chain_id information")

    if torch.is_tensor(chain_ids):
        chain_ids_np = chain_ids.cpu().numpy()
    else:
        chain_ids_np = chain_ids

    indices = []
    for idx, cid in enumerate(chain_ids_np):
        if str(cid) == chain_id:
            if atom_names is not None:
                # Filter by atom type if available
                if torch.is_tensor(atom_names):
                    atom_name = atom_names[idx].cpu().numpy() if idx < len(atom_names) else None
                else:
                    atom_name = atom_names[idx] if idx < len(atom_names) else None

                if atom_name is not None and str(atom_name).strip() == atom_type:
                    indices.append(idx)
            else:
                # No atom name filtering
                indices.append(idx)

    return torch.tensor(indices, dtype=torch.long)


def get_terminal_atom_indices(
    chain_id: str,
    feats: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Get N-terminal and C-terminal atom indices for a chain.

    Args:
        chain_id: Chain identifier
        feats: Feature dictionary

    Returns:
        Tuple of (n_term_idx, c_term_idx)
    """
    chain_ids = feats.get("chain_id", None)
    res_ids = feats.get("res_id", None)

    if chain_ids is None or res_ids is None:
        raise ValueError("Feature dict missing chain_id or res_id information")

    if torch.is_tensor(chain_ids):
        chain_ids_np = chain_ids.cpu().numpy()
    else:
        chain_ids_np = chain_ids

    if torch.is_tensor(res_ids):
        res_ids_np = res_ids.cpu().numpy()
    else:
        res_ids_np = res_ids

    # Find atoms belonging to this chain
    chain_atom_indices = []
    for idx, cid in enumerate(chain_ids_np):
        if str(cid) == chain_id:
            chain_atom_indices.append((idx, res_ids_np[idx]))

    if not chain_atom_indices:
        raise ValueError(f"No atoms found for chain: {chain_id}")

    # Sort by residue ID
    chain_atom_indices.sort(key=lambda x: x[1])

    # Get first and last atom
    n_term_idx = chain_atom_indices[0][0]
    c_term_idx = chain_atom_indices[-1][0]

    return n_term_idx, c_term_idx


def build_chain_to_atom_mapping(
    feats: Dict[str, Any],
) -> Dict[str, Tuple[int, int]]:
    """
    Build a mapping from chain IDs to atom index ranges.

    Args:
        feats: Feature dictionary (supports both direct chain_id and boltz asym_id formats)

    Returns:
        Dict mapping chain ID to (start_idx, end_idx)
    """
    # Try different field names for chain IDs
    chain_ids = feats.get("chain_id", None)
    if chain_ids is None:
        # Boltz uses asym_id at token level, need to expand to atom level
        asym_id = feats.get("asym_id", None)
        if asym_id is not None:
            # Get atom_to_token mapping to expand token-level to atom-level
            atom_to_token = feats.get("atom_to_token", None)
            if atom_to_token is not None:
                if torch.is_tensor(atom_to_token):
                    # Handle batch dimension
                    if atom_to_token.dim() == 3:
                        # [batch, n_atoms, n_tokens] - take argmax to get token index per atom
                        atom_to_token = atom_to_token.squeeze(0).argmax(dim=-1)
                    elif atom_to_token.dim() == 2:
                        # Could be [batch, n_atoms] or [n_atoms, n_tokens]
                        if atom_to_token.shape[0] == 1:
                            atom_to_token = atom_to_token.squeeze(0)
                        elif atom_to_token.shape[-1] > atom_to_token.shape[0]:
                            # [n_atoms, n_tokens] - take argmax
                            atom_to_token = atom_to_token.argmax(dim=-1)
                    atom_to_token_np = atom_to_token.cpu().numpy()
                else:
                    atom_to_token_np = list(atom_to_token)

                if torch.is_tensor(asym_id):
                    # Handle batch dimension: squeeze if present
                    if asym_id.dim() > 1:
                        asym_id = asym_id.squeeze(0)
                    asym_id_np = asym_id.cpu().numpy()
                else:
                    asym_id_np = list(asym_id)

                # Expand asym_id to atom level
                chain_ids = [asym_id_np[tok_idx] for tok_idx in atom_to_token_np]
            else:
                # Use asym_id directly (might be at token level)
                chain_ids = asym_id

    if chain_ids is None:
        return {}

    if torch.is_tensor(chain_ids):
        chain_ids_np = chain_ids.cpu().numpy()
    else:
        chain_ids_np = list(chain_ids)

    # Map numeric asym_id to chain letters (0->A, 1->B, etc.)
    def to_chain_letter(cid):
        # Handle various numeric types (int, float, numpy, torch)
        try:
            cid_int = int(cid)
            if 0 <= cid_int < 26:
                return chr(ord('A') + cid_int)
        except (ValueError, TypeError):
            pass
        return str(cid)

    mapping = {}
    current_chain = None
    start_idx = 0

    for idx, cid in enumerate(chain_ids_np):
        cid_str = to_chain_letter(cid)
        if cid_str != current_chain:
            if current_chain is not None:
                # Only add if this chain hasn't been seen before
                # This handles cases where chain A atoms appear in multiple places
                if current_chain not in mapping:
                    mapping[current_chain] = (start_idx, idx)
            current_chain = cid_str
            start_idx = idx

    # Add last chain (only if not already in mapping)
    if current_chain is not None and current_chain not in mapping:
        mapping[current_chain] = (start_idx, len(chain_ids_np))

    return mapping


def validate_atom_spec(spec: str) -> bool:
    """
    Validate an atom specification string format.

    Args:
        spec: Atom specification string

    Returns:
        True if valid format, False otherwise
    """
    pattern = r'^[A-Za-z0-9]+:\d+(?::[A-Za-z0-9]+)?$'
    return bool(re.match(pattern, spec))


def validate_group_list(groups: List[str]) -> bool:
    """
    Validate a list of chain IDs.

    Args:
        groups: List of chain ID strings

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(groups, list):
        return False

    pattern = r'^[A-Za-z0-9]+$'
    return all(isinstance(g, str) and re.match(pattern, g) for g in groups)
