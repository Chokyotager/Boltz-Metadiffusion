from pathlib import Path
from typing import Optional, Tuple

import yaml
from rdkit.Chem.rdchem import Mol

from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Target


def parse_yaml(
    path: Path,
    ccd: dict[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a yaml file with the following format:

    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
    templates:
        - path: /path/to/template.pdb
          ids: [A] # optional, specify which chains to template

    version: 1

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.
    boltz2 : bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Target
        The parsed target.

    """
    with path.open("r") as file:
        data = yaml.safe_load(file)

    name = path.stem
    return parse_boltz_schema(name, data, ccd, mol_dir, boltz2)


def parse_yaml_with_metadiffusion(
    path: Path,
    ccd: dict[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
) -> Tuple[Target, Optional["MetadiffusionConfig"]]:
    """Parse a Boltz input yaml / json with metadiffusion configuration.

    This function extends parse_yaml to also extract the metadiffusion
    configuration section from the YAML file.

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    ccd : dict[str, Mol]
        Dictionary of CCD components.
    mol_dir : Path
        Path to the molecule directory.
    boltz2 : bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Tuple[Target, Optional[MetadiffusionConfig]]
        The parsed target and metadiffusion configuration (if present).
    """
    from boltz.data.parse.metadiffusion import (
        MetadiffusionConfig,
        parse_metadiffusion_section,
    )

    with path.open("r") as file:
        data = yaml.safe_load(file)

    # Validate early: metadiffusion features are only compatible with Boltz2
    if not boltz2:
        boltz1_incompatible = []
        if "metadiffusion" in data and data["metadiffusion"]:
            boltz1_incompatible.append("metadiffusion")
        if "noise_scale" in data:
            boltz1_incompatible.append("noise_scale")
        if "denoise_clip" in data:
            boltz1_incompatible.append("denoise_clip")

        if boltz1_incompatible:
            features = ", ".join(boltz1_incompatible)
            msg = (
                f"The following features are only compatible with Boltz2, not Boltz1: {features}. "
                f"Either remove these from your YAML file or use --model boltz2."
            )
            raise ValueError(msg)

    name = path.stem
    target = parse_boltz_schema(name, data, ccd, mol_dir, boltz2)

    # Parse metadiffusion section if present
    metadiffusion_config = parse_metadiffusion_section(data)

    return target, metadiffusion_config


def parse_metadiffusion_from_yaml(path: Path) -> Optional["MetadiffusionConfig"]:
    """Parse only the metadiffusion section from a YAML file.

    This is a lightweight function for extracting metadiffusion configuration
    without parsing the full target structure.

    Parameters
    ----------
    path : Path
        Path to the YAML input format.

    Returns
    -------
    Optional[MetadiffusionConfig]
        The parsed metadiffusion configuration, or None if not present.
    """
    from boltz.data.parse.metadiffusion import parse_metadiffusion_section

    with path.open("r") as file:
        data = yaml.safe_load(file)

    return parse_metadiffusion_section(data)
