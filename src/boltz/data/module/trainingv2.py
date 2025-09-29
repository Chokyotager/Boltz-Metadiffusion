import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from rdkit.Chem import Mol
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.filter.dynamic.filter import DynamicFilter
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.pad import pad_to_max
from boltz.data.sample.v2.sampler import Sample, Sampler
from boltz.data.template.feature import (
    compute_template_features,
    load_dummy_templates_v2,
)
from boltz.data.tokenize.tokenizer import Tokenizer
from boltz.data.types import (
    MSA,
    InputTraining,
    Manifest,
    Record,
    StructureV2,
    Template,
    TokenizedTraining,
)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    msa_dir: str
    prob: Optional[float]
    sampler: Sampler
    cropper: Cropper
    template_dir: Optional[str] = None
    filters: Optional[list[DynamicFilter]] = None
    split: Optional[str] = None
    has_structure_label: bool = True
    has_affinity_label: bool = False
    symmetry_correction: bool = True
    val_group: Optional[str] = "RCSB"
    use_train_subset: Optional[float] = None
    moldir: Optional[str] = None
    override_bfactor: Optional[bool] = False
    override_method: Optional[str] = None


@dataclass
class DataConfigV2:
    """Data configuration."""

    datasets: list[DatasetConfig]
    featurizer: BoltzFeaturizer
    tokenizer: Tokenizer
    max_atoms: int
    max_tokens: int
    max_seqs: int
    samples_per_epoch: int
    batch_size: int
    num_workers: int
    random_seed: int
    pin_memory: bool
    atoms_per_window_queries: int
    min_dist: float
    max_dist: float
    num_bins: int
    checkpoint_monitor_val_group: str
    num_ensembles_train: int = 1
    num_ensembles_val: int = 1
    disto_use_ensemble: Optional[bool] = False
    fix_single_ensemble: Optional[bool] = True
    overfit: Optional[int] = None
    pad_to_max_tokens: bool = False
    pad_to_max_atoms: bool = False
    pad_to_max_seqs: bool = False
    return_train_symmetries: bool = False
    return_val_symmetries: bool = True
    train_binder_pocket_conditioned_prop: float = 0.0
    val_binder_pocket_conditioned_prop: float = 0.0
    train_contact_conditioned_prop: float = 0.0
    val_contact_conditioned_prop: float = 0.0
    binder_pocket_cutoff_min: float = 4.0
    binder_pocket_cutoff_max: float = 20.0
    binder_pocket_cutoff_val: float = 6.0
    binder_pocket_sampling_geometric_p: float = 0.0
    val_batch_size: int = 1
    return_train_affinity: bool = False
    return_val_affinity: bool = False
    single_sequence_prop_training: float = 0.0
    msa_sampling_training: bool = False
    use_templates: bool = False
    use_templates_v2: bool = False
    max_templates_train: int = 4
    max_templates_val: int = 4
    no_template_prob_train: float = 1.0
    no_template_prob_val: float = 1.0
    moldir: Optional[str] = None
    compute_frames: bool = False
    bfactor_md_correction: Optional[bool] = False


@dataclass
class Dataset:
    """Data holder."""

    samples: pd.DataFrame
    struct_dir: Path
    msa_dir: Path
    record_dir: Path
    template_dir: Path
    prob: float
    cropper: Cropper
    tokenizer: Tokenizer
    featurizer: BoltzFeaturizer
    val_group: str
    has_structure_label: bool = True
    has_affinity_label: bool = False
    symmetry_correction: bool = True
    moldir: Optional[str] = None
    override_bfactor: Optional[bool] = False
    override_method: Optional[str] = None


def load_record(record_id: str, record_dir: Path) -> Record:
    """Load the given record.

    Parameters
    ----------
    record_id : str
        The record id to load.
    record_dir : Path
        The path to the record directory.

    Returns
    -------
    Record
        The loaded record.
    """
    return Record.load(record_dir / f"{record_id}.json")


def load_structure(record: Record, struct_dir: Path) -> StructureV2:
    """Load the given input data.

    Parameters
    ----------
    record : str
        The record to load.
    target_dir : Path
        The path to the data directory.

    Returns
    -------
    InputTraining
        The loaded input.

    """
    if (struct_dir / f"{record.id}.npz").exists():
        structure_path = struct_dir / f"{record.id}.npz"
    else:
        structure_path = struct_dir / f"{record.id}" / f"{record.id}_model_0.npz"
    return StructureV2.load(structure_path)


def load_msas(chain_ids: set[int], record: Record, msa_dir: Path) -> InputTraining:
    """Load the given input data.

    Parameters
    ----------
    chain_ids : set[int]
        The chain ids to load.
    record : Record
        The record to load.
    msa_dir : Path
        The path to the MSA directory.

    Returns
    -------
    InputTraining
        The loaded input.

    """
    msas = {}
    for chain in record.chains:
        if chain.chain_id not in chain_ids:
            continue

        msa_id = chain.msa_id
        if msa_id != -1:
            msa_path = msa_dir / f"{msa_id}.npz"
            msa = MSA.load(msa_path)
            msas[chain.chain_id] = msa

    return msas


def load_templates(
    chain_ids: set[int],
    record: Record,
    template_dir: Path,
    max_templates: int,
    no_template_prob: float,
    training: bool,
    random: np.random.Generator,
) -> dict[str, list[Template]]:
    """Load the given input data.

    Parameters
    ----------
    record : str
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to the MSA directory.
    template_dir : Path
        The path to the template directory.
    max_templates : int
        The maximum number of templates to load.
    no_template_prob : float
        The probability of not loading any templates.
    training : bool
        Whether the data is for training.
    random : np.random.Generator
        The random number generator.

    Returns
    -------
    dict[str, list[Template]]
        The loaded templates.

    """
    templates = {}
    for chain in record.chains:
        if chain.chain_id not in chain_ids:
            continue

        # Check if chain has templates, skipping non proteins
        template_ids = chain.template_ids
        if template_ids is None:
            continue

        # Pick how many templates to sample
        max_chain_templates = min(max_templates, len(template_ids))

        # If 0, skips
        if (max_chain_templates == 0) or (random.random() < no_template_prob):
            continue

        # Sample for training, pick firsts for validation
        if training:
            max_chain_templates = random.integers(1, max_chain_templates + 1)
            template_indices = torch.randperm(len(template_ids))
            template_indices = template_indices[:max_chain_templates]
            template_ids = [template_ids[idx.item()] for idx in template_indices]
        else:
            template_ids = template_ids[:max_chain_templates]

        # Load templates
        templates[chain.chain_id] = []
        for template_name in template_ids:
            template_path = template_dir / f"{template_name}.npz"
            template = Template.load(template_path)
            templates[chain.chain_id].append(template)

    return templates


def load_templates_v2(
    tokenized: TokenizedTraining,
    record: Record,
    template_dir: Path,
    structure_dir: Path,
    tokenizer: Tokenizer,
    max_templates: int,
    no_template_prob: float,
    training: bool,
    random: np.random.Generator,
    max_tokens: int,
) -> dict[str, list[Template]]:
    """Load the given input data.

    Parameters
    ----------
    entities : dict[int, list[int]]
        The entities to load.
    record : str
        The record to load.
    template_dir : Path
        The path to the template directory.
    max_templates : int
        The maximum number of templates to load.
    no_template_prob : float
        The probability of not loading any templates.
    training : bool
        Whether the data is for training.
    random : np.random.Generator
        The random number generator.

    Returns
    -------
    dict[str, list[Template]]
        The loaded templates.

    """
    # Check if template record exists
    template_record_path = template_dir / f"{record.id}.json"
    if not template_record_path.exists():
        return load_dummy_templates_v2(tdim=max_templates, num_tokens=max_tokens)

    # Load template record
    with Path(template_record_path).open() as f:
        record = json.load(f)
        record = {k.split("_")[1]: v for k, v in record.items()}

    # Get prot moltype
    prot_moltype = const.chain_type_ids["PROTEIN"]

    # Extract entity to cropped indices and entity to chains mappings
    entity_to_crop = {}
    entity_to_chains = {}
    for t in tokenized.tokens:
        if (t["mol_type"] != prot_moltype) or (str(t["entity_id"]) not in record):
            continue
        entity_to_crop.setdefault(str(t["entity_id"]), set()).add(int(t["res_idx"]))
        entity_to_chains.setdefault(str(t["entity_id"]), set()).add(int(t["asym_id"]))

    # Select the number of templates to sample per query entity
    entity_counter = {}
    for entity in entity_to_chains:
        if (str(entity) not in record) or (random.random() < no_template_prob):
            entity_counter[entity] = 0
            continue
        entity_counter[entity] = random.integers(1, max_templates + 1)

    # Select templates per entity and group them by PDB ID
    pdb_id_to_templates = {}
    entity_to_pdb_ids = {}

    for entity in entity_counter:
        for tmpl in record[str(entity)]:
            # Skip templates with no overlap to the query crop
            query_range = set(range(tmpl["query_st"], tmpl["query_en"]))
            if not (query_range & entity_to_crop[entity]):
                continue

            # Add to the PDB ID to entities mapping
            pdb_id = tmpl["template_pdb"]
            entity_to_pdb_ids.setdefault(entity, set()).add(pdb_id)
            pdb_id_to_templates.setdefault(pdb_id, []).append((entity, tmpl))

    # Check if query is monomeric
    chains = tokenized.structure.chains
    protein_chains = chains[chains["mol_type"] == prot_moltype]
    is_query_monomeric = (protein_chains["sym_id"] == 0).sum()

    # Create "rows" of template features
    visited_pdbs = set()
    template_features = []

    for _ in range(max_templates):
        # Get entities to fill with a template
        entities_to_fill = {e for e in entity_to_chains if entity_counter[e] > 0}
        for entity in entities_to_fill:
            entity_counter[entity] -= 1

        # Fill in the row for each entity, here again group by PDB ID,
        # so mapping is from PDB ID to entities that will be templated
        row_pdbs = {}
        while entities_to_fill:
            # Sample an entity and its PDB ID
            entity = random.choice(list(entities_to_fill))
            entities_to_fill.remove(entity)

            pdbs = entity_to_pdb_ids.get(entity, set()) - visited_pdbs
            if not pdbs:
                continue

            pdb_id = random.choice(list(pdbs))
            visited_pdbs.add(pdb_id)
            row_pdbs.setdefault(pdb_id, set()).add(entity)

            # Expand this PDB to other query entities it contains
            pdb_entities = {x[0] for x in pdb_id_to_templates[pdb_id]}
            overlapping_entities = pdb_entities & entities_to_fill
            row_pdbs[pdb_id].update(overlapping_entities)
            entities_to_fill -= overlapping_entities

        # Prepare the template row
        row_tokens = []

        # Load, tokenize, crop and featurize structures
        for idx, (pdb_id, entities) in enumerate(row_pdbs.items()):
            # Load the template structure
            structure_path = structure_dir / f"{pdb_id}.npz"
            if not structure_path.exists():
                continue
            structure: StructureV2 = StructureV2.load(structure_path)
            tmpl_tokenized = tokenizer.tokenize(structure)

            # Map template entities to template chain ids
            tmpl_entity_to_chains = {}
            for c in structure.chains:
                tmpl_entity_to_chains.setdefault(str(c["entity_id"]), []).append(
                    c["asym_id"]
                )

            # Now map query chains to template chains:
            # If symmetrical, for now only map one template chain
            # to all the query chains and mask them from one another
            tmpl_prot_chains = structure.chains[
                structure.chains["mol_type"] == prot_moltype
            ]
            is_tmpl_monomeric = (tmpl_prot_chains["sym_id"] == 0).sum()
            is_monomeric = is_tmpl_monomeric and is_query_monomeric

            for entity, tmpl in pdb_id_to_templates[pdb_id]:
                # Skip if entity is not in the row
                if entity not in entities:
                    continue

                # Compute the offset
                offset = tmpl["template_st"] - tmpl["query_st"]

                # Get the relevant template and query chains
                tmpl_entity = tmpl["template_id"].split("_")[1].split("=")[1]
                tmpl_chains = tmpl_entity_to_chains[tmpl_entity]
                query_chains = entity_to_chains[entity]

                # TODO: try to do a clever assignment here
                for query_chain in query_chains:
                    # Use the first template chain for now
                    tmpl_chain = tmpl_chains[0]

                    # Get query and template tokens to map residues
                    query_tokens = tokenized.tokens
                    q_tokens = query_tokens[query_tokens["asym_id"] == query_chain]
                    q_indices = dict(zip(q_tokens["res_idx"], q_tokens["token_idx"]))

                    # Get the template tokens at the query residues
                    tmpl_tokens = tmpl_tokenized.tokens
                    toks = tmpl_tokens[tmpl_tokens["asym_id"] == tmpl_chain]
                    toks = [t for t in toks if t["res_idx"] - offset in q_indices]
                    for t in toks:
                        q_idx = q_indices[t["res_idx"] - offset]
                        row_tokens.append(
                            {
                                "token": t,
                                "pdb_id": idx,
                                "q_idx": q_idx,
                                "is_monomeric": is_monomeric,
                            }
                        )

        # Compute template features for each row
        row_features = compute_template_features(tokenized, row_tokens, max_tokens)
        template_features.append(row_features)

    # Stack each feature
    out = {}
    for k in template_features[0]:
        out[k] = torch.stack([f[k] for f in template_features])

    return out


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "chain_swaps",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "activity_name",
            "activity_qualifier",
            "sid",
            "cid",
            "normalized_protein_accession",
            "pair_id",
            "ligand_edge_index",
            "ligand_edge_lower_bounds",
            "ligand_edge_upper_bounds",
            "ligand_edge_bond_mask",
            "ligand_edge_angle_mask",
            "connections_edge_index",
            "ligand_chiral_atom_index",
            "ligand_chiral_check_mask",
            "ligand_chiral_atom_orientations",
            "ligand_stereo_bond_index",
            "ligand_stereo_check_mask",
            "ligand_stereo_bond_orientations",
            "ligand_aromatic_5_ring_index",
            "ligand_aromatic_6_ring_index",
            "ligand_planar_double_bond_index",
            "pdb_id",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class TrainingDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        canonicals: dict[str, Mol],
        moldir: str,
        samples_per_epoch: int,
        max_atoms: int,
        max_tokens: int,
        max_seqs: int,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        num_ensembles: int = 1,
        ensemble_sample_replacement: Optional[bool] = True,
        disto_use_ensemble: Optional[bool] = False,
        fix_single_ensemble: Optional[bool] = True,
        overfit: Optional[int] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        contact_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff_min: Optional[float] = 4.0,
        binder_pocket_cutoff_max: Optional[float] = 20.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        return_symmetries: Optional[bool] = False,
        return_affinity: bool = False,
        use_templates: bool = False,
        use_templates_v2: bool = False,
        max_templates: int = 4,
        no_template_prob: float = 0.6,
        single_sequence_prop: Optional[float] = 0.0,
        msa_sampling: bool = False,
        compute_frames: bool = False,
        bfactor_md_correction: bool = False,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        datasets : List[Dataset]
            The datasets to sample from.
        samplers : List[Sampler]
            The samplers to sample from each dataset.
        probs : List[float]
            The probabilities to sample from each dataset.
        samples_per_epoch : int
            The number of samples per epoch.
        max_tokens : int
            The maximum number of tokens.

        """
        super().__init__()

        self.datasets = datasets
        self.canonicals = canonicals
        self.moldir = moldir
        self.probs = [d.prob for d in datasets]
        self.samples_per_epoch = samples_per_epoch
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.num_ensembles = num_ensembles
        self.ensemble_sample_replacement = ensemble_sample_replacement
        self.disto_use_ensemble = disto_use_ensemble
        self.fix_single_ensemble = fix_single_ensemble
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.contact_conditioned_prop = contact_conditioned_prop
        self.binder_pocket_cutoff_min = binder_pocket_cutoff_min
        self.binder_pocket_cutoff_max = binder_pocket_cutoff_max
        self.binder_pocket_sampling_geometric_p = binder_pocket_sampling_geometric_p
        self.return_symmetries = return_symmetries
        self.return_affinity = return_affinity
        self.single_sequence_prop = single_sequence_prop
        self.msa_sampling = msa_sampling
        self.use_templates = use_templates
        self.use_templates_v2 = use_templates_v2
        self.max_templates = max_templates
        self.no_template_prob = no_template_prob
        self.overfit = overfit
        self.compute_frames = compute_frames
        self.bfactor_md_correction = bfactor_md_correction

        self.samples: list[pd.DataFrame] = []
        for d in datasets:
            self.samples.append(d.samples[:overfit] if overfit else d.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Set a random state
        random = np.random.default_rng()

        # Pick a random dataset
        dataset_idx = random.choice(len(self.datasets), p=self.probs)
        dataset = self.datasets[dataset_idx]

        # Get a sample from the dataset
        samples = self.samples[dataset_idx]
        sample_idx = random.choice(
            len(samples),
            p=(
                samples["weight"] / np.sum(samples["weight"])
                if self.overfit
                else samples["weight"]
            ),
        )
        sample = samples.iloc[sample_idx].to_dict()
        sample: Sample = Sample(
            record_id=str(sample["record_id"]),
            chain_id=(
                int(sample["chain_id"]) if sample["chain_id"] is not None else None
            ),
            interface_id=(
                int(sample["interface_id"])
                if sample["interface_id"] is not None
                else None
            ),
            weight=float(sample["weight"]),
        )

        # Load record
        record = load_record(sample.record_id, dataset.record_dir)

        # Get the structure
        try:
            structure = load_structure(record, dataset.struct_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(structure)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Compute crop
        try:
            if self.max_tokens is not None:
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_atoms=self.max_atoms,
                    max_tokens=self.max_tokens,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                    random=random,
                )
                if len(tokenized.tokens) == 0:
                    msg = "No tokens in cropped structure."
                    raise ValueError(msg)  # noqa: TRY301
        except Exception as e:  # noqa: BLE001
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Get unique chain ids
        chain_ids = set(tokenized.tokens["asym_id"])

        # Load msas and templates
        try:
            msas = load_msas(
                chain_ids=chain_ids,
                record=record,
                msa_dir=dataset.msa_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"MSA loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Load templates
        templates = FileNotFoundError
        if self.use_templates and dataset.template_dir is not None:
            try:
                if self.use_templates_v2:
                    templates = None
                    templates_features = load_templates_v2(
                        tokenized=tokenized,
                        record=record,
                        structure_dir=dataset.struct_dir,
                        template_dir=dataset.template_dir,
                        tokenizer=dataset.tokenizer,
                        max_templates=self.max_templates,
                        no_template_prob=self.no_template_prob,
                        training=True,
                        random=random,
                        max_tokens=self.max_tokens,
                    )
                else:
                    templates = load_templates(
                        chain_ids=chain_ids,
                        record=record,
                        template_dir=dataset.template_dir,
                        max_templates=self.max_templates,
                        no_template_prob=self.no_template_prob,
                        training=True,
                        random=random,
                    )
            except Exception as e:  # noqa: BLE001
                print(
                    f"Template loading failed for {record.id} with error {e}. Using no templates."
                )
                templates = None
                if self.use_templates_v2:
                    templates_features = load_dummy_templates_v2(
                        tdim=self.max_templates, num_tokens=self.max_tokens
                    )
        else:
            templates = None
            if self.use_templates_v2:
                templates_features = load_dummy_templates_v2(
                    tdim=self.max_templates, num_tokens=self.max_tokens
                )

        # Load molecules
        try:
            # Try to find molecules in the dataset moldir if provided
            # Find missing ones in global moldir and check if all found
            molecules = {}
            molecules.update(self.canonicals)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(self.canonicals.keys())
            if dataset.moldir is not None:
                molecules.update(load_molecules(dataset.moldir, mol_names))

            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.moldir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Finalize input data
        input_data = InputTraining(
            tokens=tokenized.tokens,
            bonds=tokenized.bonds,
            structure=structure,
            msa=msas,
            templates=templates,
            record=record,
        )

        # Compute features
        try:
            features: dict = dataset.featurizer.process(
                input_data,
                molecules=molecules,
                random=random,
                training=True,
                max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
                max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                num_ensembles=self.num_ensembles,
                ensemble_sample_replacement=self.ensemble_sample_replacement,
                disto_use_ensemble=self.disto_use_ensemble,
                fix_single_ensemble=self.fix_single_ensemble,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                contact_conditioned_prop=self.contact_conditioned_prop,
                binder_pocket_cutoff_min=self.binder_pocket_cutoff_min,
                binder_pocket_cutoff_max=self.binder_pocket_cutoff_max,
                binder_pocket_sampling_geometric_p=self.binder_pocket_sampling_geometric_p,
                compute_affinity=self.return_affinity,
                single_sequence_prop=self.single_sequence_prop,
                msa_sampling=self.msa_sampling,
                use_templates=self.use_templates,
                max_templates=self.max_templates,
                override_bfactor=dataset.override_bfactor,
                override_method=dataset.override_method,
                compute_frames=self.compute_frames,
                use_templates_v2=self.use_templates_v2,
                bfactor_md_correction=self.bfactor_md_correction,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            import traceback

            traceback.print_exc()
            return self.__getitem__(idx)

        # Compute template features
        if self.use_templates_v2:
            features.update(templates_features)

        features["pdb_id"] = record.id
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


class ValidationDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        canonicals: dict[str, Mol],
        moldir: str,
        seed: int,
        max_atoms: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_seqs: Optional[int] = None,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        num_ensembles: int = 1,
        ensemble_sample_replacement: Optional[bool] = False,
        disto_use_ensemble: Optional[bool] = False,
        fix_single_ensemble: Optional[bool] = True,
        overfit: Optional[int] = None,
        return_symmetries: Optional[bool] = False,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        contact_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        return_affinity: bool = False,
        use_templates: bool = False,
        use_templates_v2: bool = False,
        max_templates: int = 4,
        no_template_prob: float = 0.0,
        compute_frames: bool = False,
        bfactor_md_correction: bool = False,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        datasets : List[Dataset]
            The datasets to sample from.
        seed : int
            The random seed.
        max_tokens : int
            The maximum number of tokens.
        overfit : bool
            Whether to overfit the dataset

        """
        super().__init__()
        self.datasets = datasets
        self.canonicals = canonicals
        self.moldir = moldir
        self.max_atoms = max_atoms
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.seed = seed
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.overfit = overfit
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.num_ensembles = num_ensembles
        self.ensemble_sample_replacement = ensemble_sample_replacement
        self.disto_use_ensemble = disto_use_ensemble
        self.fix_single_ensemble = fix_single_ensemble
        self.return_symmetries = return_symmetries
        self.return_affinity = return_affinity
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.contact_conditioned_prop = contact_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff
        self.use_templates = use_templates
        self.use_templates_v2 = use_templates_v2
        self.max_templates = max_templates
        self.no_template_prob = no_template_prob
        self.compute_frames = compute_frames
        self.bfactor_md_correction = bfactor_md_correction

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Set random state
        seed = self.seed if self.overfit is None else None
        random = np.random.default_rng(seed)

        # Pick dataset based on idx
        for idx_dataset, dataset in enumerate(self.datasets):  # noqa: B007
            size = len(dataset.samples)
            if self.overfit is not None:
                size = min(size, self.overfit)
            if idx < size:
                break
            idx -= size

        # Get a sample from the dataset
        sample = Sample(**dataset.samples.iloc[idx].to_dict())
        record = load_record(sample.record_id, dataset.record_dir)

        # Get the structure
        try:
            structure = load_structure(record, dataset.struct_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(structure)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Get unique chains
        chain_ids = set(np.unique(tokenized.tokens["asym_id"]).tolist())

        # Load msas and templates
        try:
            msas = load_msas(chain_ids, record, dataset.msa_dir)
        except Exception as e:  # noqa: BLE001
            print(f"MSA loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Load templates
        if self.use_templates and dataset.template_dir is not None:
            try:
                if self.use_templates_v2:
                    templates = None
                    templates_features = load_templates_v2(
                        tokenized=tokenized,
                        record=record,
                        structure_dir=dataset.struct_dir,
                        template_dir=dataset.template_dir,
                        tokenizer=dataset.tokenizer,
                        max_templates=self.max_templates,
                        no_template_prob=self.no_template_prob,
                        training=False,
                        random=random,
                        max_tokens=len(tokenized.tokens),
                    )
                else:
                    templates = load_templates(
                        chain_ids=chain_ids,
                        record=record,
                        template_dir=dataset.template_dir,
                        max_templates=self.max_templates,
                        no_template_prob=self.no_template_prob,
                        training=False,
                        random=random,
                    )
            except Exception as e:  # noqa: BLE001
                print(
                    f"Template loading failed for {record.id} with error {e}. Using no templates."
                )
                templates = None
                if self.use_templates_v2:
                    templates_features = load_dummy_templates_v2(
                        tdim=self.max_templates, num_tokens=len(tokenized.tokens)
                    )
        else:
            templates = None
            if self.use_templates_v2:
                templates_features = load_dummy_templates_v2(
                    tdim=self.max_templates, num_tokens=len(tokenized.tokens)
                )

        try:
            # Try to find molecules in the dataset moldir if provided
            # Find missing ones in global moldir and check if all found
            molecules = {}
            molecules.update(self.canonicals)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(self.canonicals.keys())
            if dataset.moldir is not None:
                molecules.update(load_molecules(dataset.moldir, mol_names))

            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(self.moldir, mol_names))
        except Exception as e:  # noqa: BLE001
            print(f"Molecule loading failed for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Finalize input data
        input_data = InputTraining(
            tokens=tokenized.tokens,
            bonds=tokenized.bonds,
            structure=structure,
            msa=msas,
            templates=templates,
            record=record,
        )

        # Compute features
        try:
            features: dict = dataset.featurizer.process(
                input_data,
                molecules=molecules,
                random=random,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                num_ensembles=self.num_ensembles,
                ensemble_sample_replacement=self.ensemble_sample_replacement,
                disto_use_ensemble=self.disto_use_ensemble,
                fix_single_ensemble=self.fix_single_ensemble,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                contact_conditioned_prop=self.contact_conditioned_prop,
                binder_pocket_cutoff_min=self.binder_pocket_cutoff,
                binder_pocket_cutoff_max=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=1.0,  # this will only sample a single pocket token
                only_ligand_binder_pocket=True,
                only_pp_contact=True,
                compute_affinity=self.return_affinity,
                single_sequence_prop=0.0,
                use_templates=self.use_templates,
                use_templates_v2=self.use_templates_v2,
                max_templates=self.max_templates,
                override_method=dataset.override_method,
                compute_frames=self.compute_frames,
                bfactor_md_correction=self.bfactor_md_correction,
            )

        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        if self.use_templates_v2:
            features.update(templates_features)

        # Add dataset idx
        idx_dataset = torch.tensor([idx_dataset], dtype=torch.long)
        features.update({"idx_dataset": idx_dataset})
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataaset.

        """
        if self.overfit is not None:
            length = sum(len(d.samples[: self.overfit]) for d in self.datasets)
        else:
            length = sum(len(d.samples) for d in self.datasets)

        return length


class Boltz2TrainingDataModule(pl.LightningDataModule):
    """DataModule for Boltz2."""

    def __init__(self, cfg: DataConfigV2) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfigV2
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg

        assert self.cfg.val_batch_size == 1, "Validation only works with batch size=1."
        # Load datasets
        train: list[Dataset] = []
        val: list[Dataset] = []

        for data_config in cfg.datasets:
            # Get relevant directories
            manifest_path = Path(data_config.target_dir) / "manifest.json"
            struct_dir = Path(data_config.target_dir) / "structures"
            record_dir = Path(data_config.target_dir) / "records"
            msa_dir = Path(data_config.msa_dir)

            # Get template_dir, if any
            template_dir = data_config.template_dir
            template_dir = Path(template_dir) if template_dir is not None else None

            # Get moldir, if any
            moldir = data_config.moldir
            moldir = Path(moldir) if moldir is not None else None

            # Load all records
            manifest: Manifest = Manifest.load(manifest_path)

            # Split records if givens
            if data_config.split is not None:
                with Path(data_config.split).open("r") as f:
                    split = {x.lower() for x in f.read().splitlines()}

                train_records = []
                val_records = []
                for record in manifest.records:
                    if record.id.lower() in split:
                        val_records.append(record)
                    else:
                        train_records.append(record)
            else:
                train_records = manifest.records
                if cfg.overfit is None:
                    val_records = []
                else:
                    print("Warning: modified overfit val behavior.")
                    val_records = manifest.records[: cfg.overfit]

            print("train_records before filter", len(train_records))

            # Apply dataset-specific filters
            if data_config.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in data_config.filters)
                ]

            # Train with subset of data
            if data_config.use_train_subset is not None:
                # Shuffle train_records list
                assert 0 < data_config.use_train_subset < 1.0
                rng = np.random.default_rng(cfg.random_seed)
                rng.shuffle(train_records)
                train_records = train_records[
                    0 : int(len(train_records) * data_config.use_train_subset)
                ]
            print("train_records after filter", len(train_records))
            print("val_records after filter", len(val_records))

            # Get samples
            train_samples: list[Sample] = data_config.sampler.sample(train_records)
            val_samples: list[Sample] = [Sample(r.id) for r in val_records]
            del manifest, train_records, val_records

            # Convert samples to pandas dataframe to avoid copy-on-write behavior
            train_samples = pd.DataFrame(
                [
                    (
                        r.record_id,
                        r.chain_id,
                        r.interface_id,
                        r.weight,
                    )
                    for r in train_samples
                ],
                columns=["record_id", "chain_id", "interface_id", "weight"],
            )
            val_samples = pd.DataFrame(
                [s.record_id for s in val_samples], columns=["record_id"]
            )

            # Use appropriate string type
            train_samples = train_samples.replace({np.nan: None})
            val_samples = val_samples.replace({np.nan: None})
            train_samples["record_id"] = train_samples["record_id"].astype("string")
            val_samples["record_id"] = val_samples["record_id"].astype("string")

            # Create train dataset
            if data_config.prob > 0:
                train.append(
                    Dataset(
                        samples=train_samples,
                        record_dir=record_dir,
                        struct_dir=struct_dir,
                        msa_dir=msa_dir,
                        template_dir=template_dir,
                        moldir=moldir,
                        prob=data_config.prob,
                        cropper=data_config.cropper,
                        tokenizer=cfg.tokenizer,
                        featurizer=cfg.featurizer,
                        val_group=data_config.val_group,
                        has_structure_label=data_config.has_structure_label,
                        has_affinity_label=data_config.has_affinity_label,
                        symmetry_correction=data_config.symmetry_correction,
                        override_bfactor=data_config.override_bfactor,
                        override_method=data_config.override_method,
                    )
                )

            # Create validation dataset
            if len(val_samples) > 0:
                val.append(
                    Dataset(
                        samples=val_samples,
                        record_dir=record_dir,
                        struct_dir=struct_dir,
                        msa_dir=msa_dir,
                        template_dir=template_dir,
                        moldir=moldir,
                        prob=data_config.prob,
                        cropper=data_config.cropper,
                        tokenizer=cfg.tokenizer,
                        featurizer=cfg.featurizer,
                        val_group=data_config.val_group,
                        has_structure_label=data_config.has_structure_label,
                        has_affinity_label=data_config.has_affinity_label,
                        symmetry_correction=data_config.symmetry_correction,
                    )
                )

        # Print dataset sizes
        for dataset in train:
            dataset: Dataset
            print(f"Training dataset size: {len(dataset.samples)}")

        self.val_group_mapper = defaultdict(dict)
        for i, dataset in enumerate(train if cfg.overfit is not None else val):
            dataset: Dataset
            print(f"Validation dataset size: {len(dataset.samples)}")
            self.val_group_mapper[i]["label"] = dataset.val_group
            self.val_group_mapper[i]["symmetry_correction"] = (
                dataset.symmetry_correction
            )

        # Load canonical molecules
        canonicals = load_canonicals(cfg.moldir)

        # Create wrapper datasets
        self._train_set = TrainingDataset(
            datasets=train,
            canonicals=canonicals,
            moldir=cfg.moldir,
            samples_per_epoch=cfg.samples_per_epoch,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            num_ensembles=cfg.num_ensembles_train,
            ensemble_sample_replacement=True,
            disto_use_ensemble=cfg.disto_use_ensemble,
            fix_single_ensemble=cfg.fix_single_ensemble,
            overfit=cfg.overfit,
            binder_pocket_conditioned_prop=cfg.train_binder_pocket_conditioned_prop,
            contact_conditioned_prop=cfg.train_contact_conditioned_prop,
            binder_pocket_cutoff_min=cfg.binder_pocket_cutoff_min,
            binder_pocket_cutoff_max=cfg.binder_pocket_cutoff_max,
            binder_pocket_sampling_geometric_p=cfg.binder_pocket_sampling_geometric_p,
            return_symmetries=cfg.return_train_symmetries,
            return_affinity=cfg.return_train_affinity,
            single_sequence_prop=cfg.single_sequence_prop_training,
            msa_sampling=cfg.msa_sampling_training,
            use_templates=cfg.use_templates,
            use_templates_v2=cfg.use_templates_v2,
            max_templates=cfg.max_templates_train,
            no_template_prob=cfg.no_template_prob_train,
            compute_frames=cfg.compute_frames,
            bfactor_md_correction=cfg.bfactor_md_correction,
        )
        self._val_set = ValidationDataset(
            datasets=train if cfg.overfit is not None else val,
            canonicals=canonicals,
            moldir=cfg.moldir,
            seed=cfg.random_seed,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            num_ensembles=cfg.num_ensembles_val,
            ensemble_sample_replacement=False,
            disto_use_ensemble=cfg.disto_use_ensemble,
            fix_single_ensemble=cfg.fix_single_ensemble,
            overfit=cfg.overfit,
            return_symmetries=cfg.return_val_symmetries,
            binder_pocket_conditioned_prop=cfg.val_binder_pocket_conditioned_prop,
            contact_conditioned_prop=cfg.val_contact_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff_val,
            return_affinity=cfg.return_val_affinity,
            use_templates=cfg.use_templates,
            use_templates_v2=cfg.use_templates_v2,
            max_templates=cfg.max_templates_val,
            no_template_prob=cfg.no_template_prob_val,
            compute_frames=cfg.compute_frames,
            bfactor_md_correction=cfg.bfactor_md_correction,
        )

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002 (unused)
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.s

        """
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )
