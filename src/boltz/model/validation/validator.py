from collections import defaultdict
from typing import Optional

import torch
import torch._dynamo
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric

from boltz.data import const
from boltz.model.loss.distogram import distogram_loss
from boltz.model.loss.inference import (
    compute_chain_clashes,
    compute_pb_flatness_metrics,
    compute_pb_geometry_metrics,
    compute_stereo_metrics,
)
from boltz.model.loss.validation import (
    compute_pae_mae,
    compute_pde_mae,
    compute_plddt_mae,
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
)


class Validator(nn.Module):
    """Compute validation step and aggregation."""

    def __init__(
        self,
        val_names: list[str],
        confidence_prediction: bool = False,
        physicalism_metrics: bool = False,
        override_val_method: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.val_names = val_names

        self.override_val_method = override_val_method
        if override_val_method is not None:
            override_val_method = override_val_method.lower()
            assert override_val_method in const.method_types_ids, "Invalid method type."
            self.override_val_method = const.method_types_ids[override_val_method]

        self.num_val_datasets = num_val_datasets = len(val_names)

        msg = "Only one dataset supported for now per validator. Define multiple validators for multiple datasets."
        assert num_val_datasets == 1, msg

        # Folding metrics
        folding_metric_labels = [
            "lddt",
            "disto_lddt",
            "complex_lddt",
            # "rmsd",
            "disto_loss",
        ]
        self.folding_metrics = nn.ModuleDict(
            {
                k: nn.ModuleList([nn.ModuleDict() for _ in range(num_val_datasets)])
                for k in folding_metric_labels
            }
        )

        self.physicalism_metrics = physicalism_metrics
        if physicalism_metrics:
            # Physical realism metrics
            physicalism_metric_labels = ["clash", "pb"]
            pb_metric_labels = [
                "bond_length",
                "bond_angle",
                "internal_clash",
                "atom_chirality",
                "bond_stereochemistry",
                "ring_5_flatness",
                "ring_6_flatness",
                "double_bond_flatness",
            ]
            self.physicalism_metrics = nn.ModuleDict(
                {
                    k: nn.ModuleList([nn.ModuleDict() for _ in range(num_val_datasets)])
                    for k in physicalism_metric_labels
                }
            )

        # Confidence metrics
        confidence_metric_prefixes = [
            "top1",
            "iplddt_top1",
            "ipde_top1",
            "pde_top1",
            "ptm_top1",
            "iptm_top1",
            "ligand_iptm_top1",
            "protein_iptm_top1",
            "avg",
        ]
        mae_metric_labels = ["plddt_mae", "pde_mae", "pae_mae"]
        lddt_confidence_metric_labels = [
            prefix + "_lddt" for prefix in confidence_metric_prefixes
        ]
        if physicalism_metrics:
            clash_confidence_metric_labels = [
                prefix + "_clash" for prefix in confidence_metric_prefixes
            ]
            pb_confidence_metric_labels = [
                prefix + "_pb" for prefix in confidence_metric_prefixes
            ]
        else:
            clash_confidence_metric_labels, pb_confidence_metric_labels = [], []

        if confidence_prediction:
            self.confidence_metrics = nn.ModuleDict(
                {
                    k: nn.ModuleList([nn.ModuleDict() for _ in range(num_val_datasets)])
                    for k in lddt_confidence_metric_labels
                    + mae_metric_labels
                    + clash_confidence_metric_labels
                    + pb_confidence_metric_labels
                }
            )

        # Initialize metrics for datasets
        for val_idx in range(num_val_datasets):
            for m_ in [
                *const.out_types,
                "pocket_ligand_protein",
                "contact_protein_protein",
            ]:
                self.folding_metrics["disto_lddt"][val_idx][m_] = MeanMetric()
                for suffix in [":recall", ":precision", ":diversity"]:
                    m = m_ + suffix
                    self.folding_metrics["lddt"][val_idx][m] = MeanMetric()
                    self.folding_metrics["complex_lddt"][val_idx][m] = MeanMetric()
                    if confidence_prediction:
                        for k in lddt_confidence_metric_labels + mae_metric_labels:
                            self.confidence_metrics[k][val_idx][m_] = MeanMetric()

            for m in const.out_single_types:
                if confidence_prediction:
                    self.confidence_metrics["plddt_mae"][val_idx][m] = MeanMetric()

            # for m in [
            #    "rmsd",
            #    "best_rmsd:recall",
            #    "best_rmsd:precision",
            #    "rmsd:diversity",
            #    "rmsf_score",
            # ]:
            #    self.folding_metrics["rmsd"][val_idx][m] = MeanMetric()

            for m in ["disto_loss"]:
                self.folding_metrics["disto_loss"][val_idx][m] = MeanMetric()

            if self.physicalism_metrics:
                for m_ in const.out_single_types:
                    m = "sym_" + m_
                    self.physicalism_metrics["clash"][val_idx][m] = MeanMetric()
                    if confidence_prediction:
                        for k in clash_confidence_metric_labels:
                            self.confidence_metrics[k][val_idx][m] = MeanMetric()

                for m_ in const.clash_types:
                    m = "asym_" + m_
                    self.physicalism_metrics["clash"][val_idx][m] = MeanMetric()
                    if confidence_prediction:
                        for k in clash_confidence_metric_labels:
                            self.confidence_metrics[k][val_idx][m] = MeanMetric()

                for m in pb_metric_labels:
                    self.physicalism_metrics["pb"][val_idx][m] = MeanMetric()
                    if confidence_prediction:
                        for k in pb_confidence_metric_labels:
                            self.confidence_metrics[k][val_idx][m] = MeanMetric()

    def run_model(
        self, model: LightningModule, batch: dict[str, torch.Tensor], idx_dataset: int
    ) -> dict[str, torch.Tensor]:
        """Compute the forward pass."""
        if self.override_val_method is not None:
            new_feature = batch["method_feature"] * 0 + self.override_val_method
            batch["method_feature"] = new_feature

        out = model(
            batch,
            recycling_steps=model.validation_args.recycling_steps,
            num_sampling_steps=model.validation_args.sampling_steps,
            diffusion_samples=model.validation_args.diffusion_samples,
            run_confidence_sequentially=model.validation_args.get(
                "run_confidence_sequentially", False
            ),
        )

        return out

    # @abstractmethod
    def process(
        self,
        model: LightningModule,
        batch: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        idx_dataset: int,
        n_samples: int,
    ) -> None:
        """Run a validation step.

        Parameters
        ----------
        model : LightningModule
            The LightningModule model.
        batch : Dict[str, torch.Tensor]
            The batch input.
        out : Dict[str, torch.Tensor]
            The output of the model.

        """
        raise NotImplementedError

    def get_local_val_index(self, model: LightningModule, idx_dataset: int) -> int:
        """Get the local validation index.

        Parameters
        ----------
        idx_dataset : int
            The dataset index.

        Returns
        -------
        int
            The local validation index.
        """
        val_name = model.val_group_mapper[idx_dataset]["label"]
        return self.val_names.index(val_name)

    def compute_disto_loss(
        self,
        model: LightningModule,
        out: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        idx_dataset: int,
    ) -> None:
        """Compute distogram loss."""
        # Compute validation disto loss
        val_disto_loss, _ = distogram_loss(
            out, batch, aggregate_distogram=model.aggregate_distogram
        )

        return val_disto_loss

    def compute_disto_lddt(self, model, batch, out, idx_dataset) -> tuple[dict, dict]:
        """Compute distogram lddt."""
        boundaries = torch.linspace(model.min_dist, model.max_dist, model.num_bins - 1)
        lower = torch.tensor([1.0])
        upper = torch.tensor([model.max_dist + 5.0])
        exp_boundaries = torch.cat((lower, boundaries, upper))
        mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
            out["pdistogram"]
        )

        # Compute true distogram
        K = batch["coords"].shape[1]
        true_center = batch["disto_coords_ensemble"].reshape(K, -1, 3)  # (K, L, 3)

        batch["token_disto_mask"] = batch["token_disto_mask"] * batch["has_structure"]

        # Compute distogram lddt by looping over predicted distograms
        disto_lddt_dict = defaultdict(
            lambda: torch.zeros(K, model.num_distograms).to(model.device)
        )
        disto_total_dict = defaultdict(
            lambda: torch.zeros(K, model.num_distograms).to(model.device)
        )
        for i in range(model.num_distograms):
            # Compute predicted dists
            preds = out["pdistogram"][:, :, :, i]
            pred_softmax = torch.softmax(preds, dim=-1)
            pred_softmax = pred_softmax.argmax(dim=-1)
            pred_softmax = torch.nn.functional.one_hot(
                pred_softmax, num_classes=preds.shape[-1]
            )
            pred_dist_i = (pred_softmax * mid_points).sum(dim=-1)  # (B, L, L)
            del pred_softmax

            # Compute true distances for each conformer
            # Implemented in a loop to avoid memory issues with large number of
            # conformers. Batched version over K factored_token_lddt_dist_loss_ensemble
            # more efficient for small K.
            for k in range(K):
                true_dists_k = torch.cdist(true_center[k], true_center[k])[
                    None
                ]  # (1, L * L)

                # Compute lddt
                disto_lddt_dict_, disto_total_dict_ = factored_token_lddt_dist_loss(
                    feats=batch,
                    true_d=true_dists_k,
                    pred_d=pred_dist_i,
                )

            for key in disto_lddt_dict_:
                disto_lddt_dict[key][k, i] = disto_lddt_dict_[key].item()
                disto_total_dict[key][k, i] = disto_total_dict_[key].item()

        for key in disto_lddt_dict:
            # Take min over distograms and average over conformers. Add batch dimension.
            disto_lddt_dict[key] = (
                disto_lddt_dict[key].min(dim=1).values.mean(dim=0)[None]
            )
            disto_total_dict[key] = (
                disto_total_dict[key].min(dim=1).values.mean(dim=0)[None]
            )
        del true_center
        del preds

        return disto_lddt_dict, disto_total_dict

    def get_true_coords(
        self,
        model,
        batch,
        out,
        diffusion_samples,
        symmetry_correction,
        expand_to_diffusion_samples,
    ) -> dict[str, torch.Tensor]:
        # Get true coordinates
        # TODO modiy for each validator, for now using default from model
        return model.get_true_coordinates(
            batch=batch,
            out=out,
            diffusion_samples=diffusion_samples,
            symmetry_correction=symmetry_correction,
            expand_to_diffusion_samples=expand_to_diffusion_samples,
        )

    def get_lddt_metrics(
        self,
        model,
        batch,
        out,
        idx_dataset,
        n_samples,
        true_coords_resolved_mask,
        true_coords,
        expand_to_diffusion_samples,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        K = batch["coords"].shape[1]

        if not expand_to_diffusion_samples:
            true_coords_resolved_mask = true_coords_resolved_mask.unsqueeze(0).repeat(
                (n_samples, 1)
            )

        ### Compute lddt ###
        # Implemented in a loop to avoid memory issues with large number
        # of conformers
        all_lddt_dict = defaultdict(list)
        all_total_dict = defaultdict(list)
        for ensemble_idx in range(K):
            # This OOM for large n_samples. Need to chunk or loop over samples.

            if expand_to_diffusion_samples:
                true_coords_k = true_coords[:, ensemble_idx]
            else:
                true_coords_k = (
                    true_coords[ensemble_idx].unsqueeze(0).repeat((n_samples, 1, 1))
                )

            all_lddt_dict_s, all_total_dict_s = factored_lddt_loss(
                feats=batch,
                atom_mask=true_coords_resolved_mask,
                true_atom_coords=true_coords_k,  # (multiplicity, L, 3)
                pred_atom_coords=out["sample_atom_coords"],
                multiplicity=n_samples,
                representative_lddt=model.representative_lddt,
                exclude_ions=model.exclude_ions_from_lddt,
            )
            for key in all_lddt_dict_s:
                all_lddt_dict[key].append(all_lddt_dict_s[key])
                all_total_dict[key].append(all_total_dict_s[key])

        for key in all_lddt_dict:
            all_lddt_dict[key] = torch.stack(
                all_lddt_dict[key], dim=1
            )  # (multiplicity, K)
            all_total_dict[key] = torch.stack(all_total_dict[key], dim=1)
        return all_lddt_dict, all_total_dict

    def get_clash_metrics(
        self,
        batch,
        out,
    ):
        pair_clash_dict, pair_total_dict = compute_chain_clashes(
            pred_atom_coords=out["sample_atom_coords"],
            feats=batch,
        )

        return pair_clash_dict, pair_total_dict

    def get_pb_metrics(
        self,
        batch,
        out,
    ):
        (
            num_bond_length_failures,
            num_bond_angle_failures,
            num_internal_clash_failures,
            num_ligands,
        ) = compute_pb_geometry_metrics(
            pred_atom_coords=out["sample_atom_coords"],
            feats=batch,
        )
        (
            num_chiral_atom_violations,
            num_chiral_atoms,
            num_stereo_bond_violations,
            num_stereo_bonds,
        ) = compute_stereo_metrics(
            pred_atom_coords=out["sample_atom_coords"], feats=batch
        )

        (
            num_aromatic_5_violations,
            num_aromatic_5_rings,
            num_aromatic_6_violations,
            num_aromatic_6_rings,
            num_double_bond_violations,
            num_double_bonds,
        ) = compute_pb_flatness_metrics(
            pred_atom_coords=out["sample_atom_coords"], feats=batch
        )

        pb_failure_dict = {
            "bond_length": num_bond_length_failures,
            "bond_angle": num_bond_angle_failures,
            "internal_clash": num_internal_clash_failures,
            "atom_chirality": num_chiral_atom_violations,
            "bond_stereochemistry": num_stereo_bond_violations,
            "ring_5_flatness": num_aromatic_5_violations,
            "ring_6_flatness": num_aromatic_6_violations,
            "double_bond_flatness": num_double_bond_violations,
        }
        pb_total_dict = {
            "bond_length": num_ligands,
            "bond_angle": num_ligands,
            "internal_clash": num_ligands,
            "atom_chirality": num_chiral_atoms,
            "bond_stereochemistry": num_stereo_bonds,
            "ring_5_flatness": num_aromatic_5_rings,
            "ring_6_flatness": num_aromatic_6_rings,
            "double_bond_flatness": num_double_bonds,
        }
        return pb_failure_dict, pb_total_dict

    def compute_best_lddt_metrics(
        self,
        model,
        batch,
        all_lddt_dict,
        all_total_dict,
        all_lddt_dict_div,
        all_total_dict_div,
        n_samples,
    ):
        K = batch["coords"].shape[1]

        # if the multiplicity used is > 1 then we take the best lddt of the different samples
        # AF3 combines this with the confidence based filtering
        best_lddt_dict, best_total_dict = {}, {}
        best_complex_lddt_dict, best_complex_total_dict = {}, {}
        # B = true_coords.shape[0] // n_samples
        if n_samples > 1 and batch["has_structure"].sum() > 0:
            # NOTE: we can change the way we aggregate the lddt
            complex_total = 0
            complex_lddt = 0
            for key in all_lddt_dict:
                complex_lddt += all_lddt_dict[key] * all_total_dict[key]
                complex_total += all_total_dict[key]
            complex_lddt /= complex_total + 1e-7

            # Take best over samples, average over conformers: recall groundtruth
            # conformers
            suffix = ":recall"
            best_complex_idx = complex_lddt.argmax(dim=0)
            for key in all_lddt_dict:
                # take best across diffusion samples
                best_idx = all_lddt_dict[key].argmax(dim=0)
                best_lddt_dict[key + suffix] = all_lddt_dict[key][
                    best_idx, torch.arange(K)
                ].mean(dim=0)[None]  # take average across conformers in ensemble
                best_total_dict[key + suffix] = all_total_dict[key][
                    best_idx, torch.arange(K)
                ].mean(dim=0)[None]
                # mean(dim=0) since samples was argmaxed, add back batch dim
                best_complex_lddt_dict[key + suffix] = all_lddt_dict[key][
                    best_complex_idx, torch.arange(K)
                ].mean(dim=0)[None]
                best_complex_total_dict[key + suffix] = all_total_dict[key][
                    best_complex_idx, torch.arange(K)
                ].mean(dim=0)[None]  # sum here ? TODO

            # Take best over conformers, average over samples: precision
            suffix = ":precision"
            best_complex_idx = complex_lddt.argmax(dim=1)
            for key in all_lddt_dict:
                # take best across diffusion conformers
                best_idx = all_lddt_dict[key].argmax(dim=1)
                best_lddt_dict[key + suffix] = all_lddt_dict[key][
                    torch.arange(n_samples), best_idx
                ].mean(dim=0)[None]  # take average across samples in ensemble
                best_total_dict[key + suffix] = all_total_dict[key][
                    torch.arange(n_samples), best_idx
                ].mean(dim=0)[None]
                # dim 0 since samples was argmaxed, add back batch dim [None]
                best_complex_lddt_dict[key + suffix] = all_lddt_dict[key][
                    torch.arange(n_samples), best_complex_idx
                ].mean(dim=0)[None]
                best_complex_total_dict[key + suffix] = all_total_dict[key][
                    torch.arange(n_samples), best_complex_idx
                ].mean(dim=0)[None]

            # Diversity metrics
            if all_lddt_dict_div is not None and all_total_dict_div is not None:
                suffix = ":diversity"
                for key in all_lddt_dict_div:
                    # Take average across all samples
                    best_lddt_dict[key + suffix] = all_lddt_dict_div[key].mean(dim=0)[
                        None
                    ]
                    best_total_dict[key + suffix] = all_total_dict_div[key].mean(dim=0)[
                        None
                    ]
                    best_complex_lddt_dict[key + suffix] = all_lddt_dict_div[key].mean(
                        dim=0
                    )[None]  # complex_lddt same as lddt
                    best_complex_total_dict[key + suffix] = all_total_dict_div[
                        key
                    ].mean(dim=0)[None]
        else:
            # Take average across conformers in ensemble.
            for key in all_lddt_dict:
                for suffix in [":recall", ":precision"]:
                    best_lddt_dict[key + suffix] = (
                        all_lddt_dict[key].max(dim=1).values[None]
                    )  # (sample, K) -> (B=1, samples=1)
                    best_total_dict[key + suffix] = (
                        all_total_dict[key].max(dim=1).values[None]
                    )
                    best_complex_lddt_dict[key + suffix] = (
                        all_lddt_dict[key].max(dim=1).values[None]
                    )
                    best_complex_total_dict[key + suffix] = (
                        all_total_dict[key].max(dim=1).values[None]
                    )
                suffix = ":diversity"
                best_lddt_dict[key + suffix] = torch.tensor([0.0]).to(model.device)
                best_total_dict[key + suffix] = torch.tensor([1.0]).to(model.device)
                best_complex_lddt_dict[key + suffix] = torch.tensor([0.0]).to(
                    model.device
                )
                best_complex_total_dict[key + suffix] = torch.tensor([1.0]).to(
                    model.device
                )

        return (
            best_lddt_dict,
            best_total_dict,
            best_complex_lddt_dict,
            best_complex_total_dict,
        )

    def get_confidence_metrics(
        self,
        model,
        batch,
        out,
        idx_dataset,
        n_samples,
        true_coords,
        true_coords_resolved_mask,
        expand_to_diffusion_samples,
    ):
        K = batch["coords"].shape[1]
        # note: for now we don't have pae predictions so have to use pLDDT instead of pTM
        # also, while AF3 differentiates the best prediction per confidence type we are currently not doing it
        # consider this in the future as well as weighing the different pLLDT types before aggregation

        msg = "Confidence_prediction is not supported for num_ensembles_val > 1"
        assert batch["coords"].shape[1] == 1, msg

        mae_plddt_dicts = defaultdict(list)
        total_mae_plddt_dicts = defaultdict(list)
        mae_pde_dicts = defaultdict(list)
        total_mae_pde_dicts = defaultdict(list)
        mae_pae_dicts = defaultdict(list)
        total_mae_pae_dicts = defaultdict(list)

        # All ensembles have same mask
        if not expand_to_diffusion_samples:
            true_coords_resolved_mask = true_coords_resolved_mask.unsqueeze(0).repeat(
                (n_samples, 1)
            )

        for ensemble_idx in range(K):
            if expand_to_diffusion_samples:
                true_coords_k = true_coords[:, ensemble_idx]
            else:
                true_coords_k = (
                    true_coords[ensemble_idx].unsqueeze(0).repeat((n_samples, 1, 1))
                )

            mae_plddt_dict, total_mae_plddt_dict = compute_plddt_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords_k,
                pred_lddt=out["plddt"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                token_level_confidence=model.token_level_confidence,
                multiplicity=n_samples,
            )
            for key in mae_plddt_dict:
                mae_plddt_dicts[key].append(mae_plddt_dict[key])
                total_mae_plddt_dicts[key].append(total_mae_plddt_dict[key])

            mae_pde_dict, total_mae_pde_dict = compute_pde_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords_k,
                pred_pde=out["pde"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )

            for key in mae_pde_dict:
                mae_pde_dicts[key].append(mae_pde_dict[key])
                total_mae_pde_dicts[key].append(total_mae_pde_dict[key])

            mae_pae_dict, total_mae_pae_dict = compute_pae_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords_k,
                pred_pae=out["pae"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )

            for key in mae_pae_dict:
                mae_pae_dicts[key].append(mae_pae_dict[key])
                total_mae_pae_dicts[key].append(total_mae_pae_dict[key])

        # Take mean over ensembles
        for key in mae_plddt_dicts:
            mae_plddt_dicts[key] = torch.stack(mae_plddt_dicts[key], dim=0).mean(dim=0)
            total_mae_plddt_dicts[key] = torch.stack(
                total_mae_plddt_dicts[key], dim=0
            ).mean(dim=0)

        for key in mae_pde_dicts:
            mae_pde_dicts[key] = torch.stack(mae_pde_dicts[key], dim=0).mean(dim=0)
            total_mae_pde_dicts[key] = torch.stack(
                total_mae_pde_dicts[key], dim=0
            ).mean(dim=0)

        for key in mae_pae_dicts:
            mae_pae_dicts[key] = torch.stack(mae_pae_dicts[key], dim=0).mean(dim=0)
            total_mae_pae_dicts[key] = torch.stack(
                total_mae_pae_dicts[key], dim=0
            ).mean(dim=0)

        return (
            mae_plddt_dicts,
            total_mae_plddt_dicts,
            mae_pde_dicts,
            total_mae_pde_dicts,
            mae_pae_dicts,
            total_mae_pae_dicts,
        )

    def update_confidence_metrics(
        self,
        batch,
        out,
        idx_dataset,
        n_samples,
        all_lddt_dict,
        all_total_dict,
        mae_plddt_dicts,
        total_mae_plddt_dicts,
        mae_pde_dicts,
        total_mae_pde_dicts,
        mae_pae_dicts,
        total_mae_pae_dicts,
        pair_clash_dict,
        pair_total_dict,
        pb_failure_dict,
        pb_total_dict,
        physicalism_metrics,
    ):
        K = batch["coords"].shape[1]

        for confidence_metric_name in [
            "complex_plddt",
            "complex_iplddt",
            "complex_pde",
            "complex_ipde",
            "ptm",
            "iptm",
            "ligand_iptm",
            "protein_iptm",
        ]:
            confidence_metric_val = out[confidence_metric_name].reshape(-1, n_samples)
            top1_idx = confidence_metric_val.argmax(dim=1)
            if confidence_metric_name == "complex_plddt":
                confidence_metric_prefix = "top1"
            elif "complex" in confidence_metric_name:
                confidence_metric_prefix = (
                    confidence_metric_name.split("_")[1] + "_top1"
                )
            else:
                confidence_metric_prefix = confidence_metric_name + "_top1"
            for key in all_lddt_dict:
                if key == "modified":
                    continue
                top1_val = (
                    all_lddt_dict[key]
                    .reshape(n_samples, K)[top1_idx, torch.arange(K)]
                    .mean(dim=0)
                )
                top1_total = (
                    all_total_dict[key]
                    .reshape(n_samples, K)[top1_idx, torch.arange(K)]
                    .mean(dim=0)
                )
                self.confidence_metrics[confidence_metric_prefix + "_lddt"][
                    idx_dataset
                ][key].update(top1_val, top1_total)

            if physicalism_metrics:
                for key in pair_clash_dict:
                    top1_val = pair_clash_dict[key][top1_idx]
                    top1_total = pair_total_dict[key][top1_idx]
                    self.confidence_metrics[confidence_metric_prefix + "_clash"][
                        idx_dataset
                    ][key].update(top1_val, top1_total)
                for key in pb_failure_dict:
                    top1_val = pb_failure_dict[key][top1_idx]
                    top1_total = pb_total_dict[key][top1_idx]
                    self.confidence_metrics[confidence_metric_prefix + "_pb"][
                        idx_dataset
                    ][key].update(top1_val, top1_total)

        for key in all_lddt_dict:
            if key == "modified":
                continue
            self.confidence_metrics["avg_lddt"][idx_dataset][key].update(
                all_lddt_dict[key], all_total_dict[key]
            )
            self.confidence_metrics["pde_mae"][idx_dataset][key].update(
                mae_pde_dicts[key], total_mae_pde_dicts[key]
            )
            self.confidence_metrics["pae_mae"][idx_dataset][key].update(
                mae_pae_dicts[key], total_mae_pae_dicts[key]
            )
        for key in mae_plddt_dicts:
            self.confidence_metrics["plddt_mae"][idx_dataset][key].update(
                mae_plddt_dicts[key], total_mae_plddt_dicts[key]
            )

        if physicalism_metrics:
            for key in pair_clash_dict:
                self.confidence_metrics["avg_clash"][idx_dataset][key].update(
                    pair_clash_dict[key], pair_total_dict[key]
                )
            for key in pb_failure_dict:
                self.confidence_metrics["avg_pb"][idx_dataset][key].update(
                    pb_failure_dict[key], pb_total_dict[key]
                )

    def update_lddt_rmsd_metrics(
        self,
        batch,
        disto_lddt_dict,
        disto_total_dict,
        best_lddt_dict,
        best_total_dict,
        best_complex_lddt_dict,
        best_complex_total_dict,
        # rmsds,
        idx_dataset,
        return_dict,
    ):
        # Folding metrics
        for m_ in const.out_types:
            if m_ == "ligand_protein":
                if torch.any(
                    batch["contact_conditioning"][
                        :, :, :, const.contact_conditioning_info["BINDER>POCKET"]
                    ].bool()
                ):
                    self.folding_metrics["disto_lddt"][idx_dataset][
                        "pocket_ligand_protein"
                    ].update(disto_lddt_dict[m_], disto_total_dict[m_])
                    for suffix in [":recall", ":precision", ":diversity"]:
                        m = m_ + suffix
                        self.folding_metrics["lddt"][idx_dataset][
                            "pocket_ligand_protein" + suffix
                        ].update(best_lddt_dict[m], best_total_dict[m])
                        self.folding_metrics["complex_lddt"][idx_dataset][
                            "pocket_ligand_protein" + suffix
                        ].update(best_complex_lddt_dict[m], best_complex_total_dict[m])
                else:
                    self.folding_metrics["disto_lddt"][idx_dataset][
                        "ligand_protein"
                    ].update(disto_lddt_dict[m_], disto_total_dict[m_])
                    for suffix in [":recall", ":precision", ":diversity"]:
                        m = m_ + suffix
                        self.folding_metrics["lddt"][idx_dataset][
                            "ligand_protein" + suffix
                        ].update(best_lddt_dict[m], best_total_dict[m])
                        self.folding_metrics["complex_lddt"][idx_dataset][
                            "ligand_protein" + suffix
                        ].update(best_complex_lddt_dict[m], best_complex_total_dict[m])

            elif m_ == "protein_protein":
                if torch.any(
                    batch["contact_conditioning"][
                        :, :, :, const.contact_conditioning_info["CONTACT"]
                    ].bool()
                ):
                    self.folding_metrics["disto_lddt"][idx_dataset][
                        "contact_protein_protein"
                    ].update(disto_lddt_dict[m_], disto_total_dict[m_])
                    for suffix in [":recall", ":precision", ":diversity"]:
                        m = m_ + suffix
                        self.folding_metrics["lddt"][idx_dataset][
                            "contact_protein_protein" + suffix
                        ].update(best_lddt_dict[m], best_total_dict[m])
                        self.folding_metrics["complex_lddt"][idx_dataset][
                            "contact_protein_protein" + suffix
                        ].update(best_complex_lddt_dict[m], best_complex_total_dict[m])
                else:
                    self.folding_metrics["disto_lddt"][idx_dataset][
                        "protein_protein"
                    ].update(disto_lddt_dict[m_], disto_total_dict[m_])
                    for suffix in [":recall", ":precision", ":diversity"]:
                        m = m_ + suffix
                        self.folding_metrics["lddt"][idx_dataset][
                            "protein_protein" + suffix
                        ].update(best_lddt_dict[m], best_total_dict[m])
                        self.folding_metrics["complex_lddt"][idx_dataset][
                            "protein_protein" + suffix
                        ].update(best_complex_lddt_dict[m], best_complex_total_dict[m])

            else:
                self.folding_metrics["disto_lddt"][idx_dataset][m_].update(
                    disto_lddt_dict[m_], disto_total_dict[m_]
                )
                for suffix in [":recall", ":precision", ":diversity"]:
                    m = m_ + suffix
                    self.folding_metrics["lddt"][idx_dataset][m].update(
                        best_lddt_dict[m], best_total_dict[m]
                    )
                    self.folding_metrics["complex_lddt"][idx_dataset][m].update(
                        best_complex_lddt_dict[m], best_complex_total_dict[m]
                    )

        # self.folding_metrics["rmsd"][idx_dataset]["rmsd"].update(rmsds)

        # RMSD metrics
        # if "best_rmsd_recall" in return_dict:
        #    best_rmsd_recall = return_dict["best_rmsd_recall"]
        #    self.folding_metrics["rmsd"][idx_dataset]["best_rmsd:recall"].update(
        #        best_rmsd_recall
        #    )
        # if "best_rmsd_precision" in return_dict:
        #    best_rmsd_precision = return_dict["best_rmsd_precision"]
        #    self.folding_metrics["rmsd"][idx_dataset]["best_rmsd:precision"].update(
        #        best_rmsd_precision
        #    )
        # if "rmsd_diversity" in return_dict:
        #    rmsd_diversity = return_dict["rmsd_diversity"]
        #    self.folding_metrics["rmsd"][idx_dataset]["rmsd:diversity"].update(
        #        rmsd_diversity
        #    )
        # if "rmsf_score" in return_dict:
        #    rmsf_score = return_dict["rmsf_score"]
        #    self.folding_metrics["rmsd"][idx_dataset]["rmsf_score"].update(rmsf_score)

    def update_physcialism_metrics(
        self,
        pair_clash_dict,
        pair_total_dict,
        pb_failure_dict,
        pb_total_dict,
        idx_dataset,
    ):
        for key in pair_clash_dict:
            self.physicalism_metrics["clash"][idx_dataset][key].update(
                pair_clash_dict[key], pair_total_dict[key]
            )

        for key in pb_failure_dict:
            self.physicalism_metrics["pb"][idx_dataset][key].update(
                pb_failure_dict[key], pb_total_dict[key]
            )

    def common_val_step(
        self,
        model: LightningModule,
        batch: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        idx_dataset: int,
        expand_to_diffusion_samples: bool = True,
    ) -> None:
        """Run a common validation step.

        Parameters
        ----------
        model : LightningModule
            The LightningModule model.
        batch : dict[str, torch.Tensor]
            The batch input.
        out : dict[str, torch.Tensor]
            The output of the model.
        """
        symmetry_correction = model.val_group_mapper[idx_dataset][
            "symmetry_correction"
        ]  # global val index

        # Get the local validation index from the global index
        idx_dataset = self.get_local_val_index(model, idx_dataset)

        n_samples = model.validation_args.diffusion_samples

        # Compute distogram loss and update metrics
        val_disto_loss = self.compute_disto_loss(model, out, batch, idx_dataset)

        # Compute distogram lddt and update metrics
        disto_lddt_dict, disto_total_dict = self.compute_disto_lddt(
            model, batch, out, idx_dataset
        )

        # Get true coords
        return_dict = self.get_true_coords(
            model,
            batch,
            out,
            n_samples,
            symmetry_correction,
            expand_to_diffusion_samples=expand_to_diffusion_samples,
        )

        # Get diversity metrics
        # if n_samples > 1:
        #    return_dict = self.get_diversity_metrics(
        #        batch, out, idx_dataset, return_dict, n_samples
        #    )

        # Move this and do better as to when to interleave
        true_coords = return_dict[
            "true_coords"
        ]  # (multiplicity, K, L, 3) if expand_to_diffusion_samples else (K, L, 3)
        true_coords_resolved_mask = return_dict[
            "true_coords_resolved_mask"
        ]  # (multiplicity, L) if expand_to_diffusion_samples else (L)
        # rmsds = return_dict["rmsds"]

        # Get lddt metrics
        all_lddt_dict, all_total_dict = self.get_lddt_metrics(
            model,
            batch,
            out,
            idx_dataset,
            n_samples,
            true_coords_resolved_mask,
            true_coords,
            expand_to_diffusion_samples,
        )

        # Get diversity lddt metrics
        all_lddt_dict_div, all_total_dict_div = None, None
        if n_samples > 1:
            all_lddt_dict_div, all_total_dict_div = self.get_diversity_lddt_metrics(
                model,
                batch,
                out,
                idx_dataset,
                n_samples,
                true_coords_resolved_mask,
                expand_to_diffusion_samples,
            )

        # Get physical realism metrics
        if self.physicalism_metrics:
            pair_clash_dict, pair_total_dict = self.get_clash_metrics(
                batch,
                out,
            )
            pb_failure_dict, pb_total_dict = self.get_pb_metrics(
                batch,
                out,
            )
        else:
            pair_clash_dict, pair_total_dict = None, None
            pb_failure_dict, pb_total_dict = None, None

        # Compute best lddt metrics based on oracle lddt and
        # average across conformers
        (
            best_lddt_dict,
            best_total_dict,
            best_complex_lddt_dict,
            best_complex_total_dict,
        ) = self.compute_best_lddt_metrics(
            model,
            batch,
            all_lddt_dict,
            all_total_dict,
            all_lddt_dict_div,
            all_total_dict_div,
            n_samples,
        )
        # Filtering based on confidence
        if model.confidence_prediction and n_samples > 1:
            (
                mae_plddt_dicts,
                total_mae_plddt_dicts,
                mae_pde_dicts,
                total_mae_pde_dicts,
                mae_pae_dicts,
                total_mae_pae_dicts,
            ) = self.get_confidence_metrics(
                model,
                batch,
                out,
                idx_dataset,
                n_samples,
                true_coords,
                true_coords_resolved_mask,
                expand_to_diffusion_samples,
            )

        # Update distogram loss
        self.folding_metrics["disto_loss"][idx_dataset]["disto_loss"].update(
            val_disto_loss
        )

        # Update folding metrics
        self.update_lddt_rmsd_metrics(
            batch,
            disto_lddt_dict,
            disto_total_dict,
            best_lddt_dict,
            best_total_dict,
            best_complex_lddt_dict,
            best_complex_total_dict,
            # rmsds,
            idx_dataset,
            return_dict,
        )

        # Update physcial realism metrics
        if self.physicalism_metrics:
            self.update_physcialism_metrics(
                pair_clash_dict,
                pair_total_dict,
                pb_failure_dict,
                pb_total_dict,
                idx_dataset,
            )

        # Update confidence metrics
        if model.confidence_prediction and n_samples > 1:
            self.update_confidence_metrics(
                batch,
                out,
                idx_dataset,
                n_samples,
                all_lddt_dict,
                all_total_dict,
                mae_plddt_dicts,
                total_mae_plddt_dicts,
                mae_pde_dicts,
                total_mae_pde_dicts,
                mae_pae_dicts,
                total_mae_pae_dicts,
                pair_clash_dict,
                pair_total_dict,
                pb_failure_dict,
                pb_total_dict,
                physicalism_metrics=self.physicalism_metrics,
            )

    def on_epoch_end(self, model: LightningModule):
        raise NotImplementedError

    def common_on_epoch_end(self, model: LightningModule):
        avg_lddt = [{} for _ in range(self.num_val_datasets)]
        avg_disto_lddt = [{} for _ in range(self.num_val_datasets)]
        avg_complex_lddt = [{} for _ in range(self.num_val_datasets)]
        avg_clash = [{} for _ in range(self.num_val_datasets)]
        avg_pb = [{} for _ in range(self.num_val_datasets)]

        if model.confidence_prediction:
            avg_top1_lddt = [{} for _ in range(self.num_val_datasets)]
            avg_iplddt_top1_lddt = [{} for _ in range(self.num_val_datasets)]
            avg_pde_top1_lddt = [{} for _ in range(self.num_val_datasets)]
            avg_ipde_top1_lddt = [{} for _ in range(self.num_val_datasets)]
            avg_ptm_top1_lddt = [{} for _ in range(self.num_val_datasets)]
            avg_iptm_top1_lddt = [{} for _ in range(self.num_val_datasets)]
            avg_ligand_iptm_top1_lddt = [{} for _ in range(self.num_val_datasets)]
            avg_protein_iptm_top1_lddt = [{} for _ in range(self.num_val_datasets)]

            avg_avg_lddt = [{} for _ in range(self.num_val_datasets)]
            avg_mae_plddt = [{} for _ in range(self.num_val_datasets)]
            avg_mae_pde = [{} for _ in range(self.num_val_datasets)]
            avg_mae_pae = [{} for _ in range(self.num_val_datasets)]
            avg_avg_clash = [{} for _ in range(self.num_val_datasets)]
            avg_avg_pb = [{} for _ in range(self.num_val_datasets)]

            avg_top1_clash = [{} for _ in range(self.num_val_datasets)]
            avg_iplddt_top1_clash = [{} for _ in range(self.num_val_datasets)]
            avg_pde_top1_clash = [{} for _ in range(self.num_val_datasets)]
            avg_ipde_top1_clash = [{} for _ in range(self.num_val_datasets)]
            avg_ptm_top1_clash = [{} for _ in range(self.num_val_datasets)]
            avg_iptm_top1_clash = [{} for _ in range(self.num_val_datasets)]
            avg_ligand_iptm_top1_clash = [{} for _ in range(self.num_val_datasets)]
            avg_protein_iptm_top1_clash = [{} for _ in range(self.num_val_datasets)]

            avg_top1_pb = [{} for _ in range(self.num_val_datasets)]
            avg_iplddt_top1_pb = [{} for _ in range(self.num_val_datasets)]
            avg_pde_top1_pb = [{} for _ in range(self.num_val_datasets)]
            avg_ipde_top1_pb = [{} for _ in range(self.num_val_datasets)]
            avg_ptm_top1_pb = [{} for _ in range(self.num_val_datasets)]
            avg_iptm_top1_pb = [{} for _ in range(self.num_val_datasets)]
            avg_ligand_iptm_top1_pb = [{} for _ in range(self.num_val_datasets)]
            avg_protein_iptm_top1_pb = [{} for _ in range(self.num_val_datasets)]

        for idx_dataset in range(self.num_val_datasets):  # local idx_dataset
            dataset_name_ori = self.val_names[
                idx_dataset
            ]  # self.val_group_mapper[idx_dataset]["label"]

            # TODO this is harcodeded for now to compare with Boltz-1 metrics
            dataset_name = "" if dataset_name_ori == "RCSB" else f"__{dataset_name_ori}"

            for m_ in [
                *const.out_types,
                "pocket_ligand_protein",
                "contact_protein_protein",
            ]:
                avg_disto_lddt[idx_dataset][m_] = self.folding_metrics["disto_lddt"][
                    idx_dataset
                ][m_].compute()

                avg_disto_lddt[idx_dataset][m_] = (
                    0.0
                    if torch.isnan(avg_disto_lddt[idx_dataset][m_])
                    else avg_disto_lddt[idx_dataset][m_].item()
                )
                self.folding_metrics["disto_lddt"][idx_dataset][m_].reset()
                model.log(
                    f"val/disto_lddt_{m_}{dataset_name}",
                    avg_disto_lddt[idx_dataset][m_],
                )

                # TODO: this is hardcoded for now to RCSB compare with Boltz-1 metrics
                # RCSB recall == preicion
                suffixes = (
                    [":recall", ":diversity"]
                    if dataset_name_ori == "RCSB"
                    else [":recall", ":precision", ":diversity"]
                )
                for suffix in suffixes:
                    m = m_ + suffix

                    # TODO: this is hardcoded for now to RCSB compare with Boltz-1
                    # metrics
                    if dataset_name_ori == "RCSB" and suffix == ":recall":
                        # We want recall to not show suffix when logging for RCSB
                        m_label = m_
                    else:
                        # Append suffix otherwise
                        m_label = m

                    # TODO remove this
                    # if self.folding_metrics["lddt"][idx_dataset][m].weight > 1:
                    avg_lddt[idx_dataset][m] = self.folding_metrics["lddt"][
                        idx_dataset
                    ][m].compute()
                    avg_lddt[idx_dataset][m] = (
                        0.0
                        if torch.isnan(avg_lddt[idx_dataset][m])
                        else avg_lddt[idx_dataset][m].item()
                    )
                    self.folding_metrics["lddt"][idx_dataset][m].reset()
                    model.log(
                        f"val/lddt_{m_label}{dataset_name}",
                        avg_lddt[idx_dataset][m],
                    )

                    avg_complex_lddt[idx_dataset][m] = self.folding_metrics[
                        "complex_lddt"
                    ][idx_dataset][m].compute()
                    avg_complex_lddt[idx_dataset][m] = (
                        0.0
                        if torch.isnan(avg_complex_lddt[idx_dataset][m])
                        else avg_complex_lddt[idx_dataset][m].item()
                    )
                    self.folding_metrics["complex_lddt"][idx_dataset][m].reset()
                    model.log(
                        f"val/complex_lddt_{m_label}{dataset_name}",
                        avg_complex_lddt[idx_dataset][m],
                    )

                if model.confidence_prediction:
                    avg_top1_lddt[idx_dataset][m_] = self.confidence_metrics[
                        "top1_lddt"
                    ][idx_dataset][m_].compute()
                    avg_top1_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_top1_lddt[idx_dataset][m_])
                        else avg_top1_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["top1_lddt"][idx_dataset][m_].reset()
                    model.log(
                        f"val/top1_lddt_{m_}{dataset_name}",
                        avg_top1_lddt[idx_dataset][m_],
                    )

                    avg_iplddt_top1_lddt[idx_dataset][m_] = self.confidence_metrics[
                        "iplddt_top1_lddt"
                    ][idx_dataset][m_].compute()
                    avg_iplddt_top1_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_iplddt_top1_lddt[idx_dataset][m_])
                        else avg_iplddt_top1_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["iplddt_top1_lddt"][idx_dataset][m_].reset()
                    model.log(
                        f"val/iplddt_top1_lddt_{m_}{dataset_name}",
                        avg_iplddt_top1_lddt[idx_dataset][m_],
                    )

                    avg_pde_top1_lddt[idx_dataset][m_] = self.confidence_metrics[
                        "pde_top1_lddt"
                    ][idx_dataset][m_].compute()
                    avg_pde_top1_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_pde_top1_lddt[idx_dataset][m_])
                        else avg_pde_top1_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["pde_top1_lddt"][idx_dataset][m_].reset()
                    model.log(
                        f"val/pde_top1_lddt_{m_}{dataset_name}",
                        avg_pde_top1_lddt[idx_dataset][m_],
                    )

                    avg_ipde_top1_lddt[idx_dataset][m_] = self.confidence_metrics[
                        "ipde_top1_lddt"
                    ][idx_dataset][m_].compute()
                    avg_ipde_top1_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_ipde_top1_lddt[idx_dataset][m_])
                        else avg_ipde_top1_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["ipde_top1_lddt"][idx_dataset][m_].reset()
                    model.log(
                        f"val/ipde_top1_lddt_{m_}{dataset_name}",
                        avg_ipde_top1_lddt[idx_dataset][m_],
                    )

                    avg_ptm_top1_lddt[idx_dataset][m_] = self.confidence_metrics[
                        "ptm_top1_lddt"
                    ][idx_dataset][m_].compute()
                    avg_ptm_top1_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_ptm_top1_lddt[idx_dataset][m_])
                        else avg_ptm_top1_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["ptm_top1_lddt"][idx_dataset][m_].reset()
                    model.log(
                        f"val/ptm_top1_lddt_{m_}{dataset_name}",
                        avg_ptm_top1_lddt[idx_dataset][m_],
                    )

                    avg_iptm_top1_lddt[idx_dataset][m_] = self.confidence_metrics[
                        "iptm_top1_lddt"
                    ][idx_dataset][m_].compute()
                    avg_iptm_top1_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_iptm_top1_lddt[idx_dataset][m_])
                        else avg_iptm_top1_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["iptm_top1_lddt"][idx_dataset][m_].reset()
                    model.log(
                        f"val/iptm_top1_lddt_{m_}{dataset_name}",
                        avg_iptm_top1_lddt[idx_dataset][m_],
                    )

                    avg_ligand_iptm_top1_lddt[idx_dataset][m_] = (
                        self.confidence_metrics["ligand_iptm_top1_lddt"][idx_dataset][
                            m_
                        ].compute()
                    )
                    avg_ligand_iptm_top1_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_ligand_iptm_top1_lddt[idx_dataset][m_])
                        else avg_ligand_iptm_top1_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["ligand_iptm_top1_lddt"][idx_dataset][
                        m_
                    ].reset()
                    model.log(
                        f"val/ligand_iptm_top1_lddt_{m_}{dataset_name}",
                        avg_ligand_iptm_top1_lddt[idx_dataset][m_],
                    )

                    avg_protein_iptm_top1_lddt[idx_dataset][m_] = (
                        self.confidence_metrics["protein_iptm_top1_lddt"][idx_dataset][
                            m_
                        ].compute()
                    )
                    avg_protein_iptm_top1_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_protein_iptm_top1_lddt[idx_dataset][m_])
                        else avg_protein_iptm_top1_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["protein_iptm_top1_lddt"][idx_dataset][
                        m_
                    ].reset()
                    model.log(
                        f"val/protein_iptm_top1_lddt_{m_}{dataset_name}",
                        avg_protein_iptm_top1_lddt[idx_dataset][m_],
                    )

                    avg_avg_lddt[idx_dataset][m_] = self.confidence_metrics["avg_lddt"][
                        idx_dataset
                    ][m_].compute()
                    avg_avg_lddt[idx_dataset][m_] = (
                        0.0
                        if torch.isnan(avg_avg_lddt[idx_dataset][m_])
                        else avg_avg_lddt[idx_dataset][m_].item()
                    )
                    self.confidence_metrics["avg_lddt"][idx_dataset][m_].reset()
                    model.log(
                        f"val/avg_lddt_{m_}{dataset_name}",
                        avg_avg_lddt[idx_dataset][m_],
                    )

                    avg_mae_pde[idx_dataset][m_] = (
                        self.confidence_metrics["pde_mae"][idx_dataset][m_]
                        .compute()
                        .item()
                    )
                    self.confidence_metrics["pde_mae"][idx_dataset][m_].reset()
                    model.log(
                        f"val/MAE_pde_{m_}{dataset_name}",
                        avg_mae_pde[idx_dataset][m_],
                    )

                    avg_mae_pae[idx_dataset][m_] = (
                        self.confidence_metrics["pae_mae"][idx_dataset][m_]
                        .compute()
                        .item()
                    )
                    self.confidence_metrics["pae_mae"][idx_dataset][m_].reset()
                    model.log(
                        f"val/MAE_pae_{m_}{dataset_name}",
                        avg_mae_pae[idx_dataset][m_],
                    )

            for m in const.out_single_types:
                if model.confidence_prediction:
                    avg_mae_plddt[idx_dataset][m] = (
                        self.confidence_metrics["plddt_mae"][idx_dataset][m]
                        .compute()
                        .item()
                    )
                    self.confidence_metrics["plddt_mae"][idx_dataset][m].reset()
                    model.log(
                        f"val/MAE_plddt_{m}{dataset_name}",
                        avg_mae_plddt[idx_dataset][m],
                    )

            overall_disto_lddt = sum(
                avg_disto_lddt[idx_dataset][m] * w
                for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            model.log(
                f"val/disto_lddt{dataset_name}",
                overall_disto_lddt,
            )

            # TODO: this is hardcoded for now to RCSB compare with Boltz-1 metrics
            # RCSB recall == preicion and diversity is not defined.
            suffixes = (
                [":recall", ":diversity"]
                if dataset_name_ori == "RCSB"
                else [":recall", ":precision", ":diversity"]
            )

            for suffix in suffixes:
                # TODO: this is hardcoded for now to RCSB compare with Boltz-1
                # metrics
                if dataset_name_ori == "RCSB" and suffix == ":recall":
                    # We want recall to not show suffix when logging for RCSB
                    suffix_label = ""
                else:
                    # Append suffix otherwise
                    suffix_label = suffix

                # TODO REMOVE DEBUG HERE
                # if m + suffix in avg_lddt[idx_dataset]:
                overall_lddt = sum(
                    avg_lddt[idx_dataset][m + suffix] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/lddt{suffix_label}{dataset_name}",
                    overall_lddt,
                )

                overall_complex_lddt = sum(
                    avg_complex_lddt[idx_dataset][m + suffix] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/complex_lddt{suffix_label}{dataset_name}",
                    overall_complex_lddt,
                )

            if model.confidence_prediction:
                overall_top1_lddt = sum(
                    avg_top1_lddt[idx_dataset][m] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/top1_lddt{dataset_name}",
                    overall_top1_lddt,
                )

                overall_iplddt_top1_lddt = sum(
                    avg_iplddt_top1_lddt[idx_dataset][m] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/iplddt_top1_lddt{dataset_name}",
                    overall_iplddt_top1_lddt,
                )

                overall_pde_top1_lddt = sum(
                    avg_pde_top1_lddt[idx_dataset][m] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/pde_top1_lddt{dataset_name}",
                    overall_pde_top1_lddt,
                )

                overall_ipde_top1_lddt = sum(
                    avg_ipde_top1_lddt[idx_dataset][m] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/ipde_top1_lddt{dataset_name}",
                    overall_ipde_top1_lddt,
                )

                overall_ptm_top1_lddt = sum(
                    avg_ptm_top1_lddt[idx_dataset][m] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/ptm_top1_lddt{dataset_name}",
                    overall_ptm_top1_lddt,
                )

                overall_iptm_top1_lddt = sum(
                    avg_iptm_top1_lddt[idx_dataset][m] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/iptm_top1_lddt{dataset_name}",
                    overall_iptm_top1_lddt,
                )

                overall_avg_lddt = sum(
                    avg_avg_lddt[idx_dataset][m] * w
                    for (m, w) in const.out_types_weights.items()
                ) / sum(const.out_types_weights.values())
                model.log(
                    f"val/avg_lddt{dataset_name}",
                    overall_avg_lddt,
                )

            # RMSD metrics
            # model.log(
            #    f"val/rmsd{dataset_name}",
            #    self.folding_metrics["rmsd"][idx_dataset]["rmsd"].compute(),
            # )
            # self.folding_metrics["rmsd"][idx_dataset]["rmsd"].reset()

            # if self.folding_metrics["rmsd"][idx_dataset]["best_rmsd:recall"].weight > 0:
            #    model.log(
            #        f"val/best_rmsd_recall{dataset_name}",
            #        self.folding_metrics["rmsd"][idx_dataset][
            #            "best_rmsd:recall"
            #        ].compute(),
            #    )
            #    self.folding_metrics["rmsd"][idx_dataset]["best_rmsd:recall"].reset()

            # if (
            #    self.folding_metrics["rmsd"][idx_dataset]["best_rmsd:precision"].weight
            #    > 0
            # ):
            #    model.log(
            #        f"val/best_rmsd_precision{dataset_name}",
            #        self.folding_metrics["rmsd"][idx_dataset][
            #            "best_rmsd:precision"
            #        ].compute(),
            #    )
            #    self.folding_metrics["rmsd"][idx_dataset]["best_rmsd:precision"].reset()

            # if self.folding_metrics["rmsd"][idx_dataset]["rmsd:diversity"].weight > 0:
            #   model.log(
            #        f"val/rmsd_diversity{dataset_name}",
            #        self.folding_metrics["rmsd"][idx_dataset][
            #            "rmsd:diversity"
            #        ].compute(),
            #    )
            #    self.folding_metrics["rmsd"][idx_dataset]["rmsd:diversity"].reset()

            # if self.folding_metrics["rmsd"][idx_dataset]["rmsf_score"].weight > 0:
            #    model.log(
            #        f"val/rmsf_score{dataset_name}",
            #        self.folding_metrics["rmsd"][idx_dataset]["rmsf_score"].compute(),
            #    )
            #    self.folding_metrics["rmsd"][idx_dataset]["rmsf_score"].reset()

            # Distogram loss
            r = self.folding_metrics["disto_loss"][idx_dataset]["disto_loss"].compute()
            model.log(f"val/disto_loss{dataset_name}", r)
            self.folding_metrics["disto_loss"][idx_dataset]["disto_loss"].reset()

            # Physical realism metrics
            if self.physicalism_metrics:
                for m in ["asym_" + m_ for m_ in const.clash_types] + [
                    "sym_" + m_ for m_ in const.out_single_types
                ]:
                    avg_clash[idx_dataset][m] = self.physicalism_metrics["clash"][
                        idx_dataset
                    ][m].compute()
                    avg_clash[idx_dataset][m] = (
                        0.0
                        if torch.isnan(avg_clash[idx_dataset][m])
                        else avg_clash[idx_dataset][m].item()
                    )
                    self.physicalism_metrics["clash"][idx_dataset][m].reset()
                    model.log(
                        f"val/clash_{m}{dataset_name}",
                        avg_clash[idx_dataset][m],
                    )

                    if model.confidence_prediction:
                        avg_top1_clash[idx_dataset][m] = self.confidence_metrics[
                            "top1_clash"
                        ][idx_dataset][m].compute()
                        avg_top1_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_top1_clash[idx_dataset][m])
                            else avg_top1_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["top1_clash"][idx_dataset][m].reset()
                        model.log(
                            f"val/top1_clash_{m}{dataset_name}",
                            avg_top1_clash[idx_dataset][m],
                        )

                        avg_iplddt_top1_clash[idx_dataset][m] = self.confidence_metrics[
                            "iplddt_top1_clash"
                        ][idx_dataset][m].compute()
                        avg_iplddt_top1_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_iplddt_top1_clash[idx_dataset][m])
                            else avg_iplddt_top1_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["iplddt_top1_clash"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/iplddt_top1_clash_{m}{dataset_name}",
                            avg_iplddt_top1_clash[idx_dataset][m],
                        )

                        avg_pde_top1_clash[idx_dataset][m] = self.confidence_metrics[
                            "pde_top1_clash"
                        ][idx_dataset][m].compute()
                        avg_pde_top1_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_pde_top1_clash[idx_dataset][m])
                            else avg_pde_top1_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["pde_top1_clash"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/pde_top1_clash_{m}{dataset_name}",
                            avg_pde_top1_clash[idx_dataset][m],
                        )

                        avg_ipde_top1_clash[idx_dataset][m] = self.confidence_metrics[
                            "ipde_top1_clash"
                        ][idx_dataset][m].compute()
                        avg_ipde_top1_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_ipde_top1_clash[idx_dataset][m])
                            else avg_ipde_top1_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["ipde_top1_clash"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/ipde_top1_clash_{m}{dataset_name}",
                            avg_ipde_top1_clash[idx_dataset][m],
                        )

                        avg_ptm_top1_clash[idx_dataset][m] = self.confidence_metrics[
                            "ptm_top1_clash"
                        ][idx_dataset][m].compute()
                        avg_ptm_top1_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_ptm_top1_clash[idx_dataset][m])
                            else avg_ptm_top1_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["ptm_top1_clash"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/ptm_top1_clash_{m}{dataset_name}",
                            avg_ptm_top1_clash[idx_dataset][m],
                        )

                        avg_iptm_top1_clash[idx_dataset][m] = self.confidence_metrics[
                            "iptm_top1_clash"
                        ][idx_dataset][m].compute()
                        avg_iptm_top1_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_iptm_top1_clash[idx_dataset][m])
                            else avg_iptm_top1_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["iptm_top1_clash"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/iptm_top1_clash_{m}{dataset_name}",
                            avg_iptm_top1_clash[idx_dataset][m],
                        )

                        avg_ligand_iptm_top1_clash[idx_dataset][m] = (
                            self.confidence_metrics["ligand_iptm_top1_clash"][
                                idx_dataset
                            ][m].compute()
                        )
                        avg_ligand_iptm_top1_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_ligand_iptm_top1_clash[idx_dataset][m])
                            else avg_ligand_iptm_top1_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["ligand_iptm_top1_clash"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/ligand_iptm_top1_clash_{m}{dataset_name}",
                            avg_ligand_iptm_top1_clash[idx_dataset][m],
                        )

                        avg_protein_iptm_top1_clash[idx_dataset][m] = (
                            self.confidence_metrics["protein_iptm_top1_clash"][
                                idx_dataset
                            ][m].compute()
                        )
                        avg_protein_iptm_top1_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_protein_iptm_top1_clash[idx_dataset][m])
                            else avg_protein_iptm_top1_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["protein_iptm_top1_clash"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/protein_iptm_top1_clash_{m}{dataset_name}",
                            avg_protein_iptm_top1_clash[idx_dataset][m],
                        )

                        avg_avg_clash[idx_dataset][m] = self.confidence_metrics[
                            "avg_clash"
                        ][idx_dataset][m].compute()
                        avg_avg_clash[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_avg_clash[idx_dataset][m])
                            else avg_avg_clash[idx_dataset][m].item()
                        )
                        self.confidence_metrics["avg_clash"][idx_dataset][m].reset()
                        model.log(
                            f"val/avg_clash_{m}{dataset_name}",
                            avg_avg_clash[idx_dataset][m],
                        )

                for m in [
                    "bond_length",
                    "bond_angle",
                    "internal_clash",
                    "atom_chirality",
                    "bond_stereochemistry",
                    "ring_5_flatness",
                    "ring_6_flatness",
                    "double_bond_flatness",
                ]:
                    avg_pb[idx_dataset][m] = self.physicalism_metrics["pb"][
                        idx_dataset
                    ][m].compute()
                    avg_pb[idx_dataset][m] = (
                        0.0
                        if torch.isnan(avg_pb[idx_dataset][m])
                        else avg_pb[idx_dataset][m].item()
                    )
                    self.physicalism_metrics["pb"][idx_dataset][m].reset()
                    model.log(
                        f"val/pb_{m}{dataset_name}",
                        avg_pb[idx_dataset][m],
                    )

                    if model.confidence_prediction:
                        avg_top1_pb[idx_dataset][m] = self.confidence_metrics[
                            "top1_pb"
                        ][idx_dataset][m].compute()
                        avg_top1_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_top1_pb[idx_dataset][m])
                            else avg_top1_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["top1_pb"][idx_dataset][m].reset()
                        model.log(
                            f"val/top1_pb_{m}{dataset_name}",
                            avg_top1_pb[idx_dataset][m],
                        )

                        avg_iplddt_top1_pb[idx_dataset][m] = self.confidence_metrics[
                            "iplddt_top1_pb"
                        ][idx_dataset][m].compute()
                        avg_iplddt_top1_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_iplddt_top1_pb[idx_dataset][m])
                            else avg_iplddt_top1_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["iplddt_top1_pb"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/iplddt_top1_pb_{m}{dataset_name}",
                            avg_iplddt_top1_pb[idx_dataset][m],
                        )

                        avg_pde_top1_pb[idx_dataset][m] = self.confidence_metrics[
                            "pde_top1_pb"
                        ][idx_dataset][m].compute()
                        avg_pde_top1_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_pde_top1_pb[idx_dataset][m])
                            else avg_pde_top1_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["pde_top1_pb"][idx_dataset][m].reset()
                        model.log(
                            f"val/pde_top1_pb_{m}{dataset_name}",
                            avg_pde_top1_pb[idx_dataset][m],
                        )

                        avg_ipde_top1_pb[idx_dataset][m] = self.confidence_metrics[
                            "ipde_top1_pb"
                        ][idx_dataset][m].compute()
                        avg_ipde_top1_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_ipde_top1_pb[idx_dataset][m])
                            else avg_ipde_top1_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["ipde_top1_pb"][idx_dataset][m].reset()
                        model.log(
                            f"val/ipde_top1_pb_{m}{dataset_name}",
                            avg_ipde_top1_pb[idx_dataset][m],
                        )

                        avg_ptm_top1_pb[idx_dataset][m] = self.confidence_metrics[
                            "ptm_top1_pb"
                        ][idx_dataset][m].compute()
                        avg_ptm_top1_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_ptm_top1_pb[idx_dataset][m])
                            else avg_ptm_top1_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["ptm_top1_pb"][idx_dataset][m].reset()
                        model.log(
                            f"val/ptm_top1_pb_{m}{dataset_name}",
                            avg_ptm_top1_pb[idx_dataset][m],
                        )

                        avg_iptm_top1_pb[idx_dataset][m] = self.confidence_metrics[
                            "iptm_top1_pb"
                        ][idx_dataset][m].compute()
                        avg_iptm_top1_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_iptm_top1_pb[idx_dataset][m])
                            else avg_iptm_top1_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["iptm_top1_pb"][idx_dataset][m].reset()
                        model.log(
                            f"val/iptm_top1_pb_{m}{dataset_name}",
                            avg_iptm_top1_pb[idx_dataset][m],
                        )

                        avg_ligand_iptm_top1_pb[idx_dataset][m] = (
                            self.confidence_metrics["ligand_iptm_top1_pb"][idx_dataset][
                                m
                            ].compute()
                        )
                        avg_ligand_iptm_top1_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_ligand_iptm_top1_pb[idx_dataset][m])
                            else avg_ligand_iptm_top1_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["ligand_iptm_top1_pb"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/ligand_iptm_top1_pb_{m}{dataset_name}",
                            avg_ligand_iptm_top1_pb[idx_dataset][m],
                        )

                        avg_protein_iptm_top1_pb[idx_dataset][m] = (
                            self.confidence_metrics["protein_iptm_top1_pb"][
                                idx_dataset
                            ][m].compute()
                        )
                        avg_protein_iptm_top1_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_protein_iptm_top1_pb[idx_dataset][m])
                            else avg_protein_iptm_top1_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["protein_iptm_top1_pb"][idx_dataset][
                            m
                        ].reset()
                        model.log(
                            f"val/protein_iptm_top1_pb_{m}{dataset_name}",
                            avg_protein_iptm_top1_pb[idx_dataset][m],
                        )

                        avg_avg_pb[idx_dataset][m] = self.confidence_metrics["avg_pb"][
                            idx_dataset
                        ][m].compute()
                        avg_avg_pb[idx_dataset][m] = (
                            0.0
                            if torch.isnan(avg_avg_pb[idx_dataset][m])
                            else avg_avg_pb[idx_dataset][m].item()
                        )
                        self.confidence_metrics["avg_pb"][idx_dataset][m].reset()
                        model.log(
                            f"val/avg_pb_{m}{dataset_name}",
                            avg_avg_pb[idx_dataset][m],
                        )
