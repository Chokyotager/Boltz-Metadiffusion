import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import Coords, Interface, Record, Structure, StructureV2
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        boltz2: bool = False,
        write_embeddings: bool = False,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_embeddings = write_embeddings

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        # Get ranking
        if "confidence_score" in prediction:
            argsort = torch.argsort(prediction["confidence_score"], descending=True)
            idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
        # Handles cases where confidence summary is False
        else:
            idx_to_rank = {i: i for i in range(len(records))}

        # Iterate over the records
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                structure: StructureV2 = StructureV2.load(path)
            else:
                structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    structure: StructureV2
                    coord_unpad = [(x,) for x in coord_unpad]
                    coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                if self.boltz2:
                    new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )
                else:
                    new_structure: Structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                if self.boltz2 and record.affinity and idx_to_rank[model_idx] == 0:
                    path = struct_dir / f"pre_affinity_{record.id}.npz"
                    np.savez_compressed(path, **asdict(new_structure))
                    np.array(atoms["coords"][:, None], dtype=Coords)

                # Save confidence summary
                if "plddt" in prediction:
                    path = (
                        struct_dir
                        / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    )
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = (
                        struct_dir
                        / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                        struct_dir
                        / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                        struct_dir
                        / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())

            # Save SAXS P(r) results if present
            if "saxs_pr_results" in prediction and prediction["saxs_pr_results"]:
                saxs_dir = struct_dir / "saxs"
                saxs_dir.mkdir(exist_ok=True)

                saxs_data = prediction["saxs_pr_results"]

                # Save NumPy data
                npz_path = saxs_dir / f"saxs_pr_fit_{record.id}.npz"
                np.savez_compressed(
                    npz_path,
                    r_grid=saxs_data['r_grid'],
                    pr_exp=saxs_data['pr_exp'],
                    pr_calc=saxs_data['pr_calc'],
                    rg_exp=saxs_data['rg_exp'],
                    rg_calc=saxs_data['rg_calc'],
                    w1_loss=saxs_data['w1_loss'],
                    mse_loss=saxs_data['mse_loss'],
                )

                # Generate comparison plot
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # Non-interactive backend
                    import matplotlib.pyplot as plt

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                    # P(r) comparison plot
                    ax1.plot(saxs_data['r_grid'], saxs_data['pr_exp'], 'b-', linewidth=2, label='Experimental')
                    ax1.plot(saxs_data['r_grid'], saxs_data['pr_calc'], 'r--', linewidth=2, label='Calculated (ensemble)')
                    ax1.fill_between(saxs_data['r_grid'], saxs_data['pr_exp'], alpha=0.3, color='blue')
                    ax1.set_xlabel('r (Å)', fontsize=12)
                    ax1.set_ylabel('P(r)', fontsize=12)
                    ax1.set_title(f'SAXS P(r) Fit: {record.id}', fontsize=14)
                    ax1.legend(fontsize=10)
                    ax1.grid(True, alpha=0.3)

                    # Add Rg annotation
                    ax1.text(0.95, 0.95,
                             f"Rg_exp: {saxs_data['rg_exp']:.1f} Å\nRg_calc: {saxs_data['rg_calc']:.1f} Å",
                             transform=ax1.transAxes, fontsize=10,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    # Residual plot
                    residual = saxs_data['pr_calc'] - saxs_data['pr_exp']
                    ax2.plot(saxs_data['r_grid'], residual, 'g-', linewidth=1.5)
                    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
                    ax2.fill_between(saxs_data['r_grid'], residual, alpha=0.3, color='green')
                    ax2.set_xlabel('r (Å)', fontsize=12)
                    ax2.set_ylabel('Residual (calc - exp)', fontsize=12)
                    ax2.set_title('Residual Plot', fontsize=14)
                    ax2.grid(True, alpha=0.3)

                    # Add loss metrics annotation
                    ax2.text(0.95, 0.95,
                             f"W1 Loss: {saxs_data['w1_loss']:.4f}\nMSE Loss: {saxs_data['mse_loss']:.6f}",
                             transform=ax2.transAxes, fontsize=10,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    plt.tight_layout()

                    # Save figure
                    fig_path = saxs_dir / f"saxs_pr_fit_{record.id}.png"
                    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                except ImportError:
                    pass  # matplotlib not available, skip plot generation
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to generate SAXS plot: {e}")

            # Save chemical shift results if present
            if "cheshift_results" in prediction and prediction["cheshift_results"]:
                cheshift_dir = struct_dir / "cheshift"
                cheshift_dir.mkdir(exist_ok=True)

                cheshift_data = prediction["cheshift_results"]

                # Save JSON data
                json_path = cheshift_dir / f"cheshift_fit_{record.id}.json"
                with open(json_path, 'w') as f:
                    json.dump(cheshift_data, f, indent=2)

                # Generate comparison plot
                try:
                    import matplotlib
                    matplotlib.use('Agg')  # Non-interactive backend
                    import matplotlib.pyplot as plt

                    nuclei = cheshift_data.get('nuclei', {})
                    n_nuclei = len(nuclei)

                    if n_nuclei > 0:
                        fig, axes = plt.subplots(2, n_nuclei, figsize=(6 * n_nuclei, 10))
                        if n_nuclei == 1:
                            axes = axes.reshape(2, 1)

                        for col, (nucleus, data) in enumerate(nuclei.items()):
                            residue_nums = data['residue_nums']
                            exp_shifts = np.array(data['exp_shifts'])
                            calc_shifts = np.array(data['calc_shifts'])
                            calc_stds = np.array(data['calc_stds'])

                            # Top plot: Shifts vs residue number
                            ax1 = axes[0, col]
                            ax1.errorbar(residue_nums, calc_shifts, yerr=calc_stds,
                                        fmt='o-', color='red', markersize=4,
                                        linewidth=1, capsize=2, label='Calculated')
                            ax1.plot(residue_nums, exp_shifts, 's-', color='blue',
                                    markersize=4, linewidth=1, label='Experimental')
                            ax1.set_xlabel('Residue Number', fontsize=12)
                            ax1.set_ylabel(f'{nucleus} Chemical Shift (ppm)', fontsize=12)
                            ax1.set_title(f'{nucleus} Chemical Shifts: {record.id}', fontsize=14)
                            ax1.legend(fontsize=10)
                            ax1.grid(True, alpha=0.3)

                            # Add metrics annotation
                            ax1.text(0.95, 0.05,
                                    f"RMSD: {data['rmsd']:.2f} ppm\n"
                                    f"MAE: {data['mae']:.2f} ppm\n"
                                    f"R: {data['correlation']:.3f}",
                                    transform=ax1.transAxes, fontsize=10,
                                    verticalalignment='bottom', horizontalalignment='right',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                            # Bottom plot: Correlation plot
                            ax2 = axes[1, col]
                            ax2.errorbar(exp_shifts, calc_shifts, yerr=calc_stds,
                                        fmt='o', color='green', markersize=6,
                                        capsize=2, alpha=0.7)

                            # Add identity line
                            min_val = min(exp_shifts.min(), calc_shifts.min()) - 1
                            max_val = max(exp_shifts.max(), calc_shifts.max()) + 1
                            ax2.plot([min_val, max_val], [min_val, max_val],
                                    'k--', linewidth=1, label='y=x')

                            ax2.set_xlabel(f'Experimental {nucleus} (ppm)', fontsize=12)
                            ax2.set_ylabel(f'Calculated {nucleus} (ppm)', fontsize=12)
                            ax2.set_title(f'{nucleus} Correlation Plot', fontsize=14)
                            ax2.set_xlim(min_val, max_val)
                            ax2.set_ylim(min_val, max_val)
                            ax2.set_aspect('equal')
                            ax2.grid(True, alpha=0.3)
                            ax2.legend(fontsize=10)

                        plt.tight_layout()

                        # Save figure
                        fig_path = cheshift_dir / f"cheshift_fit_{record.id}.png"
                        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)

                except ImportError:
                    pass  # matplotlib not available, skip plot generation
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to generate chemical shift plot: {e}")

            # Save bias histories (hills, repulsion energies) and CV histories if present
            # All data is consolidated into a single JSON file
            has_biases = "bias_histories" in prediction and prediction["bias_histories"]
            has_cv_histories = "cv_histories" in prediction and prediction["cv_histories"]
            if has_biases or has_cv_histories:
                bias_json_path = struct_dir / f"bias_histories_{record.id}.json"
                bias_output = {
                    'record_id': record.id,
                }
                # Add sample-to-rank mapping so users can match cv_histories indices to model numbers
                # cv_histories stores values by internal sample index (0, 1, 2, ...)
                # Output files are named by rank (model_0 = highest confidence)
                # sample_to_model_rank[i] = rank means sample i became model_{rank}
                bias_output['sample_to_model_rank'] = idx_to_rank
                if has_biases:
                    bias_output['num_biases'] = len(prediction["bias_histories"])
                    bias_output['biases'] = prediction["bias_histories"]
                if has_cv_histories:
                    bias_output['cv_histories'] = prediction["cv_histories"]
                with open(bias_json_path, 'w') as f:
                    json.dump(bias_output, f, indent=2)

            # Save embeddings
            if self.write_embeddings and "s" in prediction and "z" in prediction:
                s = prediction["s"].cpu().numpy()
                z = prediction["z"].cpu().numpy()

                path = (
                    struct_dir
                    / f"embeddings_{record.id}.npz"
                )
                np.savez_compressed(path, s=s, z=z)

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201


class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return
        # Dump affinity summary
        affinity_summary = {}
        pred_affinity_value = prediction["affinity_pred_value"]
        pred_affinity_probability = prediction["affinity_probability_binary"]
        affinity_summary = {
            "affinity_pred_value": pred_affinity_value.item(),
            "affinity_probability_binary": pred_affinity_probability.item(),
        }
        if "affinity_pred_value1" in prediction:
            pred_affinity_value1 = prediction["affinity_pred_value1"]
            pred_affinity_probability1 = prediction["affinity_probability_binary1"]
            pred_affinity_value2 = prediction["affinity_pred_value2"]
            pred_affinity_probability2 = prediction["affinity_probability_binary2"]
            affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
            affinity_summary["affinity_probability_binary1"] = (
                pred_affinity_probability1.item()
            )
            affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
            affinity_summary["affinity_probability_binary2"] = (
                pred_affinity_probability2.item()
            )

        # Save the affinity summary
        struct_dir = self.output_dir / batch["record"][0].id
        struct_dir.mkdir(exist_ok=True)
        path = struct_dir / f"affinity_{batch['record'][0].id}.json"

        with path.open("w") as f:
            f.write(json.dumps(affinity_summary, indent=4))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
