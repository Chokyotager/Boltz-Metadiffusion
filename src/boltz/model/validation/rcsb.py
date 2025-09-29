from typing import Optional

import torch
from pytorch_lightning import LightningModule

from boltz.model.validation.validator import Validator


class RCSBValidator(Validator):
    """Validation step implementation for RCSB."""

    def __init__(
        self,
        val_names: list[str],
        confidence_prediction: bool = False,
        override_val_method: Optional[str] = None,
    ) -> None:
        super().__init__(
            val_names=val_names,
            confidence_prediction=confidence_prediction,
            override_val_method=override_val_method,
        )

    def process(
        self,
        model: LightningModule,
        batch: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        idx_dataset: int,
    ) -> None:
        """Compute features.

        Parameters
        ----------
        model : LightningModule
            The LightningModule model.
        batch : Dict[str, torch.Tensor]
            The batch input.
        out : Dict[str, torch.Tensor]
            The output of the model.

        """
        symmetry_correction = model.val_group_mapper[idx_dataset]["symmetry_correction"]
        expand_to_diffusion_samples = (
            symmetry_correction  # True # TODO Mateo why is this set to sym correction?
        )

        # For now all was dumped into the common operation in the parent Validator class
        self.common_val_step(
            model,
            batch,
            out,
            idx_dataset,
            expand_to_diffusion_samples=expand_to_diffusion_samples,
        )

        # TODO: Implement the RCSB specific validation step

    def on_epoch_end(self, model: LightningModule) -> None:
        # For now all was dumped into the common operation in the parent Validator class
        self.common_on_epoch_end(model)
