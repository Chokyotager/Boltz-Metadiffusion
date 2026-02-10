# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt
from typing import Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.loss.diffusionv2 import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encodersv2 import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SingleConditioning,
)
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    log,
)
from boltz.model.potentials import potentials as potentials_module
from boltz.model.potentials.potentials import get_potentials


def get_max_cutoff_from_steering_args(steering_args: dict, default_cutoff: float = 0.75) -> float:
    """Get the maximum cutoff value from all metadiffusion configs.

    Each potential can have its own warmup/cutoff. This returns the maximum
    cutoff so that guidance_enabled stays True until all potentials are done.
    """
    if steering_args is None:
        return default_cutoff

    max_cutoff = default_cutoff

    # Check opt_configs
    for config in (steering_args.get("opt_configs") or []):
        max_cutoff = max(max_cutoff, config.get("cutoff", 0.75))

    # Check steering_configs
    for config in (steering_args.get("steering_configs") or []):
        max_cutoff = max(max_cutoff, config.get("cutoff", 0.75))

    # Check explore_configs (and legacy bias_configs)
    for config in (steering_args.get("explore_configs") or steering_args.get("bias_configs") or []):
        max_cutoff = max(max_cutoff, config.get("cutoff", 0.75))

    # Check saxs_configs
    for config in (steering_args.get("saxs_configs") or []):
        max_cutoff = max(max_cutoff, config.get("cutoff", 0.9))

    # Check chemical_shift_configs
    for config in (steering_args.get("chemical_shift_configs") or []):
        max_cutoff = max(max_cutoff, config.get("cutoff", 0.9))

    return max_cutoff


class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            # post_layer_norm=transformer_post_ln,
        )

        self.a_norm = nn.LayerNorm(
            2 * token_s
        )  # if not transformer_post_ln else nn.Identity()

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            # transformer_post_layer_norm=transformer_post_ln,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing and self.training:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning[
                "token_trans_bias"
            ].float(),  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return r_update


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised_coords

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        show_progress=False,
        **network_condition_kwargs,
    ):
        # Initialize potentials to None - only set if steering is enabled
        potentials = None

        if steering_args is not None and (
            steering_args["fk_steering"]
            or steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
            or steering_args.get("rg_steering", False)
            or steering_args.get("saxs_pr_steering", False)
            or steering_args.get("metadynamics", False)
            or steering_args.get("explore_configs")
            or steering_args.get("steering_configs")
            or steering_args.get("opt_configs")
        ):
            potentials = get_potentials(
                steering_args, boltz2=True,
                feats=network_condition_kwargs.get("feats"),
                debug=steering_args.get("debug", False),
            )

        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
            or steering_args.get("rg_steering", False)
            or steering_args.get("saxs_pr_steering", False)
            or steering_args.get("metadynamics", False)
            or steering_args.get("explore_configs")
            or steering_args.get("steering_configs")
            or steering_args.get("opt_configs")
        ):
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        if self.training and self.step_scale_random is not None:
            step_scale = np.random.choice(self.step_scale_random)
        else:
            step_scale = self.step_scale

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        token_repr = None
        atom_coords_denoised = None

        # Initialize comprehensive CV tracking for ALL potentials
        # This tracks CV values at each step for any potential
        cv_histories = {}  # {potential_name: [{'step': int, 'cv_values': [...]}, ...]}

        # gradually denoise
        step_iterator = enumerate(sigmas_and_gammas)
        if show_progress:
            step_iterator = tqdm(
                step_iterator,
                total=len(sigmas_and_gammas),
                desc="Diffusion",
                leave=False,
            )
        for step_idx, (sigma_tm, sigma_t, gamma) in step_iterator:
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)

            # Apply random augmentation first
            atom_coords = (
                torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
            )

            if atom_coords_denoised is not None:
                atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
                atom_coords_denoised = (
                    torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R)
                    + random_tr
                )
            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
                or steering_args.get("rg_steering", False)
                or steering_args.get("saxs_pr_steering", False)
            ) and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)

            # Use noise_scale from YAML if provided, otherwise use Boltz default
            # Set to 0.0 for deterministic sampling
            effective_noise_scale = self.noise_scale  # Boltz default (1.003 for Boltz2)
            if steering_args and steering_args.get("noise_scale") is not None:
                effective_noise_scale = steering_args["noise_scale"]
            noise_var = effective_noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            # === GUIDANCE MODE ===
            # - "combine" (default): compute both pre and post gradients, add displacement vectors, apply to x_0
            # - "post": apply guidance to x_0 prediction after denoising only
            # - "pre": apply guidance to noisy coords before denoising only
            guidance_mode = steering_args.get("guidance_mode", "combine")
            # Get max cutoff from all configured potentials (respects per-potential cutoff settings)
            relaxation_cutoff = get_max_cutoff_from_steering_args(steering_args, default_cutoff=0.75)

            # Check if any guidance is enabled
            guidance_enabled = (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
                or steering_args.get("rg_steering", False)
                or steering_args.get("saxs_pr_steering", False)
                or steering_args.get("metadynamics", False)
                or steering_args.get("explore_configs")
                or steering_args.get("steering_configs")
                or steering_args.get("opt_configs")
            ) and step_idx < num_sampling_steps * relaxation_cutoff

            # === PRE-DENOISING GUIDANCE ===
            # Compute for "pre" mode (apply before model) or "combine" mode (save for later)
            pre_guidance_update = None
            if guidance_mode in ("pre", "combine") and guidance_enabled:
                pre_guidance_update = torch.zeros_like(atom_coords_noisy)
                min_bias_tempering = None  # Track minimum across all potentials

                for guidance_step in range(steering_args["num_gd_steps"]):
                    potentials_module.SAXSPrPotential.clear_cache(step_idx * 1000 + guidance_step)

                    energy_gradient = torch.zeros_like(atom_coords_noisy)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        parameters['_step_idx'] = step_idx
                        parameters['_relaxation'] = steering_t  # For warmup/cutoff
                        if (
                            parameters["guidance_weight"] > 0
                            and (guidance_step) % parameters["guidance_interval"] == 0
                        ):
                            pot_gradient = parameters["guidance_weight"] * potential.compute_gradient(
                                atom_coords_noisy + pre_guidance_update,
                                network_condition_kwargs["feats"],
                                parameters,
                            )

                            # Track minimum bias_tempering across potentials
                            pot_bias_tempering = parameters.get("bias_tempering")
                            if pot_bias_tempering is not None:
                                if min_bias_tempering is None:
                                    min_bias_tempering = pot_bias_tempering
                                else:
                                    min_bias_tempering = min(min_bias_tempering, pot_bias_tempering)

                            energy_gradient += pot_gradient
                    pre_guidance_update -= energy_gradient

                # Apply bias tempering to pre_guidance_update
                if min_bias_tempering is not None:
                    sigma_scale_bias = max(1.0, sigma_t / 100.0)
                    effective_bias_limit = min_bias_tempering * sigma_scale_bias
                    guidance_norms = pre_guidance_update.norm(dim=-1)
                    max_norms = guidance_norms.max(dim=-1, keepdim=True).values
                    scale = torch.where(
                        max_norms > effective_bias_limit,
                        effective_bias_limit / max_norms.clamp(min=1e-8),
                        torch.ones_like(max_norms)
                    )
                    pre_guidance_update = pre_guidance_update * scale.unsqueeze(-1)

                # For "pre" mode only: apply to noisy coords before denoising
                if guidance_mode == "pre":
                    # Apply total_bias_tempering (global limit on all potentials combined)
                    total_bias_tempering = steering_args.get("total_bias_tempering", None)
                    if total_bias_tempering is not None and total_bias_tempering > 0:
                        sigma_scale_total = max(1.0, sigma_t / 100.0)
                        effective_total_limit = total_bias_tempering * sigma_scale_total
                        guidance_norms = pre_guidance_update.norm(dim=-1)
                        max_norms = guidance_norms.max(dim=-1, keepdim=True).values
                        # Only scale if max_norms > 0 (avoid division issues with zero gradients)
                        needs_scaling = max_norms > effective_total_limit
                        if needs_scaling.any():
                            scale = torch.where(
                                needs_scaling,
                                effective_total_limit / max_norms.clamp(min=1e-8),
                                torch.ones_like(max_norms)
                            )
                            pre_guidance_update = pre_guidance_update * scale.unsqueeze(-1)

                    atom_coords_noisy = atom_coords_noisy + pre_guidance_update

                    # Track for scaled_guidance_update (used in FK resampling)
                    scaled_guidance_update = (
                        pre_guidance_update
                        * -1
                        * self.step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    )

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                num_chunks = (multiplicity + max_parallel_samples - 1) // max_parallel_samples
                sample_ids_chunks = sample_ids.chunk(num_chunks)

                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk = self.preconditioned_network_forward(
                        atom_coords_noisy[sample_ids_chunk],
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=sample_ids_chunk.numel(),
                            **network_condition_kwargs,
                        ),
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    # Compute energy of x_0 prediction
                    energy = torch.zeros(multiplicity, device=self.device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    # Compute log G values
                    if step_idx == 0:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    # Compute ll difference between guided and unguided transition distribution
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                        or steering_args.get("rg_steering", False)
                        or steering_args.get("saxs_pr_steering", False)
                    ) and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(energy)

                    # Compute resampling weights
                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )

                # === POST-DENOISING GUIDANCE ===
                # Compute for "post" mode or "combine" mode
                # For "combine": add pre_guidance_update to post_guidance_update
                if guidance_mode in ("post", "combine") and guidance_enabled:
                    guidance_update = torch.zeros_like(atom_coords_denoised)
                    min_bias_tempering = None  # Track minimum across all potentials

                    for guidance_step in range(steering_args["num_gd_steps"]):
                        # Clear SAXS P(r) cache at start of each guidance step
                        # Use composite key: step_idx * 1000 + guidance_step to uniquely identify
                        potentials_module.SAXSPrPotential.clear_cache(step_idx * 1000 + guidance_step)

                        energy_gradient = torch.zeros_like(atom_coords_denoised)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            # Add step index and relaxation for warmup/cutoff
                            parameters['_step_idx'] = step_idx
                            parameters['_relaxation'] = steering_t  # For warmup/cutoff
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"]
                                == 0
                            ):
                                # Compute this potential's gradient
                                pot_gradient = parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )

                                # Track minimum bias_tempering across potentials
                                pot_bias_tempering = parameters.get("bias_tempering")
                                if pot_bias_tempering is not None:
                                    if min_bias_tempering is None:
                                        min_bias_tempering = pot_bias_tempering
                                    else:
                                        min_bias_tempering = min(min_bias_tempering, pot_bias_tempering)

                                energy_gradient += pot_gradient
                        guidance_update -= energy_gradient

                    # Apply bias tempering to post guidance_update
                    if min_bias_tempering is not None:
                        sigma_scale_bias = max(1.0, sigma_t / 100.0)
                        effective_bias_limit = min_bias_tempering * sigma_scale_bias
                        guidance_norms = guidance_update.norm(dim=-1)
                        max_norms = guidance_norms.max(dim=-1, keepdim=True).values
                        scale = torch.where(
                            max_norms > effective_bias_limit,
                            effective_bias_limit / max_norms.clamp(min=1e-8),
                            torch.ones_like(max_norms)
                        )
                        guidance_update = guidance_update * scale.unsqueeze(-1)

                    # For "combine" mode: add pre_guidance_update to the displacement
                    if guidance_mode == "combine" and pre_guidance_update is not None:
                        guidance_update = guidance_update + pre_guidance_update

                    # Apply total_bias_tempering (global limit on all potentials combined)
                    total_bias_tempering = steering_args.get("total_bias_tempering", None)
                    if total_bias_tempering is not None and total_bias_tempering > 0:
                        sigma_scale_total = max(1.0, sigma_t / 100.0)
                        effective_total_limit = total_bias_tempering * sigma_scale_total
                        guidance_norms = guidance_update.norm(dim=-1)
                        max_norms = guidance_norms.max(dim=-1, keepdim=True).values
                        # Only scale if max_norms > 0 (avoid division issues with zero gradients)
                        needs_scaling = max_norms > effective_total_limit
                        if needs_scaling.any():
                            scale = torch.where(
                                needs_scaling,
                                effective_total_limit / max_norms.clamp(min=1e-8),
                                torch.ones_like(max_norms)
                            )
                            guidance_update = guidance_update * scale.unsqueeze(-1)

                    atom_coords_denoised += guidance_update

                    # Track CV values and deposit metadynamics hills after gradient descent
                    for potential in potentials:
                        # Get potential name for tracking
                        pot_name = getattr(potential, '_name', None) or getattr(potential, 'name', None) or type(potential).__name__

                        # Try to compute CV value for this potential
                        try:
                            if hasattr(potential, 'cv_function'):
                                # Metadynamics-style potentials with cv_function
                                cv_values, _ = potential.cv_function(
                                    atom_coords_denoised,
                                    network_condition_kwargs["feats"]
                                )
                                cv_list = cv_values.detach().cpu().tolist() if torch.is_tensor(cv_values) else [cv_values]
                            elif hasattr(potential, 'compute_variable'):
                                # Standard potentials with compute_variable
                                # Create dummy index (most potentials don't need it for CV computation)
                                dummy_index = torch.zeros(1, 1, dtype=torch.long, device=atom_coords_denoised.device)
                                cv_values = potential.compute_variable(
                                    atom_coords_denoised, dummy_index, compute_gradient=False
                                )
                                cv_list = cv_values.detach().cpu().tolist() if torch.is_tensor(cv_values) else [cv_values]
                            else:
                                cv_list = None

                            # Record CV values if we got them
                            if cv_list is not None:
                                if pot_name not in cv_histories:
                                    cv_histories[pot_name] = []
                                cv_histories[pot_name].append({
                                    'step': step_idx,
                                    'cv_values': cv_list if isinstance(cv_list, list) else [cv_list],
                                })
                        except Exception:
                            # Some potentials may not support CV computation this way
                            pass

                        # Deposit metadynamics hills if applicable
                        if hasattr(potential, 'deposit_hill'):
                            hill_interval = potential.parameters.get('hill_interval', 5)
                            if step_idx % hill_interval == 0:
                                # Compute CV value (average over samples)
                                cv_values, _ = potential.cv_function(
                                    atom_coords_denoised,
                                    network_condition_kwargs["feats"]
                                )
                                cv_mean = cv_values.mean().item()
                                potential.deposit_hill(cv_mean, step_idx)

                    scaled_guidance_update = (
                        guidance_update
                        * -1
                        * self.step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    )

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1]
                            if step_idx < num_sampling_steps - 1
                            else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    energy_traj = energy_traj[resample_indices]
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                        or steering_args.get("rg_steering", False)
                        or steering_args.get("saxs_pr_steering", False)
                        or steering_args.get("metadynamics", False)
                    ):
                        scaled_guidance_update = scaled_guidance_update[
                            resample_indices
                        ]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            # Enhanced debug: trace all stages at key steps
            # if step_idx % 50 == 0:
            #     noisy_max = atom_coords_noisy.abs().max().item()
            #     denoised_max = atom_coords_denoised.abs().max().item()
            #     diff_max = (atom_coords_noisy - atom_coords_denoised).abs().max().item()
            #     print(f"DEBUG step={step_idx}: sigma_t={sigma_t:.2f}, t_hat={t_hat:.2f}, step_scale={step_scale:.4f}", flush=True)
            #     print(f"DEBUG step={step_idx}: noisy_max={noisy_max:.1f}, denoised_max={denoised_max:.1f}, diff_max={diff_max:.1f}", flush=True)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            update = step_scale * (sigma_t - t_hat) * denoised_over_sigma

            # Apply denoising tempering if configured
            # Limits max per-atom displacement while preserving relative magnitudes
            # With reduced noise_scale, tempering should work without causing drift
            denoise_tempering = steering_args.get("denoise_tempering", None) if steering_args else None
            if denoise_tempering is not None:
                # Scale the tempering threshold by sigma_t to allow larger updates at high noise levels
                # At sigma_t=100, use base threshold; at sigma_t=1000, allow 10x larger updates
                sigma_scale = max(1.0, sigma_t / 100.0)
                effective_limit = denoise_tempering * sigma_scale

                # Compute per-atom displacement norms: [multiplicity, N_atoms]
                update_norms = update.norm(dim=-1)
                # Find max displacement across all atoms (per sample): [multiplicity, 1]
                max_norms = update_norms.max(dim=-1, keepdim=True).values
                # Scale factor: if max > limit, scale down uniformly
                scale = torch.where(
                    max_norms > effective_limit,
                    effective_limit / max_norms.clamp(min=1e-8),
                    torch.ones_like(max_norms)
                )
                # Apply uniform scaling (preserves relative magnitudes)
                update = update * scale.unsqueeze(-1)
                # if step_idx % 50 == 0:
                #     after_norms = update.norm(dim=-1)
                #     print(f"DEBUG denoise step={step_idx}: AFTER max_update_norm={after_norms.max().item():.4f}", flush=True)

            atom_coords_next = atom_coords_noisy + update

            # Debug: check coords range
            # if step_idx % 50 == 0:
            #     coord_max = atom_coords_next.abs().max().item()
            #     print(f"DEBUG step={step_idx}: coord_max={coord_max:.1f}", flush=True)

            atom_coords = atom_coords_next

        # Compute final SAXS P(r) fit if SAXS steering was used
        saxs_pr_results = None
        if steering_args is not None and steering_args.get("saxs_pr_steering", False):
            # Try new composable system (saxs_pr_data_cache) first, then legacy (saxs_pr_data)
            saxs_data = steering_args.get("saxs_pr_data")
            saxs_data_cache = steering_args.get("saxs_pr_data_cache")
            if saxs_data is None and saxs_data_cache:
                # Use first entry from cache (primary SAXS file)
                first_file = next(iter(saxs_data_cache))
                saxs_data = saxs_data_cache[first_file]
            if saxs_data is not None and potentials is not None:
                # Find the SAXSPrPotential
                from boltz.model.potentials.potentials import SAXSPrPotential
                saxs_potential = None
                for pot in potentials:
                    if isinstance(pot, SAXSPrPotential):
                        saxs_potential = pot
                        break

                if saxs_potential is not None:
                    # Compute final ensemble P(r) from final coordinates
                    r_grid = saxs_data['r_grid'].to(atom_coords.device)
                    pr_exp = saxs_data['pr_exp'].to(atom_coords.device)

                    # Initialize potential's internal state if not already set
                    # (needed when compute_variable is called directly without compute_args)
                    if not hasattr(saxs_potential, '_r_grid') or saxs_potential._r_grid is None:
                        saxs_potential._r_grid = r_grid
                        saxs_potential._sigma_bin = saxs_potential.parameters.get('sigma_bin', 0.5)
                        saxs_potential._per_sample_steering = False
                        saxs_potential._use_rep_atoms = saxs_potential.parameters.get('use_rep_atoms', False)
                        saxs_potential._ca_indices = None
                        saxs_potential._n_atoms = atom_coords.shape[1]

                    # Use the potential's compute_variable method
                    # Create dummy index for CA atoms (not used in atomistic mode)
                    dummy_index = torch.zeros(1, 1, dtype=torch.long, device=atom_coords.device)

                    # Compute ensemble P(r) from final coordinates (no gradient needed)
                    pr_calc = saxs_potential.compute_variable(
                        atom_coords, dummy_index, compute_gradient=False
                    )

                    # Compute Rg from P(r) curves
                    dr = r_grid[1] - r_grid[0]
                    rg_calc = torch.sqrt(
                        torch.sum(r_grid ** 2 * pr_calc * dr) / (2.0 * pr_calc.sum() * dr + 1e-8)
                    )
                    rg_exp = torch.sqrt(
                        torch.sum(r_grid ** 2 * pr_exp * dr) / (2.0 * pr_exp.sum() * dr + 1e-8)
                    )

                    # Compute loss metrics
                    cdf_calc = torch.cumsum(pr_calc * dr, dim=0)
                    cdf_exp = torch.cumsum(pr_exp * dr, dim=0)
                    w1_loss = torch.abs(cdf_calc - cdf_exp).sum() * dr

                    mse_loss = ((pr_calc - pr_exp) ** 2).sum() * dr

                    saxs_pr_results = {
                        'r_grid': r_grid.cpu().numpy(),
                        'pr_exp': pr_exp.cpu().numpy(),
                        'pr_calc': pr_calc.cpu().numpy(),
                        'rg_exp': rg_exp.item(),
                        'rg_calc': rg_calc.item(),
                        'w1_loss': w1_loss.item(),
                        'mse_loss': mse_loss.item(),
                    }

        # Compute final chemical shift fit if chemical shift steering was used
        cheshift_results = None
        if potentials is not None:
            from boltz.model.potentials.chemical_shift import ChemicalShiftPotential
            cheshift_potential = None
            for pot in potentials:
                if isinstance(pot, ChemicalShiftPotential):
                    cheshift_potential = pot
                    break

            if cheshift_potential is not None:
                try:
                    # Get feats from network_condition_kwargs
                    feats = network_condition_kwargs.get("feats")
                    if feats is None:
                        raise ValueError("feats not found in network_condition_kwargs")

                    # Get experimental shifts
                    exp_shifts = cheshift_potential._exp_shifts

                    # Compute predicted shifts for ensemble average
                    # atom_coords is [multiplicity, n_atoms, 3]
                    shifts_calc = cheshift_potential._compute_shifts(atom_coords, feats)

                    cheshift_results = {'nuclei': {}}

                    for nucleus in ['CA', 'CB']:
                        if nucleus not in exp_shifts or nucleus not in shifts_calc:
                            continue

                        exp_dict = exp_shifts[nucleus]
                        calc_tensor = shifts_calc[nucleus]  # [mult, n_residues]

                        # Ensemble average
                        calc_mean = calc_tensor.mean(dim=0)  # [n_residues]
                        calc_std = calc_tensor.std(dim=0) if calc_tensor.shape[0] > 1 else torch.zeros_like(calc_mean)

                        # Build per-residue data
                        residue_nums = sorted(exp_dict.keys())
                        exp_values = []
                        calc_values = []
                        calc_stds = []

                        for res_id in residue_nums:
                            res_idx = res_id - 1  # Convert to 0-indexed
                            if 0 <= res_idx < calc_mean.shape[0]:
                                exp_values.append(exp_dict[res_id])
                                calc_values.append(calc_mean[res_idx].item())
                                calc_stds.append(calc_std[res_idx].item())

                        if exp_values:
                            import numpy as np
                            exp_arr = np.array(exp_values)
                            calc_arr = np.array(calc_values)
                            std_arr = np.array(calc_stds)

                            # Filter out chemically unreasonable predictions
                            # CA: typically 40-75 ppm, CB: typically 15-75 ppm
                            # Zero values indicate invalid CheShift grid regions
                            if nucleus == 'CA':
                                valid_mask = (calc_arr >= 40) & (calc_arr <= 75)
                            else:  # CB
                                valid_mask = (calc_arr >= 10) & (calc_arr <= 80)
                            n_valid = valid_mask.sum()
                            n_total = len(calc_arr)

                            if n_valid > 0:
                                exp_valid = exp_arr[valid_mask]
                                calc_valid = calc_arr[valid_mask]
                                std_valid = std_arr[valid_mask]
                                res_valid = np.array(residue_nums)[valid_mask]

                                # Apply reference offset (same as in compute_function)
                                if cheshift_potential._auto_offset:
                                    # Auto-offset: estimate from data
                                    offset = exp_valid.mean() - calc_valid.mean()
                                else:
                                    # Fixed offset from YAML parameters
                                    if nucleus == 'CA':
                                        offset = cheshift_potential._ca_dss_offset
                                    else:
                                        offset = cheshift_potential._cb_dss_offset
                                calc_valid_adjusted = calc_valid + offset

                                # Compute metrics using offset-adjusted values
                                rmsd = np.sqrt(np.mean((calc_valid_adjusted - exp_valid) ** 2))
                                mae = np.mean(np.abs(calc_valid_adjusted - exp_valid))
                                correlation = np.corrcoef(exp_valid, calc_valid_adjusted)[0, 1] if len(exp_valid) > 1 else 0.0

                                cheshift_results['nuclei'][nucleus] = {
                                    'residue_nums': res_valid.tolist(),
                                    'exp_shifts': exp_valid.tolist(),
                                    'calc_shifts': calc_valid_adjusted.tolist(),  # Store adjusted values
                                    'calc_stds': std_valid.tolist(),
                                    'rmsd': float(rmsd),
                                    'mae': float(mae),
                                    'correlation': float(correlation),
                                    'n_valid': int(n_valid),
                                    'n_total': int(n_total),
                                    'offset_applied': float(offset),  # Record the offset used
                                }

                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to compute chemical shift results: {e}")

        result = dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)
        if saxs_pr_results is not None:
            result['saxs_pr_results'] = saxs_pr_results
        if cheshift_results is not None:
            result['cheshift_results'] = cheshift_results

        # Include bias histories from metadynamics potentials
        bias_histories = []
        if potentials is not None:
            from boltz.model.potentials.metadynamics import MetadynamicsPotential
            for pot in potentials:
                if isinstance(pot, MetadynamicsPotential):
                    pot_data = pot.export_data()
                    if pot_data.get('hills') or pot_data.get('repulsion_history'):
                        bias_histories.append(pot_data)

        if bias_histories:
            result['bias_histories'] = bias_histories

        # Add comprehensive CV histories for all tracked potentials
        if cv_histories:
            result['cv_histories'] = cv_histories

        # Clear SAXS cache to free GPU memory before confidence prediction
        try:
            from boltz.model.potentials.potentials import SAXSPrPotential
            SAXSPrPotential.clear_cache(-1)  # Force clear with invalid step
        except ImportError:
            pass

        return result

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        denoised_atom_coords = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        return {
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        filter_by_plddt=0.0,
    ):
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            if filter_by_plddt > 0:
                plddt_mask = feats["plddt"] > filter_by_plddt
                resolved_atom_mask_uni = resolved_atom_mask_uni * plddt_mask.float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            align_weights = denoised_atom_coords.new_ones(denoised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            atom_coords = out_dict["aligned_true_atom_coords"].float()
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach(),
                denoised_atom_coords.detach(),
                align_weights.detach(),
                mask=feats["atom_resolved_mask"]
                .float()
                .repeat_interleave(multiplicity, 0)
                .detach(),
            )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)
            mse_loss = torch.sum(
                mse_loss * align_weights * resolved_atom_mask, dim=-1
            ) / (torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) + 1e-5)

            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
