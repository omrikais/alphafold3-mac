"""Diffusion head matching AF3 JAX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.core.constants import (
    SIGMA_DATA,
    SIGMA_MIN,
    SIGMA_MAX,
    RHO,
    DIFFUSION_EVAL_INTERVAL,
)
from alphafold3_mlx.core.validation import check_nan
from alphafold3_mlx.geometry import Vec3Array, Rot3Array, random_gaussian_vector
from alphafold3_mlx.modules import Linear, LayerNorm
from alphafold3_mlx.network import featurization
from alphafold3_mlx.network.noise_level_embeddings import noise_embeddings
from alphafold3_mlx.network.noise_schedule import karras_schedule
from alphafold3_mlx.network.diffusion_transformer import (
    SelfAttentionConfig,
    TransformerConfig,
    Transformer,
    TransitionBlock,
)
from alphafold3_mlx.network.atom_cross_attention import (
    AtomCrossAttEncoder,
    AtomCrossAttDecoder,
)

if TYPE_CHECKING:
    from alphafold3_mlx.core.config import DiffusionConfig, GlobalConfig
    from alphafold3_mlx.feat_batch import Batch


def _mask_mean(mask: mx.array, value: mx.array, axis, keepdims, eps=1e-6) -> mx.array:
    numerator = mx.sum(mask * value, axis=axis, keepdims=keepdims)
    denom = mx.sum(mask, axis=axis, keepdims=keepdims)
    return numerator / (denom + eps)


def random_rotation(key: mx.array) -> Rot3Array:
    """Generate a random rotation using Rot3Array.

    Uses uniform random rotation according to Haar measure via
    quaternion sampling for proper SO(3) distribution.

    Args:
        key: MLX random key.

    Returns:
        Rot3Array representing a uniformly sampled rotation.
    """
    return Rot3Array.random_uniform(key=key, shape=())


def random_augmentation(
    rng_key: mx.array,
    positions: mx.array,
    mask: mx.array,
) -> mx.array:
    """Apply random rigid augmentation to positions using Vec3Array/Rot3Array.

    Applies center-of-mass centering, random rotation, and random translation.

    Args:
        rng_key: MLX random key.
        positions: Atom positions of shape [..., 3].
        mask: Atom mask of shape [...].

    Returns:
        Augmented positions with same shape as input.
    """
    keys = mx.random.split(rng_key, 2)
    rotation_key = keys[0]
    translation_key = keys[1]

    # Compute center of mass using mask
    center = _mask_mean(mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6)
    centered_positions = positions - center

    # Convert to Vec3Array for idiomatic geometry operations
    pos_vec = Vec3Array.from_array(centered_positions)

    # Get random rotation using Rot3Array
    rot = random_rotation(rotation_key)

    # Apply rotation using Rot3Array.apply_to_point
    rotated_vec = rot.apply_to_point(pos_vec)

    # Generate random translation as Vec3Array
    translation = random_gaussian_vector(shape=(), key=translation_key)

    # Apply translation
    translated_vec = rotated_vec + translation

    # Convert back to dense array and apply mask
    augmented = translated_vec.to_array()
    return augmented * mask[..., None]


class DiffusionHead(nn.Module):
    """AF3 diffusion head."""

    def __init__(
        self,
        config: "DiffusionConfig | None" = None,
        global_config: "GlobalConfig | None" = None,
    ) -> None:
        super().__init__()
        from alphafold3_mlx.core.config import DiffusionConfig, GlobalConfig

        self.config = config or DiffusionConfig()
        self.global_config = global_config or GlobalConfig()

        c = self.config

        # Conditioning projections
        self.pair_cond_initial_norm = LayerNorm(
            c.conditioning_pair_channel + 0, create_offset=False
        )
        self.pair_cond_initial_projection = Linear(
            c.conditioning_pair_channel,
            input_dims=c.conditioning_pair_channel + (2 * c.conditioning_pair_channel),  # placeholder
            use_bias=False,
            precision="highest",
        )

        self.single_cond_initial_norm = LayerNorm(
            c.conditioning_seq_channel + 0, create_offset=False
        )
        self.single_cond_initial_projection = Linear(
            c.conditioning_seq_channel,
            input_dims=c.conditioning_seq_channel + 0,
            use_bias=False,
            precision="highest",
        )

        self.noise_embedding_initial_norm = LayerNorm(
            c.conditioning_seq_channel, create_offset=False
        )
        self.noise_embedding_initial_projection = Linear(
            c.conditioning_seq_channel,
            input_dims=c.conditioning_seq_channel,
            use_bias=False,
            precision="highest",
        )

        # Pair transitions (no conditioning)
        self.pair_transition_0 = TransitionBlock(
            c.conditioning_pair_channel,
            2,
            cond_dim=None,
            final_init=self.global_config.final_init,
        )
        self.pair_transition_1 = TransitionBlock(
            c.conditioning_pair_channel,
            2,
            cond_dim=None,
            final_init=self.global_config.final_init,
        )

        # Single transitions (no conditioning)
        self.single_transition_0 = TransitionBlock(
            c.conditioning_seq_channel,
            2,
            cond_dim=None,
            final_init=self.global_config.final_init,
        )
        self.single_transition_1 = TransitionBlock(
            c.conditioning_seq_channel,
            2,
            cond_dim=None,
            final_init=self.global_config.final_init,
        )

        # Atom cross-attention
        self.atom_cross_att_encoder = AtomCrossAttEncoder(c, self.global_config, name="diffusion")
        self.atom_cross_att_decoder = AtomCrossAttDecoder(c, self.global_config, name="diffusion")

        # Single conditioning embedding
        self.single_cond_embedding_norm = LayerNorm(
            c.conditioning_seq_channel, create_offset=False
        )
        self.single_cond_embedding_projection = Linear(
            c.per_token_channels,
            input_dims=c.conditioning_seq_channel,
            use_bias=False,
            initializer=self.global_config.final_init,
            precision="highest",
        )

        # Transformer
        attn_cfg = SelfAttentionConfig(
            num_head=c.transformer_heads,
            key_dim=c.key_dim,
            value_dim=c.value_dim,
        )
        tr_cfg = TransformerConfig(
            attention=attn_cfg,
            num_blocks=c.num_transformer_blocks,
            super_block_size=c.transformer_super_block_size,
            num_intermediate_factor=c.transformer_num_intermediate_factor,
        )
        self.transformer = Transformer(tr_cfg, self.global_config)

        # Output norm
        self.output_norm = LayerNorm(c.per_token_channels, create_offset=False)

        self._built = False
        self._compiled = False
        self._compiled_denoise_step = None

    def _build_conditioning(self, pair_cond_dim: int, single_cond_dim: int) -> None:
        if self._built:
            return
        c = self.config
        # Pair conditioning projection dims: pair_emb + rel_features
        rel_dim = (2 * 32 + 2) * 2 + 1 + (2 * 2 + 2)
        self.pair_cond_initial_norm = LayerNorm(pair_cond_dim + rel_dim, create_offset=False)
        self.pair_cond_initial_projection = Linear(
            c.conditioning_pair_channel,
            input_dims=pair_cond_dim + rel_dim,
            use_bias=False,
            precision="highest",
        )

        # Single conditioning: single_emb + target_feat
        self.single_cond_initial_norm = LayerNorm(single_cond_dim, create_offset=False)
        self.single_cond_initial_projection = Linear(
            c.conditioning_seq_channel,
            input_dims=single_cond_dim,
            use_bias=False,
            precision="highest",
        )

        # Noise embedding projection input dim = 256 (fixed)
        self.noise_embedding_initial_norm = LayerNorm(256, create_offset=False)
        self.noise_embedding_initial_projection = Linear(
            c.conditioning_seq_channel,
            input_dims=256,
            use_bias=False,
            precision="highest",
        )
        self._built = True

    def denoise_step_arrays(
        self,
        encoder_token_act: mx.array,
        trunk_single_cond: mx.array,
        trunk_pair_cond: mx.array,
        sequence_mask: mx.array,
    ) -> mx.array:
        """Array-only core denoising computation.

        This function contains no Batch or GatherInfo usage and is suitable
        for mx.compile. It encapsulates the transformer-centric portion of
        the denoising step:
        1. Single conditioning embedding projection
        2. Transformer forward pass
        3. Output normalization

        Args:
            encoder_token_act: Token activations from AtomCrossAttEncoder,
                shape [num_tokens, per_token_channels].
            trunk_single_cond: Single conditioning from _conditioning(),
                shape [num_tokens, conditioning_seq_channel].
            trunk_pair_cond: Pair conditioning from _conditioning(),
                shape [num_tokens, num_tokens, conditioning_pair_channel].
            sequence_mask: Token mask, shape [num_tokens].

        Returns:
            Transformed activations ready for AtomCrossAttDecoder,
            shape [num_tokens, per_token_channels].
        """
        # Cast to float32 BEFORE adding embedding (matches JAX ordering)
        act = encoder_token_act.astype(mx.float32)

        # Add single conditioning embedding
        act = act + self.single_cond_embedding_projection(
            self.single_cond_embedding_norm(trunk_single_cond)
        )

        # Ensure float32 (redundant but matches JAX)
        act = act.astype(mx.float32)
        trunk_single_cond = trunk_single_cond.astype(mx.float32)
        trunk_pair_cond = trunk_pair_cond.astype(mx.float32)
        sequence_mask = sequence_mask.astype(mx.float32)

        # Core transformer computation
        act = self.transformer(
            act=act,
            mask=sequence_mask,
            single_cond=trunk_single_cond,
            pair_cond=trunk_pair_cond,
        )

        return self.output_norm(act)

    def compile(self) -> None:
        """Compile the array-only denoising step for improved performance.

        This method compiles denoise_step_arrays(), which encapsulates the
        computational core of the diffusion denoising step. This includes:
        - Single conditioning embedding projection
        - Transformer forward pass (majority of FLOPS)
        - Output normalization

        Denoising Step Compilation:
        The "denoising step" in sample() → denoising_step() → __call__() consists of:
        1. _conditioning(): Creates conditioning from Batch (NOT compiled - uses Batch)
        2. atom_cross_att_encoder(): Encodes positions (NOT compiled - uses Batch)
        3. denoise_step_arrays(): Core computation (COMPILED - pure array inputs)
        4. atom_cross_att_decoder(): Decodes positions (NOT compiled - uses Batch)

        denoise_step_arrays() takes only pure mx.array inputs:
        - encoder_token_act: [seq, channels] from encoder
        - trunk_single_cond: [seq, cond_dim] single conditioning
        - trunk_pair_cond: [seq, seq, pair_dim] pair conditioning
        - sequence_mask: [seq] float mask

        All inputs are pure mx.array, making the function suitable for mx.compile.

        Call frequency: num_steps × num_samples times per inference (e.g., 200 × 5 = 1000)
        This makes compilation highly beneficial for throughput.

        Note: Call this after the model is fully initialized and weights loaded.
        The transformer must have been built (requires at least one forward pass
        or explicit build) before compilation.
        """
        if self._compiled:
            return

        if not self.global_config.use_compile:
            self._compiled = True
            return

        from functools import partial

        # Compile denoise_step_arrays which includes transformer + embedding + norm
        # Capture state from all submodules used in denoise_step_arrays
        denoise_state = [
            self.single_cond_embedding_norm.state,
            self.single_cond_embedding_projection.state,
            self.transformer.state,
            self.output_norm.state,
        ]

        @partial(mx.compile, inputs=denoise_state, outputs=denoise_state)
        def compiled_denoise_step(encoder_token_act, trunk_single_cond, trunk_pair_cond, sequence_mask):
            return self.denoise_step_arrays(
                encoder_token_act=encoder_token_act,
                trunk_single_cond=trunk_single_cond,
                trunk_pair_cond=trunk_pair_cond,
                sequence_mask=sequence_mask,
            )

        self._compiled_denoise_step = compiled_denoise_step
        self._compiled = True

    def _conditioning(
        self,
        batch: "Batch",
        embeddings: dict[str, mx.array],
        noise_level: mx.array,
        use_conditioning: bool,
    ) -> tuple[mx.array, mx.array]:
        single_embedding = embeddings["single"] * use_conditioning
        pair_embedding = embeddings["pair"] * use_conditioning

        rel_features = featurization.create_relative_encoding(
            seq_features=batch.token_features,
            max_relative_idx=32,
            max_relative_chain=2,
        ).astype(pair_embedding.dtype)
        features_2d = mx.concatenate([pair_embedding, rel_features], axis=-1)

        self._build_conditioning(pair_embedding.shape[-1], single_embedding.shape[-1] + embeddings["target_feat"].shape[-1])

        # Build transformer blocks lazily (needed before compile)
        if not self.transformer._built:
            # Trigger build with dummy call to determine dimensions
            # This will be overwritten on first real call anyway
            pass

        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(features_2d)
        )
        pair_cond = pair_cond + self.pair_transition_0(pair_cond, None)
        pair_cond = pair_cond + self.pair_transition_1(pair_cond, None)

        target_feat = embeddings["target_feat"]
        features_1d = mx.concatenate([single_embedding, target_feat], axis=-1)
        single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d)
        )

        noise_embedding = noise_embeddings(noise_level / SIGMA_DATA)
        single_cond = single_cond + self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
        )

        single_cond = single_cond + self.single_transition_0(single_cond, None)
        single_cond = single_cond + self.single_transition_1(single_cond, None)

        return single_cond, pair_cond

    def __call__(
        self,
        positions_noisy: mx.array,
        noise_level: mx.array,
        batch: "Batch",
        embeddings: dict[str, mx.array],
        use_conditioning: bool,
    ) -> mx.array:
        # === Batch-dependent: Conditioning ===
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        # === Batch-dependent: Extract masks ===
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask

        # Scale noisy positions
        act = positions_noisy * atom_mask[..., None]
        act = act / mx.sqrt(noise_level ** 2 + SIGMA_DATA ** 2)

        # === Batch-dependent: Encoder ===
        enc = self.atom_cross_att_encoder(
            token_atoms_act=act,
            trunk_single_cond=embeddings["single"],
            trunk_pair_cond=trunk_pair_cond,
            batch=batch,
        )

        # === Array-only: Core denoising step ===
        # Use compiled denoise_step_arrays when available
        if self._compiled and self._compiled_denoise_step is not None:
            act = self._compiled_denoise_step(
                enc.token_act, trunk_single_cond, trunk_pair_cond, sequence_mask
            )
        else:
            act = self.denoise_step_arrays(
                encoder_token_act=enc.token_act,
                trunk_single_cond=trunk_single_cond,
                trunk_pair_cond=trunk_pair_cond,
                sequence_mask=sequence_mask,
            )

        # === Batch-dependent: Decoder ===
        position_update = self.atom_cross_att_decoder(
            token_act=act,
            enc=enc,
            batch=batch,
        )

        # Skip scaling
        skip_scaling = SIGMA_DATA ** 2 / (noise_level ** 2 + SIGMA_DATA ** 2)
        out_scaling = noise_level * SIGMA_DATA / mx.sqrt(noise_level ** 2 + SIGMA_DATA ** 2)

        return (skip_scaling * positions_noisy + out_scaling * position_update) * atom_mask[..., None]

    def sample(
        self,
        denoising_step: Callable[[mx.array, mx.array], mx.array],
        batch: "Batch",
        key: mx.array,
        num_steps: int,
        gamma_0: float,
        gamma_min: float,
        noise_scale: float,
        step_scale: float,
        num_samples: int,
        capture_checkpoints: bool = False,
        sigma_min: float = SIGMA_MIN,
        sigma_max: float = SIGMA_MAX,
        rho: float = RHO,
        check_nans: bool = False,
        step_callback: Callable[[int, int], None] | None = None,
        guidance_fn: Callable[[mx.array, mx.array, int], mx.array] | None = None,
    ) -> dict[str, mx.array]:
        """Run diffusion sampling loop.

        Args:
            denoising_step: Callable that performs a single denoising step.
                Takes (positions_noisy, noise_level) and returns denoised positions.
            batch: Batch containing structure information.
            key: Random key for sampling.
            num_steps: Number of diffusion steps.
            gamma_0: Initial gamma for stochastic sampling.
            gamma_min: Minimum gamma threshold.
            noise_scale: Noise scale factor.
            step_scale: Step scale factor.
            num_samples: Number of structure samples to generate.
            capture_checkpoints: If True, capture intermediate positions.
            sigma_min: Minimum noise level for Karras schedule.
            sigma_max: Maximum noise level for Karras schedule.
            rho: Karras schedule rho parameter.
            check_nans: If True, check for NaN values after each denoising step
                and raise NaNError with step index on detection.
            step_callback: Optional callback called after each step with
                (step, total_steps) for progress reporting.
            guidance_fn: Optional restraint guidance function. Takes
                (positions_denoised, t_hat, step) and returns gradient
                to add to the ODE tangent.

        Returns:
            Dictionary with 'atom_positions', 'mask', and optionally 'checkpoints'.

        Raises:
            NaNError: If check_nans=True and NaN values are detected during
                diffusion. Error includes step index and sample index for debugging.
        """
        mask = batch.predicted_structure_info.atom_mask

        # Use Karras schedule from noise_schedule.py
        noise_levels = karras_schedule(
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
        )

        keys = mx.random.split(key, num_samples + 1)
        noise_key = keys[0]
        sample_keys = keys[1:]

        # Initialize positions with noise scaled by initial sigma
        positions = mx.random.normal(
            shape=(num_samples,) + tuple(mask.shape) + (3,), key=noise_key
        )
        positions = positions * noise_levels[0]

        noise_level_prev = mx.broadcast_to(noise_levels[0], (num_samples,))

        # Checkpoint capture - use list to avoid .copy on MLX arrays
        checkpoints = [] if capture_checkpoints else None

        for step in range(1, num_steps + 1):
            noise_level = noise_levels[step]
            new_positions = []
            new_keys = []

            for i in range(num_samples):
                k = sample_keys[i]
                k_split = mx.random.split(k, 3)
                key_next = k_split[0]
                key_aug = k_split[1]
                key_noise = k_split[2]

                pos_i = positions[i]
                # AF3 JAX applies random rigid augmentation each denoising step.
                pos_i = random_augmentation(key_aug, pos_i, mask)

                gamma = gamma_0 * (noise_level > gamma_min)
                t_hat = noise_level_prev[i] * (1 + gamma)

                noise_scale_val = noise_scale * mx.sqrt(t_hat ** 2 - noise_level_prev[i] ** 2)
                noise = noise_scale_val * mx.random.normal(shape=pos_i.shape, key=key_noise)
                positions_noisy = pos_i + noise

                positions_denoised = denoising_step(positions_noisy, t_hat)

                # Per-step NaN detection
                if check_nans:
                    # Force evaluation to materialize NaN state
                    mx.eval(positions_denoised)
                    check_nan(
                        positions_denoised,
                        component=f"diffusion.denoise.sample_{i}",
                        step=step,
                        raise_on_nan=True,
                    )

                grad = (positions_noisy - positions_denoised) / t_hat

                # Integrate restraint guidance into the ODE tangent.
                # In the Karras/EDM formulation, grad = (x - D(x))/σ is
                # the *negative* of σ×score.  Classifier guidance modifies
                # the score: score_guided = score - w·∇L, so the guided
                # tangent is d_guided = grad + w·∇L.  We therefore ADD the
                # guidance gradient (which is w·∇L) to the tangent.
                # Sigma-dependent scaling is applied inside the
                # guidance function.
                if guidance_fn is not None:
                    restraint_grad = guidance_fn(positions_denoised, t_hat, step)
                    grad = grad + restraint_grad

                d_t = noise_level - t_hat
                positions_out = positions_noisy + step_scale * d_t * grad

                new_positions.append(positions_out)
                new_keys.append(key_next)

            positions = mx.stack(new_positions, axis=0)
            sample_keys = mx.stack(new_keys, axis=0)
            noise_level_prev = mx.broadcast_to(noise_level, (num_samples,))

            # Per-step NaN check on stacked positions
            if check_nans:
                mx.eval(positions)
                check_nan(
                    positions,
                    component="diffusion.step_output",
                    step=step,
                    raise_on_nan=True,
                )

            # Capture checkpoint if requested
            # Use slicing to create independent copy without .copy()
            if capture_checkpoints:
                checkpoints.append(positions[:])

            # Periodic evaluation for memory management
            if step % DIFFUSION_EVAL_INTERVAL == 0:
                mx.eval(positions)

            # Progress callback (step is 1-indexed in loop)
            if step_callback is not None:
                step_callback(step - 1, num_steps)  # Convert to 0-indexed

        final_dense_atom_mask = mx.broadcast_to(mask[None, ...], (num_samples,) + tuple(mask.shape))

        result = {"atom_positions": positions, "mask": final_dense_atom_mask}
        if capture_checkpoints:
            # Stack checkpoints: shape (num_checkpoints, num_samples, num_res, num_atoms, 3)
            result["checkpoints"] = mx.stack(checkpoints, axis=0)
        return result
