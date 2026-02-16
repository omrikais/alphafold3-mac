"""Inference runner for AlphaFold 3 MLX pipeline.

This module provides the high-level InferenceRunner class that orchestrates
the full inference pipeline .

Uses the real AlphaFold 3 data pipeline and featurisation modules.

Example:
    runner = InferenceRunner(args, progress_reporter)
    result = runner.run()
"""

from __future__ import annotations

import datetime
import json
import logging
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from alphafold3_mlx.pipeline.errors import (
    InputError,
    ResourceError,
    InferenceError,
    InterruptError,
    FailureLog,
)
from alphafold3_mlx.pipeline.progress import ProgressReporter, TimingData

logger = logging.getLogger(__name__)

# MSA depth cap for MLX inference to avoid OOM in MSA column attention.
# AF3 pipeline defaults to msa_crop_size=16384 which is too large for MLX SDPA.
MAX_MSA_SEQS = 4096

if TYPE_CHECKING:
    from alphafold3_mlx.pipeline.cli import CLIArguments
    from alphafold3_mlx.pipeline.input_handler import FoldInput
    from alphafold3_mlx.pipeline.output_handler import OutputBundle
    from alphafold3_mlx.pipeline.ranking import SampleRanking


class InterruptHandler:
    """Handles SIGINT (Ctrl+C) for graceful shutdown.

    Catches SIGINT signal and sets interrupted flag. Runner should
    check this flag at safe points to perform cleanup.
    """

    def __init__(self) -> None:
        """Initialize interrupt handler."""
        self.interrupted = False
        self._original_handler: signal.Handlers | None = None

    def install(self) -> None:
        """Install signal handler (skipped when not on main thread)."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            return
        self._original_handler = signal.signal(signal.SIGINT, self._handler)

    def uninstall(self) -> None:
        """Restore original signal handler."""
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None

    def _handler(self, signum: int, frame: Any) -> None:
        """Signal handler for SIGINT."""
        self.interrupted = True
        print("\nInterrupt received, cleaning up...", file=sys.stderr)

    def check(self) -> None:
        """Check if interrupted and raise if so.

        Raises:
            InterruptError: If user pressed Ctrl+C.
        """
        if self.interrupted:
            raise InterruptError("Interrupted by user. Partial outputs cleaned up.")


@dataclass
class InferenceRunner:
    """High-level inference orchestration.

    Orchestrates the full inference pipeline:
    1. Load model weights
    2. Prepare input features using real AF3 data pipeline
    3. Run inference
    4. Rank samples and write outputs

    Attributes:
        args: CLI arguments.
        fold_input: Parsed FoldInput wrapper around AF3 folding_input.Input.
        progress: Progress reporter for status updates.
        interrupt_handler: Handler for Ctrl+C signals.
        temp_files: List of temporary files to clean up on interrupt.
    """

    args: "CLIArguments"
    input_json: "FoldInput"  # Named input_json for backward compatibility
    progress: ProgressReporter = field(default_factory=ProgressReporter)
    interrupt_handler: InterruptHandler = field(default_factory=InterruptHandler)
    temp_files: list[Path] = field(default_factory=list)

    # Callbacks for diffusion/recycling/confidence progress
    _on_diffusion_step: Callable[[int, int], None] | None = None
    _on_recycling_iteration: Callable[[int, int], None] | None = None
    _on_confidence: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        """Initialize runner state."""
        # Set up progress callbacks
        self._on_diffusion_step = self.progress.on_diffusion_step
        self._on_recycling_iteration = self.progress.on_recycling_iteration
        self._on_confidence = self._confidence_callback

    def _confidence_callback(self, phase: str) -> None:
        """Handle confidence stage progress.

        Args:
            phase: "start" or "end" to indicate phase of confidence computation.
        """
        if phase == "start":
            self.progress.on_confidence_start()
        elif phase == "end":
            self.progress.on_confidence_end()

    def run(self) -> "OutputBundle":
        """Execute full inference pipeline.

        Returns:
            OutputBundle with paths to all generated output files.

        Raises:
            InputError: If input validation fails.
            ResourceError: If insufficient resources.
            InferenceError: If model inference fails.
            InterruptError: If user interrupts with Ctrl+C.
        """
        from alphafold3_mlx.pipeline.input_handler import check_memory_available
        from alphafold3_mlx.pipeline.output_handler import (
            OutputBundle,
            create_output_directory,
            handle_existing_outputs,
            write_mmcif_file,
            write_confidence_scores,
            write_timing,
            write_ranking_debug,
        )
        from alphafold3_mlx.pipeline.ranking import rank_samples, auto_detect_complex

        logger.info("Starting inference pipeline")
        logger.debug(
            "Pipeline config: num_samples=%d, diffusion_steps=%d, precision=%s",
            self.args.num_samples,
            self.args.diffusion_steps,
            self.args.precision,
        )

        # Install interrupt handler
        self.interrupt_handler.install()

        try:
            # Create output directory (not a timed stage per spec)
            create_output_directory(self.args.output_dir)
            handle_existing_outputs(
                self.args.output_dir,
                self.args.num_samples,
                self.args.no_overwrite,
            )
            output_bundle = OutputBundle(output_dir=self.args.output_dir)
            output_bundle.initialize_structure_files(self.args.num_samples)

            # Check memory requirements (not a timed stage per spec)
            check_memory_available(
                self.input_json,
                self.args.num_samples,
                safety_factor=0.8,
            )
            self.interrupt_handler.check()

            # Early fail-fast restraint validation (before expensive operations)
            # This catches format errors, invalid chain/residue/atom references,
            # and bad guidance config BEFORE running MSA search or featurization.
            # Index resolution (which requires featurized batch) happens later
            # in _build_guidance_fn().
            self._early_validate_restraints()
            self.interrupt_handler.check()

            # Load model weights - timing.json stage: weight_loading
            self.progress.on_stage_start("weight_loading")
            model, config = self._load_model()
            self.progress.on_stage_end("weight_loading")
            self.interrupt_handler.check()

            # Run data pipeline if enabled (MSA/template search)
            if self.args.run_data_pipeline:
                self.progress.on_stage_start("data_pipeline")
                self._run_data_pipeline()
                self.progress.on_stage_end("data_pipeline")
                self.interrupt_handler.check()

            # Prepare input features - timing.json stage: feature_preparation
            self.progress.on_stage_start("feature_preparation")
            batch = self._prepare_features()
            self.progress.on_stage_end("feature_preparation")
            self.interrupt_handler.check()

            # Run inference - timing for recycling, diffusion, confidence comes from InferenceStats
            result, stats = self._run_inference(model, batch)
            self.interrupt_handler.check()

            # Rank samples (not a timed stage per spec)
            is_complex = auto_detect_complex(self.input_json.chain_ids)
            ranking = self._rank_samples(result, is_complex)

            # Write outputs - timing.json stage: output_writing
            self.progress.on_stage_start("output_writing")
            self._write_outputs(result, ranking, output_bundle, batch, stats)
            self.progress.on_stage_end("output_writing")

            # Pass inference component timing to progress reporter for verbose output
            if stats is not None:
                self.progress.set_inference_timing(
                    recycling_seconds=getattr(stats, "evoformer_duration_seconds", 0.0),
                    diffusion_seconds=getattr(stats, "diffusion_duration_seconds", 0.0),
                    confidence_seconds=getattr(stats, "confidence_duration_seconds", 0.0),
                )

            # Complete
            self.progress.on_complete()
            logger.info("Inference pipeline completed successfully")

            return output_bundle

        except InterruptError:
            logger.warning("Inference interrupted by user")
            self._cleanup_temps()
            raise

        finally:
            self.interrupt_handler.uninstall()

    def _early_validate_restraints(self) -> None:
        """Early fail-fast validation of restraints before expensive operations.

        Performs restraint validation using only the input sequences, WITHOUT
        requiring featurized batches or index resolution. This catches errors
        BEFORE expensive MSA search and feature preparation stages.

        **Validations performed:**
        - Chain existence
        - Residue range checks
        - Atom name validity
        - Guidance config parameter ranges
        - Restraint count warnings
        - Conflict detection

        **Validations deferred to later stages:**
        - Index resolution (requires batch.token_features) - happens in _build_guidance_fn()

        Raises:
            InputError: If restraint validation fails.
        """
        restraint_config = getattr(self.input_json, "_restraints", None)
        guidance_config = getattr(self.input_json, "_guidance", None)

        if restraint_config is None or restraint_config.is_empty:
            return  # No restraints to validate

        logger.debug(
            "Early restraint validation: %d distance, %d contact, %d repulsive",
            len(restraint_config.distance),
            len(restraint_config.contact),
            len(restraint_config.repulsive),
        )

        from alphafold3_mlx.restraints.types import GuidanceConfig
        from alphafold3_mlx.restraints.validate import (
            build_chain_info_from_input,
            validate_restraints,
        )

        # Build chain info from input sequences (no featurization needed)
        chains = build_chain_info_from_input(self.input_json.input)

        # Collect all chain IDs (protein + non-protein) for better error messages
        all_chain_ids = {c.id for c in self.input_json.input.chains}

        # Use default guidance config if not provided
        if guidance_config is None:
            guidance_config = GuidanceConfig()

        # Validate restraints using only sequence information
        num_steps = self.args.diffusion_steps
        errors = validate_restraints(
            restraint_config, chains, guidance_config, num_steps,
            all_chain_ids=all_chain_ids,
        )

        if errors:
            raise InputError(
                "Invalid restraint references (caught before MSA search):\n  "
                + "\n  ".join(errors)
            )

        logger.info(
            "Early restraint validation passed: %d total restraints",
            restraint_config.total_count,
        )

    def _load_model(self) -> tuple[Any, Any]:
        """Load model and weights.

        Uses Model.from_pretrained() to load model with weights in one step.

        Returns:
            Tuple of (model, config).

        Raises:
            ResourceError: If weights not found.
        """
        logger.debug("Loading model from %s", self.args.model_dir)
        try:
            from alphafold3_mlx import Model, ModelConfig
            from alphafold3_mlx.core import DiffusionConfig, GlobalConfig
            from alphafold3_mlx.weights import WeightsNotFoundError
        except ImportError as e:
            logger.error("Failed to import MLX modules: %s", e)
            raise InferenceError(f"Failed to import MLX modules: {e}")

        # Create configuration
        precision = self.args.precision
        if precision is None:
            from alphafold3_mlx.pipeline.cli import auto_select_precision
            precision = auto_select_precision()

        diffusion_config = DiffusionConfig(
            num_samples=self.args.num_samples,
            num_steps=self.args.diffusion_steps,
        )
        global_config = GlobalConfig(precision=precision)
        config = ModelConfig(diffusion=diffusion_config, global_config=global_config)

        try:
            # Seed MLX global RNG for deterministic parameter initialization.
            # This ensures that when --seed is provided, any randomly initialized
            # params (e.g., unmapped weights) are consistent across runs.
            if self.args.seed is not None:
                import mlx.core as mx
                mx.random.seed(self.args.seed)

            # Use from_pretrained to load model with weights via Phase 1 loader
            model = Model.from_pretrained(self.args.model_dir, config)
            logger.info("Model weights loaded successfully")
        except WeightsNotFoundError as e:
            logger.error("Failed to load weights: %s", e)
            raise ResourceError(
                f"Model weights not found at: {self.args.model_dir}. "
                "Expected directory containing af3.bin.zst"
            )
        except Exception as e:
            logger.error("Failed to load weights: %s", e)
            raise ResourceError(f"Failed to load weights: {e}")

        return model, config

    def _run_data_pipeline(self) -> None:
        """Run MSA/template search via AF3 data pipeline.

        Populates MSA and template data on the underlying AF3 Input object.
        Results are cached to ``msa_cache_dir`` when configured.
        """
        af3_input = self.input_json.input

        try:
            from alphafold3.data import pipeline as af3_data_pipeline
            from alphafold3.common import folding_input
            from alphafold3_mlx.data.validation import (
                DataPipelineNotConfiguredError,
                build_af3_data_pipeline_config,
            )
        except ImportError as e:
            logger.error("Failed to import AF3 data pipeline modules: %s", e)
            raise InferenceError(f"Failed to import AF3 data pipeline modules: {e}")

        require_rna = any(
            isinstance(chain, folding_input.RnaChain) for chain in af3_input.chains
        )
        try:
            dp_cfg, resolved, _tried = build_af3_data_pipeline_config(
                db_dir=self.args.db_dir, require_rna=require_rna
            )
        except DataPipelineNotConfiguredError as e:
            raise ResourceError(str(e))
        except Exception as e:
            raise ResourceError(f"Failed to configure AF3 data pipeline: {e}")

        logger.info("Running AF3 data pipeline (MSA/template search)")
        logger.debug("Data pipeline resolved paths: %s", {k: str(v) for k, v in resolved.items()})

        # Check MSA cache before running the expensive HMMER search
        msa_cache = None
        if self.args.msa_cache_dir is not None:
            from alphafold3_mlx.pipeline.msa_cache import MSACache
            msa_cache = MSACache(self.args.msa_cache_dir)
            cached = msa_cache.get(af3_input)
            if cached is not None:
                logger.info("MSA cache hit — skipping HMMER search")
                # Preserve current job's RNG seeds on the cached Input.
                # MSA/template data is seed-independent, but featurisation
                # uses rng_seeds from the Input for crop randomization.
                object.__setattr__(cached, 'rng_seeds', tuple(af3_input.rng_seeds))
                self._processed_input = cached
                return

        processed = af3_data_pipeline.DataPipeline(dp_cfg).process(af3_input)
        if msa_cache is not None:
            msa_cache.put(af3_input, processed)
        self._processed_input = processed

    def _prepare_features(self) -> Any:
        """Prepare input features using real AF3 featurisation.

        Uses alphafold3.data.featurisation.featurise_input to generate proper
        feature batches from the folding_input.Input.

        Returns:
            FeatureBatch for model input.
        """
        logger.debug("Preparing input features for %d residues", self.input_json.total_residues)
        import numpy as np
        try:
            import mlx.core as mx
            from alphafold3_mlx import FeatureBatch
            from alphafold3.data import featurisation
            from alphafold3.constants import chemical_components
        except ImportError as e:
            logger.error("Failed to import feature modules: %s", e)
            raise InferenceError(f"Failed to import feature modules: {e}")

        # Use pipeline-processed input if available, otherwise raw input
        af3_input = getattr(self, "_processed_input", None)
        if af3_input is None:
            af3_input = self.input_json.input
            # Fill missing MSA/template fields with defaults if needed.
            # This allows running without the full data pipeline.
            af3_input = af3_input.fill_missing_fields()

        # Create CCD with user-defined components if any
        ccd = chemical_components.Ccd(user_ccd=af3_input.user_ccd)

        # Determine bucket sizes for padding
        buckets = [256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]
        # M-06: Filter buckets by --max_tokens if set
        max_tokens = getattr(self.args, "max_tokens", None)
        if max_tokens is not None:
            buckets = [b for b in buckets if b <= max_tokens]
            if not buckets:
                buckets = [max_tokens]

        # M-07: Use configurable max_template_date
        max_template_date_str = getattr(self.args, "max_template_date", "2021-09-30")
        ref_max_modified_date = datetime.date.fromisoformat(max_template_date_str)

        try:
            # Run the real AF3 featurisation pipeline
            featurised_examples = featurisation.featurise_input(
                fold_input=af3_input,
                ccd=ccd,
                buckets=buckets,
                ref_max_modified_date=ref_max_modified_date,
                conformer_max_iterations=None,  # Use RDKit defaults
                resolve_msa_overlaps=True,  # Match AF3 paper method
                verbose=self.args.verbose,
            )

            # Use the first featurised example (seed selection happens in inference)
            if not featurised_examples:
                raise InferenceError("Featurisation produced no batches")

            batch_dict = featurised_examples[0]
            num_tokens = (
                batch_dict["aatype"].shape[0] if "aatype" in batch_dict else 0
            )
            logger.info(
                "Featurised %d tokens from %d chains",
                num_tokens,
                len(af3_input.chains),
            )

            # M-06: Hard-reject inputs that exceed --max_tokens.
            # Bucket filtering alone is insufficient because
            # calculate_bucket_size creates an oversized bucket when
            # num_tokens exceeds the largest configured bucket.
            if max_tokens is not None and num_tokens > max_tokens:
                raise InputError(
                    f"Input has {num_tokens} tokens, which exceeds "
                    f"--max_tokens={max_tokens}. Use a larger --max_tokens "
                    f"value or reduce the input size."
                )

            # Cap MSA depth for MLX memory limits.
            # AF3 pipeline defaults to msa_crop_size=16384, which is too large
            # for our current MLX memory budget on long chains.
            msa = batch_dict.get("msa")
            if msa is not None and msa.shape[0] > MAX_MSA_SEQS:
                original_msa = msa.shape[0]
                # Match AF3 behavior more closely: shuffle rows before truncation.
                # Taking the first rows can bias towards low-diversity hits and
                # degrade conditioning quality for real pipeline runs.
                rng_seed = self.args.seed if self.args.seed is not None else 0
                rng = np.random.default_rng(rng_seed)
                keep = rng.permutation(original_msa)[:MAX_MSA_SEQS]

                batch_dict["msa"] = msa[keep]
                if "msa_mask" in batch_dict and batch_dict["msa_mask"] is not None:
                    batch_dict["msa_mask"] = batch_dict["msa_mask"][keep]
                if "deletion_matrix" in batch_dict and batch_dict["deletion_matrix"] is not None:
                    batch_dict["deletion_matrix"] = batch_dict["deletion_matrix"][keep]
                logger.warning(
                    "Truncated MSA depth from %d to %d sequences for MLX memory limits",
                    original_msa,
                    MAX_MSA_SEQS,
                )

            # Heuristic placeholder disabling is only applicable to the
            # fill_missing_fields() (sequence-only) path. When MSAs/templates
            # come from the real data pipeline, keep them unchanged.
            if not self.args.run_data_pipeline:
                # AF3 fill_missing_fields() creates placeholder MSA/templates for
                # sequence-only inputs. Treat these as "no MSA/template" so the
                # model relies on target/profile conditioning instead of synthetic
                # stack activations.
                is_placeholder_msa, msa_metrics = self._detect_placeholder_msa(batch_dict)
                if is_placeholder_msa:
                    batch_dict["msa"] = None
                    batch_dict["msa_mask"] = None
                    batch_dict["deletion_matrix"] = None
                    logger.warning(
                        "Detected placeholder MSA (active_rows=%d, unique_rows=%d, "
                        "mask_density=%.6f, profile_entropy=%.6f); disabling MSA stack",
                        int(msa_metrics["active_rows"]),
                        int(msa_metrics["unique_rows"]),
                        msa_metrics["mask_density"],
                        msa_metrics["profile_entropy"],
                    )

                is_placeholder_templates, template_metrics = self._detect_placeholder_templates(
                    batch_dict
                )
                if is_placeholder_templates:
                    batch_dict["template_aatype"] = None
                    batch_dict["template_all_atom_positions"] = None
                    batch_dict["template_all_atom_mask"] = None
                    batch_dict["template_atom_positions"] = None
                    batch_dict["template_atom_mask"] = None
                    logger.warning(
                        "Detected placeholder templates (nonzero_mask_atoms=%d); disabling template stack",
                        int(template_metrics["nonzero_mask_atoms"]),
                    )
            return FeatureBatch.from_numpy(batch_dict)

        except Exception as e:
            logger.error("Featurisation failed: %s", e)
            raise InferenceError(f"Featurisation failed: {e}")

    def _detect_placeholder_msa(self, batch_dict: dict[str, Any]) -> tuple[bool, dict[str, float]]:
        """Detect fill_missing_fields placeholder MSA tensors.

        Placeholder MSAs produced by AF3 `fill_missing_fields()` are padded to
        very large depth and typically contain only one duplicated active row.
        """
        import numpy as np

        metrics = {
            "active_rows": 0.0,
            "unique_rows": 0.0,
            "mask_density": 0.0,
            "profile_entropy": 0.0,
        }

        msa = batch_dict.get("msa")
        msa_mask = batch_dict.get("msa_mask")
        deletion_matrix = batch_dict.get("deletion_matrix")
        if msa is None or msa_mask is None:
            return False, metrics

        msa_np = np.asarray(msa)
        msa_mask_np = np.asarray(msa_mask)
        if msa_np.ndim != 2 or msa_mask_np.ndim != 2 or msa_np.shape != msa_mask_np.shape:
            return False, metrics

        row_activity = msa_mask_np.sum(axis=1)
        active_mask = row_activity > 0
        active_rows = int(active_mask.sum())
        mask_density = float(msa_mask_np.mean())
        depth = int(msa_np.shape[0])

        if active_rows > 0:
            active_msa = msa_np[active_mask]
            unique_rows = int(np.unique(active_msa, axis=0).shape[0])

            entropies: list[float] = []
            for col in range(active_msa.shape[1]):
                _, counts = np.unique(active_msa[:, col], return_counts=True)
                probs = counts / counts.sum()
                entropies.append(float(-(probs * np.log(probs + 1e-12)).sum()))
            profile_entropy = float(np.mean(entropies))
        else:
            unique_rows = 0
            profile_entropy = 0.0

        deletion_sum = 0.0
        if deletion_matrix is not None:
            deletion_sum = float(np.asarray(deletion_matrix).sum())

        metrics.update(
            {
                "active_rows": float(active_rows),
                "unique_rows": float(unique_rows),
                "mask_density": mask_density,
                "profile_entropy": profile_entropy,
            }
        )

        is_placeholder = (
            depth >= 1024
            and active_rows <= 2
            and unique_rows <= 1
            and mask_density < 1e-3
            and deletion_sum == 0.0
        )
        return is_placeholder, metrics

    def _detect_placeholder_templates(
        self, batch_dict: dict[str, Any]
    ) -> tuple[bool, dict[str, float]]:
        """Detect fill_missing_fields placeholder template tensors."""
        import numpy as np

        metrics = {"nonzero_mask_atoms": 0.0}

        template_atom_mask = batch_dict.get("template_atom_mask")
        if template_atom_mask is None:
            template_atom_mask = batch_dict.get("template_all_atom_mask")
        if template_atom_mask is None:
            return False, metrics

        mask_np = np.asarray(template_atom_mask)
        if mask_np.size == 0:
            return True, metrics

        nonzero_mask_atoms = int(mask_np.sum())
        metrics["nonzero_mask_atoms"] = float(nonzero_mask_atoms)
        return nonzero_mask_atoms == 0, metrics

    def _run_inference(self, model: Any, batch: Any) -> tuple[Any, Any]:
        """Run model inference.

        Args:
            model: AlphaFold 3 model.
            batch: Input feature batch.

        Returns:
            Tuple of (ModelResult, InferenceStats).
        """
        import mlx.core as mx
        import numpy as np

        # Generate random key
        if self.args.seed is not None:
            key = mx.random.key(self.args.seed)
        else:
            key = mx.random.key(int(time.time() * 1000) % 2**32)

        # Set up progress callbacks for diffusion/recycling
        # The model should call these during inference
        def diffusion_callback(step: int, total: int) -> None:
            if self._on_diffusion_step:
                self._on_diffusion_step(step, total)
            self.interrupt_handler.check()

        def recycling_callback(iteration: int, total: int) -> None:
            if self._on_recycling_iteration:
                self._on_recycling_iteration(iteration, total)
            self.interrupt_handler.check()

        # Build restraint guidance function if restraints are present
        try:
            guidance_fn = self._build_guidance_fn(batch, model)
        except Exception as e:
            from alphafold3_mlx.restraints.resolve import RestraintResolutionError
            if isinstance(e, RestraintResolutionError):
                raise InputError(
                    f"Restraint resolution failed: {e}"
                )
            logger.error("Failed to build restraint guidance: %s", e)
            raise InputError(f"Failed to build restraint guidance: {e}")

        try:
            from alphafold3_mlx.model.inference import run_inference
            logger.debug("Starting model inference with seed=%s", self.args.seed)
            result, stats = run_inference(
                model=model,
                batch=batch,
                key=key,
                track_memory=self.args.verbose,
                diffusion_callback=diffusion_callback,
                recycling_callback=recycling_callback,
                confidence_callback=self._on_confidence,
                guidance_fn=guidance_fn,
            )
            logger.info("Model inference completed")
            return result, stats
        except Exception as e:
            logger.error("Inference failed: %s", e)
            if "NaN" in str(e):
                raise InferenceError(f"NaN detected during inference: {e}")
            raise InferenceError(f"Inference failed: {e}")

    def _build_guidance_fn(self, batch: Any, model: Any) -> Any:
        """Build restraint guidance function if restraints are present.

        This performs LATE-STAGE validation and index resolution that requires
        the featurized batch (token layout, asym_id, residue_index). Early-stage
        validation (chain/residue/atom existence) was already done in
        _early_validate_restraints() before expensive MSA search.

        **Late-stage operations:**
        - Resolve (chain, residue, atom) → (token_index, atom37_index)
        - Build guidance closure for diffusion

        Args:
            batch: Feature batch with token metadata.
            model: Model (for num_steps config).

        Returns:
            Guidance function or None if no restraints.
        """
        import numpy as np

        restraint_config = getattr(self.input_json, "_restraints", None)
        guidance_config = getattr(self.input_json, "_guidance", None)

        if restraint_config is None or restraint_config.is_empty:
            return None

        logger.info(
            "Resolving restraint indices: %d distance, %d contact, %d repulsive",
            len(restraint_config.distance),
            len(restraint_config.contact),
            len(restraint_config.repulsive),
        )

        from alphafold3_mlx.restraints.types import GuidanceConfig
        from alphafold3_mlx.restraints.resolve import resolve_restraints
        from alphafold3_mlx.restraints.guidance import build_guidance_fn

        # Use default guidance config if not provided
        if guidance_config is None:
            guidance_config = GuidanceConfig()

        # Early validation already happened in _early_validate_restraints().
        # Here we only need to resolve indices using the featurized batch.

        # Get num_steps from model config
        num_steps = model.config.diffusion.num_steps

        # Resolve restraint indices using batch token layout
        chain_ids = self.input_json.chain_ids
        asym_id = np.array(batch.token_features.asym_id)
        residue_index = np.array(batch.token_features.residue_index)
        aatype = np.array(batch.token_features.aatype)
        mask = np.array(batch.token_features.mask)
        if mask.ndim > 1:
            mask = mask.reshape(-1)

        resolved_distance, resolved_contact, resolved_repulsive = resolve_restraints(
            config=restraint_config,
            chain_ids=chain_ids,
            asym_id=asym_id,
            residue_index=residue_index,
            aatype=aatype,
            mask=mask,
        )

        # Store resolved restraints for satisfaction output
        self._resolved_distance = resolved_distance
        self._resolved_contact = resolved_contact
        self._resolved_repulsive = resolved_repulsive

        logger.info(
            "Resolved %d distance, %d contact, %d repulsive restraints",
            len(resolved_distance), len(resolved_contact), len(resolved_repulsive),
        )

        # Compute chain token ranges for inter-chain CoM coupling.
        # asym_id is 1-indexed; each unique value maps to a contiguous
        # token range (start, end) where end is exclusive.
        chain_token_ranges: dict[int, tuple[int, int]] = {}
        unique_asym = np.unique(asym_id)
        for aid in unique_asym:
            indices = np.where(asym_id == aid)[0]
            chain_token_ranges[int(aid)] = (int(indices[0]), int(indices[-1]) + 1)

        # Build guidance function
        guidance_fn = build_guidance_fn(
            resolved_distance=resolved_distance,
            resolved_contact=resolved_contact,
            resolved_repulsive=resolved_repulsive,
            guidance=guidance_config,
            num_steps=num_steps,
            chain_token_ranges=chain_token_ranges,
        )

        logger.info(
            "Built guidance_fn: scale=%.2f annealing=%s steps=%d-%d/%d",
            guidance_config.scale, guidance_config.annealing,
            guidance_config.start_step,
            guidance_config.end_step or num_steps,
            num_steps,
        )

        return guidance_fn

    def _rank_samples(self, result: Any, is_complex: bool) -> "SampleRanking":
        """Rank samples by confidence.

        Args:
            result: Model result with confidence scores.
            is_complex: Whether input is a multi-chain complex.

        Returns:
            SampleRanking with ranked indices.
        """
        import numpy as np
        from alphafold3_mlx.pipeline.ranking import rank_samples

        np_result = result.to_numpy()

        # Extract confidence scores for each sample
        num_samples = result.num_samples
        ptm_scores = [float(np_result["ptm"][i]) for i in range(num_samples)]
        iptm_scores = [float(np_result["iptm"][i]) for i in range(num_samples)]

        # plddt is [num_samples, num_residues, max_atoms]
        # Need per-residue mean pLDDT for ranking
        plddt_per_sample = np_result["plddt"]  # [samples, residues, atoms]
        atom_mask = np_result["atom_mask"]  # [samples, residues, atoms]

        plddt_scores = []
        for i in range(num_samples):
            # Compute mean pLDDT per residue using atom mask
            sample_plddt = plddt_per_sample[i]  # [residues, atoms]
            sample_mask = atom_mask[i]  # [residues, atoms]

            # Support mixed layouts (e.g. pLDDT in dense-atom space, mask in atom37).
            if sample_plddt.shape[-1] != sample_mask.shape[-1]:
                atom_dim = min(sample_plddt.shape[-1], sample_mask.shape[-1])
                sample_plddt = sample_plddt[:, :atom_dim]
                sample_mask = sample_mask[:, :atom_dim]

            # Mean over atoms for each residue (masked mean)
            masked_plddt = sample_plddt * sample_mask
            residue_denominator = sample_mask.sum(axis=-1)
            residue_mean_plddt = masked_plddt.sum(axis=-1) / np.maximum(residue_denominator, 1.0)
            residue_mask = residue_denominator > 0
            residue_mean_plddt = residue_mean_plddt[residue_mask]
            plddt_scores.append(residue_mean_plddt.tolist())

        return rank_samples(ptm_scores, iptm_scores, plddt_scores, is_complex)

    def _write_outputs(
        self,
        result: Any,
        ranking: "SampleRanking",
        output_bundle: "OutputBundle",
        batch: Any,
        stats: Any = None,
    ) -> None:
        """Write all output files.

        Uses the centralized write_ranked_outputs function to ensure
        structure files are written in ranked order.

        Args:
            result: Model result.
            ranking: Sample ranking.
            output_bundle: Output file paths.
            batch: Feature batch with token metadata for mmCIF chain labels.
            stats: InferenceStats with recycling/diffusion/confidence timing.
        """
        from alphafold3_mlx.pipeline.output_handler import write_ranked_outputs
        import numpy as np

        # Get timing data for timing.json
        # Combines runner-level stages with inference component timing from stats
        timing_data = self._build_timing_data(stats)

        # Extract token metadata from batch for correct mmCIF chain labeling
        # These come from the featurisation pipeline and preserve chain order
        token_metadata = {
            "aatype": np.array(batch.token_features.aatype),
            "residue_index": np.array(batch.token_features.residue_index),
            "asym_id": np.array(batch.token_features.asym_id),
        }
        token_mask = np.array(batch.token_features.mask)
        if token_mask.ndim > 1:
            token_mask = token_mask.reshape(-1)
        token_mask = token_mask > 0

        # Build restraint satisfaction data if restraints were used
        restraint_data = None
        restraint_config = getattr(self.input_json, "_restraints", None)
        if restraint_config is not None and not restraint_config.is_empty:
            restraint_data = {
                "resolved_distance": getattr(self, "_resolved_distance", []),
                "resolved_contact": getattr(self, "_resolved_contact", []),
                "resolved_repulsive": getattr(self, "_resolved_repulsive", []),
                "restraint_config": restraint_config,
            }

        # Write all outputs using centralized function
        write_ranked_outputs(
            result=result,
            output_bundle=output_bundle,
            ranking=ranking,
            timing_data=timing_data,
            token_metadata=token_metadata,
            token_mask=token_mask,
            restraint_data=restraint_data,
        )

    def _build_timing_data(self, stats: Any = None) -> dict[str, Any]:
        """Build timing.json data per spec contract.

        Combines runner-level stages (weight_loading, feature_preparation,
        output_writing) with inference component timing from InferenceStats
        (recycling, diffusion, confidence).

        Args:
            stats: InferenceStats from run_inference, or None.

        Returns:
            Dictionary with total_seconds and stages matching spec.
            All required stage keys are always present (even if 0.0).
        """
        # Pass inference component timing to progress reporter first
        if stats is not None:
            self.progress.set_inference_timing(
                recycling_seconds=getattr(stats, "evoformer_duration_seconds", 0.0),
                diffusion_seconds=getattr(stats, "diffusion_duration_seconds", 0.0),
                confidence_seconds=getattr(stats, "confidence_duration_seconds", 0.0),
            )

        # Get combined timing with all required keys
        reporter_timing = self.progress.get_timing_data(include_all_stages=True)

        return {
            "total_seconds": reporter_timing.total_seconds,
            "stages": reporter_timing.stages,
        }

    def _cleanup_temps(self) -> None:
        """Clean up temporary files on interrupt."""
        for temp_path in self.temp_files:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass  # Best effort cleanup
        self.temp_files.clear()


def create_failure_log(
    exc: Exception,
    progress: ProgressReporter,
    output_dir: Path | None = None,
) -> FailureLog:
    """Create failure log from exception.

    Args:
        exc: The exception that caused the failure.
        progress: Progress reporter with timing data.
        output_dir: Optional output directory to write failure_log.json.

    Returns:
        FailureLog instance.
    """
    failure_log = FailureLog.from_exception(
        exc=exc,
        stage_reached=progress.get_current_stage(),
        timing_snapshot=progress.get_timing_snapshot(),
    )

    if output_dir is not None:
        from alphafold3_mlx.pipeline.output_handler import write_failure_log
        try:
            write_failure_log(failure_log.to_dict(), output_dir)
        except Exception:
            pass  # Best effort

    return failure_log
