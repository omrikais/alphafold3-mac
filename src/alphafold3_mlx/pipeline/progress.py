"""Progress reporting for AlphaFold 3 MLX pipeline.

This module provides progress reporting during inference .

Example:
    reporter = ProgressReporter(verbose=True)
    reporter.on_stage_start("weight_loading")
    # ... do work ...
    reporter.on_stage_end("weight_loading")
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, TextIO


@dataclass
class StageInfo:
    """Information about a processing stage.

    Attributes:
        name: Stage name.
        start_time: Unix timestamp when stage started.
        end_time: Unix timestamp when stage ended (None if not complete).
    """

    name: str
    start_time: float
    end_time: float | None = None

    @property
    def duration(self) -> float | None:
        """Get stage duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class TimingData:
    """Timing data for timing.json.

    Attributes:
        total_seconds: Total inference time.
        stages: Per-stage timing breakdown.
    """

    total_seconds: float
    stages: dict[str, float] = field(default_factory=dict)

    # Standard stage names (spec enum)
    STAGE_STARTUP: ClassVar[str] = "startup"
    STAGE_WEIGHT_LOADING: ClassVar[str] = "weight_loading"
    STAGE_FEATURE_PREP: ClassVar[str] = "feature_preparation"
    STAGE_RECYCLING: ClassVar[str] = "recycling"
    STAGE_DIFFUSION: ClassVar[str] = "diffusion"
    STAGE_CONFIDENCE: ClassVar[str] = "confidence"
    STAGE_OUTPUT: ClassVar[str] = "output_writing"

    # Valid stages per spec
    VALID_STAGES: ClassVar[frozenset[str]] = frozenset({
        "startup",
        "weight_loading",
        "feature_preparation",
        "recycling",
        "diffusion",
        "confidence",
        "output_writing",
    })

    # Mapping from internal stage names to spec stage names
    STAGE_MAPPING: ClassVar[dict[str, str]] = {
        "platform_validation": "startup",
        "input_validation": "startup",
        "input_parsing": "startup",
    }

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "total_seconds": self.total_seconds,
            "stages": self.stages,
        }


class ProgressReporter:
    """Displays inference progress to stderr.

    Provides callbacks for various stages of inference and generates
    progress output for the user.

    Attributes:
        verbose: If True, show detailed per-stage timing.
        output: Output stream (defaults to stderr).
    """

    def __init__(
        self,
        verbose: bool = False,
        output: TextIO | None = None,
    ) -> None:
        """Initialize progress reporter.

        Args:
            verbose: Enable detailed timing output.
            output: Output stream (defaults to stderr).
        """
        self.verbose = verbose
        self.output = output or sys.stderr

        self._stages: list[StageInfo] = []
        self._current_stage: StageInfo | None = None
        self._start_time: float = time.time
        self._last_diffusion_report: int = -20 # Force first report
        # Inference component timing (from InferenceStats)
        self._inference_timing: dict[str, float] = {}

    def on_stage_start(self, stage: str) -> None:
        """Called when major stage begins.

        Args:
            stage: Name of the stage starting.
        """
        self._current_stage = StageInfo(name=stage, start_time=time.time)
        self._print(f"[{stage}] Starting...")

    def on_stage_end(self, stage: str) -> None:
        """Called when major stage ends.

        Args:
            stage: Name of the stage ending.
        """
        if self._current_stage and self._current_stage.name == stage:
            self._current_stage.end_time = time.time
            self._stages.append(self._current_stage)

            if self.verbose:
                duration = self._current_stage.duration
                if duration is not None:
                    self._print(f"[{stage}] Complete ({duration:.1f}s)")
            else:
                self._print(f"[{stage}] Complete")

            self._current_stage = None

    def on_diffusion_step(self, step: int, total: int) -> None:
        """Called during diffusion (every 20 steps).

        Args:
            step: Current step (0-indexed).
            total: Total number of steps.
        """
        # Report every 20 steps or at the last step
        if step % 20 == 0 or step == total - 1:
            if step > self._last_diffusion_report or step == total - 1:
                self._print(f" Diffusion: step {step + 1}/{total}")
                self._last_diffusion_report = step

    def on_recycling_iteration(self, iteration: int, total: int) -> None:
        """Called during recycling.

        Args:
            iteration: Current iteration (0-indexed).
            total: Total number of iterations.
        """
        self._print(f" Recycling: iteration {iteration + 1}/{total}")

    def on_confidence_start(self) -> None:
        """Called when confidence computation begins."""
        self._print("[confidence] Starting...")

    def on_confidence_end(self) -> None:
        """Called when confidence computation ends."""
        self._print("[confidence] Complete")

    def on_sample_start(self, sample: int, total: int) -> None:
        """Called when starting a new sample.

        Args:
            sample: Current sample (0-indexed).
            total: Total number of samples.
        """
        self._print(f" Processing sample {sample + 1}/{total}")

    def set_inference_timing(
        self,
        recycling_seconds: float = 0.0,
        diffusion_seconds: float = 0.0,
        confidence_seconds: float = 0.0,
    ) -> None:
        """Set timing for inference components from InferenceStats.

        This allows the runner to pass timing from the model's internal
        measurements for recycling, diffusion, and confidence stages.

        Args:
            recycling_seconds: Time spent in recycling/evoformer.
            diffusion_seconds: Time spent in diffusion denoising.
            confidence_seconds: Time spent computing confidence scores.
        """
        self._inference_timing = {
            "recycling": recycling_seconds,
            "diffusion": diffusion_seconds,
            "confidence": confidence_seconds,
        }

    def on_complete(self) -> None:
        """Called when inference completes.

        Prints total elapsed time and optionally per-stage timing.
        """
        total_time = time.time - self._start_time
        self._print(f"\nInference complete! Total time: {total_time:.1f}s")

        if self.verbose:
            self._print("\nTiming breakdown:")
            # Display stages in canonical order
            stage_order = [
                "weight_loading",
                "feature_preparation",
                "recycling",
                "diffusion",
                "confidence",
                "output_writing",
            ]
            # Build timing dict from recorded stages
            stage_times: dict[str, float] = {}
            for stage in self._stages:
                if stage.duration is not None:
                    stage_times[stage.name] = stage.duration
            # Add inference component timing
            for key, value in self._inference_timing.items:
                if value > 0:
                    stage_times[key] = value
            # Print in order
            for stage_name in stage_order:
                if stage_name in stage_times:
                    self._print(f" {stage_name}: {stage_times[stage_name]:.1f}s")

    def get_timing_data(self, include_all_stages: bool = False) -> TimingData:
        """Return timing data for timing.json.

        Args:
            include_all_stages: If True, include all required stage keys
                even if they have zero duration.

        Returns:
            TimingData with total time and per-stage breakdown.
        """
        total_time = time.time - self._start_time
        stages: dict[str, float] = {}

        for stage in self._stages:
            if stage.duration is not None:
                stages[stage.name] = stage.duration

        # Add inference component timing
        for key, value in self._inference_timing.items:
            if value > 0 or include_all_stages:
                stages[key] = value

        # Ensure all required stages are present if requested
        if include_all_stages:
            required_stages = [
                "weight_loading",
                "feature_preparation",
                "recycling",
                "diffusion",
                "confidence",
                "output_writing",
            ]
            for stage_name in required_stages:
                if stage_name not in stages:
                    stages[stage_name] = 0.0

        return TimingData(total_seconds=total_time, stages=stages)

    def get_timing_snapshot(self) -> dict[str, float]:
        """Get current timing snapshot for failure logging.

        Returns:
            Dictionary of normalized stage names to durations.
            Stage names are normalized to spec enum values.
        """
        snapshot: dict[str, float] = {}
        for stage in self._stages:
            if stage.duration is not None:
                normalized = self._normalize_stage(stage.name)
                # Accumulate time for stages that map to the same normalized name
                snapshot[normalized] = snapshot.get(normalized, 0.0) + stage.duration

        # Include current stage if active
        if self._current_stage:
            elapsed = time.time - self._current_stage.start_time
            normalized = self._normalize_stage(self._current_stage.name)
            snapshot[normalized] = snapshot.get(normalized, 0.0) + elapsed

        return snapshot

    def get_current_stage(self) -> str:
        """Get name of current stage normalized to spec enum.

        Returns:
            Normalized stage name from spec enum:
            startup, weight_loading, feature_preparation, recycling,
            diffusion, confidence, or output_writing.
        """
        if self._current_stage:
            return self._normalize_stage(self._current_stage.name)
        return "startup"

    def _normalize_stage(self, stage: str) -> str:
        """Normalize stage name to spec enum.

        Args:
            stage: Internal stage name.

        Returns:
            Normalized stage name from spec enum.
        """
        # Check mapping first for internal stages
        if stage in TimingData.STAGE_MAPPING:
            return TimingData.STAGE_MAPPING[stage]
        # If already a valid spec stage, return as-is
        if stage in TimingData.VALID_STAGES:
            return stage
        # Unknown stages default to startup
        return "startup"

    def _print(self, message: str) -> None:
        """Print message to output stream.

        Args:
            message: Message to print.
        """
        print(message, file=self.output)
        self.output.flush
