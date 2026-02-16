"""Golden outputs dataclass for reference testing."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.inputs import AttentionInputs


@dataclass
class GoldenOutputs:
    """Persisted golden outputs from JAX reference harness.

    Attributes:
        case_id: Unique identifier for this test case
        description: Human-readable description
        inputs: Input tensors used
        output: Reference output array
        intermediates: Dictionary of intermediate arrays
        config: Configuration used to generate outputs
        seed: Random seed used
        numpy_version: NumPy version used for generation
        rtol: Relative tolerance for this case
        atol: Absolute tolerance for this case
    """

    case_id: str
    description: str
    inputs: AttentionInputs
    output: np.ndarray
    intermediates: dict[str, np.ndarray]
    config: AttentionConfig
    seed: int
    numpy_version: str
    rtol: float
    atol: float

    @classmethod
    def load(cls, npz_path: Path, meta_path: Path | None = None) -> "GoldenOutputs":
        """Load from NPZ + JSON metadata files.

        Args:
            npz_path: Path to NPZ file
            meta_path: Path to metadata JSON (defaults to {npz_path}.meta.json)

        Returns:
            GoldenOutputs instance
        """
        if meta_path is None:
            meta_path = npz_path.with_suffix(".meta.json")

        # Load NPZ arrays
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = dict(data)

        # Load metadata
        with open(meta_path) as f:
            meta = json.load(f)

        # Reconstruct inputs
        inputs = AttentionInputs(
            q=arrays["q"],
            k=arrays["k"],
            v=arrays["v"],
            boolean_mask=arrays.get("boolean_mask"),
            additive_bias=arrays.get("additive_bias"),
        )

        # Reconstruct intermediates
        intermediates = {}
        for key in ["logits_pre_mask", "logits_masked", "weights"]:
            if key in arrays:
                intermediates[key] = arrays[key]

        return cls(
            case_id=meta["case_id"],
            description=meta["description"],
            inputs=inputs,
            output=arrays["output"],
            intermediates=intermediates,
            config=AttentionConfig.from_dict(meta["config"]),
            seed=meta["seed"],
            numpy_version=meta["numpy_version"],
            rtol=meta["rtol"],
            atol=meta["atol"],
        )

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        """Save to NPZ + JSON metadata files.

        Args:
            output_dir: Directory to save files

        Returns:
            Tuple of (npz_path, meta_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        npz_path = output_dir / f"{self.case_id}.npz"
        meta_path = output_dir / f"{self.case_id}.meta.json"

        # Prepare arrays for saving
        arrays = {
            "q": np.ascontiguousarray(self.inputs.q),
            "k": np.ascontiguousarray(self.inputs.k),
            "v": np.ascontiguousarray(self.inputs.v),
            "output": np.ascontiguousarray(self.output),
        }

        if self.inputs.boolean_mask is not None:
            arrays["boolean_mask"] = np.ascontiguousarray(self.inputs.boolean_mask)
        if self.inputs.additive_bias is not None:
            arrays["additive_bias"] = np.ascontiguousarray(self.inputs.additive_bias)

        for key, arr in self.intermediates.items():
            arrays[key] = np.ascontiguousarray(arr)

        # Save NPZ
        np.savez_compressed(npz_path, **arrays)

        # Prepare metadata
        meta = {
            "case_id": self.case_id,
            "description": self.description,
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
            "config": self.config.to_dict(),
            "seed": self.seed,
            "numpy_version": self.numpy_version,
            "rtol": self.rtol,
            "atol": self.atol,
        }

        # Save metadata
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return npz_path, meta_path
