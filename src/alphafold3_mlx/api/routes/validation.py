"""Input validation endpoint (no job created)."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request

from alphafold3_mlx.api.models import ValidationRequest, ValidationResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["validation"])


@router.post("/validate", response_model=ValidationResult)
async def validate_input(body: ValidationRequest, request: Request) -> ValidationResult:
    """Validate input JSON without submitting a job.

    Checks sequence validity, estimates memory requirements, validates
    restraint references, and returns any validation errors.
    """
    from alphafold3_mlx.pipeline.input_handler import (
        parse_input_json,
        validate_input as validate_fold_input,
        estimate_memory_gb,
    )
    from alphafold3_mlx.pipeline.errors import InputError

    # Build AlphaFold Server format JSON
    input_json: dict = {
        "name": body.name,
        "modelSeeds": body.modelSeeds,
        "sequences": body.sequences,
    }
    # Include restraints/guidance so parse_input_json can extract them
    if body.restraints is not None:
        input_json["restraints"] = body.restraints
    if body.guidance is not None:
        input_json["guidance"] = body.guidance

    # Use server-configured diffusion steps as fallback (match jobs route behavior)
    config = request.app.state.api_config
    diffusion_steps = body.diffusionSteps if body.diffusionSteps is not None else config.diffusion_steps

    # Write to a temp file for parse_input_json
    errors: list[str] = []
    num_residues: int | None = None
    num_chains: int | None = None
    estimated_memory: float | None = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_json, f)
            tmp_path = Path(f.name)

        try:
            fold_input = parse_input_json(tmp_path)
            errors = validate_fold_input(fold_input)
            num_residues = fold_input.total_residues
            num_chains = len(fold_input.chain_ids)

            # Validate restraint references if present
            if fold_input._restraints is not None and not fold_input._restraints.is_empty:
                _validate_restraints(
                    fold_input, errors,
                    num_diffusion_steps=diffusion_steps,
                )

            if not errors:
                estimated_memory = estimate_memory_gb(fold_input)
        except InputError as e:
            errors.append(str(e))
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception as e:
        errors.append(f"Validation error: {e}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        estimated_memory_gb=estimated_memory,
        num_residues=num_residues,
        num_chains=num_chains,
    )


def _validate_restraints(
    fold_input,
    errors: list[str],
    num_diffusion_steps: int = 200,
) -> None:
    """Validate restraint references against input sequences.

    Uses the shared validate_restraints() function from restraints.validate
    to ensure API and runtime produce identical error messages.
    """
    from alphafold3_mlx.restraints.validate import (
        build_chain_info_from_input,
        validate_restraints,
    )

    # Use shared chain metadata extraction (protein chains only)
    chains = build_chain_info_from_input(fold_input._input)

    # Collect all chain IDs (protein + non-protein) for better error messages
    all_chain_ids = {chain.id for chain in fold_input._input.chains}

    # Validate using the shared function — same code path as runtime
    restraint_errors = validate_restraints(
        fold_input._restraints,
        chains,
        guidance=fold_input._guidance,
        num_diffusion_steps=num_diffusion_steps,
        all_chain_ids=all_chain_ids,
    )
    errors.extend(restraint_errors)
