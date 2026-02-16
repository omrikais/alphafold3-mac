# Restraint-Guided Docking

Restraint-guided docking steers AlphaFold 3's diffusion process toward
conformations that satisfy experimentally known spatial constraints. This is
useful for multi-chain docking when you have information from crosslinking
mass spectrometry (XL-MS), NMR chemical shift perturbation (CSP), FRET, or
mutagenesis experiments.

## How It Works

AlphaFold 3 generates atomic coordinates through a 200-step denoising
process. At each step, the model predicts clean coordinates from noisy
input, then takes an Euler step along the denoising trajectory.

Restraint guidance adds a small correction to each step, nudging the
trajectory toward structures that satisfy your restraints. The correction is
the gradient of a restraint loss function, computed via automatic
differentiation on the denoised coordinates. This is the same
classifier-guided diffusion technique used in image generation (Dhariwal &
Nichol, 2021), adapted to protein structure space.

The guidance is strongest at low noise levels (late refinement steps) and
weakest at high noise levels (early steps), following the natural
signal-to-noise curve of the diffusion process.

## Restraint Types

### Distance restraints

Specify that two atoms should be a given distance apart. Useful for
crosslinking data (e.g., DSS crosslinks imply CA-CA distance < 30 A) or
NMR NOEs.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chain_i` | string | required | Chain ID of first residue |
| `residue_i` | integer | required | Residue number of first residue |
| `chain_j` | string | required | Chain ID of second residue |
| `residue_j` | integer | required | Residue number of second residue |
| `target_distance` | float | required | Target distance in Angstroms |
| `atom_i` | string | `"CA"` | Atom name on first residue |
| `atom_j` | string | `"CA"` | Atom name on second residue |
| `sigma` | float | `1.0` | Tolerance (Angstroms) -- larger means softer |
| `weight` | float | `1.0` | Relative importance |

Loss: `weight * ((distance - target_distance) / sigma)^2`

### Contact restraints

Specify that a residue should be near *at least one* of several candidate
residues. Useful for NMR CSP data where you know a residue is at the
interface but not its exact partner.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chain_i` | string | required | Chain ID of source residue |
| `residue_i` | integer | required | Residue number of source |
| `candidates` | list | required | List of `{chain_j, residue_j}` targets |
| `threshold` | float | `8.0` | Contact distance threshold (Angstroms) |
| `weight` | float | `1.0` | Relative importance |

Uses CA atoms. The loss is zero when any candidate is within the threshold.

### Repulsive restraints

Specify that two residues should be *at least* a minimum distance apart.
Useful for encoding known non-contacts or preventing steric clashes.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chain_i` | string | required | Chain ID of first residue |
| `residue_i` | integer | required | Residue number of first residue |
| `chain_j` | string | required | Chain ID of second residue |
| `residue_j` | integer | required | Residue number of second residue |
| `min_distance` | float | required | Minimum distance in Angstroms |
| `weight` | float | `1.0` | Relative importance |

Uses CA atoms. Loss: `weight * max(0, min_distance - distance)^2`

## Guidance Parameters

These control how the restraint gradients are applied during diffusion:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale` | `1.0` | Overall guidance strength. Higher values enforce restraints more aggressively but may reduce structural quality. Start with 1.0 and increase if restraints are not satisfied. |
| `annealing` | `"linear"` | How guidance strength changes over diffusion steps. `"linear"` ramps up linearly, `"cosine"` uses a cosine curve, `"constant"` applies uniform strength. |
| `start_step` | `0` | First diffusion step where guidance is active. |
| `end_step` | `null` | Last step (null = all 200 steps). |

## Input Format

Restraints are specified in the input JSON under a top-level `restraints`
key. You can also pass a standalone restraints file via `--restraints` on the
CLI.

### Inline (in input JSON)

```json
{
  "name": "my_docking_job",
  "sequences": [ ... ],
  "restraints": {
    "distance": [
      {
        "chain_i": "A",
        "residue_i": 45,
        "chain_j": "B",
        "residue_j": 12,
        "target_distance": 8.0,
        "sigma": 2.0
      }
    ],
    "contact": [
      {
        "chain_i": "A",
        "residue_i": 72,
        "candidates": [
          {"chain_j": "B", "residue_j": 30},
          {"chain_j": "B", "residue_j": 31},
          {"chain_j": "B", "residue_j": 32}
        ],
        "threshold": 10.0
      }
    ],
    "repulsive": [
      {
        "chain_i": "A",
        "residue_i": 1,
        "chain_j": "B",
        "residue_j": 1,
        "min_distance": 20.0
      }
    ],
    "guidance": {
      "scale": 1.0,
      "annealing": "linear"
    }
  }
}
```

### Standalone file (CLI)

```bash
python3 run_alphafold_mlx.py \
  --input input.json \
  --restraints restraints.json \
  --output_dir output/
```

Where `restraints.json` has the same structure as the `restraints` object
above.

### Web interface

The web UI provides a visual restraint editor under the job submission form.
Add restraints by type, specify chains and residues, and adjust guidance
parameters. Validation warnings appear inline for common errors (missing
chains, out-of-range residues, etc.). See the
[Web Interface](web-interface.md#restraint-editor) page for details.

## Output

After prediction, each sample's confidence output includes a
`restraint_satisfaction` section reporting whether each restraint was
satisfied:

```json
{
  "restraint_satisfaction": {
    "distance": [
      {
        "chain_i": "A", "residue_i": 45,
        "chain_j": "B", "residue_j": 12,
        "target_distance": 8.0,
        "actual_distance": 7.3,
        "satisfied": true
      }
    ],
    "contact": [ ... ],
    "repulsive": [ ... ]
  }
}
```

The 3D structure viewer in the web UI overlays restraint satisfaction
results, color-coded by satisfaction status.

## Tips

- **Start with `scale: 1.0`** and increase only if restraints are not
  satisfied. Values above 5.0 can distort structure quality.
- **Use MSA search** (`--run_data_pipeline`) for best results. Without MSA,
  the model has limited evolutionary context and docking quality degrades
  significantly.
- **Crosslinking data**: For DSS/BS3 crosslinks, use distance restraints
  with `target_distance: 20.0` and `sigma: 5.0` (CA-CA, allowing for lysine
  sidechain flexibility).
- **NMR CSP data**: Use contact restraints with multiple candidates per
  perturbed residue and a threshold of 8-10 A.
- **Sparse restraints are fine**: Even 3-5 distance restraints can
  substantially improve docking orientation for a two-chain complex.
- **Multi-chain coupling**: For complexes with very few inter-chain
  restraints, the system automatically adds a soft center-of-mass coupling
  to prevent chains from drifting apart during diffusion. This activates
  only when chains are more than 30 A apart and has no effect on the final
  aligned RMSD.

## Background

The restraint type taxonomy (distance, contact, repulsive) is inspired by
[ColabDock](https://github.com/JeffSHF/ColabDock) (Feng et al., *Nature
Communications*, 2024), which demonstrated restraint-guided docking for
AlphaFold 2. The mechanism here is fundamentally different: ColabDock
backpropagates through AF2's frozen recycling network to optimize input
logits, while this implementation uses classifier-guided diffusion on AF3's
denoising trajectory. AF3's diffusion architecture makes restraint guidance
particularly natural since guided diffusion is well-studied
(Dhariwal & Nichol, 2021).
