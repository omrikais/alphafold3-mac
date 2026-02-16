# Web Interface Guide

This guide documents the full web UI shipped with the AlphaFold3 Mac Port.

## Navigation and Status Bar

The top navigation bar is available on every page and includes:

- **Home / reset**: clicking the logo returns to the home page and resets the current prediction form.
- **System status badge**: shows model availability and hardware summary.
- **Theme toggle**: switches between light and dark themes.

## Home Page: Submit Predictions

The home page is split into two main areas:

- **Prediction setup** (left/main)
- **Job history** (right or below on smaller screens)

### Prediction Setup

The prediction setup workflow includes these UI elements.

1. **Job metadata controls**
- Job name field (with random-name helper).
- Seed controls (manual seed or auto-seed behavior).
- Data pipeline toggle to enable/disable MSA/template search.

2. **Input import tools**
- **PDB fetch** panel for importing structures by 4-character PDB ID.
- **File upload** for JSON/PDB/CIF/mmCIF inputs.
- Import warnings/error banner when parsing fails or fields are adjusted.

3. **Entity builder**
- Add entities by type: protein, RNA, DNA, ligand, ion.
- Reorder entity cards with drag-and-drop.
- Configure copy count and delete entities.
- Per-entity input controls:
  - Protein/RNA/DNA sequence editor with validation and residue counters.
  - Protein PTM editor with residue grid and per-residue PTM selection.
  - Ligand editor with CCD code mode or SMILES mode.
  - Ion selector for common ion types.

4. **Restraint editor (restraint-guided docking)**
- Add/remove restraints by category:
  - Distance restraints
  - Contact restraints
  - Repulsive restraints
- Per-restraint controls include chain, residue, atom, target/sigma/threshold, and weight fields.
- Guidance controls include guidance scale, annealing mode, and diffusion start/end steps.
- Validation warnings appear inline (for example, missing chains or invalid ranges).

5. **Preview modal before submission**
- Job summary: name, seeds, pipeline mode, chain/residue counts, and entity list.
- MSA cache status:
  - Checking
  - Cache hit (with size/timestamp)
  - Cache miss
- Optional cache usage toggle.
- Final submit/cancel actions.

### Job History Panel

The job history UI includes:

- Search box
- Status filters (all/running/completed/failed/queued)
- Pagination controls
- Job cards showing:
  - Status badge
  - Time metadata
  - Chain/residue/sample counts
  - Progress bar for active jobs
  - Inline error details for failed jobs
  - Actions: cancel, reuse inputs, delete (with confirmation)

## Job Page: Progress and Results

When a job is selected, the `/job` view shows either live progress or completed results.

### Live Progress View

For pending/running jobs, the page shows:

- Job metadata header (name, status, chain/residue/sample counts)
- Cancel button (when cancellation is valid)
- Stage indicator across pipeline/model phases
- Progress bar with percentage and stage-specific messaging
- Error banner for failed jobs

The page receives progress events over WebSocket and updates in real time.

### Completed Results View

For completed jobs, the page provides the following sections.

1. **Results header**
- Back to jobs
- Reuse inputs
- Job metadata and sample selector

2. **Confidence summary cards**
- pTM
- ipTM (complexes)
- pLDDT

3. **3D structure viewer**
- Interactive Mol* visualization of the selected CIF.
- Restraint overlays (distance/contact/repulsive) with pass/fail coloring.

4. **Analysis plots**
- Per-residue pLDDT bar chart with confidence bands.
- PAE heatmap with hover tooltips.

5. **Sidebar actions and tables**
- Downloads:
  - Selected sample CIF
  - Confidence JSON
  - ZIP bundle of all outputs
  - Open output directory in Finder (local-only)
- Sample ranking table with pTM/ipTM/pLDDT and top-ranked marker.
- Restraint satisfaction panel:
  - Satisfied/total summary
  - Per-restraint details with expected vs observed distances

## UX Notes

- The UI is optimized for local execution on macOS and expects API access from the same machine by default.
- Validation is performed both client-side and server-side. Use `/api/validate` for preflight validation in automation workflows.
- For keyboard-driven workflows, keep using CLI mode and the API together with this UI.
