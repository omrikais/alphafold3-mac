# Installation

This guide walks through installing AlphaFold 3 on an Apple Silicon Mac using
the interactive installer. The installer handles all dependencies, builds the
required tools, and creates a ready-to-use server configuration.

## Prerequisites

### Hardware

AlphaFold 3 runs natively on Apple Silicon Macs. The following chips are
supported:

| Chip | Unified Memory | Recommended For |
|------|---------------|-----------------|
| M2/M3/M4 Max | 64--128 GB | Small to medium proteins (up to ~2,000 tokens) |
| M2/M3 Ultra | 192 GB | Large proteins (up to ~3,500 tokens) |
| M4 Ultra | up to 512 GB | Large complexes (up to ~5,000+ tokens) |

!!! warning "Intel Macs are not supported"
    AlphaFold 3 requires Apple Silicon (M2 or later). The installer will check
    for this and exit if run on an Intel Mac.

### Software

The installer checks for and, where possible, automatically installs these
dependencies:

| Dependency | Version | Notes |
|-----------|---------|-------|
| macOS | ARM64 (Apple Silicon) | Required |
| Xcode Command Line Tools | Any | Auto-installed if missing |
| Python | 3.12+ | `brew install python@3.12` if not found |
| Homebrew | Any | Auto-installed if missing |
| autoconf + automake | Any | Auto-installed via Homebrew |
| Node.js | 20+ | Only required for full install (Web UI). `brew install node@20` |

!!! tip "CLI-only mode"
    If you don't need the web interface, choose **CLI-only** mode during
    installation. This skips the Node.js requirement entirely.

### Disk Space

| Component | Size |
|-----------|------|
| Repository + dependencies | ~3 GB |
| Model weights | ~1 GB |
| Genetic databases (optional) | ~500 GB |
| Job outputs | Varies |

At least 5 GB of free disk space is required. The installer will warn if
disk space is low.

## Step 1: Obtain Model Weights

AlphaFold 3 model weights are provided by Google DeepMind under a separate
license. You must request access before installing.

1. Complete the [model parameters request form](https://forms.gle/svvpY4u2jsHEwWYS6).
2. Google DeepMind will review your request (typically 2--3 business days).
3. Once approved, download the weight file (`af3.bin.zst`, approximately 973 MB).
4. Place it in a directory of your choice. The default location is:

```
~/.alphafold3/weights/model/af3.bin.zst
```

!!! note "You can install without weights"
    The installer will proceed even if weights are not yet available. The
    server will start, but prediction jobs will fail until you place the
    weight file in the configured directory.

### Recognized Weight Formats

The installer and runtime accept any of these file patterns:

- `af3.bin.zst` (single compressed file)
- `af3.bin` (uncompressed)
- `af3.0.bin.zst` (sharded, numbered)
- `af3.bin.zst.0`, `af3.bin.zst.1`, ... (split shards)

## Step 2: Clone the Repository

```bash
git clone https://github.com/omrikais/alphafold3-mac.git
cd alphafold3-mac
```

## Step 3: Run the Installer

There are two ways to start the installer:

=== "Terminal"

    ```bash
    ./scripts/install.sh
    ```

=== "Finder (double-click)"

    Double-click the `install.command` file in the repository root. This opens
    a Terminal window and runs the installer.

### Dry Run

To preview what the installer will do without making any changes:

```bash
./scripts/install.sh --dry-run
```

This exercises all prompts and validation logic but skips actual installations
and file writes.

### Install Mode Selection

The installer presents two modes:

| Mode | What's Installed | Requirements |
|------|-----------------|--------------|
| **Full install** (recommended) | Model + CLI + Web UI | Node.js >= 20 |
| **CLI-only install** | Model + CLI | No Node.js needed |

Choose **Full install** to get the web interface with 3D structure visualization,
job management, and real-time progress tracking. Choose **CLI-only** if you
prefer command-line workflows or don't have Node.js installed.

### Configuration Prompts

The installer asks for four pieces of configuration:

#### Model Weights Directory

```
Enter weights directory [~/.alphafold3/weights/model]:
```

Point this to the directory containing your downloaded `af3.bin.zst` file.
The installer validates that a recognized weight file exists in this directory.

#### Genetic Databases (Optional)

The installer presents three options for database setup:

| Option | Description |
|--------|-------------|
| **[1] I already have the databases** | Enter the path to your existing databases directory |
| **[2] Download databases now** | Downloads all databases from Google Cloud Storage (~252 GB download, ~630 GB on disk) |
| **[3] Skip for now** (default) | Sequence-only mode; you can download databases later |

**Option 1** validates that the directory contains the required database files:

- `uniref90_2022_05.fa`
- `mgy_clusters_2022_05.fa`
- `bfd-first_non_consensus_sequences.fasta`
- `uniprot_all_2021_04.fa`
- `pdb_seqres_2022_09_28.fasta`
- `mmcif_files/` (or `pdb_2022_09_28_mmcif_files/mmcif_files/`)

RNA databases are checked but are optional:

- `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta`
- `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta`
- `rnacentral_active_seq_id_90_cov_80_linclust.fasta`

**Option 2** prompts for a target directory (default `~/public_databases`),
checks disk space (warns if less than 650 GB free), verifies that `curl`,
`tar`, and `zstd` are available (offering to install `zstd` via Homebrew if
missing), and then downloads all databases after the main installation steps
complete. Downloads are resumable -- if interrupted, re-run the installer
and choose option 2 with the same directory, or run the fetch script directly:

```bash
bash fetch_databases.sh ~/public_databases
```

If the download is incomplete, the installer defaults to clearing the
database path so the server starts in sequence-only mode. You can
explicitly choose to keep a partial path if you plan to finish downloading
later.

**Option 3** skips database setup entirely. The server runs in sequence-only
mode, which still produces high-quality predictions for many cases.

#### Data Directory

```
Enter data directory [~/.alphafold3_mlx/data]:
```

This is where job history (SQLite database), prediction outputs, and MSA cache
files are stored. The default is `~/.alphafold3_mlx/data`.

#### Server Port

```
Enter port number [8642]:
```

The port for the web server. Must be between 1024 and 65535. The installer
checks that the port is not already in use.

### What the Installer Does

After collecting configuration, the installer runs these steps:

| Step | Description | Time |
|------|-------------|------|
| 1 | Install [uv](https://docs.astral.sh/uv/) package manager | ~10 seconds |
| 2 | Create Python virtual environment (Python 3.12) | ~5 seconds |
| 3 | Install Python dependencies | ~1--2 minutes |
| 4 | Build chemical components database (CCD) | ~1--2 minutes |
| 5 | Build HMMER 3.4 with seq_limit patch | ~2--3 minutes |
| 6 | Install frontend dependencies (full mode only) | ~30 seconds |
| 7 | Build web interface (full mode only) | ~30 seconds |
| 8 | Download genetic databases (if option 2 was chosen) | Hours (network-dependent) |

Each step shows a progress indicator and elapsed time. If a step fails, the
installer offers to retry it before exiting.

### What Gets Created

After a successful installation, the following files and directories exist:

| Path | Description |
|------|-------------|
| `.venv/` | Python virtual environment (in the repository) |
| `~/.alphafold3_mlx/config.env` | Persistent configuration file |
| `~/.alphafold3_mlx/hmmer/bin/` | HMMER binaries (jackhmmer, hmmsearch, etc.) |
| `~/.alphafold3_mlx/data/` | Job database and outputs |
| `frontend/out/` | Built web interface (full mode only) |
| `~/Desktop/AlphaFold3.command` | Desktop launcher (full mode only) |

## Step 4: Start the Server

After installation, there are several ways to start the server.

=== "Desktop Launcher (Full Install)"

    Double-click **AlphaFold3** on your Desktop. This opens a Terminal window,
    loads your configuration, and starts the server. Close the Terminal window
    to stop the server.

=== "Terminal"

    ```bash
    cd /path/to/alphafold3-mac
    ./scripts/start.sh
    ```

=== "Direct Python Command"

    ```bash
    cd /path/to/alphafold3-mac
    source .venv/bin/activate
    python3 -m alphafold3_mlx.api --port 8642
    ```

When the server starts, you'll see:

```
  AlphaFold 3 MLX
  ────────────────────────────────
  Web UI:  http://127.0.0.1:8642
  API:     http://127.0.0.1:8642/api

  Press Ctrl+C to stop the server.
```

Open `http://127.0.0.1:8642` in your browser to access the web interface.

### Server Command-Line Options

When using `python3 -m alphafold3_mlx.api` directly, the following options are
available:

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | `AF3_WEIGHTS_DIR` env, then `~/.alphafold3/weights/model` | Model weights directory |
| `--data-dir` | `AF3_DATA_DIR` env, then `~/.alphafold3_mlx/data` | Job and output storage |
| `--db-dir` | `AF3_DB_DIR` env | Genetic databases (enables MSA search) |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8642` | Bind port |
| `--num-samples` | `5` | Default structure samples per job |
| `--diffusion-steps` | `200` | Default diffusion steps |
| `--precision` | Auto | `float32`, `float16`, or `bfloat16` |
| `--run-data-pipeline` | Auto | Enable MSA/template search (auto-enabled if `--db-dir` is set) |
| `--api-key` | `AF3_API_KEY` env | Require Bearer token authentication |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

!!! tip "Using `start.sh` vs direct invocation"
    The `start.sh` script automatically loads your `config.env` and passes the
    appropriate flags. Use `python3 -m alphafold3_mlx.api` directly only if you
    need to override specific settings.

### Stopping the Server

Press `Ctrl+C` in the Terminal window running the server. If you used the
Desktop launcher, simply close the Terminal window.

## Step 5: Verify Installation

Run these checks after first startup to confirm the install is complete.

### 1) API health check

```bash
curl http://127.0.0.1:8642/api/system/status
```

Expected result: JSON including model status, queue status, and hardware info.

### 2) Input validation check

```bash
curl -X POST http://127.0.0.1:8642/api/validate \
  -H "Content-Type: application/json" \
  -d @examples/desi1_monomer.json
```

Expected result: JSON with `valid: true` for a valid sample input.

### 3) CLI smoke test

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_monomer.json \
  --output_dir output/install_verify \
  --num_samples 1 \
  --diffusion_steps 20
```

Expected result: successful completion and output files under `output/install_verify`.

## Using the CLI Instead

If you prefer command-line workflows, you can run predictions directly without
the web server:

```bash
cd /path/to/alphafold3-mac
./scripts/alphafold3.sh \
    --input examples/desi1_monomer.json \
    --output_dir output/my_prediction
```

Or equivalently:

```bash
source .venv/bin/activate
python3 run_alphafold_mlx.py \
    --input examples/desi1_monomer.json \
    --output_dir output/my_prediction
```

The `alphafold3.sh` wrapper automatically loads your `config.env` (setting
weight paths, HMMER paths, etc.) before running the CLI.

See the [CLI Reference](../reference/cli.md) for all available options.

## Obtaining Genetic Databases (Optional)

Genetic databases enable MSA (Multiple Sequence Alignment) search, which
can improve prediction quality. Without databases, AlphaFold 3 runs in
**sequence-only mode** using placeholder MSA data.

!!! info "Sequence-only mode still works well"
    Sequence-only mode produces high-quality predictions for many cases.
    Databases are most beneficial for novel or divergent sequences where
    evolutionary information helps.

### Downloading Databases

The easiest way to download databases is to choose **option 2** during
installation. The installer handles disk space checks, tool dependencies,
and download progress automatically.

If you skipped databases during installation or need to download them
separately, use the provided fetch script:

```bash
bash fetch_databases.sh ~/public_databases
```

The script supports resuming interrupted downloads -- files that already
exist are skipped automatically. If a download was interrupted mid-file,
re-running the script resumes from where it left off.

This downloads and decompresses the following from Google Cloud Storage:

| Database | Description |
|----------|-------------|
| UniRef90 | Clustered UniProt sequences |
| BFD | Big Fantastic Database (metagenomic sequences) |
| MGnify | Metagenomic protein database |
| UniProt | Comprehensive protein database |
| PDB seqres | PDB sequence database |
| PDB mmCIF | PDB structure files (~200k files) |
| NT RNA | NCBI nucleotide database (for RNA) |
| RFam | RNA families database |
| RNACentral | Non-coding RNA database |

!!! warning "Storage requirements"
    The download is approximately 252 GB and expands to roughly 630 GB.
    SSD storage is strongly recommended for search performance.

### Recovering from Interrupted Downloads

If the database download was interrupted (by Ctrl+C, network loss, or
system sleep), you have two options:

1. **Re-run the installer** and choose option 2 with the same target
   directory. Completed files are skipped automatically.

2. **Run the fetch script directly:**

    ```bash
    bash fetch_databases.sh ~/public_databases
    ```

After the download completes, update your configuration:

```bash
nano ~/.alphafold3_mlx/config.env
# Set: AF3_DB_DIR="/path/to/public_databases"
```

Then restart the server.

### Configuring Database Paths

If you skipped databases during installation and want to add them later, edit
your configuration file:

```bash
nano ~/.alphafold3_mlx/config.env
```

Set the `AF3_DB_DIR` value to your databases directory:

```
AF3_DB_DIR="/path/to/public_databases"
```

Then restart the server. When `AF3_DB_DIR` is set, the server automatically
enables the data pipeline for MSA and template search.

### Per-Database Environment Variables

For advanced setups where databases are split across multiple locations, you
can set individual database paths:

| Variable | Database |
|----------|----------|
| `AF3_UNIREF90_DB` | UniRef90 |
| `AF3_MGNIFY_DB` | MGnify |
| `AF3_SMALL_BFD_DB` | BFD |
| `AF3_UNIPROT_DB` | UniProt |
| `AF3_PDB_SEQRES_DB` | PDB sequence database |
| `AF3_PDB_MMCIF_DIR` | PDB mmCIF directory |

These override the automatic lookup from `AF3_DB_DIR`.

## Configuration Reference

All configuration is stored in `~/.alphafold3_mlx/config.env`. This file is
created by the installer and loaded by `start.sh` and `alphafold3.sh`.

### Config File Format

```bash
# AlphaFold 3 MLX Configuration
# Edit this file to change settings, then restart the server.

AF3_WEIGHTS_DIR="/Users/you/.alphafold3/weights/model"
AF3_DATA_DIR="/Users/you/.alphafold3_mlx/data"
AF3_DB_DIR=""
AF3_PORT="8642"
AF3_INSTALL_DIR="/Users/you/alphafold3-mac"
AF3_JACKHMMER="/Users/you/.alphafold3_mlx/hmmer/bin/jackhmmer"
AF3_HMMSEARCH="/Users/you/.alphafold3_mlx/hmmer/bin/hmmsearch"
AF3_HMMBUILD="/Users/you/.alphafold3_mlx/hmmer/bin/hmmbuild"
AF3_NHMMER="/Users/you/.alphafold3_mlx/hmmer/bin/nhmmer"
AF3_HMMALIGN="/Users/you/.alphafold3_mlx/hmmer/bin/hmmalign"
```

### Config Keys

| Key | Description |
|-----|-------------|
| `AF3_WEIGHTS_DIR` | Directory containing `af3.bin.zst` model weights |
| `AF3_DATA_DIR` | Directory for job database, outputs, and MSA cache |
| `AF3_DB_DIR` | Genetic databases root (empty = sequence-only mode) |
| `AF3_PORT` | Server port number |
| `AF3_INSTALL_DIR` | Path to the cloned repository |
| `AF3_JACKHMMER` | Path to the jackhmmer binary |
| `AF3_HMMSEARCH` | Path to the hmmsearch binary |
| `AF3_HMMBUILD` | Path to the hmmbuild binary |
| `AF3_NHMMER` | Path to the nhmmer binary |
| `AF3_HMMALIGN` | Path to the hmmalign binary |

To apply changes, edit the file and restart the server.

## API Authentication (Optional)

To protect your server with Bearer token authentication:

1. Set the `AF3_API_KEY` environment variable before starting the server:

    ```bash
    export AF3_API_KEY="your-secret-token"
    ./scripts/start.sh
    ```

2. Or pass it directly:

    ```bash
    source .venv/bin/activate
    python3 -m alphafold3_mlx.api --api-key "your-secret-token"
    ```

When enabled, all API requests must include the token:

```bash
curl -H "Authorization: Bearer your-secret-token" \
     http://127.0.0.1:8642/api/system/status
```

Or as a query parameter: `http://127.0.0.1:8642/api/system/status?token=your-secret-token`

## Development Mode

For contributors or developers who want to modify the web interface with
hot-reload:

```bash
./scripts/dev.sh
```

This starts:

- **FastAPI backend** on port `8642` (with auto-reload)
- **Next.js dev server** on port `3001` (with hot module replacement)

Open `http://localhost:3001` for the development frontend. Logs are written
to `.dev-logs/fastapi.log` and `.dev-logs/nextjs.log`.

To stop both servers:

```bash
./scripts/dev-stop.sh
```

## Re-running the Installer

It is safe to re-run the installer at any time. It will:

- Skip steps that are already complete (existing venv, already-installed tools)
- Overwrite `~/.alphafold3_mlx/config.env` with new settings
- Rebuild the frontend if in full mode

```bash
./scripts/install.sh
```

## Troubleshooting

### "Python 3.12+ required"

Install Python 3.12 via Homebrew:

```bash
brew install python@3.12
```

Then re-run the installer. It will detect `python3.12` automatically.

!!! warning "Python 3.14 is not supported"
    Several dependencies (notably rdkit) do not yet have Python 3.14 wheels.
    Use Python 3.12 or 3.13.

### "Node.js >= 20 required"

Install Node.js 20 via Homebrew:

```bash
brew install node@20
```

Or choose **CLI-only** mode during installation to skip the Node.js requirement.

### "Port already in use"

Another process is using the configured port. Either:

- Stop the process using that port
- Choose a different port during installation
- Edit `~/.alphafold3_mlx/config.env` and change `AF3_PORT`

### HMMER Build Fails

If the HMMER build step fails, try building it manually:

```bash
./scripts/build_hmmer_macos.sh --prefix=$HOME/.alphafold3_mlx/hmmer
```

This requires `autoconf` and `automake` (installed via Homebrew).

### Server Starts but Jobs Fail

Check that model weights are in place:

```bash
ls ~/.alphafold3/weights/model/af3.bin.zst
```

If the file doesn't exist, download it from Google DeepMind (see
[Step 1](#step-1-obtain-model-weights)) and place it in the configured
weights directory.

### Memory Errors on Large Inputs

For inputs with many tokens (large proteins or complexes), you may run out of
unified memory. Try:

- Using `--precision float16` or `--precision bfloat16` (M3/M4) to halve
  memory usage
- Reducing `--num-samples` to 1
- Closing other memory-intensive applications

See [Performance](../user-guide/performance.md) for memory optimization
guidance.

## Next Steps

- [Quickstart](quickstart.md) -- Run your first prediction
- [Input Format](../user-guide/input-format.md) -- Learn how to prepare input
  JSON files
- [Output Format](../user-guide/output-format.md) -- Understand prediction
  outputs
- [Pipeline Setup](pipeline-setup.md) -- Configure MSA and template search
- [CLI Reference](../reference/cli.md) -- Full command-line options
