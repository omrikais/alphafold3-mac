# Updating

This guide covers how to update an existing AlphaFold 3 installation to the
latest version. Updates bring new features, bug fixes, and performance
improvements.

## Quick Update

Most updates require only three steps:

```bash
cd /path/to/alphafold3-mac
git pull
./scripts/install.sh
```

The installer is idempotent — it detects what is already installed and only
rebuilds what has changed. A typical update takes under two minutes.

!!! tip "When in doubt, re-run the installer"
    Re-running `install.sh` is always safe. It will not overwrite your
    configuration unless you choose to, and it skips steps that are already
    complete (venv creation, HMMER build, etc.).

## Minimal Update (Advanced)

If you know the update does not add new dependencies, you can skip the
installer and update manually:

=== "Full Install (Web UI)"

    ```bash
    cd /path/to/alphafold3-mac
    git pull
    cd frontend && npm run build && cd ..
    ```

    Then restart the server.

=== "CLI-only Install"

    ```bash
    cd /path/to/alphafold3-mac
    git pull
    ```

    No further steps needed — run predictions as usual.

The minimal approach is faster but does not check for new Python or Node.js
dependencies. If anything fails after a minimal update, fall back to the
full update with `./scripts/install.sh`.

## When Extra Steps Are Needed

Certain updates may require additional action. Release notes will call these
out explicitly, but here is a summary of what to watch for:

| Change | What to Do |
|--------|-----------|
| New Python dependency | Re-run `./scripts/install.sh` (handles `pip install` automatically) |
| New Node.js dependency | Re-run `./scripts/install.sh` (handles `npm install` automatically) |
| New configuration key | Edit `~/.alphafold3_mlx/config.env` — see release notes for details |
| Database schema change | Re-run `./scripts/install.sh` — migrations run automatically |
| Model weight update | Download the new weight file and place it in your weights directory |

Most feature updates — including recent additions like restraint-guided
docking — require **none** of the above. They ship as pure code changes with
no new dependencies, configuration, or migrations.

## Checking Your Version

To see which version you are running:

```bash
cd /path/to/alphafold3-mac
git log --oneline -1
```

Or from within Python:

```bash
source .venv/bin/activate
python3 -c "import alphafold3; print(alphafold3.__version__)" 2>/dev/null \
    || echo "Version not available (pre-release build)"
```

## Preserving Local Changes

If you have made local modifications to the code:

```bash
# Stash your changes before pulling
git stash

# Pull the update
git pull

# Re-apply your changes
git stash pop
```

If there are conflicts, Git will mark them in the affected files. Resolve
the conflicts, then re-run the installer if needed.

## Restart After Updating

After any update, restart the server to pick up code changes:

=== "Desktop Launcher"

    Close the Terminal window running the server, then double-click
    **AlphaFold3** on your Desktop.

=== "Terminal"

    Press `Ctrl+C` to stop the running server, then:

    ```bash
    ./scripts/start.sh
    ```

=== "CLI Users"

    No server to restart. Just run your next prediction — it will use the
    updated code automatically.

## Rollback

If an update causes problems, you can revert to any previous version:

```bash
cd /path/to/alphafold3-mac
git log --oneline -10          # find the commit you want
git checkout <commit-hash>     # switch to that version
./scripts/install.sh           # rebuild if needed
```

!!! warning
    After rolling back, do not run `git pull` until you are ready to
    update again. Use `git checkout main` to return to the latest version.
