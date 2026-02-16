# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Script for building intermediate data."""

from importlib import resources
import logging
import pathlib
import site
import sys
import threading

import alphafold3.constants.converters
from alphafold3.constants.converters import ccd_pickle_gen
from alphafold3.constants.converters import chemical_component_sets_gen

logger = logging.getLogger(__name__)

_ensure_lock = threading.Lock()
_data_ensured = False


def _find_components_cif() -> pathlib.Path | None:
  """Find components.cif from rdkit/libcifpp installation."""
  search_roots = list(site.getsitepackages())
  # Also check the venv prefix (not always in getsitepackages)
  venv_site = pathlib.Path(sys.prefix) / 'lib'
  for child in venv_site.glob('python*/site-packages'):
    search_roots.append(str(child))

  for site_path in search_roots:
    for subdir in ('share/libcifpp', 'var/cache/libcifpp'):
      path = pathlib.Path(site_path) / subdir / 'components.cif'
      if path.exists():
        return path
  return None


def _pickle_paths() -> tuple[pathlib.Path, pathlib.Path]:
  """Return (ccd_pickle_path, chemical_component_sets_pickle_path)."""
  out_root = resources.files(alphafold3.constants.converters)
  ccd = pathlib.Path(str(out_root.joinpath('ccd.pickle')))
  sets = pathlib.Path(str(out_root.joinpath('chemical_component_sets.pickle')))
  return ccd, sets


def ensure_data() -> None:
  """Ensure CCD pickle files exist, generating them if needed.

  Thread-safe; only runs generation once per process. Broken symlinks
  (e.g. from a developer checkout pointing at a non-existent path) are
  removed before regeneration.
  """
  global _data_ensured
  if _data_ensured:
    return

  with _ensure_lock:
    if _data_ensured:
      return

    ccd_path, sets_path = _pickle_paths()

    if ccd_path.exists() and sets_path.exists():
      _data_ensured = True
      return

    # Remove broken symlinks so open(..., 'wb') doesn't follow them
    for p in (ccd_path, sets_path):
      if p.is_symlink() and not p.exists():
        p.unlink()

    cif_path = _find_components_cif()
    if cif_path is None:
      # Can't auto-generate — let the caller hit the original
      # FileNotFoundError with a helpful log message.
      logger.warning(
          'CCD pickle files are missing and components.cif was not found. '
          'Run `build_data` after installing dependencies (pip install -e .).'
      )
      return

    logger.info(
        'CCD pickle files missing — generating from %s (this is a one-time '
        'operation and may take a minute)...', cif_path,
    )
    build_data()
    _data_ensured = True
    logger.info('CCD pickle files generated successfully.')


def build_data():
  """Builds intermediate data."""
  cif_path = _find_components_cif()
  if cif_path is None:
    raise ValueError(
        'Could not find components.cif. Ensure rdkit is installed '
        '(pip install -e .) so that libcifpp data is available.'
    )

  ccd_pickle_path, chemical_component_sets_pickle_path = _pickle_paths()
  ccd_pickle_gen.main(['', str(cif_path), str(ccd_pickle_path)])
  chemical_component_sets_gen.main(
      ['', str(chemical_component_sets_pickle_path)]
  )


if __name__ == '__main__':
  build_data()
