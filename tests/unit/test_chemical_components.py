"""Regression tests for CCD overrides used by golden data tests."""

from alphafold3.common import resources
from alphafold3.constants import chemical_components


def test_user_ccd_does_not_mutate_cached_dictionary(monkeypatch):
    cached = {"7BU": {"_chem_comp_atom.charge": ("-1",)}}
    monkeypatch.setattr(
        chemical_components, "_load_ccd_pickle_cached", lambda _: cached
    )
    old_7bu = (
        resources.ROOT / "test_data/7bu_ccd_2021.cif"
    ).read_text()

    overridden = chemical_components.Ccd(user_ccd=old_7bu)

    assert overridden["7BU"]["_chem_comp_atom.charge"][27] == "0"
    assert chemical_components.Ccd()["7BU"] is cached["7BU"]
    assert cached["7BU"]["_chem_comp_atom.charge"] == ("-1",)
