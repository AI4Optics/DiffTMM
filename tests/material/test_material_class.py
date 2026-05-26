import pytest
import torch

from difftmm.material import Material


class TestAirMaterial:
    def test_default_name_is_air(self):
        mat = Material()
        assert mat.name == "air"
        assert mat.dispersion == "sellmeier"

    @pytest.mark.parametrize("alias", ["air", "AIR", "vacuum", "occluder", "Air"])
    def test_air_aliases_normalize(self, alias):
        assert Material(alias).name == "air"

    def test_air_ior_returns_complex_one(self, wvln_vis):
        mat = Material("air")
        n = mat.ior(wvln_vis)
        assert n.dtype == torch.complex64
        assert n.shape == wvln_vis.shape
        torch.testing.assert_close(n.real, torch.ones_like(wvln_vis))
        torch.testing.assert_close(n.imag, torch.zeros_like(wvln_vis))

    def test_air_n_attribute_is_unity(self):
        assert Material("air").n == pytest.approx(1.0)


class TestAGFLoading:
    def test_n_bk7_loads_as_sellmeier(self):
        mat = Material("N-BK7")
        assert mat.dispersion == "sellmeier"
        # Six Sellmeier coefficients should be populated (non-zero)
        assert any(c != 0.0 for c in [mat.k1, mat.k2, mat.k3, mat.l1, mat.l2, mat.l3])

    def test_n_bk7_d_line_index(self):
        # Published nd for N-BK7 is 1.5168
        mat = Material("N-BK7")
        wvln = torch.tensor([0.5876])  # d-line
        n = mat.ior(wvln).real.item()
        assert abs(n - 1.5168) < 1e-3

    def test_case_insensitive_lookup(self):
        m1 = Material("N-BK7")
        m2 = Material("n-bk7")
        assert m1.n == m2.n

    def test_unknown_material_raises(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            Material("not_a_real_material_xyz")

    def test_schott_mode_entries_are_skipped(self):
        # PLASTIC2022.AGF entries are mostly mode=1 (Schott). They should NOT
        # be loadable as Sellmeier in v1.
        # PMMA is in materials_data.json's SCHOTT_TABLE only — not yet supported,
        # so it should raise.
        with pytest.raises(NotImplementedError):
            Material("PMMA")  # depends: must not match an AGF Sellmeier entry


class TestJSONSellmeier:
    def test_bk7_lowercase_from_json(self):
        # 'bk7' is in materials_data.json SELLMEIER_TABLE
        mat = Material("bk7")
        assert mat.dispersion == "sellmeier"
        wvln = torch.tensor([0.5876])
        n = mat.ior(wvln).real.item()
        assert abs(n - 1.5168) < 1e-3
