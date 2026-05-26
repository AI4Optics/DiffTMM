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
