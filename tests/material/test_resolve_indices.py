import pytest
import torch

from difftmm.material import Material, resolve_indices


@pytest.fixture
def wvln():
    return torch.tensor([0.45, 0.55, 0.65])


class TestResolveIndices:
    def test_float_spec_broadcasts(self, wvln):
        out = resolve_indices(1.5, wvln, torch.device("cpu"))
        assert out.shape == (3,)
        assert out.dtype == torch.complex64
        torch.testing.assert_close(out.real, torch.full((3,), 1.5))
        torch.testing.assert_close(out.imag, torch.zeros(3))

    def test_complex_spec_broadcasts(self, wvln):
        out = resolve_indices(1.5 + 0.01j, wvln, torch.device("cpu"))
        torch.testing.assert_close(out.imag, torch.full((3,), 0.01))

    def test_string_spec_creates_material(self, wvln):
        out = resolve_indices("SiO2", wvln, torch.device("cpu"))
        assert out.dtype == torch.complex64
        # SiO2 should have n ~1.46 in visible
        assert (out.real > 1.4).all() and (out.real < 1.5).all()

    def test_material_spec_calls_ior(self, wvln):
        mat = Material("air")
        out = resolve_indices(mat, wvln, torch.device("cpu"))
        torch.testing.assert_close(out.real, torch.ones(3))

    def test_tuple_spec_selects_axis(self, wvln):
        spec = (1.0, 1.5, 2.0)
        out_x = resolve_indices(spec, wvln, torch.device("cpu"), axis=0)
        out_y = resolve_indices(spec, wvln, torch.device("cpu"), axis=1)
        out_z = resolve_indices(spec, wvln, torch.device("cpu"), axis=2)
        assert out_x.real[0].item() == 1.0
        assert out_y.real[0].item() == 1.5
        assert out_z.real[0].item() == 2.0
