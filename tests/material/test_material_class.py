import pytest
import torch

from difftmm.material import Material


class TestAirMaterial:
    def test_default_name_is_air(self):
        mat = Material()
        assert mat.name == "air"
        assert mat.dispersion == "sellmeier"

    @pytest.mark.parametrize("alias", ["air", "AIR", "Air"])
    def test_air_name_normalizes(self, alias):
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
        # PLASTIC2022.AGF entries are mostly mode=1 (Schott) — not supported in v1.
        with pytest.raises(NotImplementedError):
            Material("PMMA")


class TestInterpNK:
    def test_sio2_zero_extinction(self, wvln_vis):
        mat = Material("SiO2")
        assert mat.dispersion == "interp"
        n = mat.ior(wvln_vis)
        torch.testing.assert_close(n.imag, torch.zeros_like(wvln_vis))

    def test_silver_has_extinction(self):
        mat = Material("Ag")
        n = mat.ior(torch.tensor([0.55]))
        assert n.imag.item() > 0.5  # k is large for Ag in visible

    def test_silver_lowercase_lookup(self):
        # NK_TABLE materials are looked up case-insensitively
        m1 = Material("Ag")
        m2 = Material("ag")
        assert m1.dispersion == m2.dispersion
        # Same table => identical output
        wvln = torch.tensor([0.55])
        torch.testing.assert_close(m1.ior(wvln), m2.ior(wvln))


class TestMaterialDeviceAndGrad:
    def test_to_device_moves_cached_tensors(self):
        mat = Material("SiO2")
        out_cpu = mat.ior(torch.tensor([0.55])).device
        assert out_cpu.type == "cpu"
        # to() should be a no-op for cpu->cpu
        mat.to("cpu")
        assert mat._ref_wvlns.device.type == "cpu"

    def test_autograd_through_sellmeier(self):
        mat = Material("N-BK7")
        wvln = torch.tensor([0.55], requires_grad=True)
        n = mat.ior(wvln)
        n.real.sum().backward()
        assert wvln.grad is not None
        assert torch.isfinite(wvln.grad).all()

    def test_autograd_through_interp(self):
        mat = Material("SiO2")
        wvln = torch.tensor([0.55], requires_grad=True)
        n = mat.ior(wvln)
        n.real.sum().backward()
        assert wvln.grad is not None
        assert torch.isfinite(wvln.grad).all()


class TestHelpers:
    def test_list_materials_returns_known_names(self):
        from difftmm.material import list_materials
        names = list_materials()
        assert "air" in names
        assert "n-bk7" in names  # from AGF
        assert "sio2" in names  # from thin_film_materials INTERP_NK_TABLE (case-folded)

    def test_refractive_index_scalar_input(self):
        mat = Material("air")
        result = mat.refractive_index(0.55)
        assert isinstance(result, complex)
        assert result.real == pytest.approx(1.0)

    def test_refractive_index_tensor_input(self, wvln_vis):
        mat = Material("air")
        result = mat.refractive_index(wvln_vis)
        assert torch.is_tensor(result)
        assert result.dtype == torch.complex64
