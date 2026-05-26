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

    def test_s_til6_json_only_material(self):
        """s-til6 exists only in JSON SELLMEIER_TABLE, not in any AGF catalog."""
        mat = Material("s-til6")
        assert mat.dispersion == "sellmeier"
        assert abs(mat.n - 1.5317) < 1e-4
        assert abs(mat.V - 48.84) < 1e-2
        wvln = torch.tensor([0.5876])
        n = mat.ior(wvln).real.item()
        assert abs(n - 1.5317) < 1e-3


class TestInterpRealN:
    def test_fused_silica_at_d_line(self):
        # 'fused_silica' is in materials_data.json INTERP_TABLE
        mat = Material("fused_silica")
        assert mat.dispersion == "interp"
        wvln = torch.tensor([0.5876])
        n = mat.ior(wvln)
        # At 0.5876, table values bracket; the result should be a real value ~1.46
        assert n.dtype == torch.complex64
        assert abs(n.real.item() - 1.4596) < 0.005
        assert n.imag.item() == pytest.approx(0.0)

    def test_interp_vector_output(self, wvln_vis):
        mat = Material("fused_silica")
        n = mat.ior(wvln_vis)
        assert n.shape == wvln_vis.shape
        assert n.dtype == torch.complex64

    def test_interp_endpoints_match_table(self):
        mat = Material("fused_silica")
        # First table point is (0.40, 1.4701)
        wvln = torch.tensor([0.40])
        assert abs(mat.ior(wvln).real.item() - 1.4701) < 1e-4


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
