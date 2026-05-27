import pytest
import torch

from difftmm import IsotropicFilmSolver, FilmSolver
from difftmm.material import Material


@pytest.fixture
def cpu():
    return torch.device("cpu")


class TestIsotropicSolverMaterials:
    def test_string_inputs(self, cpu):
        solver = IsotropicFilmSolver(
            mat_in="air",
            mat_out="N-BK7",
            mat_ls=["TiO2", "SiO2"],
            thickness_ls=[0.06, 0.10],
            device=cpu,
        )
        theta = torch.tensor([0.0])
        wvln = torch.tensor([0.55])
        ts, tp, rs, rp = solver.simulate(theta=theta, wvln=wvln)
        assert ts.shape == (1, 1, 1)
        assert torch.isfinite(ts).all()

    def test_mixed_scalar_and_material(self, cpu):
        solver = IsotropicFilmSolver(
            mat_in=1.0,
            mat_out=1.52,
            mat_ls=[2.4, Material("SiO2", device=cpu), "TiO2"],
            thickness_ls=[0.06, 0.10, 0.06],
            device=cpu,
        )
        ts, tp, rs, rp = solver.simulate(
            theta=torch.tensor([0.1]),
            wvln=torch.tensor([0.45, 0.55, 0.65]),
        )
        assert ts.shape == (1, 3, 1)

    def test_scalar_only_still_works(self, cpu):
        solver = IsotropicFilmSolver(
            mat_in=1.0,
            mat_out=1.52,
            mat_ls=[2.1, 1.46, 2.1],
            thickness_ls=[0.08, 0.12, 0.08],
            device=cpu,
        )
        ts, _, _, _ = solver.simulate(
            theta=torch.tensor([0.1]),
            wvln=torch.tensor([0.55]),
        )
        assert torch.isfinite(ts).all()

    def test_unknown_material_name_fails_fast(self, cpu):
        with pytest.raises(NotImplementedError):
            IsotropicFilmSolver(
                mat_in="air",
                mat_out=1.52,
                mat_ls=["NotAMaterial"],
                thickness_ls=[0.1],
                device=cpu,
            )


class TestIsotropicCheckpoint:
    def test_roundtrip_material_stack(self, tmp_path, cpu):
        path = tmp_path / "ckpt.pt"
        solver1 = IsotropicFilmSolver(
            mat_in="air",
            mat_out="N-BK7",
            mat_ls=["TiO2", "SiO2"],
            thickness_ls=[0.06, 0.10],
            device=cpu,
        )
        solver1.save_ckpt(path)
        solver2 = IsotropicFilmSolver(
            mat_in="air",
            mat_out="N-BK7",
            mat_ls=["TiO2", "SiO2"],
            thickness_ls=[0.001, 0.001],
            device=cpu,
        )
        solver2.load_ckpt(path)
        torch.testing.assert_close(
            solver1.get_film_thickness(), solver2.get_film_thickness()
        )
        from difftmm.material import Material
        assert isinstance(solver2._n_layer_specs[0], Material)
        assert solver2._n_layer_specs[0].name == "tio2"

    def test_roundtrip_scalar_stack(self, tmp_path, cpu):
        path = tmp_path / "ckpt.pt"
        solver = IsotropicFilmSolver(
            mat_in=1.0,
            mat_out=1.52,
            mat_ls=[2.10, 1.46, 2.10],
            thickness_ls=[0.08, 0.12, 0.08],
            device=cpu,
        )
        solver.save_ckpt(path)
        solver.load_ckpt(path)


class TestAnisotropicSolverIsotropicInputs:
    def test_string_inputs_isotropic_path(self, cpu):
        solver = FilmSolver(
            mat_in="air",
            mat_out="N-BK7",
            mat_ls=["TiO2", "SiO2"],
            thickness_ls=[0.06, 0.10],
            device=cpu,
        )
        ts, tp, rs, rp = solver.simulate(
            theta=torch.tensor([0.0]), wvln=torch.tensor([0.55])
        )
        assert ts.shape == (1, 1, 1)
