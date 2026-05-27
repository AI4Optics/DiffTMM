import torch

from difftmm import FilmSolver
from difftmm.material import Material


def test_three_equal_materials_matches_isotropic():
    cpu = torch.device("cpu")
    solver_iso = FilmSolver(
        mat_n_in="air", mat_n_out="N-BK7",
        mat_n_ls=["SiO2"], thickness_ls=[0.10], device=cpu,
    )
    solver_aniso = FilmSolver(
        mat_n_in="air", mat_n_out="N-BK7",
        mat_n_ls=[("SiO2", "SiO2", "SiO2")], thickness_ls=[0.10], device=cpu,
    )
    theta = torch.tensor([0.1])
    wvln = torch.tensor([0.55])
    ts1, _, _, _ = solver_iso.simulate(theta=theta, wvln=wvln)
    ts2, _, _, _ = solver_aniso.simulate(theta=theta, wvln=wvln)
    torch.testing.assert_close(ts1, ts2, rtol=1e-5, atol=1e-6)


def test_birefringent_layer_produces_polarization_difference():
    cpu = torch.device("cpu")
    solver = FilmSolver(
        mat_n_in="air", mat_n_out=1.52,
        mat_n_ls=[(2.4, 1.5, 1.5)],
        thickness_ls=[0.10], device=cpu,
    )
    ts, tp, rs, rp = solver.simulate(
        theta=torch.tensor([0.5]),
        wvln=torch.tensor([0.55]),
    )
    assert abs(abs(ts).item() - abs(tp).item()) > 1e-3


def test_anisotropic_3tuple_checkpoint_roundtrip(tmp_path):
    cpu = torch.device("cpu")
    path = tmp_path / "ckpt.pt"
    solver = FilmSolver(
        mat_n_in="air", mat_n_out=1.52,
        mat_n_ls=[("SiO2", "TiO2", "SiO2"), (2.4, 2.4, 1.5)],
        thickness_ls=[0.1, 0.06], device=cpu,
    )
    solver.save_ckpt(path)
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["layer_specs"][0] == ("sio2", "tio2", "sio2")
    assert all(isinstance(v, complex) for v in ckpt["layer_specs"][1])
