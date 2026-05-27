"""Regression tests: the scalar-only API must produce identical outputs after
the Material refactor."""
import torch

from difftmm import IsotropicFilmSolver, FilmSolver


def test_isotropic_scalar_api_outputs_finite():
    solver = IsotropicFilmSolver(
        mat_in=1.0,
        mat_out=1.52,
        mat_ls=[2.10, 1.46, 2.10],
        thickness_ls=[0.080, 0.120, 0.080],
        device=torch.device("cpu"),
    )
    theta = torch.linspace(0, 1.2, 10)
    wvln = [0.45, 0.55, 0.65]
    ts, tp, rs, rp = solver.simulate(theta=theta, wvln=wvln)
    assert ts.shape == (1, 3, 10)
    assert torch.isfinite(ts).all() and torch.isfinite(rp).all()


def test_isotropic_scalar_api_energy_conservation_at_normal_incidence():
    """|R|^2 + |T|^2 = 1 for lossless stack with n_in == n_out at normal incidence."""
    solver = IsotropicFilmSolver(
        mat_in=1.0,
        mat_out=1.0,
        mat_ls=[2.10, 1.46, 2.10],
        thickness_ls=[0.080, 0.120, 0.080],
        device=torch.device("cpu"),
    )
    ts, tp, rs, rp = solver.simulate(
        theta=torch.tensor([0.0]), wvln=torch.tensor([0.55])
    )
    R = (rs.abs() ** 2).item()
    T = (ts.abs() ** 2).item()
    assert abs(R + T - 1.0) < 1e-3


def test_top_level_imports():
    from difftmm import (
        Material,
        list_materials,
        IsotropicFilmSolver,
        FilmSolver,
    )
    assert callable(Material)
    assert "air" in list_materials()
