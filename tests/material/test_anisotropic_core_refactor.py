"""Regression tests for create_jones_matrix_AOIAz per-wavelength tensor support.

T14: Refactor create_jones_matrix_AOIAz to accept per-wavelength tensors.
"""
import torch

from difftmm.film_solver_anisotropic import create_jones_matrix_AOIAz


def test_per_wvln_dispersive_n_2d():
    """n_2d shape (batch, n_wvln, n_layer, 3) — dispersive per layer."""
    batch, n_wv, n_layer = 1, 3, 2
    a_2d = torch.zeros((batch, n_layer, 3), dtype=torch.complex64)
    n_static = torch.tensor(
        [[[1.46, 1.46, 1.46], [2.4, 2.4, 2.4]]], dtype=torch.complex64
    )  # (batch, n_layer, 3)
    d = torch.tensor([[0.10, 0.06]], dtype=torch.complex64)
    wv = torch.tensor([[0.45, 0.55, 0.65]], dtype=torch.float32)
    th_x = torch.tensor([[0.1]], dtype=torch.float32)
    th_y = torch.tensor([[0.0]], dtype=torch.float32)

    Jt1, Jr1 = create_jones_matrix_AOIAz(a_2d, n_static, d, wv, 1.0, 1.52, th_x, th_y)

    n_dispersive = n_static.unsqueeze(1).expand(-1, n_wv, -1, -1)
    Jt2, Jr2 = create_jones_matrix_AOIAz(a_2d, n_dispersive, d, wv, 1.0, 1.52, th_x, th_y)

    torch.testing.assert_close(Jt1, Jt2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(Jr1, Jr2, rtol=1e-5, atol=1e-6)
