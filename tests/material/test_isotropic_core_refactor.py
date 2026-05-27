"""Regression tests for create_jones_matrix_isotropic per-wavelength tensor support.

T12: Refactor create_jones_matrix_isotropic to accept per-wavelength tensors.
"""
import pytest
import torch

from difftmm.film_solver_isotropic import create_jones_matrix_isotropic


@pytest.fixture
def stack():
    """A small isotropic stack: air | TiO2 | SiO2 | glass."""
    n_layers = torch.tensor([[2.4 + 0j, 1.46 + 0j]], dtype=torch.complex64)
    d = torch.tensor([[0.06, 0.10]], dtype=torch.complex64)
    wv = torch.tensor([[0.45, 0.55, 0.65]], dtype=torch.float32)
    theta = torch.tensor([[0.0, 0.2, 0.4]], dtype=torch.float32)
    return n_layers, d, wv, theta


def test_per_wvln_scalar_n_in_n_out_matches_old_behavior(stack):
    """Passing n_in / n_out as a (batch, n_wvln) tensor of constants should
    yield identical results to passing them as Python floats (back-compat)."""
    n_layers, d, wv, theta = stack
    ts1, tp1, rs1, rp1 = create_jones_matrix_isotropic(n_layers, d, wv, 1.0, 1.52, theta)

    # Pass n_in / n_out as broadcasted complex tensors of shape (batch, n_wvln)
    n_in_t = torch.full(wv.shape, 1.0 + 0j, dtype=torch.complex64)
    n_out_t = torch.full(wv.shape, 1.52 + 0j, dtype=torch.complex64)
    ts2, tp2, rs2, rp2 = create_jones_matrix_isotropic(n_layers, d, wv, n_in_t, n_out_t, theta)

    torch.testing.assert_close(ts1, ts2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rp1, rp2, rtol=1e-5, atol=1e-6)


def test_per_wvln_dispersive_n_layers_supported(stack):
    """n_layers shape (batch, n_wvln, n_layer) — dispersive per layer."""
    n_layers_static, d, wv, theta = stack
    # Build dispersive: same value for all wvln (should match static result)
    n_layers_dispersive = n_layers_static.unsqueeze(1).expand(-1, wv.shape[1], -1)

    ts1, tp1, rs1, rp1 = create_jones_matrix_isotropic(n_layers_static, d, wv, 1.0, 1.52, theta)
    ts2, tp2, rs2, rp2 = create_jones_matrix_isotropic(n_layers_dispersive, d, wv, 1.0, 1.52, theta)

    torch.testing.assert_close(ts1, ts2, rtol=1e-5, atol=1e-6)
