"""Tests for incoherent TMM support in DiffTMM."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# Make sibling `tmm_numpy` importable when running tests from repo root.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tmm_numpy.tmm_core import R_from_r, T_from_t, coh_tmm  # noqa: E402

from difftmm.film_solver_isotropic import (  # noqa: E402
    create_jones_matrix_isotropic,
    coh_stack_power_RT_isotropic,
)


DEVICE = torch.device("cpu")
RTOL = 1e-5
ATOL = 1e-6


def _wrap_inputs(n_layers, d_layers, wv, theta):
    """Helper: convert plain lists to the batched tensor shapes the API expects."""
    n_t = torch.tensor(n_layers, dtype=torch.complex64, device=DEVICE).unsqueeze(0)
    d_t = torch.tensor(d_layers, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    wv_t = torch.tensor([wv], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    th_t = torch.tensor([theta], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return n_t, d_t, wv_t, th_t


def test_coh_stack_power_RT_matches_tmm_numpy():
    """coh_stack_power_RT_isotropic must match tmm_numpy power R/T for a coherent stack."""
    # 3-layer coherent stack: air | 100nm Ta2O5 | 120nm SiO2 | 80nm Ta2O5 | glass
    # All units in um.
    n_in, n_out = 1.0, 1.52
    n_layers = [2.10, 1.46, 2.10]
    d_layers = [0.100, 0.120, 0.080]
    wv = 0.55
    theta = 0.3  # ~17.2 degrees

    n_t, d_t, wv_t, th_t = _wrap_inputs(n_layers, d_layers, wv, theta)

    Rs, Rp, Ts, Tp = coh_stack_power_RT_isotropic(
        n_t, d_t, wv_t, n_in, n_out, th_t
    )

    # Reference: tmm_numpy.coh_tmm. d_list must start/end with inf.
    ref_n = [n_in] + n_layers + [n_out]
    ref_d = [np.inf] + d_layers + [np.inf]
    ref_s = coh_tmm("s", ref_n, ref_d, theta, wv)
    ref_p = coh_tmm("p", ref_n, ref_d, theta, wv)

    assert np.allclose(Rs.item(), ref_s["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Rp.item(), ref_p["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Ts.item(), ref_s["T"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Tp.item(), ref_p["T"], rtol=RTOL, atol=ATOL)


def test_coh_stack_power_RT_asymmetric_and_energy_conservation():
    """Asymmetric stack must still match tmm_numpy, and R + T = 1 for lossless layers."""
    n_in, n_out = 1.0, 1.52
    # Asymmetric: distinct indices in different positions.
    n_layers = [2.50, 1.46, 1.80]
    d_layers = [0.120, 0.080, 0.150]
    wv = 0.633
    theta = 0.9  # ~51.5 deg, comfortably away from normal incidence

    n_t, d_t, wv_t, th_t = _wrap_inputs(n_layers, d_layers, wv, theta)
    Rs, Rp, Ts, Tp = coh_stack_power_RT_isotropic(
        n_t, d_t, wv_t, n_in, n_out, th_t
    )

    ref_n = [n_in] + n_layers + [n_out]
    ref_d = [np.inf] + d_layers + [np.inf]
    ref_s = coh_tmm("s", ref_n, ref_d, theta, wv)
    ref_p = coh_tmm("p", ref_n, ref_d, theta, wv)

    assert np.allclose(Rs.item(), ref_s["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Rp.item(), ref_p["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Ts.item(), ref_s["T"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Tp.item(), ref_p["T"], rtol=RTOL, atol=ATOL)
    # Lossless: energy conservation must hold to float32 precision.
    assert np.allclose(Rs.item() + Ts.item(), 1.0, atol=1e-5)
    assert np.allclose(Rp.item() + Tp.item(), 1.0, atol=1e-5)


from difftmm.film_solver_isotropic import group_layers_by_coherence  # noqa: E402


def test_group_layers_all_incoherent():
    """All-incoherent stack: no coherent groups, each layer is its own incoherent unit."""
    groups = group_layers_by_coherence(["i", "i", "i"])
    assert groups["num_inc_layers"] == 3
    assert groups["num_stacks"] == 0
    assert groups["stack_alllayer_indices"] == []
    assert groups["inc_alllayer_indices"] == [0, 1, 2]
    assert groups["stack_after_inc"] == [None, None, None]
    # inc_after_stack is one entry per stack
    assert groups["inc_after_stack"] == []


def test_group_layers_single_coherent_stack_inside():
    """i | c c | i — one stack of two coherent layers."""
    groups = group_layers_by_coherence(["i", "c", "c", "i"])
    assert groups["num_inc_layers"] == 2
    assert groups["num_stacks"] == 1
    # The stack spans alllayer indices 1, 2 plus its incoherent bookends (0 and 3).
    assert groups["stack_alllayer_indices"] == [[0, 1, 2, 3]]
    assert groups["inc_alllayer_indices"] == [0, 3]
    # Incoherent layer 0 is followed by stack 0; incoherent layer 1 is followed by no stack.
    assert groups["stack_after_inc"] == [0, None]
    # Stack 0 comes after incoherent layer 0.
    assert groups["inc_after_stack"] == [0]


def test_group_layers_multiple_stacks():
    """i | c | i | c c | i — two stacks separated by an incoherent layer."""
    groups = group_layers_by_coherence(["i", "c", "i", "c", "c", "i"])
    assert groups["num_inc_layers"] == 3
    assert groups["num_stacks"] == 2
    assert groups["stack_alllayer_indices"] == [[0, 1, 2], [2, 3, 4, 5]]
    assert groups["inc_alllayer_indices"] == [0, 2, 5]
    assert groups["stack_after_inc"] == [0, 1, None]
    assert groups["inc_after_stack"] == [0, 1]


def test_group_layers_endpoints_must_be_incoherent():
    """First and last layers are semi-infinite, must be 'i'."""
    with pytest.raises(ValueError, match="must start and end with"):
        group_layers_by_coherence(["c", "c", "i"])
    with pytest.raises(ValueError, match="must start and end with"):
        group_layers_by_coherence(["i", "c", "c"])


def test_group_layers_rejects_unknown_codes():
    with pytest.raises(ValueError, match="entries must be"):
        group_layers_by_coherence(["i", "x", "i"])


from difftmm.film_solver_isotropic import interface_power_RT  # noqa: E402
from tmm_numpy.tmm_core import interface_R as ref_interface_R  # noqa: E402
from tmm_numpy.tmm_core import interface_T as ref_interface_T  # noqa: E402
from tmm_numpy.tmm_core import snell as ref_snell  # noqa: E402


def test_interface_power_RT_real_indices():
    """Single interface R/T must match Fresnel and obey R + T = 1 for real indices."""
    n_i, n_f = 1.0, 1.52
    theta_i = 0.4
    theta_f = ref_snell(n_i, n_f, theta_i)  # complex but imag ~ 0

    n_i_t = torch.tensor(n_i, dtype=torch.complex64)
    n_f_t = torch.tensor(n_f, dtype=torch.complex64)
    cos_i = torch.tensor(np.cos(theta_i), dtype=torch.complex64)
    cos_f = torch.tensor(np.cos(theta_f), dtype=torch.complex64)

    Rs, Rp, Ts, Tp = interface_power_RT(n_i_t, n_f_t, cos_i, cos_f)

    assert np.allclose(Rs.item(), ref_interface_R("s", n_i, n_f, theta_i, theta_f), atol=ATOL)
    assert np.allclose(Rp.item(), ref_interface_R("p", n_i, n_f, theta_i, theta_f), atol=ATOL)
    assert np.allclose(Ts.item(), ref_interface_T("s", n_i, n_f, theta_i, theta_f), atol=ATOL)
    assert np.allclose(Tp.item(), ref_interface_T("p", n_i, n_f, theta_i, theta_f), atol=ATOL)
    # Energy conservation for real n, real theta.
    assert np.allclose(Rs.item() + Ts.item(), 1.0, atol=ATOL)
    assert np.allclose(Rp.item() + Tp.item(), 1.0, atol=ATOL)


from difftmm.film_solver_isotropic import create_intensity_RT_isotropic  # noqa: E402
from tmm_numpy.tmm_core import inc_tmm as ref_inc_tmm  # noqa: E402

INF = float("inf")


def test_inc_three_real_layers_matches_reference():
    """3-incoherent-layer real-index stack: matches tmm_numpy.inc_tmm closed form."""
    n_list = [1.0, 2.0, 3.0]
    d_list_inc = [INF, 0.567, INF]  # in um (567 nm)
    theta = float(np.pi / 3)
    wv = 0.400  # 400 nm in um

    # Reference values from tmm_numpy
    ref_d_list = [INF, 567.0, INF]  # nm
    ref_c_list = ["i", "i", "i"]
    ref_s = ref_inc_tmm("s", n_list, ref_d_list, ref_c_list, theta, 400.0)
    ref_p = ref_inc_tmm("p", n_list, ref_d_list, ref_c_list, theta, 400.0)

    n_t = torch.tensor([n_list[1:-1]], dtype=torch.complex64)
    d_t = torch.tensor([d_list_inc[1:-1]], dtype=torch.float32)
    wv_t = torch.tensor([[wv]], dtype=torch.float32)
    th_t = torch.tensor([[theta]], dtype=torch.float32)

    Rs, Rp, Ts, Tp = create_intensity_RT_isotropic(
        n_t, d_t, wv_t, n_in=n_list[0], n_out=n_list[-1], theta_1d=th_t,
        c_list=["i"],  # interior only; the full sequence is ['i', 'i', 'i']
    )

    assert np.allclose(Rs.item(), ref_s["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Rp.item(), ref_p["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Ts.item(), ref_s["T"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Tp.item(), ref_p["T"], rtol=RTOL, atol=ATOL)
    # Energy conservation for real indices.
    assert np.allclose(Rs.item() + Ts.item(), 1.0, atol=1e-5)
    assert np.allclose(Rp.item() + Tp.item(), 1.0, atol=1e-5)
