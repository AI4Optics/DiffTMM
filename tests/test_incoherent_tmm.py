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
