# Incoherent TMM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native incoherent / partly-incoherent TMM support to DiffTMM's `IsotropicFilmSolver` so users can mark individual layers as incoherent (e.g. thick substrates) and obtain ripple-free intensity-based R/T, while keeping autograd, GPU batching, and the existing coherent path intact.

**Architecture:** Port the algorithm from `tmm_numpy/tmm_core.py::inc_tmm` (intensity transfer matrices between incoherent layers, with each coherent "stack" reduced to its (R, T) via the existing 2x2 coherent solver). All new code is appended to the existing `difftmm/film_solver_isotropic.py` module — keeping coherent and incoherent isotropic TMM together since they share the same underlying 2x2 math. Adds a new `c_list` parameter (per-layer 'c'/'i') and returns power-domain `(Rs, Rp, Ts, Tp)` rather than complex amplitudes (since incoherent calculation discards phase). Scope is restricted to the 2x2 isotropic solver — extending the 4x4 anisotropic solver to incoherent is left as future work (called out at the end of the plan), as the issue's primary use case (thin films on thick isotropic substrates) is fully covered by the 2x2 path.

**Tech Stack:** PyTorch (autograd, complex64), Python 3.9+, pytest for tests, NumPy + the bundled `tmm_numpy` reference library for validation.

---

## File Structure

**New files:**
- `tests/__init__.py` — empty (marker file).
- `tests/test_incoherent_tmm.py` — pytest suite validating against `tmm_numpy.inc_tmm` and analytic formulas.
- `benchmarks/4_compare_incoherent.py` — visual comparison plot vs `tmm_numpy.inc_tmm` (mirrors style of `1_compare_angle_response_isotropic.py`).

**Modified files:**
- `difftmm/film_solver_isotropic.py` — append the incoherent TMM helpers, the public `create_intensity_RT_isotropic` function, and the `IncoherentIsotropicFilmSolver` class to the existing module. Estimated ~330 lines added (final file ~870 lines).
- `difftmm/__init__.py` — re-export `IncoherentIsotropicFilmSolver` and `create_intensity_RT_isotropic`.
- `pyproject.toml` — add `tests/` to package excludes; nothing else.
- `README.md` — add a short "Incoherent layers" subsection with an example.

**Why no new module:** The coherent and incoherent 2x2 paths share the same Snell/Fresnel math and the same `_compute_isotropic_tmm` core. Keeping them in one file makes it easy to see how the incoherent path reuses the coherent stack solver, and matches the existing convention in this repo of multi-hundred-line solver files (`film_solver_anisotropic.py` is already ~720 lines).

---

## Algorithm Summary (reference: `tmm_numpy/tmm_core.py:751-949`)

Given a per-layer coherence list `c_list ∈ {'c','i'}^N` (semi-infinite first and last layers must be `'i'`):

1. **Group layers** into consecutive coherent "stacks", each bounded by incoherent layers. Track index mappings.
2. **Solve each coherent stack** with the existing 2x2 solver to get power-domain `(R, T)` looking forward and `(R', T')` looking backward through that stack.
3. **For each purely-incoherent interface** between two adjacent incoherent layers (no stack between them), compute Fresnel `R`, `T` from interface amplitudes.
4. **Compute single-pass absorption** `P_i = exp(-4π · d_i · Im(n_i · cos θ_i) / λ)` for each finite incoherent layer.
5. **Assemble intensity transfer matrices** `L_i` between adjacent incoherent layers (formula in `tmm_core.py:887-892`).
6. **Multiply** `Ltilde = L_initial · L_1 · L_2 · … · L_{N_inc-2}` and read off:
   - `T_total = 1 / Ltilde[0,0]`
   - `R_total = Ltilde[1,0] / Ltilde[0,0]`

All `R, T` returned are real, in `[0, 1]`. Autograd flows through because each step is differentiable (`exp`, `cos`, `abs`, matrix multiply on real or complex tensors).

---

## Task 1: Add coherent-stack (R, T) helper to the isotropic solver

**Files:**
- Modify: `difftmm/film_solver_isotropic.py` (append a new public helper near the existing `_compute_isotropic_tmm`)
- Test: `tests/test_incoherent_tmm.py` (created in Task 4, but we add a stub test now to drive this helper)

**Why:** `_compute_isotropic_tmm` returns complex amplitudes `(ts, tp, rs, rp)`. The incoherent algorithm needs *power* `(Rs, Rp, Ts, Tp)` for each coherent stack, going both forward and backward. We need a thin wrapper that converts amplitudes → power using the same formulas as `T_from_t` / `R_from_r` in `tmm_numpy/tmm_core.py:156-182`.

- [ ] **Step 1: Create the `tests/` directory and write the failing test**

Create `tests/__init__.py` as an empty file.

Then create `tests/test_incoherent_tmm.py` with this content (more tests added in later tasks):

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_incoherent_tmm.py::test_coh_stack_power_RT_matches_tmm_numpy -v`
Expected: FAIL with `ImportError: cannot import name 'coh_stack_power_RT_isotropic'`.

- [ ] **Step 3: Implement `coh_stack_power_RT_isotropic`**

Append this function to `difftmm/film_solver_isotropic.py`, just below `create_jones_matrix_isotropic` and above the `IsotropicFilmSolver` class:

```python
def coh_stack_power_RT_isotropic(
    n_layers_1d,
    d_1d,
    wv_1d,
    n_in,
    n_out,
    theta_1d,
):
    """Power-domain (Rs, Rp, Ts, Tp) for a coherent isotropic stack.

    Thin wrapper around ``create_jones_matrix_isotropic`` that converts
    complex amplitudes to real power coefficients. Used as a building block
    for the incoherent TMM solver, where each coherent stack contributes its
    forward/backward (R, T) to the intensity transfer matrix.

    Args:
        n_layers_1d: refractive index of each interior layer, shape (batch, n_layer). Complex.
        d_1d: thickness of each interior layer in um, shape (batch, n_layer). Real.
        wv_1d: wavelengths in um, shape (batch, n_wls). Real.
        n_in: scalar incident refractive index (top medium).
        n_out: scalar exit refractive index (bottom medium).
        theta_1d: incident angles in radians, shape (batch, n_angles). Real, in [0, pi/2].

    Returns:
        Rs, Rp, Ts, Tp: real tensors, each shape (batch, n_wls, n_angles), in [0, 1].
    """
    ts, tp, rs, rp = create_jones_matrix_isotropic(
        n_layers_1d, d_1d, wv_1d, n_in, n_out, theta_1d
    )

    # Reflectance is |r|^2 for both polarizations.
    Rs = (rs.real ** 2 + rs.imag ** 2)
    Rp = (rp.real ** 2 + rp.imag ** 2)

    # Transmittance uses the standard intensity correction:
    # s-pol:  T = |t|^2 * Re(n_out * cos th_out) / Re(n_in * cos th_in)
    # p-pol:  T = |t|^2 * Re(n_out * conj(cos th_out)) / Re(n_in * conj(cos th_in))
    # For real n_in, n_out and theta_in in [0, pi/2], the conj() ops are no-ops.
    device = n_layers_1d.device
    dtype = torch.complex64

    n_in_t = torch.tensor(n_in, dtype=dtype, device=device)
    n_out_t = torch.tensor(n_out, dtype=dtype, device=device)
    cos_th_in = torch.cos(theta_1d.to(dtype)).unsqueeze(1)  # (batch, 1, angles)
    sin_th_in = torch.sin(theta_1d.to(dtype)).unsqueeze(1)
    sin_th_out = n_in_t * sin_th_in / n_out_t
    cos_th_out = torch.sqrt(1 - sin_th_out ** 2)

    # s-pol denominator/numerator (real parts).
    num_s = (n_out_t * cos_th_out).real
    den_s = (n_in_t * cos_th_in).real
    Ts = (ts.real ** 2 + ts.imag ** 2) * (num_s / den_s)

    # p-pol uses conj(cos theta).
    num_p = (n_out_t * torch.conj(cos_th_out)).real
    den_p = (n_in_t * torch.conj(cos_th_in)).real
    Tp = (tp.real ** 2 + tp.imag ** 2) * (num_p / den_p)

    return Rs, Rp, Ts, Tp
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_incoherent_tmm.py::test_coh_stack_power_RT_matches_tmm_numpy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_incoherent_tmm.py difftmm/film_solver_isotropic.py
git commit -m "feat(solver): add coh_stack_power_RT_isotropic helper for incoherent TMM

Returns real power (Rs, Rp, Ts, Tp) for a coherent stack — needed as
the building block for the upcoming inc_tmm port. Validated against
tmm_numpy.coh_tmm."
```

---

## Task 2: Add the layer-grouping helper

**Files:**
- Modify: `difftmm/film_solver_isotropic.py` (append)
- Test: `tests/test_incoherent_tmm.py`

**Why:** Before computing anything, we need to decompose the layer list into coherent stacks and incoherent layers. This mirrors `inc_group_layers` in `tmm_numpy/tmm_core.py:641`. We do it in pure Python (no tensors) because the grouping depends only on `c_list`, which is fixed per simulation — there's no need to differentiate through it or run it on the GPU.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_incoherent_tmm.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_incoherent_tmm.py -k group_layers -v`
Expected: FAIL with `ImportError: cannot import name 'group_layers_by_coherence'`.

- [ ] **Step 3: Append `group_layers_by_coherence` to `difftmm/film_solver_isotropic.py`**

Add `from typing import Dict, List, Optional, Sequence` to the existing import block at the top of the file if it isn't already imported.

Then append this section at the end of the file (after the existing `IsotropicFilmSolver` class is fine — keep functions and classes interleaved as the existing file already does):

```python
# ===========================================
# Incoherent / partly-incoherent TMM
# ===========================================
# Differentiable implementation of partly-incoherent TMM that lets users mark
# individual layers as coherent ('c') or incoherent ('i'). Reuses the
# coherent 2x2 solver above for each coherent stack, then propagates
# intensities between incoherent layers via real 2x2 transfer matrices.
#
# Algorithm reference: tmm_numpy/tmm_core.py::inc_tmm (Steven Byrnes, MIT).
# Physics reference:   https://arxiv.org/abs/1603.02720 (S. Byrnes, 2016).


# =========================
# Layer grouping
# =========================
def group_layers_by_coherence(c_list: Sequence[str]) -> Dict[str, object]:
    """Group a layer-coherence list into coherent stacks and incoherent layers.

    A "stack" is a maximal run of consecutive coherent layers, bracketed on
    each side by an incoherent layer (which is required to be present because
    the first and last layers are semi-infinite and must be 'i').

    Args:
        c_list: Per-layer coherence flags. Each entry is 'i' (incoherent) or
            'c' (coherent). First and last entries must be 'i' because those
            layers are semi-infinite.

    Returns:
        Dict with keys:
            - num_inc_layers (int): number of incoherent layers.
            - num_stacks (int): number of coherent stacks.
            - inc_alllayer_indices (List[int]): for each incoherent layer i,
              its index in the original layer list.
            - stack_alllayer_indices (List[List[int]]): for each stack s,
              the original indices of layers in [bracketing_inc, coh_layers..., bracketing_inc].
            - stack_after_inc (List[Optional[int]]): for each incoherent layer i,
              the stack-index of the stack immediately after it, or None if the
              next layer is also incoherent (or there is no next layer).
            - inc_after_stack (List[int]): for each stack s, the incoherent-layer
              index that immediately precedes the stack.

    Raises:
        ValueError: if the first or last entry is not 'i', or any entry is
            neither 'i' nor 'c'.
    """
    if len(c_list) < 2 or c_list[0] != "i" or c_list[-1] != "i":
        raise ValueError("c_list must start and end with 'i' (semi-infinite layers).")
    for code in c_list:
        if code not in ("i", "c"):
            raise ValueError("c_list entries must be 'i' or 'c'.")

    inc_alllayer_indices: List[int] = []
    stack_alllayer_indices: List[List[int]] = []
    stack_after_inc: List[Optional[int]] = []
    inc_after_stack: List[int] = []

    inc_index = -1  # incremented when we visit an 'i' layer
    in_stack = False
    current_stack: List[int] = []

    for layer_idx, code in enumerate(c_list):
        if code == "i":
            inc_index += 1
            inc_alllayer_indices.append(layer_idx)
            if in_stack:
                # Close out the stack with this incoherent layer as its right bracket.
                current_stack.append(layer_idx)
                stack_alllayer_indices.append(current_stack)
                # Whoever opened this stack was the previous incoherent layer,
                # which is at incoherent index (inc_index - 1).
                inc_after_stack.append(inc_index - 1)
                current_stack = []
                in_stack = False
                # This 'i' has no stack following it *yet*; will be patched if the next 'c' opens one.
                stack_after_inc.append(None)
            else:
                stack_after_inc.append(None)
        else:  # 'c'
            if not in_stack:
                # Open a stack: left bracket is the previous incoherent layer.
                in_stack = True
                current_stack = [inc_alllayer_indices[-1], layer_idx]
                # The most recent incoherent layer is followed by a stack whose
                # index will be len(stack_alllayer_indices) once the stack closes.
                stack_after_inc[-1] = len(stack_alllayer_indices)
            else:
                current_stack.append(layer_idx)

    return {
        "num_inc_layers": len(inc_alllayer_indices),
        "num_stacks": len(stack_alllayer_indices),
        "inc_alllayer_indices": inc_alllayer_indices,
        "stack_alllayer_indices": stack_alllayer_indices,
        "stack_after_inc": stack_after_inc,
        "inc_after_stack": inc_after_stack,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_incoherent_tmm.py -k group_layers -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/film_solver_isotropic.py tests/test_incoherent_tmm.py
git commit -m "feat(solver): add group_layers_by_coherence helper

Pure-Python decomposition of c_list into coherent stacks and
incoherent layers. Mirrors tmm_numpy.inc_group_layers but with a
simplified output schema tailored to the batched-PyTorch path."
```

---

## Task 3: Add the interface Fresnel R/T helper for incoherent-incoherent boundaries

**Files:**
- Modify: `difftmm/film_solver_isotropic.py` (append)
- Test: `tests/test_incoherent_tmm.py`

**Why:** When two incoherent layers are adjacent (no coherent stack between them), we need plain Fresnel interface reflectance/transmittance — not a stack solve. This mirrors `interface_R` / `interface_T` in `tmm_numpy/tmm_core.py:205-217` plus `interface_r` / `interface_t` (lines 118-154).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_incoherent_tmm.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_incoherent_tmm.py::test_interface_power_RT_real_indices -v`
Expected: FAIL with `ImportError: cannot import name 'interface_power_RT'`.

- [ ] **Step 3: Implement `interface_power_RT`**

Append to `difftmm/film_solver_isotropic.py` (immediately after `group_layers_by_coherence`):

```python
# =========================
# Single-interface Fresnel power R, T
# =========================
def interface_power_RT(n_i, n_f, cos_i, cos_f):
    """Fresnel power reflectance/transmittance at a single interface.

    All inputs are complex tensors broadcastable to a common shape.
    Returns (Rs, Rp, Ts, Tp) as real tensors of that shape.

    Math:
        r_s = (n_i cos_i - n_f cos_f) / (n_i cos_i + n_f cos_f)
        r_p = (n_f cos_i - n_i cos_f) / (n_f cos_i + n_i cos_f)
        t_s = 2 n_i cos_i / (n_i cos_i + n_f cos_f)
        t_p = 2 n_i cos_i / (n_f cos_i + n_i cos_f)
        R = |r|^2
        T_s = |t_s|^2 * Re(n_f cos_f) / Re(n_i cos_i)
        T_p = |t_p|^2 * Re(n_f conj(cos_f)) / Re(n_i conj(cos_i))
    """
    n_i_cos_i = n_i * cos_i
    n_f_cos_f = n_f * cos_f
    n_f_cos_i = n_f * cos_i
    n_i_cos_f = n_i * cos_f

    r_s = (n_i_cos_i - n_f_cos_f) / (n_i_cos_i + n_f_cos_f)
    r_p = (n_f_cos_i - n_i_cos_f) / (n_f_cos_i + n_i_cos_f)
    t_s = 2 * n_i_cos_i / (n_i_cos_i + n_f_cos_f)
    t_p = 2 * n_i_cos_i / (n_f_cos_i + n_i_cos_f)

    Rs = r_s.real ** 2 + r_s.imag ** 2
    Rp = r_p.real ** 2 + r_p.imag ** 2

    ts_sq = t_s.real ** 2 + t_s.imag ** 2
    tp_sq = t_p.real ** 2 + t_p.imag ** 2

    Ts = ts_sq * (n_f_cos_f.real / n_i_cos_i.real)
    Tp = tp_sq * (
        (n_f * torch.conj(cos_f)).real / (n_i * torch.conj(cos_i)).real
    )

    return Rs, Rp, Ts, Tp
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_incoherent_tmm.py::test_interface_power_RT_real_indices -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/film_solver_isotropic.py tests/test_incoherent_tmm.py
git commit -m "feat(solver): add interface_power_RT Fresnel helper

Differentiable power reflectance/transmittance for a single interface
between two incoherent layers. Validated against tmm_numpy.interface_R/T."
```

---

## Task 4: Implement the top-level `create_intensity_RT_isotropic` function — 3-layer all-incoherent base case

**Files:**
- Modify: `difftmm/film_solver_isotropic.py` (append)
- Test: `tests/test_incoherent_tmm.py`

**Why:** Build up the main function incrementally. The simplest case — three incoherent semi-infinite-ish layers (`['i','i','i']`) — exercises the intensity transfer matrix path with zero coherent stacks. Subsequent tasks add coherent stacks and the multi-incoherent-layer pathway.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_incoherent_tmm.py`:

```python
from difftmm.film_solver_isotropic import create_intensity_RT_isotropic  # noqa: E402
from tmm_numpy.tmm_core import inc_tmm as ref_inc_tmm  # noqa: E402
from tmm_numpy.tmm_core import snell  # noqa: E402

INF = float("inf")


def test_inc_three_real_layers_matches_reference():
    """3-incoherent-layer real-index stack: matches tmm_numpy.inc_tmm closed form."""
    n_list = [1.0, 2.0, 3.0]
    d_list_inc = [INF, 0.567, INF]  # in um (567 nm)
    c_list = ["i", "i", "i"]
    theta = float(np.pi / 3)
    wv = 0.400  # 400 nm in um

    # Reference values from tmm_numpy
    ref_d_list = [INF, 567.0, INF]  # nm
    ref_s = ref_inc_tmm("s", n_list, ref_d_list, c_list, theta, 400.0)
    ref_p = ref_inc_tmm("p", n_list, ref_d_list, c_list, theta, 400.0)

    n_t = torch.tensor([n_list[1:-1]], dtype=torch.complex64)
    d_t = torch.tensor([d_list_inc[1:-1]], dtype=torch.float32)
    wv_t = torch.tensor([[wv]], dtype=torch.float32)
    th_t = torch.tensor([[theta]], dtype=torch.float32)

    Rs, Rp, Ts, Tp = create_intensity_RT_isotropic(
        n_t, d_t, wv_t, n_in=n_list[0], n_out=n_list[-1], theta_1d=th_t, c_list=c_list,
    )

    assert np.allclose(Rs.item(), ref_s["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Rp.item(), ref_p["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Ts.item(), ref_s["T"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Tp.item(), ref_p["T"], rtol=RTOL, atol=ATOL)
    # Energy conservation for real indices.
    assert np.allclose(Rs.item() + Ts.item(), 1.0, atol=1e-5)
    assert np.allclose(Rp.item() + Tp.item(), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_incoherent_tmm.py::test_inc_three_real_layers_matches_reference -v`
Expected: FAIL with `ImportError: cannot import name 'create_intensity_RT_isotropic'`.

- [ ] **Step 3: Implement `create_intensity_RT_isotropic`**

Append to `difftmm/film_solver_isotropic.py` (after `interface_power_RT`):

```python
# =========================
# Main incoherent TMM
# =========================
def create_intensity_RT_isotropic(
    n_layers_1d,
    d_1d,
    wv_1d,
    n_in,
    n_out,
    theta_1d,
    c_list,
):
    """Partly-incoherent intensity TMM for isotropic multi-layer films.

    Each interior layer is marked coherent ('c') or incoherent ('i') via
    ``c_list``. The two semi-infinite media (n_in, n_out) are always
    incoherent, so ``c_list`` describes only the *interior* layers and the
    full coherence sequence is ``['i'] + c_list + ['i']``.

    The algorithm:
      1. Group consecutive coherent layers into stacks.
      2. For each stack, run the existing coherent 2x2 solver in both
         directions to get (R_fwd, T_fwd, R_bwd, T_bwd).
      3. Compute single-pass absorption P_i for each interior incoherent
         layer.
      4. Build real 2x2 intensity transfer matrices L_i and multiply.
      5. Read off total R and T.

    Args:
        n_layers_1d: refractive indices of interior layers, shape (batch, n_layer). Complex.
        d_1d: thicknesses of interior layers in um, shape (batch, n_layer). Real.
        wv_1d: wavelengths in um, shape (batch, n_wls). Real.
        n_in: scalar incident refractive index (top medium).
        n_out: scalar exit refractive index (bottom medium).
        theta_1d: incident angles in radians, shape (batch, n_angles). Real, in [0, pi/2].
        c_list: list of 'c'/'i' codes, length n_layer (interior layers only).

    Returns:
        Rs, Rp, Ts, Tp: real tensors, each shape (batch, n_wls, n_angles), in [0, 1].
    """
    if len(c_list) != n_layers_1d.shape[1]:
        raise ValueError(
            f"c_list length {len(c_list)} does not match interior-layer count "
            f"{n_layers_1d.shape[1]}."
        )

    device = n_layers_1d.device
    real_dtype = torch.float32
    complex_dtype = torch.complex64

    batchsize = n_layers_1d.shape[0]
    num_wv = wv_1d.shape[1]
    num_angles = theta_1d.shape[1]

    # Full coherence sequence including semi-infinite endpoints.
    full_c_list = ["i"] + list(c_list) + ["i"]
    groups = group_layers_by_coherence(full_c_list)

    num_inc = groups["num_inc_layers"]
    num_stacks = groups["num_stacks"]
    inc_alllayer_indices = groups["inc_alllayer_indices"]
    stack_alllayer_indices = groups["stack_alllayer_indices"]
    stack_after_inc = groups["stack_after_inc"]
    inc_after_stack = groups["inc_after_stack"]

    # Build a full per-layer index tensor for n and d, with the endpoints in place.
    # Shape: (batch, n_full) for n; (batch, n_full) for d. The two endpoint
    # thicknesses are infinite (unused for incoherent layers because P is only
    # computed for finite interior incoherent layers).
    n_in_col = torch.full(
        (batchsize, 1), n_in, dtype=complex_dtype, device=device
    )
    n_out_col = torch.full(
        (batchsize, 1), n_out, dtype=complex_dtype, device=device
    )
    n_full = torch.cat([n_in_col, n_layers_1d.to(complex_dtype), n_out_col], dim=1)

    d_in_col = torch.full((batchsize, 1), float("inf"), dtype=real_dtype, device=device)
    d_out_col = torch.full((batchsize, 1), float("inf"), dtype=real_dtype, device=device)
    d_full = torch.cat([d_in_col, d_1d.to(real_dtype), d_out_col], dim=1)

    # Snell's law to get cos(theta) in each layer.
    # theta_1d shape: (batch, angles) -> broadcast to (batch, n_full, angles)
    sin_th_in = torch.sin(theta_1d).to(complex_dtype)  # (batch, angles)
    # n_full shape: (batch, n_full); broadcast with sin_th_in: (batch, 1, angles)
    n_full_b = n_full.unsqueeze(-1)  # (batch, n_full, 1)
    n_in_b = n_in_col.unsqueeze(-1)  # (batch, 1, 1)
    sin_th_in_b = sin_th_in.unsqueeze(1)  # (batch, 1, angles)
    sin_th_layers = n_in_b * sin_th_in_b / n_full_b  # (batch, n_full, angles)
    cos_th_layers = torch.sqrt(1 - sin_th_layers ** 2)  # complex

    # Single-pass absorption P_i for each *interior* incoherent layer.
    # Formula: P = exp(-4 pi d Im(n cos th) / lambda)
    # Shape target: (batch, num_inc, n_wv, angles). The first and last
    # incoherent layers are the semi-infinite media (no P needed).
    wv_b = wv_1d.unsqueeze(-1).unsqueeze(1)  # (batch, 1, n_wv, 1)
    # Helper: gather the (n_full) index for each interior incoherent layer.
    interior_inc_alllayer = [
        inc_alllayer_indices[i] for i in range(1, num_inc - 1)
    ]
    if len(interior_inc_alllayer) > 0:
        idx = torch.tensor(interior_inc_alllayer, dtype=torch.long, device=device)
        # n_full[:, idx]: (batch, n_interior_inc); d_full[:, idx]: (batch, n_interior_inc)
        n_inc_interior = n_full.index_select(1, idx)  # (batch, n_int_inc)
        d_inc_interior = d_full.index_select(1, idx)  # (batch, n_int_inc)
        cos_inc_interior = cos_th_layers.index_select(1, idx)  # (batch, n_int_inc, angles)
        # Broadcast to (batch, n_int_inc, n_wv, angles).
        n_inc_b = n_inc_interior.unsqueeze(-1).unsqueeze(-1)  # (batch, n_int_inc, 1, 1)
        d_inc_b = d_inc_interior.unsqueeze(-1).unsqueeze(-1)
        cos_inc_b = cos_inc_interior.unsqueeze(2)  # (batch, n_int_inc, 1, angles)
        imag_part = (n_inc_b * cos_inc_b).imag  # (batch, n_int_inc, 1, angles)
        P_interior = torch.exp(-4 * torch.pi * d_inc_b.real * imag_part / wv_b.real)
        # Clamp to avoid divide-by-zero in the L matrix.
        P_interior = torch.clamp(P_interior, min=1e-30)
        # P_interior shape: (batch, n_int_inc, n_wv, angles).
    else:
        P_interior = None  # used only when num_inc > 2

    # Build per-incoherent-interface (R_fwd, T_fwd, R_bwd, T_bwd) for s and p.
    # The interface from incoherent layer i to i+1 is either:
    #   (a) a bare Fresnel interface (next layer is also incoherent), or
    #   (b) a coherent stack between them.
    # We compute these one polarization at a time to keep the code simple.
    Rs_total, Rp_total, Ts_total, Tp_total = _inc_total_RT_for_all_pols(
        n_full=n_full,
        d_full=d_full,
        cos_th_layers=cos_th_layers,
        wv_1d=wv_1d,
        theta_1d=theta_1d,
        n_in=n_in,
        n_out=n_out,
        groups=groups,
        P_interior=P_interior,
    )

    return Rs_total, Rp_total, Ts_total, Tp_total


def _inc_total_RT_for_all_pols(
    n_full,
    d_full,
    cos_th_layers,
    wv_1d,
    theta_1d,
    n_in,
    n_out,
    groups,
    P_interior,
):
    """Inner driver: assemble L matrices and return (Rs, Rp, Ts, Tp).

    Separated from the public entry point to keep the public signature
    focused. See create_intensity_RT_isotropic for argument meanings.
    """
    device = n_full.device
    complex_dtype = torch.complex64
    real_dtype = torch.float32

    batchsize = n_full.shape[0]
    num_wv = wv_1d.shape[1]
    num_angles = theta_1d.shape[1]

    num_inc = groups["num_inc_layers"]
    num_stacks = groups["num_stacks"]
    inc_alllayer_indices = groups["inc_alllayer_indices"]
    stack_alllayer_indices = groups["stack_alllayer_indices"]
    stack_after_inc = groups["stack_after_inc"]

    # Precompute power R/T for every coherent stack, forward and backward.
    # stack_RT[s] = dict with keys 'Rs_fwd', 'Rp_fwd', 'Ts_fwd', 'Tp_fwd',
    #                              'Rs_bwd', 'Rp_bwd', 'Ts_bwd', 'Tp_bwd'.
    # Each value has shape (batch, n_wv, n_angles).
    stack_RT = []
    for s_idx in range(num_stacks):
        layer_idxs = stack_alllayer_indices[s_idx]  # [left_inc, c..., right_inc]
        left_inc_alllayer = layer_idxs[0]
        right_inc_alllayer = layer_idxs[-1]
        coh_alllayer = layer_idxs[1:-1]  # the actual coherent layers

        # n_in / n_out for this sub-stack are the surrounding incoherent layers.
        n_left = n_full[:, left_inc_alllayer]  # (batch,) complex
        n_right = n_full[:, right_inc_alllayer]
        # All batch entries share the same n_left/n_right (since n_full came
        # from a single n_in/n_out and per-layer n_layers_1d), but each batch
        # entry might have different interior n. We pass scalars by taking [0].
        # (If a future use case needs per-batch n_in/n_out, this will need a
        # batched coh_stack_power_RT_isotropic.)
        n_left_scalar = complex(n_left[0].item())
        n_right_scalar = complex(n_right[0].item())

        idx_coh = torch.tensor(coh_alllayer, dtype=torch.long, device=device)
        n_coh = n_full.index_select(1, idx_coh)  # (batch, n_coh)
        d_coh = d_full.index_select(1, idx_coh).to(real_dtype)  # (batch, n_coh)

        Rs_fwd, Rp_fwd, Ts_fwd, Tp_fwd = coh_stack_power_RT_isotropic(
            n_coh, d_coh, wv_1d, n_left_scalar, n_right_scalar, theta_1d
        )

        # Backward: reverse layer order and swap media.
        n_coh_rev = torch.flip(n_coh, dims=[1])
        d_coh_rev = torch.flip(d_coh, dims=[1])
        # The angle on the right side is theta in the right medium (via Snell).
        # Compute it: sin_th_right = n_in * sin_th_in / n_right
        sin_th_in = torch.sin(theta_1d).to(complex_dtype)
        sin_th_right = (n_in * sin_th_in) / n_right_scalar
        # Real-valued angle for input (use .real; for evanescent cases this
        # falls back to arcsin of a clamped value, but the path that triggers
        # that is non-physical for real n media). Take the real part of the
        # complex angle as float.
        theta_right = torch.arcsin(torch.clamp(sin_th_right.real, -1.0, 1.0)).to(real_dtype)
        Rs_bwd, Rp_bwd, Ts_bwd, Tp_bwd = coh_stack_power_RT_isotropic(
            n_coh_rev, d_coh_rev, wv_1d, n_right_scalar, n_left_scalar, theta_right
        )
        stack_RT.append({
            "Rs_fwd": Rs_fwd, "Rp_fwd": Rp_fwd, "Ts_fwd": Ts_fwd, "Tp_fwd": Tp_fwd,
            "Rs_bwd": Rs_bwd, "Rp_bwd": Rp_bwd, "Ts_bwd": Ts_bwd, "Tp_bwd": Tp_bwd,
        })

    # For each incoherent-to-incoherent interface i -> i+1, compute
    # R_fwd, T_fwd, R_bwd, T_bwd (s and p separately).
    # We accumulate the L-matrix product on the fly:
    # Ltilde_init = [[1, -R_10], [R_01, T_10*T_01 - R_10*R_01]] / T_01
    # For i = 1..num_inc-2: L_i = diag(1/P_i, P_i) @ [[1, -R_{i+1,i}], [R_{i,i+1}, T_{i+1,i}*T_{i,i+1} - R_{i+1,i}*R_{i,i+1}]] / T_{i,i+1}
    # Ltilde *= L_i
    # T = 1 / Ltilde[0,0],  R = Ltilde[1,0] / Ltilde[0,0]
    def _interface_RT(inc_i):
        """Return (R_fwd, T_fwd, R_bwd, T_bwd) for the interface inc_i -> inc_i+1, for s and p.

        Each is a real tensor (batch, n_wv, n_angles).
        """
        nxt_stack = stack_after_inc[inc_i]
        if nxt_stack is None:
            # Direct incoherent-incoherent interface.
            a_idx = inc_alllayer_indices[inc_i]
            b_idx = inc_alllayer_indices[inc_i + 1]
            n_a = n_full[:, a_idx].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            n_b = n_full[:, b_idx].unsqueeze(-1).unsqueeze(-1)
            cos_a = cos_th_layers[:, a_idx].unsqueeze(1)  # (batch, 1, angles)
            cos_b = cos_th_layers[:, b_idx].unsqueeze(1)
            Rs_f, Rp_f, Ts_f, Tp_f = interface_power_RT(n_a, n_b, cos_a, cos_b)
            Rs_b, Rp_b, Ts_b, Tp_b = interface_power_RT(n_b, n_a, cos_b, cos_a)
            # Broadcast over wavelength axis.
            Rs_f = Rs_f.expand(-1, num_wv, -1)
            Rp_f = Rp_f.expand(-1, num_wv, -1)
            Ts_f = Ts_f.expand(-1, num_wv, -1)
            Tp_f = Tp_f.expand(-1, num_wv, -1)
            Rs_b = Rs_b.expand(-1, num_wv, -1)
            Rp_b = Rp_b.expand(-1, num_wv, -1)
            Ts_b = Ts_b.expand(-1, num_wv, -1)
            Tp_b = Tp_b.expand(-1, num_wv, -1)
            return (Rs_f, Rp_f, Ts_f, Tp_f, Rs_b, Rp_b, Ts_b, Tp_b)
        else:
            d = stack_RT[nxt_stack]
            return (
                d["Rs_fwd"], d["Rp_fwd"], d["Ts_fwd"], d["Tp_fwd"],
                d["Rs_bwd"], d["Rp_bwd"], d["Ts_bwd"], d["Tp_bwd"],
            )

    def _step_L(Rfwd, Tfwd, Rbwd, Tbwd):
        """L matrix factor (without the leading diag(1/P, P)). Real, shape (batch, n_wv, angles, 2, 2)."""
        eps = 1e-30
        Tfwd_safe = torch.clamp(Tfwd, min=eps)
        det = Tbwd * Tfwd - Rbwd * Rfwd
        # Stack into (..., 2, 2).
        row0 = torch.stack([torch.ones_like(Rfwd), -Rbwd], dim=-1)
        row1 = torch.stack([Rfwd, det], dim=-1)
        M = torch.stack([row0, row1], dim=-2) / Tfwd_safe.unsqueeze(-1).unsqueeze(-1)
        return M

    def _accumulate(pol):
        """Run the L-matrix accumulation for one polarization. Returns (R, T) tensors."""
        # Pull per-interface power R/T for this polarization (s or p).
        # Indices into the tuple returned by _interface_RT:
        #   s: Rs_f, Ts_f, Rs_b, Ts_b -> positions 0, 2, 4, 6
        #   p: Rp_f, Tp_f, Rp_b, Tp_b -> positions 1, 3, 5, 7
        if pol == "s":
            sel = (0, 2, 4, 6)
        else:
            sel = (1, 3, 5, 7)

        # Initial L from interface 0 -> 1.
        i0 = _interface_RT(0)
        Rfwd, Tfwd, Rbwd, Tbwd = i0[sel[0]], i0[sel[1]], i0[sel[2]], i0[sel[3]]
        Ltilde = _step_L(Rfwd, Tfwd, Rbwd, Tbwd)

        # Subsequent L_i for i = 1..num_inc-2.
        for i in range(1, num_inc - 1):
            # Interface i -> i+1.
            ii = _interface_RT(i)
            Rfwd, Tfwd, Rbwd, Tbwd = ii[sel[0]], ii[sel[1]], ii[sel[2]], ii[sel[3]]
            M = _step_L(Rfwd, Tfwd, Rbwd, Tbwd)
            # P factor for incoherent layer i (interior, so P_interior index is i - 1).
            P_i = P_interior[:, i - 1]  # (batch, n_wv, angles)
            P_safe = torch.clamp(P_i, min=1e-30)
            # diag(1/P, P) @ M
            inv_P = 1.0 / P_safe
            # Apply diag scaling.
            M[..., 0, 0] = M[..., 0, 0] * inv_P
            M[..., 0, 1] = M[..., 0, 1] * inv_P
            M[..., 1, 0] = M[..., 1, 0] * P_safe
            M[..., 1, 1] = M[..., 1, 1] * P_safe
            Ltilde = torch.matmul(Ltilde, M)

        a = Ltilde[..., 0, 0]
        c = Ltilde[..., 1, 0]
        # Guard against division by zero.
        a_safe = torch.where(a.abs() < 1e-30, torch.full_like(a, 1e-30), a)
        T_total = 1.0 / a_safe
        R_total = c / a_safe
        return R_total, T_total

    Rs_t, Ts_t = _accumulate("s")
    Rp_t, Tp_t = _accumulate("p")
    return Rs_t, Rp_t, Ts_t, Tp_t
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_incoherent_tmm.py::test_inc_three_real_layers_matches_reference -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/film_solver_isotropic.py tests/test_incoherent_tmm.py
git commit -m "feat(solver): add create_intensity_RT_isotropic main function

Implements the intensity transfer matrix accumulation. Currently
covers the all-incoherent case (no coherent stacks). Validated
against tmm_numpy.inc_tmm for a 3-layer real-index stack."
```

---

## Task 5: Cover the mixed coherent/incoherent path

**Files:**
- Test: `tests/test_incoherent_tmm.py`

**Why:** Task 4 already implemented the full code path including coherent stacks, but only one test exercises it. Add the targeted tests that match the reference incoherent test cases in `tmm_numpy/tests.py:252-307` to catch regressions in the coherent-stack handling.

- [ ] **Step 1: Add tests for the mixed case**

Append to `tests/test_incoherent_tmm.py`:

```python
def test_inc_one_coherent_layer_between_incoherent_matches_coh_tmm():
    """i | c | i with all real n: should equal coh_tmm result (no incoherent thickness)."""
    n_list = [1.0, 2.0, 3.0]
    d_list = [INF, 0.100, INF]  # 100 nm
    c_list_full = ["i", "c", "i"]
    c_list_interior = ["c"]
    theta = float(np.pi / 4)
    wv = 0.500  # 500 nm

    ref_d = [INF, 100.0, INF]
    ref_s = ref_inc_tmm("s", n_list, ref_d, c_list_full, theta, 500.0)
    ref_p = ref_inc_tmm("p", n_list, ref_d, c_list_full, theta, 500.0)

    n_t = torch.tensor([n_list[1:-1]], dtype=torch.complex64)
    d_t = torch.tensor([d_list[1:-1]], dtype=torch.float32)
    wv_t = torch.tensor([[wv]], dtype=torch.float32)
    th_t = torch.tensor([[theta]], dtype=torch.float32)

    Rs, Rp, Ts, Tp = create_intensity_RT_isotropic(
        n_t, d_t, wv_t, n_in=n_list[0], n_out=n_list[-1],
        theta_1d=th_t, c_list=c_list_interior,
    )
    assert np.allclose(Rs.item(), ref_s["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Rp.item(), ref_p["R"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Ts.item(), ref_s["T"], rtol=RTOL, atol=ATOL)
    assert np.allclose(Tp.item(), ref_p["T"], rtol=RTOL, atol=ATOL)


def test_inc_thin_film_on_thick_substrate_matches_random_avg_of_coh_tmm():
    """The motivating use case: thin film on a thick incoherent substrate.

    Stack: air | 100nm Ta2O5 | thick glass substrate | air.
    c_list interior: ['c', 'i'] -- the substrate is incoherent.
    Per tmm_numpy/tests.py:309-334, this should equal the average of the
    coherent solver over a range of (random) substrate thicknesses.
    """
    n_air, n_film, n_sub = 1.0, 2.10 + 0.001j, 1.52
    d_film = 0.100  # 100 nm
    wv = 0.550
    theta = 0.0

    n_list_full = [n_air, n_film, n_sub, n_air]
    c_list_full = ["i", "c", "i", "i"]
    c_list_interior = ["c", "i"]

    # Reference: tmm_numpy with c_list=['i','c','i','i'] (substrate thickness doesn't matter).
    ref_d = [INF, 100.0, 1.0, INF]  # nm; substrate thickness irrelevant for inc_tmm
    ref_s = ref_inc_tmm("s", n_list_full, ref_d, c_list_full, theta, 550.0)
    ref_p = ref_inc_tmm("p", n_list_full, ref_d, c_list_full, theta, 550.0)

    # DiffTMM inputs: interior is [film, substrate]; substrate thickness
    # 0.5 mm = 500 um (any nonzero value works since c='i' and Im(n_sub)=0).
    n_t = torch.tensor([[n_film, n_sub]], dtype=torch.complex64)
    d_t = torch.tensor([[d_film, 500.0]], dtype=torch.float32)
    wv_t = torch.tensor([[wv]], dtype=torch.float32)
    th_t = torch.tensor([[theta]], dtype=torch.float32)

    Rs, Rp, Ts, Tp = create_intensity_RT_isotropic(
        n_t, d_t, wv_t, n_in=n_air, n_out=n_air,
        theta_1d=th_t, c_list=c_list_interior,
    )

    assert np.allclose(Rs.item(), ref_s["R"], rtol=1e-4, atol=1e-5)
    assert np.allclose(Rp.item(), ref_p["R"], rtol=1e-4, atol=1e-5)
    assert np.allclose(Ts.item(), ref_s["T"], rtol=1e-4, atol=1e-5)
    assert np.allclose(Tp.item(), ref_p["T"], rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `pytest tests/test_incoherent_tmm.py -v`
Expected: All tests so far PASS (including the two new ones).

If any fail, the L-matrix accumulation or stack-RT handling in `create_intensity_RT_isotropic` has a bug. Fix it before continuing — do not skip the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_incoherent_tmm.py
git commit -m "test(solver): add mixed coherent/incoherent inc_tmm cases

Covers the motivating use case (thin film on thick substrate) and
validates one-coherent-layer-between-incoherent against tmm_numpy."
```

---

## Task 6: Sweep test over angle and wavelength (vectorization correctness)

**Files:**
- Test: `tests/test_incoherent_tmm.py`

**Why:** The above tests only check single (theta, wvln) points. The whole point of DiffTMM is batched computation. We need to confirm that a 1-D sweep over angles (and wavelengths) matches looping over the reference one point at a time.

- [ ] **Step 1: Add the sweep test**

Append to `tests/test_incoherent_tmm.py`:

```python
def test_inc_sweep_angles_and_wavelengths():
    """Batched (theta, wvln) sweep matches reference looped element-wise."""
    n_air, n_film, n_sub = 1.0, 2.10 + 0.002j, 1.52
    d_film = 0.100
    c_list_full = ["i", "c", "i", "i"]
    c_list_interior = ["c", "i"]
    n_list_full = [n_air, n_film, n_sub, n_air]

    thetas = np.linspace(0.0, np.pi / 3, 5)
    wvs_um = np.linspace(0.450, 0.700, 4)

    # Reference: loop.
    ref_Rs = np.zeros((4, 5))
    ref_Rp = np.zeros((4, 5))
    ref_Ts = np.zeros((4, 5))
    ref_Tp = np.zeros((4, 5))
    for i, w in enumerate(wvs_um):
        ref_d = [INF, 100.0, 1.0, INF]
        for j, th in enumerate(thetas):
            ds = ref_inc_tmm("s", n_list_full, ref_d, c_list_full, float(th), float(w) * 1000)
            dp = ref_inc_tmm("p", n_list_full, ref_d, c_list_full, float(th), float(w) * 1000)
            ref_Rs[i, j] = ds["R"]
            ref_Ts[i, j] = ds["T"]
            ref_Rp[i, j] = dp["R"]
            ref_Tp[i, j] = dp["T"]

    # DiffTMM batched call.
    n_t = torch.tensor([[n_film, n_sub]], dtype=torch.complex64)
    d_t = torch.tensor([[d_film, 500.0]], dtype=torch.float32)
    wv_t = torch.tensor([wvs_um.tolist()], dtype=torch.float32)
    th_t = torch.tensor([thetas.tolist()], dtype=torch.float32)

    Rs, Rp, Ts, Tp = create_intensity_RT_isotropic(
        n_t, d_t, wv_t, n_in=n_air, n_out=n_air,
        theta_1d=th_t, c_list=c_list_interior,
    )

    # Shape: (1, n_wv=4, n_angles=5).
    assert Rs.shape == (1, 4, 5)
    np.testing.assert_allclose(Rs[0].cpu().numpy(), ref_Rs, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(Rp[0].cpu().numpy(), ref_Rp, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(Ts[0].cpu().numpy(), ref_Ts, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(Tp[0].cpu().numpy(), ref_Tp, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_incoherent_tmm.py::test_inc_sweep_angles_and_wavelengths -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_incoherent_tmm.py
git commit -m "test(solver): add batched angle+wavelength sweep test for inc_tmm"
```

---

## Task 7: Verify autograd through the incoherent path

**Files:**
- Test: `tests/test_incoherent_tmm.py`

**Why:** The whole reason DiffTMM exists is differentiability. We must confirm that gradients flow through `create_intensity_RT_isotropic` w.r.t. layer thicknesses (the typical inverse-design parameter). Use `torch.autograd.gradcheck` on a small example, and a smoke-test that a simple optimization loop drives the loss down.

- [ ] **Step 1: Add the gradient test**

Append to `tests/test_incoherent_tmm.py`:

```python
def test_inc_autograd_flows_through_thickness():
    """Loss = (R - target)^2 should have a non-zero gradient w.r.t. layer thickness."""
    n_air, n_film, n_sub = 1.0, 2.10 + 0.002j, 1.52
    c_list_interior = ["c", "i"]

    n_t = torch.tensor([[n_film, n_sub]], dtype=torch.complex64)
    d_t = torch.tensor([[0.080, 0.500]], dtype=torch.float32, requires_grad=True)
    wv_t = torch.tensor([[0.550]], dtype=torch.float32)
    th_t = torch.tensor([[0.0]], dtype=torch.float32)

    Rs, Rp, Ts, Tp = create_intensity_RT_isotropic(
        n_t, d_t, wv_t, n_in=n_air, n_out=n_air,
        theta_1d=th_t, c_list=c_list_interior,
    )
    loss = (Rs.mean() - 0.5) ** 2
    loss.backward()

    grad = d_t.grad
    assert grad is not None
    # The coherent film thickness must influence R (interference). The substrate
    # is incoherent and lossless, so its gradient should be ~0; the film's
    # gradient must be non-zero.
    assert abs(grad[0, 0].item()) > 1e-6, "Film thickness gradient should be non-zero"
    assert abs(grad[0, 1].item()) < 1e-6, "Lossless incoherent substrate thickness gradient should be ~0"


def test_inc_optimization_loop_reduces_loss():
    """A short Adam loop on film thickness should reduce the loss."""
    torch.manual_seed(0)
    n_air, n_film, n_sub = 1.0, 2.10, 1.52
    c_list_interior = ["c", "i"]

    target_R = torch.tensor(0.30)
    d_param = torch.tensor([[0.050, 0.500]], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([d_param], lr=0.01)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        n_t = torch.tensor([[n_film, n_sub]], dtype=torch.complex64)
        wv_t = torch.tensor([[0.550]], dtype=torch.float32)
        th_t = torch.tensor([[0.0]], dtype=torch.float32)
        Rs, Rp, Ts, Tp = create_intensity_RT_isotropic(
            n_t, d_param, wv_t, n_in=n_air, n_out=n_air,
            theta_1d=th_t, c_list=c_list_interior,
        )
        loss = (Rs.mean() - target_R) ** 2
        loss.backward()
        opt.step()
        # Keep thicknesses positive.
        with torch.no_grad():
            d_param.clamp_(min=1e-4)
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"Optimization didn't reduce loss: start={losses[0]:.4f}, end={losses[-1]:.4f}"
    )
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_incoherent_tmm.py -k "autograd or optimization" -v`
Expected: Both PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_incoherent_tmm.py
git commit -m "test(solver): verify autograd through incoherent TMM and optimization loop"
```

---

## Task 8: Add the `IncoherentIsotropicFilmSolver` class

**Files:**
- Modify: `difftmm/film_solver_isotropic.py` (append)
- Test: `tests/test_incoherent_tmm.py`

**Why:** The repo's UX convention is a solver *class* with `.simulate(theta, wvln)`, mirroring `IsotropicFilmSolver`. We add the matching class so users can do:

```python
solver = IncoherentIsotropicFilmSolver(
    mat_n_in=1.0, mat_n_out=1.0,
    mat_n_ls=[2.10, 1.52],
    thickness_ls=[0.100, 500.0],
    c_list=["c", "i"],
    device=torch.device("cuda"),
)
Rs, Rp, Ts, Tp = solver.simulate(theta=angles, wvln=[0.55])
```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_incoherent_tmm.py`:

```python
from difftmm.film_solver_isotropic import IncoherentIsotropicFilmSolver  # noqa: E402


def test_incoherent_solver_class_matches_functional_api():
    """The IncoherentIsotropicFilmSolver class must match the functional create_intensity_RT_isotropic."""
    n_air, n_film, n_sub = 1.0, 2.10 + 0.002j, 1.52
    c_list = ["c", "i"]
    thetas = torch.linspace(0.0, 1.0, 8)
    wvs = [0.500, 0.600]

    solver = IncoherentIsotropicFilmSolver(
        mat_n_in=n_air,
        mat_n_out=n_air,
        mat_n_ls=[n_film, n_sub],
        thickness_ls=[0.100, 500.0],
        c_list=c_list,
        device=torch.device("cpu"),
    )
    Rs_c, Rp_c, Ts_c, Tp_c = solver.simulate(theta=thetas, wvln=wvs)

    # Functional reference.
    n_t = torch.tensor([[n_film, n_sub]], dtype=torch.complex64)
    d_t = torch.tensor([[0.100, 500.0]], dtype=torch.float32)
    wv_t = torch.tensor([wvs], dtype=torch.float32)
    th_t = thetas.unsqueeze(0)
    Rs_f, Rp_f, Ts_f, Tp_f = create_intensity_RT_isotropic(
        n_t, d_t, wv_t, n_in=n_air, n_out=n_air, theta_1d=th_t, c_list=c_list,
    )

    assert torch.allclose(Rs_c, Rs_f, atol=1e-6)
    assert torch.allclose(Rp_c, Rp_f, atol=1e-6)
    assert torch.allclose(Ts_c, Ts_f, atol=1e-6)
    assert torch.allclose(Tp_c, Tp_f, atol=1e-6)
    # Shapes: (batch=1, n_wv=2, n_angles=8)
    assert Rs_c.shape == (1, 2, 8)


def test_incoherent_solver_requires_c_list_length_to_match_layers():
    with pytest.raises(ValueError, match="c_list length"):
        IncoherentIsotropicFilmSolver(
            mat_n_in=1.0, mat_n_out=1.0,
            mat_n_ls=[2.0, 1.5],
            thickness_ls=[0.100, 500.0],
            c_list=["c"],  # too short
            device=torch.device("cpu"),
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_incoherent_tmm.py -k incoherent_solver -v`
Expected: FAIL with `ImportError: cannot import name 'IncoherentIsotropicFilmSolver'`.

- [ ] **Step 3: Implement `IncoherentIsotropicFilmSolver`**

Append to `difftmm/film_solver_isotropic.py` (at the end of the file, after `create_intensity_RT_isotropic`):

```python
# =========================
# Solver class
# =========================
class IncoherentIsotropicFilmSolver:
    """Multi-layer coating solver with partly-incoherent layer support.

    Same UX as IsotropicFilmSolver but additionally accepts a ``c_list``
    argument that marks each interior layer as coherent ('c') or
    incoherent ('i'). Returns real power coefficients (Rs, Rp, Ts, Tp)
    rather than complex amplitudes, because incoherent calculations
    discard phase by design.
    """

    def __init__(
        self,
        mat_n_in,
        mat_n_out,
        mat_n_ls,
        c_list,
        thickness_ls=None,
        thickness_min=0.0,
        thickness_max=1000.0,  # in um; allow thick substrates by default
        batch_size=1,
        sigmoid_param=False,
        device=torch.device("cuda"),
    ):
        """Initialize the incoherent isotropic film solver.

        Args:
            mat_n_in: scalar incident refractive index.
            mat_n_out: scalar exit refractive index.
            mat_n_ls: refractive indices of interior layers (list or tensor of length N).
            c_list: list of 'c'/'i' codes, length N. One per interior layer.
            thickness_ls: thicknesses of interior layers in um, length N. If None,
                          random.
            thickness_min: minimum thickness in um (used for sigmoid bounds).
            thickness_max: maximum thickness in um.
            batch_size: number of film stacks in the batch.
            sigmoid_param: if True, use sigmoid parameterization.
            device: torch device.

        Raises:
            ValueError: if c_list length does not match the number of interior layers
                        or contains invalid codes.
        """
        self.batch_size = batch_size
        self.mat_n_in = float(mat_n_in) if not isinstance(mat_n_in, complex) else complex(mat_n_in)
        self.mat_n_out = float(mat_n_out) if not isinstance(mat_n_out, complex) else complex(mat_n_out)
        self.device = device

        if torch.is_tensor(mat_n_ls):
            n_layers_t = mat_n_ls.to(torch.complex64)
        else:
            n_layers_t = torch.tensor(mat_n_ls, dtype=torch.complex64)
        self.num_layers = len(n_layers_t)

        if len(c_list) != self.num_layers:
            raise ValueError(
                f"c_list length {len(c_list)} does not match number of interior "
                f"layers {self.num_layers}."
            )
        for code in c_list:
            if code not in ("c", "i"):
                raise ValueError("c_list entries must be 'c' or 'i'.")
        self.c_list = list(c_list)

        self.refract_idx_layers = n_layers_t.unsqueeze(0).expand(batch_size, -1).clone()

        self.thickness_min = thickness_min
        self.thickness_max = thickness_max
        self._thickness_range = thickness_max - thickness_min

        self.sigmoid_param = sigmoid_param
        if thickness_ls is not None:
            if not torch.is_tensor(thickness_ls):
                thickness_ls = torch.tensor(thickness_ls, dtype=torch.float32)
            normalized = (
                thickness_ls.clamp(thickness_min, thickness_max) - thickness_min
            ) / max(self._thickness_range, 1e-30)
            self.film_params = normalized.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            self.film_params = torch.randn(batch_size, self.num_layers) * 0.01 + 0.5

        if self.sigmoid_param:
            self.film_params = inv_sigmoid(self.film_params.clamp(1e-6, 1 - 1e-6))

        self.to(device)

    def to(self, device):
        self.device = device
        self.film_params = self.film_params.to(device, non_blocking=True)
        self.refract_idx_layers = self.refract_idx_layers.to(device, non_blocking=True)
        return self

    def get_film_thickness(self):
        if self.sigmoid_param:
            return (
                torch.sigmoid(self.film_params) * self._thickness_range + self.thickness_min
            )
        thickness = self.film_params * self._thickness_range + self.thickness_min
        return thickness.clamp(self.thickness_min, self.thickness_max)

    def simulate(self, theta, wvln):
        """Compute (Rs, Rp, Ts, Tp) for the configured stack.

        Args:
            theta: angles in radians. 1D of shape (n_angles,) or 2D (batch, n_angles).
            wvln: wavelengths in um. Scalar, list, or 1D tensor.

        Returns:
            Rs, Rp, Ts, Tp: real tensors of shape (batch, n_wvlns, n_angles).
        """
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        theta = theta.to(self.device)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand(self.batch_size, -1)

        if torch.is_tensor(wvln):
            wv = wvln.to(self.device)
            if wv.dim() == 0:
                wv = wv.unsqueeze(0)
        elif isinstance(wvln, (list, tuple)):
            wv = torch.tensor(wvln, dtype=torch.float32, device=self.device)
        else:
            wv = torch.tensor([wvln], dtype=torch.float32, device=self.device)
        wv_batch = wv.unsqueeze(0).expand(self.batch_size, -1)

        d_batch = self.get_film_thickness()
        return create_intensity_RT_isotropic(
            self.refract_idx_layers, d_batch, wv_batch,
            self.mat_n_in, self.mat_n_out, theta, self.c_list,
        )

    def __call__(self, theta, wvln):
        return self.simulate(theta, wvln)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_incoherent_tmm.py -k incoherent_solver -v`
Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/film_solver_isotropic.py tests/test_incoherent_tmm.py
git commit -m "feat(solver): add IncoherentIsotropicFilmSolver class

Mirrors IsotropicFilmSolver UX but with a c_list argument and real
(Rs, Rp, Ts, Tp) output. Validated against the functional API."
```

---

## Task 9: Expose the new API from `difftmm.__init__`

**Files:**
- Modify: `difftmm/__init__.py`
- Test: `tests/test_incoherent_tmm.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_incoherent_tmm.py`:

```python
def test_public_api_exposes_incoherent_symbols():
    import difftmm
    assert hasattr(difftmm, "IncoherentIsotropicFilmSolver")
    assert hasattr(difftmm, "create_intensity_RT_isotropic")
    assert "IncoherentIsotropicFilmSolver" in difftmm.__all__
    assert "create_intensity_RT_isotropic" in difftmm.__all__
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_incoherent_tmm.py::test_public_api_exposes_incoherent_symbols -v`
Expected: FAIL with `AttributeError: module 'difftmm' has no attribute 'IncoherentIsotropicFilmSolver'`.

- [ ] **Step 3: Update `difftmm/__init__.py`**

Replace the entire contents with:

```python
from .film_solver_isotropic import (
    IsotropicFilmSolver,
    IncoherentIsotropicFilmSolver,
    create_jones_matrix_isotropic,
    create_intensity_RT_isotropic,
)
from .film_solver_anisotropic import (
    FilmSolver,
    create_jones_matrix_AOIAz,
)

AnisotropicFilmSolver = FilmSolver

__all__ = [
    "IsotropicFilmSolver",
    "FilmSolver",
    "AnisotropicFilmSolver",
    "IncoherentIsotropicFilmSolver",
    "create_jones_matrix_isotropic",
    "create_jones_matrix_AOIAz",
    "create_intensity_RT_isotropic",
]
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_incoherent_tmm.py::test_public_api_exposes_incoherent_symbols -v`
Expected: PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add difftmm/__init__.py tests/test_incoherent_tmm.py
git commit -m "feat(api): expose IncoherentIsotropicFilmSolver from difftmm package"
```

---

## Task 10: Add the visual benchmark script

**Files:**
- Create: `benchmarks/4_compare_incoherent.py`

**Why:** The repo already uses `benchmarks/*.py` scripts to produce comparison plots vs `tmm_numpy`. Add a counterpart for incoherent TMM that mirrors the style of `1_compare_angle_response_isotropic.py` and visually demonstrates the elimination of ripples on a thick-substrate stack.

- [ ] **Step 1: Create the benchmark script**

```python
"""Compare DiffTMM incoherent TMM against tmm_numpy.inc_tmm.

Reproduces the motivating use case from issue #2: a thin film on a thick
incoherent substrate. The coherent calculation produces dense Fabry-Perot
ripples; the incoherent calculation smooths them out.

Output: benchmarks/incoherent_tmm_comparison.png
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import inf, linspace, pi

from difftmm import IncoherentIsotropicFilmSolver, IsotropicFilmSolver
from tmm_numpy.tmm_core import coh_tmm, inc_tmm

DEGREE = pi / 180


def main():
    # Stack: air | 100 nm TiO2 | 1 mm glass | air. Wavelength sweep at normal incidence.
    n_air = 1.0
    n_film = 2.40 + 0.001j  # TiO2 with a touch of loss for stability
    n_sub = 1.52
    d_film_um = 0.100
    d_sub_um = 1000.0  # 1 mm

    wvs_nm = np.linspace(400, 800, 401)
    wvs_um = wvs_nm / 1000.0
    theta = 0.0

    # 1. Reference: tmm_numpy coherent (thick substrate -> dense ripples).
    R_coh_np = np.zeros_like(wvs_nm)
    for i, w_nm in enumerate(wvs_nm):
        ref = coh_tmm(
            "s",
            [n_air, n_film, n_sub, n_air],
            [inf, d_film_um * 1000, d_sub_um * 1000, inf],
            theta,
            w_nm,
        )
        R_coh_np[i] = ref["R"]

    # 2. Reference: tmm_numpy incoherent (smooth).
    R_inc_np = np.zeros_like(wvs_nm)
    for i, w_nm in enumerate(wvs_nm):
        ref = inc_tmm(
            "s",
            [n_air, n_film, n_sub, n_air],
            [inf, d_film_um * 1000, d_sub_um * 1000, inf],
            ["i", "c", "i", "i"],
            theta,
            w_nm,
        )
        R_inc_np[i] = ref["R"]

    # 3. DiffTMM incoherent (batched).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = IncoherentIsotropicFilmSolver(
        mat_n_in=n_air,
        mat_n_out=n_air,
        mat_n_ls=[n_film, n_sub],
        c_list=["c", "i"],
        thickness_ls=[d_film_um, d_sub_um],
        device=device,
    )
    Rs, Rp, Ts, Tp = solver.simulate(theta=torch.tensor([theta]), wvln=wvs_um.tolist())
    R_inc_torch = Rs[0, :, 0].cpu().numpy()

    # 4. DiffTMM coherent (for comparison: dense ripples).
    coh_solver = IsotropicFilmSolver(
        mat_n_in=n_air,
        mat_n_out=n_air,
        mat_n_ls=[n_film, n_sub],
        thickness_ls=[d_film_um, d_sub_um],
        thickness_max=2 * d_sub_um,
        device=device,
    )
    ts, tp, rs_amp, rp_amp = coh_solver.simulate(theta=torch.tensor([theta]), wvln=wvs_um.tolist())
    R_coh_torch = (rs_amp[0, :, 0].abs() ** 2).cpu().numpy()

    # Plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(wvs_nm, R_coh_np, "C0", lw=0.4, alpha=0.7, label="tmm_numpy (coherent)")
    ax1.plot(wvs_nm, R_coh_torch, "C3--", lw=0.4, alpha=0.7, label="DiffTMM (coherent)")
    ax1.set_title("Coherent: dense Fabry-Perot ripples (thick substrate)")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance R_s")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(wvs_nm, R_inc_np, "C0", lw=1.5, label="tmm_numpy (incoherent)")
    ax2.plot(wvs_nm, R_inc_torch, "C3--", lw=1.5, label="DiffTMM (incoherent)")
    ax2.set_title("Incoherent: smooth, ripple-free")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Reflectance R_s")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Incoherent TMM: thin film on thick substrate\n"
        "stack = air | 100 nm TiO2 | 1 mm glass | air, normal incidence",
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "incoherent_tmm_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    print(
        f"Max abs(R_inc_torch - R_inc_np) = {np.max(np.abs(R_inc_torch - R_inc_np)):.2e}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

Run: `python benchmarks/4_compare_incoherent.py`
Expected: Prints `Saved benchmarks/incoherent_tmm_comparison.png` and a max-abs-error < 1e-4.

If running on a CPU-only machine, this will still work (the script falls back to CPU).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/4_compare_incoherent.py benchmarks/incoherent_tmm_comparison.png
git commit -m "bench: add incoherent TMM comparison plot vs tmm_numpy

Shows the motivating use case: thin film on a 1 mm thick glass substrate.
The coherent calculation produces dense Fabry-Perot ripples; the
incoherent calculation is smooth and matches tmm_numpy.inc_tmm."
```

---

## Task 11: Document the new API in README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Insert a new "Incoherent layers" subsection**

Open `README.md`. After the "Two Solvers" section (currently ending around line 101 in the version on disk) and *before* the "Accuracy Validation" section (line 103), insert this content:

```markdown
### Incoherent Layers (thick substrates)

For stacks containing layers thicker than the source's coherence length
(typically anything thicker than ~10 μm for broadband illumination), the
fully-coherent calculation produces dense Fabry-Perot ripples that do not
appear in real measurements. The `IncoherentIsotropicFilmSolver` lets you
mark individual layers as incoherent (`'i'`) while keeping thin films
coherent (`'c'`):

```python
import torch
from difftmm import IncoherentIsotropicFilmSolver

# Stack: air | 100 nm TiO2 | 1 mm glass | air
solver = IncoherentIsotropicFilmSolver(
    mat_n_in=1.0,
    mat_n_out=1.0,
    mat_n_ls=[2.40, 1.52],
    c_list=["c", "i"],           # TiO2 coherent, glass incoherent
    thickness_ls=[0.100, 1000.0],
    device=torch.device("cuda"),
)
Rs, Rp, Ts, Tp = solver.simulate(
    theta=torch.tensor([0.0]),
    wvln=[0.55],
)
# Returns real power coefficients in [0, 1].
```

`c_list` is per-interior-layer; the two semi-infinite media are always
treated as incoherent. The coherent path (`IsotropicFilmSolver`) and the
incoherent path (`IncoherentIsotropicFilmSolver`) share the same forward
mathematics for coherent stacks, so an all-`'c'` `c_list` (with
incoherent semi-infinite endpoints) produces results consistent with the
coherent solver up to the loss of complex phase.

Only the 2x2 isotropic solver supports incoherent layers today.
Anisotropic incoherent TMM is tracked as future work.
```

- [ ] **Step 2: Update the "Repository Structure" tree**

Find the `Repository Structure` section in `README.md` (line ~132). The existing `difftmm/` block already lists the two solver files. Update the comment on `film_solver_isotropic.py` to reflect that it now also hosts the incoherent path:

```
├── difftmm/                          # Importable package
│   ├── __init__.py                   #   Public API
│   ├── film_solver_isotropic.py      #   2x2 isotropic solver (coherent + incoherent)
│   └── film_solver_anisotropic.py    #   4x4 anisotropic solver (general)
```

And add a new `tests/` entry directly under the `difftmm/` block (sibling of `benchmarks/`):

```
├── tests/                          # pytest suite
│   └── test_incoherent_tmm.py
```

- [ ] **Step 3: Verify the README still renders**

Spot-check by previewing the diff:

```bash
git diff README.md
```

The new code block should be inside a single fenced section; the inner ```` ```python ```` and outer ```` ``` ```` must be balanced (the example uses a tilde-style outer fence? No — both are triple backticks. To avoid breaking the outer fence, we use four backticks for the outer fence if needed. If using a stricter Markdown renderer, change the outer fence to four backticks ```` ```` ```` to wrap the inner ```` ``` ```` block. Verify the rendered output looks right before committing.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): document IncoherentIsotropicFilmSolver API and use case"
```

---

## Task 12: Update `pyproject.toml` to exclude `tests/` from packaging and add `tests` to dev extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inspect current exclude list**

Read `pyproject.toml` lines 72-75. The exclude list currently has `["benchmarks*", "tmm_numpy*", "__pycache__*"]`.

- [ ] **Step 2: Add `tests*` to the exclude list**

Open `pyproject.toml`. Find the line:

```toml
exclude = ["benchmarks*", "tmm_numpy*", "__pycache__*"]
```

Replace it with:

```toml
exclude = ["benchmarks*", "tmm_numpy*", "tests*", "__pycache__*"]
```

This keeps the published wheel small and avoids shipping test data.

- [ ] **Step 3: Verify the build still produces a valid wheel**

Run: `python -m build --wheel --outdir /tmp/difftmm-wheel-test`
Expected: A `difftmm-*.whl` file is produced. Inspect with `unzip -l /tmp/difftmm-wheel-test/difftmm-*.whl` and confirm there are no `tests/` files in the listing.

If `python -m build` is not installed, install with `pip install build` first. Skip this step if the dev environment doesn't support it; the next commit on CI will catch a regression.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: exclude tests/ from source distribution and wheel"
```

---

## Self-Review (already performed)

- **Spec coverage:**
  - "Native incoherent layer support in the differentiable PyTorch solvers (IsotropicFilmSolver and FilmSolver)" — covered for `IsotropicFilmSolver` via the new `IncoherentIsotropicFilmSolver` (Task 8). The 4x4 `FilmSolver` extension is **explicitly deferred** (mentioned in README per Task 11) because (a) the issue's motivating use case is isotropic substrates, and (b) the 4x4 path uses eigenvalue decomposition and would roughly double the implementation surface — better as a follow-up issue. If the reviewer wants 4x4 included in this PR, that is a separate plan.
  - "Ports the intensity-based transfer matrix logic into the differentiable PyTorch pipeline" — Tasks 1-4 + 7 (autograd verification).
  - "Modeling thin film stacks on both sides of thick substrates while maintaining coherence in thin films and breaking phase coherence at the thick substrate" — the algorithm allows arbitrary interleaving of `'c'` and `'i'` stacks, validated by Task 5's "thin film on thick substrate" test.

- **Placeholder scan:** No "TBD", "implement later", "add error handling", or `// removed` patterns. All code blocks are complete. Step 1 of Task 1 establishes test infra; subsequent tasks reuse the same imports and helper `_wrap_inputs`.

- **Type consistency:**
  - `c_list` is consistently a `List[str]` of `'c'/'i'` codes (the *interior*-only variant passed to `create_intensity_RT_isotropic` and `IncoherentIsotropicFilmSolver`); the *full* `['i'] + c_list + ['i']` variant is built internally and never exposed publicly.
  - `n_layers_1d`, `d_1d`, `wv_1d`, `theta_1d` shapes match those used by the existing `create_jones_matrix_isotropic`: 2D `(batch, ·)` tensors.
  - Return tuple is always `(Rs, Rp, Ts, Tp)` — real tensors, shape `(batch, n_wvlns, n_angles)`, in `[0, 1]`.
  - Function name `group_layers_by_coherence` is used in both the implementation (Task 2) and in `create_intensity_RT_isotropic` (Task 4).
  - `coh_stack_power_RT_isotropic` is defined in Task 1 in `film_solver_isotropic.py` and called by `create_intensity_RT_isotropic` (Task 4) and indirectly by `IncoherentIsotropicFilmSolver` (Task 8) — all live in the same module, so no cross-module imports are needed.

- **Out-of-scope items deliberately not included** (for the reviewer's awareness):
  - 4x4 anisotropic incoherent TMM (would be a separate ~10-task plan).
  - `inc_absorp_in_each_layer` analog (per-layer absorption breakdown). Useful but not part of the requested R/T output; can be added later.
  - GPU benchmarks for incoherent path. Existing speed benchmarks compare amplitude solvers; an incoherent speed benchmark can be added once the core is in.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-26-incoherent-tmm.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
