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
    # thickness_max must accommodate d_sub_um=1000 um; we pass it explicitly.
    R_coh_torch = None
    try:
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
    except Exception as exc:
        # The 1 mm substrate causes excessive float32 phase wrap in the coherent
        # solver; gracefully skip the DiffTMM coherent curve if it errors out.
        print(f"DiffTMM coherent solver skipped ({exc}); showing tmm_numpy coherent only.")

    # Plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(wvs_nm, R_coh_np, "C0", lw=0.4, alpha=0.7, label="tmm_numpy (coherent)")
    if R_coh_torch is not None:
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
