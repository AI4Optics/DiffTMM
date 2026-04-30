"""
https://github.com/sbyrnes321/tmm?tab=readme-ov-file
"""

from __future__ import absolute_import, division, print_function

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

import matplotlib.pyplot as plt
import torch
from numpy import array, inf, linspace, pi

from difftmm import create_jones_matrix_AOIAz, create_jones_matrix_isotropic
from tmm_numpy.tmm_core import (
    coh_tmm,
    ellips,
    find_in_structure_with_inf,
    position_resolved,
    unpolarized_RT,
)

try:
    import colorpy.colormodels
    import colorpy.illuminants

    try:
        from . import color
    except ImportError:
        import color

    colors_were_imported = True
except ImportError:
    # without colorpy, you can't run sample5(), but everything else is fine.
    colors_were_imported = False


# "5 * degree" is 5 degrees expressed in radians
# "1.2 / degree" is 1.2 radians expressed in degrees
degree = pi / 180


def example6_tmm():
    """
    An example reflection plot with a surface plasmon resonance (SPR) dip.
    Compare with http://doi.org/10.2320/matertrans.M2010003 ("Spectral and
    Angular Responses of Surface Plasmon Resonance Based on the Kretschmann
    Prism Configuration") Fig 6a
    """
    # list of layer thicknesses in nm
    d_list = [inf, 5, 30, 5, inf]
    # list of refractive indices
    n_list = [1.9, 1.46, 2.13, 1.46, 1.9]
    # wavelength in nm
    lam_vac = 633
    # list of angles to plot
    theta_list = linspace(0 * degree, 90 * degree, num=300)
    # initialize lists of y-values to plot
    Rp = []
    for theta in theta_list:
        Rp.append(coh_tmm("p", n_list, d_list, theta, lam_vac)["R"])

    plt.figure(figsize=(10, 6))
    plt.plot(theta_list / degree, Rp, "navy", linewidth=2.5)
    plt.xlabel("theta (degree)", fontsize=12)
    plt.ylabel("Fraction reflected", fontsize=12)
    plt.xlim(0, 90)
    plt.ylim(0, 1)
    plt.title(
        "Reflection of p-polarized light with Surface Plasmon Resonance\n"
        "Compare with http://doi.org/10.2320/matertrans.M2010003 Fig 6a",
        fontsize=14,
        fontweight="bold",
    )
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.grid(True, alpha=0.2, linestyle="--")
    plt.savefig(os.path.join(SCRIPT_DIR, "surface_plasmon_resonance_tmm.png"), dpi=300, bbox_inches="tight")


def example6_torch():
    """
    Same SPR calculation as example6_tmm() but using PyTorch-based film_solver.

    Uses create_jones_matrix_AOIAz() which implements the 4x4 transfer matrix method
    for potentially anisotropic media.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Film stack parameters (same as example6_tmm)
    # Original: d_list = [inf, 5, 30, 5, inf], n_list = [1.9, 1.46, 2.13, 1.46, 1.9]
    # For film_solver: only the middle layers (excluding semi-infinite media)
    n_layers = 3  # middle 3 layers
    batchsize = 1

    # Layer thicknesses in um (film_solver uses um): [5, 30, 5] nm -> [0.005, 0.030, 0.005] um
    d_1d = torch.tensor([[0.005, 0.030, 0.005]], dtype=torch.complex64, device=device)

    # Refractive indices for each layer (shape: batchsize, n_layers, 3)
    # Using isotropic materials, so all 3 components are the same
    n_2d = torch.zeros((batchsize, n_layers, 3), dtype=torch.complex64, device=device)
    n_2d[:, 0, :] = 1.46  # SiO2
    n_2d[:, 1, :] = 2.13  # TiO2
    n_2d[:, 2, :] = 1.46  # SiO2

    # Azimuth angles of materials (zero for isotropic)
    a_2d = torch.zeros((batchsize, n_layers, 3), dtype=torch.complex64, device=device)

    # Wavelength in um: 633 nm -> 0.633 um
    wv_1d = torch.tensor([[0.633]], dtype=torch.float64, device=device)

    # Incident and output media refractive indices
    n_in = 1.9  # glass (incident medium)
    n_out = 1.9  # glass (output medium)

    # Angles of incidence: 0° to 90° (300 points)
    n_angles = 300
    theta_list = torch.linspace(
        0 * degree, 90 * degree, n_angles, dtype=torch.float64, device=device
    )
    AOI_1d = theta_list.unsqueeze(0)  # shape: (batchsize, n_angles)

    # Azimuth angle (single value, set to 0)
    Az_1d = torch.zeros((batchsize, 1), dtype=torch.float64, device=device)

    # Calculate Jones matrices for transmission and reflection
    Jt, Jr = create_jones_matrix_AOIAz(
        a_2d, n_2d, d_1d, wv_1d, n_in, n_out, AOI_1d, Az_1d
    )
    # Output shape: (batchsize, n_wls, n_aoi_angles, n_az_angles, 2, 2)

    # For p-polarized light, input is [1, 0]
    p_in = torch.tensor(
        [[1.0 + 0.0j], [0.0 + 0.0j]], dtype=torch.complex64, device=device
    )
    p_in = p_in.reshape(1, 1, 1, 1, 2, 1).expand(batchsize, 1, n_angles, 1, -1, -1)

    # For s-polarized light, input is [0, 1]
    s_in = torch.tensor(
        [[0.0 + 0.0j], [1.0 + 0.0j]], dtype=torch.complex64, device=device
    )
    s_in = s_in.reshape(1, 1, 1, 1, 2, 1).expand(batchsize, 1, n_angles, 1, -1, -1)

    # Calculate reflected and transmitted fields
    r_vec_p = torch.matmul(Jr, p_in)  # p-polarized reflection
    r_vec_s = torch.matmul(Jr, s_in)  # s-polarized reflection
    t_vec_p = torch.matmul(Jt, p_in)  # p-polarized transmission
    t_vec_s = torch.matmul(Jt, s_in)  # s-polarized transmission

    # Extract coefficients (p-polarized: first component, s-polarized: second component)
    rp = r_vec_p[:, :, :, :, 0, 0].squeeze()  # p-reflection amplitude
    rs = r_vec_s[:, :, :, :, 1, 0].squeeze()  # s-reflection amplitude
    tp = t_vec_p[:, :, :, :, 0, 0].squeeze()  # p-transmission amplitude
    ts = t_vec_s[:, :, :, :, 1, 0].squeeze()  # s-transmission amplitude

    # Power coefficients = |amplitude|^2
    Rp = (rp.abs() ** 2).cpu().numpy()
    Rs = (rs.abs() ** 2).cpu().numpy()
    Tp = (tp.abs() ** 2).cpu().numpy()
    Ts = (ts.abs() ** 2).cpu().numpy()

    theta_degrees = theta_list.cpu().numpy() / degree

    # Plot results (only Rp for now, but could extend to show all coefficients)
    plt.figure(figsize=(10, 6))
    plt.plot(theta_degrees, Rp, "darkgreen", label="Ours (Rp)", linewidth=2.5)
    plt.xlabel("theta (degree)", fontsize=12)
    plt.ylabel("Fraction reflected", fontsize=12)
    plt.xlim(0, 90)
    plt.ylim(0, max(0.01, Rp.max() * 1.1))
    plt.title(
        "Reflection of p-polarized light (PyTorch film_solver)\n"
        "Surface Plasmon Resonance simulation",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.grid(True, alpha=0.2, linestyle="--")
    plt.savefig(os.path.join(SCRIPT_DIR, "surface_plasmon_resonance_torch.png"), dpi=300, bbox_inches="tight")

    print(f"PyTorch simulation completed. Device: {device}")
    print(f"Rp range: [{Rp.min():.6f}, {Rp.max():.6f}]")
    print(f"Rs range: [{Rs.min():.6f}, {Rs.max():.6f}]")
    print(f"Tp range: [{Tp.min():.6f}, {Tp.max():.6f}]")
    print(f"Ts range: [{Ts.min():.6f}, {Ts.max():.6f}]")

    return theta_degrees, Rp, Rs, Tp, Ts


def example6_isotropic():
    """
    Same SPR calculation using the fast isotropic film solver.

    Uses create_jones_matrix_isotropic() which implements the standard 2x2 
    transfer matrix method - much faster than the 4x4 anisotropic formulation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Film stack parameters (same as example6_tmm)
    # Original: d_list = [inf, 5, 30, 5, inf], n_list = [1.9, 1.46, 2.13, 1.46, 1.9]
    # For film_solver: only the middle layers (excluding semi-infinite media)
    n_layers = 3  # middle 3 layers
    batchsize = 1

    # Layer thicknesses in um: [5, 30, 5] nm -> [0.005, 0.030, 0.005] um
    d_1d = torch.tensor([[0.005, 0.030, 0.005]], dtype=torch.float32, device=device)

    # Refractive indices for each layer (shape: batchsize, n_layers)
    # Isotropic solver uses single refractive index per layer
    n_layers_1d = torch.tensor([[1.46, 2.13, 1.46]], dtype=torch.complex64, device=device)

    # Wavelength in um: 633 nm -> 0.633 um
    wv_1d = torch.tensor([[0.633]], dtype=torch.float32, device=device)

    # Incident and output media refractive indices
    n_in = 1.9  # glass (incident medium)
    n_out = 1.9  # glass (output medium)

    # Angles of incidence: 0° to 90° (300 points)
    n_angles = 300
    theta_list = torch.linspace(
        0 * degree, 90 * degree, n_angles, dtype=torch.float32, device=device
    )
    theta_1d = theta_list.unsqueeze(0)  # shape: (batchsize, n_angles)

    # Calculate transmission and reflection coefficients directly
    ts, tp, rs, rp = create_jones_matrix_isotropic(
        n_layers_1d, d_1d, wv_1d, n_in, n_out, theta_1d
    )
    # Output shape: (batchsize, n_wls, n_angles)

    # Power coefficients = |amplitude|^2
    Rp = (rp.abs() ** 2).squeeze().cpu().numpy()
    Rs = (rs.abs() ** 2).squeeze().cpu().numpy()
    Tp = (tp.abs() ** 2).squeeze().cpu().numpy()
    Ts = (ts.abs() ** 2).squeeze().cpu().numpy()

    theta_degrees = theta_list.cpu().numpy() / degree

    print(f"Isotropic solver completed. Device: {device}")
    print(f"Rp range: [{Rp.min():.6f}, {Rp.max():.6f}]")
    print(f"Rs range: [{Rs.min():.6f}, {Rs.max():.6f}]")
    print(f"Tp range: [{Tp.min():.6f}, {Tp.max():.6f}]")
    print(f"Ts range: [{Ts.min():.6f}, {Ts.max():.6f}]")

    return theta_degrees, Rp, Rs, Tp, Ts


def compare_tmm_torch():
    """Compare all Fresnel coefficients: TMM vs PyTorch (anisotropic) vs PyTorch (isotropic)."""
    import numpy as np

    # Get TMM results for all coefficients
    d_list = [inf, 5, 30, 5, inf]
    n_list = [1.9, 1.46, 2.13, 1.46, 1.9]
    lam_vac = 633
    theta_list_np = linspace(0 * degree, 90 * degree, num=300)

    Rs_tmm, Rp_tmm, Ts_tmm, Tp_tmm = [], [], [], []
    for theta in theta_list_np:
        # s-polarization
        s_result = coh_tmm("s", n_list, d_list, theta, lam_vac)
        Rs_tmm.append(s_result["R"])
        Ts_tmm.append(s_result["T"])

        # p-polarization
        p_result = coh_tmm("p", n_list, d_list, theta, lam_vac)
        Rp_tmm.append(p_result["R"])
        Tp_tmm.append(p_result["T"])

    Rs_tmm = array(Rs_tmm)
    Rp_tmm = array(Rp_tmm)
    Ts_tmm = array(Ts_tmm)
    Tp_tmm = array(Tp_tmm)

    # Get PyTorch anisotropic results
    theta_aniso, Rp_aniso, Rs_aniso, Tp_aniso, Ts_aniso = example6_torch()

    # Get PyTorch isotropic results
    theta_iso, Rp_iso, Rs_iso, Tp_iso, Ts_iso = example6_isotropic()

    # Create comprehensive comparison plots (3 methods)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    coefficients = [
        (Rp_tmm, Rp_aniso, Rp_iso, "Rp", "p-Reflection", 0, 0),
        (Rs_tmm, Rs_aniso, Rs_iso, "Rs", "s-Reflection", 0, 1),
        (Tp_tmm, Tp_aniso, Tp_iso, "Tp", "p-Transmission", 1, 0),
        (Ts_tmm, Ts_aniso, Ts_iso, "Ts", "s-Transmission", 1, 1),
    ]

    for tmm_data, aniso_data, iso_data, coeff_name, title, row, col in coefficients:
        ax = axes[row, col]
        ax.plot(theta_list_np / degree, tmm_data, "royalblue", label="TMM (reference)", linewidth=3)
        ax.plot(
            theta_aniso,
            aniso_data,
            "darkorange",
            label="Anisotropic (4x4)",
            linewidth=2.5,
            linestyle="--",
        )
        ax.plot(
            theta_iso,
            iso_data,
            "green",
            label="Isotropic (2x2)",
            linewidth=2,
            linestyle=":",
        )

        ax.set_xlabel("Angle of incidence (degrees)", fontsize=16)
        ax.set_ylabel(f"Fraction of {coeff_name}", fontsize=16)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, max(tmm_data.max(), aniso_data.max(), iso_data.max()) * 1.1 + 0.001)
        ax.set_title(f"{title} Coefficient", fontsize=16, fontweight="bold")
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        ax.grid(True, alpha=0.2, linestyle="--")

        # Increase tick label font size
        ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(SCRIPT_DIR, "surface_plasmon_resonance_all_coefficients.png"), dpi=300, bbox_inches="tight"
    )

    # Print comparison statistics
    print(f"\n{'=' * 70}")
    print("COMPREHENSIVE FRESNEL COEFFICIENTS COMPARISON")
    print("TMM (reference) vs Anisotropic (4x4) vs Isotropic (2x2)")
    print(f"{'=' * 70}")

    coeff_pairs = [
        ("Rp (p-reflection)", Rp_tmm, Rp_aniso, Rp_iso),
        ("Rs (s-reflection)", Rs_tmm, Rs_aniso, Rs_iso),
        ("Tp (p-transmission)", Tp_tmm, Tp_aniso, Tp_iso),
        ("Ts (s-transmission)", Ts_tmm, Ts_aniso, Ts_iso),
    ]

    max_diff_aniso = 0
    max_diff_iso = 0
    for name, tmm_data, aniso_data, iso_data in coeff_pairs:
        diff_aniso = abs(tmm_data - aniso_data).max()
        diff_iso = abs(tmm_data - iso_data).max()
        rms_aniso = np.sqrt(np.mean((tmm_data - aniso_data) ** 2))
        rms_iso = np.sqrt(np.mean((tmm_data - iso_data) ** 2))
        max_diff_aniso = max(max_diff_aniso, diff_aniso)
        max_diff_iso = max(max_diff_iso, diff_iso)
        
        print(f"\n{name}:")
        print(f"  TMM range:        [{tmm_data.min():.6f}, {tmm_data.max():.6f}]")
        print(f"  Anisotropic (4x4):")
        print(f"    Range:          [{aniso_data.min():.6f}, {aniso_data.max():.6f}]")
        print(f"    Max |diff|:     {diff_aniso:.2e}")
        print(f"    RMS diff:       {rms_aniso:.2e}")
        print(f"  Isotropic (2x2):")
        print(f"    Range:          [{iso_data.min():.6f}, {iso_data.max():.6f}]")
        print(f"    Max |diff|:     {diff_iso:.2e}")
        print(f"    RMS diff:       {rms_iso:.2e}")

    print(f"\n{'=' * 70}")
    print("SUMMARY:")
    print(f"  Anisotropic (4x4) max error: {max_diff_aniso:.2e} {'✓ PASS' if max_diff_aniso < 1e-4 else '✗ FAIL'}")
    print(f"  Isotropic (2x2) max error:   {max_diff_iso:.2e} {'✓ PASS' if max_diff_iso < 1e-4 else '✗ FAIL'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    print("Running TMM simulation...")
    example6_tmm()

    print("\nRunning PyTorch anisotropic (4x4) film_solver simulation...")
    example6_torch()

    print("\nRunning PyTorch isotropic (2x2) film_solver simulation...")
    example6_isotropic()

    print("\nGenerating comprehensive comparison (TMM vs Anisotropic vs Isotropic)...")
    compare_tmm_torch()
