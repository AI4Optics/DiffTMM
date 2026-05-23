"""Anisotropic Multi-layer Thin Film Solver (4x4 Transfer Matrix Method)

Differentiable thin film solver using the general 4x4 transfer matrix method.
Supports both isotropic and anisotropic (birefringent) materials. Computes
Fresnel coefficients (ts, tp, rs, rp) for multi-layer film stacks with full
autograd support.

Copyright (c) 2026, Xinge Yang, Qingyuan Fan, Zhaocheng Liu.
"""

import os

import torch


# =========================
# Utility functions
# =========================
def inv_sigmoid(x):
    """Inverse sigmoid function."""
    return torch.log(x / (1 - x))


def complex_arcsin(x):
    """
    Compute arcsin that returns complex values when |x| > 1,
    mimicking numpy.lib.scimath.arcsin behavior exactly.

    This is essential for correctly handling evanescent waves
    beyond the critical angle in thin film calculations.

    For |x| > 1, numpy.lib.scimath.arcsin returns pi/2 + i*arccosh(|x|) for x > 1
    and -pi/2 - i*arccosh(|x|) for x < -1.

    Args:
        x: Input tensor (real or complex)

    Returns:
        Complex tensor with arcsin values matching numpy.lib.scimath behavior
    """
    # Ensure complex type for proper handling
    if not x.is_complex():
        x_complex = x.to(torch.complex128)
    else:
        x_complex = x.to(torch.complex128)

    # For the branch cut that matches numpy.lib.scimath.arcsin:
    # arcsin(x) = -i * ln(ix + sqrt(1 - x^2))
    # where the sqrt uses the principal branch with positive imaginary part for negative real arguments

    one_minus_x2 = 1 - x_complex**2

    # Compute sqrt with the correct branch: for negative real numbers, return positive imaginary
    # numpy's sqrt of negative real is positive imaginary, so sqrt(-4) = 2j
    sqrt_term = torch.sqrt(one_minus_x2)

    # The key fix: when 1-x^2 is negative real (i.e., |x| > 1 for real x),
    # we need to ensure the imaginary part of sqrt has the correct sign
    # numpy.lib.scimath uses the convention that gives positive imaginary part
    # for arcsin of real x > 1

    # For real x > 1: 1-x^2 < 0, so sqrt(1-x^2) should be +i*sqrt(x^2-1)
    # But torch.sqrt of negative real gives -i*sqrt(|val|), so we need to fix this

    # Check where we have real x with |x| > 1 (where 1-x^2 is negative real)
    is_real_input = (
        (x.imag.abs() < 1e-10)
        if x.is_complex()
        else torch.ones_like(x, dtype=torch.bool)
    )
    needs_sign_flip = (
        is_real_input & (one_minus_x2.real < 0) & (one_minus_x2.imag.abs() < 1e-10)
    )

    # Flip the sign of sqrt where needed to match numpy convention
    sqrt_term = torch.where(needs_sign_flip, -sqrt_term, sqrt_term)

    # Now compute arcsin using the standard formula
    result = -1j * torch.log(1j * x_complex + sqrt_term)

    return result.to(torch.complex64)


# =========================
# Physics simulation functions
# =========================
def EnterExitMatrix_AOIAz(eps_in, eps_out, theta_in, theta_out):
    """
    Calculate enter and exit matrices for thin film simulation.

    Args:
        eps_in: real, permittivity of incident medium
        eps_out: real, permittivity of output medium
        theta_in: AOI angle in the shape of (batchsize, n_angles)
        theta_out: AOI angle in the shape of (batchsize, n_angles)

    Returns:
        T0 and Tinv: shape (batchsize, n_angles, 4, 4)
    """
    device = theta_in.device

    batchsize, size = theta_in.shape[:2]
    T_0 = torch.zeros((batchsize, size, 4, 4), dtype=torch.complex64).to(device)
    T_N_inv = torch.zeros((batchsize, size, 4, 4), dtype=torch.complex64).to(device)

    er_in = eps_in
    erz_in = torch.cos((theta_in)) ** 2 * eps_in
    serz_in = torch.sqrt(erz_in)

    T_0[:, :, 0, 0] = 1
    T_0[:, :, 0, 2] = 1
    T_0[:, :, 1, 0] = er_in / serz_in
    T_0[:, :, 1, 2] = -er_in / serz_in
    T_0[:, :, 2, 1] = 1
    T_0[:, :, 2, 3] = 1
    T_0[:, :, 3, 1] = serz_in
    T_0[:, :, 3, 3] = -serz_in

    er_out = eps_out
    erz_out = torch.cos((theta_out)) ** 2 * eps_out
    serz_out = torch.sqrt(erz_out)

    T_N_inv[:, :, 0, 0] = 1 / 2.0
    T_N_inv[:, :, 0, 1] = serz_out / er_out / 2.0
    T_N_inv[:, :, 1, 2] = 1 / 2.0
    T_N_inv[:, :, 1, 3] = 1 / serz_out / 2.0
    T_N_inv[:, :, 2, 0] = 1 / 2.0
    T_N_inv[:, :, 2, 1] = -serz_out / er_out / 2.0
    T_N_inv[:, :, 3, 2] = 1 / 2.0
    T_N_inv[:, :, 3, 3] = -1 / serz_out / 2.0

    return T_0, T_N_inv


def EnterExitMatrix_XY(eps_in, eps_out, theta_in_2d, theta_out_2d):
    """Calculate enter and exit matrices for XY configuration."""
    device = theta_in_2d.device

    batchsize = theta_in_2d.shape[0]
    size_1 = theta_in_2d.size()[1]
    size_2 = theta_in_2d.size()[2]
    T_0_4d = torch.zeros((batchsize, size_1, size_2, 4, 4), dtype=torch.complex64).to(
        device
    )
    T_N_inv_4d = torch.zeros(
        (batchsize, size_1, size_2, 4, 4), dtype=torch.complex64
    ).to(device)

    er_in = eps_in
    erz_in_2d = torch.pow(torch.cos((theta_in_2d)), 2) * eps_in
    serz_in_2d = torch.sqrt(erz_in_2d)

    T_0_4d[:, :, :, 0, 0] = 1
    T_0_4d[:, :, :, 0, 2] = 1
    T_0_4d[:, :, :, 1, 0] = er_in / serz_in_2d
    T_0_4d[:, :, :, 1, 2] = -er_in / serz_in_2d
    T_0_4d[:, :, :, 2, 1] = 1
    T_0_4d[:, :, :, 2, 3] = 1
    T_0_4d[:, :, :, 3, 1] = serz_in_2d
    T_0_4d[:, :, :, 3, 3] = -serz_in_2d

    er_out = eps_out
    erz_out_2d = torch.pow(torch.cos((theta_out_2d)), 2) * eps_out
    serz_out_2d = torch.sqrt(erz_out_2d)

    T_N_inv_4d[:, :, :, 0, 0] = 1 / 2.0
    T_N_inv_4d[:, :, :, 0, 1] = serz_out_2d / er_out / 2.0
    T_N_inv_4d[:, :, :, 1, 2] = 1 / 2.0
    T_N_inv_4d[:, :, :, 1, 3] = 1 / serz_out_2d / 2.0
    T_N_inv_4d[:, :, :, 2, 0] = 1 / 2.0
    T_N_inv_4d[:, :, :, 2, 1] = -serz_out_2d / er_out / 2.0
    T_N_inv_4d[:, :, :, 3, 2] = 1 / 2.0
    T_N_inv_4d[:, :, :, 3, 3] = -1 / serz_out_2d / 2.0

    return T_0_4d, T_N_inv_4d


def create_eps_matrix_AOIAz(a_2d, n_2d, Az_1d):
    """
    Create epsilon matrix used in simulation.

    Optimized: Fully vectorized without loops.

    Args:
        a_2d: azimuth angle of materials in each layer, shape (batchsize, n_layer, 3). Complex.
        n_2d: refractive index of each layer, shape (batchsize, n_layer, 3). Complex.
        Az_1d: azimuth angle of incident light, shape (batchsize, n_angles,). Real.

    Returns:
        eps_4d: epsilon tensor for all layers, shape (batchsize, n_angles, n_layers, 3, 3). Complex
    """
    device = a_2d.device
    num_layer = a_2d.size()[1]
    num_Az = Az_1d.size()[1]
    batchsize = Az_1d.shape[0]

    # Expand a_2d to (batchsize, num_Az, num_layer, 3)
    a_2d_exp = a_2d.unsqueeze(1).expand(-1, num_Az, -1, -1)

    # Expand Az_1d to (batchsize, num_Az, num_layer)
    Az_1d_exp = Az_1d.unsqueeze(-1).expand(-1, -1, num_layer)

    # Compute angles: (batchsize, num_Az, num_layer)
    phi_medium = a_2d_exp[:, :, :, 0] + Az_1d_exp + torch.pi / 2
    theta_medium = a_2d_exp[:, :, :, 1] + torch.pi / 2
    psi_medium = a_2d_exp[:, :, :, 2] + torch.pi / 2

    cos_theta = torch.cos(theta_medium)
    sin_theta = torch.sin(theta_medium)
    cos_phi = torch.cos(phi_medium)
    sin_phi = torch.sin(phi_medium)
    cos_psi = torch.cos(psi_medium)
    sin_psi = torch.sin(psi_medium)

    a2 = -sin_psi * sin_theta * cos_phi - cos_psi * sin_phi
    a3 = cos_theta * cos_phi
    b2 = -sin_psi * sin_theta * sin_phi + cos_psi * cos_phi
    b3 = cos_theta * sin_phi
    c2 = sin_psi * cos_theta
    c3 = sin_theta

    # Compute n^2 once: n_2d is (batchsize, num_layer, 3)
    nx2 = (
        n_2d[:, :, 0].real ** 2
        - n_2d[:, :, 0].imag ** 2
        + 2 * n_2d[:, :, 0].real * n_2d[:, :, 0].imag * 1j
    )
    ny2 = (
        n_2d[:, :, 1].real ** 2
        - n_2d[:, :, 1].imag ** 2
        + 2 * n_2d[:, :, 1].real * n_2d[:, :, 1].imag * 1j
    )
    nz2 = (
        n_2d[:, :, 2].real ** 2
        - n_2d[:, :, 2].imag ** 2
        + 2 * n_2d[:, :, 2].real * n_2d[:, :, 2].imag * 1j
    )

    # Expand to (batchsize, num_Az, num_layer)
    nx2 = nx2.unsqueeze(1).expand(-1, num_Az, -1)
    ny2 = ny2.unsqueeze(1).expand(-1, num_Az, -1)
    nz2 = nz2.unsqueeze(1).expand(-1, num_Az, -1)

    exx = nx2 + (ny2 - nx2) * a2**2 + (nz2 - nx2) * a3**2
    eyy = nx2 + (ny2 - nx2) * b2**2 + (nz2 - nx2) * b3**2
    ezz = nx2 + (ny2 - nx2) * c2**2 + (nz2 - nx2) * c3**2
    exy = (ny2 - nx2) * a2 * b2 + (nz2 - nx2) * a3 * b3
    exz = (ny2 - nx2) * a2 * c2 + (nz2 - nx2) * a3 * c3
    eyz = (ny2 - nx2) * c2 * b2 + (nz2 - nx2) * b3 * c3

    # Stack into eps_4d: (batchsize, num_Az, num_layer, 3, 3)
    eps_4d = torch.zeros(
        (batchsize, num_Az, num_layer, 3, 3), dtype=torch.complex64, device=device
    )
    eps_4d[:, :, :, 0, 0] = exx
    eps_4d[:, :, :, 0, 1] = exy
    eps_4d[:, :, :, 0, 2] = exz
    eps_4d[:, :, :, 1, 0] = exy  # eyx = exy
    eps_4d[:, :, :, 1, 1] = eyy
    eps_4d[:, :, :, 1, 2] = eyz
    eps_4d[:, :, :, 2, 0] = exz  # ezx = exz
    eps_4d[:, :, :, 2, 1] = eyz  # ezy = eyz
    eps_4d[:, :, :, 2, 2] = ezz

    return eps_4d


def create_eps_matrix_XY(a_2d, n_2d, Az_2d):
    """
    Create epsilon matrix used in simulation for XY configuration.

    Optimized: Fully vectorized without loops.

    Args:
        a_2d: azimuth angle of materials in each layer, shape (batchsize, n_layer, 3). Complex.
        n_2d: refractive index of each layer, shape (batchsize, n_layer, 3). Complex.
        Az_2d: azimuth, zenith angle of incident light, shape (batchsize, n_angles, n_zenith). Real.

    Returns:
        eps_5d: epsilon tensor for all layers, shape (batchsize, n_aoi, n_az, n_layers, 3, 3). Complex
    """
    device = a_2d.device
    batchsize = Az_2d.shape[0]
    num_layer = a_2d.size()[1]
    num_x = Az_2d.size()[1]
    num_y = Az_2d.size()[2]

    # Expand a_2d to (batchsize, num_x, num_y, num_layer, 3)
    # a_2d is (batchsize, num_layer, 3)
    a_2d_exp = a_2d.unsqueeze(1).unsqueeze(2).expand(-1, num_x, num_y, -1, -1)

    # Az_2d is (batchsize, num_x, num_y), expand to (batchsize, num_x, num_y, num_layer)
    Az_2d_exp = Az_2d.unsqueeze(-1).expand(-1, -1, -1, num_layer)

    # Compute phi_medium: (batchsize, num_x, num_y, num_layer)
    phi_medium = a_2d_exp[:, :, :, :, 0] + Az_2d_exp + torch.pi / 2
    theta_medium = a_2d_exp[:, :, :, :, 1] + torch.pi / 2
    psi_medium = a_2d_exp[:, :, :, :, 2] + torch.pi / 2

    cos_theta = torch.cos(theta_medium)
    sin_theta = torch.sin(theta_medium)
    cos_phi = torch.cos(phi_medium)
    sin_phi = torch.sin(phi_medium)
    cos_psi = torch.cos(psi_medium)
    sin_psi = torch.sin(psi_medium)

    a2 = -sin_psi * sin_theta * cos_phi - cos_psi * sin_phi
    a3 = cos_theta * cos_phi
    b2 = -sin_psi * sin_theta * sin_phi + cos_psi * cos_phi
    b3 = cos_theta * sin_phi
    c2 = sin_psi * cos_theta
    c3 = sin_theta

    # Compute n^2 once and expand: n_2d is (batchsize, num_layer, 3)
    # Result shape: (batchsize, 1, 1, num_layer)
    nx2 = (
        n_2d[:, :, 0].real ** 2
        - n_2d[:, :, 0].imag ** 2
        + 2 * n_2d[:, :, 0].real * n_2d[:, :, 0].imag * 1j
    )
    ny2 = (
        n_2d[:, :, 1].real ** 2
        - n_2d[:, :, 1].imag ** 2
        + 2 * n_2d[:, :, 1].real * n_2d[:, :, 1].imag * 1j
    )
    nz2 = (
        n_2d[:, :, 2].real ** 2
        - n_2d[:, :, 2].imag ** 2
        + 2 * n_2d[:, :, 2].real * n_2d[:, :, 2].imag * 1j
    )

    # Expand to (batchsize, num_x, num_y, num_layer)
    nx2 = nx2.unsqueeze(1).unsqueeze(2).expand(-1, num_x, num_y, -1)
    ny2 = ny2.unsqueeze(1).unsqueeze(2).expand(-1, num_x, num_y, -1)
    nz2 = nz2.unsqueeze(1).unsqueeze(2).expand(-1, num_x, num_y, -1)

    exx = nx2 + (ny2 - nx2) * a2**2 + (nz2 - nx2) * a3**2
    eyy = nx2 + (ny2 - nx2) * b2**2 + (nz2 - nx2) * b3**2
    ezz = nx2 + (ny2 - nx2) * c2**2 + (nz2 - nx2) * c3**2
    exy = (ny2 - nx2) * a2 * b2 + (nz2 - nx2) * a3 * b3
    exz = (ny2 - nx2) * a2 * c2 + (nz2 - nx2) * a3 * c3
    eyz = (ny2 - nx2) * c2 * b2 + (nz2 - nx2) * b3 * c3

    # Stack into eps_5d: (batchsize, num_x, num_y, num_layer, 3, 3)
    eps_5d = torch.zeros(
        (batchsize, num_x, num_y, num_layer, 3, 3), dtype=torch.complex64, device=device
    )
    eps_5d[:, :, :, :, 0, 0] = exx
    eps_5d[:, :, :, :, 0, 1] = exy
    eps_5d[:, :, :, :, 0, 2] = exz
    eps_5d[:, :, :, :, 1, 0] = exy  # eyx = exy
    eps_5d[:, :, :, :, 1, 1] = eyy
    eps_5d[:, :, :, :, 1, 2] = eyz
    eps_5d[:, :, :, :, 2, 0] = exz  # ezx = exz
    eps_5d[:, :, :, :, 2, 1] = eyz  # ezy = eyz
    eps_5d[:, :, :, :, 2, 2] = ezz

    return eps_5d


def create_jones_matrix_AOIAz(
    a_2d, n_2d, d_1d, wv_1d, n_in, n_out, theta_x_1d, theta_y_1d
):
    """
    Calculate the Jones matrix for reflected and transmitted light.

    Optimized: Vectorized AOI/Az setup, batched eigenvalue decomposition,
    and using torch.linalg.solve instead of torch.inverse.

    Args:
        a_2d: azimuth angle of materials in each layer, shape (batchsize, n_layer, 3). Complex.
        n_2d: refractive index of each layer, shape (batchsize, n_layer, 3). Complex.
        d_1d: thicknesses of all layers, shape (batchsize, n_layer). Complex.
        wv_1d: wavelengths of simulations, shape (batchsize, n_wls). Real
        n_in: incident media refractive index
        n_out: transmit media refractive index
        theta_x_1d: incident Zenith angle, shape (batchsize, n_aoi_angles). Real
        theta_y_1d: azimuth angle of incident light, shape (batchsize, n_az_angles). Real.

    Returns:
        Jones_trn, Jones_ref: Jones matrices, each with shape (batchsize, n_wls, n_aoi_angles, n_az_angles, 2, 2). Complex
    """
    device = a_2d.device

    batchsize = d_1d.shape[0]
    num_wv = wv_1d.size()[1]
    num_x = theta_x_1d.size()[1]
    num_y = theta_y_1d.size()[1]
    num_layer = d_1d.size()[1]

    # Vectorized AOI and Az calculation (no loops)
    # theta_x_1d: (batchsize, num_x), theta_y_1d: (batchsize, num_y)
    # AOI_2d: (batchsize, num_x, num_y) - broadcast theta_x over y dimension
    AOI_2d = theta_x_1d.unsqueeze(-1).expand(-1, -1, num_y).to(torch.complex64)
    # Az_2d: (batchsize, num_x, num_y) - broadcast theta_y over x dimension
    Az_2d = theta_y_1d.unsqueeze(1).expand(-1, num_x, -1).to(torch.float64)

    k0_1d = 2 * torch.pi / wv_1d
    ng_1d = torch.sqrt(
        (n_2d[:, :, 0] ** 2 + n_2d[:, :, 1] ** 2 + n_2d[:, :, 2] ** 2) / 3
    )
    eps_in = n_in**2
    eps_out = n_out**2

    theta_inc_air_2d = AOI_2d
    # Use complex_arcsin to properly handle evanescent waves beyond critical angle
    theta_inc_sub_2d = complex_arcsin(n_in * torch.sin(AOI_2d) / n_out)

    ng_3d = ng_1d.reshape((batchsize, 1, 1, -1)).expand(-1, num_x, num_y, -1)
    AOI_3d = AOI_2d.unsqueeze(-1).expand(-1, -1, -1, num_layer)

    # Use complex_arcsin for angles in each layer - critical for TIR handling
    theta_inc_medium_3d = complex_arcsin(n_in * torch.sin(AOI_3d) / ng_3d)
    sin_Vt_3d = ng_3d * torch.sin(theta_inc_medium_3d)

    eps_5d = create_eps_matrix_XY(a_2d, n_2d, Az_2d)

    # Extract epsilon components and expand for wavelengths
    exx_4d = eps_5d[:, :, :, :, 0, 0].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)
    exy_4d = eps_5d[:, :, :, :, 0, 1].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)
    exz_4d = eps_5d[:, :, :, :, 0, 2].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)
    eyx_4d = eps_5d[:, :, :, :, 1, 0].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)
    eyy_4d = eps_5d[:, :, :, :, 1, 1].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)
    eyz_4d = eps_5d[:, :, :, :, 1, 2].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)
    ezx_4d = eps_5d[:, :, :, :, 2, 0].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)
    ezy_4d = eps_5d[:, :, :, :, 2, 1].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)
    ezz_4d = eps_5d[:, :, :, :, 2, 2].unsqueeze(1).expand(-1, num_wv, -1, -1, -1)

    sin_Vt_4d = sin_Vt_3d.unsqueeze(1).expand(-1, num_wv, -1, -1, -1)

    # Build Q matrix for all layers at once
    Q_6d = torch.zeros(
        (batchsize, num_wv, num_x, num_y, num_layer, 4, 4),
        dtype=torch.complex64,
        device=device,
    )
    Q_6d[:, :, :, :, :, 0, 0] = -ezx_4d * sin_Vt_4d / ezz_4d
    Q_6d[:, :, :, :, :, 0, 1] = 1 - sin_Vt_4d**2 / ezz_4d
    Q_6d[:, :, :, :, :, 0, 2] = -ezy_4d * sin_Vt_4d / ezz_4d
    Q_6d[:, :, :, :, :, 1, 0] = exx_4d - exz_4d * ezx_4d / ezz_4d
    Q_6d[:, :, :, :, :, 1, 1] = -exz_4d * sin_Vt_4d / ezz_4d
    Q_6d[:, :, :, :, :, 1, 2] = exy_4d - exz_4d * ezy_4d / ezz_4d
    Q_6d[:, :, :, :, :, 2, 3] = 1.0
    Q_6d[:, :, :, :, :, 3, 0] = eyx_4d - eyz_4d * ezx_4d / ezz_4d
    Q_6d[:, :, :, :, :, 3, 1] = -eyz_4d * sin_Vt_4d / ezz_4d
    Q_6d[:, :, :, :, :, 3, 2] = eyy_4d - eyz_4d * ezy_4d / ezz_4d - sin_Vt_4d**2

    # Batch eigenvalue decomposition for ALL layers at once
    # Q_6d shape: (batchsize, num_wv, num_x, num_y, num_layer, 4, 4)
    # Reshape to (batchsize * num_wv * num_x * num_y * num_layer, 4, 4)
    Q_flat = Q_6d.view(-1, 4, 4)
    eigenvalues_flat, eigenvectors_flat = torch.linalg.eig(Q_flat)

    # Ensure consistent dtype (complex64) for all tensors
    dtype = torch.complex64
    eigenvalues_flat = eigenvalues_flat.to(dtype)
    eigenvectors_flat = eigenvectors_flat.to(dtype)

    # Reshape back: (batchsize, num_wv, num_x, num_y, num_layer, 4)
    eigenvalues_all = eigenvalues_flat.view(
        batchsize, num_wv, num_x, num_y, num_layer, 4
    )
    # (batchsize, num_wv, num_x, num_y, num_layer, 4, 4)
    eigenvectors_all = eigenvectors_flat.view(
        batchsize, num_wv, num_x, num_y, num_layer, 4, 4
    )

    # Compute k0 * d for all layers: (batchsize, num_wv, 1, 1, num_layer)
    k0_1d_exp = k0_1d.reshape(batchsize, num_wv, 1, 1, 1).to(dtype)
    d_1d_exp = d_1d.reshape(batchsize, 1, 1, 1, num_layer).to(dtype)
    k0d = k0_1d_exp * d_1d_exp  # (batchsize, num_wv, 1, 1, num_layer)

    # Compute phase factors for all layers: exp(1j * k0 * d * eigenvalue)
    # eigenvalues_all: (batchsize, num_wv, num_x, num_y, num_layer, 4)
    phase_factors = torch.exp(1j * k0d.unsqueeze(-1) * eigenvalues_all)
    # phase_factors: (batchsize, num_wv, num_x, num_y, num_layer, 4)

    # Build diagonal U matrices efficiently using diag_embed
    # U shape: (batchsize, num_wv, num_x, num_y, num_layer, 4, 4)
    U_all = torch.diag_embed(phase_factors)

    # Compute Pn = V @ U @ V^(-1) for all layers
    # For 4x4 matrices, direct inverse is efficient
    V_flat = eigenvectors_all.reshape(-1, 4, 4)
    V_inv_flat = torch.linalg.inv(V_flat)
    V_inv_all = V_inv_flat.view(batchsize, num_wv, num_x, num_y, num_layer, 4, 4)

    # Pn = V @ U @ V_inv for each layer (batched matmul)
    Pn_all = torch.matmul(torch.matmul(eigenvectors_all, U_all), V_inv_all)

    # Sequential multiplication of layer transfer matrices P = Pn[n-1] @ ... @ Pn[1] @ Pn[0]
    # Start with first layer's transfer matrix
    P_5d = Pn_all[:, :, :, :, 0, :, :].clone()

    for i_layer in range(1, num_layer):
        P_5d = torch.matmul(Pn_all[:, :, :, :, i_layer, :, :], P_5d)

    T0_4d, T_N_inv_4d = EnterExitMatrix_XY(
        eps_in, eps_out, theta_inc_air_2d, theta_inc_sub_2d
    )

    T_N_inv_5d = T_N_inv_4d.unsqueeze(1).expand(-1, num_wv, -1, -1, -1, -1)
    T0_5d = T0_4d.unsqueeze(1).expand(-1, num_wv, -1, -1, -1, -1)
    N_5d = torch.matmul(torch.matmul(T_N_inv_5d, P_5d), T0_5d)

    N11_5d = N_5d[:, :, :, :, :2, :2]
    N12_5d = N_5d[:, :, :, :, :2, 2:]
    N21_5d = N_5d[:, :, :, :, 2:, :2]
    N22_5d = N_5d[:, :, :, :, 2:, 2:]

    # Reshape to scattering matrix S
    # For 2x2 matrices, direct inverse is efficient
    S12_5d = torch.linalg.inv(N22_5d)
    S11_5d = torch.matmul(-S12_5d, N21_5d)
    S21_5d = N11_5d + torch.matmul(N12_5d, S11_5d)

    Jones_trans = S21_5d
    Jones_rflc = S11_5d

    return Jones_trans, Jones_rflc


# ===========================================
# Film Solver Class
# ===========================================
class FilmSolver:
    """Multi-layer coating physical film solver using transfer matrix method.

    This solver calculates (ts, tp, rs, rp) with phase shifts using
    rigorous electromagnetic wave propagation through multi-layer coating stacks.
    """

    def __init__(
        self,
        mat_n_in,
        mat_n_out,
        mat_n_ls,
        thickness_ls=None,
        thickness_min=0.0,
        thickness_max=0.2,
        batch_size=1,
        sigmoid_param=False,
        device=torch.device("cuda"),
    ):
        """
        Initialize the anisotropic film solver.

        Args:
            mat_n_in: Refractive index of incident medium (scalar).
            mat_n_out: Refractive index of exit medium (scalar).
            mat_n_ls: Refractive indices of interior layers.
                      For isotropic: list/tensor of length N (scalar per layer).
                      For anisotropic: tensor of shape (N, 3) with [nx, ny, nz] per layer.
            thickness_ls: Thicknesses of interior layers in um, list or tensor of length N.
                          If None, randomly initializes thicknesses.
            thickness_min: Minimum layer thickness in um.
            thickness_max: Maximum layer thickness in um.
            batch_size: Number of film stacks in the batch dimension.
            sigmoid_param: If True, use sigmoid parameterization for thickness.
            device: PyTorch device.
        """
        # Parameters
        self.batch_size = batch_size
        self.mat_n_in = float(mat_n_in)
        self.mat_n_out = float(mat_n_out)
        self.device = device

        # Set up refractive indices for each layer (N, 3) for anisotropic
        if torch.is_tensor(mat_n_ls):
            n_layers_t = mat_n_ls.float()
        else:
            n_layers_t = torch.tensor(mat_n_ls, dtype=torch.float32)
        if n_layers_t.dim() == 1:
            # Isotropic: expand scalar per layer to (N, 3)
            n_layers_t = n_layers_t.unsqueeze(-1).expand(-1, 3)
        self.num_layers = n_layers_t.shape[0]
        self.refract_idx_layers = n_layers_t.unsqueeze(0).expand(self.batch_size, -1, -1).clone()

        # Min and max single layer film thickness in [um]
        self.thickness_min = thickness_min
        self.thickness_max = thickness_max
        self._thickness_range = self.thickness_max - self.thickness_min

        # Initialize film_params in normalized [0, 1] space
        self.sigmoid_param = sigmoid_param
        if thickness_ls is not None:
            if not torch.is_tensor(thickness_ls):
                thickness_ls = torch.tensor(thickness_ls, dtype=torch.float32)
            normalized = (thickness_ls.clamp(self.thickness_min, self.thickness_max) - self.thickness_min) / self._thickness_range
            self.film_params = normalized.unsqueeze(0).expand(self.batch_size, -1).clone()
        else:
            self.film_params = torch.randn(self.batch_size, self.num_layers) * 0.01 + 0.5

        # Convert to sigmoid (logit) parameterization if requested
        if self.sigmoid_param:
            self.film_params = inv_sigmoid(self.film_params.clamp(1e-6, 1 - 1e-6))

        # Move to device
        self.to(device)

    def to(self, device):
        """Move tensors to specified device."""
        self.device = device
        self.film_params = self.film_params.to(device)
        self.refract_idx_layers = self.refract_idx_layers.to(device)
        return self

    def load_ckpt(self, ckpt_path):
        """Load checkpoint from file path."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        # Load film thickness
        film_thickness = torch.clamp(ckpt["film_thickness"], self.thickness_min, self.thickness_max)
        film_thickness_normalized = (film_thickness - self.thickness_min) / (
            self.thickness_max - self.thickness_min
        )

        # Convert to optimizable parameters
        if self.sigmoid_param:
            film_thickness_normalized = torch.clamp(
                film_thickness_normalized, 1e-6, 1 - 1e-6
            )
            self.film_params = inv_sigmoid(film_thickness_normalized).to(self.device)
        else:
            self.film_params = film_thickness_normalized.to(self.device)

    def save_ckpt(self, save_path):
        """Save checkpoint."""
        torch.save(
            {
                "film_thickness": self.get_film_thickness().cpu(),
                "batch_size": self.batch_size,
                "num_layers": self.num_layers,
                "n_in": self.mat_n_in,
                "n_out": self.mat_n_out,
                "refract_idx_layers": self.refract_idx_layers.cpu(),
            },
            save_path,
        )

    # ===========================================
    # Film simulation
    # ===========================================

    def get_film_thickness(self):
        """Convert optimization-friendly film parameters to real film thickness.

        Returns:
            film_thickness: tensor of shape (batch_size, num_layers), in [um].
        """
        if self.sigmoid_param:
            film_thickness = (
                torch.sigmoid(self.film_params) * self._thickness_range + self.thickness_min
            )
        else:
            film_thickness = self.film_params * self._thickness_range + self.thickness_min
            film_thickness = film_thickness.clamp(self.thickness_min, self.thickness_max)

        return film_thickness

    def simulate(self, theta, wvln):
        """
        Calculate (ts, tp, rs, rp) using 4x4 TMM for specified angles and wavelengths.

        Args:
            theta: Incident angles in radians. Can be:
                   - 1D tensor of shape (n_angles,): same angles for all mirrors
                   - 2D tensor of shape (batch_size, n_angles): different angles per film stack
            wvln: Wavelengths in micrometers. Can be:
                  - List or 1D tensor of shape (n_wvlns,)
                  - Scalar or 0D tensor: single wavelength

        Returns:
            ts, tp, rs, rp: Complex transmission/reflection coefficients.
                           Shape: (batch_size, n_wvlns, n_angles)
        """
        # Handle theta input
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        theta = theta.to(self.device)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand(self.batch_size, -1)

        # Handle wavelength input
        if torch.is_tensor(wvln):
            wv = wvln.to(self.device)
            if wv.dim() == 0:
                wv = wv.unsqueeze(0)
        elif isinstance(wvln, (list, tuple)):
            wv = torch.tensor(wvln, dtype=torch.float32, device=self.device)
        else:
            wv = torch.tensor([wvln], dtype=torch.float32, device=self.device)

        d_1d = self.get_film_thickness()
        wv_1d = wv.unsqueeze(0).expand(self.batch_size, -1)
        n_wvlns = wv.shape[0]
        n_angles = theta.shape[1]

        # 4x4 anisotropic solver (handles both isotropic and anisotropic materials)
        a_2d = (
            torch.zeros((self.batch_size, self.num_layers, 3))
            .to(torch.complex64)
            .to(self.device)
        )
        n_2d = self.refract_idx_layers.to(torch.complex64)
        d_1d_complex = d_1d.to(torch.complex64)

        Az_1d = torch.zeros((self.batch_size, 1)).to(self.device)

        Jt, Jr = create_jones_matrix_AOIAz(
            a_2d, n_2d, d_1d_complex, wv_1d, self.mat_n_in, self.mat_n_out, theta, Az_1d
        )

        # Set input polarization status
        p_in_lab = torch.tensor([[1.0 + 0.0j], [0.0 + 0.0j]], dtype=torch.complex64).to(
            self.device
        )
        s_in_lab = torch.tensor([[0.0 + 0.0j], [1.0 + 0.0j]], dtype=torch.complex64).to(
            self.device
        )
        p_in_5d = p_in_lab.reshape((1, 1, 1, 1, 2, 1)).expand(
            self.batch_size, n_wvlns, n_angles, 1, -1, -1
        )
        s_in_5d = s_in_lab.reshape((1, 1, 1, 1, 2, 1)).expand(
            self.batch_size, n_wvlns, n_angles, 1, -1, -1
        )

        # T and R at the first film
        t1_vec_p_5d = torch.matmul(Jt, p_in_5d)
        r1_vec_p_5d = torch.matmul(Jr, p_in_5d)
        t1_vec_s_5d = torch.matmul(Jt, s_in_5d)
        r1_vec_s_5d = torch.matmul(Jr, s_in_5d)

        tp = t1_vec_p_5d[:, :, :, :, 0, 0].squeeze(-1)
        ts = t1_vec_s_5d[:, :, :, :, 1, 0].squeeze(-1)
        rp = r1_vec_p_5d[:, :, :, :, 0, 0].squeeze(-1)
        rs = r1_vec_s_5d[:, :, :, :, 1, 0].squeeze(-1)

        return ts, tp, rs, rp

    def __call__(self, theta, wvln):
        """Forward pass using simulate."""
        return self.simulate(theta, wvln)
