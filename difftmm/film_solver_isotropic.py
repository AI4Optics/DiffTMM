"""Isotropic Multi-layer Thin Film Solver (2x2 Transfer Matrix Method)

Differentiable thin film solver for isotropic materials using the standard
2x2 transfer matrix method. Computes Fresnel coefficients (ts, tp, rs, rp)
for multi-layer film stacks with full autograd support.

Copyright (c) 2026, Xinge Yang, Qingyuan Fan, Zhaocheng Liu.
"""

from typing import Dict, List, Optional, Sequence

import torch


# =========================
# Utility functions
# =========================
def inv_sigmoid(x):
    """Inverse sigmoid function."""
    return torch.log(x / (1 - x))


@torch.jit.script
def _batch_2x2_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized batched 2x2 matrix multiplication.
    
    For 2x2 matrices, explicit formula is faster than torch.matmul.
    A, B: shape (..., 2, 2)
    Returns: A @ B with shape (..., 2, 2)
    """
    # Extract elements
    a00, a01 = A[..., 0, 0], A[..., 0, 1]
    a10, a11 = A[..., 1, 0], A[..., 1, 1]
    b00, b01 = B[..., 0, 0], B[..., 0, 1]
    b10, b11 = B[..., 1, 0], B[..., 1, 1]
    
    # Compute product elements
    c00 = a00 * b00 + a01 * b10
    c01 = a00 * b01 + a01 * b11
    c10 = a10 * b00 + a11 * b10
    c11 = a10 * b01 + a11 * b11
    
    # Stack result
    return torch.stack([
        torch.stack([c00, c01], dim=-1),
        torch.stack([c10, c11], dim=-1)
    ], dim=-2)


# =========================
# Isotropic TMM functions
# =========================
def _compute_isotropic_tmm(
    n_layers,
    d,
    wv,
    n_in_t,
    n_out_t,
    theta,
    batchsize,
    num_wv,
    num_angles,
    num_layer,
    device,
    dtype,
):
    """
    Core TMM computation for one propagation direction.

    Optimized version:
    - Pre-allocates epsilon tensor once
    - Builds all layer matrices in parallel
    - Uses optimized 2x2 matrix multiplication
    - Minimizes memory allocations in the loop

    Args:
        n_layers: refractive indices, shape (batch, 1, 1, layer)
        d: thicknesses, shape (batch, 1, 1, layer)
        wv: wavelengths, shape (batch, wv, 1, 1)
        n_in_t, n_out_t: incident/output refractive indices (scalar tensors)
        theta: angles, shape (batch, 1, angles, 1)
        Other args: dimension info

    Returns:
        ts, tp, rs, rp for this direction
    """
    eps = 1e-10
    # Pre-allocate epsilon tensor once (avoid repeated tensor creation)
    eps_tensor = torch.tensor(eps, dtype=dtype, device=device)

    # Compute sin(theta) in incident medium - this is conserved (Snell's law)
    sin_theta_in = torch.sin(theta)  # (batch, 1, angles, 1)

    # sin(theta) is conserved: n_in * sin(theta_in) = n_layer * sin(theta_layer)
    sin_theta_layers = n_in_t * sin_theta_in / n_layers  # (batch, 1, angles, layer)

    # Compute cos(theta) in each layer using cos = sqrt(1 - sin^2)
    # For evanescent waves, this will be purely imaginary
    cos_theta_layers = torch.sqrt(1 - sin_theta_layers**2)
    cos_theta_layers = torch.where(
        cos_theta_layers.abs() < eps, eps_tensor, cos_theta_layers
    )

    # cos(theta) in incident and output media
    cos_theta_in = torch.cos(theta)  # (batch, 1, angles, 1)
    cos_theta_in = torch.where(cos_theta_in.abs() < eps, eps_tensor, cos_theta_in)
    sin_theta_out = n_in_t * sin_theta_in / n_out_t
    cos_theta_out = torch.sqrt(1 - sin_theta_out**2)
    cos_theta_out = torch.where(cos_theta_out.abs() < eps, eps_tensor, cos_theta_out)

    # Phase shift in each layer: delta = 2*pi*n*d*cos(theta)/lambda
    k0 = 2 * torch.pi / wv  # (batch, wv, 1, 1)
    delta = k0 * n_layers * d * cos_theta_layers  # (batch, wv, angles, layer)

    # Compute effective indices for s and p polarizations
    n_eff_s = n_layers * cos_theta_layers  # (batch, wv, angles, layer)
    n_eff_p = n_layers / cos_theta_layers  # (batch, wv, angles, layer)

    # Input/output effective indices
    n_in_s = n_in_t * cos_theta_in  # (batch, 1, angles, 1)
    n_in_p = n_in_t / cos_theta_in
    n_out_s = n_out_t * cos_theta_out
    n_out_p = n_out_t / cos_theta_out

    # Precompute cos and sin of delta for all layers at once
    cos_delta = torch.cos(delta)  # (batch, wv, angles, layer)
    sin_delta = torch.sin(delta)

    # Build ALL layer matrices at once (vectorized over layers)
    # Shape: (batch, wv, angles, layer, 2, 2)
    M_all_s = torch.zeros(
        (batchsize, num_wv, num_angles, num_layer, 2, 2), dtype=dtype, device=device
    )
    M_all_p = torch.zeros(
        (batchsize, num_wv, num_angles, num_layer, 2, 2), dtype=dtype, device=device
    )

    # Fill all matrices at once (no loop for matrix construction)
    M_all_s[..., 0, 0] = cos_delta
    M_all_s[..., 0, 1] = 1j * sin_delta / n_eff_s
    M_all_s[..., 1, 0] = 1j * n_eff_s * sin_delta
    M_all_s[..., 1, 1] = cos_delta

    M_all_p[..., 0, 0] = cos_delta
    M_all_p[..., 0, 1] = 1j * sin_delta / n_eff_p
    M_all_p[..., 1, 0] = 1j * n_eff_p * sin_delta
    M_all_p[..., 1, 1] = cos_delta

    # Sequential matrix multiplication (still needed due to non-commutativity)
    # But now we use the optimized 2x2 matmul and avoid allocations
    M_s = M_all_s[..., 0, :, :]  # Start with first layer
    M_p = M_all_p[..., 0, :, :]

    for i_layer in range(1, num_layer):
        M_s = _batch_2x2_matmul(M_all_s[..., i_layer, :, :], M_s)
        M_p = _batch_2x2_matmul(M_all_p[..., i_layer, :, :], M_p)

    # Extract matrix elements (contiguous access pattern)
    M_s_00 = M_s[..., 0, 0]
    M_s_01 = M_s[..., 0, 1]
    M_s_10 = M_s[..., 1, 0]
    M_s_11 = M_s[..., 1, 1]
    M_p_00 = M_p[..., 0, 0]
    M_p_01 = M_p[..., 0, 1]
    M_p_10 = M_p[..., 1, 0]
    M_p_11 = M_p[..., 1, 1]

    # Squeeze input/output indices
    n_in_s = n_in_s.squeeze(-1)
    n_in_p = n_in_p.squeeze(-1)
    n_out_s = n_out_s.squeeze(-1)
    n_out_p = n_out_p.squeeze(-1)

    # Compute reflection and transmission coefficients
    # Pre-compute common terms to avoid redundant computation
    n_in_s_M00 = n_in_s * M_s_00
    n_in_s_n_out_s_M01 = n_in_s * n_out_s * M_s_01
    n_out_s_M11 = n_out_s * M_s_11

    denom_s = n_in_s_M00 + n_in_s_n_out_s_M01 + M_s_10 + n_out_s_M11
    rs = (n_in_s_M00 + n_in_s_n_out_s_M01 - M_s_10 - n_out_s_M11) / denom_s
    ts = 2 * n_in_s / denom_s

    n_in_p_M00 = n_in_p * M_p_00
    n_in_p_n_out_p_M01 = n_in_p * n_out_p * M_p_01
    n_out_p_M11 = n_out_p * M_p_11

    denom_p = n_in_p_M00 + n_in_p_n_out_p_M01 + M_p_10 + n_out_p_M11
    rp = (n_in_p_M00 + n_in_p_n_out_p_M01 - M_p_10 - n_out_p_M11) / denom_p
    tp = 2 * n_in_p / denom_p

    return ts, tp, rs, rp


def create_jones_matrix_isotropic(n_layers_1d, d_1d, wv_1d, n_in, n_out, theta_1d):
    """
    Fast Jones matrix calculation for isotropic multi-layer films.

    Uses the standard 2x2 transfer matrix method which is much faster
    than the general 4x4 anisotropic formulation when materials are isotropic.
    Avoids eigenvalue decomposition entirely.

    Supports bidirectional propagation:
    - theta in [0, pi/2]: Forward direction (top to bottom, n_in -> layers -> n_out)
    - theta in [pi/2, pi]: Reverse direction (bottom to top, n_out -> layers -> n_in)
      Internally converts to equivalent forward problem with swapped media and reversed layers.

    Args:
        n_layers_1d: refractive index of each layer, shape (batchsize, n_layer). Complex.
        d_1d: thicknesses of all layers, shape (batchsize, n_layer). Real or Complex.
        wv_1d: wavelengths of simulations, shape (batchsize, n_wls). Real.
        n_in: incident media refractive index (top medium). Scalar.
        n_out: transmit media refractive index (bottom medium). Scalar.
        theta_1d: incident angles, shape (batchsize, n_angles). Real.
                  Range [0, pi]. Angles > pi/2 represent reverse propagation.

    Returns:
        ts, tp, rs, rp: complex transmission/reflection coefficients
                        each with shape (batchsize, n_wls, n_angles)
    """
    device = n_layers_1d.device
    dtype = torch.complex64

    batchsize = d_1d.shape[0]
    num_wv = wv_1d.shape[1]
    num_angles = theta_1d.shape[1]
    num_layer = d_1d.shape[1]

    n_in_t = torch.tensor(n_in, dtype=dtype, device=device)
    n_out_t = torch.tensor(n_out, dtype=dtype, device=device)

    # Identify forward (theta <= pi/2) and reverse (theta > pi/2) angles
    pi_half = torch.pi / 2
    is_reverse = theta_1d > pi_half  # (batch, angles)

    # Fast path: if n_in == n_out (symmetric media), we can use a simpler approach
    # For symmetric media with symmetric layer stack, |r(theta)| = |r(pi-theta)|
    # We compute forward angles only and map results for reverse angles
    if abs(n_in - n_out) < 1e-10:
        # Map all angles to [0, pi/2] range
        theta_mapped = torch.where(is_reverse, torch.pi - theta_1d, theta_1d)

        # Expand dimensions (combined for fewer operations)
        n_layers = n_layers_1d.unsqueeze(1).unsqueeze(2).to(dtype)
        d = d_1d.unsqueeze(1).unsqueeze(2).to(dtype)
        wv = wv_1d.unsqueeze(2).unsqueeze(3).to(dtype)
        theta = theta_mapped.unsqueeze(1).unsqueeze(3).to(dtype)

        # Single forward computation with mapped angles
        ts, tp, rs, rp = _compute_isotropic_tmm(
            n_layers,
            d,
            wv,
            n_in_t,
            n_out_t,
            theta,
            batchsize,
            num_wv,
            num_angles,
            num_layer,
            device,
            dtype,
        )

        return ts, tp, rs, rp

    # Check if we have any reverse angles
    has_forward = (~is_reverse).any()
    has_reverse = is_reverse.any()

    # Pre-expand common tensors (avoid redundant expansion)
    wv = wv_1d.unsqueeze(2).unsqueeze(3).to(dtype)

    # Prepare output tensors
    ts_out = torch.zeros((batchsize, num_wv, num_angles), dtype=dtype, device=device)
    tp_out = torch.zeros((batchsize, num_wv, num_angles), dtype=dtype, device=device)
    rs_out = torch.zeros((batchsize, num_wv, num_angles), dtype=dtype, device=device)
    rp_out = torch.zeros((batchsize, num_wv, num_angles), dtype=dtype, device=device)

    # Process forward angles (theta <= pi/2)
    if has_forward:
        # Get forward angle indices
        forward_mask = ~is_reverse  # (batch, angles)

        # For simplicity, process all angles but only use results for forward ones
        # Expand dimensions for broadcasting
        n_layers = n_layers_1d.unsqueeze(1).unsqueeze(2).to(dtype)  # (batch, 1, 1, layer)
        d = d_1d.unsqueeze(1).unsqueeze(2).to(dtype)
        theta = theta_1d.unsqueeze(1).unsqueeze(3).to(dtype)  # (batch, 1, angles, 1)

        ts_fwd, tp_fwd, rs_fwd, rp_fwd = _compute_isotropic_tmm(
            n_layers,
            d,
            wv,
            n_in_t,
            n_out_t,
            theta,
            batchsize,
            num_wv,
            num_angles,
            num_layer,
            device,
            dtype,
        )

        # Copy forward results
        forward_mask_exp = forward_mask.unsqueeze(1).expand(-1, num_wv, -1)
        ts_out = torch.where(forward_mask_exp, ts_fwd, ts_out)
        tp_out = torch.where(forward_mask_exp, tp_fwd, tp_out)
        rs_out = torch.where(forward_mask_exp, rs_fwd, rs_out)
        rp_out = torch.where(forward_mask_exp, rp_fwd, rp_out)

    # Process reverse angles (theta > pi/2)
    if has_reverse:
        # For reverse direction:
        # 1. Use supplementary angle: theta_rev = pi - theta
        # 2. Swap incident and output media
        # 3. Reverse layer order

        # Supplementary angle
        theta_rev = torch.pi - theta_1d  # (batch, angles)

        # Reverse layer order
        n_layers_rev = torch.flip(n_layers_1d, dims=[1])
        d_rev = torch.flip(d_1d, dims=[1])

        # Expand dimensions
        n_layers_rev_exp = n_layers_rev.unsqueeze(1).unsqueeze(2).to(dtype)
        d_rev_exp = d_rev.unsqueeze(1).unsqueeze(2).to(dtype)
        theta_rev_exp = theta_rev.unsqueeze(1).unsqueeze(3).to(dtype)

        # Compute with swapped media (n_out -> n_in)
        ts_rev, tp_rev, rs_rev, rp_rev = _compute_isotropic_tmm(
            n_layers_rev_exp,
            d_rev_exp,
            wv,
            n_out_t,
            n_in_t,
            theta_rev_exp,
            batchsize,
            num_wv,
            num_angles,
            num_layer,
            device,
            dtype,
        )

        # Copy reverse results
        reverse_mask = is_reverse  # (batch, angles)
        reverse_mask_exp = reverse_mask.unsqueeze(1).expand(-1, num_wv, -1)
        ts_out = torch.where(reverse_mask_exp, ts_rev, ts_out)
        tp_out = torch.where(reverse_mask_exp, tp_rev, tp_out)
        rs_out = torch.where(reverse_mask_exp, rs_rev, rs_out)
        rp_out = torch.where(reverse_mask_exp, rp_rev, rp_out)

    return ts_out, tp_out, rs_out, rp_out


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

    Layer convention: layers are ordered from the incident medium (n_in) to
    the exit medium (n_out), i.e. the same physical order used by
    ``tmm_numpy.coh_tmm``.  Internally, layers are reversed before being
    passed to ``create_jones_matrix_isotropic``, which stores them in the
    reverse physical order due to its characteristic-matrix accumulation
    convention.

    Args:
        n_layers_1d: refractive index of each interior layer in physical order
            (n_in-side first), shape (batch, n_layer). Complex.
        d_1d: thickness of each interior layer in um, same physical ordering
            as n_layers_1d, shape (batch, n_layer). Real.
        wv_1d: wavelengths in um, shape (batch, n_wls). Real.
        n_in: scalar incident refractive index (top medium).
        n_out: scalar exit refractive index (bottom medium).
        theta_1d: incident angles in radians, shape (batch, n_angles). Real, in [0, pi/2].

    Returns:
        Rs, Rp, Ts, Tp: real tensors, each shape (batch, n_wls, n_angles), in [0, 1].

    Note:
        This helper is public-API: it is exposed via ``difftmm.__init__`` in
        Task 9. Other modules in this package use it as a building block for
        the incoherent TMM solver, but external users may also call it
        directly when they need power coefficients without phase.
    """
    # Layer convention: callers pass layers in physical order (n_in-side first),
    # matching the convention used by tmm_numpy.coh_tmm. However,
    # create_jones_matrix_isotropic empirically interprets its n_layers_1d
    # argument as ordered from the n_out-side first -- verified by direct
    # comparison with tmm_numpy.coh_tmm in test_coh_stack_power_RT_matches_tmm_numpy.
    # We therefore flip along the layer axis before delegating. If the upstream
    # convention is ever clarified or changed, update this flip and the test
    # will tell us immediately.
    n_layers_rev = torch.flip(n_layers_1d, dims=[1])
    d_rev = torch.flip(d_1d, dims=[1])

    ts, tp, rs, rp = create_jones_matrix_isotropic(
        n_layers_rev, d_rev, wv_1d, n_in, n_out, theta_1d
    )

    # Reflectance is |r|^2 for both polarizations.
    Rs = (rs.real ** 2 + rs.imag ** 2)
    Rp = (rp.real ** 2 + rp.imag ** 2)

    # Transmittance: |t|^2 must be scaled by an intensity-correction factor that
    # depends on how the amplitudes ts/tp are normalized inside
    # _compute_isotropic_tmm. That solver builds the characteristic matrix using
    # two *different* effective indices:
    #     n_eff_s = n * cos(theta)   for s-pol
    #     n_eff_p = n / cos(theta)   for p-pol
    # So ts/tp are normalized to those effective indices, and the power-correction
    # is the ratio of effective indices -- NOT the |E|^2-based ratio
    #     (n_out * conj(cos)) / (n_in * conj(cos))
    # used by tmm_numpy.T_from_t. Concretely:
    #     s-pol: T = |t|^2 * Re(n_out * cos_th_out) / Re(n_in * cos_th_in)
    #     p-pol: T = |t|^2 * Re(n_out / cos_th_out) / Re(n_in / cos_th_in)
    # Energy conservation (R+T = 1 for real lossless stacks) and agreement with
    # tmm_numpy.coh_tmm to ~1e-7 (see test_coh_stack_power_RT_matches_tmm_numpy)
    # confirm this is the consistent power formula for this amplitude normalization.
    device = n_layers_1d.device
    dtype = torch.complex64

    n_in_t = torch.tensor(n_in, dtype=dtype, device=device)
    n_out_t = torch.tensor(n_out, dtype=dtype, device=device)
    cos_th_in = torch.cos(theta_1d.to(dtype)).unsqueeze(1)  # (batch, 1, angles)
    sin_th_in = torch.sin(theta_1d.to(dtype)).unsqueeze(1)
    sin_th_out = n_in_t * sin_th_in / n_out_t
    cos_th_out = torch.sqrt(1 - sin_th_out ** 2)

    # s-pol: ratio of (n * cos_theta) quantities.
    num_s = (n_out_t * cos_th_out).real   # n_out * cos(th_out)
    den_s = (n_in_t * cos_th_in).real    # n_in  * cos(th_in)
    Ts = (ts.real ** 2 + ts.imag ** 2) * (num_s / den_s)

    # p-pol: ratio of (n / cos_theta) quantities.
    num_p = (n_out_t / cos_th_out).real   # n_out / cos(th_out)
    den_p = (n_in_t / cos_th_in).real    # n_in  / cos(th_in)
    Tp = (tp.real ** 2 + tp.imag ** 2) * (num_p / den_p)

    return Rs, Rp, Ts, Tp


# ===========================================
# Isotropic Film Solver Class
# ===========================================
class IsotropicFilmSolver:
    """Multi-layer coating film solver for isotropic materials.

    Uses the standard 2x2 transfer matrix method which is much faster
    than the general 4x4 anisotropic formulation. This solver calculates 
    (ts, tp, rs, rp) with phase shifts using rigorous electromagnetic wave 
    propagation through multi-layer coating stacks.
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
        Initialize the isotropic film solver.

        Args:
            mat_n_in: Refractive index of incident medium (scalar).
            mat_n_out: Refractive index of exit medium (scalar).
            mat_n_ls: Refractive indices of interior layers, list or tensor of length N.
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

        # Set up refractive indices for each layer
        if torch.is_tensor(mat_n_ls):
            n_layers_t = mat_n_ls.to(torch.complex64)
        else:
            n_layers_t = torch.tensor(mat_n_ls, dtype=torch.complex64)
        self.num_layers = len(n_layers_t)
        self.refract_idx_layers = n_layers_t.unsqueeze(0).expand(self.batch_size, -1).clone()

        # Min and max single layer film thickness in [um]
        self.thickness_min = thickness_min
        self.thickness_max = thickness_max
        self._thickness_range = self.thickness_max - self.thickness_min  # Pre-compute

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
        self.film_params = self.film_params.to(device, non_blocking=True)
        self.refract_idx_layers = self.refract_idx_layers.to(device, non_blocking=True)
        return self

    def load_ckpt(self, ckpt_path):
        """Load checkpoint from file path."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        film_thickness = torch.clamp(ckpt["film_thickness"], self.thickness_min, self.thickness_max)
        film_thickness_normalized = (film_thickness - self.thickness_min) / (
            self.thickness_max - self.thickness_min
        )

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
        Calculate (ts, tp, rs, rp) using TMM for specified angles and wavelengths.

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
        wv_batch = wv.unsqueeze(0).expand(self.batch_size, -1)

        # Get film thickness and compute TMM results
        d_batch = self.get_film_thickness()
        ts, tp, rs, rp = create_jones_matrix_isotropic(
            self.refract_idx_layers, d_batch, wv_batch, self.mat_n_in, self.mat_n_out, theta
        )

        return ts, tp, rs, rp

    def __call__(self, theta, wvln):
        """Forward pass using simulate."""
        return self.simulate(theta, wvln)


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

    Note:
        Inside any coherent sub-stack, the refractive indices of the
        bracketing incoherent layers must be identical across the batch
        dimension. Heterogeneous per-batch indices in stack-bracketing
        layers raise ValueError. Per-batch interior coherent-layer indices
        are still supported.
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
    inc_alllayer_indices = groups["inc_alllayer_indices"]
    stack_alllayer_indices = groups["stack_alllayer_indices"]
    stack_after_inc = groups["stack_after_inc"]

    # Build full per-layer index tensors with the endpoints in place.
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
    sin_th_in = torch.sin(theta_1d).to(complex_dtype)  # (batch, angles)
    n_full_b = n_full.unsqueeze(-1)           # (batch, n_full, 1)
    n_in_b = n_in_col.unsqueeze(-1)           # (batch, 1, 1)
    sin_th_in_b = sin_th_in.unsqueeze(1)      # (batch, 1, angles)
    sin_th_layers = n_in_b * sin_th_in_b / n_full_b   # (batch, n_full, angles)
    cos_th_layers = torch.sqrt(1 - sin_th_layers ** 2)  # complex

    # Single-pass absorption P_i for each *interior* incoherent layer.
    # wv_b shape: (batch, 1, n_wv, 1) for broadcasting over n_int_inc and angles.
    wv_b = wv_1d.unsqueeze(-1).unsqueeze(1)  # (batch, 1, n_wv, 1)
    interior_inc_alllayer = [
        inc_alllayer_indices[i] for i in range(1, num_inc - 1)
    ]
    if len(interior_inc_alllayer) > 0:
        idx = torch.tensor(interior_inc_alllayer, dtype=torch.long, device=device)
        n_inc_interior = n_full.index_select(1, idx)          # (batch, n_int_inc)
        d_inc_interior = d_full.index_select(1, idx)          # (batch, n_int_inc)
        cos_inc_interior = cos_th_layers.index_select(1, idx) # (batch, n_int_inc, angles)
        n_inc_b = n_inc_interior.unsqueeze(-1).unsqueeze(-1)  # (batch, n_int_inc, 1, 1)
        d_inc_b = d_inc_interior.unsqueeze(-1).unsqueeze(-1)  # (batch, n_int_inc, 1, 1)
        cos_inc_b = cos_inc_interior.unsqueeze(2)             # (batch, n_int_inc, 1, angles)
        imag_part = (n_inc_b * cos_inc_b).imag
        P_interior = torch.exp(-4 * torch.pi * d_inc_b.real * imag_part / wv_b.real)
        # shape: (batch, n_int_inc, n_wv, angles)
        P_interior = torch.clamp(P_interior, min=1e-30)
    else:
        P_interior = None

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

    See create_intensity_RT_isotropic for argument meanings.
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
    stack_RT = []
    for s_idx in range(num_stacks):
        layer_idxs = stack_alllayer_indices[s_idx]  # [left_inc, c..., right_inc]
        left_inc_alllayer = layer_idxs[0]
        right_inc_alllayer = layer_idxs[-1]
        coh_alllayer = layer_idxs[1:-1]

        n_left = n_full[:, left_inc_alllayer]   # (batch,) complex
        n_right = n_full[:, right_inc_alllayer]

        # Sub-stack solver requires scalar n_in/n_out per stack. Reject heterogeneous
        # batches that would silently use only the first batch row's index.
        if n_left.numel() > 1 and not torch.all(n_left == n_left[0]):
            raise ValueError(
                f"Stack {s_idx} has heterogeneous refractive index in its left "
                f"bracketing layer across the batch dimension. Per-batch n_in/n_out "
                f"in coherent sub-stacks is not yet supported. Use a single set of "
                f"refractive indices across the batch."
            )
        if n_right.numel() > 1 and not torch.all(n_right == n_right[0]):
            raise ValueError(
                f"Stack {s_idx} has heterogeneous refractive index in its right "
                f"bracketing layer across the batch dimension. Per-batch n_in/n_out "
                f"in coherent sub-stacks is not yet supported."
            )

        n_left_scalar = complex(n_left[0].item())
        n_right_scalar = complex(n_right[0].item())

        idx_coh = torch.tensor(coh_alllayer, dtype=torch.long, device=device)
        n_coh = n_full.index_select(1, idx_coh)            # (batch, n_coh)
        d_coh = d_full.index_select(1, idx_coh).to(real_dtype)

        # Compute local angles in the bracketing media via Snell propagation.
        # cos_th_layers is already Snell-propagated from the user's theta_1d in n_in.
        sin_left = torch.sqrt(1 - cos_th_layers[:, left_inc_alllayer] ** 2)   # (batch, angles), complex
        theta_left = torch.arcsin(torch.clamp(sin_left.real, -1.0, 1.0)).to(real_dtype)
        sin_right = torch.sqrt(1 - cos_th_layers[:, right_inc_alllayer] ** 2)
        theta_right = torch.arcsin(torch.clamp(sin_right.real, -1.0, 1.0)).to(real_dtype)

        Rs_fwd, Rp_fwd, Ts_fwd, Tp_fwd = coh_stack_power_RT_isotropic(
            n_coh, d_coh, wv_1d, n_left_scalar, n_right_scalar, theta_left
        )

        # Backward: reverse layer order and swap media, use the angle in the
        # right bracketing medium as the incident angle.
        n_coh_rev = torch.flip(n_coh, dims=[1])
        d_coh_rev = torch.flip(d_coh, dims=[1])
        Rs_bwd, Rp_bwd, Ts_bwd, Tp_bwd = coh_stack_power_RT_isotropic(
            n_coh_rev, d_coh_rev, wv_1d, n_right_scalar, n_left_scalar, theta_right
        )
        stack_RT.append({
            "Rs_fwd": Rs_fwd, "Rp_fwd": Rp_fwd, "Ts_fwd": Ts_fwd, "Tp_fwd": Tp_fwd,
            "Rs_bwd": Rs_bwd, "Rp_bwd": Rp_bwd, "Ts_bwd": Ts_bwd, "Tp_bwd": Tp_bwd,
        })

    def _interface_RT(inc_i):
        """Return (Rs_f, Rp_f, Ts_f, Tp_f, Rs_b, Rp_b, Ts_b, Tp_b) for interface inc_i -> inc_i+1.

        Each is a real tensor (batch, n_wv, n_angles).
        Tuple positions: 0=Rs_f, 1=Rp_f, 2=Ts_f, 3=Tp_f, 4=Rs_b, 5=Rp_b, 6=Ts_b, 7=Tp_b.
        """
        nxt_stack = stack_after_inc[inc_i]
        if nxt_stack is None:
            # Direct bare interface between consecutive incoherent layers.
            a_idx = inc_alllayer_indices[inc_i]
            b_idx = inc_alllayer_indices[inc_i + 1]
            n_a = n_full[:, a_idx].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            n_b = n_full[:, b_idx].unsqueeze(-1).unsqueeze(-1)
            cos_a = cos_th_layers[:, a_idx].unsqueeze(1)  # (batch, 1, angles)
            cos_b = cos_th_layers[:, b_idx].unsqueeze(1)
            Rs_f, Rp_f, Ts_f, Tp_f = interface_power_RT(n_a, n_b, cos_a, cos_b)
            Rs_b, Rp_b, Ts_b, Tp_b = interface_power_RT(n_b, n_a, cos_b, cos_a)
            # Expand wavelength dimension: (batch, 1, angles) -> (batch, n_wv, angles)
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
        """Build 2x2 intensity L-matrix factor (without the leading diag(1/P, P)).

        Returns real tensor of shape (batch, n_wv, n_angles, 2, 2).
        """
        eps = 1e-30
        Tfwd_safe = torch.clamp(Tfwd, min=eps)
        det = Tbwd * Tfwd - Rbwd * Rfwd
        row0 = torch.stack([torch.ones_like(Rfwd), -Rbwd], dim=-1)
        row1 = torch.stack([Rfwd, det], dim=-1)
        M = torch.stack([row0, row1], dim=-2) / Tfwd_safe.unsqueeze(-1).unsqueeze(-1)
        return M

    def _accumulate(pol):
        """Run the L-matrix accumulation for one polarization. Returns (R, T) tensors."""
        # sel indices into _interface_RT tuple:
        #   (Rs_f, Rp_f, Ts_f, Tp_f, Rs_b, Rp_b, Ts_b, Tp_b)
        #   s-pol: Rfwd=0, Tfwd=2, Rbwd=4, Tbwd=6
        #   p-pol: Rfwd=1, Tfwd=3, Rbwd=5, Tbwd=7
        if pol == "s":
            sel = (0, 2, 4, 6)
        else:
            sel = (1, 3, 5, 7)

        i0 = _interface_RT(0)
        Rfwd = i0[sel[0]]
        Tfwd = i0[sel[1]]
        Rbwd = i0[sel[2]]
        Tbwd = i0[sel[3]]
        Ltilde = _step_L(Rfwd, Tfwd, Rbwd, Tbwd)

        for i in range(1, num_inc - 1):
            ii = _interface_RT(i)
            Rfwd = ii[sel[0]]
            Tfwd = ii[sel[1]]
            Rbwd = ii[sel[2]]
            Tbwd = ii[sel[3]]
            M = _step_L(Rfwd, Tfwd, Rbwd, Tbwd)

            # Apply P factor for interior incoherent layer i.
            # P_interior index: interior inc layers are indices 1..(num_inc-2),
            # stored in P_interior[:, 0..n_int_inc-1], so layer i maps to column i-1.
            P_i = P_interior[:, i - 1]  # (batch, n_wv, angles)
            P_safe = torch.clamp(P_i, min=1e-30)
            inv_P = 1.0 / P_safe

            # Left-multiply M by diag(1/P, P): out-of-place to preserve autograd.
            M_scaled_00 = M[..., 0, 0] * inv_P
            M_scaled_01 = M[..., 0, 1] * inv_P
            M_scaled_10 = M[..., 1, 0] * P_safe
            M_scaled_11 = M[..., 1, 1] * P_safe
            row0 = torch.stack([M_scaled_00, M_scaled_01], dim=-1)
            row1 = torch.stack([M_scaled_10, M_scaled_11], dim=-1)
            M_scaled = torch.stack([row0, row1], dim=-2)

            Ltilde = torch.matmul(Ltilde, M_scaled)

        a = Ltilde[..., 0, 0]
        c = Ltilde[..., 1, 0]
        a_safe = torch.where(a.abs() < 1e-30, torch.full_like(a, 1e-30), a)
        T_total = 1.0 / a_safe
        R_total = c / a_safe
        return R_total, T_total

    Rs_t, Ts_t = _accumulate("s")
    Rp_t, Tp_t = _accumulate("p")
    return Rs_t, Rp_t, Ts_t, Tp_t


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
        """Move tensors to specified device."""
        self.device = device
        self.film_params = self.film_params.to(device, non_blocking=True)
        self.refract_idx_layers = self.refract_idx_layers.to(device, non_blocking=True)
        return self

    def get_film_thickness(self):
        """Convert optimization-friendly film parameters to real film thickness.

        Returns:
            film_thickness: tensor of shape (batch_size, num_layers), in [um].
        """
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
        """Forward pass using simulate."""
        return self.simulate(theta, wvln)

