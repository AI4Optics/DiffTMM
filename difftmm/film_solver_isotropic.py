"""Isotropic Multi-layer Thin Film Solver (2x2 Transfer Matrix Method)

Differentiable thin film solver for isotropic materials using the standard
2x2 transfer matrix method. Computes Fresnel coefficients (ts, tp, rs, rp)
for multi-layer film stacks with full autograd support.

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


def _serialize_spec(spec):
    """Serialize one refractive-index spec to a checkpoint-safe value.

    Material → its lowercase name (str); scalar → complex; 3-tuple → tuple of
    the above (anisotropic; not used by the isotropic solver but supported for
    symmetry).
    """
    from .material import Material

    if isinstance(spec, Material):
        return spec.name
    if isinstance(spec, tuple):
        return tuple(_serialize_spec(s) for s in spec)
    if isinstance(spec, (int, float, complex)):
        return complex(spec)
    raise TypeError(f"Cannot serialize spec of type {type(spec).__name__}")


def _deserialize_spec(value, device):
    """Inverse of _serialize_spec — rewraps strings as Material(name)."""
    from .material import Material

    if isinstance(value, str):
        return Material(value, device=device)
    if isinstance(value, tuple):
        return tuple(_deserialize_spec(v, device) for v in value)
    return value


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
        n_layers_1d: refractive index of each layer.
                     Shape (batchsize, n_layer) — wavelength-independent, OR
                     shape (batchsize, n_wls, n_layer) — dispersive per wavelength. Complex.
        d_1d: thicknesses of all layers, shape (batchsize, n_layer). Real or Complex.
        wv_1d: wavelengths of simulations, shape (batchsize, n_wls). Real.
        n_in: incident media refractive index (top medium).
              Scalar (Python float/complex), OR tensor of shape (batchsize, n_wls)
              for per-wavelength dispersive incident medium.
        n_out: transmit media refractive index (bottom medium).
               Scalar (Python float/complex), OR tensor of shape (batchsize, n_wls)
               for per-wavelength dispersive exit medium.
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

    # Normalize n_in / n_out to (batch, num_wv, 1, 1) complex tensors so they
    # broadcast correctly with the (batch, num_wv, num_angles, num_layer) tensors
    # used inside _compute_isotropic_tmm.
    def _to_per_wvln(x):
        if torch.is_tensor(x) and x.dim() == 2:
            # Already (batch, num_wv)
            return x.to(dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
        if torch.is_tensor(x) and x.dim() == 0:
            x = x.item()
        # Scalar (Python or 0-d tensor that we just unwrapped)
        return torch.tensor(complex(x), dtype=dtype, device=device).view(1, 1, 1, 1)

    n_in_t = _to_per_wvln(n_in)
    n_out_t = _to_per_wvln(n_out)

    # Identify forward (theta <= pi/2) and reverse (theta > pi/2) angles
    pi_half = torch.pi / 2
    is_reverse = theta_1d > pi_half  # (batch, angles)

    # Fast path: if n_in == n_out (symmetric media), we can use a simpler approach
    # For symmetric media with symmetric layer stack, |r(theta)| = |r(pi-theta)|
    # We compute forward angles only and map results for reverse angles
    # Fast path only when both media are scalar and equal
    is_scalar_pair = (
        not torch.is_tensor(n_in) and not torch.is_tensor(n_out)
        and abs(n_in - n_out) < 1e-10
    )
    if is_scalar_pair:
        # Map all angles to [0, pi/2] range
        theta_mapped = torch.where(is_reverse, torch.pi - theta_1d, theta_1d)

        # Expand dimensions (combined for fewer operations)
        # Normalize n_layers to (batch, num_wv, 1, num_layer) complex
        if n_layers_1d.dim() == 2:
            # (batch, num_layer) — wavelength-independent
            n_layers = n_layers_1d.unsqueeze(1).unsqueeze(2).to(dtype=dtype, device=device)
        elif n_layers_1d.dim() == 3:
            # (batch, num_wv, num_layer) — wavelength-dependent
            n_layers = n_layers_1d.unsqueeze(2).to(dtype=dtype, device=device)
        else:
            raise ValueError(f"n_layers_1d must be 2-D or 3-D, got shape {n_layers_1d.shape}")
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
        # Normalize n_layers to (batch, num_wv, 1, num_layer) complex
        if n_layers_1d.dim() == 2:
            # (batch, num_layer) — wavelength-independent
            n_layers = n_layers_1d.unsqueeze(1).unsqueeze(2).to(dtype=dtype, device=device)
        elif n_layers_1d.dim() == 3:
            # (batch, num_wv, num_layer) — wavelength-dependent
            n_layers = n_layers_1d.unsqueeze(2).to(dtype=dtype, device=device)
        else:
            raise ValueError(f"n_layers_1d must be 2-D or 3-D, got shape {n_layers_1d.shape}")
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
        # Flip the layer axis (last dim works for both 2-D and 3-D)
        n_layers_rev = torch.flip(n_layers_1d, dims=[-1])
        d_rev = torch.flip(d_1d, dims=[1])

        # Expand dimensions
        # Normalize n_layers to (batch, num_wv, 1, num_layer) complex
        if n_layers_rev.dim() == 2:
            # (batch, num_layer) — wavelength-independent
            n_layers_rev_exp = n_layers_rev.unsqueeze(1).unsqueeze(2).to(dtype=dtype, device=device)
        elif n_layers_rev.dim() == 3:
            # (batch, num_wv, num_layer) — wavelength-dependent
            n_layers_rev_exp = n_layers_rev.unsqueeze(2).to(dtype=dtype, device=device)
        else:
            raise ValueError(f"n_layers_1d must be 2-D or 3-D, got shape {n_layers_1d.shape}")
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
            mat_n_in: Refractive index of incident medium. May be a scalar
                      (int/float/complex), a str material name, or a Material.
            mat_n_out: Refractive index of exit medium. Same types as mat_n_in.
            mat_n_ls: Refractive indices of interior layers. May be a list/tensor
                      of scalars, str material names, and/or Material objects.
            thickness_ls: Thicknesses of interior layers in um, list or tensor of length N.
                          If None, randomly initializes thicknesses.
            thickness_min: Minimum layer thickness in um.
            thickness_max: Maximum layer thickness in um.
            batch_size: Number of film stacks in the batch dimension.
            sigmoid_param: If True, use sigmoid parameterization for thickness.
            device: PyTorch device.
        """
        from .material import Material

        self.batch_size = batch_size
        self.device = device

        def _normalize(spec):
            if isinstance(spec, str):
                return Material(spec, device=device)
            return spec

        self._n_in_spec = _normalize(mat_n_in)
        self._n_out_spec = _normalize(mat_n_out)

        if torch.is_tensor(mat_n_ls):
            specs = [complex(v.item()) for v in mat_n_ls.flatten()]
        else:
            specs = list(mat_n_ls)
        normalized_specs = []
        for s in specs:
            if isinstance(s, tuple):
                raise ValueError(
                    "IsotropicFilmSolver does not support anisotropic (3-tuple) "
                    "material specs. Use FilmSolver for anisotropic stacks."
                )
            normalized_specs.append(_normalize(s))
        self._n_layer_specs = normalized_specs
        self.num_layers = len(normalized_specs)

        all_scalar = all(
            isinstance(s, (int, float, complex)) for s in self._n_layer_specs
        )
        if all_scalar:
            t = torch.tensor(
                [complex(s) for s in self._n_layer_specs], dtype=torch.complex64
            )
            self.refract_idx_layers = t.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            self.refract_idx_layers = None

        self.mat_n_in = (
            float(mat_n_in)
            if isinstance(mat_n_in, (int, float)) and not isinstance(mat_n_in, bool)
            else None
        )
        self.mat_n_out = (
            float(mat_n_out)
            if isinstance(mat_n_out, (int, float)) and not isinstance(mat_n_out, bool)
            else None
        )

        self.thickness_min = thickness_min
        self.thickness_max = thickness_max
        self._thickness_range = self.thickness_max - self.thickness_min

        self.sigmoid_param = sigmoid_param
        if thickness_ls is not None:
            if not torch.is_tensor(thickness_ls):
                thickness_ls = torch.tensor(thickness_ls, dtype=torch.float32)
            normalized = (
                thickness_ls.clamp(self.thickness_min, self.thickness_max)
                - self.thickness_min
            ) / self._thickness_range
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
        if self.refract_idx_layers is not None:
            self.refract_idx_layers = self.refract_idx_layers.to(device, non_blocking=True)
        return self

    def load_ckpt(self, ckpt_path):
        """Load thicknesses (and spec metadata) from a checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        film_thickness = torch.clamp(
            ckpt["film_thickness"], self.thickness_min, self.thickness_max
        )
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

        if "layer_specs" in ckpt:
            self._n_in_spec = _deserialize_spec(ckpt["n_in_spec"], self.device)
            self._n_out_spec = _deserialize_spec(ckpt["n_out_spec"], self.device)
            self._n_layer_specs = [
                _deserialize_spec(v, self.device) for v in ckpt["layer_specs"]
            ]

    def save_ckpt(self, save_path):
        """Save thicknesses and material specs to a checkpoint.

        Material objects are persisted by name; scalars are persisted by value.
        Per-axis 3-tuples are persisted element-wise.
        """
        payload = {
            "film_thickness": self.get_film_thickness().cpu(),
            "batch_size": self.batch_size,
            "num_layers": self.num_layers,
            "n_in_spec":  _serialize_spec(self._n_in_spec),
            "n_out_spec": _serialize_spec(self._n_out_spec),
            "layer_specs": [_serialize_spec(s) for s in self._n_layer_specs],
            "n_in":  self.mat_n_in,
            "n_out": self.mat_n_out,
            "refract_idx_layers": (
                self.refract_idx_layers.cpu()
                if self.refract_idx_layers is not None
                else None
            ),
        }
        torch.save(payload, save_path)

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
        from .material import resolve_indices

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

        n_in_t = resolve_indices(self._n_in_spec, wv, self.device).unsqueeze(0).expand(
            self.batch_size, -1
        )
        n_out_t = resolve_indices(self._n_out_spec, wv, self.device).unsqueeze(0).expand(
            self.batch_size, -1
        )
        per_layer = [resolve_indices(s, wv, self.device) for s in self._n_layer_specs]
        n_layers_t = torch.stack(per_layer, dim=-1)
        n_layers_t = n_layers_t.unsqueeze(0).expand(self.batch_size, -1, -1)

        d_batch = self.get_film_thickness()
        ts, tp, rs, rp = create_jones_matrix_isotropic(
            n_layers_t, d_batch, wv_batch, n_in_t, n_out_t, theta
        )
        return ts, tp, rs, rp

    def __call__(self, theta, wvln):
        """Forward pass using simulate."""
        return self.simulate(theta, wvln)

