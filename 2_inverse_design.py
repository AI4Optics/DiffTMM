"""Example 2: Differentiable Optimization of Film Thicknesses

Step 1: Create a "ground truth" film stack with known thicknesses, compute
        Fresnel coefficients at sampled (wavelength, angle) pairs.
Step 2: Create a new film stack with same refractive indices but unknown
        (random) thicknesses, then use gradient-based optimization to
        recover the ground truth thicknesses from the target coefficients.

Film stack: Air | Ta2O5 | SiO2 | Ta2O5 | SiO2 | Ta2O5 | Glass
  - N+2 = 7 refractive indices (2 outer media + 5 layers)
  - N = 5 optimizable layer thicknesses
"""

import torch
from film_solver_isotropic import create_jones_matrix_isotropic


def forward_tmm(n_list, d_list, n_in, n_out, inp):
    """
    Forward TMM calculation for arbitrary (wavelength, angle) queries.

    Args:
        n_list: refractive indices of interior layers, shape (N,). Complex or real.
        d_list: thicknesses of interior layers in um, shape (N,).
        n_in: incident medium refractive index (scalar).
        n_out: exit medium refractive index (scalar).
        inp: query tensor of shape (B, 2), columns = [wavelength_um, angle_rad].

    Returns:
        out: complex tensor of shape (B, 4), columns = [ts, tp, rs, rp].
    """
    device = inp.device
    B = inp.shape[0]

    wvlns = inp[:, 0]  # (B,)
    angles = inp[:, 1]  # (B,)

    n_layers_1d = n_list.unsqueeze(0).expand(B, -1).to(torch.complex64).to(device)
    d_1d = d_list.unsqueeze(0).expand(B, -1).to(device)
    wv_1d = wvlns.unsqueeze(1)
    theta_1d = angles.unsqueeze(1)

    ts, tp, rs, rp = create_jones_matrix_isotropic(
        n_layers_1d, d_1d, wv_1d, n_in, n_out, theta_1d
    )
    out = torch.stack([ts[:, 0, 0], tp[:, 0, 0], rs[:, 0, 0], rp[:, 0, 0]], dim=-1)
    return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # ---- Film stack definition ----
    n_in = 1.0   # air
    n_out = 1.52  # glass substrate
    n_list = torch.tensor([2.10, 1.46, 2.10, 1.46, 2.10], device=device)  # 5 layers

    # Ground truth thicknesses (um)
    d_gt = torch.tensor([0.060, 0.130, 0.085, 0.110, 0.070], device=device)

    # ---- Step 1: Generate target data ----
    B = 1024
    wavelengths = torch.lerp(
        torch.full((B,), 0.40, device=device),
        torch.full((B,), 0.70, device=device),
        torch.rand(B, device=device),
    )
    angles = torch.rand(B, device=device) * (torch.pi / 3)  # 0 to 60 degrees
    inp = torch.stack([wavelengths, angles], dim=-1)  # (B, 2)

    with torch.no_grad():
        target = forward_tmm(n_list, d_gt, n_in, n_out, inp)  # (B, 4) complex

    print("=" * 60)
    print("Step 1: Generated target Fresnel coefficients")
    print(f"  Film stack: Air | Ta2O5 | SiO2 | Ta2O5 | SiO2 | Ta2O5 | Glass")
    print(f"  Ground truth d (nm): {(d_gt * 1000).tolist()}")
    print(f"  Input samples: {B} random (wavelength, angle) pairs")
    print(f"  Target shape: {target.shape}")

    # ---- Step 2: Optimize unknown thicknesses ----
    print("\n" + "=" * 60)
    print("Step 2: Differentiable optimization to recover thicknesses")

    # Initialize with random unconstrained parameters; sigmoid maps to [d_min, d_max]
    d_param = torch.nn.Parameter(torch.randn(5, device=device) * 0.5)
    d_min, d_max = 0.01, 0.20  # thickness range in um

    def param_to_thickness(p):
        return torch.sigmoid(p) * (d_max - d_min) + d_min

    optimizer = torch.optim.Adam([d_param], lr=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)

    print(f"  Initial d (nm): {(param_to_thickness(d_param.data) * 1000).tolist()}")
    print(f"  Optimizing (Adam)...")

    for step in range(1, 3001):
        optimizer.zero_grad()
        d_current = param_to_thickness(d_param)
        pred = forward_tmm(n_list, d_current, n_in, n_out, inp)
        diff = pred - target
        loss = (diff.real ** 2 + diff.imag ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0 or step == 1:
            d_nm = (d_current.detach() * 1000).tolist()
            print(f"  Step {step:4d}  loss={loss.item():.2e}  d(nm)=[{', '.join(f'{x:.1f}' for x in d_nm)}]")

    # ---- Evaluate results ----
    print("\n" + "=" * 60)
    print("Results")
    d_final = param_to_thickness(d_param.data)
    d_gt_nm = (d_gt * 1000).tolist()
    d_final_nm = (d_final * 1000).tolist()

    print(f"  {'Layer':<8} {'GT (nm)':>10} {'Recovered (nm)':>15} {'Error (nm)':>12}")
    print("  " + "-" * 48)
    for i in range(len(d_gt)):
        err = abs(d_gt_nm[i] - d_final_nm[i])
        print(f"  {i+1:<8} {d_gt_nm[i]:10.2f} {d_final_nm[i]:15.2f} {err:12.2f}")

    # Verify: compare Fresnel coefficients
    with torch.no_grad():
        pred_final = forward_tmm(n_list, d_final, n_in, n_out, inp)
        residual = ((pred_final - target).abs() ** 2).mean().item()
    print(f"\n  Final MSE on Fresnel coefficients: {residual:.2e}")
