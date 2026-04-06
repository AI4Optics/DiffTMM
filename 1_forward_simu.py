"""Example 1: Forward TMM Calculation using IsotropicFilmSolver

Initialize a film solver instance with refractive indices and thicknesses,
then call simulate to compute Fresnel coefficients.

Film stack: Glass | Ta2O5 | SiO2 | Ta2O5 | Glass
  - N+2 = 5 refractive indices (2 outer media + 3 layers)
  - N = 3 layer thicknesses

Input:  [B, 2] tensor — each row is [wavelength (um), incident_angle (rad)]
Output: [B, 4] tensor — each row is [ts, tp, rs, rp] (complex Fresnel coefficients)
"""

import torch
from film_solver_isotropic import IsotropicFilmSolver


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Define film stack ----
    # Glass(n=1.5) | Ta2O5(n=2.10) | SiO2(n=1.46) | Ta2O5(n=2.10) | Glass(n=1.5)
    n_in = 1.5
    n_out = 1.5
    n_layers_list = [2.10, 1.46, 2.10]  # 3 interior layers
    d_target = [0.080, 0.120, 0.080]  # layer thicknesses in um

    # ---- Init film solver ----
    wvlns = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]  # wavelengths in um
    solver = IsotropicFilmSolver(
        n_in=n_in,
        n_out=n_out,
        n_layers_list=n_layers_list,
        d_layers=d_target,
        n_mirrors=1,
        device=device,
    )

    print(f"Film stack: Glass(n={n_in}) | Ta2O5(n=2.10, d={d_target[0]*1000:.0f}nm) | SiO2(n=1.46, d={d_target[1]*1000:.0f}nm) | Ta2O5(n=2.10, d={d_target[2]*1000:.0f}nm) | Glass(n={n_out})")
    print(f"Film thickness (um): {solver.get_film_thickness().tolist()}")

    # ---- Forward calculation using simulate ----
    # Query at specific angles and wavelengths
    B = 8
    angles = torch.linspace(0.0, 1.2, B, device=device)  # 0 to ~69 degrees

    # simulate returns (ts, tp, rs, rp), each shape (n_mirrors, n_wvlns, n_angles)
    ts, tp, rs, rp = solver.simulate(theta=angles, wvln=wvlns)

    # Stack into [4, n_mirrors, n_wvlns, n_angles] then reshape to [B, 4]
    # For single mirror (index 0) and single wavelength, output is [n_angles, 4]
    print(f"\nOutput shape per coefficient: {ts.shape}  — (n_mirrors, n_wvlns, n_angles)")

    # ---- Print results for each wavelength ----
    for wi, wv in enumerate(wvlns):
        print(f"\n--- Wavelength = {wv*1000:.0f} nm ---")
        print(f"{'angle (deg)':>12} {'|ts|^2':>8} {'|tp|^2':>8} {'|rs|^2':>8} {'|rp|^2':>8} {'R+T (s)':>8}")
        print("-" * 60)
        for ai in range(B):
            Ts = (ts[0, wi, ai].abs() ** 2).item()
            Tp = (tp[0, wi, ai].abs() ** 2).item()
            Rs = (rs[0, wi, ai].abs() ** 2).item()
            Rp = (rp[0, wi, ai].abs() ** 2).item()
            ang_deg = angles[ai].item() * 180 / torch.pi
            print(f"{ang_deg:12.2f} {Ts:8.4f} {Tp:8.4f} {Rs:8.4f} {Rp:8.4f} {Rs+Ts:8.4f}")

    # ---- Demo: [B, 2] input -> [B, 4] output for a single wavelength ----
    print("\n\n=== [B, 2] -> [B, 4] demo (single wavelength) ===")
    wv_query = 0.55  # um
    inp = torch.stack([
        torch.full((B,), wv_query, device=device),
        angles,
    ], dim=-1)  # (B, 2)

    ts_q, tp_q, rs_q, rp_q = solver.simulate(theta=inp[:, 1], wvln=wv_query)
    out = torch.stack([ts_q[0, 0, :], tp_q[0, 0, :], rs_q[0, 0, :], rp_q[0, 0, :]], dim=-1)  # (B, 4)
    print(f"Input shape:  {inp.shape}  — [wavelength (um), angle (rad)]")
    print(f"Output shape: {out.shape}  — [ts, tp, rs, rp]")
