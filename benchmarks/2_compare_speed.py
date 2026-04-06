"""
Speed Comparison: TMM Library vs PyTorch film_solver (Batch Mode)

This script compares the computational performance of the standard TMM (Transfer Matrix Method)
library with the PyTorch-based film_solver implementation for multi-layer thin film calculations.

The comparison is performed with a batch size of 16 to demonstrate the advantage of
film_solver's batch processing capability. TMM library processes samples sequentially,
while film_solver processes all 16 samples in parallel.

The comparison is performed across different numbers of layers (1 to 99 layers) to show
how performance scales with film stack complexity.
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import inf, linspace, pi

from tmm_numpy.tmm_core import coh_tmm
from film_solver_anisotropic import create_jones_matrix_AOIAz
from film_solver_isotropic import create_jones_matrix_isotropic


def benchmark_tmm_library(n_layers, n_angles=100, n_repeats=5, batch_size=1):
    """
    Benchmark the standard TMM library performance.

    Args:
        n_layers: Number of film layers (excluding semi-infinite media)
        n_angles: Number of angles to evaluate
        n_repeats: Number of timing repeats for averaging
        batch_size: Number of samples to process (TMM processes sequentially)

    Returns:
        avg_time: Average time per simulation in seconds
        result: Sample result for verification
    """
    # Create alternating refractive indices: SiO2 (1.46) and TiO2 (2.13)
    n_list = [1.9]  # incident medium (glass)

    for i in range(n_layers):
        if i % 2 == 0:
            n_list.append(1.46)  # SiO2
        else:
            n_list.append(2.13)  # TiO2

    n_list.append(1.9)  # substrate (glass)

    # Create thickness list: 50nm alternating layers
    d_list = [inf] + [0.050] * n_layers + [inf]  # thicknesses in um

    # Wavelength and angles
    lam_vac = 0.633  # 633nm in um
    theta_list = linspace(0, pi / 2, n_angles)  # 0 to 90 degrees

    # Warm up
    coh_tmm("p", n_list, d_list, theta_list[0], lam_vac)

    # Time the calculations
    times = []
    for _ in range(n_repeats):
        start_time = time.time()

        # TMM library doesn't support batch processing, so we loop over batch_size
        for _ in range(batch_size):
            Rp = []
            for theta in theta_list:
                result = coh_tmm("p", n_list, d_list, theta, lam_vac)
                Rp.append(result["R"])

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    return avg_time, Rp


def benchmark_isotropic_solver(n_layers, n_angles=100, n_repeats=5, batch_size=1):
    """
    Benchmark the PyTorch Isotropic film_solver performance.

    Args:
        n_layers: Number of film layers
        n_angles: Number of angles to evaluate
        n_repeats: Number of timing repeats for averaging
        batch_size: Number of samples to process in parallel (batch dimension)

    Returns:
        avg_time: Average time per simulation in seconds
        result: Sample result for verification
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # REFRACTIVE INDICES
    n_layers_1d = torch.zeros(
        (batch_size, n_layers), dtype=torch.complex64, device=device
    )
    n_layers_1d[:, 0::2] = 1.46  # SiO2
    n_layers_1d[:, 1::2] = 2.13  # TiO2

    # THICKNESSES
    d_1d = torch.full(
        (batch_size, n_layers), 0.050, dtype=torch.complex64, device=device
    )

    # WAVELENGTHS
    wv_1d = torch.full((batch_size, 1), 0.633, dtype=torch.float32, device=device)

    # Incident and output media
    n_in = 1.9
    n_out = 1.9

    # ANGLES
    theta_1d = (
        torch.linspace(0, pi / 2, n_angles, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # Warm up
    create_jones_matrix_isotropic(n_layers_1d, d_1d, wv_1d, n_in, n_out, theta_1d)

    # Time calculations
    times = []
    for _ in range(n_repeats):
        start_time = time.time()
        create_jones_matrix_isotropic(n_layers_1d, d_1d, wv_1d, n_in, n_out, theta_1d)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)

    # Verification result (Rp)
    _, _, _, rp_amp = create_jones_matrix_isotropic(
        n_layers_1d, d_1d, wv_1d, n_in, n_out, theta_1d
    )
    Rp = (rp_amp[0, 0, :].abs() ** 2).cpu().numpy()

    return avg_time, Rp


def benchmark_film_solver(n_layers, n_angles=100, n_repeats=5, batch_size=1):
    """
    Benchmark the PyTorch film_solver performance with batch processing.

    Args:
        n_layers: Number of film layers
        n_angles: Number of angles to evaluate
        n_repeats: Number of timing repeats for averaging
        batch_size: Number of samples to process in parallel (batch dimension)

    Returns:
        avg_time: Average time per simulation in seconds
        result: Sample result for verification
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create alternating refractive indices with batch dimension
    n_2d = torch.zeros((batch_size, n_layers, 3), dtype=torch.complex64, device=device)
    for i in range(n_layers):
        if i % 2 == 0:
            n_2d[:, i, :] = 1.46  # SiO2
        else:
            n_2d[:, i, :] = 2.13  # TiO2

    # Create alternating thicknesses: 50nm layers with batch dimension
    d_1d = torch.full(
        (batch_size, n_layers), 0.050, dtype=torch.complex64, device=device
    )

    # Azimuth angles (zero for isotropic) with batch dimension
    a_2d = torch.zeros((batch_size, n_layers, 3), dtype=torch.complex64, device=device)

    # Wavelength: 633nm with batch dimension
    wv_1d = torch.full((batch_size, 1), 0.633, dtype=torch.float64, device=device)

    # Incident and output media
    n_in = 1.9  # glass
    n_out = 1.9  # glass

    # Angles of incidence with batch dimension
    AOI_1d = (
        torch.linspace(0, pi / 2, n_angles, dtype=torch.float64, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # Azimuth angle with batch dimension
    Az_1d = torch.zeros((batch_size, 1), dtype=torch.float64, device=device)

    # Warm up
    Jt, Jr = create_jones_matrix_AOIAz(
        a_2d, n_2d, d_1d, wv_1d, n_in, n_out, AOI_1d, Az_1d
    )

    # Time the calculations
    times = []
    for _ in range(n_repeats):
        start_time = time.time()

        Jt, Jr = create_jones_matrix_AOIAz(
            a_2d, n_2d, d_1d, wv_1d, n_in, n_out, AOI_1d, Az_1d
        )

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)

    # Get sample result (p-polarized reflection) - just from first batch element
    p_in = torch.tensor(
        [[1.0 + 0.0j], [0.0 + 0.0j]], dtype=torch.complex64, device=device
    )
    p_in = p_in.reshape(1, 1, 1, 1, 2, 1).expand(batch_size, 1, n_angles, 1, -1, -1)
    r_vec_p = torch.matmul(Jr, p_in)
    Rp = (r_vec_p[0, :, :, :, 0, 0].squeeze().abs() ** 2).cpu().numpy()

    return avg_time, Rp


def run_speed_comparison():
    """
    Run the complete speed comparison experiment with batch processing.
    """
    print("Running speed comparison: TMM Library vs PyTorch film_solver (Batch Mode)")
    print("=" * 70)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Layer counts to test (odd numbers from 1 to 99)
    layer_counts = list(range(1, 41, 2))

    # Number of repeats for averaging to reduce variance
    n_repeats = 5
    n_angles = 180
    batch_size = 16

    tmm_times = []
    film_solver_times = []
    iso_solver_times = []

    print(f"Batch size: {batch_size}")
    print(f"Benchmarking different layer counts (averaged over {n_repeats} runs)...")
    print("Layers | TMM (s)   | Aniso (s)  | Iso (s)    | Spd(Aniso) | Spd(Iso)")
    print("-------|-----------|------------|------------|------------|---------")

    for n_layers in layer_counts:
        print(f"{n_layers:2d}", end="", flush=True)

        # Benchmark TMM library (processes batch_size samples sequentially)
        tmm_time, _ = benchmark_tmm_library(
            n_layers, n_angles=n_angles, n_repeats=n_repeats, batch_size=batch_size
        )
        tmm_times.append(tmm_time)

        # Benchmark film_solver (processes batch_size samples in parallel)
        fs_time, _ = benchmark_film_solver(
            n_layers, n_angles=n_angles, n_repeats=n_repeats, batch_size=batch_size
        )
        film_solver_times.append(fs_time)

        # Benchmark isotropic solver
        iso_time, _ = benchmark_isotropic_solver(
            n_layers, n_angles=n_angles, n_repeats=n_repeats, batch_size=batch_size
        )
        iso_solver_times.append(iso_time)

        speedup_fs = tmm_time / fs_time if fs_time > 0 else float("inf")
        speedup_iso = tmm_time / iso_time if iso_time > 0 else float("inf")

        print(
            f"     | {tmm_time:9.4f} | {fs_time:10.4f} | {iso_time:10.4f} | {speedup_fs:9.1f}x | {speedup_iso:7.1f}x"
        )

    # Create plot (computation time only)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    ax1.plot(
        layer_counts,
        tmm_times,
        "o-",
        color="royalblue",
        label=f"TMM (NumPy, {batch_size}x sequential)",
        linewidth=2,
        markersize=5,
    )
    ax1.plot(
        layer_counts,
        film_solver_times,
        "s-",
        color="#FF8C00",
        label=f"Anisotropic 4x4 (PyTorch, batch={batch_size})",
        linewidth=2,
        markersize=5,
    )
    ax1.plot(
        layer_counts,
        iso_solver_times,
        "d-",
        color="green",
        label=f"Isotropic 2x2 (PyTorch, batch={batch_size})",
        linewidth=2,
        markersize=5,
    )
    ax1.set_xlabel("Number of Layers")
    ax1.set_ylabel("Computation Time (seconds)")
    ax1.set_title(
        f"Computation Time vs Number of Layers\n(Batch Size: {batch_size}, Device: {device})"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    plt.tight_layout()
    plt.savefig(
        os.path.join(SCRIPT_DIR, "speed_comparison_tmm_vs_film_solver_batch.png"), dpi=300, bbox_inches="tight"
    )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (Batch Mode)")
    print("=" * 70)

    avg_tmm_time = np.mean(tmm_times)
    avg_fs_time = np.mean(film_solver_times)
    avg_iso_time = np.mean(iso_solver_times)
    # Recompute speedups to be safe
    speedups = [t / f if f > 0 else 0 for t, f in zip(tmm_times, film_solver_times)]
    speedups_iso = [t / i if i > 0 else 0 for t, i in zip(tmm_times, iso_solver_times)]
    avg_speedup = np.mean(speedups)
    avg_speedup_iso = np.mean(speedups_iso)

    print(f"Batch size: {batch_size}")
    print(f"Average TMM time:       {avg_tmm_time:.4f} s")
    print(f"Average Anisotropic time: {avg_fs_time:.4f} s")
    print(f"Average Isotropic time:   {avg_iso_time:.4f} s")
    print(f"Average Aniso Speedup:    {avg_speedup:.1f}x")
    print(f"Average Iso Speedup:      {avg_speedup_iso:.1f}x")
    print()
    print("Performance characteristics:")
    print(f"  Layers tested: {len(layer_counts)} configurations")
    print(f"  Layer range: {min(layer_counts)} to {max(layer_counts)} layers")
    print(f"  Repeats per measurement: {n_repeats} (for variance reduction)")
    print(f"  Batch size: {batch_size} samples")
    print(f"  TMM: processes {batch_size} samples sequentially")
    print(f"  film_solver: processes {batch_size} samples in parallel")

    # Test with specific layer counts
    test_layers = [3, 11, 25, 51, 99]
    print("\nDetailed results for selected layer counts:")
    print("Layers | TMM (s) | Aniso (s) | Iso (s)  | Spd(Aniso) | Spd(Iso)")
    print("-------|---------|-----------|----------|------------|---------")

    for nl in test_layers:
        try:
            idx = layer_counts.index(nl)
            t_time = tmm_times[idx]
            f_time = film_solver_times[idx]
            i_time = iso_solver_times[idx]
            spd = speedups[idx]
            spd_iso = speedups_iso[idx]
            print(
                f" {nl:2d}    | {t_time:7.4f} | {f_time:8.4f}  | {i_time:8.4f} | {spd:9.1f}x | {spd_iso:8.1f}x"
            )
        except ValueError:
            pass  # Layer count not found in tested list

    return layer_counts, tmm_times, film_solver_times


if __name__ == "__main__":
    # Run the comparison
    layer_counts, tmm_times, film_solver_times = run_speed_comparison()

    print("\nSpeed comparison (batch mode) completed!")
    print("Results saved to: speed_comparison_tmm_vs_film_solver_batch.png")
