"""
Memory Comparison: Isotropic vs Anisotropic Film Solvers (Differentiable Mode)

This script compares the peak GPU memory consumption of the PyTorch-based
anisotropic and isotropic film solvers during a training step (Forward + Backward).
Standard TMM library is CPU-based, so its GPU memory usage is 0.

Key features:
- Measurements include both forward pass and backward pass (gradients).
- Uses `torch.cuda.max_memory_allocated()` for accurate peak memory usage.
- Clears cache between runs to ensure isolation.
"""

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import pi

from difftmm import create_jones_matrix_AOIAz, create_jones_matrix_isotropic


def measure_peak_memory(func, *args, **kwargs):
    """
    Measure peak GPU memory allocation for a function call (forward + backward).
    
    Args:
        func: Function to execute
        args, kwargs: Arguments for the function
        
    Returns:
        peak_memory_mb: Peak memory usage in MB
    """
    # Force garbage collection and empty cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Record baseline memory (should be close to 0 if cache is empty, but useful to subtract static overhead)
    baseline_memory = torch.cuda.memory_allocated()
    
    # Execute function (Forward)
    # The function is expected to return tensors that require grad
    outputs = func(*args, **kwargs)
    
    # Compute dummy loss and backward (Backward)
    # Use intensity (power) loss which is physically meaningful and phase-invariant
    loss = 0
    if isinstance(outputs, tuple):
        for out in outputs:
            loss = loss + (out * out.conj()).real.sum()
    else:
        loss = (outputs * outputs.conj()).real.sum()
        
    try:
        loss.backward()
    except RuntimeError as e:
        print(f"Warning: Backward pass failed: {e}")
        # Continue with whatever memory was measured (likely forward activations)
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated()
    
    # Cleanup to release graph
    del loss
    del outputs
    
    return (peak_memory - baseline_memory) / (1024 * 1024)  # Convert to MB


def benchmark_isotropic_memory(n_layers, n_angles=180, batch_size=16):
    """Benchmark GPU memory for Isotropic solver."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        return 0.0

    # Setup inputs with requires_grad=True for differentiable mode
    # REFRACTIVE INDICES
    n_layers_1d = torch.zeros((batch_size, n_layers), dtype=torch.complex64, device=device)
    n_layers_1d[:, 0::2] = 1.46
    n_layers_1d[:, 1::2] = 2.13
    n_layers_1d.requires_grad_(True)

    # THICKNESSES
    d_1d = torch.full((batch_size, n_layers), 0.050, dtype=torch.complex64, device=device)
    d_1d.requires_grad_(True)

    # WAVELENGTHS
    wv_1d = torch.full((batch_size, 1), 0.633, dtype=torch.float32, device=device)

    # Angles
    n_in = 1.9
    n_out = 1.9
    theta_1d = torch.linspace(0, pi / 2, n_angles, dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)

    # Wrap function to match signature for measure_peak_memory
    mem_usage = measure_peak_memory(
        create_jones_matrix_isotropic,
        n_layers_1d, d_1d, wv_1d, n_in, n_out, theta_1d
    )
    return mem_usage


def benchmark_anisotropic_memory(n_layers, n_angles=180, batch_size=16):
    """Benchmark GPU memory for Anisotropic solver."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        return 0.0

    # Setup inputs with requires_grad=True
    n_2d = torch.zeros((batch_size, n_layers, 3), dtype=torch.complex64, device=device)
    for i in range(n_layers):
        if i % 2 == 0:
            n_2d[:, i, :] = 1.46
        else:
            n_2d[:, i, :] = 2.13
    n_2d.requires_grad_(True)

    d_1d = torch.full((batch_size, n_layers), 0.050, dtype=torch.complex64, device=device)
    d_1d.requires_grad_(True)

    a_2d = torch.zeros((batch_size, n_layers, 3), dtype=torch.complex64, device=device)
    
    wv_1d = torch.full((batch_size, 1), 0.633, dtype=torch.float64, device=device) # Aniso uses float64 for wl/angles typically or complex
    
    n_in = 1.9
    n_out = 1.9
    
    AOI_1d = torch.linspace(0, pi / 2, n_angles, dtype=torch.float64, device=device).unsqueeze(0).expand(batch_size, -1)
    Az_1d = torch.zeros((batch_size, 1), dtype=torch.float64, device=device)

    mem_usage = measure_peak_memory(
        create_jones_matrix_AOIAz,
        a_2d, n_2d, d_1d, wv_1d, n_in, n_out, AOI_1d, Az_1d
    )
    return mem_usage


def run_memory_comparison():
    print("Running Memory Comparison (Differentiable Mode)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available! Memory comparison requires GPU.")
        return

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Batch Size: 16")
    print(f"Measurements include Forward + Backward pass.")
    
    layer_counts = list(range(1, 100, 4)) # slightly coarser step than speed test
    batch_size = 16
    n_angles = 180
    
    aniso_mems = []
    iso_mems = []
    
    print("\nLayers | Aniso (MB) | Iso (MB) | Ratio (Aniso/Iso)")
    print("-------|------------|----------|------------------")
    
    for n_layers in layer_counts:
        # Benchmark Anisotropic
        mem_aniso = benchmark_anisotropic_memory(n_layers, n_angles, batch_size)
        aniso_mems.append(mem_aniso)
        
        # Benchmark Isotropic
        mem_iso = benchmark_isotropic_memory(n_layers, n_angles, batch_size)
        iso_mems.append(mem_iso)
        
        ratio = mem_aniso / mem_iso if mem_iso > 0 else 0
        
        print(f" {n_layers:3d}   | {mem_aniso:10.2f} | {mem_iso:8.2f} | {ratio:15.1f}x")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(layer_counts, aniso_mems, "s-", color="#FF8C00", label="Anisotropic (4x4)", linewidth=2)
    plt.plot(layer_counts, iso_mems, "d-", color="green", label="Isotropic (2x2)", linewidth=2)
    
    plt.xlabel("Number of Layers")
    plt.ylabel("Peak GPU Memory (MB) [Log Scale]")
    plt.title(f"Peak GPU Memory vs Number of Layers (Differentiable)\nBatch={batch_size}, Angles={n_angles}, Forward+Backward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    # Note: TMM library is CPU-only, so GPU memory = 0
    plt.text(0.98, 0.02, "Note: NumPy TMM library is CPU-only (0 GPU memory, not shown)",
             transform=plt.gca().transAxes, ha="right", va="bottom",
             fontsize=9, style="italic", color="gray")
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "memory_comparison_solvers.png"), dpi=300)
    print(f"\nMemory comparison completed. Plot saved to {os.path.join(SCRIPT_DIR, 'memory_comparison_solvers.png')}")

if __name__ == "__main__":
    run_memory_comparison()
