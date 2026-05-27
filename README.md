# DiffTMM: Differentiable Transfer Matrix Method

A PyTorch-based differentiable thin film solver for multi-layer optical coatings. Supports both isotropic and anisotropic materials with full autograd for inverse design.

## Advantages over NumPy TMM

| Feature               | NumPy TMM ([sbyrnes321/tmm](https://github.com/sbyrnes321/tmm)) | DiffTMM                                                            |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| Differentiable        | No                                                           | Yes (PyTorch autograd)                                             |
| GPU acceleration      | No (CPU only)                                                | Yes (CUDA)                                                         |
| Batch processing      | No (sequential)                                              | Yes (vectorized)                                                   |
| Anisotropic materials | No (isotropic only)                                          | Yes (4x4 transfer matrix)                                          |
| Speed (batch=16)      | 1x baseline                                                  | **~190x** (isotropic 2x2), **~134x** (anisotropic 4x4) |

## Installation

```bash
git clone https://github.com/singer-yang/DiffTMM.git
cd DiffTMM
pip install torch numpy matplotlib scipy
```

## Quick Start

### Forward Simulation (`1_forward_simu.ipynb`)

Initialize a film stack with known refractive indices and thicknesses, then compute Fresnel coefficients at arbitrary wavelengths and angles.

```python
import torch
from difftmm import IsotropicFilmSolver

# Define film stack: Glass | Ta2O5 | SiO2 | Ta2O5 | Glass
solver = IsotropicFilmSolver(
    mat_in=1.5,                           # incident medium
    mat_out=1.5,                          # exit medium
    mat_ls=[2.10, 1.46, 2.10],            # interior layer indices
    thickness_ls=[0.080, 0.120, 0.080],   # thicknesses in um
    device=torch.device("cuda"),
)

# Compute Fresnel coefficients: ts, tp, rs, rp
angles = torch.linspace(0, 1.2, 100, device=solver.device)
ts, tp, rs, rp = solver.simulate(theta=angles, wvln=[0.45, 0.55, 0.65])
# Output shape: (n_mirrors, n_wvlns, n_angles)
```

### Inverse Design via Differentiable Optimization (`2_inverse_design.ipynb`)

Given target Fresnel coefficients, recover unknown film thicknesses using gradient-based optimization.

```python
import torch
from difftmm import create_jones_matrix_isotropic

# Film stack with unknown thicknesses
n_list = torch.tensor([2.10, 1.46, 2.10, 1.46, 2.10], device="cuda")
d_param = torch.nn.Parameter(torch.randn(5, device="cuda") * 0.5)

def param_to_thickness(p):
    return torch.sigmoid(p) * 0.19 + 0.01  # map to [0.01, 0.20] um

# Optimization loop
optimizer = torch.optim.Adam([d_param], lr=0.02)
for step in range(3000):
    optimizer.zero_grad()
    d = param_to_thickness(d_param)
    pred = forward_tmm(n_list, d, n_in=1.0, n_out=1.52, inp=inp)
    loss = ((pred - target).real ** 2 + (pred - target).imag ** 2).mean()
    loss.backward()
    optimizer.step()
```

**Result**: Layer thicknesses recovered from random initialization:

```
Layer     GT (nm)   Recovered (nm)    Error (nm)
  1        60.00           60.00          0.00
  2       130.00          130.00          0.00
  3        85.00           85.00          0.00
  4       110.00          110.00          0.00
  5        70.00           70.00          0.00
```

## Two Solvers

- **`difftmm.IsotropicFilmSolver`** — Fast 2x2 transfer matrix method for isotropic materials (~190x faster than NumPy TMM)
- **`difftmm.FilmSolver`** (also `AnisotropicFilmSolver`) — General 4x4 transfer matrix method for both isotropic and anisotropic materials (~134x faster than NumPy TMM)

Both solvers share the same API:

```python
solver = Solver(
    mat_in=1.0,                    # incident medium refractive index
    mat_out=1.52,                  # exit medium refractive index
    mat_ls=[2.1, 1.46],            # interior layer refractive indices
    thickness_ls=[0.08, 0.12],     # thicknesses in um (optional, random if None)
    device=torch.device("cuda"),
)
ts, tp, rs, rp = solver.simulate(theta, wvln)
```

## Real Materials

DiffTMM ships with wavelength-dependent refractive index support via the
`Material` class. Look up materials by name (case-insensitive):

```python
from difftmm import IsotropicFilmSolver, Material, list_materials

# Bundled catalogs: CDGM/SCHOTT/MISC AGF glasses + thin-film n+k tables
print(len(list_materials()), "materials available")

# Pass material names directly to a solver — they're auto-wrapped in Material()
solver = IsotropicFilmSolver(
    mat_in="air",
    mat_out="N-BK7",                    # Sellmeier (AGF)
    mat_ls=["TiO2", "SiO2"],            # n+k tables for thin-film materials
    thickness_ls=[0.06, 0.10],
)
ts, tp, rs, rp = solver.simulate(theta=angles, wvln=[0.45, 0.55, 0.65])
```

Scalars (float/complex) and strings can be mixed freely in `mat_ls`.
For the 4×4 `FilmSolver`, anisotropic per-axis dispersion is expressed as
a `(mat_x, mat_y, mat_z)` tuple per layer.

Dispersion models supported in v1: **Sellmeier** (analytical, real n) and
**linear interpolation** (lookup tables, complex n + ik).

## Incoherent Layers (thick substrates)

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
    mat_in=1.0,
    mat_out=1.0,
    mat_ls=[2.40, 1.52],
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

## Accuracy Validation

Validated against the reference NumPy TMM library ([sbyrnes321/tmm](https://github.com/sbyrnes321/tmm)) on surface plasmon resonance (SPR) calculations:

![SPR Comparison](https://raw.githubusercontent.com/singer-yang/DiffTMM/main/benchmarks/surface_plasmon_resonance_all_coefficients.png)

The anisotropic 4x4 solver is validated for energy conservation, isotropic limit accuracy, cross-polarization coupling, and reciprocity:

![Anisotropic Validation](https://raw.githubusercontent.com/singer-yang/DiffTMM/main/benchmarks/anisotropic_tmm_comparison.png)

## Performance Benchmarks

### Speed (batch=16, NVIDIA RTX 5090)

![Speed Comparison](https://raw.githubusercontent.com/singer-yang/DiffTMM/main/benchmarks/speed_comparison_tmm_vs_film_solver_batch.png)

| Layers | TMM NumPy (s) | Anisotropic 4x4 (s) | Isotropic 2x2 (s) | Speedup (4x4) | Speedup (2x2) |
| ------ | ------------- | ------------------- | ----------------- | ------------- | ------------- |
| 3      | 0.281         | 0.003               | 0.001             | 84.1x         | 233.0x        |
| 11     | 0.577         | 0.005               | 0.003             | 128.4x        | 201.1x        |
| 25     | 1.076         | 0.008               | 0.006             | 133.7x        | 186.4x        |
| 39     | 1.574         | 0.010               | 0.009             | 165.1x        | 182.2x        |

### GPU Memory (batch=16, forward + backward)

![Memory Comparison](https://raw.githubusercontent.com/singer-yang/DiffTMM/main/benchmarks/memory_comparison_solvers.png)

The isotropic 2x2 solver uses ~23x less GPU memory than the anisotropic 4x4 solver. NumPy TMM is CPU-only (0 GPU memory).

## Repository Structure

```
├── difftmm/                            # Importable package
│   ├── __init__.py                     #   Public API
│   ├── film_solver_isotropic.py        #   2x2 isotropic solver (fast)
│   ├── film_solver_anisotropic.py      #   4x4 anisotropic solver (general)
│   ├── film_solver_incoherent.py       #   2x2 isotropic solver with incoherent layers
│   └── material/                       #   Wavelength-dependent materials
│       ├── __init__.py
│       ├── materials.py                #     Material class and catalog loaders
│       └── catalogs/                   #     Bundled glass + thin-film data
├── 1_forward_simu.ipynb                # Example: forward simulation
├── 2_inverse_design.ipynb              # Example: differentiable inverse design
├── 3_real_materials.ipynb              # Example: real materials
├── 4_incoherent_film.ipynb             # Example: incoherent / thick-substrate solver
├── tmm_numpy/                          # Reference NumPy TMM library
├── benchmarks/                         # Accuracy and performance benchmarks
├── tests/                              # Pytest suite
├── pyproject.toml                      # Packaging metadata
├── CITATION.cff                        # Citation metadata
└── README.md
```

## Physics

- **2x2 transfer matrix method**: Standard formulation for isotropic multi-layer films
- **4x4 transfer matrix method**: General formulation for anisotropic (birefringent) media
- Snell's law, Fresnel equations, evanescent wave handling beyond critical angle
- Bidirectional propagation (forward and reverse through the film stack)
- Complete polarization handling via Jones calculus

## References

- S. J. Byrnes, "Multilayer optical calculations," [arXiv:1603.02720](https://arxiv.org/abs/1603.02720)
- Steven Byrnes' TMM library: [github.com/sbyrnes321/tmm](https://github.com/sbyrnes321/tmm)
- Yang, X., Liu, Z., Nie, Z., Fan, Q., Shi, Z., Bonar, J., & Heidrich, W. (2026). "End-to-end differentiable design of geometric waveguide displays." *arXiv preprint* [arXiv:2601.04370](https://arxiv.org/abs/2601.04370)

## License

DiffTMM is licensed under the [Apache License 2.0](LICENSE).

The bundled NumPy TMM reference library (`tmm_numpy/`) is by Steven Byrnes and is licensed under the [MIT License](tmm_numpy/LICENSE.txt).
