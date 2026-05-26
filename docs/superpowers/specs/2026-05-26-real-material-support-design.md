# Real Material Support — Design

**Status:** Draft (brainstorm-approved)
**Date:** 2026-05-26
**Tracking issue:** [AI4Optics/DiffTMM#3](https://github.com/AI4Optics/DiffTMM/issues/3)
**Reference implementation:** [vccimaging/DeepLens — deeplens/material/](https://github.com/vccimaging/DeepLens/tree/main/deeplens/material)

## Summary

Add wavelength-dependent refractive-index support to DiffTMM. Today, all refractive indices are static scalars; users cannot model real materials like SiO₂, TiO₂, Ta₂O₅, Ag, Au, or Si over a broadband spectrum.

This design introduces a `Material` class under `difftmm/material/` with two dispersion models — analytical **Sellmeier** and tabulated **interpolation** (linear, complex n+ik) — backed by bundled material catalogs adapted from DeepLens plus a new thin-film–focused n+k dataset. Both solvers (`IsotropicFilmSolver`, `FilmSolver`) accept `Material` objects, plain scalars, or string material names; mixing is allowed per layer.

## Motivation

The existing API:

```python
IsotropicFilmSolver(
    mat_n_in=1.0,
    mat_n_out=1.52,
    mat_n_ls=[2.10, 1.46, 2.10],
    ...,
)
```

treats each refractive index as a single scalar — independent of wavelength. This blocks accurate simulation of:

- **Broadband coatings** (anti-reflection, dichroic, beam-splitter) where dispersion is the dominant effect across visible/NIR.
- **Plasmonic and absorbing layers** (Ag, Au, Al) where n + ik varies steeply with wavelength.
- **High-index dielectrics** (TiO₂, Ta₂O₅, Si) used in modern thin-film stacks.

Issue #3 explicitly requests a `RefractiveIndexTable` (lookup) and `SellmeierMaterial` (analytical) with differentiable complex tensors, full back-compat, and anisotropic per-axis support.

## Goals

- Wavelength-dependent **complex** refractive index `n + ik` available throughout the TMM pipeline.
- Material lookup by name (catalog-backed).
- Full backward compatibility with the existing scalar API. Existing notebooks and benchmarks must run unchanged and produce bit-identical results.
- Anisotropic per-axis dispersion in the 4×4 solver via a per-layer 3-tuple of materials.
- Differentiable through `torch.autograd` for inverse design.

## Non-goals (v1)

- Schott and Cauchy dispersion equations (kept out of scope; v1 ships `sellmeier` + `interp` only).
- Inline `"n/V"` Cauchy syntax (DeepLens has this; we drop it).
- `optimizable` dispersion mode (learnable nd, V) — used for lens material discovery; out of scope for thin-film v1.
- Cubic / PCHIP interpolation — v1 is linear only.
- Persisting `Material`-based solvers to checkpoints — only scalar stacks round-trip through `save_ckpt`/`load_ckpt`.

## Decisions

| Topic | Decision |
|---|---|
| Class structure | Single monolithic `Material` class with a `.dispersion` attribute (DeepLens-style) |
| Dispersion modes shipped | `sellmeier`, `interp` |
| `.ior(wvln)` return dtype | Always `torch.complex64` (k = 0 implied for non-absorbing materials) |
| Solver input flexibility | `float \| complex \| str \| Material` per slot, mixable per layer |
| Anisotropic representation | Per-layer 3-tuple `(mat_x, mat_y, mat_z)` of any of the above |
| Bundled data | 4 DeepLens AGF catalogs (Sellmeier entries only) + DeepLens `materials_data.json` (Sellmeier+Interp tables only) + new `thin_film_materials.json` (n+ik tables) |
| Interpolation method | Linear only |
| `DeepObj` dependency | Dropped — `Material` is a standalone class with no external base |
| New runtime dependencies | None (drop numpy by replacing `np.interp` with `torch.searchsorted`) |

## File layout

```
difftmm/material/
├── __init__.py                       # Exports: Material, MATERIAL_data, list_materials, resolve_indices
├── materials.py                      # Material class + AGF/JSON loader (~400 LoC)
└── catalogs/
    ├── CDGM.AGF                      # From DeepLens (verbatim)
    ├── SCHOTT.AGF                    # From DeepLens (verbatim)
    ├── MISC.AGF                      # From DeepLens (verbatim)
    ├── PLASTIC2022.AGF               # From DeepLens (verbatim — mostly Schott, mostly skipped)
    ├── materials_data.json           # From DeepLens (verbatim; SELLMEIER_TABLE + INTERP_TABLE used)
    └── thin_film_materials.json      # NEW: INTERP_NK_TABLE with n+k for thin-film materials
```

### `thin_film_materials.json` schema (new)

```json
{
  "_info": {
    "INTERP_NK_TABLE": "Wavelength (um), refractive index n, extinction k. References: refractiveindex.info."
  },
  "INTERP_NK_TABLE": {
    "SiO2":  {"wvlns": [0.30, 0.31, ...], "n": [...], "k": [...]},
    "TiO2":  {"wvlns": [...], "n": [...], "k": [...]},
    "Ta2O5": {"wvlns": [...], "n": [...], "k": [...]},
    "MgF2":  {"wvlns": [...], "n": [...], "k": [...]},
    "Si":    {"wvlns": [...], "n": [...], "k": [...]},
    "Ag":    {"wvlns": [...], "n": [...], "k": [...]},
    "Au":    {"wvlns": [...], "n": [...], "k": [...]},
    "Al":    {"wvlns": [...], "n": [...], "k": [...]},
    "ITO":   {"wvlns": [...], "n": [...], "k": [...]}
  }
}
```

Sampling: ≤10 nm step in the visible (linear interp is accurate at this density). Sources: refractiveindex.info — citations recorded in `_info` comments per material.

## `Material` class API

```python
class Material:
    """Optical material with wavelength-dependent complex refractive index.

    Attributes:
        name (str): Lowercase material name.
        dispersion (str): 'sellmeier' | 'interp'.
        n (float): Nominal refractive index at d-line (587 nm).
        V (float): Abbe number (1e38 for non-dispersive 'air').
        device (torch.device): Active device.
    """

    def __init__(self, name: str | None = None, device: torch.device | str = "cpu"):
        """Look up `name` in (in priority order):
          1. Air alias (`None`, `"air"`, `"vacuum"`, `"occluder"`) → non-dispersive n=1.
          2. AGF Sellmeier entries (`calculate_mode == 2`).
          3. DeepLens JSON `SELLMEIER_TABLE`.
          4. DeepLens JSON `INTERP_TABLE` (real n, k = 0).
          5. New `INTERP_NK_TABLE` (complex n+ik).
        Raises NotImplementedError if not found.
        """

    def ior(self, wvln: torch.Tensor) -> torch.Tensor:
        """Compute the refractive index at given wavelengths.

        Args:
            wvln: real torch.Tensor of arbitrary shape, wavelengths in μm.
        Returns:
            torch.complex64 tensor with the same shape as `wvln`. For non-absorbing
            materials the imaginary part is zero.
        """

    def refractive_index(self, wvln) -> complex | torch.Tensor:
        """Thin wrapper around `ior` that accepts a Python float and returns a
        Python complex, for interactive/scripting use."""

    def to(self, device) -> "Material":
        """Move cached interpolation tensors to `device`."""

    @classmethod
    def list_materials(cls) -> list[str]:
        """Return all known material names from all catalogs."""
```

### Dispersion formulae

- **Sellmeier:** `n² = 1 + k1·λ²/(λ²-l1) + k2·λ²/(λ²-l2) + k3·λ²/(λ²-l3)`; return `n + 0j`.
- **Interp:** linear interpolation through cached `(wvlns, n, k)` tensors using `torch.searchsorted` for index lookup; return `n + 1j·k`.

### Module-level exports (`difftmm/material/__init__.py`)

```python
from .materials import Material, MATERIAL_data, list_materials, resolve_indices

__all__ = ["Material", "MATERIAL_data", "list_materials", "resolve_indices"]
```

### Top-level re-exports (`difftmm/__init__.py`)

```python
from .material import Material, list_materials, MATERIAL_data
```

## `resolve_indices` helper

A small utility shared between both solvers. Lives in `difftmm/material/materials.py`.

```python
def resolve_indices(
    spec,
    wvln: torch.Tensor,
    device,
    *,
    axis: int | None = None,
) -> torch.Tensor:
    """Resolve a refractive-index spec at given wavelengths.

    Args:
        spec: One of:
            - float, complex, or 0-d torch.Tensor (treated as wavelength-independent).
            - str: looked up via Material(spec) (cached internally).
            - Material: `.ior(wvln)` is called.
            - 3-tuple of the above (anisotropic; `axis` selects which element).
        wvln: real 1-D tensor of wavelengths in μm.
        device: torch.device.
        axis: For 3-tuple specs, which axis (0, 1, 2). For non-tuple specs, ignored.

    Returns:
        torch.complex64 tensor of shape `(n_wvlns,)`. Scalar specs broadcast.
    """
```

String specs are wrapped to `Material(spec)` and cached on the solver instance so repeated `simulate()` calls don't re-lookup the catalog. Material name validation happens eagerly at solver `__init__` time — invalid names raise `NotImplementedError` immediately.

## Solver integration

### Constructor signatures

Unchanged at the parameter-name level; only the accepted types broaden.

```python
IsotropicFilmSolver(
    mat_n_in:  float | complex | str | Material,
    mat_n_out: float | complex | str | Material,
    mat_n_ls:  list[float | complex | str | Material] | torch.Tensor,
    thickness_ls=None,
    thickness_min=0.0,
    thickness_max=0.2,
    batch_size=1,
    sigmoid_param=False,
    device=torch.device("cuda"),
)

FilmSolver(  # also exported as AnisotropicFilmSolver
    mat_n_in:  float | complex | str | Material,
    mat_n_out: float | complex | str | Material,
    mat_n_ls:  list[
        float | complex | str | Material |
        tuple[float|complex|str|Material, float|complex|str|Material, float|complex|str|Material]
    ] | torch.Tensor,
    thickness_ls=None,
    thickness_min=0.0,
    thickness_max=0.2,
    batch_size=1,
    sigmoid_param=False,
    device=torch.device("cuda"),
)
```

The `IsotropicFilmSolver` rejects 3-tuple layer entries with a clear `ValueError` pointing the user to `FilmSolver`.

The `FilmSolver` broadcasts an isotropic entry (scalar or single Material) across all three axes internally.

### Internal data model

New per-solver attributes — primary state:

```python
self._n_in_spec:  float | complex | Material
self._n_out_spec: float | complex | Material
self._n_layer_specs: list[
    float | complex | Material
    | tuple[float|complex|Material, float|complex|Material, float|complex|Material]
]
```

Strings are normalized to `Material(name)` at `__init__` time, so downstream code sees only `float | complex | Material | tuple[...]`.

For backward-compatible checkpointing, the existing `self.refract_idx_layers` attribute is retained but becomes **a derived view** that is only materialized when every layer spec is a scalar/complex constant (no `Material` objects). When `Material`-based specs are present, `self.refract_idx_layers` is set to `None` and `save_ckpt` raises a clear error referring users to material-name reconstruction on load.

### `simulate(theta, wvln)` flow

```python
def simulate(self, theta, wvln):
    # ... existing wvln/theta tensor normalization ...

    # NEW: resolve all refractive indices at this wvln
    n_in_t  = resolve_indices(self._n_in_spec,  wv, self.device)  # (n_wvlns,)
    n_out_t = resolve_indices(self._n_out_spec, wv, self.device)  # (n_wvlns,)

    # Isotropic solver:
    n_layers_t = torch.stack([
        resolve_indices(s, wv, self.device) for s in self._n_layer_specs
    ], dim=-1)                                                    # (n_wvlns, n_layers)
    # Add batch axis (and broadcast for now — no per-batch index variation in v1)
    n_layers_batched = n_layers_t.unsqueeze(0).expand(self.batch_size, -1, -1)

    # Anisotropic solver: stack 3 axes per layer
    n_layers_t = torch.stack([
        torch.stack([resolve_indices(s, wv, self.device, axis=ax)
                     for ax in (0, 1, 2)], dim=-1)
        for s in self._n_layer_specs
    ], dim=-2)                                                    # (n_wvlns, n_layers, 3)

    # ... rest of existing TMM computation, with broadened shapes ...
```

### Core TMM function signature changes

Both `create_jones_matrix_isotropic` ([film_solver_isotropic.py:196](difftmm/film_solver_isotropic.py:196)) and `create_jones_matrix_AOIAz` ([film_solver_anisotropic.py:361](difftmm/film_solver_anisotropic.py:361)) currently accept `n_in` and `n_out` as Python scalars. They must change to accept tensors broadcastable to `(batch, n_wvlns)`:

- **Before:**
  - `n_in: float`, `n_out: float`
  - `n_layers_1d: (batch, n_layer)` complex
- **After:**
  - `n_in:  (batch, n_wvlns)` complex (or scalar, broadcastable)
  - `n_out: (batch, n_wvlns)` complex (or scalar, broadcastable)
  - `n_layers_1d: (batch, n_wvlns, n_layer)` complex (isotropic)
  - `n_2d: (batch, n_wvlns, n_layer, 3)` complex (anisotropic)

The change is mostly removing two `unsqueeze` operations inside the TMM core and adjusting which broadcast axis carries wavelengths.

### Checkpoint compatibility

- For **scalar-only** stacks, `save_ckpt` / `load_ckpt` continue to use the derived `self.refract_idx_layers` tensor — byte-level back-compat with existing checkpoints is preserved.
- For **Material-bearing** stacks, `save_ckpt` raises a `NotImplementedError` with a clear message: persist thicknesses separately and reconstruct the solver with the same material list at load time. This v1 limitation is documented in the docstring and notebook. A future v2 could serialize material names too.

## Backward compatibility

The existing scalar API continues to work bit-identically. Concretely:

- `IsotropicFilmSolver(mat_n_in=1.0, mat_n_out=1.52, mat_n_ls=[2.10, 1.46, 2.10], ...)` → no `Material` lookup, the scalar path inside `resolve_indices` produces the same tensor as today.
- `1_forward_simu.ipynb` and `2_inverse_design.ipynb` outputs must match exactly (regression-tested).
- All benchmarks under `benchmarks/` continue to compile and pass.

## Testing strategy

Tests live under `tests/material/` (new directory; `tests/` itself is created if absent).

| File | Coverage |
|---|---|
| `test_material_class.py` | • `Material('air')` returns `1.0 + 0j` for any wvln.<br>• `Material('N-BK7').ior(0.5876)` ≈ `1.5168 + 0j` (Sellmeier, published nd).<br>• `Material('SiO2').ior(0.55)` matches the bundled table value.<br>• `Material('Ag').ior(0.55)` returns a tensor with nontrivial imaginary part.<br>• Autograd: `wvln.requires_grad_(); Material('SiO2').ior(wvln).sum().backward()` produces finite gradients.<br>• `Material('SiO2').to('cuda')` moves cached tensors to GPU. |
| `test_solver_with_materials.py` | • `IsotropicFilmSolver(mat_n_in='air', mat_n_ls=['SiO2','TiO2'], ...).simulate(...)` runs without error.<br>• Mixed input `mat_n_ls=[1.5, Material('SiO2'), 'TiO2']` produces the expected per-layer ior.<br>• String input == Material input == manual scalar (when material is non-dispersive at the query wvln). |
| `test_anisotropic_materials.py` | • `FilmSolver` with `mat_n_ls=[('SiO2','SiO2','SiO2')]` matches `mat_n_ls=['SiO2']` (degenerate isotropic).<br>• Birefringence: `mat_n_ls=[('TiO2','SiO2','SiO2')]` produces non-zero cross-polarization in expected regimes. |
| `test_backward_compat.py` | • Output of `1_forward_simu.ipynb`-style call matches a frozen baseline tensor.<br>• Output of `2_inverse_design.ipynb`-style optimization recovers the same layer thicknesses. |

Reference values come from:

- Published nd for N-BK7, N-SF11, etc. (well-known Sellmeier results).
- refractiveindex.info raw tables for n+k materials (the bundled JSON is derived from these).
- Hash-stamped frozen tensors saved alongside the back-compat tests.

## Example notebook

`3_real_materials.ipynb` — new, demonstrates:

1. **Material introspection**: `list_materials()`, plot `n(λ)`, `k(λ)` for SiO₂, TiO₂, Ag, Au across 0.40-0.80 μm.
2. **Anti-reflection coating**: broadband reflectance of `'air' | 'SiO2' | 'TiO2' | 'N-BK7'` stack.
3. **Surface plasmon resonance** with `'Ag'` — shows the absorption signature in `R(θ)` from non-zero k, validated against the existing `tmm_numpy` reference.
4. **(Optional)** Anisotropic layer with a uniaxial crystal if one ends up in `thin_film_materials.json`.

`1_forward_simu.ipynb` and `2_inverse_design.ipynb` are not modified — they continue to exercise the scalar API.

## Packaging

- `pyproject.toml`:
  - `[tool.setuptools.packages.find]` — add `difftmm.material` and `difftmm.material.catalogs` (or rely on `find` auto-discovery with the package marker).
  - Add `[tool.setuptools.package-data]` entry:
    ```toml
    [tool.setuptools.package-data]
    "difftmm.material" = ["catalogs/*.AGF", "catalogs/*.json"]
    ```
- `MANIFEST.in`: add `recursive-include difftmm/material/catalogs *.AGF *.json`.
- Update `README.md`:
  - "Real Materials" subsection with the string-API quick-start.
  - Update the "Repository Structure" tree.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Broadcasting bugs when extending TMM core from scalar `n_in` to per-wvln tensor | Tight broadcasting-axis tests; explicit `.shape` assertions in dev; regression tests against pre-change outputs. |
| `torch.searchsorted` + `torch.where` interpolation has subtle gradient holes at exact knot wavelengths | Standard linear-interp gradient passes through smoothly via the weighted-sum formulation already used in DeepLens. Reuse that pattern. |
| AGF mode-1 (Schott) entries silently disappearing surprises users coming from DeepLens | Log a single `INFO` line at catalog load: `"Skipped N Schott-mode entries (only Sellmeier supported in v1)."`. Document in README. |
| Catalog files inflate wheel size (~1 MB total) | Acceptable for a scientific package. Consider lazy loading later if needed (load each AGF on first lookup of a name in that catalog). |
| Material name collisions between AGF and JSON catalogs | Use DeepLens's merge order (`MISC < PLASTIC < CDGM < SCHOTT`) so SCHOTT wins; new `INTERP_NK_TABLE` is checked last, so user-curated thin-film materials only fill gaps. Document precedence. |
| Loss of `optimizable` mode confuses lens-design users coming from DeepLens | Out of scope; document the difference in README. Easy to add back as a v2 feature. |

## Implementation order (rough phasing)

1. Standalone `Material` class with Sellmeier + Interp + AGF/JSON loaders, no solver integration yet.
2. Curate `thin_film_materials.json` (~10 materials) from refractiveindex.info.
3. `resolve_indices` helper.
4. Refactor TMM core to accept per-wavelength `n_in` / `n_out` / `n_layers`.
5. Wire Material support into `IsotropicFilmSolver`.
6. Wire Material support into `FilmSolver` (including anisotropic 3-tuple).
7. Tests.
8. Notebook `3_real_materials.ipynb`.
9. Packaging metadata + README updates.

(Order will be detailed in the implementation plan produced by writing-plans.)
