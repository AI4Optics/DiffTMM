# Real Material Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add wavelength-dependent complex refractive index support (`n + ik`) to DiffTMM via a new `difftmm/material/` package, backed by bundled material catalogs, integrated into both solvers with full backward compatibility.

**Architecture:** Monolithic `Material` class with a `.dispersion` attribute switching between `sellmeier` (analytical) and `interp` (linear interpolation, supports complex n+ik). Solvers accept `float | complex | str | Material` per slot; strings auto-wrap to `Material(name)` eagerly at `__init__` for fail-fast validation. Refractive indices are materialized at `simulate(wvln)` time via a `resolve_indices` helper. TMM core functions are refactored to accept per-wavelength `n_in` / `n_out` / `n_layers` tensors.

**Tech Stack:** PyTorch (already a dependency). No new runtime dependencies — `numpy` is *not* added; `torch.searchsorted` + tensor ops replace `np.interp`.

**Reference:** Design spec at [docs/superpowers/specs/2026-05-26-real-material-support-design.md](docs/superpowers/specs/2026-05-26-real-material-support-design.md). Original implementation reference: [vccimaging/DeepLens — deeplens/material/materials.py](https://github.com/vccimaging/DeepLens/blob/main/deeplens/material/materials.py).

---

## File Structure

**Created:**

```
difftmm/material/
├── __init__.py                            # Public API re-exports
├── materials.py                           # Material class, loaders, resolve_indices (~450 LoC)
└── catalogs/
    ├── CDGM.AGF                           # Verbatim from DeepLens
    ├── SCHOTT.AGF                         # Verbatim from DeepLens
    ├── MISC.AGF                           # Verbatim from DeepLens
    ├── PLASTIC2022.AGF                    # Verbatim from DeepLens
    ├── materials_data.json                # Verbatim from DeepLens
    └── thin_film_materials.json           # NEW — n+k tables for thin-film materials

tests/
├── __init__.py
└── material/
    ├── __init__.py
    ├── conftest.py
    ├── test_material_class.py
    ├── test_solver_with_materials.py
    ├── test_anisotropic_materials.py
    └── test_backward_compat.py

3_real_materials.ipynb                     # NEW notebook (top-level alongside existing examples)
```

**Modified:**

- `difftmm/__init__.py` — re-export `Material`, `list_materials`, `MATERIAL_data`
- `difftmm/film_solver_isotropic.py` — accept str/Material/scalar in constructor; refactor TMM core
- `difftmm/film_solver_anisotropic.py` — accept str/Material/scalar/3-tuple in constructor; refactor TMM core
- `pyproject.toml` — package-data, packages.find
- `MANIFEST.in` — include catalog files
- `README.md` — Real Materials subsection + updated repo tree

---

## Phase 1 — Material module scaffolding

### Task 1: Create material package skeleton and bundle DeepLens catalog files

**Files:**
- Create: `difftmm/material/__init__.py`
- Create: `difftmm/material/materials.py`
- Create: `difftmm/material/catalogs/CDGM.AGF`
- Create: `difftmm/material/catalogs/SCHOTT.AGF`
- Create: `difftmm/material/catalogs/MISC.AGF`
- Create: `difftmm/material/catalogs/PLASTIC2022.AGF`
- Create: `difftmm/material/catalogs/materials_data.json`

- [ ] **Step 1: Create empty package and stub module**

```python
# difftmm/material/__init__.py
"""Material support for DiffTMM — wavelength-dependent refractive indices."""
```

```python
# difftmm/material/materials.py
"""Optical material with wavelength-dependent complex refractive index."""
```

- [ ] **Step 2: Download DeepLens catalog files**

Run from repo root:

```bash
mkdir -p difftmm/material/catalogs
for f in CDGM.AGF SCHOTT.AGF MISC.AGF PLASTIC2022.AGF materials_data.json; do
  curl -fsSL "https://raw.githubusercontent.com/vccimaging/DeepLens/main/deeplens/material/$f" \
    -o "difftmm/material/catalogs/$f"
done
ls -la difftmm/material/catalogs/
```

Expected: all five files present and non-empty.

- [ ] **Step 3: Verify files load cleanly**

Run:

```bash
python -c "import json; json.load(open('difftmm/material/catalogs/materials_data.json'))"
python -c "open('difftmm/material/catalogs/CDGM.AGF', encoding='utf-8', errors='ignore').read()[:100]"
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add difftmm/material/__init__.py difftmm/material/materials.py difftmm/material/catalogs/
git commit -m "feat(material): scaffold material package and bundle DeepLens catalogs"
```

---

### Task 2: Bundle thin-film n+k JSON with seed materials

**Files:**
- Create: `difftmm/material/catalogs/thin_film_materials.json`

- [ ] **Step 1: Create the seed JSON with SiO₂, TiO₂, Ag**

Write to `difftmm/material/catalogs/thin_film_materials.json`. Values below are abbreviated; the engineer should consult refractiveindex.info for full curated tables. The schema is what matters for v1 — additional materials (Ta₂O₅, MgF₂, Si, Au, Al, ITO) can be added in follow-up commits.

```json
{
  "_info": {
    "INTERP_NK_TABLE": "Wavelength (um), refractive index n, extinction coefficient k. References: https://refractiveindex.info/.",
    "schema_version": 1
  },
  "INTERP_NK_TABLE": {
    "SiO2": {
      "_source": "Malitson 1965, https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson",
      "wvlns": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
      "n":     [1.4701, 1.4656, 1.4623, 1.4601, 1.4580, 1.4565, 1.4553, 1.4542, 1.4533],
      "k":     [0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0]
    },
    "TiO2": {
      "_source": "Sarkar 2019 (rutile), https://refractiveindex.info/?shelf=main&book=TiO2&page=Sarkar",
      "wvlns": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
      "n":     [3.05, 2.78, 2.65, 2.58, 2.54, 2.51, 2.49, 2.47, 2.46],
      "k":     [0.10, 0.02, 0.005,0.001,0.0,  0.0,  0.0,  0.0,  0.0]
    },
    "Ag": {
      "_source": "Johnson and Christy 1972, https://refractiveindex.info/?shelf=main&book=Ag&page=Johnson",
      "wvlns": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
      "n":     [0.173, 0.141, 0.129, 0.123, 0.124, 0.130, 0.142, 0.157, 0.176],
      "k":     [1.952, 2.460, 2.880, 3.282, 3.661, 4.030, 4.397, 4.768, 5.139]
    }
  }
}
```

- [ ] **Step 2: Verify JSON parses**

Run:

```bash
python -c "import json; d=json.load(open('difftmm/material/catalogs/thin_film_materials.json')); print(list(d['INTERP_NK_TABLE'].keys()))"
```

Expected: `['SiO2', 'TiO2', 'Ag']`

- [ ] **Step 3: Commit**

```bash
git add difftmm/material/catalogs/thin_film_materials.json
git commit -m "feat(material): seed thin-film n+k catalog with SiO2, TiO2, Ag"
```

---

### Task 3: Set up tests/ directory

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/material/__init__.py`
- Create: `tests/material/conftest.py`

- [ ] **Step 1: Create empty package files**

```python
# tests/__init__.py
```

```python
# tests/material/__init__.py
```

- [ ] **Step 2: Create conftest with a CPU device fixture**

```python
# tests/material/conftest.py
import pytest
import torch


@pytest.fixture
def device():
    """CPU device — tests should not require CUDA."""
    return torch.device("cpu")


@pytest.fixture
def wvln_vis(device):
    """A visible-light wavelength vector in micrometers."""
    return torch.tensor([0.45, 0.55, 0.65], device=device)
```

- [ ] **Step 3: Verify pytest discovers the new test root**

Run:

```bash
python -m pytest tests/ --collect-only
```

Expected: `no tests ran` (no test files yet) with no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: scaffold tests/material/ with shared fixtures"
```

---

### Task 4: TDD `Material('air')` — non-dispersive base case

**Files:**
- Modify: `difftmm/material/materials.py`
- Modify: `difftmm/material/__init__.py`
- Create: `tests/material/test_material_class.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/material/test_material_class.py
import pytest
import torch

from difftmm.material import Material


class TestAirMaterial:
    def test_default_name_is_air(self):
        mat = Material()
        assert mat.name == "air"
        assert mat.dispersion == "sellmeier"

    @pytest.mark.parametrize("alias", ["air", "AIR", "vacuum", "occluder", "Air"])
    def test_air_aliases_normalize(self, alias):
        assert Material(alias).name == "air"

    def test_air_ior_returns_complex_one(self, wvln_vis):
        mat = Material("air")
        n = mat.ior(wvln_vis)
        assert n.dtype == torch.complex64
        assert n.shape == wvln_vis.shape
        torch.testing.assert_close(n.real, torch.ones_like(wvln_vis))
        torch.testing.assert_close(n.imag, torch.zeros_like(wvln_vis))

    def test_air_n_attribute_is_unity(self):
        assert Material("air").n == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/material/test_material_class.py::TestAirMaterial -v
```

Expected: FAIL with `ImportError` or `AttributeError` (no Material yet).

- [ ] **Step 3: Implement `Material` with air-only support**

Replace contents of `difftmm/material/materials.py`:

```python
"""Optical material with wavelength-dependent complex refractive index."""

from __future__ import annotations

import torch


_AIR_ALIASES = {"air", "vacuum", "occluder"}


class Material:
    """Optical material with wavelength-dependent complex refractive index.

    Attributes:
        name (str): Lowercase material name.
        dispersion (str): 'sellmeier' | 'interp'.
        n (float): Nominal refractive index at d-line (587 nm).
        V (float): Abbe number (1e38 for non-dispersive 'air').
    """

    def __init__(
        self,
        name: str | None = None,
        device: torch.device | str = "cpu",
    ):
        raw = "air" if name is None else name.strip().lower()
        self.name = "air" if raw in _AIR_ALIASES else raw
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self._load_dispersion()

    def _load_dispersion(self) -> None:
        if self.name == "air":
            self.dispersion = "sellmeier"
            self.k1 = self.l1 = self.k2 = self.l2 = self.k3 = self.l3 = 0.0
            self.n = 1.0
            self.V = 1e38
            return
        raise NotImplementedError(f"Material {self.name!r} not implemented.")

    def ior(self, wvln: torch.Tensor) -> torch.Tensor:
        """Compute the complex refractive index at given wavelengths.

        Args:
            wvln: real tensor of wavelengths in μm.
        Returns:
            torch.complex64 tensor with the same shape as `wvln`.
        """
        if self.dispersion == "sellmeier":
            wvln2 = wvln**2
            n2 = (
                1.0
                + self.k1 * wvln2 / (wvln2 - self.l1 + 1e-30)
                + self.k2 * wvln2 / (wvln2 - self.l2 + 1e-30)
                + self.k3 * wvln2 / (wvln2 - self.l3 + 1e-30)
            )
            n = torch.sqrt(torch.clamp(n2, min=1e-30))
            return (n + 0j).to(torch.complex64)
        raise NotImplementedError(f"Dispersion {self.dispersion!r} not implemented.")
```

And update the package `__init__.py`:

```python
# difftmm/material/__init__.py
"""Material support for DiffTMM — wavelength-dependent refractive indices."""

from .materials import Material

__all__ = ["Material"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/material/test_material_class.py::TestAirMaterial -v
```

Expected: all 7 (4 named + 3 parametrized variants of `test_air_aliases_normalize` cover 5 cases — total 7 PASSED counts).

- [ ] **Step 5: Commit**

```bash
git add difftmm/material/materials.py difftmm/material/__init__.py tests/material/test_material_class.py
git commit -m "feat(material): add Material class with air-only support"
```

---

### Task 5: TDD AGF file loader (Sellmeier entries only)

**Files:**
- Modify: `difftmm/material/materials.py`
- Modify: `tests/material/test_material_class.py`

- [ ] **Step 1: Write failing tests for AGF loading**

Append to `tests/material/test_material_class.py`:

```python
class TestAGFLoading:
    def test_n_bk7_loads_as_sellmeier(self):
        mat = Material("N-BK7")
        assert mat.dispersion == "sellmeier"
        # Six Sellmeier coefficients should be populated (non-zero)
        assert any(c != 0.0 for c in [mat.k1, mat.k2, mat.k3, mat.l1, mat.l2, mat.l3])

    def test_n_bk7_d_line_index(self):
        # Published nd for N-BK7 is 1.5168
        mat = Material("N-BK7")
        wvln = torch.tensor([0.5876])  # d-line
        n = mat.ior(wvln).real.item()
        assert abs(n - 1.5168) < 1e-3

    def test_case_insensitive_lookup(self):
        m1 = Material("N-BK7")
        m2 = Material("n-bk7")
        assert m1.n == m2.n

    def test_unknown_material_raises(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            Material("not_a_real_material_xyz")

    def test_schott_mode_entries_are_skipped(self):
        # PLASTIC2022.AGF entries are mostly mode=1 (Schott). They should NOT
        # be loadable as Sellmeier in v1.
        # PMMA is in materials_data.json's SCHOTT_TABLE only — not yet supported,
        # so it should raise.
        with pytest.raises(NotImplementedError):
            Material("PMMA")  # depends: must not match an AGF Sellmeier entry
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/material/test_material_class.py::TestAGFLoading -v
```

Expected: 5 FAIL (no AGF loader yet).

- [ ] **Step 3: Implement the AGF loader**

Add to `difftmm/material/materials.py` above the `Material` class:

```python
import os
import re

_CATALOGS_DIR = os.path.join(os.path.dirname(__file__), "catalogs")


def _read_agf(file_path: str) -> dict:
    """Parse an AGF catalog and return a dict of Sellmeier (mode=2) entries.

    Schott-mode (mode=1) entries are silently skipped — v1 only supports Sellmeier.
    """
    encodings = ("utf-8", "utf-16")
    lines: list[str] | None = None
    for enc in encodings:
        try:
            with open(file_path, encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        raise OSError(f"Could not read {file_path} with utf-8 or utf-16.")

    nm_lines = [ln for ln in lines if re.match(r"^NM\b", ln)]
    cd_lines = [ln for ln in lines if re.match(r"^CD\b", ln)]
    materials: dict = {}
    for nm, cd in zip(nm_lines, cd_lines):
        nm_parts = nm.split()
        cd_parts = cd.split()
        mode = float(nm_parts[2])
        if mode != 2:  # Skip non-Sellmeier
            continue
        materials[nm_parts[1].lower()] = {
            "k1": float(cd_parts[1]),
            "l1": float(cd_parts[2]),
            "k2": float(cd_parts[3]),
            "l2": float(cd_parts[4]),
            "k3": float(cd_parts[5]),
            "l3": float(cd_parts[6]),
            "nd": float(nm_parts[4]),
            "vd": float(nm_parts[5]),
        }
    return materials


def _load_all_agf() -> dict:
    """Merge all AGF Sellmeier entries. Precedence: MISC < PLASTIC < CDGM < SCHOTT."""
    files = ("MISC.AGF", "PLASTIC2022.AGF", "CDGM.AGF", "SCHOTT.AGF")
    merged: dict = {}
    for fname in files:
        path = os.path.join(_CATALOGS_DIR, fname)
        if os.path.exists(path):
            merged.update(_read_agf(path))
    return merged


_AGF_DATA: dict = _load_all_agf()
MATERIAL_data: dict = dict(_AGF_DATA)  # Public — exported via package __init__
```

And update `Material._load_dispersion`:

```python
def _load_dispersion(self) -> None:
    if self.name == "air":
        self.dispersion = "sellmeier"
        self.k1 = self.l1 = self.k2 = self.l2 = self.k3 = self.l3 = 0.0
        self.n = 1.0
        self.V = 1e38
        return

    if self.name in _AGF_DATA:
        entry = _AGF_DATA[self.name]
        self.dispersion = "sellmeier"
        self.k1 = entry["k1"]
        self.l1 = entry["l1"]
        self.k2 = entry["k2"]
        self.l2 = entry["l2"]
        self.k3 = entry["k3"]
        self.l3 = entry["l3"]
        self.n = entry["nd"]
        self.V = entry["vd"]
        return

    raise NotImplementedError(f"Material {self.name!r} not implemented.")
```

Update `difftmm/material/__init__.py`:

```python
"""Material support for DiffTMM — wavelength-dependent refractive indices."""

from .materials import MATERIAL_data, Material

__all__ = ["Material", "MATERIAL_data"]
```

- [ ] **Step 4: Run AGF tests**

Run:

```bash
python -m pytest tests/material/test_material_class.py -v
```

Expected: all tests PASS (including the earlier air tests).

- [ ] **Step 5: Commit**

```bash
git add difftmm/material/materials.py difftmm/material/__init__.py tests/material/test_material_class.py
git commit -m "feat(material): load Sellmeier entries from bundled AGF catalogs"
```

---

### Task 6: TDD JSON SELLMEIER_TABLE loader (DeepLens custom Sellmeier)

**Files:**
- Modify: `difftmm/material/materials.py`
- Modify: `tests/material/test_material_class.py`

- [ ] **Step 1: Write failing test**

Append to `tests/material/test_material_class.py`:

```python
class TestJSONSellmeier:
    def test_bk7_lowercase_from_json(self):
        # 'bk7' is in materials_data.json SELLMEIER_TABLE
        mat = Material("bk7")
        assert mat.dispersion == "sellmeier"
        wvln = torch.tensor([0.5876])
        n = mat.ior(wvln).real.item()
        assert abs(n - 1.5168) < 1e-3
```

- [ ] **Step 2: Run test to verify failure**

```bash
python -m pytest tests/material/test_material_class.py::TestJSONSellmeier -v
```

Expected: FAIL (only AGF lookup so far; `bk7` is in JSON but AGF has `N-BK7`).

- [ ] **Step 3: Implement JSON loading and SELLMEIER_TABLE branch**

Add to `difftmm/material/materials.py`:

```python
import json


def _read_json_catalog(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


_CUSTOM_DATA: dict = _read_json_catalog(
    os.path.join(_CATALOGS_DIR, "materials_data.json")
)
_SELLMEIER_TABLE: dict = _CUSTOM_DATA.get("SELLMEIER_TABLE", {})
_MATERIAL_TABLE: dict = _CUSTOM_DATA.get("MATERIAL_TABLE", {})
```

Update `Material._load_dispersion` (insert after AGF branch, before `raise`):

```python
    if self.name in _SELLMEIER_TABLE:
        coeffs = _SELLMEIER_TABLE[self.name]
        self.dispersion = "sellmeier"
        self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = coeffs
        nv = _MATERIAL_TABLE.get(self.name, [None, None])
        self.n = nv[0] if nv[0] is not None else 0.0
        self.V = nv[1] if nv[1] is not None else 1e38
        return
```

Extend `MATERIAL_data` at module level so it advertises JSON entries too:

```python
MATERIAL_data: dict = {**_AGF_DATA, **{k: {"source": "json"} for k in _SELLMEIER_TABLE}}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/material/test_material_class.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/material/materials.py tests/material/test_material_class.py
git commit -m "feat(material): load custom Sellmeier entries from materials_data.json"
```

---

### Task 7: TDD interp dispersion — real n only

**Files:**
- Modify: `difftmm/material/materials.py`
- Modify: `tests/material/test_material_class.py`

- [ ] **Step 1: Write failing test**

Append to `tests/material/test_material_class.py`:

```python
class TestInterpRealN:
    def test_fused_silica_at_d_line(self):
        # 'fused_silica' is in materials_data.json INTERP_TABLE
        mat = Material("fused_silica")
        assert mat.dispersion == "interp"
        wvln = torch.tensor([0.5876])
        n = mat.ior(wvln)
        # At 0.5876, table values bracket; the result should be a real value ~1.46
        assert n.dtype == torch.complex64
        assert abs(n.real.item() - 1.4596) < 0.005
        assert n.imag.item() == pytest.approx(0.0)

    def test_interp_vector_output(self, wvln_vis):
        mat = Material("fused_silica")
        n = mat.ior(wvln_vis)
        assert n.shape == wvln_vis.shape
        assert n.dtype == torch.complex64

    def test_interp_endpoints_match_table(self):
        mat = Material("fused_silica")
        # First table point is (0.40, 1.4701)
        wvln = torch.tensor([0.40])
        assert abs(mat.ior(wvln).real.item() - 1.4701) < 1e-4
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/material/test_material_class.py::TestInterpRealN -v
```

Expected: 3 FAIL (no interp support yet).

- [ ] **Step 3: Implement interp dispersion (real n only path)**

Add to `difftmm/material/materials.py`:

```python
_INTERP_TABLE: dict = _CUSTOM_DATA.get("INTERP_TABLE", {})


def _linear_interp_complex(
    wvln: torch.Tensor,
    ref_wvlns: torch.Tensor,
    ref_n: torch.Tensor,
    ref_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable linear interpolation of (n, k) into complex output."""
    num_pts = ref_wvlns.numel()
    i = torch.searchsorted(ref_wvlns, wvln, side="right")
    idx_low = torch.clamp(i - 1, 0, num_pts - 1)
    idx_high = torch.clamp(i, 0, num_pts - 1)

    w_low = ref_wvlns[idx_low]
    w_high = ref_wvlns[idx_high]
    n_low = ref_n[idx_low]
    n_high = ref_n[idx_high]

    denom = w_high - w_low
    has_interval = denom != 0
    safe_denom = torch.where(has_interval, denom, torch.ones_like(denom))
    weight_high = torch.where(
        has_interval, (wvln - w_low) / safe_denom, torch.zeros_like(wvln)
    )
    weight_low = 1.0 - weight_high
    n_real = n_low * weight_low + n_high * weight_high
    if ref_k is not None:
        k_low = ref_k[idx_low]
        k_high = ref_k[idx_high]
        k_real = k_low * weight_low + k_high * weight_high
    else:
        k_real = torch.zeros_like(n_real)
    return torch.complex(n_real, k_real).to(torch.complex64)
```

Extend `Material._load_dispersion` (after `_SELLMEIER_TABLE` branch):

```python
    if self.name in _INTERP_TABLE:
        entry = _INTERP_TABLE[self.name]
        self.dispersion = "interp"
        self._ref_wvlns = torch.tensor(entry["wvlns"], dtype=torch.float32)
        self._ref_n = torch.tensor(entry["n"], dtype=torch.float32)
        self._ref_k = None
        # Compute nd, V from the table for completeness
        d_wvln = torch.tensor([0.5876])
        F_wvln = torch.tensor([0.4861])
        C_wvln = torch.tensor([0.6563])
        nd = _linear_interp_complex(d_wvln, self._ref_wvlns, self._ref_n).real.item()
        nF = _linear_interp_complex(F_wvln, self._ref_wvlns, self._ref_n).real.item()
        nC = _linear_interp_complex(C_wvln, self._ref_wvlns, self._ref_n).real.item()
        self.n = nd
        self.V = (nd - 1) / (nF - nC) if nF != nC else 1e38
        return
```

Extend `Material.ior` (after sellmeier branch):

```python
        if self.dispersion == "interp":
            self._ref_wvlns = self._ref_wvlns.to(wvln.device)
            self._ref_n = self._ref_n.to(wvln.device)
            ref_k = self._ref_k.to(wvln.device) if self._ref_k is not None else None
            return _linear_interp_complex(wvln, self._ref_wvlns, self._ref_n, ref_k)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/material/test_material_class.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/material/materials.py tests/material/test_material_class.py
git commit -m "feat(material): add linear-interp dispersion with real-n table support"
```

---

### Task 8: TDD interp dispersion with n+k from thin_film_materials.json

**Files:**
- Modify: `difftmm/material/materials.py`
- Modify: `tests/material/test_material_class.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/material/test_material_class.py`:

```python
class TestInterpNK:
    def test_sio2_zero_extinction(self, wvln_vis):
        mat = Material("SiO2")
        assert mat.dispersion == "interp"
        n = mat.ior(wvln_vis)
        torch.testing.assert_close(n.imag, torch.zeros_like(wvln_vis))

    def test_silver_has_extinction(self):
        mat = Material("Ag")
        n = mat.ior(torch.tensor([0.55]))
        assert n.imag.item() > 0.5  # k is large for Ag in visible

    def test_silver_lowercase_lookup(self):
        # NK_TABLE materials are looked up case-insensitively
        m1 = Material("Ag")
        m2 = Material("ag")
        assert m1.dispersion == m2.dispersion
        # Same table => identical output
        wvln = torch.tensor([0.55])
        torch.testing.assert_close(m1.ior(wvln), m2.ior(wvln))
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/material/test_material_class.py::TestInterpNK -v
```

Expected: 3 FAIL.

- [ ] **Step 3: Implement n+k loader**

Add to `difftmm/material/materials.py`:

```python
_THINFILM_DATA: dict = _read_json_catalog(
    os.path.join(_CATALOGS_DIR, "thin_film_materials.json")
)
# Build a case-insensitive lookup map name_lower -> (orig_name, entry)
_INTERP_NK_TABLE: dict = {
    k.lower(): v for k, v in _THINFILM_DATA.get("INTERP_NK_TABLE", {}).items()
}
```

Extend `Material._load_dispersion` (after `_INTERP_TABLE` branch):

```python
    if self.name in _INTERP_NK_TABLE:
        entry = _INTERP_NK_TABLE[self.name]
        self.dispersion = "interp"
        self._ref_wvlns = torch.tensor(entry["wvlns"], dtype=torch.float32)
        self._ref_n = torch.tensor(entry["n"], dtype=torch.float32)
        self._ref_k = torch.tensor(entry["k"], dtype=torch.float32)
        d_wvln = torch.tensor([0.5876])
        F_wvln = torch.tensor([0.4861])
        C_wvln = torch.tensor([0.6563])
        nd = _linear_interp_complex(d_wvln, self._ref_wvlns, self._ref_n).real.item()
        nF = _linear_interp_complex(F_wvln, self._ref_wvlns, self._ref_n).real.item()
        nC = _linear_interp_complex(C_wvln, self._ref_wvlns, self._ref_n).real.item()
        self.n = nd
        self.V = (nd - 1) / (nF - nC) if nF != nC else 1e38
        return
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/material/test_material_class.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/material/materials.py tests/material/test_material_class.py
git commit -m "feat(material): add n+k interp loader from thin_film_materials.json"
```

---

### Task 9: TDD `Material.to(device)` and autograd

**Files:**
- Modify: `difftmm/material/materials.py`
- Modify: `tests/material/test_material_class.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/material/test_material_class.py`:

```python
class TestMaterialDeviceAndGrad:
    def test_to_device_moves_cached_tensors(self):
        mat = Material("SiO2")
        out_cpu = mat.ior(torch.tensor([0.55])).device
        assert out_cpu.type == "cpu"
        # to() should be a no-op for cpu->cpu
        mat.to("cpu")
        assert mat._ref_wvlns.device.type == "cpu"

    def test_autograd_through_sellmeier(self):
        mat = Material("N-BK7")
        wvln = torch.tensor([0.55], requires_grad=True)
        n = mat.ior(wvln)
        n.real.sum().backward()
        assert wvln.grad is not None
        assert torch.isfinite(wvln.grad).all()

    def test_autograd_through_interp(self):
        mat = Material("SiO2")
        wvln = torch.tensor([0.55], requires_grad=True)
        n = mat.ior(wvln)
        n.real.sum().backward()
        assert wvln.grad is not None
        assert torch.isfinite(wvln.grad).all()
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/material/test_material_class.py::TestMaterialDeviceAndGrad -v
```

Expected: 1 PASS (autograd works already), 2 FAIL (no `to()` method).

- [ ] **Step 3: Implement `to()`**

Add to `Material`:

```python
    def to(self, device: torch.device | str) -> "Material":
        """Move cached interpolation tensors to the given device.

        Returns self for chaining.
        """
        device = torch.device(device) if not isinstance(device, torch.device) else device
        self.device = device
        if hasattr(self, "_ref_wvlns") and self._ref_wvlns is not None:
            self._ref_wvlns = self._ref_wvlns.to(device)
            self._ref_n = self._ref_n.to(device)
            if self._ref_k is not None:
                self._ref_k = self._ref_k.to(device)
        return self
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/material/test_material_class.py::TestMaterialDeviceAndGrad -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/material/materials.py tests/material/test_material_class.py
git commit -m "feat(material): add Material.to(device) + verify autograd"
```

---

### Task 10: `list_materials()` and `refractive_index()` helpers

**Files:**
- Modify: `difftmm/material/materials.py`
- Modify: `difftmm/material/__init__.py`
- Modify: `tests/material/test_material_class.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/material/test_material_class.py`:

```python
class TestHelpers:
    def test_list_materials_returns_known_names(self):
        from difftmm.material import list_materials
        names = list_materials()
        assert "air" in names
        assert "n-bk7" in names  # from AGF
        assert "sio2" in names  # from thin_film_materials INTERP_NK_TABLE (case-folded)

    def test_refractive_index_scalar_input(self):
        mat = Material("air")
        result = mat.refractive_index(0.55)
        assert isinstance(result, complex)
        assert result.real == pytest.approx(1.0)

    def test_refractive_index_tensor_input(self, wvln_vis):
        mat = Material("air")
        result = mat.refractive_index(wvln_vis)
        assert torch.is_tensor(result)
        assert result.dtype == torch.complex64
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/material/test_material_class.py::TestHelpers -v
```

Expected: 3 FAIL.

- [ ] **Step 3: Implement helpers**

Add to `difftmm/material/materials.py` (at module level):

```python
def list_materials() -> list[str]:
    """Return all known material names from all bundled catalogs (sorted)."""
    names = set()
    names.add("air")
    names.update(_AGF_DATA.keys())
    names.update(_SELLMEIER_TABLE.keys())
    names.update(_INTERP_TABLE.keys())
    names.update(_INTERP_NK_TABLE.keys())
    return sorted(names)
```

Add `refractive_index` method to `Material`:

```python
    def refractive_index(self, wvln):
        """Return the complex refractive index.

        Args:
            wvln: float (returns Python complex) or tensor (returns complex tensor).
        """
        if isinstance(wvln, (int, float)):
            t = torch.tensor([float(wvln)], device=self.device)
            return complex(self.ior(t).item())
        return self.ior(wvln)
```

Update `__init__.py`:

```python
"""Material support for DiffTMM — wavelength-dependent refractive indices."""

from .materials import MATERIAL_data, Material, list_materials

__all__ = ["Material", "MATERIAL_data", "list_materials"]
```

- [ ] **Step 4: Run all material tests**

```bash
python -m pytest tests/material/test_material_class.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/material/materials.py difftmm/material/__init__.py tests/material/test_material_class.py
git commit -m "feat(material): add list_materials() and refractive_index() wrapper"
```

---

## Phase 2 — Solver integration

### Task 11: TDD `resolve_indices` helper

**Files:**
- Modify: `difftmm/material/materials.py`
- Modify: `difftmm/material/__init__.py`
- Create: `tests/material/test_resolve_indices.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/material/test_resolve_indices.py
import pytest
import torch

from difftmm.material import Material, resolve_indices


@pytest.fixture
def wvln():
    return torch.tensor([0.45, 0.55, 0.65])


class TestResolveIndices:
    def test_float_spec_broadcasts(self, wvln):
        out = resolve_indices(1.5, wvln, torch.device("cpu"))
        assert out.shape == (3,)
        assert out.dtype == torch.complex64
        torch.testing.assert_close(out.real, torch.full((3,), 1.5))
        torch.testing.assert_close(out.imag, torch.zeros(3))

    def test_complex_spec_broadcasts(self, wvln):
        out = resolve_indices(1.5 + 0.01j, wvln, torch.device("cpu"))
        torch.testing.assert_close(out.imag, torch.full((3,), 0.01))

    def test_string_spec_creates_material(self, wvln):
        out = resolve_indices("SiO2", wvln, torch.device("cpu"))
        assert out.dtype == torch.complex64
        # SiO2 should have n ~1.46 in visible
        assert (out.real > 1.4).all() and (out.real < 1.5).all()

    def test_material_spec_calls_ior(self, wvln):
        mat = Material("air")
        out = resolve_indices(mat, wvln, torch.device("cpu"))
        torch.testing.assert_close(out.real, torch.ones(3))

    def test_tuple_spec_selects_axis(self, wvln):
        spec = (1.0, 1.5, 2.0)
        out_x = resolve_indices(spec, wvln, torch.device("cpu"), axis=0)
        out_y = resolve_indices(spec, wvln, torch.device("cpu"), axis=1)
        out_z = resolve_indices(spec, wvln, torch.device("cpu"), axis=2)
        assert out_x.real[0].item() == 1.0
        assert out_y.real[0].item() == 1.5
        assert out_z.real[0].item() == 2.0
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/material/test_resolve_indices.py -v
```

Expected: 5 FAIL (no `resolve_indices` yet).

- [ ] **Step 3: Implement `resolve_indices`**

Add to `difftmm/material/materials.py`:

```python
def resolve_indices(
    spec,
    wvln: torch.Tensor,
    device: torch.device | str = "cpu",
    *,
    axis: int | None = None,
) -> torch.Tensor:
    """Resolve a refractive-index spec at given wavelengths.

    Args:
        spec: one of:
            - float, complex, 0-d torch.Tensor (wavelength-independent).
            - str: looked up via Material(spec).
            - Material: `.ior(wvln)` is called.
            - 3-tuple of any of the above (anisotropic; `axis` selects element).
        wvln: real 1-D tensor of wavelengths in μm.
        device: target torch device.
        axis: For 3-tuple specs, which axis (0, 1, 2). Ignored otherwise.

    Returns:
        torch.complex64 tensor of shape `(n_wvlns,)`.
    """
    if isinstance(spec, tuple):
        if axis is None or not (0 <= axis < 3):
            raise ValueError(f"axis must be 0, 1, or 2 for tuple spec, got {axis!r}")
        if len(spec) != 3:
            raise ValueError(f"anisotropic spec must be a 3-tuple, got len {len(spec)}")
        return resolve_indices(spec[axis], wvln, device)

    if isinstance(spec, Material):
        return spec.to(device).ior(wvln)

    if isinstance(spec, str):
        return Material(spec, device=device).ior(wvln)

    # Scalar paths
    if isinstance(spec, (int, float, complex)):
        val = complex(spec)
        return torch.full(wvln.shape, val, dtype=torch.complex64, device=wvln.device)

    if torch.is_tensor(spec):
        if spec.dim() == 0:
            val = complex(spec.item())
            return torch.full(wvln.shape, val, dtype=torch.complex64, device=wvln.device)
        # Per-wvln tensor — must already match shape
        if spec.shape != wvln.shape:
            raise ValueError(f"tensor spec shape {spec.shape} != wvln shape {wvln.shape}")
        return spec.to(torch.complex64).to(wvln.device)

    raise TypeError(f"Unsupported refractive-index spec type: {type(spec).__name__}")
```

Update `__init__.py`:

```python
"""Material support for DiffTMM — wavelength-dependent refractive indices."""

from .materials import MATERIAL_data, Material, list_materials, resolve_indices

__all__ = ["Material", "MATERIAL_data", "list_materials", "resolve_indices"]
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/material/test_resolve_indices.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/material/materials.py difftmm/material/__init__.py tests/material/test_resolve_indices.py
git commit -m "feat(material): add resolve_indices helper for solver dispatch"
```

---

### Task 12: Refactor `create_jones_matrix_isotropic` to accept per-wavelength tensors

**Files:**
- Modify: `difftmm/film_solver_isotropic.py:196-357`
- Create: `tests/material/test_isotropic_core_refactor.py`

- [ ] **Step 1: Write a regression test against the current scalar behavior**

```python
# tests/material/test_isotropic_core_refactor.py
import pytest
import torch

from difftmm.film_solver_isotropic import create_jones_matrix_isotropic


@pytest.fixture
def stack():
    """A small isotropic stack: air | TiO2 | SiO2 | glass."""
    n_layers = torch.tensor([[2.4 + 0j, 1.46 + 0j]], dtype=torch.complex64)
    d = torch.tensor([[0.06, 0.10]], dtype=torch.complex64)
    wv = torch.tensor([[0.45, 0.55, 0.65]], dtype=torch.float32)
    theta = torch.tensor([[0.0, 0.2, 0.4]], dtype=torch.float32)
    return n_layers, d, wv, theta


def test_per_wvln_scalar_n_in_n_out_matches_old_behavior(stack):
    """Passing n_in / n_out as a (batch, n_wvln) tensor of constants should
    yield identical results to passing them as Python floats (back-compat)."""
    n_layers, d, wv, theta = stack
    ts1, tp1, rs1, rp1 = create_jones_matrix_isotropic(n_layers, d, wv, 1.0, 1.52, theta)

    # Pass n_in / n_out as broadcasted complex tensors of shape (batch, n_wvln)
    n_in_t = torch.full(wv.shape, 1.0 + 0j, dtype=torch.complex64)
    n_out_t = torch.full(wv.shape, 1.52 + 0j, dtype=torch.complex64)
    ts2, tp2, rs2, rp2 = create_jones_matrix_isotropic(n_layers, d, wv, n_in_t, n_out_t, theta)

    torch.testing.assert_close(ts1, ts2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rp1, rp2, rtol=1e-5, atol=1e-6)


def test_per_wvln_dispersive_n_layers_supported(stack):
    """n_layers shape (batch, n_wvln, n_layer) — dispersive per layer."""
    n_layers_static, d, wv, theta = stack
    # Build dispersive: same value for all wvln (should match static result)
    n_layers_dispersive = n_layers_static.unsqueeze(1).expand(-1, wv.shape[1], -1)

    ts1, tp1, rs1, rp1 = create_jones_matrix_isotropic(n_layers_static, d, wv, 1.0, 1.52, theta)
    ts2, tp2, rs2, rp2 = create_jones_matrix_isotropic(n_layers_dispersive, d, wv, 1.0, 1.52, theta)

    torch.testing.assert_close(ts1, ts2, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/material/test_isotropic_core_refactor.py -v
```

Expected: first test PASS (current code accepts floats), second test FAIL (current code rejects 3D `n_layers_1d`).

- [ ] **Step 3: Refactor `create_jones_matrix_isotropic`**

Open `difftmm/film_solver_isotropic.py`. Replace the function signature and body of `_compute_isotropic_tmm` and `create_jones_matrix_isotropic` to accept the new shapes. Key changes:

**In `_compute_isotropic_tmm` (lines 54-193):**

The function already accepts `n_layers` of shape `(batch, wv, angles, layer)` and `n_in_t`, `n_out_t` as scalars. We need to extend `n_in_t` / `n_out_t` to per-wavelength.

Replace the parameter docstring near line 78:

```python
        n_in_t, n_out_t: incident/output refractive indices.
            Either scalar complex tensors (broadcast across all wvlns) OR
            tensors of shape (batch, wv, 1, 1) broadcastable per-wavelength.
```

No structural change needed inside `_compute_isotropic_tmm` — the math broadcasts already.

**In `create_jones_matrix_isotropic` (lines 196-357):**

Change the docstring and add input normalization at the top. Replace lines `230-232`:

```python
    n_in_t = torch.tensor(n_in, dtype=dtype, device=device)
    n_out_t = torch.tensor(n_out, dtype=dtype, device=device)
```

with:

```python
    # Normalize n_in / n_out to (batch, num_wv, 1, 1) complex tensors
    def _to_per_wvln(x):
        if torch.is_tensor(x) and x.dim() == 2:
            # Already (batch, num_wv)
            return x.to(dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
        if torch.is_tensor(x) and x.dim() == 0:
            x = x.item()
        # Scalar (Python or 0-d)
        return torch.tensor(complex(x), dtype=dtype, device=device).view(1, 1, 1, 1)

    n_in_t = _to_per_wvln(n_in)
    n_out_t = _to_per_wvln(n_out)
```

Then in the symmetric fast-path test at line 240, replace:

```python
    if abs(n_in - n_out) < 1e-10:
```

with:

```python
    # Fast path only when both media are scalar and equal
    is_scalar_pair = (
        not torch.is_tensor(n_in) and not torch.is_tensor(n_out)
        and abs(n_in - n_out) < 1e-10
    )
    if is_scalar_pair:
```

Adjust `n_layers_1d` handling at lines 245-247 and 288-290 to support either `(batch, n_layer)` or `(batch, n_wvln, n_layer)`:

Replace the per-layer expansion block (line 245):

```python
    n_layers = n_layers_1d.unsqueeze(1).unsqueeze(2).to(dtype)
```

with:

```python
    # Normalize n_layers to (batch, num_wv, 1, num_layer) complex
    if n_layers_1d.dim() == 2:
        # (batch, num_layer) — wavelength-independent
        n_layers = n_layers_1d.unsqueeze(1).unsqueeze(2).to(dtype=dtype, device=device)
    elif n_layers_1d.dim() == 3:
        # (batch, num_wv, num_layer) — wavelength-dependent
        n_layers = n_layers_1d.unsqueeze(2).to(dtype=dtype, device=device)
    else:
        raise ValueError(f"n_layers_1d must be 2-D or 3-D, got shape {n_layers_1d.shape}")
```

Apply the same change in the forward-only branch around line 288 and the reverse branch around line 329.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/material/test_isotropic_core_refactor.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Run existing benchmarks to confirm back-compat**

```bash
python benchmarks/1_compare_angle_response_isotropic.py
```

Expected: runs without errors, comparison plot regenerated (visual diff with main is OK).

- [ ] **Step 6: Commit**

```bash
git add difftmm/film_solver_isotropic.py tests/material/test_isotropic_core_refactor.py
git commit -m "refactor(isotropic): accept per-wavelength n_in/n_out/n_layers tensors"
```

---

### Task 13: Wire `IsotropicFilmSolver` to accept str/Material

**Files:**
- Modify: `difftmm/film_solver_isotropic.py:363-end`
- Create: `tests/material/test_solver_with_materials.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/material/test_solver_with_materials.py
import pytest
import torch

from difftmm import IsotropicFilmSolver, FilmSolver
from difftmm.material import Material


@pytest.fixture
def cpu():
    return torch.device("cpu")


class TestIsotropicSolverMaterials:
    def test_string_inputs(self, cpu):
        solver = IsotropicFilmSolver(
            mat_n_in="air",
            mat_n_out="N-BK7",
            mat_n_ls=["TiO2", "SiO2"],
            thickness_ls=[0.06, 0.10],
            device=cpu,
        )
        theta = torch.tensor([0.0])
        wvln = torch.tensor([0.55])
        ts, tp, rs, rp = solver.simulate(theta=theta, wvln=wvln)
        assert ts.shape == (1, 1, 1)
        assert torch.isfinite(ts).all()

    def test_mixed_scalar_and_material(self, cpu):
        solver = IsotropicFilmSolver(
            mat_n_in=1.0,
            mat_n_out=1.52,
            mat_n_ls=[2.4, Material("SiO2", device=cpu), "TiO2"],
            thickness_ls=[0.06, 0.10, 0.06],
            device=cpu,
        )
        ts, tp, rs, rp = solver.simulate(
            theta=torch.tensor([0.1]),
            wvln=torch.tensor([0.45, 0.55, 0.65]),
        )
        assert ts.shape == (1, 3, 1)

    def test_scalar_only_still_works(self, cpu):
        solver = IsotropicFilmSolver(
            mat_n_in=1.0,
            mat_n_out=1.52,
            mat_n_ls=[2.1, 1.46, 2.1],
            thickness_ls=[0.08, 0.12, 0.08],
            device=cpu,
        )
        ts, _, _, _ = solver.simulate(
            theta=torch.tensor([0.1]),
            wvln=torch.tensor([0.55]),
        )
        assert torch.isfinite(ts).all()

    def test_unknown_material_name_fails_fast(self, cpu):
        with pytest.raises(NotImplementedError):
            IsotropicFilmSolver(
                mat_n_in="air",
                mat_n_out=1.52,
                mat_n_ls=["NotAMaterial"],
                thickness_ls=[0.1],
                device=cpu,
            )
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/material/test_solver_with_materials.py::TestIsotropicSolverMaterials -v
```

Expected: most FAIL — solver currently rejects str / Material in `mat_n_ls`.

- [ ] **Step 3: Refactor `IsotropicFilmSolver.__init__` and `simulate`**

Open `difftmm/film_solver_isotropic.py`. Replace the `__init__` method (line 372) to accept new types:

```python
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
        from .material import Material

        self.batch_size = batch_size
        self.device = device

        # Normalize spec inputs:
        # - str -> Material(str), validated eagerly
        # - Material -> as-is
        # - scalar (int/float/complex) -> as-is
        def _normalize(spec):
            if isinstance(spec, str):
                return Material(spec, device=device)
            return spec

        self._n_in_spec = _normalize(mat_n_in)
        self._n_out_spec = _normalize(mat_n_out)

        # Layer list normalization: tuple-of-3 is anisotropic — REJECT for 2x2 solver
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

        # Back-compat scalar refract_idx_layers (only if all specs are scalar)
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

        # Numeric back-compat attributes for save_ckpt (only valid for all-scalar)
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

        # Thickness setup (unchanged from existing logic)
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
```

Now update `simulate` (line 492). Replace the body starting at line 526:

```python
    def simulate(self, theta, wvln):
        from .material import resolve_indices

        # Handle theta input (unchanged)
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        theta = theta.to(self.device)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand(self.batch_size, -1)

        # Handle wavelength input (unchanged)
        if torch.is_tensor(wvln):
            wv = wvln.to(self.device)
            if wv.dim() == 0:
                wv = wv.unsqueeze(0)
        elif isinstance(wvln, (list, tuple)):
            wv = torch.tensor(wvln, dtype=torch.float32, device=self.device)
        else:
            wv = torch.tensor([wvln], dtype=torch.float32, device=self.device)
        wv_batch = wv.unsqueeze(0).expand(self.batch_size, -1)

        # NEW: resolve refractive indices at this wvln
        n_in_t = resolve_indices(self._n_in_spec, wv, self.device).unsqueeze(0).expand(
            self.batch_size, -1
        )
        n_out_t = resolve_indices(self._n_out_spec, wv, self.device).unsqueeze(0).expand(
            self.batch_size, -1
        )
        # Stack per-layer: shape (n_wvln, n_layer), then add batch
        per_layer = [resolve_indices(s, wv, self.device) for s in self._n_layer_specs]
        n_layers_t = torch.stack(per_layer, dim=-1)  # (n_wvln, n_layer)
        n_layers_t = n_layers_t.unsqueeze(0).expand(self.batch_size, -1, -1)

        d_batch = self.get_film_thickness()
        ts, tp, rs, rp = create_jones_matrix_isotropic(
            n_layers_t, d_batch, wv_batch, n_in_t, n_out_t, theta
        )
        return ts, tp, rs, rp
```

- [ ] **Step 4: Make `save_ckpt` / `load_ckpt` persist material names**

In `difftmm/film_solver_isotropic.py`, add a helper near the top of the file (right after `inv_sigmoid`):

```python
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
    return value  # complex / float
```

Replace the body of `save_ckpt` (line 458) with:

```python
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
            # Back-compat scalar-only fields (None when material objects present)
            "n_in":  self.mat_n_in,
            "n_out": self.mat_n_out,
            "refract_idx_layers": (
                self.refract_idx_layers.cpu()
                if self.refract_idx_layers is not None
                else None
            ),
        }
        torch.save(payload, save_path)
```

Replace the body of `load_ckpt` (line 442) with:

```python
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

        # If new-format spec metadata is present, restore it
        if "layer_specs" in ckpt:
            self._n_in_spec = _deserialize_spec(ckpt["n_in_spec"], self.device)
            self._n_out_spec = _deserialize_spec(ckpt["n_out_spec"], self.device)
            self._n_layer_specs = [
                _deserialize_spec(v, self.device) for v in ckpt["layer_specs"]
            ]
```

Add tests in `tests/material/test_solver_with_materials.py`:

```python
class TestIsotropicCheckpoint:
    def test_roundtrip_material_stack(self, tmp_path, cpu):
        path = tmp_path / "ckpt.pt"
        solver1 = IsotropicFilmSolver(
            mat_n_in="air",
            mat_n_out="N-BK7",
            mat_n_ls=["TiO2", "SiO2"],
            thickness_ls=[0.06, 0.10],
            device=cpu,
        )
        solver1.save_ckpt(path)
        # Reconstruct with the same material spec (load_ckpt restores thicknesses)
        solver2 = IsotropicFilmSolver(
            mat_n_in="air",
            mat_n_out="N-BK7",
            mat_n_ls=["TiO2", "SiO2"],
            thickness_ls=[0.001, 0.001],  # placeholder, will be overwritten
            device=cpu,
        )
        solver2.load_ckpt(path)
        torch.testing.assert_close(
            solver1.get_film_thickness(), solver2.get_film_thickness()
        )
        # The deserialized layer specs should be Material objects with the right names
        from difftmm.material import Material
        assert isinstance(solver2._n_layer_specs[0], Material)
        assert solver2._n_layer_specs[0].name == "tio2"

    def test_roundtrip_scalar_stack(self, tmp_path, cpu):
        path = tmp_path / "ckpt.pt"
        solver = IsotropicFilmSolver(
            mat_n_in=1.0,
            mat_n_out=1.52,
            mat_n_ls=[2.10, 1.46, 2.10],
            thickness_ls=[0.08, 0.12, 0.08],
            device=cpu,
        )
        solver.save_ckpt(path)
        solver.load_ckpt(path)  # should not raise
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/material/test_solver_with_materials.py tests/material/test_isotropic_core_refactor.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add difftmm/film_solver_isotropic.py tests/material/test_solver_with_materials.py
git commit -m "feat(isotropic): accept str / Material / scalar mix in mat_n_ls"
```

---

### Task 14: Refactor `create_jones_matrix_AOIAz` to accept per-wavelength tensors

**Files:**
- Modify: `difftmm/film_solver_anisotropic.py:361-487`
- Create: `tests/material/test_anisotropic_core_refactor.py`

- [ ] **Step 1: Write failing test**

```python
# tests/material/test_anisotropic_core_refactor.py
import torch

from difftmm.film_solver_anisotropic import create_jones_matrix_AOIAz


def test_per_wvln_dispersive_n_2d():
    """n_2d shape (batch, n_wvln, n_layer, 3) — dispersive per layer."""
    batch, n_wv, n_layer = 1, 3, 2
    a_2d = torch.zeros((batch, n_layer, 3), dtype=torch.complex64)
    n_static = torch.tensor(
        [[[1.46, 1.46, 1.46], [2.4, 2.4, 2.4]]], dtype=torch.complex64
    )  # (batch, n_layer, 3)
    d = torch.tensor([[0.10, 0.06]], dtype=torch.complex64)
    wv = torch.tensor([[0.45, 0.55, 0.65]], dtype=torch.float32)
    th_x = torch.tensor([[0.1]], dtype=torch.float32)
    th_y = torch.tensor([[0.0]], dtype=torch.float32)

    # Static path
    Jt1, Jr1 = create_jones_matrix_AOIAz(a_2d, n_static, d, wv, 1.0, 1.52, th_x, th_y)

    # Dispersive path — same value across wvln
    n_dispersive = n_static.unsqueeze(1).expand(-1, n_wv, -1, -1)
    Jt2, Jr2 = create_jones_matrix_AOIAz(a_2d, n_dispersive, d, wv, 1.0, 1.52, th_x, th_y)

    torch.testing.assert_close(Jt1, Jt2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(Jr1, Jr2, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/material/test_anisotropic_core_refactor.py -v
```

Expected: FAIL — current code only accepts `n_2d` of shape `(batch, n_layer, 3)`.

- [ ] **Step 3: Refactor `create_jones_matrix_AOIAz`**

Open `difftmm/film_solver_anisotropic.py`. The core change: allow `n_2d` to be either `(batch, n_layer, 3)` or `(batch, n_wvln, n_layer, 3)`.

Around line 399 (the `ng_1d` computation):

```python
    ng_1d = torch.sqrt(
        (n_2d[:, :, 0] ** 2 + n_2d[:, :, 1] ** 2 + n_2d[:, :, 2] ** 2) / 3
    )
```

Add a normalization helper at the top of the function (after `device = a_2d.device`):

```python
    # Normalize n_2d to (batch, num_wv, n_layer, 3)
    if n_2d.dim() == 3:
        # (batch, n_layer, 3) — broadcast across wvlns
        n_2d_w = n_2d.unsqueeze(1).expand(-1, num_wv, -1, -1)
    elif n_2d.dim() == 4:
        # (batch, num_wv, n_layer, 3) — already dispersive
        n_2d_w = n_2d
    else:
        raise ValueError(f"n_2d must be 3-D or 4-D, got shape {n_2d.shape}")
```

Then replace `ng_1d` and downstream uses to use `n_2d_w`:

```python
    # ng now has shape (batch, num_wv, n_layer)
    ng_4d = torch.sqrt(
        (n_2d_w[..., 0] ** 2 + n_2d_w[..., 1] ** 2 + n_2d_w[..., 2] ** 2) / 3
    )
```

Adjust the existing `ng_3d` expansion. Replace:

```python
    ng_3d = ng_1d.reshape((batchsize, 1, 1, -1)).expand(-1, num_x, num_y, -1)
```

with:

```python
    # ng_4d is (batch, num_wv, n_layer); expand to (batch, num_wv, num_x, num_y, n_layer)
    ng_5d = ng_4d.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_x, num_y, -1)
```

This requires propagating the new `num_wv` axis through the entire function. Specifically:

- All `*_4d` tensors built around line 410 (`AOI_3d`, `theta_inc_medium_3d`, `sin_Vt_3d`) become 5-D `(batch, num_wv, num_x, num_y, num_layer)`.
- Adjust `create_eps_matrix_XY` to accept the 5-D `n_2d_w` (signature change, see below) so it returns shape `(batch, num_wv, num_x, num_y, num_layer, 3, 3)` rather than the current `(batch, num_x, num_y, num_layer, 3, 3)`.

Update `create_eps_matrix_XY` signature (line 268) to take `n_2d_w` of shape `(batch, num_wv, num_layer, 3)` and produce a 6-D eps tensor. Concretely, replace lines 314-335 (`nx2`, `ny2`, `nz2` extraction and expansion):

```python
    # n_2d_w is (batch, num_wv, num_layer, 3)
    nx2 = (
        n_2d_w[..., 0].real ** 2
        - n_2d_w[..., 0].imag ** 2
        + 2 * n_2d_w[..., 0].real * n_2d_w[..., 0].imag * 1j
    )  # (batch, num_wv, num_layer)
    ny2 = (
        n_2d_w[..., 1].real ** 2
        - n_2d_w[..., 1].imag ** 2
        + 2 * n_2d_w[..., 1].real * n_2d_w[..., 1].imag * 1j
    )
    nz2 = (
        n_2d_w[..., 2].real ** 2
        - n_2d_w[..., 2].imag ** 2
        + 2 * n_2d_w[..., 2].real * n_2d_w[..., 2].imag * 1j
    )

    # Expand to (batch, num_wv, num_x, num_y, num_layer)
    nx2 = nx2.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_x, num_y, -1)
    ny2 = ny2.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_x, num_y, -1)
    nz2 = nz2.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_x, num_y, -1)
```

Update `a_2d_exp` to include the wvln axis:

```python
    # a_2d is (batch, num_layer, 3) — no wvln dependence
    # Expand to (batch, num_wv, num_x, num_y, num_layer, 3)
    a_2d_exp = a_2d.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, num_2d_w_dim_1 := nx2.shape[1], num_x, num_y, -1, -1)
```

Actually, simpler: pull `num_wv` from `n_2d_w.shape[1]` and pass it as an arg. Refactor signature:

```python
def create_eps_matrix_XY(a_2d, n_2d_w, Az_2d):
    """...
    Args:
        a_2d: shape (batch, n_layer, 3)
        n_2d_w: shape (batch, num_wv, n_layer, 3)
        Az_2d: shape (batch, num_x, num_y)
    Returns:
        eps_6d: shape (batch, num_wv, num_x, num_y, n_layer, 3, 3)
    """
    device = a_2d.device
    batchsize, num_wv, num_layer, _ = n_2d_w.shape
    num_x = Az_2d.size(1)
    num_y = Az_2d.size(2)

    # a_2d_exp: (batch, num_wv, num_x, num_y, num_layer, 3)
    a_2d_exp = (
        a_2d.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        .expand(-1, num_wv, num_x, num_y, -1, -1)
    )
    # Az_2d_exp: (batch, num_wv, num_x, num_y, num_layer)
    Az_2d_exp = (
        Az_2d.unsqueeze(1).unsqueeze(-1)
        .expand(-1, num_wv, -1, -1, num_layer)
    )

    phi_medium = a_2d_exp[..., 0] + Az_2d_exp + torch.pi / 2
    theta_medium = a_2d_exp[..., 1] + torch.pi / 2
    psi_medium = a_2d_exp[..., 2] + torch.pi / 2

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

    nx2 = (
        n_2d_w[..., 0].real ** 2 - n_2d_w[..., 0].imag ** 2
        + 2j * n_2d_w[..., 0].real * n_2d_w[..., 0].imag
    )
    ny2 = (
        n_2d_w[..., 1].real ** 2 - n_2d_w[..., 1].imag ** 2
        + 2j * n_2d_w[..., 1].real * n_2d_w[..., 1].imag
    )
    nz2 = (
        n_2d_w[..., 2].real ** 2 - n_2d_w[..., 2].imag ** 2
        + 2j * n_2d_w[..., 2].real * n_2d_w[..., 2].imag
    )
    # Each is (batch, num_wv, num_layer); expand to (batch, num_wv, num_x, num_y, num_layer)
    nx2 = nx2.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_x, num_y, -1)
    ny2 = ny2.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_x, num_y, -1)
    nz2 = nz2.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_x, num_y, -1)

    exx = nx2 + (ny2 - nx2) * a2**2 + (nz2 - nx2) * a3**2
    eyy = nx2 + (ny2 - nx2) * b2**2 + (nz2 - nx2) * b3**2
    ezz = nx2 + (ny2 - nx2) * c2**2 + (nz2 - nx2) * c3**2
    exy = (ny2 - nx2) * a2 * b2 + (nz2 - nx2) * a3 * b3
    exz = (ny2 - nx2) * a2 * c2 + (nz2 - nx2) * a3 * c3
    eyz = (ny2 - nx2) * c2 * b2 + (nz2 - nx2) * b3 * c3

    eps_6d = torch.zeros(
        (batchsize, num_wv, num_x, num_y, num_layer, 3, 3),
        dtype=torch.complex64, device=device,
    )
    eps_6d[..., 0, 0] = exx
    eps_6d[..., 0, 1] = exy
    eps_6d[..., 0, 2] = exz
    eps_6d[..., 1, 0] = exy
    eps_6d[..., 1, 1] = eyy
    eps_6d[..., 1, 2] = eyz
    eps_6d[..., 2, 0] = exz
    eps_6d[..., 2, 1] = eyz
    eps_6d[..., 2, 2] = ezz
    return eps_6d
```

Now in `create_jones_matrix_AOIAz`, replace the `create_eps_matrix_XY` call and the subsequent `unsqueeze(1).expand` over `num_wv` (lines 416-429): the eps tensor is already 6-D, so just consume it directly without re-expanding the wvln axis. The downstream `Q_6d` and `Pn_all` shapes stay the same (lines 432-456) — they were already 7-D = `(batch, num_wv, num_x, num_y, num_layer, 4, 4)`.

Similarly, normalize `n_in` and `n_out` per-wavelength. Replace lines 402-403:

```python
    eps_in = n_in**2
    eps_out = n_out**2
```

with:

```python
    # n_in / n_out: scalar OR shape (batch, num_wv). Normalize to (batch, num_wv, 1, 1)
    def _per_wvln(x):
        if torch.is_tensor(x) and x.dim() == 2:
            return (x ** 2).to(torch.complex64).unsqueeze(-1).unsqueeze(-1)
        return torch.tensor(complex(x) ** 2, dtype=torch.complex64, device=device).view(1, 1, 1, 1)

    eps_in = _per_wvln(n_in)
    eps_out = _per_wvln(n_out)
```

And `EnterExitMatrix_XY` (line 135) must also accept per-wavelength `eps_in` / `eps_out`. Since `eps_in` is now `(batch, num_wv, 1, 1)` and `theta_in_2d` is `(batch, num_x, num_y)`, expand the theta tensors to `(batch, num_wv, num_x, num_y)` inside `create_jones_matrix_AOIAz` before passing into the helper:

```python
    theta_inc_air_3d = theta_inc_air_2d.unsqueeze(1).expand(-1, num_wv, -1, -1)
    theta_inc_sub_3d = theta_inc_sub_2d.unsqueeze(1).expand(-1, num_wv, -1, -1)
    T0_5d, T_N_inv_5d = EnterExitMatrix_XY(eps_in, eps_out, theta_inc_air_3d, theta_inc_sub_3d)
```

And update `EnterExitMatrix_XY` to handle the new 4-D theta and 4-D eps inputs, returning 6-D matrices `(batch, num_wv, num_x, num_y, 4, 4)`. (Modify the existing 3-D code to add the wvln axis everywhere.)

Replace the `T_N_inv_5d = T_N_inv_4d.unsqueeze(1).expand(...)` lines (469-470) — they're no longer needed since the helper now returns 5-D/6-D directly.

(This is a sizable refactor — implement it carefully. The shape contract is: every internal tensor in `create_jones_matrix_AOIAz` carries an explicit `num_wv` axis.)

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/material/test_anisotropic_core_refactor.py -v
```

Expected: PASS.

- [ ] **Step 5: Run benchmark sanity check**

```bash
python benchmarks/1_compare_angle_response_anisotropic.py
```

Expected: runs without error, plot regenerated.

- [ ] **Step 6: Commit**

```bash
git add difftmm/film_solver_anisotropic.py tests/material/test_anisotropic_core_refactor.py
git commit -m "refactor(anisotropic): accept per-wavelength n_in/n_out/n_2d tensors"
```

---

### Task 15: Wire `FilmSolver` to accept str/Material and anisotropic 3-tuples

**Files:**
- Modify: `difftmm/film_solver_anisotropic.py:493-end`
- Modify: `tests/material/test_solver_with_materials.py`
- Create: `tests/material/test_anisotropic_materials.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/material/test_solver_with_materials.py`:

```python
class TestAnisotropicSolverIsotropicInputs:
    def test_string_inputs_isotropic_path(self, cpu):
        solver = FilmSolver(
            mat_n_in="air",
            mat_n_out="N-BK7",
            mat_n_ls=["TiO2", "SiO2"],
            thickness_ls=[0.06, 0.10],
            device=cpu,
        )
        ts, tp, rs, rp = solver.simulate(
            theta=torch.tensor([0.0]), wvln=torch.tensor([0.55])
        )
        assert ts.shape == (1, 1, 1)
```

Create `tests/material/test_anisotropic_materials.py`:

```python
import torch

from difftmm import FilmSolver
from difftmm.material import Material


def test_three_equal_materials_matches_isotropic():
    cpu = torch.device("cpu")
    solver_iso = FilmSolver(
        mat_n_in="air", mat_n_out="N-BK7",
        mat_n_ls=["SiO2"], thickness_ls=[0.10], device=cpu,
    )
    solver_aniso = FilmSolver(
        mat_n_in="air", mat_n_out="N-BK7",
        mat_n_ls=[("SiO2", "SiO2", "SiO2")], thickness_ls=[0.10], device=cpu,
    )
    theta = torch.tensor([0.1])
    wvln = torch.tensor([0.55])
    ts1, _, _, _ = solver_iso.simulate(theta=theta, wvln=wvln)
    ts2, _, _, _ = solver_aniso.simulate(theta=theta, wvln=wvln)
    torch.testing.assert_close(ts1, ts2, rtol=1e-5, atol=1e-6)


def test_birefringent_layer_produces_polarization_difference():
    cpu = torch.device("cpu")
    # ne != no: ts and tp will differ noticeably even at non-zero angle
    solver = FilmSolver(
        mat_n_in="air", mat_n_out=1.52,
        mat_n_ls=[(2.4, 1.5, 1.5)],  # uniaxial-like
        thickness_ls=[0.10], device=cpu,
    )
    ts, tp, rs, rp = solver.simulate(
        theta=torch.tensor([0.5]),  # ~28 deg
        wvln=torch.tensor([0.55]),
    )
    # Birefringence → ts and tp magnitudes differ
    assert abs(abs(ts).item() - abs(tp).item()) > 1e-3
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest tests/material/test_solver_with_materials.py::TestAnisotropicSolverIsotropicInputs tests/material/test_anisotropic_materials.py -v
```

Expected: FAIL.

- [ ] **Step 3: Refactor `FilmSolver.__init__` and `simulate`**

Open `difftmm/film_solver_anisotropic.py`. Replace `FilmSolver.__init__` (line 500) following the pattern from Task 13, but allow 3-tuple specs:

```python
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
        from .material import Material

        self.batch_size = batch_size
        self.device = device

        def _normalize_scalar_or_str(spec):
            if isinstance(spec, str):
                return Material(spec, device=device)
            return spec

        def _normalize_layer(spec):
            # Layer can be: scalar | str | Material | 3-tuple of those
            if isinstance(spec, tuple):
                if len(spec) != 3:
                    raise ValueError(
                        f"anisotropic layer spec must be 3-tuple, got len {len(spec)}"
                    )
                return tuple(_normalize_scalar_or_str(s) for s in spec)
            return _normalize_scalar_or_str(spec)

        self._n_in_spec = _normalize_scalar_or_str(mat_n_in)
        self._n_out_spec = _normalize_scalar_or_str(mat_n_out)

        # Layer list: accept torch.Tensor (back-compat scalar path) or list-like
        if torch.is_tensor(mat_n_ls):
            if mat_n_ls.dim() == 1:
                self._n_layer_specs = [complex(v.item()) for v in mat_n_ls]
            elif mat_n_ls.dim() == 2 and mat_n_ls.shape[1] == 3:
                self._n_layer_specs = [
                    (complex(row[0].item()), complex(row[1].item()), complex(row[2].item()))
                    for row in mat_n_ls
                ]
            else:
                raise ValueError(f"mat_n_ls tensor must be 1-D or (N,3), got {mat_n_ls.shape}")
        else:
            self._n_layer_specs = [_normalize_layer(s) for s in mat_n_ls]
        self.num_layers = len(self._n_layer_specs)

        # Back-compat refract_idx_layers (only if all specs are scalar/3-scalar)
        def _is_scalar_or_scalar_tuple(s):
            if isinstance(s, tuple):
                return all(isinstance(x, (int, float, complex)) for x in s)
            return isinstance(s, (int, float, complex))

        all_scalar = all(_is_scalar_or_scalar_tuple(s) for s in self._n_layer_specs)
        if all_scalar:
            rows = []
            for s in self._n_layer_specs:
                if isinstance(s, tuple):
                    rows.append([complex(s[0]), complex(s[1]), complex(s[2])])
                else:
                    rows.append([complex(s), complex(s), complex(s)])
            t = torch.tensor(rows, dtype=torch.complex64)
            self.refract_idx_layers = t.unsqueeze(0).expand(batch_size, -1, -1).clone()
        else:
            self.refract_idx_layers = None

        self.mat_n_in = (
            float(mat_n_in) if isinstance(mat_n_in, (int, float)) and not isinstance(mat_n_in, bool)
            else None
        )
        self.mat_n_out = (
            float(mat_n_out) if isinstance(mat_n_out, (int, float)) and not isinstance(mat_n_out, bool)
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
```

Replace `simulate` body (line 627). Insert the resolution block after the wvln/theta handling and before `create_jones_matrix_AOIAz`:

```python
    def simulate(self, theta, wvln):
        from .material import resolve_indices

        # ... existing wvln/theta normalization (unchanged) ...

        d_1d = self.get_film_thickness()
        wv_1d = wv.unsqueeze(0).expand(self.batch_size, -1)
        n_wvlns = wv.shape[0]
        n_angles = theta.shape[1]

        # NEW: resolve per-axis n at each wavelength
        n_in_t = resolve_indices(self._n_in_spec, wv, self.device).unsqueeze(0).expand(
            self.batch_size, -1
        )
        n_out_t = resolve_indices(self._n_out_spec, wv, self.device).unsqueeze(0).expand(
            self.batch_size, -1
        )

        # Build n_2d_w: shape (batch, n_wvln, n_layer, 3)
        per_layer_axes = []  # list of (n_wvln, 3) tensors
        for s in self._n_layer_specs:
            if isinstance(s, tuple):
                cols = torch.stack(
                    [resolve_indices(s, wv, self.device, axis=ax) for ax in (0, 1, 2)],
                    dim=-1,
                )
            else:
                col = resolve_indices(s, wv, self.device)
                cols = col.unsqueeze(-1).expand(-1, 3)  # isotropic -> broadcast
            per_layer_axes.append(cols)
        n_2d_w = torch.stack(per_layer_axes, dim=-2)  # (n_wvln, n_layer, 3)
        n_2d_w = n_2d_w.unsqueeze(0).expand(self.batch_size, -1, -1, -1)

        a_2d = torch.zeros((self.batch_size, self.num_layers, 3),
                           dtype=torch.complex64, device=self.device)
        d_1d_complex = d_1d.to(torch.complex64)
        Az_1d = torch.zeros((self.batch_size, 1), device=self.device)

        Jt, Jr = create_jones_matrix_AOIAz(
            a_2d, n_2d_w, d_1d_complex, wv_1d, n_in_t, n_out_t, theta, Az_1d
        )

        # ... existing polarization extraction (unchanged) ...
        # (keep the Jones-matrix projection code at lines 681-704)

        return ts, tp, rs, rp
```

- [ ] **Step 4: Make `save_ckpt` / `load_ckpt` persist material names (incl. 3-tuples)**

Add the same helpers at the top of `difftmm/film_solver_anisotropic.py` (after `inv_sigmoid`):

```python
def _serialize_spec(spec):
    """Serialize one refractive-index spec to a checkpoint-safe value."""
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
```

Replace `save_ckpt` (line 593):

```python
    def save_ckpt(self, save_path):
        """Save thicknesses and material specs to a checkpoint."""
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
```

Replace `load_ckpt` (line 575) — same structure as the isotropic counterpart:

```python
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
```

Add a test for anisotropic 3-tuple checkpoint roundtrip in `tests/material/test_anisotropic_materials.py`:

```python
def test_anisotropic_3tuple_checkpoint_roundtrip(tmp_path):
    cpu = torch.device("cpu")
    path = tmp_path / "ckpt.pt"
    solver = FilmSolver(
        mat_n_in="air", mat_n_out=1.52,
        mat_n_ls=[("SiO2", "TiO2", "SiO2"), (2.4, 2.4, 1.5)],
        thickness_ls=[0.1, 0.06], device=cpu,
    )
    solver.save_ckpt(path)
    ckpt = torch.load(path, weights_only=False)
    # Material 3-tuple should serialize to a (str, str, str) tuple
    assert ckpt["layer_specs"][0] == ("sio2", "tio2", "sio2")
    # Scalar 3-tuple should serialize to a (complex, complex, complex) tuple
    assert all(isinstance(v, complex) for v in ckpt["layer_specs"][1])
```

- [ ] **Step 5: Run all solver tests**

```bash
python -m pytest tests/material/ -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add difftmm/film_solver_anisotropic.py tests/material/test_solver_with_materials.py tests/material/test_anisotropic_materials.py
git commit -m "feat(anisotropic): accept str/Material/scalar/3-tuple in mat_n_ls"
```

---

### Task 16: Top-level re-exports and backward-compatibility regression test

**Files:**
- Modify: `difftmm/__init__.py`
- Create: `tests/material/test_backward_compat.py`

- [ ] **Step 1: Write failing test**

```python
# tests/material/test_backward_compat.py
"""Regression tests: the scalar-only API must produce identical outputs after
the Material refactor."""
import torch

from difftmm import IsotropicFilmSolver, FilmSolver


def test_isotropic_scalar_api_outputs_finite():
    solver = IsotropicFilmSolver(
        mat_n_in=1.0,
        mat_n_out=1.52,
        mat_n_ls=[2.10, 1.46, 2.10],
        thickness_ls=[0.080, 0.120, 0.080],
        device=torch.device("cpu"),
    )
    theta = torch.linspace(0, 1.2, 10)
    wvln = [0.45, 0.55, 0.65]
    ts, tp, rs, rp = solver.simulate(theta=theta, wvln=wvln)
    assert ts.shape == (1, 3, 10)
    assert torch.isfinite(ts).all() and torch.isfinite(rp).all()


def test_isotropic_scalar_api_energy_conservation_at_normal_incidence():
    """|R|² + |T|²·(n_out cos θ_out / n_in cos θ_in) = 1 for lossless stack."""
    solver = IsotropicFilmSolver(
        mat_n_in=1.0,
        mat_n_out=1.0,  # symmetric so R + T = 1 at normal incidence
        mat_n_ls=[2.10, 1.46, 2.10],
        thickness_ls=[0.080, 0.120, 0.080],
        device=torch.device("cpu"),
    )
    ts, tp, rs, rp = solver.simulate(
        theta=torch.tensor([0.0]), wvln=torch.tensor([0.55])
    )
    R = (rs.abs() ** 2).item()
    T = (ts.abs() ** 2).item()
    assert abs(R + T - 1.0) < 1e-3


def test_top_level_imports():
    from difftmm import (
        Material,
        MATERIAL_data,
        list_materials,
        IsotropicFilmSolver,
        FilmSolver,
    )
    assert callable(Material)
    assert isinstance(MATERIAL_data, dict)
    assert "air" in list_materials()
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/material/test_backward_compat.py -v
```

Expected: `test_top_level_imports` FAILS (Material not re-exported yet); other tests should PASS (back-compat already preserved).

- [ ] **Step 3: Update `difftmm/__init__.py`**

```python
from .film_solver_isotropic import (
    IsotropicFilmSolver,
    create_jones_matrix_isotropic,
)
from .film_solver_anisotropic import (
    FilmSolver,
    create_jones_matrix_AOIAz,
)
from .material import (
    MATERIAL_data,
    Material,
    list_materials,
    resolve_indices,
)

AnisotropicFilmSolver = FilmSolver

__all__ = [
    "IsotropicFilmSolver",
    "FilmSolver",
    "AnisotropicFilmSolver",
    "create_jones_matrix_isotropic",
    "create_jones_matrix_AOIAz",
    "Material",
    "MATERIAL_data",
    "list_materials",
    "resolve_indices",
]
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/material/ -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add difftmm/__init__.py tests/material/test_backward_compat.py
git commit -m "feat(api): re-export Material, MATERIAL_data, list_materials at top level"
```

---

## Phase 3 — Examples, packaging, docs

### Task 17: Example notebook `3_real_materials.ipynb`

**Files:**
- Create: `3_real_materials.ipynb`

- [ ] **Step 1: Create the notebook**

Write a notebook that runs end-to-end. Cells (use `jupyter nbconvert --to notebook` or write the JSON directly):

Cell 1 (markdown):
```
# 3 — Real Materials

This notebook demonstrates wavelength-dependent refractive indices in DiffTMM.
```

Cell 2 (code):
```python
import torch
import matplotlib.pyplot as plt

from difftmm import IsotropicFilmSolver, Material, list_materials

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"{len(list_materials())} known materials")
```

Cell 3 (markdown): `## 1. Plot n(λ), k(λ)`

Cell 4 (code):
```python
wvln = torch.linspace(0.40, 0.80, 100)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for name in ["SiO2", "TiO2", "Ag"]:
    mat = Material(name, device=device)
    n = mat.ior(wvln.to(device)).cpu()
    ax1.plot(wvln.numpy(), n.real.numpy(), label=name)
    ax2.plot(wvln.numpy(), n.imag.numpy(), label=name)
ax1.set_xlabel("wavelength (μm)"); ax1.set_ylabel("n"); ax1.legend(); ax1.grid()
ax2.set_xlabel("wavelength (μm)"); ax2.set_ylabel("k"); ax2.legend(); ax2.grid()
plt.tight_layout(); plt.show()
```

Cell 5 (markdown): `## 2. Anti-reflection coating broadband reflectance`

Cell 6 (code):
```python
solver = IsotropicFilmSolver(
    mat_n_in="air",
    mat_n_out="N-BK7",
    mat_n_ls=["TiO2", "SiO2"],  # 2-layer V-coat
    thickness_ls=[0.06, 0.10],
    device=device,
)
wvln = torch.linspace(0.45, 0.75, 60).to(device)
theta = torch.tensor([0.0], device=device)
ts, tp, rs, rp = solver.simulate(theta=theta, wvln=wvln)
R = (rs.abs() ** 2).squeeze().cpu()
plt.plot(wvln.cpu(), R)
plt.xlabel("λ (μm)"); plt.ylabel("R"); plt.title("Reflectance"); plt.grid(); plt.show()
```

Cell 7 (markdown): `## 3. Surface plasmon resonance with Ag`

Cell 8 (code):
```python
# Kretschmann SPR: prism | metal | air; sweep angle at 633 nm
solver = IsotropicFilmSolver(
    mat_n_in=1.52,  # prism
    mat_n_out="air",
    mat_n_ls=["Ag"],
    thickness_ls=[0.05],  # 50 nm Ag
    device=device,
)
theta = torch.linspace(0.7, 1.2, 200).to(device)  # ~40-70 deg
wvln = torch.tensor([0.633], device=device)
_, _, _, rp = solver.simulate(theta=theta, wvln=wvln)
R_p = (rp.abs() ** 2).squeeze().cpu()
plt.plot(theta.cpu() * 180 / 3.14159, R_p)
plt.xlabel("angle (deg)"); plt.ylabel("R_p"); plt.title("Kretschmann SPR — 633 nm")
plt.grid(); plt.show()
```

- [ ] **Step 2: Execute the notebook end-to-end**

```bash
jupyter nbconvert --to notebook --execute 3_real_materials.ipynb --output 3_real_materials.ipynb
```

Expected: completes without errors, plots rendered.

- [ ] **Step 3: Commit**

```bash
git add 3_real_materials.ipynb
git commit -m "docs(notebooks): add 3_real_materials.ipynb"
```

---

### Task 18: Packaging — pyproject.toml and MANIFEST.in

**Files:**
- Modify: `pyproject.toml`
- Modify: `MANIFEST.in`

- [ ] **Step 1: Update `pyproject.toml`**

Edit `pyproject.toml`. After line 74 (`exclude = [...]`), add:

```toml
[tool.setuptools.package-data]
"difftmm.material" = ["catalogs/*.AGF", "catalogs/*.json"]
```

Also verify `[tool.setuptools.packages.find]` block at lines 72-74 still picks up `difftmm.material` — it should, since `find` is recursive and `difftmm.material/__init__.py` exists.

- [ ] **Step 2: Update `MANIFEST.in`**

Append to `MANIFEST.in`:

```
recursive-include difftmm/material/catalogs *.AGF *.json
```

- [ ] **Step 3: Build and verify catalog files ship in the sdist**

```bash
pip install --upgrade build
python -m build --sdist
tar -tzf dist/difftmm-*.tar.gz | grep -E "(\.AGF|materials_data\.json|thin_film_materials\.json)"
```

Expected: 6 catalog files listed.

- [ ] **Step 4: Build wheel and verify**

```bash
python -m build --wheel
unzip -l dist/difftmm-*.whl | grep -E "(\.AGF|\.json)" | grep material
```

Expected: 6 catalog files listed.

- [ ] **Step 5: Cleanup build artifacts**

```bash
rm -rf build/ dist/ difftmm.egg-info/
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml MANIFEST.in
git commit -m "build(package): ship material catalogs as package data"
```

---

### Task 19: Update README with Real Materials section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Real Materials" subsection**

Open `README.md`. After the "Two Solvers" section (around line 102), insert a new subsection:

```markdown
## Real Materials

DiffTMM ships with wavelength-dependent refractive index support via the
`Material` class. Look up materials by name (case-insensitive):

```python
from difftmm import IsotropicFilmSolver, Material, list_materials

# Bundled catalogs: CDGM/SCHOTT/MISC AGF glasses + thin-film n+k tables
print(len(list_materials()), "materials available")

# Pass material names directly to a solver — they're auto-wrapped in Material()
solver = IsotropicFilmSolver(
    mat_n_in="air",
    mat_n_out="N-BK7",                  # Sellmeier (AGF)
    mat_n_ls=["TiO2", "SiO2"],          # n+k tables for thin-film materials
    thickness_ls=[0.06, 0.10],
)
ts, tp, rs, rp = solver.simulate(theta=angles, wvln=[0.45, 0.55, 0.65])
```

Scalars, strings, and `Material` objects can be mixed freely in `mat_n_ls`.
For the 4×4 `FilmSolver`, anisotropic per-axis dispersion is expressed as
a `(mat_x, mat_y, mat_z)` tuple per layer.

Dispersion models supported in v1: **Sellmeier** (analytical, real n) and
**linear interpolation** (lookup tables, complex n + ik).
```

- [ ] **Step 2: Update the Repository Structure tree**

Find the existing tree block around line 134. Replace it with:

```markdown
## Repository Structure

```
├── difftmm/                            # Importable package
│   ├── __init__.py                     #   Public API
│   ├── film_solver_isotropic.py        #   2x2 isotropic solver (fast)
│   ├── film_solver_anisotropic.py      #   4x4 anisotropic solver (general)
│   └── material/                       #   Wavelength-dependent materials
│       ├── __init__.py
│       ├── materials.py                #     Material class, loaders, resolve_indices
│       └── catalogs/                   #     Bundled glass + thin-film data
├── 1_forward_simu.ipynb                # Example: forward simulation
├── 2_inverse_design.ipynb              # Example: differentiable inverse design
├── 3_real_materials.ipynb              # Example: real materials
├── tmm_numpy/                          # Reference NumPy TMM library
├── benchmarks/                         # Accuracy and performance benchmarks
├── tests/                              # Pytest suite
├── pyproject.toml
└── README.md
```
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document real material support and updated layout"
```

---

## Final verification

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS, no errors, no warnings about deprecated APIs.

- [ ] **Step 2: Run all benchmarks**

```bash
python benchmarks/1_compare_angle_response_isotropic.py
python benchmarks/1_compare_angle_response_anisotropic.py
python benchmarks/2_compare_speed.py
python benchmarks/3_compare_memory.py
```

Expected: all run to completion; visual outputs match pre-change baselines within numerical noise.

- [ ] **Step 3: Smoke-test the notebooks**

```bash
jupyter nbconvert --to notebook --execute 1_forward_simu.ipynb --output 1_forward_simu.ipynb
jupyter nbconvert --to notebook --execute 2_inverse_design.ipynb --output 2_inverse_design.ipynb
jupyter nbconvert --to notebook --execute 3_real_materials.ipynb --output 3_real_materials.ipynb
```

Expected: all execute without errors.

- [ ] **Step 4: Final commit and summary**

If any cleanup commits accumulated above, ensure they're committed and tagged with the issue:

```bash
git log --oneline f2bd717..HEAD
```

Each commit message should mention the issue (#3) or feature area (`feat(material)`, `refactor(isotropic)`, etc.).
