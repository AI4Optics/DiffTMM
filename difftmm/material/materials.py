"""Optical material with wavelength-dependent complex refractive index."""

from __future__ import annotations

import json
import os
import re

import torch


_AIR_ALIASES = {"air", "vacuum", "occluder"}

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


def _read_json_catalog(file_path: str) -> dict:
    """Read a JSON catalog file and return its contents as a dict."""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


_AGF_DATA: dict = _load_all_agf()

_CUSTOM_DATA: dict = _read_json_catalog(
    os.path.join(_CATALOGS_DIR, "materials_data.json")
)
_SELLMEIER_TABLE: dict = _CUSTOM_DATA.get("SELLMEIER_TABLE", {})
_MATERIAL_TABLE: dict = _CUSTOM_DATA.get("MATERIAL_TABLE", {})
_INTERP_TABLE: dict = _CUSTOM_DATA.get("INTERP_TABLE", {})

_THINFILM_DATA: dict = _read_json_catalog(
    os.path.join(_CATALOGS_DIR, "thin_film_materials.json")
)
# Build a case-insensitive lookup map name_lower -> entry
_INTERP_NK_TABLE: dict = {
    k.lower(): v for k, v in _THINFILM_DATA.get("INTERP_NK_TABLE", {}).items()
}

MATERIAL_data: dict = {
    **_AGF_DATA,
    **{k: {"source": "json"} for k in _SELLMEIER_TABLE if k not in _AGF_DATA},
}  # Public — exported via package __init__


def _linear_interp_complex(
    wvln: torch.Tensor,
    ref_wvlns: torch.Tensor,
    ref_n: torch.Tensor,
    ref_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable linear interpolation of (n, k) into complex output.

    Args:
        wvln: Query wavelengths (μm), shape (...,).
        ref_wvlns: Sorted table wavelengths, shape (M,).
        ref_n: Real refractive index at each table point, shape (M,).
        ref_k: Extinction coefficient at each table point, shape (M,), or None.

    Returns:
        torch.complex64 tensor with the same shape as `wvln`.
    """
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

        if self.name in _SELLMEIER_TABLE:
            coeffs = _SELLMEIER_TABLE[self.name]
            self.dispersion = "sellmeier"
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = coeffs
            nv = _MATERIAL_TABLE.get(self.name, [None, None])
            self.n = nv[0] if nv[0] is not None else 0.0
            self.V = nv[1] if nv[1] is not None else 1e38
            return

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

        raise NotImplementedError(f"Material {self.name!r} not implemented.")

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
        if self.dispersion == "interp":
            self._ref_wvlns = self._ref_wvlns.to(wvln.device)
            self._ref_n = self._ref_n.to(wvln.device)
            ref_k = self._ref_k.to(wvln.device) if self._ref_k is not None else None
            return _linear_interp_complex(wvln, self._ref_wvlns, self._ref_n, ref_k)
        raise NotImplementedError(f"Dispersion {self.dispersion!r} not implemented.")
