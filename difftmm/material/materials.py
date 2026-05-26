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

MATERIAL_data: dict = {**_AGF_DATA, **{k: {"source": "json"} for k in _SELLMEIER_TABLE}}  # Public — exported via package __init__


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
