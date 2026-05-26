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
