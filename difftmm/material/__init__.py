"""Material support for DiffTMM — wavelength-dependent refractive indices."""

from .materials import MATERIAL_data, Material, list_materials, resolve_indices

__all__ = ["Material", "MATERIAL_data", "list_materials", "resolve_indices"]
