"""Material support for DiffTMM — wavelength-dependent refractive indices."""

from .materials import (
    MATERIAL_data,
    Material,
    _deserialize_spec,
    _serialize_spec,
    list_materials,
    resolve_indices,
)

__all__ = [
    "Material",
    "MATERIAL_data",
    "list_materials",
    "resolve_indices",
    "_serialize_spec",
    "_deserialize_spec",
]
