"""Material support for DiffTMM — wavelength-dependent refractive indices."""

from .materials import (
    Material,
    _deserialize_spec,
    _serialize_spec,
    list_materials,
    resolve_indices,
)

__all__ = [
    "Material",
    "list_materials",
    "resolve_indices",
    "_serialize_spec",
    "_deserialize_spec",
]
