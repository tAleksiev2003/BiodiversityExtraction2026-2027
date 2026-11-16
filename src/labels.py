"""Allowed label values, read only from the schema files in data/schemas/.

The schemas are the single source of truth (SPEC.md, section 4): no label string is ever
typed into the code. This module turns each schema into a flat list of fields such as
`habitat` or `ecosystem_type.term`, each with its allowed values (or None for open text).
The evaluation and the baselines both work on these flat fields.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REF_PREFIX = "#/$defs/"


@dataclass(frozen=True)
class FieldSpec:
    """One scorable field, e.g. `habitat` or the subfield `ecosystem_type.term`."""

    name: str
    parent: str
    subfield: str | None
    allowed_values: tuple[str, ...] | None  # None = open text (no fixed list)

    @property
    def is_enum(self) -> bool:
        return self.allowed_values is not None


def load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_enum(schema: dict[str, Any], enum_name: str) -> list[str]:
    """Return the allowed values of one enum, exactly as written in the schema."""
    return list(schema["$defs"][enum_name]["enum"])


def _resolve_ref(node: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Follow a `$ref` like `#/$defs/HabitatEnum`; other nodes are returned unchanged."""
    reference = node.get("$ref")
    if reference is None:
        return node
    return schema["$defs"][reference.removeprefix(REF_PREFIX)]


def _allowed_values(node: dict[str, Any], schema: dict[str, Any]) -> tuple[str, ...] | None:
    """Return the enum behind a node, or None if the node is open text.

    Optional subfields are written as `anyOf: [{$ref: Enum}, {type: null}]`, so every
    option of an `anyOf` is checked.
    """
    options = node.get("anyOf", [node])
    for option in options:
        resolved = _resolve_ref(option, schema)
        if "enum" in resolved:
            return tuple(resolved["enum"])
    return None


def field_specs_from_schema(schema: dict[str, Any]) -> dict[str, FieldSpec]:
    """List every field and subfield of one schema with its allowed values."""
    specs: dict[str, FieldSpec] = {}
    for parent, property_node in schema["properties"].items():
        item_node = _resolve_ref(property_node["items"], schema)
        if item_node.get("type") == "object":
            for subfield, subfield_node in item_node["properties"].items():
                name = f"{parent}.{subfield}"
                specs[name] = FieldSpec(name, parent, subfield, _allowed_values(subfield_node, schema))
        else:
            specs[parent] = FieldSpec(parent, parent, None, _allowed_values(property_node["items"], schema))
    return specs


def load_field_specs(schema_paths: list[Path]) -> dict[str, FieldSpec]:
    """Merge the fields of several schemas (C1 and C2b share `taxa` and `biodiversity_level`)."""
    merged: dict[str, FieldSpec] = {}
    for schema_path in schema_paths:
        for name, spec in field_specs_from_schema(load_schema(schema_path)).items():
            if name in merged and merged[name] != spec:
                raise ValueError(f"Field {name!r} is defined differently in {schema_path}")
            merged[name] = spec
    return merged


def load_term_to_category(path: Path) -> dict[str, dict[str, str]]:
    """Load {field: {term: category}} built by src/scripts/build_term_to_category.py."""
    return json.loads(path.read_text(encoding="utf-8"))
