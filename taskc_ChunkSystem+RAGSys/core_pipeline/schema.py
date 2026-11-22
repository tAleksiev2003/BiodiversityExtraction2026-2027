"""Load the official schemas and deterministic English-to-German catalogues."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator

DATA = Path(__file__).parent / "schemas"
FIELDS = ("habitat", "biodiversity_level", "ecosystem_type", "taxa",
          "biodiversity_variable", "ecosystem_service")


def load_schema():
    """Merge the official C1/C2b schemas and require all output arrays."""
    core = json.loads((DATA / "core_fields.json").read_text())
    bio = json.loads((DATA / "biodiversity.json").read_text())
    for name, definition in bio["$defs"].items():
        if name in core["$defs"] and core["$defs"][name] != definition:
            raise ValueError(f"Conflicting schema definition: {name}")
        core["$defs"][name] = definition
    core["properties"].update(bio["properties"])
    core["properties"]["zotitem_ptr_id"] = {"type": "string", "minLength": 1}
    core["required"] = ["zotitem_ptr_id", *FIELDS]
    # Task C2b requests category/term, not optional service details.
    core["$defs"]["EcosystemService"]["properties"].pop("details", None)
    Draft202012Validator.check_schema(core)
    return core


def catalogue(schema):
    """Build reversible English aliases while retaining exact official enum strings."""
    glosses = json.loads((DATA / "english_glosses.json").read_text())
    result = {}
    for name, definition in schema["$defs"].items():
        if "enum" not in definition:
            continue
        english = glosses[name]
        german = definition["enum"]
        if len(english) != len(german) or len(set(english)) != len(english):
            raise ValueError(f"Incomplete or ambiguous glossary: {name}")
        result[name] = dict(zip(english, german))
    return result


def translate_value(value, spec, schema, labels):
    """Map enum aliases recursively and reject unsupported keys or values."""
    if "$ref" in spec:
        name = spec["$ref"].rsplit("/", 1)[-1]
        if name in labels:
            if not isinstance(value, str):
                raise ValueError("Enum must be a string")
            if value in labels[name]:
                return labels[name][value]
            if value in labels[name].values():
                return value
            raise ValueError(f"Unknown enum value: {value}")
        return translate_value(value, schema["$defs"][name], schema, labels)
    if "anyOf" in spec:
        if value is None:
            return None
        return translate_value(value, spec["anyOf"][0], schema, labels)
    if spec.get("type") == "object":
        if not isinstance(value, dict) or set(value) - set(spec["properties"]):
            raise ValueError("Invalid nested object keys")
        if set(spec.get("required", [])) - set(value):
            raise ValueError("Missing required nested field")
        return {k: translate_value(v, spec["properties"][k], schema, labels)
                for k, v in value.items()}
    if spec.get("type") == "string":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Expected non-empty text")
        return value.strip()
    raise ValueError("Unsupported value type")


def validate_record(record, schema):
    """Validate the complete submission record against the merged JSON Schema."""
    Draft202012Validator(schema).validate(record)


def prompt_schema(schema):
    """Return an independent schema copy without the externally supplied paper ID."""
    result = copy.deepcopy(schema)
    result["properties"].pop("zotitem_ptr_id")
    result["required"].remove("zotitem_ptr_id")
    return result
