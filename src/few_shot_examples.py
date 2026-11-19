"""The training examples of prompt v4: how the database labelled similar documents.

The systems propose many labels the database does not give (SPEC.md, section 13,
2026-11-19), and the database's conventions are not written down anywhere. Prompt v4
therefore shows, per document, the database labels of its most similar TRAIN papers
(src/retrieval.py writes them to neighbours_<split>.json). The template's `{examples}`
placeholder is filled with the text built here.

Per example only: title, bibtex type, and the labels of habitat, ecosystem_type,
biodiversity_level, taxa (at most `few_shot.max_taxa_per_example`) and ecosystem_service.
Nothing else of the example (no free text, no descriptions), so the model learns which
fields are filled and at which level, not what the example is about.
"""
import json
import logging
from typing import Any

from data import load_references
from prompts import quoted
from retrieval import neighbours_path

logger = logging.getLogger(__name__)

EMPTY = "(empty)"
UNKNOWN = "(unknown)"
NULL = "null"
NO_EXAMPLES = "(no similar training documents found for this document)"
SUBFIELD_SEPARATOR = " / "
ENTRY_SEPARATOR = "; "

# Field names (not label values) shown per example, in prompt order, with the subfields
# of each entry. A field without subfields is a list of strings.
EXAMPLE_FIELDS: dict[str, tuple[str, ...]] = {
    "habitat": (),
    "ecosystem_type": ("category", "term"),
    "biodiversity_level": (),
    "taxa": ("scientific_name", "german_name", "species_group"),
    "ecosystem_service": ("category", "term"),
}
TAXA_FIELD = "taxa"


def value_text(value: Any) -> str:
    """A label in double quotes as in the field lists of the prompt; a missing one as null."""
    if isinstance(value, str) and value:
        return quoted([value])
    return NULL


def field_text(reference: dict[str, Any], field: str, subfields: tuple[str, ...],
               max_entries: int | None) -> str:
    """All labels of one field of one reference, in the order the database gives them."""
    entries = reference.get(field) or []
    if not entries:
        return EMPTY
    shown = entries if max_entries is None else entries[:max_entries]
    if subfields:
        parts = [SUBFIELD_SEPARATOR.join(value_text(entry.get(name)) for name in subfields)
                 for entry in shown]
    else:
        parts = [value_text(entry) for entry in shown]
    text = ENTRY_SEPARATOR.join(parts)
    hidden = len(entries) - len(shown)
    if hidden:
        text += f" (and {hidden} more)"
    return text


def format_example(number: int, reference: dict[str, Any], few_shot: dict[str, Any]) -> str:
    """One example: heading line, then one line per field."""
    title = reference.get(few_shot["title_field"]) or UNKNOWN
    bibtex_type = reference.get(few_shot["type_field"]) or UNKNOWN
    lines = [f"Example {number}: {json.dumps(title, ensure_ascii=False)} ({few_shot['type_field']}: {bibtex_type})"]
    for field, subfields in EXAMPLE_FIELDS.items():
        max_entries = few_shot["max_taxa_per_example"] if field == TAXA_FIELD else None
        label = f"{field} ({SUBFIELD_SEPARATOR.join(subfields)})" if subfields else field
        lines.append(f"- {label}: {field_text(reference, field, subfields, max_entries)}")
    return "\n".join(lines)


def format_examples(references: list[dict[str, Any]], few_shot: dict[str, Any]) -> str:
    """The text for `{examples}`: all examples, separated by an empty line."""
    if not references:
        return NO_EXAMPLES
    return "\n\n".join(format_example(number, reference, few_shot)
                       for number, reference in enumerate(references, start=1))


class FewShotExamples:
    """Example text per document of one split, from neighbours_<split>.json and the train references."""

    def __init__(self, neighbour_ids: dict[str, list[str]], train_references: dict[str, dict[str, Any]],
                 few_shot: dict[str, Any]) -> None:
        unknown = {train_id for ids in neighbour_ids.values() for train_id in ids} - set(train_references)
        if unknown:
            raise ValueError(f"neighbours that are not train references: {sorted(unknown)}")
        own = [document_id for document_id, ids in neighbour_ids.items() if document_id in ids]
        if own:
            raise ValueError(f"documents listed as their own neighbour: {own}")
        self.neighbour_ids = neighbour_ids
        self.train_references = train_references
        self.few_shot = few_shot

    @classmethod
    def from_settings(cls, settings: dict[str, Any], split: str) -> "FewShotExamples":
        """Fails if the neighbour file was made with another method or k than the settings,
        so a run never uses examples the settings (and the run's config.yaml) do not describe."""
        path = neighbours_path(settings, split)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python3 src/retrieval.py --split {split}")
        content = json.loads(path.read_text(encoding="utf-8"))
        retrieval = settings["retrieval"]
        for key in ("method", "k"):
            if content[key] != retrieval[key]:
                raise ValueError(f"{path} was made with {key}={content[key]!r}, the settings say "
                                 f"{retrieval[key]!r}: run python3 src/retrieval.py --split {split} again")
        neighbour_ids = {document_id: [entry["id"] for entry in entries]
                         for document_id, entries in content["neighbours"].items()}
        train_references = load_references(settings["paths"]["references_train"])
        return cls(neighbour_ids, train_references, settings["few_shot"])

    def text_for(self, document_id: str) -> str:
        """The `{examples}` text of one document; a document without neighbours (no
        first-pages text) gets a note instead of examples."""
        ids = self.neighbour_ids.get(document_id)
        if not ids:
            logger.warning("%s: no neighbours, prompt without examples", document_id)
            return NO_EXAMPLES
        return format_examples([self.train_references[train_id] for train_id in ids], self.few_shot)
