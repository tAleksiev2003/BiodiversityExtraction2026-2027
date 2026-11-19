"""Build the extraction prompt from a template file (SPEC.md, section 6).

Pipeline F uses `paths.prompt_extraction`, pipeline B `paths.prompt_extraction_chunked`.

The template has sections marked by lines like `=== SYSTEM ===`. Its placeholders are
filled here from the schema files and data/labels/, so the model sees every allowed
value exactly as the schema spells it and no label string is typed into the code:

- `{habitat_labels}`, `{biodiversity_level_labels}`, `{species_group_labels}`,
  `{ecosystem_service_category_labels}`: the enum values, each in double quotes;
- `{ecosystem_terms_by_category}`: every ecosystem type category with its terms
  (from term_to_category.json), so the model can only form consistent pairs (rule R6);
- `{ecosystem_service_terms}`: all service terms of the schema, one per line;
- `{variable_vocabulary}`: the `prompt.vocabulary_top_n` most frequent training values;
- `{<field>_question}` (since prompt v3), e.g. `{taxa_question}`: the guiding question
  of the database for that field, the `description` of the property in the schema;
- `{min_quote_words}`, `{max_quote_words}` (since prompt v3): the quote length the
  verbatim checker accepts (`verbatim_check` in settings.yaml), so prompt and check C3
  cannot disagree;
- `{document_id}`, `{chunk_id}`, `{chunk_text}`: the text of one call;
- `{examples}` (since prompt v4): the database labels of the most similar training
  papers of this document (src/few_shot_examples.py), different for every document.

The fixed parts are filled once per run (`ExtractionPrompt.from_settings`); only the
text-specific placeholders change per call. A template may leave out any of these
placeholders (v1 and v2 have no questions, v1 to v3 no examples), so older versions give
the same prompt as before and their answers still come from the cache.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labels import FieldSpec, load_field_questions, load_field_specs, load_term_to_category
from llm_cache import Message

SECTION_PATTERN = re.compile(r"^=== (.+?) ===$", re.MULTILINE)
SYSTEM_SECTION = "SYSTEM"
USER_SECTION = "USER"
LIST_INDENT = "   "
ECOSYSTEM_TYPE_FIELD = "ecosystem_type"
ECOSYSTEM_SERVICE_FIELD = "ecosystem_service"
QUESTION_PLACEHOLDER_SUFFIX = "_question"
EXAMPLES_PLACEHOLDER = "{examples}"
PER_CALL_PLACEHOLDERS = ("document_id", "chunk_id", "chunk_text", "examples")


def read_sections(template_path: Path) -> dict[str, str]:
    """{section name: text} for a template with `=== NAME ===` lines."""
    template = template_path.read_text(encoding="utf-8")
    parts = SECTION_PATTERN.split(template)
    # split() gives [text before the first marker, name 1, text 1, name 2, text 2, ...]
    names, texts = parts[1::2], parts[2::2]
    return {name: text.strip() for name, text in zip(names, texts)}


def quoted(values: list[str] | tuple[str, ...]) -> str:
    """Values in double quotes, so the model copies them character by character (rule R1)."""
    return ", ".join(json.dumps(value, ensure_ascii=False) for value in values)


def terms_by_category(
    categories: tuple[str, ...], terms: tuple[str, ...], term_to_category: dict[str, str]
) -> str:
    """One line per category with its terms, both in schema order."""
    lines = []
    for category in categories:
        category_terms = [term for term in terms if term_to_category.get(term) == category]
        lines.append(f"{LIST_INDENT}- category {quoted([category])}: terms {quoted(category_terms)}")
    return "\n".join(lines)


def one_per_line(values: tuple[str, ...]) -> str:
    """Long values (service terms) are easier to read and copy one per line."""
    return "".join(f"\n{LIST_INDENT}  - {quoted([value])}" for value in values)


def top_vocabulary(vocabulary_path: Path, top_n: int) -> list[str]:
    """The `top_n` most frequent values of biodiversity_variable in the training references."""
    entries: list[dict[str, Any]] = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    by_count = sorted(entries, key=lambda entry: entry["count"], reverse=True)  # stable for ties
    return [entry["value"] for entry in by_count[:top_n]]


def allowed(specs: dict[str, FieldSpec], name: str) -> tuple[str, ...]:
    values = specs[name].allowed_values
    if values is None:
        raise ValueError(f"{name} has no list of allowed values in the schema")
    return values


@dataclass(frozen=True)
class ExtractionPrompt:
    """The extraction template with all label lists filled in."""

    system_text: str
    user_template: str  # still contains {document_id}, {chunk_id}, {chunk_text} (and {examples})

    @classmethod
    def from_settings(cls, settings: dict[str, Any], template_path: Path | None = None) -> "ExtractionPrompt":
        """`template_path`: default `paths.prompt_extraction` (pipeline F); pipeline B
        passes `paths.prompt_extraction_chunked`. Both use the same placeholders."""
        paths = settings["paths"]
        schema_paths = [paths["schema_core"], paths["schema_biodiversity"]]
        specs = load_field_specs(schema_paths)
        term_to_category = load_term_to_category(paths["term_to_category"])
        sections = read_sections(template_path or paths["prompt_extraction"])
        label_lists = {
            "habitat_labels": quoted(allowed(specs, "habitat")),
            "ecosystem_terms_by_category": terms_by_category(
                allowed(specs, f"{ECOSYSTEM_TYPE_FIELD}.category"),
                allowed(specs, f"{ECOSYSTEM_TYPE_FIELD}.term"),
                term_to_category[ECOSYSTEM_TYPE_FIELD],
            ),
            "biodiversity_level_labels": quoted(allowed(specs, "biodiversity_level")),
            "species_group_labels": quoted(allowed(specs, "taxa.species_group")),
            "variable_vocabulary": LIST_INDENT + quoted(top_vocabulary(
                paths["variable_vocabulary"], settings["prompt"]["vocabulary_top_n"])),
            "ecosystem_service_category_labels": quoted(allowed(specs, f"{ECOSYSTEM_SERVICE_FIELD}.category")),
            "ecosystem_service_terms": one_per_line(allowed(specs, f"{ECOSYSTEM_SERVICE_FIELD}.term")),
            "min_quote_words": settings["verbatim_check"]["min_quote_words"],
            "max_quote_words": settings["verbatim_check"]["max_quote_words"],
        }
        questions = {
            field + QUESTION_PLACEHOLDER_SUFFIX: question
            for field, question in load_field_questions(schema_paths).items()
        }
        # Fill the label lists now and keep the per-call placeholders for later. The
        # per-call placeholders are passed as themselves, so format() leaves them as they are.
        keep_for_later = {name: "{" + name + "}" for name in PER_CALL_PLACEHOLDERS}
        user_template = sections[USER_SECTION].format(**label_lists, **questions, **keep_for_later)
        return cls(system_text=sections[SYSTEM_SECTION], user_template=user_template)

    @property
    def uses_examples(self) -> bool:
        """True for templates with training examples (v4): the caller must pass `examples`."""
        return EXAMPLES_PLACEHOLDER in self.user_template

    def messages(self, document_id: str, chunk_id: str, text: str,
                 examples: str | None = None) -> list[Message]:
        """System and user message for one call.

        Plain replace() instead of format(): the document text may contain braces, and
        the label lists are already filled in. The examples are filled before the text,
        so a document that happens to contain "{examples}" is sent unchanged.
        """
        if self.uses_examples and examples is None:
            raise ValueError("this prompt has an {examples} section: pass the training examples")
        user_text = self.user_template
        if self.uses_examples:
            user_text = user_text.replace(EXAMPLES_PLACEHOLDER, examples)
        user_text = (
            user_text
            .replace("{document_id}", document_id)
            .replace("{chunk_id}", chunk_id)
            .replace("{chunk_text}", text)
        )
        return [
            {"role": "system", "content": self.system_text},
            {"role": "user", "content": user_text},
        ]
