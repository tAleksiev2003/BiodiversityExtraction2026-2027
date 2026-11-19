"""Build the verification prompt from `paths.prompt_verify` (src/verification.py).

The template has the same `=== SYSTEM ===` / `=== USER ===` sections as the extraction
prompts and three placeholders, all filled per call:

- `{examples}`: the database labels of the most similar training papers, the same text
  as in extraction prompt v4 (src/few_shot_examples.py), so the model sees how often the
  database leaves a field empty for documents like this one;
- `{candidates}`: one line per candidate with its id, field, value and quote;
- `{document_text}`: the text F sent for this paper (or this part of it).

Braces in the template's JSON example are written `{{ }}`, as in the extraction
prompts; they are turned into single braces once, when the template is read.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from llm_cache import Message
from prompts import SYSTEM_SECTION, USER_SECTION, read_sections
from verification import Candidate
from verbatim_checker import LabelEntry

PER_CALL_PLACEHOLDERS = ("examples", "candidates", "document_text")
NULL = "null"
SUBFIELD_SEPARATOR = ", "


def value_text(entry: LabelEntry) -> str:
    """A plain value in double quotes; an object as `name "value"` pairs (null if missing)."""
    if isinstance(entry, str):
        return json.dumps(entry, ensure_ascii=False)
    return SUBFIELD_SEPARATOR.join(
        f"{name} {json.dumps(value, ensure_ascii=False) if value is not None else NULL}"
        for name, value in entry.items()
    )


def candidate_line(candidate: Candidate) -> str:
    """E.g. `- C3 | field: taxa | value: species_group "...", ... | quote: "..."`."""
    quote = json.dumps(candidate.quote, ensure_ascii=False) if candidate.quote is not None else NULL
    return (f"- {candidate.candidate_id} | field: {candidate.field} | "
            f"value: {value_text(candidate.entry)} | quote: {quote}")


def candidates_text(candidates: list[Candidate]) -> str:
    return "\n".join(candidate_line(candidate) for candidate in candidates)


@dataclass(frozen=True)
class VerifyPrompt:
    system_text: str
    user_template: str  # still contains {examples}, {candidates}, {document_text}

    @classmethod
    def from_file(cls, template_path: Path) -> "VerifyPrompt":
        sections = read_sections(template_path)
        # format() with the placeholders passed as themselves: they stay for later, and
        # the {{ }} of the JSON example become single braces.
        keep_for_later = {name: "{" + name + "}" for name in PER_CALL_PLACEHOLDERS}
        return cls(sections[SYSTEM_SECTION], sections[USER_SECTION].format(**keep_for_later))

    def messages(self, examples: str, candidates: list[Candidate], document_text: str) -> list[Message]:
        """System and user message for one call.

        Plain replace() in this order: examples and candidates first, the document text
        last, so braces or placeholder names inside the document are sent unchanged.
        """
        user_text = (
            self.user_template
            .replace("{examples}", examples)
            .replace("{candidates}", candidates_text(candidates))
            .replace("{document_text}", document_text)
        )
        return [
            {"role": "system", "content": self.system_text},
            {"role": "user", "content": user_text},
        ]
