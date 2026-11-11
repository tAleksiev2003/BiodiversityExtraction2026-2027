#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Architecture Model 1: Zero-Shot LLM with Annotation Guidelines
========================================================================================
- Concept & Goal:
    First architecture-level model (Phase 4.1 of the roadmap): a Large Language Model
    performs schema-guided zero-shot information extraction on full biodiversity research
    papers. Inspired by GoLLIE (Sainz et al., 2024) — instead of mere class names, the
    prompt contains the *official annotation guidelines*: the exact German enum values and
    the schema descriptions from `schemas/core_fields.json` + `schemas/biodiversity.json`,
    plus the annotation conventions we discovered during data exploration (empty fields
    are frequent and valid, collective taxonomic groups carry Latin collective names
    such as 'Pisces'/'Aves'/'Plantae' as scientific_name with '-' only as fallback,
    ecosystem_service only on explicit evidence). This model establishes the LLM reference
    point against which all later ablations (Few-Shot RAG, decomposed prompting, chunking)
    will be measured.
- How It Works:
    1. Full-Text Extraction: Extracts the *complete* PDF text per Dev paper with `pypdf`
       (not just the first 2 pages as in the base models) and caches it under
       `data/parsed_text/dev_full/` — the LLM sees Title + Abstract + Methods + Results.
    2. Guideline Prompt: A system prompt defines the annotator role and hard conventions;
       the user prompt embeds (a) the paper metadata + full text, (b) the field
       specification generated directly from the official schema files (descriptions +
       all allowed enum values), and (c) a strict JSON output template.
    3. Provider-Agnostic Client: Any OpenAI-compatible chat API works by switching
       `--provider` (default: Google Gemini via its OpenAI-compatible endpoint, free tier,
       1M-token context). Temperature 0 for maximal determinism; rate-limit aware.
    4. Robust Output Handling: JSON is extracted from raw model output (tolerates markdown
       fences), then every enum field is normalized (exact -> whitespace/case-insensitive
       -> fuzzy difflib match, threshold 0.85; unmatchable values are dropped, not kept as
       false positives). The collective-term convention is enforced as a safety net
       (collective_term=true without a scientific_name -> "-" fallback; missing
       german_name -> the German species_group enum value). One repair
       retry per paper on invalid JSON; total failures fall back to an all-empty prediction.
    5. Reproducibility: Checkpointed per-paper predictions (crashed runs resume for free),
       token accounting, and `save_reproducible_run()` artifacts + leaderboard update.
    Offline Self-Test (`--dry-run`): replaces the API with a mock that returns the
    majority-baseline predictions wrapped in *deliberately malformed* formatting (markdown
    fence + corrupted enum casing + missing "-" convention). The full pipeline must then
    reproduce the known Majority Baseline scores exactly (C1 Macro-F1 = 0.2055), which
    verifies parsing, normalization, and evaluation without spending a single API token.
- Results on Dev Set (100 valid papers):
    * Dry-Run Self-Test  : C1 Macro-F1 = 0.2055 (= Majority Baseline, pipeline verified)
    * Real API Run       : PENDING — execute with:
        GEMINI_API_KEY=<key> python3 src/models/architectures/llm_zeroshot_guidelines.py
      (then update this block with the measured scores)
========================================================================================
"""

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

try:
    import pypdf
except ImportError:  # PDFs are mandatory for this model; fail loudly if missing
    pypdf = None

from src.evaluation.evaluator import (
    evaluate_predictions,
    load_jsonl_by_id,
    print_report,
    save_reproducible_run,
)
from src.models.base_models.majority_baseline import (
    fit_majority_model,
    predict_single_paper as majority_predict_single_paper,
)

# --------------------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------------------

MODEL_ID = "llm_zeroshot_guidelines"
MODEL_TITLE = "Architecture 1: Zero-Shot LLM + Annotation Guidelines (Gemini Flash)"
CATEGORY = "architectures"
SEED = 42

# OpenAI-compatible provider registry: any chat-completion API can be added here.
# `key_env` names the environment variable holding the API key.
PROVIDERS: dict[str, dict[str, Any]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "openai": {
        "base_url": None,  # default OpenAI endpoint
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "google/gemini-2.5-flash",
    },
}

# Fixed prompt conventions distilled from the official schema descriptions and our
# exploratory data analysis of the 908 training annotations.
CONVENTIONS: list[str] = [
    "Output ONLY a single JSON object — no explanations, no markdown fences.",
    "Use the EXACT German enum strings listed in the field specification, character for "
    "character (including umlauts, hyphens, and parentheses).",
    "A field the study does not address gets an EMPTY list []. Empty predictions are "
    "correct and frequent: e.g. ecosystem_service is absent in ~78% of studies, "
    "ecosystem_type in ~50%, biodiversity_level in ~42%.",
    "Never invent values that are not supported by the text.",
    "ecosystem_service: predict ONLY if the study explicitly investigates a service that "
    "nature provides to humans (e.g. pollination, water retention, carbon storage, "
    "recreation). Pure biodiversity monitoring is NOT an ecosystem service.",
    "taxa: list every distinct taxon the study focuses on. For a specific species output "
    "collective_term=false, the Latin binomial as scientific_name, and the German name "
    "(or null if unknown).",
    "taxa: if the study examines a broad taxonomic group or many species at once "
    "(e.g. 'fishes', 'birds', 'vascular plants'), output ONE entry with "
    "collective_term=true. For scientific_name use the LATIN collective name of the "
    "group (e.g. 'Pisces' for fishes, 'Aves' for birds, 'Plantae' or 'Tracheophyta' for "
    "plants, 'Bacteria', 'Mollusca', 'Arthropoda', 'Nematoda'); use \"-\" (a hyphen) "
    "ONLY when the text gives no usable collective designation. german_name = the "
    "German collective name (e.g. 'Fische').",
    "biodiversity_level: distinguish carefully between 'Artenvielfalt' (HOW MANY "
    "species — richness/counts) and 'Zusammensetzung der Artengemeinschaft' (WHICH "
    "species — community composition, identity shifts, community structure).",
    "biodiversity_variable: short German nouns describing WHAT was measured about the "
    "taxa (e.g. \"Abundanz\", \"Artenzahl\", \"Biomasse\", \"Deckungsgrad\").",
]

# Maximum attempts per paper when the model output fails JSON parsing or validation.
MAX_ATTEMPTS = 2


# --------------------------------------------------------------------------------------
# Schema loading: turn the official JSON schemas into a compact field specification
# --------------------------------------------------------------------------------------

def load_field_spec() -> dict[str, dict[str, Any]]:
    """
    Loads both official schema files and extracts, per output field, the human-readable
    German description and the complete list of allowed enum values (if categorical).

    Returns a dict keyed by top-level field name; nested objects (ecosystem_type,
    ecosystem_service, taxa) carry their subfields under 'subfields'.
    """
    core = json.loads((REPO_ROOT / "schemas" / "core_fields.json").read_text(encoding="utf-8"))
    biod = json.loads((REPO_ROOT / "schemas" / "biodiversity.json").read_text(encoding="utf-8"))
    defs_core = core["$defs"]
    defs_biod = biod["$defs"]

    def enum_values(ref: str, defs: dict[str, Any]) -> list[str]:
        """Resolves a '#/$defs/<Name>' reference to its enum value list."""
        name = ref.split("/")[-1]
        return defs[name].get("enum", [])

    def sub(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
        """Resolves a $ref to its full definition dict."""
        return defs[ref.split("/")[-1]]

    habitat_desc = defs_core["HabitatEnum"].get("description", "")
    field_spec: dict[str, dict[str, Any]] = {
        "habitat": {
            "description": habitat_desc
            or "Lebensraum/Lebensräume, die in der Studie untersucht werden.",
            "type": "enum_list",
            "values": enum_values("#/$defs/HabitatEnum", defs_core),
        },
        "biodiversity_level": {
            "description": defs_core["BiodiversityLevelEnum"].get(
                "description", "Aspekte der Biodiversität, die untersucht werden."
            ),
            "type": "enum_list",
            "values": enum_values("#/$defs/BiodiversityLevelEnum", defs_core),
        },
        "biodiversity_variable": {
            "description": biod["properties"]["biodiversity_variable"].get(
                "description", "Was wurde an den Taxa gemessen?"
            ),
            "type": "string_list",
        },
        "ecosystem_type": {
            "description": defs_core["EcosystemTypeSimple"].get("description", ""),
            "type": "object_list",
            "subfields": {
                "category": {
                    "description": "Kategorie des Ökosystemtyps",
                    "values": enum_values("#/$defs/EcosystemTypeCategoryEnum", defs_core),
                },
                "term": {
                    "description": "Name des Ökosystemtyps (null wenn nicht bestimmbar)",
                    "values": enum_values("#/$defs/EcosystemTypeTermEnum", defs_core),
                },
            },
        },
        "ecosystem_service": {
            "description": defs_biod["EcosystemService"].get("description", ""),
            "type": "object_list",
            "subfields": {
                "category": {
                    "description": "Kategorie der Ökosystemdienstleistung",
                    "values": enum_values("#/$defs/EcosystemServiceCategoryEnum", defs_biod),
                },
                "term": {
                    "description": "Konkrete Ökosystemleistung",
                    "values": enum_values("#/$defs/EcosystemServiceTermEnum", defs_biod),
                },
            },
        },
        "taxa": {
            "description": defs_core["Taxa"].get("description", "Untersuchte Taxa."),
            "type": "object_list",
            "subfields": {
                "species_group": {
                    "description": "Taxonomische Gruppe",
                    "values": enum_values("#/$defs/SpeciesGroupEnum", defs_core),
                },
                "scientific_name": {"description": "Wissenschaftlicher Name", "values": None},
                "german_name": {"description": "Deutscher Name", "values": None},
            },
        },
    }
    return field_spec


def format_field_spec(field_spec: dict[str, dict[str, Any]]) -> str:
    """Renders the field specification as a compact, LLM-readable German prompt block."""
    lines: list[str] = []
    for fname, spec in field_spec.items():
        lines.append(f"### Feld `{fname}`")
        if spec.get("description"):
            desc = " ".join(spec["description"].split())  # collapse schema line breaks
            if len(desc) > 600:  # keep the huge ecosystem_type guideline readable
                desc = desc[:600] + " ..."
            lines.append(f"Beschreibung: {desc}")
        if spec["type"] == "enum_list":
            lines.append(f"Erlaubte Werte ({len(spec['values'])}): " + " | ".join(spec["values"]))
        elif spec["type"] == "string_list":
            lines.append("Typ: Liste freier deutscher Kurzbezeichnungen (Strings).")
        else:  # object_list
            for sname, sspec in spec["subfields"].items():
                if sspec.get("values"):
                    lines.append(
                        f"  `{fname}.{sname}` ({len(sspec['values'])} erlaubte Werte): "
                        + " | ".join(sspec["values"])
                    )
                else:
                    lines.append(
                        f"  `{fname}.{sname}`: {sspec['description']} (freier String oder null)"
                    )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Full-text PDF extraction (all pages, cached)
# --------------------------------------------------------------------------------------

def extract_fulltext_cached(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
    cache_dir: Path,
) -> tuple[str, bool]:
    """
    Returns the paper text (original casing!) combining `bibtex_title` and ALL PDF pages.
    In contrast to the base models (first 2 pages, lowercased for keyword matching),
    the LLM reads the natural-cased full document. Results are cached under
    `data/parsed_text/dev_full/` so repeated runs cost nothing.
    """
    title = str(rec.get("bibtex_title") or "").strip()
    cache_file = cache_dir / f"{zot_id}_full.txt"

    if cache_file.exists():
        return f"{title}\n\n{cache_file.read_text(encoding='utf-8')}", True

    pdf_path = pdf_dir / f"{zot_id}.pdf"
    if pypdf is not None and pdf_path.exists():
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            pages_text = "\n".join((page.extract_text() or "") for page in reader.pages)
            pages_text = re.sub(r"[ \t]+", " ", pages_text)  # collapse double spaces
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(pages_text, encoding="utf-8")
            return f"{title}\n\n{pages_text}", True
        except Exception:
            pass
    # Fallback: title only (paper has no readable PDF)
    return title, False


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Du bist ein Experte für die Annotation von Biodiversitäts-Forschungspublikationen "
    "für eine deutsche Forschungsdatenbank. Du extrahierst strukturierte Metadaten aus "
    "wissenschaftlichen Texten und befolgst die offiziellen Annotationsrichtlinien "
    "absolut exakt. Du antwortet ausschließlich mit validem JSON."
)


def build_user_prompt(
    rec: dict[str, Any], paper_text: str, field_spec_str: str, max_chars: int
) -> str:
    """Assembles the full zero-shot guideline prompt for one paper."""
    truncated = paper_text[:max_chars]
    if len(paper_text) > max_chars:
        truncated += "\n[... TEXT ABGESCHNITTEN ...]"
    meta = (
        f"Titel: {rec.get('bibtex_title', '')}\n"
        f"Jahr: {rec.get('bibtex_year', '')}\n"
        f"Typ: {rec.get('bibtex_type', '')}\n"
    )
    json_template = (
        '{\n'
        '  "habitat": ["<HabitatEnum>"],\n'
        '  "biodiversity_level": ["<BiodiversityLevelEnum>"],\n'
        '  "ecosystem_type": [{"category": "<Kategorie>", "term": "<Name oder null>"}],\n'
        '  "taxa": [{"species_group": "<Gruppe>", "scientific_name": "<Latein oder \'-\'>", '
        '"german_name": "<Deutsch oder null>", "collective_term": true|false}],\n'
        '  "biodiversity_variable": ["<Gemessene Variable>"],\n'
        '  "ecosystem_service": [{"category": "<Kategorie>", "term": "<Leistung oder null>"}]\n'
        '}'
    )
    return (
        "# Aufgabe\n"
        "Extrahiere aus der folgenden Forschungspublikation die Felder gemäß der "
        "Feldspezifikation. Gib ausschließlich ein JSON-Objekt mit genau diesen Schlüsseln "
        "zurück: habitat, biodiversity_level, ecosystem_type, taxa, biodiversity_variable, "
        "ecosystem_service.\n\n"
        "# Verbindliche Annotationskonventionen\n"
        + "\n".join(f"- {c}" for c in CONVENTIONS)
        + "\n\n# Feldspezifikation (offizielle Schemas)\n"
        + field_spec_str
        + "\n# JSON-Ausgabeformat (Vorlage)\n"
        + json_template
        + "\n\n# Publikation\n"
        + meta
        + "\n## Volltext\n"
        + truncated
    )


# --------------------------------------------------------------------------------------
# LLM clients (real API + deterministic offline mock for --dry-run)
# --------------------------------------------------------------------------------------

class OpenAICompatClient:
    """Thin wrapper around any OpenAI-compatible chat completion API."""

    def __init__(self, provider: str, model: str, temperature: float, rpm: int):
        from openai import OpenAI  # imported lazily so --dry-run needs no key/package

        cfg = PROVIDERS[provider]
        api_key = os.environ.get(cfg["key_env"])
        if not api_key:
            raise SystemExit(
                f"ERROR: environment variable {cfg['key_env']} is not set.\n"
                f"Get a free Gemini key at https://aistudio.google.com/apikey and run:\n"
                f"  export {cfg['key_env']}='<your key>'"
            )
        self._client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        self.model = model
        self.temperature = temperature
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0
        # Aggregated token accounting for the run report
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_calls = 0

    def chat(self, system: str, user: str) -> str:
        """Sends one chat request, respecting the provider rate limit. Returns raw text."""
        if self._min_interval:
            waited = time.time() - self._last_call
            if waited < self._min_interval:
                time.sleep(self._min_interval - waited)
        self._last_call = time.time()

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            seed=SEED,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self.n_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
        content = response.choices[0].message.content
        return content if content is not None else ""


class DryRunMockClient:
    """
    Deterministic offline mock for pipeline verification (`--dry-run`).

    Instead of an API call it returns the Majority-Baseline predictions for every paper —
    but wrapped in deliberately hostile formatting: markdown code fences, one habitat
    value with corrupted casing/whitespace, and a collective-term taxon whose
    scientific_name violates the '-' convention. If parsing, enum normalization, and
    convention enforcement work correctly, the pipeline must reproduce the Majority
    Baseline's published scores exactly (C1 Macro-F1 = 0.2055).
    """

    def __init__(self, train_records: dict[str, dict[str, Any]]):
        self._learned = fit_majority_model(train_records)

    def chat(self, system: str, user: str) -> str:
        # Recover the paper ID from the user prompt's metadata header is brittle;
        # the pipeline passes the expected record via `mock_context` instead.
        rec = self.mock_context
        pred = majority_predict_single_paper(rec["zotitem_ptr_id"], self._learned)
        # Deliberately sloppy formatting to exercise the repair/normalization path.
        pred["habitat"] = [str(pred["habitat"][0]).upper().replace("  ", " ")]
        if pred.get("taxa"):
            pred["taxa"][0]["collective_term"] = True
            pred["taxa"][0].pop("scientific_name", None)  # '-' convention must be re-added
        return "```json\n" + json.dumps(pred, ensure_ascii=False) + "\n```"


# --------------------------------------------------------------------------------------
# Output parsing, enum normalization, convention enforcement
# --------------------------------------------------------------------------------------

def extract_json_block(raw: str) -> dict[str, Any] | None:
    """
    Extracts the first JSON object from raw LLM output. Tolerates markdown code fences
    and surrounding prose. Returns None if no valid object can be recovered.
    """
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    candidate = fence.group(1) if fence else raw
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(candidate[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        # Common LLM slip: trailing comma before a closing brace/bracket.
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate[start : end + 1])
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def normalize_enum(value: str, allowed: list[str], fuzzy_threshold: float = 0.85) -> str | None:
    """
    Maps a model-generated string onto the closest allowed enum value:
    1. exact match, 2. case/whitespace/punctuation-insensitive match,
    3. fuzzy difflib match above `fuzzy_threshold`. Returns None if nothing fits —
    an invalid value is dropped rather than kept as a guaranteed false positive.
    """
    value = str(value).strip()
    if not value:
        return None
    if value in allowed:
        return value

    def canon(s: str) -> str:
        return re.sub(r"[\s\-–—()/.,;:]+", "", s).lower()

    canon_value = canon(value)
    for cand in allowed:
        if canon(cand) == canon_value:
            return cand

    close = difflib.get_close_matches(value, allowed, n=1, cutoff=fuzzy_threshold)
    return close[0] if close else None


def _clean_string_list(values: Any) -> list[str]:
    """Keeps unique, stripped, non-empty strings from a free-text list field."""
    if not isinstance(values, list):
        return []
    seen, out = set(), []
    for v in values:
        s = str(v).strip() if v is not None else ""
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def normalize_prediction(raw_pred: dict[str, Any], spec: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Validates and normalizes a parsed LLM JSON object into a schema-conform prediction:
    enum fields are mapped onto official values (or dropped), nested object lists are
    rebuilt field by field, duplicates removed, and the collective-term '-'
    convention enforced as a final safety net.
    """
    out: dict[str, Any] = {}

    # --- simple enum lists (habitat, biodiversity_level) ---
    for fname in ("habitat", "biodiversity_level"):
        vals = raw_pred.get(fname)
        if not isinstance(vals, list):
            out[fname] = []
            continue
        norm, seen = [], set()
        for v in vals:
            n = normalize_enum(v, spec[fname]["values"])
            if n and n not in seen:
                seen.add(n)
                norm.append(n)
        out[fname] = norm

    # --- free-text list (biodiversity_variable) ---
    out["biodiversity_variable"] = _clean_string_list(raw_pred.get("biodiversity_variable"))

    # --- nested object lists (ecosystem_type, ecosystem_service, taxa) ---
    for fname in ("ecosystem_type", "ecosystem_service"):
        items = raw_pred.get(fname)
        out[fname] = []
        if not isinstance(items, list):
            continue
        seen: set[tuple] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            cat = normalize_enum(item.get("category"), spec[fname]["subfields"]["category"]["values"])
            if cat is None:  # an object without a valid category is unusable
                continue
            term = item.get("term")
            term = (
                normalize_enum(term, spec[fname]["subfields"]["term"]["values"])
                if term
                else None
            )
            key = (cat, term)
            if key not in seen:
                seen.add(key)
                out[fname].append({"category": cat, "term": term})

    taxa_items = raw_pred.get("taxa")
    out["taxa"] = []
    if isinstance(taxa_items, list):
        seen_taxa: set[tuple] = set()
        for item in taxa_items:
            if not isinstance(item, dict):
                continue
            group = normalize_enum(
                item.get("species_group"), spec["taxa"]["subfields"]["species_group"]["values"]
            )
            if group is None:
                continue
            collective = bool(item.get("collective_term", False))
            sci = str(item.get("scientific_name") or "").strip() or None
            ger = str(item.get("german_name") or "").strip() or None
            if collective:
                # Collective groups: gold usually carries the LATIN collective name
                # ('Pisces', 'Aves', 'Plantae', ...); '-' is only the fallback when the
                # model named no usable designation. The German collective name
                # defaults to the (German) species_group enum value.
                if sci is None:
                    sci = "-"
                if ger is None:
                    ger = group
            key = (group, sci, ger)
            if key not in seen_taxa:
                seen_taxa.add(key)
                out["taxa"].append(
                    {
                        "species_group": group,
                        "scientific_name": sci,
                        "german_name": ger,
                        "collective_term": collective,
                    }
                )
    return out


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot guideline-prompted LLM extraction (Architecture Model 1)."
    )
    parser.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    parser.add_argument(
        "--model", default=None, help="Model name (default: provider default)"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rpm", type=int, default=15, help="Rate limit, requests/minute")
    parser.add_argument(
        "--max-chars", type=int, default=120_000, help="Paper text truncation limit"
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N papers")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Offline pipeline self-test with a deterministic mock (no API key needed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    # --- data loading ---
    gold = load_jsonl_by_id(REPO_ROOT / "data" / "gold" / "dev.jsonl")
    train = load_jsonl_by_id(REPO_ROOT / "data" / "gold" / "train.jsonl")
    dev_ids = [
        line.strip()
        for line in (REPO_ROOT / "data" / "gold" / "dev_100_ids.txt").read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        dev_ids = dev_ids[: args.limit]

    field_spec = load_field_spec()
    field_spec_str = format_field_spec(field_spec)
    pdf_dir = REPO_ROOT / "data" / "raw_pdfs" / "dev"
    fulltext_cache = REPO_ROOT / "data" / "parsed_text" / "dev_full"

    # --- client setup ---
    if args.dry_run:
        client: Any = DryRunMockClient(train)
    else:
        model = args.model or PROVIDERS[args.provider]["default_model"]
        client = OpenAICompatClient(args.provider, model, args.temperature, args.rpm)

    # --- checkpointing: resume interrupted runs without re-spending API tokens ---
    # The mode suffix keeps the offline mock checkpoint from masking a real API run.
    mode_tag = "dryrun" if args.dry_run else "api"
    ckpt_file = (
        REPO_ROOT / "outputs" / CATEGORY / MODEL_ID / f"{MODEL_ID}_checkpoint_{mode_tag}.jsonl"
    )
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[str, Any]] = {}
    if ckpt_file.exists():
        for line in ckpt_file.read_text().splitlines():
            if not line.strip():
                continue
            try:  # tolerate a truncated final line from a crashed run
                obj = json.loads(line)
                predictions[obj["zotitem_ptr_id"]] = obj
            except json.JSONDecodeError:
                continue

    n_retries, n_failures = 0, 0
    for i, zot_id in enumerate(dev_ids, 1):
        if zot_id in predictions:  # already done in a previous (interrupted) run
            continue
        rec = gold[zot_id]
        paper_text, had_pdf = extract_fulltext_cached(zot_id, rec, pdf_dir, fulltext_cache)
        user_prompt = build_user_prompt(rec, paper_text, field_spec_str, args.max_chars)

        if isinstance(client, DryRunMockClient):
            client.mock_context = rec  # the mock needs the record to look up its ID

        # --- generation + parse + repair loop ---
        pred_raw: dict[str, Any] | None = None
        for attempt in range(MAX_ATTEMPTS):
            raw_out = client.chat(SYSTEM_PROMPT, user_prompt)
            parsed = extract_json_block(raw_out)
            if parsed is not None:
                pred_raw = parsed
                break
            n_retries += 1

        if pred_raw is None:
            n_failures += 1
            pred = {f: [] for f in field_spec}  # safe fallback: predict nothing
        else:
            pred = normalize_prediction(pred_raw, field_spec)
        pred["zotitem_ptr_id"] = zot_id
        predictions[zot_id] = pred

        # append-write checkpoint immediately so a crash loses at most one paper
        with open(ckpt_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        status = "OK" if pred_raw is not None else "PARSE-FAIL"
        print(
            f"[{i:>3}/{len(dev_ids)}] {zot_id} {status} "
            f"habitat={len(pred['habitat'])} taxa={len(pred['taxa'])} "
            f"(pdf={'yes' if had_pdf else 'NO'})",
            flush=True,
        )

    # --- evaluation + reproducible artifacts ---
    metrics = evaluate_predictions(gold, predictions, filter_ids=set(dev_ids))
    print_report(f"{MODEL_TITLE} {'[DRY-RUN]' if args.dry_run else ''}", metrics)

    token_stats = {
        "n_calls": getattr(client, "n_calls", 0),
        "prompt_tokens": getattr(client, "prompt_tokens", 0),
        "completion_tokens": getattr(client, "completion_tokens", 0),
        "n_repair_retries": n_retries,
        "n_parse_failures": n_failures,
    }
    runtime = time.time() - start_time
    save_reproducible_run(
        model_id=MODEL_ID,
        model_title=MODEL_TITLE + (" [DRY-RUN MOCK]" if args.dry_run else ""),
        category=CATEGORY,
        predictions=predictions,
        metrics=metrics,
        runtime_seconds=runtime,
        seed=SEED,
        config={
            "mode": "dry_run_mock" if args.dry_run else "api",
            "provider": args.provider,
            "model": getattr(client, "model", "majority-baseline-mock"),
            "temperature": args.temperature,
            "max_chars": args.max_chars,
            "n_papers": len(dev_ids),
            "prompt_conventions": CONVENTIONS,
            "token_stats": token_stats,
        },
    )
    print(
        f"\nDone in {runtime:.1f}s | calls={token_stats['n_calls']} "
        f"| prompt_tokens={token_stats['prompt_tokens']} "
        f"| completion_tokens={token_stats['completion_tokens']} "
        f"| repair_retries={n_retries} | parse_failures={n_failures}"
    )


if __name__ == "__main__":
    main()
