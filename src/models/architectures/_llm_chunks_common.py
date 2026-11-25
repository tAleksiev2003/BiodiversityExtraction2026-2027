#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Shared Building Blocks for Chunked LLM Pipelines
========================================================================================
1. Concept & Goal (shared stages of the team's agreed chunk pipeline, Google-Doc diagram):
   PDF -> text -> section chunks C1..Cn, then quote verification -> EN->DE mapping
   -> predictions.jsonl. Serves two architectures: sequential state fold
   (llm_sequential_chunks.py) and parallel vote merge (llm_parallel_chunks.py).
2. How It Works (this module): provider registry (HU bridge, SAIA, Gemini),
   heading-heuristic section chunking over cached pypdf full text, compact
   English-label chunk prompts, fuzzy quote verification, deterministic EN->DE
   enum mapping, canonical dedup, and mock clients for offline dry runs.
3. Results on Dev Set: no runs of its own (shared helper, not a model);
   see llm_sequential_chunks.py and llm_parallel_chunks.py for measured scores.
========================================================================================
Implements the shared stages of the team's agreed chunk pipeline (Google-Doc diagram):

    PDF -> text -> section chunks C1..Cn
      A) Sequential: LLM fold over chunks (state passed on)   -> llm_sequential_chunks.py
      B) Parallel  : independent LLM call per chunk           -> llm_parallel_chunks.py
    Python: verify quotes -> map EN->DE labels -> predictions.jsonl

Design decisions (agreed in the team doc, 04.11./11.11.2026):
- *Section chunking instead of fixed token windows* (Shaukat et al. 2026: dynamic /
  paragraph-group chunking beats fixed-size chunks for biology papers). GROBID is not
  available locally (no Docker on the Mac), so we approximate its TEI section output
  with a heading heuristic over the pypdf full text (same text source as Model 1).
- *English labels in the prompt + deterministic EN->DE mapping in Python* (Minh Anh's
  translate-last note: LLMs reason better in English, exact German enum strings are
  resolved deterministically afterwards). German enum output from the model is still
  accepted as a fallback (matching is German-first).
- *Every label must carry a verbatim quote from its chunk*; labels whose quote cannot
  be fuzzy-verified in the chunk text are dropped (hallucination filter).
- *Compact schema in chunk prompts*: the six short enums (habitat 6, biodiversity
  level 7, species-group 16, ecosystem type categories 6, service categories 3) are
  listed in English; the two huge enums (ecosystem type terms 46, service terms 67)
  are NOT listed per chunk (token cost x ~10 chunks/paper). Terms are requested as
  free text and resolved against the German enums in Python (SchemaRAG-style schema
  reduction, Yuquan's paper).
- Provider layer is OpenAI-compatible (HU bridge 'hu-reasoning' = Qwen3.6-27B on
  llm3 with thinking enabled and sampling pinned server-side; client temperature is
  NOT forwarded for such providers — Qwen3 thinking at temperature 0 loops).
========================================================================================
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------------------
# Provider registry (subset of Model 1's registry; hu bridge is the default target)
# --------------------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, Any]] = {
    "hu": {
        "base_url": "http://127.0.0.1:4000/v1",
        "key_env": "LLM_BRIDGE_KEY",
        "key_default": "not-needed",
        "default_model": "hu-reasoning",
        # Server pins temperature=0.6 / top_p=0.95 / top_k=20 with thinking on.
        "pin_sampling": True,
    },
    # GWDG SAIA (https://docs.hpc.gwdg.de/services/ai-services/saia/): OpenAI-
    # compatible academic cloud API. Key via KISSKI booking (SAIA self-service
    # portal), same email as the Academic Cloud account. Verify exact model ids
    # against GET /v1/models after obtaining the key.
    "saia": {
        "base_url": "https://chat-ai.academiccloud.de/v1",
        "key_env": "SAIA_API_KEY",
        # Best available as of 2026-11-12 (3-way probe: all flagships 3/3 on the
        # enum-exactness probe; mistral-large-4 = largest, 7s vs 13s for qwen3.5).
        # Speed alternative for iteration: glm-5.3 (~0.6s), also 3/3.
        "default_model": "mistral-large-4-1t-a52b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
}

SEED = 42


class OpenAICompatClient:
    """Thin OpenAI-compatible chat client with RPM throttling and token accounting."""

    def __init__(self, provider: str, model: str, temperature: float, rpm: int):
        from openai import OpenAI

        cfg = PROVIDERS[provider]
        api_key = os.environ.get(cfg["key_env"]) or cfg.get("key_default")
        if not api_key:
            raise SystemExit(f"ERROR: environment variable {cfg['key_env']} is not set.")
        self._client = OpenAI(api_key=api_key, base_url=cfg["base_url"], timeout=900.0)
        self.model = model
        self.temperature = temperature
        self.pin_sampling = bool(cfg.get("pin_sampling"))
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_calls = 0

    def chat(self, system: str, user: str) -> str:
        if self._min_interval:
            waited = time.time() - self._last_call
            if waited < self._min_interval:
                time.sleep(self._min_interval - waited)
        self._last_call = time.time()
        extra: dict[str, Any] = {}
        if not self.pin_sampling:
            extra = {"temperature": self.temperature, "seed": SEED}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extra,
        )
        self.n_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
        content = response.choices[0].message.content
        return content if content is not None else ""


# --------------------------------------------------------------------------------------
# German enum values (single source of truth: the two official schema files)
# --------------------------------------------------------------------------------------

def _enum(schema: str, name: str) -> list[str]:
    data = json.loads((REPO_ROOT / "schemas" / schema).read_text(encoding="utf-8"))
    return data["$defs"][name]["enum"]

ENUMS = {
    "habitat": _enum("core_fields.json", "HabitatEnum"),
    "biodiversity_level": _enum("core_fields.json", "BiodiversityLevelEnum"),
    "ecosystem_type_category": _enum("core_fields.json", "EcosystemTypeCategoryEnum"),
    "ecosystem_type_term": _enum("core_fields.json", "EcosystemTypeTermEnum"),
    "species_group": _enum("core_fields.json", "SpeciesGroupEnum"),
    "ecosystem_service_category": _enum("biodiversity.json", "EcosystemServiceCategoryEnum"),
    "ecosystem_service_term": _enum("biodiversity.json", "EcosystemServiceTermEnum"),
}

# English prompt labels per enum. The model outputs these; Python maps them to the
# exact German schema strings via EN_TO_DE below (canonical lowercase matching).
EN_LABELS = {
    "habitat": [
        "agriculture and open land", "freshwaters and floodplains", "soil",
        "coasts and coastal waters", "urban areas", "forest",
    ],
    "biodiversity_level": [
        "species richness", "functional diversity", "genetic diversity",
        "habitat diversity", "phylogenetic diversity", "structural diversity",
        "community composition",
    ],
    "species_group": [
        "amphibians and reptiles", "other", "other plants", "other invertebrates",
        "archaea", "bacteria", "fish", "lichens", "vascular plants", "insects",
        "mosses", "fungi", "protists", "protozoa", "mammals", "birds",
    ],
    "ecosystem_type_category": [
        "marine and coastal biotopes", "inland water biotopes",
        "terrestrial and semi-terrestrial inland biotopes", "technical biotopes",
        "alpine biotopes", "other biotopes",
    ],
    "ecosystem_service_category": [
        "provisioning services", "regulating and maintenance services", "cultural services",
    ],
}

def _pairs(habitat: list[str], levels: list[str], cats: list[str], groups: list[str],
           scats: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Zips English labels to German enum values in schema order."""
    return dict(zip(
        ["habitat", "biodiversity_level", "ecosystem_type_category",
         "species_group", "ecosystem_service_category"],
        [list(zip(habitat, ENUMS["habitat"])), list(zip(levels, ENUMS["biodiversity_level"])),
         list(zip(cats, ENUMS["ecosystem_type_category"])),
         list(zip(groups, ENUMS["species_group"])),
         list(zip(scats, ENUMS["ecosystem_service_category"]))],
    ))

_EN_PAIRS = _pairs(EN_LABELS["habitat"], EN_LABELS["biodiversity_level"],
                   EN_LABELS["ecosystem_type_category"], EN_LABELS["species_group"],
                   EN_LABELS["ecosystem_service_category"])

# Long-enum English keys -> German values. Only the frequently used ecosystem-type
# terms are listed; anything else resolves via fuzzy matching or stays null.
_ECOSYSTEM_TERMS_EN = {
    "arable fields and fallows": "Äcker und Ackerbrachen",
    "north sea benthos": "Benthal der Nordsee",
    "baltic sea benthos": "Benthal der Ostsee",
    "north sea pelagic": "Pelagial der Nordsee",
    "baltic sea pelagic": "Pelagial der Ostsee",
    "flowing waters": "Fließende Gewässer",
    "standing waters": "Stehende Gewässer",
    "raised bogs and transitional mires": "Hoch-, Zwischen- und Übergangsmoore",
    "fens and marshes wet grassland": "Waldfreie Niedermoore und Sümpfe, Grünland nasser bis feuchter Standorte",
    "deciduous and mixed forests": "Laub(Misch)Wälder und -Forste (Laubbaumanteil über 50 Prozent)",
    "coniferous and mixed forests": "Nadel(Misch)Wälder und -Forste",
    "dry grassland": "Trockenrasen sowie Grünland trockener bis frischer Standorte",
    "dwarf shrub heaths": "Zwergstrauchheiden",
    "coastal dunes": "Küstendünen",
    "buildings": "Bauwerke",
    "green and urban open spaces": "Grün- und Freiflächen",
    "traffic infrastructure and squares": "Verkehrsanlagen und Plätze",
    "hedgerows and field groves": "Feldgehölze, Gebüsche, Hecken und Gehölzkulturen",
    "riparian and forest fringes": "Wald- und Ufersäume, Staudenfluren",
    "reed beds": "Röhrichte (ohne Brackwasserröhrichte)",
    "springs": "Quellen (inklusive Quellabfluss (Krenal))",
    "salt marshes": "Salzgrünland der Nordseeküste (Supralitoral)",
    "rocks and scree": "Felsen, Block- und Schutthalden, Geröllfelder, offene Bereiche mit sandigem oder bindigem Substrat",
    "caves": "Höhlen (einschließlich Stollen, Brunnenschächte etc.)",
    "groundwater": "Grundwasser und Höhlengewässer",
    "subalpine forests": "Subalpine Wälder",
    "mountain grassland": "Gebirgsrasen (subalpine bis alpine Stufe)",
    "glaciers and snowfields": "Firn, permanente Schneefelder und Gletscher",
    "beaches": "Sände, Sand-, Geröll- und Blockstrände",
    "rocky coasts": "Fels- und Steilküsten",
    "landfills": "Deponien und Rieselfelder",
    "ruderal vegetation": "Anthropogene Rohbodenstandorte und Ruderalfluren",
}

# Ecosystem-service terms: English keys -> full German CICES strings. The field is
# absent in ~78% of gold papers and scored F1=0 for every baseline so far; we map the
# common entries and leave rare ones null rather than guessing CICES codes.
_SERVICE_TERMS_EN = {
    "pollination": "2.2.2.1 Bestäubung (oder 'Gameten'-Ausbreitung in einem marinen Kontext)",
    "pest control": "2.2.3.1 Schädlingsbekämpfung (einschließlich invasiver Arten)",
    "disease control": "2.2.3.2 Krankheitsbekämpfung",
    "water regulation": "2.2.1.3 Wasserkreislauf und Wasserflussregulierung (einschließlich Hochwasserschutz und Küstenschutz)",
    "temperature and humidity regulation": "2.2.6.2 Regulierung von Temperatur und Feuchtigkeit, einschließlich Belüftung und Transpiration",
    "climate and atmospheric regulation": "2.2.6.1 Regulierung der chemischen Zusammensetzung der Atmosphäre und der Ozeane",
    "carbon storage and sequestration": "2.2.6.1 Regulierung der chemischen Zusammensetzung der Atmosphäre und der Ozeane",
    "erosion control": "2.2.1.1 Kontrolle der Erosionsraten",
    "seed dispersal": "2.2.2.2 Ausbreitung von Saatgut",
    "habitat maintenance": "2.2.2.3 Aufrechterhaltung von Aufwuchspopulationen und Lebensräumen (einschließlich Schutz des Genpools)",
    "water filtration and purification": "2.2.5.1 Regulierung des chemischen Zustands von Süßgewässern durch lebende Prozesse",
    "mass stabilization": "2.2.1.2 Pufferung und Abschwächung von Massenbewegungen",
    "wind protection": "2.2.1.4 Windschutz",
    "fire protection": "2.2.1.5 Feuerschutz",
    "soil formation and weathering": "2.2.4.1 Verwitterungsprozesse und ihre Auswirkungen auf die Bodenqualität",
    "decomposition and soil quality": "2.2.4.2 Zersetzungs- und Fixierungsprozesse und ihre Auswirkungen auf die Bodenqualität",
    "bio-remediation": "2.1.1.1 Biosanierung durch Mikroorganismen, Algen, Pflanzen und Tiere",
    "filtration and sequestration": "2.1.1.2 Filtration/Sequestrierung/Speicherung/Akkumulation durch Mikroorganismen, Algen, Pflanzen und Tiere",
    "noise reduction": "2.1.2.2 Reduktion von Lärm",
    "recreation and health": "3.1.1.2 Merkmale lebender Systeme, die Aktivitäten zur Förderung von Gesundheit, Erholung oder Vergnügen durch passive oder aktive Erholung ermöglichen",
    "outdoor recreation": "3.1.1.2 Merkmale lebender Systeme, die Aktivitäten zur Förderung von Gesundheit, Erholung oder Vergnügen durch passive oder aktive Erholung ermöglichen",
    "scientific research": "3.1.2.1 Merkmale lebender Systeme, die eine wissenschaftliche Untersuchung oder die Schaffung von traditionell ökologischen Erkenntnissen ermöglichen",
    "education": "3.1.2.2 Merkmale lebender Systeme, die Bildung und Ausbildung ermöglichen",
    "aesthetic experiences": "3.1.2.4 Merkmale lebender Systeme, die ästhetische Erfahrungen ermöglichen",
    "cultural heritage": "3.1.2.3 Merkmale lebender Systeme, die in Bezug auf die Kultur und Kulturerbe eine Rolle spielen",
    "wild plants for food": "1.1.5.1 Wildpflanzen (Land- und Wasserpflanzen, einschließlich Pilze und Algen), die zur Ernährung genutzt werden",
    "wild animals for food": "1.1.6.1 Wildtiere (Land- und Wassertiere), die zu Ernährungszwecke verwendet werden",
    "cultivated food plants": "1.1.1.1 Kultivierte Landpflanzen (einschließlich Pilze und Algen), die zu Ernährungszwecken angebaut werden",
    "drinking water": "4.2.1.1 Oberflächenwasser als Trinkwasser",
}

_TERM_MAPS = {
    "ecosystem_type_term": _ECOSYSTEM_TERMS_EN,
    "ecosystem_service_term": _SERVICE_TERMS_EN,
}


def canon(s: str) -> str:
    """Canonical form for label matching: lowercase, no spaces/punctuation."""
    return re.sub(r"[\s\-–—()/.,;:'\"&]+", "", str(s)).lower()


def map_label_to_enum(value: str, enum_key: str) -> str | None:
    """
    Maps a model-generated label onto the exact German enum value.
    Order: exact German -> canonical German -> canonical English dict -> CICES code
    prefix (service terms) -> fuzzy German (ratio >= 0.85). None if unmatchable.
    """
    if not value:
        return None
    value = str(value).strip()
    allowed = ENUMS[enum_key]
    if value in allowed:
        return value
    cv = canon(value)
    for cand in allowed:
        if canon(cand) == cv:
            return cand
    pairs = _EN_PAIRS.get(enum_key)
    if pairs:
        for en, de in pairs:
            if canon(en) == cv:
                return de
    term_map = _TERM_MAPS.get(enum_key)
    if term_map:
        for en, de in term_map.items():
            if canon(en) == cv:
                return de
        m = re.match(r"^([1-4]\.\d(?:\.\d{1,2})?(?:\.X)?)\b", value)
        if m:  # model answered with a CICES code like "2.2.3.1 Schädlingsbekämpfung..."
            prefix = m.group(1)
            for cand in allowed:
                if cand.startswith(prefix + " "):
                    return cand
    close = difflib.get_close_matches(value, allowed, n=1, cutoff=0.85)
    return close[0] if close else None


# --------------------------------------------------------------------------------------
# Section chunking (GROBID substitute): heading heuristic over the full text
# --------------------------------------------------------------------------------------

_HEADING_KEYWORDS = (
    "abstract", "introduction", "background", "related work", "methods",
    "methodology", "materials and methods", "materials", "study area", "study site",
    "study region", "results", "findings", "discussion", "conclusions", "conclusion",
    "summary", "outlook", "limitations", "data analysis", "statistical analysis",
    "experimental", " Acknowledg", "references", "literature", "appendix", "supplement",
)
_HEADING_RE = re.compile(
    r"^\s*(?:\(?\d{1,2}(?:\.\d{1,2}){0,2}\)?[.)]?\s+|[IVXLC]{1,5}[.)]\s+)?[A-Z]"
)


def split_into_sections(full_text: str, max_section_chars: int = 6500) -> list[dict[str, str]]:
    """
    Splits paper text into section chunks at detected headings. Long sections are cut
    at paragraph boundaries; boilerplate sections (references, acknowledgements) are
    dropped to save tokens. Returns [{'title': ..., 'text': ...}, ...].
    """
    lines = full_text.splitlines()
    sections: list[tuple[str, list[str]]] = [("Header", [])]
    for line in lines:
        stripped = line.strip()
        is_heading = (
            0 < len(stripped) <= 80
            and not stripped.endswith((".", ",", ";", ":", ")"))
            and (
                any(stripped.lower().startswith(k.strip()) for k in _HEADING_KEYWORDS)
                or (
                    _HEADING_RE.match(stripped) is not None
                    and re.match(r"^\s*\d{1,2}[.)]\s+\S", stripped) is not None
                )
                or (stripped.upper() == stripped and re.search(r"[A-ZÄÖÜ]{3}", stripped))
            )
        )
        if is_heading and len(" ".join(sections[-1][1])) > 200:
            sections.append((stripped[:70], []))
        else:
            sections[-1][1].append(line)

    out: list[dict[str, str]] = []
    drop_kw = ("references", "literature cited", "acknowledg", "appendix", "supplement", "author contribution", "declaration", "conflict of interest", "funding statement")
    for title, body in sections:
        text = "\n".join(body)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text or len(text) < 120:
            continue
        if any(k in title.lower() for k in drop_kw):
            continue
        # Long sections: split at paragraph boundaries into <= max_section_chars parts.
        parts: list[str] = [""]
        for para in text.split("\n\n"):
            if len(parts[-1]) + len(para) > max_section_chars and parts[-1]:
                parts.append("")
            parts[-1] += para + "\n\n"
        for j, part in enumerate(parts):
            part = part.strip()
            if len(part) >= 120:
                out.append({"title": title if j == 0 else f"{title} (part {j + 1})", "text": part[:max_section_chars]})
    return out


def extract_fulltext(zot_id: str, rec: dict[str, Any]) -> str:
    """Full paper text (cached by Model 1 under data/parsed_text/dev_full/)."""
    cache = REPO_ROOT / "data" / "parsed_text" / "dev_full" / f"{zot_id}_full.txt"
    title = str(rec.get("bibtex_title") or "").strip()
    if cache.exists():
        return f"{title}\n\n{cache.read_text(encoding='utf-8')}"
    import pypdf

    pdf = REPO_ROOT / "data" / "raw_pdfs" / "dev" / f"{zot_id}.pdf"
    if pdf.exists():
        try:
            reader = pypdf.PdfReader(str(pdf))
            pages = "\n".join((p.extract_text() or "") for p in reader.pages)
            pages = re.sub(r"[ \t]+", " ", pages)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(pages, encoding="utf-8")
            return f"{title}\n\n{pages}"
        except Exception:
            pass
    return title


# --------------------------------------------------------------------------------------
# Quote verification (fuzzy): a label only counts if its quote appears in the chunk
# --------------------------------------------------------------------------------------

def _norm_quote(s: str) -> str:
    return re.sub(r"[\s\-–—'\"`()]+", "", str(s)).lower()


def verify_quote(quote: str, chunk_text: str, min_ratio: float = 0.7) -> bool:
    """
    True if `quote` plausibly occurs verbatim in `chunk_text`: either as a normalized
    substring, or covering >= min_ratio of the quote via longest common block.
    """
    if not quote or not chunk_text:
        return False
    qn, cn = _norm_quote(quote), _norm_quote(chunk_text)
    if not qn:
        return False
    if qn in cn:
        return True
    match = difflib.SequenceMatcher(None, cn, qn, autojunk=False).find_longest_match(0, len(cn), 0, len(qn))
    return match.size / len(qn) >= min_ratio


# --------------------------------------------------------------------------------------
# Prompts (English reasoning, verbatim-quote requirement, compact schema)
# --------------------------------------------------------------------------------------

CHUNK_SYSTEM_PROMPT = (
    "You are a meticulous annotation assistant for a German biodiversity research "
    "database. You read ONE section of a scientific paper and list the study-level "
    "metadata labels that this section EVIDENCES. Every label must be justified by a "
    "verbatim quote copied exactly from the section text. You answer with valid JSON "
    "only - no markdown fences, no explanations."
)

_LABEL_LINES = {
    key: " | ".join(labels) for key, labels in EN_LABELS.items()
}


def build_chunk_prompt(section: dict[str, str], state_block: str | None = None) -> str:
    """One chunk-extraction prompt. With `state_block` set, asks for an updated state
    (Sequential approach A); without it, asks only for labels found in this section
    (Parallel approach B)."""
    task = (
        "Extract all labels that THIS section evidences about the study. Rules:\n"
        "- A label counts only if the section contains explicit textual evidence.\n"
        '- Missing information means an empty list []. An empty section is NOT evidence against a label.\n'
        '- "quote" must be copied VERBATIM from the section text (max ~25 words), or the label is invalid.\n'
        '- Use ONLY the listed English labels for enum fields.\n'
        "- For ecosystem_type.term / ecosystem_service.term output the exact German term if you know it, "
        "else a short English description, else null.\n"
        "- Free-text values (biodiversity_variable, taxa names) in German / Latin binomials.\n"
        "- taxa: one entry per distinct taxon; collective_term=true for broad groups (fishes, birds, "
        "vascular plants, ...) with the LATIN collective name as scientific_name (e.g. Pisces, Aves, "
        "Plantae) and '-' only if no usable designation; german_name in German if known."
    )
    if state_block:
        task += (
            "\n\nYou receive the CURRENT STATE of labels already collected from EARLIER "
            "sections. Return the UPDATED COMPLETE state as JSON: keep labels that still "
            "hold (with their original quote), ADD new labels evidenced by THIS section "
            "(quote from THIS section), REMOVE labels that this section clearly "
            "contradicts. Keep the exact same JSON structure."
        )
    schema_hint = (
        "{\n"
        '  "habitat": [{"label": "<one of: ' + _LABEL_LINES["habitat"] + '>", "quote": "..."}],\n'
        '  "biodiversity_level": [{"label": "<one of: ' + _LABEL_LINES["biodiversity_level"] + '>", "quote": "..."}],\n'
        '  "biodiversity_variable": [{"value": "<German short noun>", "quote": "..."}],\n'
        '  "ecosystem_type": [{"category": "<one of: ' + _LABEL_LINES["ecosystem_type_category"] + '>", "term": "<German term, English short description, or null>", "quote": "..."}],\n'
        '  "taxa": [{"species_group": "<one of: ' + _LABEL_LINES["species_group"] + '>", "scientific_name": "<Latin binomial, Latin collective name, or \'-\'>", "german_name": "<German name or null>", "collective_term": true, "quote": "..."}],\n'
        '  "ecosystem_service": [{"category": "<one of: ' + _LABEL_LINES["ecosystem_service_category"] + '>", "term": "<German CICES term, English short description, or null>", "quote": "..."}]\n'
        "}"
    )
    parts = ["# Task\n" + task]
    if state_block:
        parts.append("# Current state (earlier sections)\n" + state_block)
    parts.append("# Output JSON schema\n" + schema_hint)
    parts.append(f"# Section: {section['title']}\n\n{section['text']}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Output parsing / validation of chunk JSON
# --------------------------------------------------------------------------------------

def extract_json_block(raw: str) -> dict[str, Any] | None:
    """First JSON object in raw LLM output; tolerates fences and trailing commas."""
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    candidate = fence.group(1) if fence else raw
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    snippet = candidate[start : end + 1]
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def parse_chunk_items(parsed: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Validates the raw chunk/state JSON into per-field item lists, keeping the quote.
    Values are NOT yet mapped to German enums (done once, after merging/voting).
    """
    out: dict[str, list[dict[str, Any]]] = {}

    def _items(field: str) -> list[dict[str, Any]]:
        raw = parsed.get(field)
        return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []

    for field in ("habitat", "biodiversity_level"):
        vals = []
        for it in _items(field):
            label = str(it.get("label") or it.get("value") or "").strip()
            quote = str(it.get("quote") or "").strip()
            if label and quote:
                vals.append({"label": label, "quote": quote})
        out[field] = vals

    vals = []
    for it in _items("biodiversity_variable"):
        value = str(it.get("value") or it.get("label") or "").strip()
        quote = str(it.get("quote") or "").strip()
        if value and quote:
            vals.append({"value": value, "quote": quote})
    out["biodiversity_variable"] = vals

    for field in ("ecosystem_type", "ecosystem_service"):
        vals = []
        for it in _items(field):
            cat = str(it.get("category") or "").strip()
            term = it.get("term")
            quote = str(it.get("quote") or "").strip()
            if cat and quote:
                vals.append({
                    "category": cat,
                    "term": str(term).strip() if isinstance(term, str) and term.strip() else None,
                    "quote": quote,
                })
        out[field] = vals

    vals = []
    for it in _items("taxa"):
        group = str(it.get("species_group") or "").strip()
        quote = str(it.get("quote") or "").strip()
        sci = str(it.get("scientific_name") or "").strip() or None
        ger = str(it.get("german_name") or "").strip() or None
        if group and quote:
            vals.append({
                "species_group": group, "scientific_name": sci, "german_name": ger,
                "collective_term": bool(it.get("collective_term", False)), "quote": quote,
            })
    out["taxa"] = vals
    return out


# --------------------------------------------------------------------------------------
# Final prediction assembly (EN->DE mapping, dedup, collective-term conventions)
# --------------------------------------------------------------------------------------

def assemble_prediction(items: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """
    Turns merged/voted raw items into a schema-conform German prediction record
    (same shape as Model 1): enums mapped exactly, duplicates removed, '-' and
    german_name conventions for collective taxa enforced.
    """
    pred: dict[str, Any] = {}

    for field in ("habitat", "biodiversity_level"):
        seen, vals = set(), []
        for it in items.get(field, []):
            de = map_label_to_enum(it["label"], field)
            if de and de not in seen:
                seen.add(de)
                vals.append(de)
        pred[field] = vals

    seen, vals = set(), []
    for it in items.get("biodiversity_variable", []):
        v = it["value"]
        if v and canon(v) not in seen:
            seen.add(canon(v))
            vals.append(v)
    pred["biodiversity_variable"] = vals[:6]  # cap: gold rarely exceeds a handful

    for field, cat_key, term_key in (
        ("ecosystem_type", "ecosystem_type_category", "ecosystem_type_term"),
        ("ecosystem_service", "ecosystem_service_category", "ecosystem_service_term"),
    ):
        cat_votes: dict[str, int] = {}
        term_votes: dict[str, dict[str, int]] = {}
        for it in items.get(field, []):
            cat_de = map_label_to_enum(it["category"], cat_key)
            if cat_de is None:
                continue
            cat_votes[cat_de] = cat_votes.get(cat_de, 0) + 1
            if it.get("term"):
                term_de = map_label_to_enum(it["term"], term_key)
                if term_de:
                    term_votes.setdefault(cat_de, {})
                    term_votes[cat_de][term_de] = term_votes[cat_de].get(term_de, 0) + 1
        out_list = []
        for cat_de in sorted(cat_votes, key=cat_votes.get, reverse=True):
            best_term = None
            if term_votes.get(cat_de):
                best_term = max(term_votes[cat_de], key=term_votes[cat_de].get)
            out_list.append({"category": cat_de, "term": best_term})
        pred[field] = out_list

    # taxa: dedupe on (group, sci, ger); enforce collective-term conventions.
    seen, taxa = set(), []
    for it in items.get("taxa", []):
        group_de = map_label_to_enum(it["species_group"], "species_group")
        if group_de is None:
            continue
        sci, ger = it.get("scientific_name"), it.get("german_name")
        if it.get("collective_term"):
            sci = sci or "-"
            ger = ger or group_de
        key = canon(f"{group_de}|{sci}|{ger}")
        if key in seen:
            continue
        seen.add(key)
        taxa.append({
            "species_group": group_de, "scientific_name": sci,
            "german_name": ger, "collective_term": bool(it.get("collective_term")),
        })
    pred["taxa"] = taxa[:12]  # cap over-firing (SVM error analysis: too many taxa)
    return pred


# --------------------------------------------------------------------------------------
# Deterministic offline mocks (--dry-run): exercise fold/merge/verify without API
# --------------------------------------------------------------------------------------

class SequentialMockClient:
    """Adds one verifiable habitat label per fold step (quote from current chunk)."""

    habitat_labels = ["forest", "agriculture and open land", "coasts and coastal waters"]

    def __init__(self):
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._current_chunk: dict[str, str] = {}

    def chat(self, system: str, user: str) -> str:
        label = self.habitat_labels[self.n_calls % len(self.habitat_labels)]
        words = self._current_chunk["text"].split()
        quote = " ".join(words[10:22]) if len(words) > 22 else " ".join(words[:8])
        self.n_calls += 1
        return json.dumps({
            "habitat": [{"label": label, "quote": quote}],
            "biodiversity_level": [],
            "biodiversity_variable": [],
            "ecosystem_type": [],
            "taxa": [{"species_group": "vascular plants", "scientific_name": "Plantae",
                      "german_name": None, "collective_term": True, "quote": quote}],
            "ecosystem_service": [],
        })


class ParallelMockClient(SequentialMockClient):
    """Same per-chunk answers; the difference is only the orchestration around it."""
