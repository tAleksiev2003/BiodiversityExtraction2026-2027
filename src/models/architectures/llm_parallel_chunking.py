#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Architecture Model 2: Parallel Chunking & Quote-Verified Voting
========================================================================================
- Concept & Goal:
    Implements the group's planned "Parallel Pipeline (B)" from the team architecture
    diagram (PDF -> Chunks C1..Cn -> Parallel independent LLM calls -> Quote
    Verification -> Merge by Votes -> Final JSON predictions).
    It directly tests Architecture Axis 1:
      * Full-Text in One Monolithic Prompt (Architecture 1: `llm_zeroshot_guidelines.py`)
        vs.
      * Section Chunking + Independent Parallel Extraction + Evidence Voting (this file)
    Hypothesis: processing document sections independently with anti-hallucination
    quote verification prevents context dilution and false-positive explosion on
    long ecological papers.

- How It Works:
    1. Document Sectioning / Chunking:
       Extracts text page-by-page (pypdf) and segments the paper into logical
       multi-page chunks (Title/Abstract/Intro, Study Area/Methods, Results,
       Discussion/Management), preventing "lost in the middle" attenuation on 20+
       page papers. Papers without a readable PDF fall back to a title-only chunk.
    2. Guided Chain-of-Thought with Verbatim Quotes:
       Calls an OpenAI-compatible LLM (`mistral-large-4-1t-a52b` via SAIA or
       `hu-reasoning` via the local HU bridge) on each chunk independently. Every
       candidate finding MUST carry an exact `verbatim_quote` from that chunk as
       grounded evidence; categories absent from the section are omitted.
    3. Python Anti-Hallucination Gate (Quote Verification):
       Each quote is validated against the raw chunk text via (a) normalized exact
       substring match, (b) 30-character leading anchor match, and (c) fuzzy
       alignment (>= 75% of the quote found contiguously in the chunk). Findings
       whose quotes are not grounded in the text are discarded as hallucinations.
    4. Vote Merging & Sparsity-Aware Resolution:
       Votes for a label = number of distinct chunks with verified evidence (an
       empty chunk is NOT a negative vote — evidence is naturally localized).
       Dense fields (habitat, taxa) fall back to training-set majority values when
       no verified evidence exists; sparse fields (ecosystem_service, ~78% empty in
       gold) are only emitted with verified support.
    5. Collective-Term & Fallback Discipline:
       Broad species groups are emitted with `scientific_name = "-"` and the German
       species-group enum as `german_name`, enforced in Python regardless of what
       the model returned.

- Results on Dev Set (100 valid papers):
    * Dry-Run Self-Test  : verified=3 / rejected=1 mock findings per paper —
                          parsing, quote gate, vote merge, and fallbacks verified
                          offline without spending API tokens.
    * Real API Run       : PENDING — execute with:
        python3 src/models/architectures/llm_parallel_chunking.py --provider saia
        python3 src/models/architectures/llm_parallel_chunking.py --provider hu
      (SAIA needs SAIA_API_KEY; HU needs the local bridge on :4000 + VPN.)
      (then update this block with the measured scores)
========================================================================================
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import difflib
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluator import (
    evaluate_predictions,
    load_jsonl_by_id,
    print_report,
    save_reproducible_run,
)
from src.models.architectures.llm_zeroshot_guidelines import (
    CONVENTIONS,
    PROVIDERS,
    OpenAICompatClient,
    extract_json_block,
    load_field_spec,
    normalize_enum,
)
from src.models.base_models.majority_baseline import fit_majority_model

MODEL_ID = "llm_parallel_chunking"
MODEL_TITLE = "Architecture 2: Parallel Chunking & Quote-Verified Voting"
CATEGORY = "architectures"
SEED = 42

# Chunk configuration: grouping pages into logical document sections (0-based,
# half-open [start, end) page ranges).
CHUNK_PAGE_GROUPS = [
    (0, 2),   # Chunk 1: Title, Abstract, Introduction (pages 1-2)
    (2, 5),   # Chunk 2: Study Area, Material & Methods (pages 3-5)
    (5, 9),   # Chunk 3: Results, Systematic Species Observations (pages 6-9)
    (9, 14),  # Chunk 4: Discussion, Management / Services (pages 10-14, if present)
]

CHUNK_SYSTEM_PROMPT = """You are an expert scientific information extraction system for biodiversity and ecological literature.
You are analyzing ONE isolated section of a scientific paper.
Your task is to identify mentions of ecological study attributes strictly based on THIS section.

CRITICAL REQUIREMENT — GROUNDING EVIDENCE:
For every finding you report, you MUST provide a verbatim quote (exact contiguous text
excerpt, max ~200 characters) from the provided section that explicitly proves it.
If you cannot quote the evidence word-for-word from the section, do NOT report the finding.
If a category is not mentioned in this section, omit it completely.
Output ONLY a valid JSON object, no prose.
"""

CHUNK_USER_PROMPT_TEMPLATE = """Paper Title: {title}
Document Section ({section_name}):
\"\"\"
{chunk_text}
\"\"\"
{field_guidelines}
ALLOWED VALUES REFERENCE (use these EXACT strings for the listed fields):
{enum_reference}
(ecosystem_type.term and ecosystem_service.term: quote the term as written in the section,
 as a short German noun phrase; it is matched against the schema vocabulary afterwards.)

Extract all findings present in THIS section. Each finding needs "field", "value", and "quote".
For taxa use {{"species_group": ..., "scientific_name": ..., "german_name": ..., "collective_term": true/false}}
(collective_term=true when only a broad group is named, not a specific species).
For ecosystem_type / ecosystem_service use {{"category": ..., "term": ...}}.

Output format:
{{
  "findings": [
    {{
      "field": "habitat" | "biodiversity_level" | "ecosystem_type" | "taxa" | "ecosystem_service",
      "value": "Exact enum string or the object as specified above",
      "quote": "verbatim text excerpt from this section proving this finding"
    }}
  ]
}}
"""

# --------------------------------------------------------------------------------------
# v2: thorough per-field requirement descriptions (guideline-paper insight: detailed
# field-level decision rules beat short convention lists). Injected into the chunk
# prompt via --field-guidelines; each rule is distilled from the official schema
# descriptions, the 908-paper gold conventions, and our Dev error analysis
# (biodiversity_level + ecosystem_service over-firing were the main FP sources).
# --------------------------------------------------------------------------------------

FIELD_GUIDELINES: str = """# FIELD REQUIREMENTS (decide per finding, precision over recall)

GENERAL RULE: A finding is only valid if THIS section shows that the study itself
samples, measures, analyzes, or manages the aspect. Context mentions, related-work
citations, or outlook statements do NOT count. When unsure, OMIT — empty predictions
are correct and frequent (ecosystem_service ~78%, ecosystem_type ~50%,
biodiversity_level ~42% of studies have none).

- habitat (max 2): The ecosystem(s) where the study's own sampling/intervention
  happens. agriculture/grassland/cropland/field experiments -> "Agrar- und Offenland";
  rivers/lakes/streams/wetlands/floodplains -> "Binnengewässer und Auen"; seas/coast/
  wadden sea -> "Küsten und Küstengewässer"; forest inventories/forest stands ->
  "Wald"; soil organisms/soil processes as the subject -> "Boden"; cities/urban green/
  parks -> "Urbane Räume". Do NOT infer a habitat from a species' general ecology —
  only from where THIS study works. Skip if the section gives no concrete location.

- biodiversity_level (max 2, STRICTEST FIELD — over-prediction is penalized): Only
  predict a level the study explicitly MEASURES or analyzes as a biodiversity aspect:
  counts/richness/species numbers -> "Artenvielfalt"; WHICH species occur, community
  composition/turnover/identity shifts -> "Zusammensetzung der Artengemeinschaft";
  traits/functional groups/ecosystem processes -> "Funktionelle Diversität";
  genetic markers/DNA/alleles -> "Genetische Diversität"; diversity OF habitat
  structures -> "Habitatdiversität"; physical structure measures (layers, dead wood)
  -> "Strukturelle Diversität"; evolutionary relatedness/phylogenies ->
  "Phylogenetische Diversität". Merely sampling or listing species is NOT a
  biodiversity level. If the section only reports species records, emit NOTHING here.

- ecosystem_type (max 2): Only when the study explicitly characterizes its sites by a
  concrete biotope/ecosystem type. category: marine/coastal biotopes -> "I Biotoptypen-
  gruppen der Meere und Küsten"; inland waters -> "II ... Binnengewässer"; terrestrial/
  semiterrestrial inland (the normal case) -> "III ..."; technical/urban structures ->
  "IV Technische Biotoptypengruppen"; alpine -> "V ..."; anything else -> "VI Weitere".
  term = the specific biotope named in the text (short German phrase), null if none.

- taxa (the study's own focus organisms, not background mentions): For each distinct
  taxon: species_group from the allowed values. A SPECIES-level focus ->
  {"collective_term": false, "scientific_name": Latin binomial, "german_name": German
  name or null}. A broad GROUP focus ('fishes', 'birds', 'vascular plants') ->
  {"collective_term": true, "scientific_name": Latin collective (Pisces, Aves,
  Plantae, Bacteria, Mollusca, Arthropoda, ...) or "-" if the text names none,
  "german_name": German collective (Fische, Vögel, ...)}. Tables of species in THIS
  section's results count; species in cited literature do not.

- ecosystem_service (max 2, RAREST FIELD — highest precision bar): ONLY an explicit
  benefit that nature provides to humans which the study itself investigates:
  pollination, water retention/purification, carbon storage, flood protection,
  erosion control, recreation, timber/food provision. Biodiversity monitoring per se,
  species conservation for its own sake, or habitat management without a human-benefit
  framing are NOT services. Categories: "Versorgungsleistungen" (provisioning),
  "Regulierungs- und Erhaltungsleistungen" (regulating/maintaining),
  "Kulturelle Leistungen" (cultural). If in doubt, OMIT.

"""


# --------------------------------------------------------------------------------------
# Stage 1: Document sectioning / chunking
# --------------------------------------------------------------------------------------

def extract_document_chunks(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
) -> list[dict[str, Any]]:
    """
    Extracts text page-by-page from the paper's PDF and segments it into multi-page
    logical chunks (see CHUNK_PAGE_GROUPS). Chunks with almost no text (<120 chars,
    e.g. blank or cover pages) are dropped. If no readable PDF is available, the
    paper title becomes a single fallback chunk so the paper still yields a prediction.
    """
    pdf_path = pdf_dir / f"{zot_id}.pdf"
    title = str(rec.get("bibtex_title") or "").strip()

    try:
        import pypdf

        if pdf_path.exists():
            reader = pypdf.PdfReader(str(pdf_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            chunks: list[dict[str, Any]] = []

            for idx, (p_start, p_end) in enumerate(CHUNK_PAGE_GROUPS, 1):
                p_stop = min(p_end, len(pages))
                if p_start >= p_stop:
                    continue  # paper has fewer pages than this chunk range
                text = "\n".join(pages[p_start:p_stop]).strip()
                if len(text) > 120:  # skip near-empty page groups
                    chunks.append(
                        {
                            "chunk_id": f"C{idx}",
                            "section_name": f"Pages {p_start + 1}-{p_stop}",
                            "text": text,
                        }
                    )
            if chunks:
                return chunks
    except Exception:
        pass  # unreadable PDF -> title-only fallback below

    return [{"chunk_id": "C1", "section_name": "Title Only", "text": title}]


# --------------------------------------------------------------------------------------
# Stage 3: Python anti-hallucination gate (quote verification)
# --------------------------------------------------------------------------------------

def verify_verbatim_quote(quote: str, chunk_text: str) -> bool:
    """
    Verifies that `quote` is genuinely grounded in `chunk_text`:
      1. normalized exact substring match (lowercase, collapsed whitespace),
      2. leading 30-character anchor match (models often append "..." mid-quote),
      3. fuzzy fallback: >= 75% of the quote found as ONE contiguous block in the
         chunk (tolerates OCR hyphenation artifacts inside the quote).
    Empty or extremely short quotes are rejected outright.
    """
    if not quote or len(quote.strip()) < 8:
        return False

    norm_quote = " ".join(quote.strip().lower().split())
    norm_chunk = " ".join(chunk_text.lower().split())

    # 1. Exact normalized match
    if norm_quote in norm_chunk:
        return True

    # 2. Leading anchor match (first 30 characters)
    if len(norm_quote) >= 30 and norm_quote[:30] in norm_chunk:
        return True

    # 3. Fuzzy contiguous alignment
    matcher = difflib.SequenceMatcher(None, norm_quote, norm_chunk)
    match = matcher.find_longest_match(0, len(norm_quote), 0, len(norm_chunk))
    if match.size >= max(15, int(len(norm_quote) * 0.75)):
        return True

    return False


# --------------------------------------------------------------------------------------
# Stage 2 output parsing (shared helpers from Architecture 1 are reused)
# --------------------------------------------------------------------------------------

def parse_chunk_findings(raw_output: str) -> list[dict[str, Any]]:
    """
    Extracts the candidate finding objects from one LLM chunk response. Tolerates
    markdown fences, surrounding prose, and a top-level bare list. Returns [] when
    nothing parseable is recovered (the paper then simply falls back to defaults).
    """
    data = extract_json_block(raw_output)
    if data is None:
        # Fallback: response may be a bare JSON array rather than an object.
        start, end = raw_output.find("["), raw_output.rfind("]")
        if start != -1 and end > start:
            try:
                candidate = json.loads(raw_output[start : end + 1])
                if isinstance(candidate, list):
                    return [f for f in candidate if isinstance(f, dict)]
            except json.JSONDecodeError:
                pass
        return []

    findings = data.get("findings")
    if isinstance(findings, list):
        return [f for f in findings if isinstance(f, dict)]
    return []


def normalize_enum_value(
    field_spec: dict[str, dict[str, Any]], key: str, value: Any
) -> str | None:
    """
    Maps a model-generated value onto the official German enum for `key`
    (e.g. "habitat" or "taxa.species_group"), reusing Architecture 1's
    exact -> canonical -> fuzzy matching. Returns None when nothing fits.
    """
    if "." in key:
        parent, sub = key.split(".", 1)
        allowed = field_spec[parent]["subfields"][sub]["values"]
    else:
        allowed = field_spec[key]["values"]
    return normalize_enum(str(value), allowed)


def build_enum_reference(field_spec: dict[str, dict[str, Any]]) -> str:
    """
    Renders the compact per-chunk enum reference from the OFFICIAL schemas
    (never hardcode enum lists — the schemas are the single source of truth).
    The two huge term vocabularies (46 / 67 values) are deliberately omitted
    to keep chunk prompts small; terms are free-text and resolved in Python.
    """
    keys = [
        "habitat",
        "biodiversity_level",
        "taxa.species_group",
        "ecosystem_type.category",
        "ecosystem_service.category",
    ]
    lines = []
    for key in keys:
        if "." in key:
            parent, sub = key.split(".", 1)
            values = field_spec[parent]["subfields"][sub]["values"]
        else:
            values = field_spec[key]["values"]
        rendered = ", ".join(f'"{v}"' for v in values)
        lines.append(f"- {key}: {rendered}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Stage 4: Vote merging across chunks
# --------------------------------------------------------------------------------------

def merge_chunk_votes(
    verified_findings_by_chunk: list[list[dict[str, Any]]],
    zot_id: str,
    majority_learned: dict[str, dict[str, Any]],
    field_spec: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Merges verified findings across all chunks for one paper by vote aggregation:
      - Votes for a label = number of distinct chunks providing verified evidence
        (each chunk counts at most once per label; an empty chunk is not a vote
        against anything, because evidence is naturally localized).
      - habitat / biodiversity_level: top-2 labels by votes.
      - ecosystem_type / ecosystem_service: category-term pairs, deduplicated,
        capped at 2; services only with verified support (sparse field).
      - taxa: specific species or collective groups; the gold convention forces
        `scientific_name = "-"` and `german_name = species_group` for collective
        terms. Dense fallback to the training majority when nothing is verified.
    """
    habitat_votes: dict[str, int] = defaultdict(int)
    bio_level_votes: dict[str, int] = defaultdict(int)
    eco_type_candidates: list[dict[str, Any]] = []
    taxa_candidates: list[dict[str, Any]] = []
    eco_service_candidates: list[dict[str, Any]] = []

    for chunk_findings in verified_findings_by_chunk:
        seen_in_chunk: set[str] = set()  # one vote per label per chunk
        for f in chunk_findings:
            field = str(f.get("field") or "").lower().strip()
            val = f.get("value")

            if "habitat" in field and isinstance(val, str):
                norm = normalize_enum_value(field_spec, "habitat", val)
                if norm and norm not in seen_in_chunk:
                    habitat_votes[norm] += 1
                    seen_in_chunk.add(norm)

            elif "biodiversity" in field and isinstance(val, str):
                norm = normalize_enum_value(field_spec, "biodiversity_level", val)
                if norm and norm not in seen_in_chunk:
                    bio_level_votes[norm] += 1
                    seen_in_chunk.add(norm)

            elif "ecosystem_type" in field and isinstance(val, dict):
                eco_type_candidates.append(val)

            elif "taxa" in field:
                if isinstance(val, dict):
                    taxa_candidates.append(val)
                elif isinstance(val, str):
                    # Model reported a bare group name -> collective term by design
                    grp = normalize_enum_value(field_spec, "taxa.species_group", val)
                    if grp:
                        taxa_candidates.append(
                            {"species_group": grp, "collective_term": True}
                        )

            elif "ecosystem_service" in field and isinstance(val, dict):
                eco_service_candidates.append(val)

    # 1. Habitat: top-2 by verified votes; training majority as dense fallback
    sorted_habs = sorted(habitat_votes.items(), key=lambda kv: (-kv[1], kv[0]))
    habitats = [hab for hab, _ in sorted_habs[:2]]
    if not habitats:
        habitats = [majority_learned["habitat"]["majority_value"]]

    # 2. Biodiversity level: top-2 by verified votes (no fallback: sparser field)
    sorted_bio = sorted(bio_level_votes.items(), key=lambda kv: (-kv[1], kv[0]))
    bio_levels = [lvl for lvl, _ in sorted_bio[:2]]

    # 3. Ecosystem type: verified category-term pairs, deduplicated, capped at 2
    eco_types: list[dict[str, Any]] = []
    seen_eco: set[tuple] = set()
    for cand in eco_type_candidates:
        cat = normalize_enum_value(
            field_spec, "ecosystem_type.category", cand.get("category") or ""
        )
        if cat is None:
            continue
        term = cand.get("term")
        term = (
            normalize_enum_value(field_spec, "ecosystem_type.term", term)
            if term
            else None
        )
        key = (cat, term)
        if key not in seen_eco:
            seen_eco.add(key)
            eco_types.append({"category": cat, "term": term})
        if len(eco_types) >= 2:
            break

    # 4. Taxa: species + collective groups, '-' convention enforced in Python
    taxa_entries: list[dict[str, Any]] = []
    seen_taxa: set[tuple] = set()
    for t in taxa_candidates:
        grp = normalize_enum_value(
            field_spec, "taxa.species_group", t.get("species_group") or ""
        )
        if grp is None:
            continue
        sci = str(t.get("scientific_name") or "").strip()
        ger = str(t.get("german_name") or "").strip()
        if bool(t.get("collective_term", False)) or not sci:
            # Broad group only: gold convention is scientific_name "-",
            # German name defaults to the (German) species_group enum value.
            sci = "-"
            ger = ger or grp
        key = (grp, sci, ger)
        if key not in seen_taxa:
            seen_taxa.add(key)
            taxa_entries.append(
                {
                    "species_group": grp,
                    "scientific_name": sci or "-",
                    "german_name": ger or grp,
                }
            )
        if len(taxa_entries) >= 5:
            break

    if not taxa_entries:  # dense fallback: training-majority collective taxon
        taxa_entries.append(
            {
                "species_group": majority_learned["taxa.species_group"]["majority_value"],
                "scientific_name": "-",
                "german_name": majority_learned["taxa.german_name"]["majority_value"],
            }
        )

    # 5. Ecosystem service: sparse (~78% empty in gold) — verified support ONLY
    eco_services: list[dict[str, Any]] = []
    seen_srv: set[tuple] = set()
    for s in eco_service_candidates:
        cat = normalize_enum_value(
            field_spec, "ecosystem_service.category", s.get("category") or ""
        )
        if cat is None:
            continue
        term = s.get("term")
        term = (
            normalize_enum_value(field_spec, "ecosystem_service.term", term)
            if term
            else None
        )
        key = (cat, term)
        if key not in seen_srv:
            seen_srv.add(key)
            eco_services.append({"category": cat, "term": term})
        if len(eco_services) >= 2:
            break

    return {
        "zotitem_ptr_id": zot_id,
        "habitat": habitats,
        "biodiversity_level": bio_levels,
        "ecosystem_type": eco_types,
        "taxa": taxa_entries,
        "ecosystem_service": eco_services,
    }


# --------------------------------------------------------------------------------------
# Deterministic offline mock for --dry-run (no API key / tokens needed)
# --------------------------------------------------------------------------------------

# A sentence that cannot occur in any real paper chunk; the quote gate MUST reject it.
_HALLUCINATED_QUOTE = (
    "XXHALLUCINATED-EVIDENCE-XX this fabricated sentence appears nowhere in the section."
)


class ChunkMockClient:
    """
    Offline stand-in for the LLM that exercises every stage of the chunk pipeline:
    it returns (per paper) ONE chunk response containing
      * 3 findings whose quotes are exact substrings of the chunk text
        (majority-baseline values, one with corrupted casing, one collective taxon
        without scientific_name) -> these MUST survive the verification gate, and
      * 1 finding with a fabricated quote -> this MUST be discarded as hallucination.
    All later chunks return empty findings. Responses are wrapped in markdown fences
    plus a trailing-comma JSON error to exercise the parser's repair path.
    If the pipeline is correct, every dry-run paper predicts: majority habitat
    (normalized back to exact casing), majority biodiversity_level, one collective
    taxon with scientific_name="-", and empty ecosystem_type/ecosystem_service.
    """

    def __init__(self, train_records: dict[str, dict[str, Any]]):
        self._learned = fit_majority_model(train_records)
        self.mock_context: dict[str, Any] = {}
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @staticmethod
    def _slice(text: str, start: int, length: int) -> str:
        """Contiguous verbatim excerpt of the chunk text (grounded by construction)."""
        if len(text) <= start:
            return text[:length]  # very short (title-only) chunk: reuse the start
        return text[start : start + length]

    def chat(self, system: str, user: str) -> str:
        self.n_calls += 1
        rec = self.mock_context["rec"]
        chunk = self.mock_context["chunk"]

        if chunk["chunk_id"] != "C1":
            findings: list[dict[str, Any]] = []
        else:
            text = chunk["text"]
            group = self._learned["taxa.species_group"]["majority_value"]
            findings = [
                {
                    "field": "habitat",
                    # corrupted casing/whitespace must be repaired by enum normalization
                    "value": str(self._learned["habitat"]["majority_value"]).upper(),
                    "quote": self._slice(text, 0, 90),
                },
                {
                    "field": "biodiversity_level",
                    "value": self._learned["biodiversity_level"]["majority_value"],
                    "quote": self._slice(text, 100, 90),
                },
                {
                    # collective term WITHOUT scientific_name: merge must add "-"
                    "field": "taxa",
                    "value": {"species_group": group, "collective_term": True},
                    "quote": self._slice(text, 200, 90),
                },
                {
                    "field": "habitat",
                    "value": "Wald",
                    "quote": _HALLUCINATED_QUOTE,  # must be rejected by the gate
                },
            ]

        obj = json.dumps({"findings": findings}, ensure_ascii=False)
        # Inject a trailing comma (classic LLM JSON slip) to exercise the repair path.
        obj = obj[:-1] + ",}"
        return f"Sure, here is the JSON:\n```json\n{obj}\n```"


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Architecture Model 2: Parallel Chunking with Verbatim Quote Verification."
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="saia",
        help="API provider (default: saia)",
    )
    parser.add_argument("--model", default=None, help="Model name (default: provider default)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rpm", type=int, default=30, help="Rate limit requests/minute")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers evaluated")
    parser.add_argument(
        "--field-guidelines",
        action="store_true",
        help="v2: inject thorough per-field requirement descriptions into the chunk prompt",
    )
    parser.add_argument("--dry-run", action="store_true", help="Offline deterministic mock test")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip save_reproducible_run() (smoke tests keep artifacts/leaderboard clean)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # --- data loading ---
    gold_path = REPO_ROOT / "data" / "gold" / "dev.jsonl"
    dev_ids_path = REPO_ROOT / "data" / "gold" / "dev_100_ids.txt"
    train_path = REPO_ROOT / "data" / "gold" / "train.jsonl"
    pdf_dir = REPO_ROOT / "data" / "raw_pdfs" / "dev"

    gold = load_jsonl_by_id(gold_path)
    train_records = load_jsonl_by_id(train_path)
    majority_learned = fit_majority_model(train_records)
    field_spec = load_field_spec()
    enum_reference = build_enum_reference(field_spec)

    dev_ids = [line.strip() for line in dev_ids_path.read_text().splitlines() if line.strip()]
    if args.limit:
        dev_ids = dev_ids[: args.limit]

    # --- client setup ---
    if args.dry_run:
        client: Any = ChunkMockClient(train_records)
        actual_model = "dry-run-mock"
    else:
        actual_model = args.model or PROVIDERS[args.provider]["default_model"]
        client = OpenAICompatClient(args.provider, actual_model, args.temperature, args.rpm)

    # --- checkpointing: resume interrupted runs without re-spending API tokens ---
    # The provider+model suffix keeps parallel SAIA / HU runs from masking each other
    # (same convention as Architecture 1).
    if args.dry_run:
        run_model_id = MODEL_ID
        mode_tag = "dryrun"
    else:
        model_slug = actual_model.replace("/", "_").replace(":", "_")
        # v2 runs (field guidelines) get their own suffix so checkpoints, artifacts,
        # and leaderboard entries never mix with the v1 baseline runs.
        variant = "_fg" if args.field_guidelines else ""
        run_model_id = f"{MODEL_ID}_{args.provider}_{model_slug}{variant}"
        mode_tag = f"api_{args.provider}_{model_slug}{variant}"

    ckpt_dir = REPO_ROOT / "outputs" / CATEGORY / run_model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"{run_model_id}_checkpoint_{mode_tag}.jsonl"
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

    total_chunks_processed = 0
    total_quotes_verified = 0
    total_quotes_rejected = 0

    for i, zot_id in enumerate(dev_ids, 1):
        if zot_id in predictions:  # already done in a previous (interrupted) run
            continue

        rec = gold[zot_id]
        title = str(rec.get("bibtex_title") or "")
        chunks = extract_document_chunks(zot_id, rec, pdf_dir)

        verified_findings_by_chunk: list[list[dict[str, Any]]] = []

        # Stage 1+2: independent LLM extraction per chunk (parallel by design;
        # executed sequentially here to respect the shared provider rate limit)
        for ch in chunks:
            total_chunks_processed += 1
            user_prompt = CHUNK_USER_PROMPT_TEMPLATE.format(
                title=title,
                section_name=ch["section_name"],
                chunk_text=ch["text"],
                enum_reference=enum_reference,
                field_guidelines=FIELD_GUIDELINES if args.field_guidelines else "",
            )

            if isinstance(client, ChunkMockClient):
                client.mock_context = {"rec": rec, "chunk": ch}

            raw_out = client.chat(CHUNK_SYSTEM_PROMPT, user_prompt)
            raw_findings = parse_chunk_findings(raw_out)

            # Stage 3: anti-hallucination gate — quotes must be grounded in the chunk
            verified_chunk_findings = []
            for f in raw_findings:
                quote = str(f.get("quote") or "")
                if verify_verbatim_quote(quote, ch["text"]):
                    verified_chunk_findings.append(f)
                    total_quotes_verified += 1
                else:
                    total_quotes_rejected += 1

            verified_findings_by_chunk.append(verified_chunk_findings)

        # Stage 4: merge verified evidence across chunks into one schema-conform record
        paper_pred = merge_chunk_votes(
            verified_findings_by_chunk, zot_id, majority_learned, field_spec
        )
        predictions[zot_id] = paper_pred

        # append-write checkpoint immediately so a crash loses at most one paper
        with ckpt_file.open("a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(paper_pred, ensure_ascii=False) + "\n")

        n_verified = sum(len(v) for v in verified_findings_by_chunk)
        print(
            f"[{i:>3}/{len(dev_ids)}] {zot_id} | chunks={len(chunks)} | "
            f"hab={len(paper_pred['habitat'])} taxa={len(paper_pred['taxa'])} | "
            f"verified={n_verified}",
            flush=True,
        )

    # --- evaluation + reproducible artifacts ---
    metrics = evaluate_predictions(gold, predictions, filter_ids=set(dev_ids))
    run_title = f"{MODEL_TITLE} ({actual_model})"
    print_report(f"{run_title} {'[DRY-RUN]' if args.dry_run else ''}", metrics)

    runtime = time.time() - start_time
    token_stats = {
        "n_calls": getattr(client, "n_calls", 0),
        "prompt_tokens": getattr(client, "prompt_tokens", 0),
        "completion_tokens": getattr(client, "completion_tokens", 0),
        "total_chunks_processed": total_chunks_processed,
        "total_quotes_verified": total_quotes_verified,
        "total_quotes_rejected": total_quotes_rejected,
    }

    if args.no_save:
        print("\n--no-save set: skipping run artifacts + leaderboard update (smoke test).")
    else:
        save_reproducible_run(
            model_id=run_model_id,
            model_title=run_title + (" [DRY-RUN MOCK]" if args.dry_run else ""),
            category=CATEGORY,
            predictions=predictions,
            metrics=metrics,
            runtime_seconds=runtime,
            seed=SEED,
            config={
                "mode": "dry_run_mock" if args.dry_run else "api",
                "provider": args.provider,
                "model": actual_model,
                "temperature": args.temperature,
                "rpm_limit": args.rpm,
                "field_guidelines": bool(args.field_guidelines),
                "n_papers": len(dev_ids),
                "chunk_page_groups": [list(g) for g in CHUNK_PAGE_GROUPS],
                "quote_verification": "exact | anchor30 | fuzzy>=0.75-contiguous",
                "prompt_conventions": CONVENTIONS,
                "token_stats": token_stats,
            },
        )

    print(
        f"\nDone in {runtime:.1f}s | calls={token_stats['n_calls']} | "
        f"chunks={total_chunks_processed} | verified_quotes={total_quotes_verified} | "
        f"rejected_quotes={total_quotes_rejected}"
    )


if __name__ == "__main__":
    main()
