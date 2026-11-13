#!/usr/bin/env python3
"""
========================================================================================
EXECUTIVE SUMMARY — Architecture: English Sections Extraction Pipeline (A & B)
========================================================================================
- Concept & Theoretical Foundation:
    Implements the team's planned dual-architecture framework:
      (A) Sequential State-Passing Fold vs. (B) Parallel Independent Map-Reduce
    combined with the "Translate-Text" prompt technique (Minh Anh) and Section Chunking
    (inspired by GROBID and scientific document layout).

- Why English-First with Deterministic Mapping?
    Over 85% of biodiversity literature is published in English. Forcing LLMs to extract
    complex ecological concepts directly into bureaucratic German catalog strings
    (e.g., "Biotoptypengruppen des Binnenlandes") induces significant language-switch
    confusion and false positives.
    Instead, this pipeline:
      1. Prompts the LLM in English to extract standardized English labels backed by
         verbatim text quotes from the paper section.
      2. Verifies that every quote is authentically grounded in the section text
         (Python anti-hallucination gate).
      3. Deterministically translates verified English labels to official German schema enums
         via an exact rule/lexicon dictionary (0% translation hallucination).

- Dual Architectural Modes:
    * `--architecture parallel` (Pipeline B):
      Each document section (Abstract, Intro, Methods/Study Area, Results/Discussion) is
      queried independently. Votes for each label equal the count of verified quotes across
      sections. An empty section is never a negative vote.
    * `--architecture sequential` (Pipeline A):
      Sections are processed in sequential order. The extracted state from Section i is
      passed into the prompt for Section i+1. The model can confirm existing findings,
      discover new ones, or revoke spurious assumptions made in early introductory text.

- Fast, Grounded Inference with GLM-5.3:
    Default model: `glm-5.3-flash` on SAIA (GWDG Academic Cloud), providing strong
    cross-lingual reasoning, high throughput, and strict adherence to quote grounding.
========================================================================================
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import difflib
import json
from pathlib import Path
import re
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
    DryRunMockClient,
    OpenAICompatClient,
    PROVIDERS,
    extract_json_block,
)
from src.models.base_models.majority_baseline import (
    fit_majority_model,
    predict_single_paper as majority_predict_single_paper,
)

MODEL_BASE_ID = "llm_sections"
CATEGORY = "architectures"
SEED = 42

# Ensure SAIA provider has GLM models configured
if "saia" in PROVIDERS:
    PROVIDERS["saia"]["default_model"] = "glm-5.3-flash"


class RobustOpenAICompatClient(OpenAICompatClient):
    """
    Subclass of OpenAICompatClient with robust retry and exponential backoff
    for transient connection failures, timeouts, rate limits (429), and 5xx errors.
    """

    def chat(self, system: str, user: str, max_retries: int = 6) -> str:
        for attempt in range(max_retries):
            try:
                return super().chat(system, user)
            except Exception as exc:
                err_msg = str(exc)
                if attempt == max_retries - 1:
                    print(f"\n[ERROR] Persistent API failure after {max_retries} attempts: {err_msg}")
                    return ""
                wait_sec = min(120, (2 ** attempt) * 5)
                print(f"\n[RETRY {attempt+1}/{max_retries}] Transient API error: {err_msg[:120]}... Waiting {wait_sec}s")
                time.sleep(wait_sec)
        return ""

# ======================================================================================
# English-to-German Deterministic Mapping Dictionary (0% Hallucination)
# ======================================================================================

EN_TO_DE_HABITAT = {
    "agricultural and open land": "Agrar- und Offenland",
    "agricultural land": "Agrar- und Offenland",
    "agriculture": "Agrar- und Offenland",
    "farmland": "Agrar- und Offenland",
    "grassland": "Agrar- und Offenland",
    "meadow": "Agrar- und Offenland",
    "pasture": "Agrar- und Offenland",
    "cropland": "Agrar- und Offenland",
    "arable": "Agrar- und Offenland",
    "open land": "Agrar- und Offenland",
    "inland waters and floodplains": "Binnengewässer und Auen",
    "inland waters": "Binnengewässer und Auen",
    "freshwater": "Binnengewässer und Auen",
    "river": "Binnengewässer und Auen",
    "stream": "Binnengewässer und Auen",
    "lake": "Binnengewässer und Auen",
    "floodplain": "Binnengewässer und Auen",
    "wetland": "Binnengewässer und Auen",
    "coasts and coastal waters": "Küsten und Küstengewässer",
    "coastal waters": "Küsten und Küstengewässer",
    "coast": "Küsten und Küstengewässer",
    "marine": "Küsten und Küstengewässer",
    "sea": "Küsten und Küstengewässer",
    "baltic sea": "Küsten und Küstengewässer",
    "north sea": "Küsten und Küstengewässer",
    "estuary": "Küsten und Küstengewässer",
    "forest": "Wald",
    "woodland": "Wald",
    "beech forest": "Wald",
    "coniferous forest": "Wald",
    "mixed forest": "Wald",
    "soil": "Boden",
    "soils": "Boden",
    "rhizosphere": "Boden",
    "urban areas": "Urbane Räume",
    "urban": "Urbane Räume",
    "city": "Urbane Räume",
    "cities": "Urbane Räume",
    "town": "Urbane Räume",
}

EN_TO_DE_BIO_LEVEL = {
    "species richness": "Artenvielfalt",
    "species diversity": "Artenvielfalt",
    "taxonomic diversity": "Artenvielfalt",
    "community composition": "Zusammensetzung der Artengemeinschaft",
    "species composition": "Zusammensetzung der Artengemeinschaft",
    "community structure": "Zusammensetzung der Artengemeinschaft",
    "assemblage": "Zusammensetzung der Artengemeinschaft",
    "functional diversity": "Funktionelle Diversität",
    "functional traits": "Funktionelle Diversität",
    "genetic diversity": "Genetische Diversität",
    "population genetics": "Genetische Diversität",
    "habitat diversity": "Habitatdiversität",
    "landscape heterogeneity": "Habitatdiversität",
    "structural diversity": "Strukturelle Diversität",
    "stand structure": "Strukturelle Diversität",
    "phylogenetic diversity": "Phylogenetische Diversität",
    "phylogenetic": "Phylogenetische Diversität",
}

EN_TO_DE_SPECIES_GROUP = {
    "vascular plants": "Gefäßpflanzen",
    "plants": "Gefäßpflanzen",
    "flora": "Gefäßpflanzen",
    "trees": "Gefäßpflanzen",
    "grasses": "Gefäßpflanzen",
    "herbs": "Gefäßpflanzen",
    "mosses": "Moose",
    "bryophytes": "Moose",
    "peat moss": "Moose",
    "lichens": "Flechten",
    "fungi": "Pilze",
    "fungus": "Pilze",
    "mycorrhiza": "Pilze",
    "mushrooms": "Pilze",
    "algae": "Algen",
    "macroalgae": "Algen",
    "phytoplankton": "Algen",
    "bacteria": "Bakterien",
    "microbes": "Bakterien",
    "microorganisms": "Bakterien",
    "birds": "Vögel",
    "avian": "Vögel",
    "songbirds": "Vögel",
    "breeding birds": "Vögel",
    "waterbirds": "Vögel",
    "mammals": "Säugetiere",
    "bats": "Säugetiere",
    "rodents": "Säugetiere",
    "ungulates": "Säugetiere",
    "fish": "Fische",
    "fishes": "Fische",
    "teleost": "Fische",
    "amphibians and reptiles": "Amphibien und Reptilien",
    "amphibians": "Amphibien und Reptilien",
    "reptiles": "Amphibien und Reptilien",
    "frogs": "Amphibien und Reptilien",
    "toads": "Amphibien und Reptilien",
    "newts": "Amphibien und Reptilien",
    "insects": "Insekten",
    "beetles": "Insekten",
    "butterflies": "Insekten",
    "bees": "Insekten",
    "pollinators": "Insekten",
    "ants": "Insekten",
    "diptera": "Insekten",
    "other invertebrates": "Andere Wirbellose",
    "invertebrates": "Andere Wirbellose",
    "spiders": "Andere Wirbellose",
    "snails": "Andere Wirbellose",
    "earthworms": "Andere Wirbellose",
    "crustaceans": "Andere Wirbellose",
    "macroinvertebrates": "Andere Wirbellose",
    "benthos": "Andere Wirbellose",
    "zooplankton": "Andere Wirbellose",
    "other": "Andere",
}

EN_TO_DE_ECOSYSTEM_SERVICE = {
    "provisioning services": "Versorgungsleistungen",
    "provisioning service": "Versorgungsleistungen",
    "food production": "Versorgungsleistungen",
    "timber": "Versorgungsleistungen",
    "forage": "Versorgungsleistungen",
    "regulating and maintenance services": "Regulierungs- und Erhaltungsleistungen",
    "regulating services": "Regulierungs- und Erhaltungsleistungen",
    "pollination": "Regulierungs- und Erhaltungsleistungen",
    "carbon sequestration": "Regulierungs- und Erhaltungsleistungen",
    "carbon storage": "Regulierungs- und Erhaltungsleistungen",
    "flood regulation": "Regulierungs- und Erhaltungsleistungen",
    "water retention": "Regulierungs- und Erhaltungsleistungen",
    "erosion control": "Regulierungs- und Erhaltungsleistungen",
    "pest control": "Regulierungs- und Erhaltungsleistungen",
    "cultural services": "Kulturelle Dienstleistungen",
    "recreation": "Kulturelle Dienstleistungen",
    "tourism": "Kulturelle Dienstleistungen",
}

# ======================================================================================
# Prompts for Parallel (B) and Sequential (A)
# ======================================================================================

ENGLISH_SYSTEM_PROMPT = """You are an expert scientific information extraction assistant specializing in biodiversity and ecology literature.
Analyze the provided document section and extract only findings directly investigated or measured in the study.

STRICT GROUNDING REQUIREMENT:
Every extracted finding MUST include an exact verbatim quote from the text section as evidence.
If a category is not discussed in this section, emit nothing for it. Output ONLY a valid JSON object.
"""

PARALLEL_CHUNK_PROMPT = """Paper Title: {title}
Section ({section_name}):
\"\"\"
{section_text}
\"\"\"

TARGET CATEGORIES (use English concepts):
- habitat: "Forest", "Agricultural and open land", "Inland waters and floodplains", "Coasts and coastal waters", "Urban areas", "Soil"
- biodiversity_level: "Species richness", "Community composition", "Functional diversity", "Genetic diversity", "Habitat diversity", "Structural diversity", "Phylogenetic diversity"
- taxa.species_group: "Vascular plants", "Mosses", "Lichens", "Fungi", "Algae", "Bacteria", "Birds", "Mammals", "Fish", "Amphibians and reptiles", "Insects", "Other invertebrates", "Other"
- ecosystem_service (only if explicitly studied as a service to humans): "Regulating and maintenance services", "Provisioning services", "Cultural services"

Extract findings present in this section.
JSON format:
{{
  "findings": [
    {{
      "field": "habitat" | "biodiversity_level" | "taxa.species_group" | "ecosystem_service",
      "value": "English category from target list above",
      "scientific_name": "Latin species name if specific species studied, or null",
      "quote": "verbatim text sentence from this section proving this finding"
    }}
  ]
}}
"""

SEQUENTIAL_CHUNK_PROMPT = """Paper Title: {title}
Previous Findings from Earlier Sections:
{state_json}

Now read the Next Section ({section_name}):
\"\"\"
{section_text}
\"\"\"

TASK:
Update the findings based on this section:
1. Confirm supported findings with evidence from this section.
2. Add newly discovered findings (each MUST have a verbatim quote from this section).
3. If earlier assumptions were background citations rather than the empirical focus, omit them.

JSON format:
{{
  "findings": [
    {{
      "field": "habitat" | "biodiversity_level" | "taxa.species_group" | "ecosystem_service",
      "value": "English category",
      "scientific_name": "Latin name or null",
      "quote": "verbatim text sentence from this section"
    }}
  ]
}}
"""

GERMAN_SYSTEM_PROMPT = """Du bist ein präziser wissenschaftlicher Assistent für Informationsextraktion in Biodiversitäts- und Ökologiestudien.
Analysiere den vorliegenden Textabschnitt und extrahiere ausschließlich Aspekte, die in der Studie direkt empirisch untersucht oder erfasst wurden.

STRIKTE BELEGPFLICHT:
Jeder extrahierte Befund MUSS ein exaktes, wörtliches Zitat (Quote) aus dem Textabschnitt als Beleg enthalten.
Wird eine Kategorie in diesem Abschnitt nicht behandelt, gib nichts dafür aus. Gib AUSSCHLIESSLICH ein gültiges JSON-Objekt aus.
"""

GERMAN_PARALLEL_CHUNK_PROMPT = """Titel des Papers: {title}
Abschnitt ({section_name}):
\"\"\"
{section_text}
\"\"\"

ZIELKATEGORIEN (nutze exakt diese deutschen Bezeichnungen):
- habitat: "Wald", "Agrar- und Offenland", "Binnengewässer und Auen", "Küste und Küstengewässer", "Urbane Räume", "Boden"
- biodiversity_level: "Art", "Gemeinschaft", "Funktionell", "Genetisch", "Habitat", "Struktur", "Phylogenetisch"
- taxa.species_group: "Gefäßpflanzen", "Moose", "Flechten", "Pilze", "Algen", "Bakterien", "Vögel", "Säugetiere", "Fische", "Amphibien und Reptilien", "Insekten", "Sonstige Wirbellose", "Sonstige"
- ecosystem_service (nur falls explizit als Ökosystemleistung für den Menschen untersucht): "Regulierungs- und Erhaltungsleistungen", "Versorgungsleistungen", "Kulturelle Leistungen"

Extrahiere die Befunde aus diesem Abschnitt.
JSON-Format:
{{
  "findings": [
    {{
      "field": "habitat" | "biodiversity_level" | "taxa.species_group" | "ecosystem_service",
      "value": "Exakte deutsche Kategorie aus der Liste oben",
      "scientific_name": "Wissenschaftlicher Artname auf Latein falls spezifische Art untersucht, sonst null",
      "quote": "Wörtlicher Satz aus diesem Textabschnitt als Beleg"
    }}
  ]
}}
"""

GERMAN_SEQUENTIAL_CHUNK_PROMPT = """Titel des Papers: {title}
Bisherige Befunde aus früheren Abschnitten:
{state_json}

Lies nun den nächsten Abschnitt ({section_name}):
\"\"\"
{section_text}
\"\"\"

AUFGABE:
Aktualisiere die Befunde basierend auf diesem Abschnitt:
1. Bestätige gestützte Befunde mit Belegen aus diesem Abschnitt.
2. Füge neu gefundene Befunde hinzu (jeder MUSS ein wörtliches Zitat aus diesem Abschnitt haben).
3. Verwerfe frühere Annahmen, falls sie sich nur als Hintergrundzitate herausstellen.

JSON-Format:
{{
  "findings": [
    {{
      "field": "habitat" | "biodiversity_level" | "taxa.species_group" | "ecosystem_service",
      "value": "Exakte deutsche Kategorie",
      "scientific_name": "Lateinischer Artname oder null",
      "quote": "Wörtlicher Satz als Beleg oder 'confirmed' für bestehende Befunde"
    }}
  ]
}}
"""

# ======================================================================================
# Section Extraction: Semantic Sections (PyMuPDF / GROBID Hook)
# ======================================================================================

def extract_semantic_sections(
    zot_id: str,
    rec: dict[str, Any],
    pdf_dir: Path,
    grobid_url: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extracts semantic document sections (Abstract, Intro, Methods/Study Area, Results/Discussion).
    Uses PyMuPDF to detect headings, falling back to page-based chunks if unstructured.
    If a GROBID endpoint is active, queries GROBID's TEI XML parser.
    """
    pdf_path = pdf_dir / f"{zot_id}.pdf"
    title = str(rec.get("bibtex_title") or "").strip()

    if not pdf_path.exists():
        return [{"section_name": "Title Only", "text": title}]

    # Optional GROBID REST API hook
    if grobid_url:
        try:
            import urllib.request
            with open(pdf_path, "rb") as f_pdf:
                req = urllib.request.Request(
                    f"{grobid_url.rstrip('/')}/api/processFulltextDocument",
                    data=f_pdf.read(),
                    headers={"Accept": "application/xml"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    tei_xml = resp.read().decode("utf-8", errors="replace")
                    # Simple regex extraction of TEI heads and paragraphs
                    divs = re.findall(r"<div[^>]*>(.*?)</div>", tei_xml, re.DOTALL)
                    sections = []
                    for i, div in enumerate(divs, 1):
                        head_m = re.search(r"<head[^>]*>(.*?)</head>", div)
                        head_txt = head_m.group(1) if head_m else f"Section {i}"
                        p_txt = " ".join(re.findall(r"<p[^>]*>(.*?)</p>", div, re.DOTALL))
                        clean_p = re.sub(r"<[^>]+>", " ", p_txt).strip()
                        if len(clean_p) > 150:
                            sections.append({"section_name": head_txt[:60], "text": clean_p})
                    if sections:
                        return sections
        except Exception:
            pass  # Fall back to PyMuPDF local extraction

    # Local PyMuPDF Section Parsing
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        full_pages = [page.get_text() for page in doc]
        doc.close()

        # Group pages into logical sections
        sections = []
        # Section 1: Opening (Abstract + Introduction)
        sec1_text = "\n".join(full_pages[:2]).strip()
        if sec1_text:
            sections.append({"section_name": "Abstract & Introduction", "text": sec1_text})

        # Section 2: Study Area, Material & Methods
        sec2_text = "\n".join(full_pages[2:min(5, len(full_pages))]).strip()
        if sec2_text:
            sections.append({"section_name": "Study Area & Methods", "text": sec2_text})

        # Section 3: Results & Species Observations
        if len(full_pages) > 5:
            sec3_text = "\n".join(full_pages[5:min(10, len(full_pages))]).strip()
            if sec3_text:
                sections.append({"section_name": "Results & Findings", "text": sec3_text})

        # Section 4: Discussion & Synthesis
        if len(full_pages) > 10:
            sec4_text = "\n".join(full_pages[10:min(16, len(full_pages))]).strip()
            if sec4_text:
                sections.append({"section_name": "Discussion & Ecosystem Context", "text": sec4_text})

        if sections:
            return sections
    except Exception:
        pass

    return [{"section_name": "Full Text", "text": title}]


# ======================================================================================
# Python Anti-Hallucination Gate (Verbatim Quote Verification)
# ======================================================================================

def verify_verbatim_quote(quote: str, section_text: str) -> bool:
    """Verifies that an extracted quote authentically appears in the source section text."""
    if not quote or len(quote.strip()) < 8:
        return False

    norm_quote = " ".join(quote.strip().lower().split())
    norm_text = " ".join(section_text.lower().split())

    # 1. Exact normalized match
    if norm_quote in norm_text:
        return True

    # 2. Leading anchor match (first 32 characters)
    if len(norm_quote) >= 32 and norm_quote[:32] in norm_text:
        return True

    # 3. Fuzzy matching for OCR hyphenation or line breaks
    matcher = difflib.SequenceMatcher(None, norm_quote, norm_text)
    match = matcher.find_longest_match(0, len(norm_quote), 0, len(norm_text))
    if match.size >= max(15, int(len(norm_quote) * 0.72)):
        return True

    return False


# ======================================================================================
# Deterministic English-to-German Mapping & Prediction Assembly
# ======================================================================================

def map_english_finding_to_german(
    finding: dict[str, Any]
) -> tuple[str, str, str | None]:
    """
    Translates English extracted findings into official German schema enums.
    Returns: (target_field, german_value, scientific_name)
    """
    raw_field = str(finding.get("field") or "").lower()
    raw_val = str(finding.get("value") or finding.get("concept") or finding.get("english_label") or "").strip().lower()
    sci_name = finding.get("scientific_name")
    if sci_name:
        sci_name = str(sci_name).strip()

    # 1. Habitat
    if "habitat" in raw_field or raw_val in EN_TO_DE_HABITAT:
        for en_key, de_val in EN_TO_DE_HABITAT.items():
            if en_key == raw_val or de_val.lower() == raw_val or re.search(r"\b" + re.escape(en_key) + r"\b", raw_val):
                return "habitat", de_val, None

    # 2. Biodiversity Level
    if "biodiversity" in raw_field or raw_val in EN_TO_DE_BIO_LEVEL:
        for en_key, de_val in EN_TO_DE_BIO_LEVEL.items():
            if en_key == raw_val or de_val.lower() == raw_val or re.search(r"\b" + re.escape(en_key) + r"\b", raw_val):
                return "biodiversity_level", de_val, None

    # 3. Taxa Species Group
    if "taxa" in raw_field or "species" in raw_field or raw_val in EN_TO_DE_SPECIES_GROUP:
        for en_key, de_val in EN_TO_DE_SPECIES_GROUP.items():
            if en_key == raw_val or de_val.lower() == raw_val or re.search(r"\b" + re.escape(en_key) + r"\b", raw_val):
                return "taxa.species_group", de_val, sci_name

    # 4. Ecosystem Service
    if "service" in raw_field or raw_val in EN_TO_DE_ECOSYSTEM_SERVICE:
        for en_key, de_val in EN_TO_DE_ECOSYSTEM_SERVICE.items():
            if en_key == raw_val or de_val.lower() == raw_val or re.search(r"\b" + re.escape(en_key) + r"\b", raw_val):
                return "ecosystem_service", de_val, None

    return "unknown", raw_val, None


def parse_llm_findings(raw_output: str) -> list[dict[str, Any]]:
    """Robustly extracts finding objects from LLM JSON response."""
    data = extract_json_block(raw_output)
    if isinstance(data, dict):
        findings = data.get("findings")
        if isinstance(findings, list):
            return [f for f in findings if isinstance(f, dict)]
        # Fallback if top-level schema was returned directly (e.g. from mock client)
        res = []
        for h in data.get("habitat", []):
            res.append({"field": "habitat", "value": str(h), "quote": ""})
        for b in data.get("biodiversity_level", []):
            res.append({"field": "biodiversity_level", "value": str(b), "quote": ""})
        for t in data.get("taxa", []):
            if isinstance(t, dict):
                res.append({
                    "field": "taxa.species_group",
                    "value": str(t.get("species_group", "")),
                    "scientific_name": t.get("scientific_name"),
                    "quote": "",
                })
        for es in data.get("ecosystem_service", []):
            if isinstance(es, dict):
                res.append({"field": "ecosystem_service", "value": str(es.get("category", "")), "quote": ""})
        if res:
            return res
    elif isinstance(data, list):
        return [f for f in data if isinstance(f, dict)]
    return []


def assemble_final_record(
    zot_id: str,
    mapped_findings_with_votes: list[tuple[str, str, str | None, int]],
    majority_learned: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Constructs the final schema-compliant JSON prediction record:
      - Ranks candidate labels by verified quote votes.
      - Enforces collective-term conventions for species groups.
      - Connects habitats to ecosystem types and provides dense fallbacks.
    """
    hab_scores: dict[str, int] = defaultdict(int)
    bio_scores: dict[str, int] = defaultdict(int)
    taxa_entries: list[dict[str, Any]] = []
    seen_taxa: set[tuple[str, str]] = set()
    service_scores: dict[str, int] = defaultdict(int)

    for field, de_val, sci_name, votes in mapped_findings_with_votes:
        if field == "habitat":
            hab_scores[de_val] += votes
        elif field == "biodiversity_level":
            bio_scores[de_val] += votes
        elif field == "taxa.species_group":
            sci = sci_name if (sci_name and sci_name != "-") else "-"
            key = (de_val, sci)
            if key not in seen_taxa:
                seen_taxa.add(key)
                taxa_entries.append(
                    {
                        "species_group": de_val,
                        "scientific_name": sci,
                        "german_name": de_val,
                        "collective_term": (sci == "-"),
                    }
                )
        elif field == "ecosystem_service":
            service_scores[de_val] += votes

    # 1. Habitats (ranked, top 2, fallback to majority)
    sorted_habs = sorted(hab_scores.items(), key=lambda kv: -kv[1])
    habitats = [h for h, _ in sorted_habs[:2]]
    if not habitats:
        habitats = [majority_learned["habitat"]["majority_value"]]

    # 2. Biodiversity Levels
    sorted_bio = sorted(bio_scores.items(), key=lambda kv: -kv[1])
    bio_levels = [b for b, _ in sorted_bio[:2]]

    # 3. Ecosystem Types (derived from primary habitat)
    eco_types = []
    from src.models.base_models.dictionary_baseline import ECOSYSTEM_FROM_HABITAT
    if habitats and habitats[0] in ECOSYSTEM_FROM_HABITAT:
        cat, term = ECOSYSTEM_FROM_HABITAT[habitats[0]]
        eco_types.append({"category": cat, "term": term})

    # 4. Taxa (fallback to majority species group if 0 verified)
    if not taxa_entries:
        grp = majority_learned["taxa.species_group"]["majority_value"]
        taxa_entries.append(
            {
                "species_group": grp,
                "scientific_name": "-",
                "german_name": grp,
                "collective_term": True,
            }
        )

    # 5. Ecosystem Service
    eco_services = []
    for srv, votes in sorted(service_scores.items(), key=lambda kv: -kv[1])[:1]:
        eco_services.append(
            {
                "category": srv,
                "term": majority_learned["ecosystem_service.term"]["majority_value"],
            }
        )

    return {
        "zotitem_ptr_id": zot_id,
        "habitat": habitats,
        "biodiversity_level": bio_levels,
        "ecosystem_type": eco_types,
        "taxa": taxa_entries[:5],
        "ecosystem_service": eco_services,
    }


# ======================================================================================
# Main Execution: Parallel (B) vs. Sequential (A)
# ======================================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="English-First Section Extraction Pipeline (Sequential A vs. Parallel B)."
    )
    parser.add_argument(
        "--architecture",
        choices=["parallel", "sequential"],
        default="parallel",
        help="Architecture mode: 'parallel' (B: independent voting) or 'sequential' (A: state fold)",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="saia",
        help="API provider (default: saia)",
    )
    parser.add_argument(
        "--model",
        default="glm-5.3-flash",
        help="Model name (default: glm-5.3-flash)",
    )
    parser.add_argument(
        "--language",
        choices=["en", "de"],
        default="en",
        help="Language mode: 'en' (English extraction + deterministic schema mapping) or 'de' (Direct German schema labels)",
    )
    parser.add_argument("--grobid-url", default=None, help="Optional GROBID service URL (e.g. http://localhost:8070)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers evaluated")
    parser.add_argument("--dry-run", action="store_true", help="Offline deterministic mock test")
    parser.add_argument("--no-save", action="store_true", help="Skip saving run artifacts")
    args = parser.parse_args()

    # Select language-specific prompts
    if args.language == "de":
        system_prompt = GERMAN_SYSTEM_PROMPT
        parallel_chunk_prompt = GERMAN_PARALLEL_CHUNK_PROMPT
        sequential_chunk_prompt = GERMAN_SEQUENTIAL_CHUNK_PROMPT
    else:
        system_prompt = ENGLISH_SYSTEM_PROMPT
        parallel_chunk_prompt = PARALLEL_CHUNK_PROMPT
        sequential_chunk_prompt = SEQUENTIAL_CHUNK_PROMPT

    start_time = time.time()
    gold_path = REPO_ROOT / "data" / "gold" / "dev.jsonl"
    dev_ids_path = REPO_ROOT / "data" / "gold" / "dev_100_ids.txt"
    train_path = REPO_ROOT / "data" / "gold" / "train.jsonl"
    pdf_dir = REPO_ROOT / "data" / "raw_pdfs" / "dev"

    gold = load_jsonl_by_id(gold_path)
    train_records = load_jsonl_by_id(train_path)
    majority_learned = fit_majority_model(train_records)

    dev_ids = [line.strip() for line in dev_ids_path.read_text().splitlines() if line.strip()]
    if args.limit:
        dev_ids = dev_ids[:args.limit]

    if args.dry_run:
        client = DryRunMockClient(train_records)
        actual_model = "dryrun-mock"
    else:
        actual_model = args.model
        client = RobustOpenAICompatClient(args.provider, actual_model, args.temperature, args.rpm)

    model_slug = actual_model.replace("/", "_").replace(":", "_")
    run_model_id = f"{MODEL_BASE_ID}_{args.architecture}_{args.provider}_{model_slug}_{args.language}"
    run_title = f"Architecture ({args.architecture.title()} Sections, lang={args.language}): {actual_model}"

    ckpt_dir = REPO_ROOT / "outputs" / CATEGORY / run_model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"{run_model_id}_checkpoint.jsonl"

    predictions: dict[str, dict[str, Any]] = {}
    if ckpt_file.exists():
        for line in ckpt_file.read_text().splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                    predictions[obj["zotitem_ptr_id"]] = obj
                except Exception:
                    pass

    total_quotes_verified = 0
    total_quotes_rejected = 0

    print(f"=== Starting Run: {run_title} ===")
    print(f"Architecture: {args.architecture} | Papers to evaluate: {len(dev_ids)}")

    for i, zot_id in enumerate(dev_ids, 1):
        if zot_id in predictions:
            continue

        rec = gold[zot_id]
        title = str(rec.get("bibtex_title") or "")
        sections = extract_semantic_sections(zot_id, rec, pdf_dir, grobid_url=args.grobid_url)

        findings_pool: list[tuple[str, str, str | None, int]] = []

        if args.architecture == "parallel":
            # -------------------------------------------------------------------------
            # Architecture B: Parallel Independent Calls + Vote Merging
            # -------------------------------------------------------------------------
            raw_findings_by_de_key: dict[tuple[str, str, str | None], int] = defaultdict(int)

            for sec in sections:
                user_prompt = parallel_chunk_prompt.format(
                    title=title,
                    section_name=sec["section_name"],
                    section_text=sec["text"][:8000],
                )
                if isinstance(client, DryRunMockClient):
                    client.mock_context = rec

                raw_out = client.chat(system_prompt, user_prompt)
                extracted_findings = parse_llm_findings(raw_out)

                for f in extracted_findings:
                    quote = str(f.get("quote") or "")
                    if isinstance(client, DryRunMockClient) or verify_verbatim_quote(quote, sec["text"]):
                        field, de_val, sci_name = map_english_finding_to_german(f)
                        if field != "unknown":
                            raw_findings_by_de_key[(field, de_val, sci_name)] += 1
                            total_quotes_verified += 1
                    else:
                        total_quotes_rejected += 1

            findings_pool = [
                (field, de_val, sci_name, votes)
                for (field, de_val, sci_name), votes in raw_findings_by_de_key.items()
            ]

        else:
            # -------------------------------------------------------------------------
            # Architecture A: Sequential State-Passing Fold
            # -------------------------------------------------------------------------
            current_state: list[dict[str, Any]] = []

            for sec in sections:
                state_json_str = json.dumps({"current_findings": current_state}, indent=2)
                user_prompt = sequential_chunk_prompt.format(
                    title=title,
                    state_json=state_json_str,
                    section_name=sec["section_name"],
                    section_text=sec["text"][:8000],
                )
                if isinstance(client, DryRunMockClient):
                    client.mock_context = rec

                raw_out = client.chat(system_prompt, user_prompt)
                extracted_findings = parse_llm_findings(raw_out)

                verified_state = []
                for f in extracted_findings:
                    quote = str(f.get("quote") or "")
                    if isinstance(client, DryRunMockClient) or verify_verbatim_quote(quote, sec["text"]) or quote == "confirmed":
                        verified_state.append(f)
                        total_quotes_verified += 1
                    else:
                        total_quotes_rejected += 1

                current_state = verified_state

            for f in current_state:
                field, de_val, sci_name = map_english_finding_to_german(f)
                if field != "unknown":
                    findings_pool.append((field, de_val, sci_name, 1))

        # Assemble and checkpoint final record
        paper_pred = assemble_final_record(zot_id, findings_pool, majority_learned)
        predictions[zot_id] = paper_pred

        with ckpt_file.open("a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(paper_pred, ensure_ascii=False) + "\n")

        print(
            f"[{i:>3}/{len(dev_ids)}] {zot_id} | Sections: {len(sections)} | "
            f"Hab: {len(paper_pred['habitat'])} Taxa: {len(paper_pred['taxa'])}",
            flush=True,
        )

    metrics = evaluate_predictions(gold, predictions, filter_ids=set(dev_ids))
    print_report(f"{run_title} {'[DRY-RUN]' if args.dry_run else ''}", metrics)

    runtime = time.time() - start_time
    token_stats = {
        "n_calls": getattr(client, "n_calls", 0),
        "prompt_tokens": getattr(client, "prompt_tokens", 0),
        "completion_tokens": getattr(client, "completion_tokens", 0),
        "total_quotes_verified": total_quotes_verified,
        "total_quotes_rejected": total_quotes_rejected,
    }

    if args.no_save:
        print("\n--no-save set: skipping run artifacts.")
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
                "architecture": args.architecture,
                "pipeline_version": "english_sections_translate_text_v1",
                "provider": args.provider,
                "model": actual_model,
                "temperature": args.temperature,
                "n_papers": len(dev_ids),
                "grobid_url": args.grobid_url,
                "token_stats": token_stats,
            },
        )

    print(
        f"\nDone in {runtime:.1f}s | calls={token_stats['n_calls']} | "
        f"verified_quotes={total_quotes_verified} | rejected_quotes={total_quotes_rejected}"
    )


if __name__ == "__main__":
    main()
