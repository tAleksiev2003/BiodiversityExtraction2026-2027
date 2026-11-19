"""Prompt v4: first pages, neighbours, the example section and the vote check (invented data only).

PDFs are the invented fixture PDFs or empty files; first-pages texts, references and
neighbours are written into a temporary folder. The sentence-embedding model is replaced
by an invented bag-of-words encoder, so no model is downloaded and no request is sent.
"""
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data import ID_FIELD, write_jsonl
from few_shot_examples import EXAMPLE_FIELDS, NO_EXAMPLES, FewShotExamples, format_examples
from first_pages import LOG_NAME, STATUS_DONE, STATUS_FAILED, extract_documents, read_first_pages
from llm_client import LLMClient
from pipelines.full_text import FullTextPipeline, run_full_text
from prompts import ExtractionPrompt
from retrieval import (
    METHOD_EMBEDDING, METHOD_TFIDF, cached_embeddings, find_neighbours, neighbours_path, write_neighbours)
from settings import REPO_ROOT, load_settings

sys.path.insert(0, str(REPO_ROOT / "src" / "scripts"))
from evaluate_neighbours import vote_record, votes_needed  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures"
PROMPT_V2_EXPERIMENT = REPO_ROOT / "config" / "experiments" / "prompt_v2.yaml"
PROMPT_V4_EXPERIMENT = REPO_ROOT / "config" / "experiments" / "prompt_v4.yaml"
SECTION_HEADING = "## How the database labelled similar documents (training examples)"
TEXT_HEADING = "## Text"

# Invented first pages. The test papers share their topic with one train paper each;
# SHARED01 is in train AND test with the same text, so it would be the best match.
TRAIN_TEXTS = {
    "TRAIN001": "Carabid beetles in hedgerows: pitfall traps and beetle species richness",
    "TRAIN002": "Breeding birds in a beech forest: territory mapping of forest birds",
    "TRAIN003": "Fish in a lowland river: electrofishing of river fish communities",
    "SHARED01": "Earthworms in arable soil: earthworm density after ploughing",
    "NOREF001": "Carabid beetles in hedgerows and beetle traps",  # PDF without a reference
}
TEST_TEXTS = {
    "TEST0001": "Beetle traps in hedgerows: carabid beetles and species richness",
    "TEST0002": "Forest birds: mapping breeding birds in beech forest",
    "SHARED01": "Earthworms in arable soil: earthworm density after ploughing",
}
TEST_WITHOUT_TEXT = "TEST0003"


# --- invented train references ----------------------------------------------------------------

@pytest.fixture
def train_references(enum) -> dict[str, dict[str, Any]]:
    """Every train paper with labels; values picked by position from the schema enums."""
    def record(document_id: str, index: int, taxa_count: int = 1) -> dict[str, Any]:
        return {
            ID_FIELD: document_id,
            "bibtex_title": f"Invented title {document_id}",  # the key of the real references (few_shot.title_field)
            "bibtex_type": "article",
            "habitat": [enum("habitat", index)],
            "biodiversity_level": [] if document_id == "TRAIN003" else [enum("biodiversity_level", index)],
            "ecosystem_type": [{"category": enum("ecosystem_type.category", index),
                                "term": enum("ecosystem_type.term", index), "description": "not shown"}],
            "taxa": [{"species_group": enum("taxa.species_group", index),
                      "scientific_name": f"Inventus {document_id.lower()} {number}",
                      "german_name": f"Erfundene Art {document_id} {number}", "collective_term": False}
                     for number in range(taxa_count)],
            "biodiversity_variable": [f"Variable {document_id}"],
            "ecosystem_service": [{"category": enum("ecosystem_service.category", index % 2), "term": None,
                                   "details": "not shown"}],
        }
    records = [record("TRAIN001", 0), record("TRAIN002", 1), record("TRAIN003", 2, taxa_count=12),
               record("SHARED01", 3)]
    return {entry[ID_FIELD]: entry for entry in records}


@pytest.fixture
def retrieval_settings(invented_settings, tmp_path, train_references) -> dict[str, Any]:
    """Settings whose first pages, PDFs and references are invented files in tmp_path."""
    settings = copy.deepcopy(invented_settings)
    paths = settings["paths"]
    paths.update(first_pages=tmp_path / "first_pages", processed=tmp_path / "processed",
                 embeddings=tmp_path / "embeddings", pdfs_test=tmp_path / "pdfs" / "test",
                 references_train=tmp_path / "references" / "train.jsonl")
    settings["retrieval"].update(tfidf_min_df=1, tfidf_max_df=1.0)  # the corpus has only 4 papers
    for split, texts in (("train", TRAIN_TEXTS), ("test", TEST_TEXTS)):
        for document_id, text in texts.items():
            path = paths["first_pages"] / split / f"{document_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    paths["pdfs_test"].mkdir(parents=True)
    for document_id in [*TEST_TEXTS, TEST_WITHOUT_TEXT]:
        (paths["pdfs_test"] / f"{document_id}.pdf").touch()  # only the ids matter here
    write_jsonl(list(train_references.values()), paths["references_train"])
    return settings


class BagOfWordsEncoder:
    """Invented stand-in for the sentence-embedding model; counts the texts it embeds."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.embedded.extend(texts)
        vocabulary = sorted({word for text in [*TRAIN_TEXTS.values(), *TEST_TEXTS.values()]
                             for word in re.findall(r"\w+", text.lower())})
        rows = np.array([[re.findall(r"\w+", text.lower()).count(word) for word in vocabulary]
                         for text in texts], dtype=float)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


# --- first pages -------------------------------------------------------------------------------

def test_first_pages_are_cut_at_the_character_limit():
    full, pages = read_first_pages(FIXTURES / "with_headings.pdf", 100_000)
    short, short_pages = read_first_pages(FIXTURES / "with_headings.pdf", 50)
    assert pages == 2 and short_pages == 1  # stops reading once the limit is reached
    assert len(short) <= 50 and full.startswith(short)


def test_first_pages_skip_done_documents_and_log_failures(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "GOOD0001.pdf").write_bytes((FIXTURES / "with_headings.pdf").read_bytes())
    (pdf_dir / "BROKEN01.pdf").write_bytes(b"not a pdf")
    output_dir = tmp_path / "first_pages"

    counts = extract_documents(pdf_dir, ["GOOD0001", "BROKEN01"], output_dir, 200)
    assert counts[STATUS_DONE] == 1 and counts[STATUS_FAILED] == 1
    assert (output_dir / "GOOD0001.txt").exists() and not (output_dir / "BROKEN01.txt").exists()

    again = extract_documents(pdf_dir, ["GOOD0001", "BROKEN01"], output_dir, 200)
    assert again["skipped"] == 1 and again[STATUS_FAILED] == 1  # the failed one is tried again
    log = [json.loads(line) for line in (output_dir / LOG_NAME).read_text().splitlines()]
    assert [entry["status"] for entry in log] == [STATUS_DONE, STATUS_FAILED, STATUS_FAILED]


# --- neighbours --------------------------------------------------------------------------------

@pytest.mark.parametrize("method", [METHOD_TFIDF, METHOD_EMBEDDING])
def test_neighbours_come_only_from_train_papers_with_a_reference(retrieval_settings, method):
    neighbours, without_text = find_neighbours(retrieval_settings, "test", method, n=10,
                                               encoder=BagOfWordsEncoder())
    assert without_text == [TEST_WITHOUT_TEXT]
    assert set(neighbours) == set(TEST_TEXTS)
    allowed = {"TRAIN001", "TRAIN002", "TRAIN003"}  # not SHARED01 (in test), not NOREF001 (no reference)
    for ranked in neighbours.values():
        assert {train_id for train_id, _ in ranked} == allowed
    assert neighbours["TEST0001"][0][0] == "TRAIN001"
    assert neighbours["TEST0002"][0][0] == "TRAIN002"


def test_neighbours_are_never_found_for_the_train_split(retrieval_settings):
    with pytest.raises(ValueError):
        find_neighbours(retrieval_settings, "train", METHOD_TFIDF, n=3)


def test_embeddings_are_cached(retrieval_settings):
    encoder = BagOfWordsEncoder()
    first, _ = find_neighbours(retrieval_settings, "test", METHOD_EMBEDDING, n=2, encoder=encoder)
    embedded = len(encoder.embedded)
    assert embedded == len(TEST_TEXTS) + 3
    second, _ = find_neighbours(retrieval_settings, "test", METHOD_EMBEDDING, n=2, encoder=encoder)
    assert len(encoder.embedded) == embedded and second == first

    cache_path = retrieval_settings["paths"]["embeddings"] / "cache.npz"
    cached_embeddings({"A": "beetle"}, encoder, cache_path)
    cached_embeddings({"A": "beetle birds"}, encoder, cache_path)  # changed text: embedded again
    assert encoder.embedded[-2:] == ["beetle", "beetle birds"]


# --- the example section -----------------------------------------------------------------------

def quoted_strings(text: str) -> set[str]:
    return set(re.findall(r'"([^"]*)"', text))


def shown_values(reference: dict[str, Any], max_taxa: int) -> set[str]:
    """Title and every label of the shown fields, straight from the reference."""
    values = {reference["bibtex_title"]}
    for field, subfields in EXAMPLE_FIELDS.items():
        entries = reference.get(field) or []
        if field == "taxa":
            entries = entries[:max_taxa]
        for entry in entries:
            if subfields:
                values.update(entry.get(name) for name in subfields)
            else:
                values.add(entry)
    values.discard(None)
    return values


def test_example_section_lists_exactly_the_neighbours_gold_labels(train_references, settings):
    few_shot = settings["few_shot"]
    examples = FewShotExamples({"TEST0001": ["TRAIN001", "TRAIN003"]}, train_references, few_shot)
    text = examples.text_for("TEST0001")

    expected = (shown_values(train_references["TRAIN001"], few_shot["max_taxa_per_example"])
                | shown_values(train_references["TRAIN003"], few_shot["max_taxa_per_example"]))
    assert quoted_strings(text) == expected            # nothing of TRAIN002, nothing else
    assert "Variable" not in text and "not shown" not in text and "collective" not in text
    assert "biodiversity_level: (empty)" in text       # TRAIN003 has none
    assert "(and 2 more)" in text                       # TRAIN003 has 12 taxa, 10 are shown
    assert text.index("Example 1") < text.index("TRAIN001") < text.index("Example 2") < text.index("TRAIN003")
    assert text.count("bibtex_type: article") == 2


def test_document_without_neighbours_gets_a_note(train_references, settings):
    examples = FewShotExamples({"TEST0001": ["TRAIN001"]}, train_references, settings["few_shot"])
    assert examples.text_for(TEST_WITHOUT_TEXT) == NO_EXAMPLES == format_examples([], settings["few_shot"])


def test_examples_refuse_neighbours_outside_train(train_references, settings):
    with pytest.raises(ValueError):
        FewShotExamples({"TEST0001": ["TEST0002"]}, train_references, settings["few_shot"])
    with pytest.raises(ValueError):
        FewShotExamples({"TRAIN001": ["TRAIN001"]}, train_references, settings["few_shot"])


def test_neighbour_file_must_match_method_and_k(retrieval_settings):
    neighbours, without_text = find_neighbours(retrieval_settings, "test", METHOD_TFIDF, n=2)
    write_neighbours(retrieval_settings, "test", METHOD_TFIDF, 2, neighbours, without_text)
    assert neighbours_path(retrieval_settings, "test").name == "neighbours_test.json"

    retrieval_settings["retrieval"].update(method=METHOD_TFIDF, k=2)
    examples = FewShotExamples.from_settings(retrieval_settings, "test")
    assert "TRAIN001" in examples.text_for("TEST0001")
    retrieval_settings["retrieval"]["k"] = 3
    with pytest.raises(ValueError, match="k=2"):
        FewShotExamples.from_settings(retrieval_settings, "test")


# --- prompt v4 ---------------------------------------------------------------------------------

def prompt_with(settings: dict[str, Any], experiment: Path) -> ExtractionPrompt:
    changed = copy.deepcopy(settings)
    changed["paths"]["prompt_extraction"] = load_settings(experiment)["paths"]["prompt_extraction"]
    return ExtractionPrompt.from_settings(changed)


def test_default_is_v4_and_v2_is_an_experiment(settings):
    assert settings["paths"]["prompt_extraction"].name == "extraction_v4.txt"
    assert ExtractionPrompt.from_settings(settings).uses_examples
    assert not ExtractionPrompt.from_settings(load_settings(PROMPT_V2_EXPERIMENT)).uses_examples


def test_v4_equals_v2_except_the_example_section(invented_settings):
    text = "Invented text with {braces} and {examples}."
    v2 = prompt_with(invented_settings, PROMPT_V2_EXPERIMENT).messages("DOC00001", "1/1", text)
    v4 = prompt_with(invented_settings, PROMPT_V4_EXPERIMENT).messages("DOC00001", "1/1", text, "EXAMPLES")
    assert v4[0] == v2[0]  # system message unchanged
    user = v4[1]["content"]
    start, end = user.index(SECTION_HEADING), user.index(TEXT_HEADING)
    assert user[:start] + user[end:] == v2[1]["content"]
    assert user[start:end].endswith("\n\nEXAMPLES\n\n")
    assert user.count("EXAMPLES") == 1 and text in user  # the document text is sent unchanged


def test_v4_needs_examples(invented_settings):
    prompt = prompt_with(invented_settings, PROMPT_V4_EXPERIMENT)
    assert prompt.uses_examples
    with pytest.raises(ValueError):
        prompt.messages("DOC00001", "1/1", "Invented text.")


def test_pipeline_f_sends_each_document_its_own_examples(invented_settings, train_references, tmp_path,
                                                          word_tokenizer, enum, specs):
    # Reuses the invented documents and fake API of the pipeline F tests.
    from test_full_text_pipeline import ALL_IDS, INVENTED_DOCUMENTS, InventedLlm, build_document
    from pdf_to_text import save_document

    settings = copy.deepcopy(invented_settings)
    settings["paths"]["prompt_extraction"] = load_settings(PROMPT_V4_EXPERIMENT)["paths"]["prompt_extraction"]
    document_dir = settings["paths"]["docling"] / "dev"
    for document_id, sections in INVENTED_DOCUMENTS.items():
        save_document(build_document(sections), document_dir / f"{document_id}.json")
    examples = FewShotExamples({"DOC00001": ["TRAIN001"], "DOC00003": ["TRAIN002", "TRAIN003"]},
                               train_references, settings["few_shot"])
    with pytest.raises(ValueError):  # v4 without examples
        FullTextPipeline(settings, None, document_dir, word_tokenizer(10_000))

    llm = InventedLlm(enum, specs["ecosystem_service.term"].allowed_values[-1])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    client = LLMClient(settings, call_log_path=run_dir / "llm_calls.jsonl", openai_clients=[llm.openai_client()])
    run_full_text(settings, ALL_IDS, client, run_dir, document_dir, word_tokenizer(10_000), examples)

    prompts = {}
    for messages in llm.prompts:
        document_id = re.search(r"of document (\w+)", messages[1]["content"]).group(1)
        prompts[document_id] = messages[1]["content"]
    assert "Invented title TRAIN001" in prompts["DOC00001"] and "TRAIN002" not in prompts["DOC00001"]
    assert "Invented title TRAIN002" in prompts["DOC00003"] and "Invented title TRAIN003" in prompts["DOC00003"]
    assert NO_EXAMPLES in prompts["DOC00002"]


# --- quality check: vote of the neighbours -----------------------------------------------------

def test_votes_needed_is_at_least_half():
    assert [votes_needed(k, 0.5) for k in (1, 2, 3, 5)] == [1, 1, 2, 3]


def test_vote_keeps_labels_of_at_least_half_of_the_neighbours(train_references, enum):
    from evaluation import extract_values

    first, second, third = (train_references[i] for i in ("TRAIN001", "TRAIN002", "TRAIN003"))
    third = {**third, "habitat": first["habitat"]}  # TRAIN001's habitat now in 2 of 3
    record = vote_record("TEST0001", [first, second, third], 0.5)
    assert extract_values(record, "habitat") == set(first["habitat"])
    assert extract_values(record, "taxa.species_group") == set()  # each group only once
    assert extract_values(record, "taxa.scientific_name") == set()

    single = vote_record("TEST0001", [third], 0.5)
    for field in ("ecosystem_type.term", "taxa.scientific_name", "ecosystem_service.category"):
        assert extract_values(single, field) == extract_values(third, field)
    assert extract_values(single, "ecosystem_service.term") == set()
