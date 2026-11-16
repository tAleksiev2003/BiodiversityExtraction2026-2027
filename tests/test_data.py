"""Tests for src/data.py with invented files in tmp_path."""
import pytest

from data import (
    ID_FIELD, dev_ids_with_pdf, list_pdf_ids, load_references, make_dev_references,
    read_jsonl, select_dev_subset, write_jsonl,
)


def test_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_jsonl(path)


def test_duplicate_id_is_rejected(tmp_path):
    path = tmp_path / "refs.jsonl"
    write_jsonl([{ID_FIELD: "A"}, {ID_FIELD: "A"}], path)
    with pytest.raises(ValueError, match="duplicate id"):
        load_references(path)


def test_jsonl_round_trip_keeps_umlauts(tmp_path):
    path = tmp_path / "refs.jsonl"
    write_jsonl([{ID_FIELD: "A", "note": "Gewässer"}], path)
    assert "Gewässer" in path.read_text(encoding="utf-8")
    assert read_jsonl(path) == [{ID_FIELD: "A", "note": "Gewässer"}]


def test_select_dev_subset_is_reproducible():
    ids = [f"D{number}" for number in range(20)]
    assert select_dev_subset(ids, 5, seed=1) == select_dev_subset(ids, 5, seed=1)
    assert len(select_dev_subset(ids, 5, seed=1)) == 5
    assert select_dev_subset(ids, None, seed=1) == sorted(ids)
    assert select_dev_subset(ids, 100, seed=1) == sorted(ids)


def _invented_dev_setup(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for document_id in ["A", "B", "Z"]:  # Z has no reference record
        (pdf_dir / f"{document_id}.pdf").write_bytes(b"%PDF-invented")
    (pdf_dir / "notes.txt").write_text("not a pdf")
    references_path = tmp_path / "dev.jsonl"
    write_jsonl([{ID_FIELD: "A"}, {ID_FIELD: "B"}, {ID_FIELD: "C"}], references_path)
    return {
        "paths": {
            "pdfs_dev": pdf_dir,
            "references_dev": references_path,
            "references_dev_with_pdf": tmp_path / "processed" / "dev_with_pdf.jsonl",
        },
        "data": {"dev_subset_size": None, "dev_subset_seed": 0},
    }


def test_list_pdf_ids_uses_file_stems(tmp_path):
    settings = _invented_dev_setup(tmp_path)
    assert list_pdf_ids(settings["paths"]["pdfs_dev"]) == ["A", "B", "Z"]


def test_dev_with_pdf_keeps_only_documents_with_reference_and_pdf(tmp_path):
    settings = _invented_dev_setup(tmp_path)
    assert dev_ids_with_pdf(settings) == ["A", "B"]
    output_path = make_dev_references(settings)
    assert [record[ID_FIELD] for record in read_jsonl(output_path)] == ["A", "B"]
