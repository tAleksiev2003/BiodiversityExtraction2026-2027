# Preprocessing Benchmark & Reproducibility Log (4 PDF Parsers)

This document logs the exact configuration, commands, hardware environment, and evaluation results for comparing all four PDF-to-text/section parsers inside **Architecture 3 (`src/models/architectures/llm_rag_sections.py`)** on the **100 validated Dev papers** (`data/gold/dev_100_ids.txt`).

---

## 1. Exact Parser Configurations

| Parser | Version / Service | Execution Environment | Sectioning & Filtering | Cache Location |
| :--- | :--- | :--- | :--- | :--- |
| **1. PyMuPDF** *(Baseline)* | `pymupdf` (`fitz`) | `lenlenovo` CPU | Page-group heuristic (`pages[0:2]`, `[2:5]`, `[5:10]`, `[10:16]`), drops pages 17+ | On-the-fly (`<0.1s`/PDF) |
| **2. GROBID** | `grobid/grobid:0.9.1-crf` (`http://127.0.0.1:8070`) | `lenlenovo` Docker | Multipart upload to `/api/processFulltextDocument`; TEI `<abstract>` + `<body>/<div>` mapped by `<head>` regex into `intro`, `methods`, `results`, `discussion`; `<back>` references stripped | `data/parsed_text/grobid_dev/<id>.tei.xml` |
| **3. Marker** | `marker-pdf==2.2.0` (`PdfConverter`) | `lenserver-hp` CPU (8 cores, 10 GiB RAM free; isolated `/home/lennard/tmp_pdf_parse`) | Markdown output split on `#{1,3}` headings, stops at `References/Bibliography/Literatur/Appendix`, grouped into `<= 4` blocks; falls back to PyMuPDF on conversion errors | `data/parsed_text/marker_dev/<id>.md` |
| **4. Docling** | `docling==2.151.1` (`do_ocr=False`, `do_table_structure=True`, `document_timeout=240.0`) | `lenserver-hp` CPU (isolated `/home/lennard/tmp_pdf_parse`) | Markdown output split identically via `_split_markdown_to_sections()`; falls back to PyMuPDF on error/timeout | `data/parsed_text/docling_dev/<id>.md` |
| **5. Full Docling (Minh's JSON)** | `docling` with `do_ocr=True`, `do_table_structure=True` (Minh Anh's converter from branch `minhanh_testing`) | University GPU server batch, pulled to `lenlenovo` | `extract_minh_json_sections()`: body in reading order, new section at every `section_header`, `[page N]` markers, footers skipped, grouped into `<= 4` blocks; falls back to the AutoParser router if the JSON is missing | `data/processed/docling/dev/<id>.json` |

All four parsers feed into the exact same downstream prompt (`max_total_chars = 16000` distributed evenly across sections, Top-2 TF-IDF `train.jsonl` kNN anchors, schema guidelines + full term enums).

---

## 2. Commands Used to Run Benchmarks

```bash
# 1. PyMuPDF (default)
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider saia --model deepseek-v4.1-flash-1017 --rpm 20
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider hu --model hu-reasoning --rpm 15

# 2. GROBID
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider saia --model deepseek-v4.1-flash-1017 --grobid-url http://127.0.0.1:8070 --rpm 20
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider hu --model hu-reasoning --grobid-url http://127.0.0.1:8070 --rpm 15

# 3. Marker (using pre-converted cached Markdown in data/parsed_text/marker_dev/)
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider saia --model deepseek-v4.1-flash-1017 --marker --rpm 20
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider hu --model hu-reasoning --marker --rpm 15

# 4. Docling (using pre-converted cached Markdown in data/parsed_text/docling_dev/)
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider saia --model deepseek-v4.1-flash-1017 --docling --rpm 20
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider hu --model hu-reasoning --docling --rpm 15

# 5. Full Docling through Minh's JSON cache (fair rerun 2026-11-26; same v4 prompt as the AutoParser reference)
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider saia --model deepseek-v4.1-flash-1017 --minh-json --clean-taxa --v4-prompt --rpm 20
.venv/bin/python -u src/models/architectures/llm_rag_sections.py --provider hu --model hu-reasoning --minh-json --clean-taxa --v4-prompt --rpm 15
```

---

## 3. Benchmark Results (Dev-100)

| Parser | Model | C1 Macro-F1 | C1+C2b Macro-F1 | Micro-F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **PyMuPDF** | `deepseek-v4.1-flash-1017` | `0.5798` | `0.4273` | `0.4286` | 100/100 parsed |
| **PyMuPDF** | `hu-reasoning` (Qwen 3.6-27B) | `0.5539` | `0.4073` | `0.4126` | 100/100 parsed |
| **GROBID** | `deepseek-v4.1-flash-1017` | **`0.5957`** | **`0.4544`** | **`0.4390`** | +0.0159 C1 vs PyMuPDF (wins 25, loses 18) |
| **GROBID** | `hu-reasoning` (Qwen 3.6-27B) | **`0.5706`** | **`0.4353`** | **`0.4293`** | +0.0167 C1 vs PyMuPDF |
| **Marker** | `deepseek-v4.1-flash-1017` | `0.5985` | `0.4362` | `0.3784` | 100/100 converted |
| **Marker** | `hu-reasoning` (Qwen 3.6-27B) | `0.5837` | `0.4197` | `0.4015` | 100/100 converted |
| **Docling** | `deepseek-v4.1-flash-1017` | `0.5634` | `0.4221` | `0.4131` | 100/100 converted |
| **Docling** | `hu-reasoning` (Qwen 3.6-27B) | `0.5613` | `0.4018` | `0.3748` | 100/100 converted |
| **Full Docling (Minh's JSON, OCR on)** | `deepseek-v4.1-flash-1017` | `0.5883` | `0.4626` | `0.4804` | Fair rerun 2026-11-26 with `--clean-taxa --v4-prompt` |
| **Full Docling (Minh's JSON, OCR on)** | `hu-reasoning` (Qwen 3.6-27B) | `0.5894` | `0.4233` | `0.4426` | Fair rerun 2026-11-26 with `--clean-taxa --v4-prompt` |

Rows 1–8 share the plain baseline prompt, so they compare parsers directly. Rows 9–10 use the v4 prompt
and are only comparable with AutoParser under the same prompt: `0.5953 / 0.4744 / 0.5019`
(DeepSeek 4.1) and `0.5571 / 0.4176 / 0.4279` (Qwen3.6-27B). AutoParser stays ahead on DeepSeek 4.1
by 0.7 C1 points (within noise); full Docling is 3.2 points ahead on Qwen3.6-27B. Full tables:
[experiment_matrix.md](experiment_matrix.md) and section 3f in [experiment_log.md](experiment_log.md).

---

## 4. Planned Smart Preprocessing Router (Zero-Overfitting)

**Status (2026-11-21):** this router exists as `--auto-parser`: Marker when GROBID finds `<= 3` sections
or keeps `< 65 %` of the text, otherwise GROBID. The offline oracle analysis and its bootstrap are in
section 4 of [experiment_log.md](experiment_log.md).

Original plan — an objective, document-agnostic routing cascade (validated via bootstrap / 5-fold CV on Dev-100):
1. **Primary (Academic Multi-Section Papers):** Use **GROBID** if it detects `>= 2` structured sections and retains sufficient text coverage vs. raw PyMuPDF.
2. **Secondary (Brochures, Reports, Table-Heavy PDFs):** Route to **Marker** or **Docling** (whichever shows higher stability and accuracy on non-academic layouts).
3. **Fallback:** Fall back to **PyMuPDF** whenever Marker/Docling errors or returns `< 500` characters.
