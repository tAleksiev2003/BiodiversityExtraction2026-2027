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

---

## 4. Planned Smart Preprocessing Router (Zero-Overfitting)

Once Marker and Docling benchmarks finish, we will evaluate an objective, document-agnostic routing cascade (validated via bootstrap / 5-fold CV on Dev-100):
1. **Primary (Academic Multi-Section Papers):** Use **GROBID** if it detects `>= 2` structured sections and retains sufficient text coverage vs. raw PyMuPDF.
2. **Secondary (Brochures, Reports, Table-Heavy PDFs):** Route to **Marker** or **Docling** (whichever shows higher stability and accuracy on non-academic layouts).
3. **Fallback:** Fall back to **PyMuPDF** whenever Marker/Docling errors or returns `< 500` characters.
