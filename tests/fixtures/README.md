# Test fixtures

Claude Code develops and tests the code only with data in this folder, never with
course data. Put here:

- **invented test data** created by Claude Code: short texts with headings, tiny PDFs
  generated from them, and reference records in the same format as the real ones;
- optionally, **open-access papers (CC BY)** to test Docling on realistic layouts.
  Before adding one, check that its title does not appear in the course references.

Suggested open-access papers:

| Paper | Source |
|---|---|
| Zeng et al. (2019), The Influence of Bt Maize Cultivation on Communities of Arbuscular Mycorrhizal Fungi Revealed by MiSeq Sequencing, Frontiers in Microbiology | https://pmc.ncbi.nlm.nih.gov/articles/PMC6334669/ |
| Szoboszlay et al. (2019), Annual replication is essential in evaluating the response of the soil microbiome to the genetic modification of maize in different biogeographical regions, PLoS ONE | https://pmc.ncbi.nlm.nih.gov/articles/PMC6917299 |
| Duc et al. (2011), No Adverse Effect of Genetically Modified Antifungal Wheat on Decomposition Dynamics and the Soil Fauna Community – A Field Study, PLoS ONE | https://doi.org/10.1371/journal.pone.0025014 |

## What is here (M0)

- `experiment_override.yaml`: an invented experiment file for `tests/test_settings.py`.
- The invented reference records are **generated** in `tests/conftest.py` from the schema
  enums (e.g. "the first allowed habitat"), so no label string is retyped in the tests.

## What is here (M1)

- `with_headings.pdf`, `without_headings.pdf`: two invented 2-page PDFs (a short "paper"
  with numbered headings, and plain paragraphs only) for the Docling tests in
  `tests/test_pdf_to_text.py`. Made by `make_fixture_pdfs.py` (`python3 tests/fixtures/make_fixture_pdfs.py`).
- The chunking tests (`tests/test_chunking.py`) build tiny Docling documents in code and
  count tokens as words (`WordTokenizer` in `tests/conftest.py`).
- `tests/many_pages_document.py` builds an invented 120-page Docling document in code
  (lists and captions across pages, empty pages, ...). `tests/test_long_documents.py` uses it
  to check that the speed-ups for long documents give the same text and chunks as before.

## What is here (M2)

- `llm_smoke_prompt.txt`: an invented short study text with a question, sent to SAIA by
  the team with `python3 src/llm_client.py ask --prompt-file tests/fixtures/llm_smoke_prompt.txt`.
- The LLM client tests (`tests/test_llm_client.py`) never reach the API: they answer every
  request with an `httpx.MockTransport` and replace `sleep` and the clock, so they never wait.

## What is here (M3a)

- No new files: the pipeline F tests (`tests/test_full_text_pipeline.py`) build three
  invented Docling documents in code and answer every prompt with invented JSON through
  an `httpx.MockTransport`. `tests/conftest.py` writes invented stand-ins for
  `data/labels/term_to_category.json` and `variable_vocabulary.json` into a temporary
  folder, so no test depends on files derived from course data.
- `tests/test_parallel_requests.py` uses a fake slow API (real short delays) to show
  that requests overlap and that the per-key limits still hold.
- `tests/test_pdf_to_text.py` converts the two fixture PDFs with 1 and 2 worker processes
  and compares the results.
