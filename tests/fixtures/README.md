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
