# Field Error Analysis — where each field fails and how to fix it

Method: 7 older pipelines (2 providers × 3 architectures + ensemble, pre-v4) on Dev-100.
Per paper and field, every predicted-vs-gold mismatch was classified as FP-only (model invents),
FN-only (model misses), near-miss (string similarity ≥ 0.8), or different (both sides differ).
Counts below are paper-run mismatches (7 runs × 100 papers), so they rank the levers, not papers.
No train fitting involved — all fixes are prompt, post-processing, or evidence rules.
Caveat: pre-v4 runs; the per-paper rule will refresh this once new runs land.

## Priority order (by expected C1/C1+C2b gain, cheapest first)

### 1. biodiversity_level — default-bias to Artenvielfalt (C1 field, 30 papers never right)

- FP 192 / FN 101 / near 0 / different 72.
- Top false positives: Artenvielfalt (152×), Zusammensetzung (118×), Funktionelle Diversität (45×).
- Top missed: Zusammensetzung (66×), Habitatdiversität (66×), Artenvielfalt (59×).
- Hypothesis: the model defaults to Artenvielfalt whenever biodiversity is mentioned without a
  measurable aspect; compositional and habitat-diversity studies are systematically relabelled.
- Measures: (a) test Minh's level prompt (extra call, her level F1 0.412 → 0.455);
  (b) fold the level decision rules into the main prompt (single-call variant);
  (c) base-rate reminder in the prompt (Artenvielfalt needs an explicit diversity measure).
- Expected: biggest single C1 lever after taxa; directly attacks 30 never-right papers.

### 2. taxa.species_group — "Andere" never predicted (C1 field, 25 never right)

- FP 200 / FN 74 / near 0 / different 83.
- Top false positives: Gefäßpflanzen (165×), Andere Wirbellose (55×), Vögel (35×).
- Top missed: Andere (92×), Andere Pflanzen (39×), Andere Wirbellose (19×).
- Hypothesis: the model always picks a specific group while gold often says the collective
  "Andere" for mixed assemblages (same disease as the "-" problem in scientific names).
- Measures: collective-group rule mirroring the taxa convention (group entry only with
  species-level evidence, else "Andere"); explicit "Andere" examples in the few-shot anchors.
- Expected: C1 gain on 25 papers; pairs with the scientific-name collective fix.

### 3. ecosystem_type.category + term — terrestrial bias, water blindness (2 C1 fields)

- Category: FP 187 / FN 43. Top FP: III-terrestrisch (150×). Top missed: II-Binnengewässer (40×).
- Term: FP 205 / FN 72 / near 1. Top FP: Äcker und Ackerbrachen (77×), Laubwälder (45×),
  Trockenrasen (41×). Top missed: Fließende Gewässer (36×), Benthal/Pelagial der Nord-/Ostsee.
- Hypothesis: the model defaults to land categories; water studies get land labels while their
  water terms go unpredicted. Near-misses are ~0 — the concepts are wrong, not the spelling.
- Measures: (a) water mapping (water habitat/keywords → water category first);
  (b) test Minh's eco-type rule (drop "Äcker und Ackerbrachen" in land-only papers);
  (c) term-must-match-category check already exists — extend to category-must-match-habitat.
- Expected: two C1 fields move together; likely the largest combined C1 lever.

### 4. habitat — Agrar default, water/soil missed (C1 field, 16 never right)

- FP 81 / FN 86 / near 0 / different 47.
- Top false positives: Agrar- und Offenland (52×), Wald (46×), Boden (26×).
- Top missed: Binnengewässer und Auen (72×), Boden (65×), Wald (43×).
- Hypothesis: Agrar is the train majority and becomes the fallback; water/soil habitats need
  explicit cues the prompt does not demand.
- Measures: habitat derivation rules (water words → Binnengewässer, soil pits → Boden);
  anti-default check (Agrar only with farmland evidence).
- Expected: moderate C1 gain; supports the eco water mapping above.

### 5. taxa.scientific_name + german_name — wrong taxa set, not spelling (C2b, 47/45 never right)

- sci_name: FP-only 220 (model invents, often just "-"), different-names 116, "-" instead of
  real names 106, FN-only 56, near-miss (epithet/spp.) only 2 of 513 mismatches.
- Hypothesis: the model fails at *which* organisms (85% of cases), not at how names are spelled (1%).
- Measures: (a) evidence-bound quotes (quote must contain the taxon name);
  (b) keep species-list sections instead of cutting them (taxa-aware budget or dedicated taxa pass);
  (c) collective "-" decision as a second question, mirroring Minh's level call;
  (d) accept English source spans as evidence for translated German names.
- Expected: largest C1+C2b/Micro lever; near-miss normalization is already done and exhausted.

### 6. ecosystem_service category + term — over-prediction on empty gold (not C1, 5–6 never right)

- Category: FP 116 / FN 20. Top FP: Regulierungsleistungen (86×). Term: FP 115 / FN 20.
- Hypothesis: any regulating function mentioned becomes a predicted service; gold stays empty
  unless the study is *about* the service.
- Measures: evidence gates already exist (keep); no new runs — lowest priority, not a ranking field.

## Queued OFAT runs (cheapest first, each on both providers)

1. Isolated `--clean-taxa` (queued) — prices the taxa-cleanup lever.
2. Minh level prompt vs. folded single-prompt level rules (Priority 1).
3. Collective-group rule for `species_group` + collective-taxa second question (Priorities 2+5).
4. Water mapping + eco-type rule (Priorities 3+4).
5. Taxa-aware budget / dedicated taxa pass (Priority 5).
6. Translated names with English evidence (Priority 5, german_name).
