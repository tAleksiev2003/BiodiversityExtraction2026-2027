# Routing Plan — cheap pre-filter parser selection

Goal: spend as little compute as possible per paper, save resources, and still parse
every paper well enough. Calibrate on the 100 Dev papers, verify later on the 908 train papers.

## Principle

Run the cheapest parser that is good enough for that paper. Decide *before* converting,
from PyMuPDF signals that cost <0.1 s (no OCR, no model, no second pass).

## Signals (already computable per paper)

- Characters per page (text layer present at all?)
- Characters total and per page
- Headings count (structure GROBID would find)
- Images per page (scan detection)
- Page count and file size
- Umlaut ratio (German vs. English heuristic)

## Calibration on Dev-100 (oracle agreement)

- Per paper, the oracle is the best of the 4 parser baselines (going to 5 with full Docling).
- Only 39–42 papers have a unique best parser; on average 2.1–2.3 parsers tie per paper.
- A rule counts as right when its choice sits inside the best set (never exact-match).
- Pick the sparsest rule that loses nothing statistically: bootstrap CI lower bound first,
  greedy feature tree second, never fitted thresholds without reporting the gap.

## Verification on Train-908 (after the Gruenau batch finishes)

- The same 4 (5) parser sets for all 908 train papers, plus cheap signals for each.
- Check that the Dev-calibrated rules still route train papers into the best set ≥ ~60%.
- Deviations expected: papers are older (1970s scans), more reports, more German share.
- If a rule breaks on train (>2 points C1-equivalent loss), tighten it; otherwise lock it in.

## Target form (hypothesis bands, for later fitting)

- No text layer (scan) → GROBID skipped; Marker first, full Docling only for tables or quality.
- Text layer + few headings + low characters/page (reports, brochures) → Marker.
- Text layer + many headings + structured (journal papers) → GROBID.
- Very short and clean → PyMuPDF only; structure parsers add nothing at 2–6 pages.
- Very long + many headings → GROBID sections complet; very long + few headings → chunked conversion.

## Costs avoided if this works

- Full Docling on every paper: several days of GPU for 908, plus OCR-time waste on text papers.
- GROBID on scans: guaranteed empty output (blind) per paper.
- Marker on already-clean journal text: layout work with no gain.

## Open items until training can start

- Fair full-Docling rerun first (5th parser in the oracle).
- Signal upkeep: re-mine the cheap signals whenever caches change (seconds of CPU, no LLM).
- Fine-tuning / upper-bound check: greedy feature tree vs. sparse rules (overfit watch).
