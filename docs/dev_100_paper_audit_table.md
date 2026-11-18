# Dev-100 Gold Label & PDF Issues (Filtered Audit)

Out of **100 validated Dev papers** (`dev_100_ids.txt`), **67** show no anomalies and **33** exhibit gold-label, formatting, or PDF-text issues listed below.

## Issue Breakdown (Evaluable Summary)

| Issue Tag | Papers |
| :--- | :---: |
| `taxa_missing_sci` | 20 |
| `breadth_doc` | 7 |
| `empty_habitat` | 2 |
| `low_ocr_text` | 2 |
| `convention_floodplain` | 2 |
| `taxa_format` | 2 |
| `taxa_comma_de` | 2 |
| `term_mismatch` | 1 |
| `taxa_epithet` | 1 |
| `missing_label` | 1 |

## Flagged Papers Only (33 / 100)

| ID | Issue Tag(s) | Title | Counts | Details |
| :--- | :--- | :--- | :--- | :--- |
| `25ABQZIH` | `empty_habitat` | Informationspapier des Greifswald Moor Centru... | `H:0 L:0 E:2 T:0 S:0` | habitat is None/empty |
| `2FDHAHAR` | `taxa_missing_sci` | Climate change and eutrophication induced shi... | `H:1 L:2 E:1 T:10 S:0` | Missing scientific_name |
| `2NJD9Y9T` | `taxa_missing_sci` | Ocean acidification drives global reshuffling... | `H:1 L:2 E:0 T:5 S:0` | Missing scientific_name |
| `2P53UVJA` | `breadth_doc, low_ocr_text` | Invasion ecology | `H:6 L:6 E:0 T:0 S:0` | All 6 habitats + 6 levels; broken book OCR |
| `2RXMDX8I` | `term_mismatch` | Veränderungen der hessischen Grünlandvegetati... | `H:1 L:0 E:1 T:0 S:0` | Grassland study labeled "Äcker und Ackerbrachen" |
| `3439JHQD` | `breadth_doc` | Citizen Science in der Naturschutzarbeit: Nat... | `H:5 L:0 E:0 T:0 S:0` | 5 habitats (Citizen Science apps review) |
| `3E8X2AK3` | `breadth_doc` | Die invasiven gebietsfremden Arten der Unions... | `H:6 L:0 E:0 T:0 S:0` | All 6 habitats (EU Invasive Species List) |
| `3LGPK6BL` | `convention_floodplain, taxa_missing_sci` | Untersuchungen zur historischen Entwicklung u... | `H:2 L:2 E:1 T:1 S:0` | Floodplain forest labeled II / "Fließende Gewässer"; Missing scientific_name |
| `3RSTTAEZ` | `taxa_missing_sci` | Less Nutrients but More Phytoplankton: Long-T... | `H:1 L:0 E:1 T:1 S:0` | Taxa missing scientific_name |
| `3Z5BFIBL` | `taxa_missing_sci` | Erfassung und Bewertung ökologisch wertvoller... | `H:1 L:1 E:0 T:11 S:0` | Missing scientific_name |
| `46RX4AEN` | `breadth_doc` | The cost of policy inaction, the case of not ... | `H:6 L:7 E:0 T:0 S:0` | All 6 habitats + all 7 levels (Cost of Policy Inaction) |
| `4JF8F3WL` | `breadth_doc` | Säugetiere (Mammalia) | `H:5 L:0 E:0 T:1 S:0` | 5 habitats (Red List Mammals) |
| `4YXRYRJC` | `breadth_doc` | Deutsche Nachhaltigkeitsstrategie. Weiterentw... | `H:6 L:0 E:0 T:0 S:0` | All 6 habitats (Sustainability Strategy) |
| `4Z67G9T5` | `taxa_missing_sci` | Zustand der deutschen Ostseegewässer 2018 | `H:1 L:1 E:0 T:7 S:0` | Missing scientific_name |
| `5LC5W7KF` | `taxa_format` | Der Wald in Deutschland - Ausgewählte Ergebni... | `H:1 L:1 E:0 T:1 S:0` | scientific_name = "Bäume" (German word in Latin field) |
| `64BS3HZ3` | `taxa_missing_sci` | Gewässerschutz und Luftschadstoffe – 30 Jahre... | `H:1 L:0 E:1 T:1 S:0` | Missing scientific_name |
| `6792YBL7` | `taxa_epithet, taxa_comma_de` | Von Venuskamm, Finkensame und Hasenohr - Vom ... | `H:1 L:0 E:0 T:16 S:0` | 16 species with author epithets ("L.", "1805") + comma-joined German names |
| `6BKNRKUA` | `convention_floodplain, taxa_missing_sci` | Untersuchung der xylobionten Käferfauna im NS... | `H:1 L:1 E:1 T:1 S:0` | Saproxylic beetles -> II/Fließende Gewässer; "-" instead of Coleoptera |
| `6CMGEZPD` | `taxa_missing_sci` | Die Wasserrahmenrichtlinie – Gewässer in Deut... | `H:2 L:0 E:3 T:4 S:0` | Missing scientific_name |
| `6DCMK7GA` | `taxa_missing_sci` | Wiederansiedlung seltener und gefährdeter Ack... | `H:1 L:0 E:0 T:1 S:0` | Taxa missing scientific_name |
| `6G2THNDX` | `taxa_missing_sci` | Öekosystemforschung: Der Rhein und seine Auen | `H:1 L:2 E:1 T:8 S:1` | Missing scientific_name |
| `7AYVH7FI` | `missing_label, taxa_missing_sci` | The blue carbon wealth of nations | `H:1 L:0 E:0 T:3 S:0` | Blue Carbon (CO2 sequestration), but ecosystem_service empty; Missing scientific_name |
| `7DSY6RMR` | `taxa_format, taxa_missing_sci` | Eutrophication in the Baltic Sea – An integra... | `H:1 L:0 E:1 T:7 S:0` | "Fucus spp" missing period; Missing scientific_name |
| `8UVBVFHS` | `breadth_doc` | Ecology of Central European Forests: Vegetati... | `H:4 L:2 E:0 T:0 S:0` | 4/6 habitats, 2/7 bio_levels |
| `F582S9H8` | `taxa_missing_sci, taxa_comma_de` | Managing open habitats by wild ungulate brows... | `H:1 L:2 E:0 T:1 S:0` | Missing scientific_name; Comma-combined German taxon name |
| `FG3YVDCZ` | `taxa_missing_sci` | Unbalanced reduction of nutrient loads has cr... | `H:1 L:0 E:1 T:7 S:0` | Missing scientific_name |
| `KJ7JVAEW` | `taxa_missing_sci` | The drivers of biogeochemistry in beach ecosy... | `H:1 L:2 E:1 T:5 S:0` | Missing scientific_name |
| `KXP8FP94` | `taxa_missing_sci` | How will Ocean Acidification Affect Baltic Se... | `H:1 L:0 E:1 T:16 S:0` | Missing scientific_name |
| `PBV2BUX6` | `empty_habitat` | Drought, windthrow and forest operations stro... | `H:0 L:0 E:0 T:0 S:0` | habitat is None/empty |
| `PMJGMLUG` | `taxa_missing_sci` | Global warming and plant–pollinator mismatches | `H:2 L:0 E:0 T:2 S:0` | Missing scientific_name |
| `PW2PNF55` | `taxa_missing_sci` | No increase in marine microplastic concentrat... | `H:1 L:0 E:1 T:3 S:0` | Missing scientific_name |
| `SVXF36H4` | `taxa_missing_sci` | Langjährige Bestandsentwicklung häufiger Brut... | `H:1 L:1 E:0 T:1 S:0` | Missing scientific_name |
| `V2BVMBPF` | `low_ocr_text` | Zum Rückgang von Naß- und Feuchtbiotopen im E... | `H:1 L:2 E:1 T:1 S:0` | PDF text < 500 chars (0) |
