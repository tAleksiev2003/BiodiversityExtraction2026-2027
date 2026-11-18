# Dev-100 Gold Label & PDF Quality Audit

Audit across all **100 validated Dev papers** (`dev_100_ids.txt`): **33 flagged** (listed first) and **67 clean** (listed below).

## Key Takeaways: Where the Biggest Problems Are

1. **Missing or inconsistent `taxa.scientific_name` (20 papers — `taxa_missing_sci`):**  
   The most frequent schema/gold inconsistency: many gold `taxa` entries omit `scientific_name` completely (neither a Latin name nor the expected `"-"` placeholder for collective terms), or mix German/irregular strings (`"Bäume"`, `"Fucus spp"` without a dot).
2. **Exact-match traps on species names (`taxa_epithet`, `taxa_comma_de`):**  
   Some gold records include full botanical author epithets & years (e.g. `"Adonis aestivalis L."`, `"Bromus grossus A. P. de Candolle 1805"` in `6792YBL7`) or comma-joined German names in a single string (`"Acker-Hasenohr, Rundblättriges Hasenohr"`). Because evaluation uses exact set matching, clean predictions like `"Adonis aestivalis"` suffer both a False Positive and a False Negative simultaneously.
3. **"All-Labels" on breadth documents (7 papers — `breadth_doc`):**  
   Policy strategies, species lists, and textbooks (e.g. `46RX4AEN`, `4YXRYRJC`, `3E8X2AK3`, `2P53UVJA`) are annotated with 5–6 habitats and/or 6–7 biodiversity levels. Standard empirical-focus extractors cannot predict all global categories on policy texts, creating a natural score ceiling (~7–10% of Dev).
4. **Unwritten annotator conventions & sparse/missing labels (`convention_floodplain`, `missing_label`, `low_ocr_text`):**  
   Floodplain forest studies (`3LGPK6BL`, `6BKNRKUA`) are systematically mapped to `II / Fließende Gewässer` rather than forest; `7AYVH7FI` (*Blue Carbon*) lacks `ecosystem_service` despite studying carbon sequestration; and 2 PDFs (`2P53UVJA` book OCR, `V2BVMBPF` with 0 extracted chars) have `< 500` extractable characters.

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

## Flagged Papers (33 / 100)

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

## Clean Papers (67 / 100)

| ID | Title | Counts |
| :--- | :--- | :--- |
| `25RIYD2C` | Grünes Band Deutschland: Chronik | `H:1 L:0 E:0 T:0 S:0` |
| `29Q84X7V` | What we can learn from the german implementat... | `H:1 L:0 E:0 T:0 S:0` |
| `2E9XWUUE` | Neobiota der deutschen Nord- und Ostseeküste:... | `H:1 L:0 E:0 T:5 S:0` |
| `2EUNPHDZ` | Analysis and assessment of cumulative effects... | `H:1 L:0 E:0 T:2 S:0` |
| `2RGQT8TU` | Safeguarding food security and reinforcing th... | `H:1 L:0 E:0 T:0 S:0` |
| `33MV2F7X` | Vorsorgender Bodenschutz in Deutschland | `H:1 L:0 E:0 T:0 S:0` |
| `38EBGVBF` | Art und Ausmaß der Inanspruchnahme landwirtsc... | `H:1 L:0 E:1 T:0 S:0` |
| `3TWLQG7F` | Establishment of blue mussel beds to enhance ... | `H:1 L:1 E:1 T:11 S:0` |
| `3WEEGFGW` | GAP-Strategieplan für die Bundesrepublik Deut... | `H:1 L:0 E:0 T:0 S:0` |
| `42ELXZ2H` | Vom Kreislauf zum Durchfluss: Österreichs Agr... | `H:1 L:0 E:0 T:0 S:0` |
| `42STMMQA` | Greening: a more complex income support schem... | `H:1 L:0 E:0 T:0 S:0` |
| `49Z7RPDP` | Vielfalt und Leistungen nutzen - Regenwürmer ... | `H:1 L:0 E:1 T:1 S:1` |
| `4BAX6GKK` | Biodiversity on farmland: CAP contribution ha... | `H:1 L:1 E:0 T:0 S:0` |
| `4FCS62W5` | Elevated CO2 stimulates marsh elevation gain,... | `H:1 L:0 E:0 T:3 S:0` |
| `4ZGN9AHA` | Das Ende der Agrarmodernisierung. Die Folgen ... | `H:1 L:0 E:0 T:0 S:0` |
| `57CBE476` | Studie zu Insekten in Gewässerrandstreifen | `H:1 L:2 E:1 T:1 S:0` |
| `57ZZ3GDF` | Investigations of the Potential Influence of ... | `H:1 L:0 E:2 T:1 S:0` |
| `5CZNRP7X` | Plastic ingestion by pelagic and demersal fis... | `H:1 L:0 E:2 T:5 S:0` |
| `5I8A3W7U` | Flurbereinigung- Best-Practice-Beispiele | `H:1 L:0 E:0 T:0 S:0` |
| `5SIYLM9W` | Die Wildbienen Deutschlands | `H:1 L:0 E:0 T:1 S:0` |
| `5TMUACTH` | Globale Landflächen und Biomasse - nachhaltig... | `H:1 L:0 E:0 T:0 S:0` |
| `5XHDHCIQ` | Chapter 8: Environmental and Health Effects o... | `H:1 L:0 E:0 T:0 S:0` |
| `6332SBW4` | Aktionsprogramm Quellen in Bayern Teil 2: Que... | `H:1 L:1 E:1 T:0 S:0` |
| `64C65W5I` | Preliminary study of oxidative stress biomark... | `H:1 L:0 E:1 T:1 S:0` |
| `6D23L7B5` | Kunststoffe in der Umwelt: Emissionen in land... | `H:2 L:0 E:0 T:0 S:0` |
| `6JGB9KZ2` | Bio-Lebensmittel: Marktentwicklung und freiwi... | `H:1 L:0 E:0 T:0 S:0` |
| `6RLMM8BQ` | Losers, winners, and opportunists: How grassl... | `H:1 L:1 E:0 T:1 S:0` |
| `7EUNQX3Z` | Research on Dutch Elm Disease in Europe | `H:1 L:0 E:0 T:1 S:0` |
| `7T8NZA5Q` | Environmental Suitability of Vibrio Infection... | `H:1 L:0 E:1 T:4 S:0` |
| `84P9CN4J` | Ausgestaltung der Ökoregelungen in Deutschlan... | `H:1 L:0 E:0 T:0 S:0` |
| `84QQ9F5S` | Natural Contaminants | `H:2 L:0 E:0 T:1 S:0` |
| `885FDL5Z` | State of knowledge of soil biodiversity – Sta... | `H:2 L:2 E:0 T:0 S:0` |
| `8GEYSUXR` | The marine macroalgae of Helgoland (North Sea... | `H:1 L:1 E:1 T:4 S:0` |
| `8HFL2MPL` | "Gute Küste Niedersachsen": Reallabore für ei... | `H:1 L:0 E:1 T:0 S:0` |
| `8PFJ2KHU` | Global Epidemiology, Ecology and Control of S... | `H:1 L:0 E:0 T:3 S:0` |
| `8Q86MQJE` | Thünen-Baseline 2020 - 2030: Agrarökonomische... | `H:1 L:0 E:0 T:0 S:0` |
| `8QKB2BJE` | Strategische Lösungsansätze und Best-Practice... | `H:2 L:1 E:0 T:0 S:0` |
| `8ZESW42A` | Vogelparadies Wattenmeer | `H:1 L:2 E:1 T:1 S:0` |
| `ASEHVK97` | Methane and sulfate profiles within the subsu... | `H:1 L:2 E:1 T:2 S:0` |
| `BBDCY7DW` | Impact on Population Health of Baltic Shippin... | `H:1 L:0 E:0 T:1 S:0` |
| `BE56TLWA` | Gardening as a mental health intervention: a ... | `H:1 L:0 E:0 T:0 S:0` |
| `BMTEN2FG` | Genetic diversity and disease control in rice | `H:1 L:1 E:0 T:1 S:1` |
| `DAI3VR9M` | Longer and more frequent marine heatwaves ove... | `H:1 L:0 E:0 T:0 S:0` |
| `DFWI5DCP` | Rewetting prolongs root growing season in min... | `H:1 L:2 E:1 T:3 S:1` |
| `G4VAFJZH` | Reduktion der Ammoniak-Emissionen auf der Weide | `H:1 L:0 E:0 T:1 S:0` |
| `G7H698R6` | Effects of forest management intensity on her... | `H:1 L:0 E:1 T:1 S:0` |
| `HNT6RCEA` | Regional Conditions and Land-Use Alter the Po... | `H:2 L:1 E:0 T:5 S:1` |
| `LY4P94Q4` | An ecotope map of the trilateral Wadden Sea | `H:1 L:1 E:3 T:0 S:0` |
| `N82RZ7WL` | Conspicuous decline of Fucus in Kiel Bay (Wes... | `H:1 L:1 E:1 T:1 S:0` |
| `NGFBPJCG` | Imprint of past and present environmental con... | `H:1 L:1 E:1 T:2 S:0` |
| `NZ7ETG2K` | Seasonal dynamics and changing sea level as d... | `H:1 L:1 E:1 T:6 S:0` |
| `QGF2WV3C` | The impact of Natura 2000 designation on agri... | `H:1 L:0 E:0 T:0 S:0` |
| `QR5QGMTI` | Untersuchungen über den Durchzug der Wasservö... | `H:1 L:2 E:1 T:1 S:0` |
| `TAXBZ8G5` | Preliminary indications for diverging heat an... | `H:1 L:1 E:1 T:4 S:1` |
| `TTJ3S6P9` | Synthesising the diversity of European agri-f... | `H:1 L:0 E:0 T:0 S:0` |
| `UGNPXL8B` | Relationships between biodiversity and biolog... | `H:1 L:1 E:0 T:2 S:0` |
| `V477VCE6` | Population dynamics of pikeperch (Sander luci... | `H:1 L:1 E:1 T:1 S:0` |
| `V6SVVBUH` | Fauna der Nierdemoore | `H:1 L:1 E:1 T:17 S:0` |
| `VQT78VA5` | The soil macrofauna (Diplopoda, Isopoda, Lumb... | `H:2 L:1 E:1 T:5 S:0` |
| `WW9EZUVE` | Ecosystem services and disservices provided b... | `H:1 L:1 E:0 T:2 S:1` |
| `XG99Q5DJ` | Widespread deoxygenation of temperate lakes | `H:1 L:0 E:1 T:0 S:0` |
| `XMIP837H` | Invasion trajectory of Pacific oysters in the... | `H:1 L:0 E:1 T:2 S:0` |
| `XQ7DW4VS` | Zur aktuellen Verbreitung des Mausohrs, Myoti... | `H:1 L:1 E:2 T:1 S:0` |
| `XU5H8QJI` | Der Kalikokrebs Orconectes immunis am nördlic... | `H:1 L:0 E:1 T:1 S:0` |
| `Z6CEAMK4` | Effects of ecological flooding on the tempora... | `H:1 L:2 E:1 T:2 S:0` |
| `ZQ329JQ6` | The North Sea — A shelf sea in the Anthropocene | `H:1 L:0 E:0 T:0 S:0` |
| `ZS59ID9J` | Wieviel weniger darf´s denn sein? Düstere Zuk... | `H:1 L:0 E:1 T:10 S:1` |
