# Dev-100 Gold Label & PDF Quality Audit Table

Combines manual gold-label auditing (first 50 papers) with automated schema/format/PDF-text consistency checks across all **100 validated Dev papers** (`dev_100_ids.txt`).

**Summary:** `67` clean (`✅`) · `31` labeling/format/OCR issue (`⚠️`) · `2` special/sparse case (`🔎`)  
**Counts:** `H` = habitat, `L` = biodiversity_level, `E` = ecosystem_type, `T` = taxa, `S` = ecosystem_service

| # | ID | Status | Title | Gold Counts | Findings / Labeling Notes |
| :---: | :--- | :---: | :--- | :--- | :--- |
| 1 | `25ABQZIH` | 🔎 | Informationspapier des Greifswald Moor Centrum zu... | `H:0 L:0 E:2 T:0 S:0` | habitat empty (None) |
| 2 | `25RIYD2C` | ✅ | Grünes Band Deutschland: Chronik | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 3 | `29Q84X7V` | ✅ | What we can learn from the german implementation ... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 4 | `2E9XWUUE` | ✅ | Neobiota der deutschen Nord- und Ostseeküste: Ein... | `H:1 L:0 E:0 T:5 S:0` | No anomalies detected |
| 5 | `2EUNPHDZ` | ✅ | Analysis and assessment of cumulative effects of ... | `H:1 L:0 E:0 T:2 S:0` | No anomalies detected |
| 6 | `2FDHAHAR` | ⚠️ | Climate change and eutrophication induced shifts ... | `H:1 L:2 E:1 T:10 S:0` | Large taxa list (10 entries); Taxa missing scientific_name |
| 7 | `2NJD9Y9T` | ⚠️ | Ocean acidification drives global reshuffling of ... | `H:1 L:2 E:0 T:5 S:0` | Taxa missing scientific_name |
| 8 | `2P53UVJA` | ⚠️ | Invasion ecology | `H:6 L:6 E:0 T:0 S:0` | Breadth doc + poor book OCR: 6 habitats + 6 levels (Invasion Ecology) |
| 9 | `2RGQT8TU` | ✅ | Safeguarding food security and reinforcing the re... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 10 | `2RXMDX8I` | ⚠️ | Veränderungen der hessischen Grünlandvegetation s... | `H:1 L:0 E:1 T:0 S:0` | Term mismatch: grassland study labeled "Äcker und Ackerbrachen" |
| 11 | `33MV2F7X` | ✅ | Vorsorgender Bodenschutz in Deutschland | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 12 | `3439JHQD` | ⚠️ | Citizen Science in der Naturschutzarbeit: Natur-A... | `H:5 L:0 E:0 T:0 S:0` | Breadth doc: 5 habitats (Citizen Science apps review) |
| 13 | `38EBGVBF` | ✅ | Art und Ausmaß der Inanspruchnahme landwirtschaft... | `H:1 L:0 E:1 T:0 S:0` | No anomalies detected |
| 14 | `3E8X2AK3` | ⚠️ | Die invasiven gebietsfremden Arten der Unionslist... | `H:6 L:0 E:0 T:0 S:0` | Breadth doc: all 6 habitats (EU Invasive Alien Species list) |
| 15 | `3LGPK6BL` | ⚠️ | Untersuchungen zur historischen Entwicklung und V... | `H:2 L:2 E:1 T:1 S:0` | Floodplain convention: floodplain forest labeled II / "Fließende Gewässer" |
| 16 | `3RSTTAEZ` | ⚠️ | Less Nutrients but More Phytoplankton: Long-Term ... | `H:1 L:0 E:1 T:1 S:0` | Taxa convention: taxa missing scientific_name |
| 17 | `3TWLQG7F` | ✅ | Establishment of blue mussel beds to enhance fish... | `H:1 L:1 E:1 T:11 S:0` | Large taxa list (11 entries) |
| 18 | `3WEEGFGW` | ✅ | GAP-Strategieplan für die Bundesrepublik Deutschland | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 19 | `3Z5BFIBL` | ⚠️ | Erfassung und Bewertung ökologisch wertvoller Leb... | `H:1 L:1 E:0 T:11 S:0` | Large taxa list (11 entries); Taxa missing scientific_name |
| 20 | `42ELXZ2H` | ✅ | Vom Kreislauf zum Durchfluss: Österreichs Agrarmo... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 21 | `42STMMQA` | ✅ | Greening: a more complex income support scheme, n... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 22 | `46RX4AEN` | ⚠️ | The cost of policy inaction, the case of not meet... | `H:6 L:7 E:0 T:0 S:0` | Breadth doc: all 6 habitats + all 7 levels (Cost of Policy Inaction) |
| 23 | `49Z7RPDP` | ✅ | Vielfalt und Leistungen nutzen - Regenwürmer in b... | `H:1 L:0 E:1 T:1 S:1` | Has ecosystem_service (1) |
| 24 | `4BAX6GKK` | ✅ | Biodiversity on farmland: CAP contribution has no... | `H:1 L:1 E:0 T:0 S:0` | No anomalies detected |
| 25 | `4FCS62W5` | ✅ | Elevated CO2 stimulates marsh elevation gain, cou... | `H:1 L:0 E:0 T:3 S:0` | No anomalies detected |
| 26 | `4JF8F3WL` | ⚠️ | Säugetiere (Mammalia) | `H:5 L:0 E:0 T:1 S:0` | Species list: 5 habitats (Red List Mammals) |
| 27 | `4YXRYRJC` | ⚠️ | Deutsche Nachhaltigkeitsstrategie. Weiterentwickl... | `H:6 L:0 E:0 T:0 S:0` | Breadth doc: all 6 habitats (German Sustainability Strategy) |
| 28 | `4Z67G9T5` | ⚠️ | Zustand der deutschen Ostseegewässer 2018 | `H:1 L:1 E:0 T:7 S:0` | Taxa missing scientific_name |
| 29 | `4ZGN9AHA` | ✅ | Das Ende der Agrarmodernisierung. Die Folgen der ... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 30 | `57CBE476` | ✅ | Studie zu Insekten in Gewässerrandstreifen | `H:1 L:2 E:1 T:1 S:0` | No anomalies detected |
| 31 | `57ZZ3GDF` | ✅ | Investigations of the Potential Influence of Envi... | `H:1 L:0 E:2 T:1 S:0` | No anomalies detected |
| 32 | `5CZNRP7X` | ✅ | Plastic ingestion by pelagic and demersal fish fr... | `H:1 L:0 E:2 T:5 S:0` | No anomalies detected |
| 33 | `5I8A3W7U` | ✅ | Flurbereinigung- Best-Practice-Beispiele | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 34 | `5LC5W7KF` | ⚠️ | Der Wald in Deutschland - Ausgewählte Ergebnisse ... | `H:1 L:1 E:0 T:1 S:0` | Taxa convention: scientific_name = "Bäume" (German word in Latin field) |
| 35 | `5SIYLM9W` | ✅ | Die Wildbienen Deutschlands | `H:1 L:0 E:0 T:1 S:0` | No anomalies detected |
| 36 | `5TMUACTH` | ✅ | Globale Landflächen und Biomasse - nachhaltig und... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 37 | `5XHDHCIQ` | ✅ | Chapter 8: Environmental and Health Effects of Pe... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 38 | `6332SBW4` | ✅ | Aktionsprogramm Quellen in Bayern Teil 2: Queller... | `H:1 L:1 E:1 T:0 S:0` | No anomalies detected |
| 39 | `64BS3HZ3` | ⚠️ | Gewässerschutz und Luftschadstoffe – 30 Jahre Mon... | `H:1 L:0 E:1 T:1 S:0` | Taxa missing scientific_name |
| 40 | `64C65W5I` | ✅ | Preliminary study of oxidative stress biomarkers ... | `H:1 L:0 E:1 T:1 S:0` | No anomalies detected |
| 41 | `6792YBL7` | ⚠️ | Von Venuskamm, Finkensame und Hasenohr - Vom Auss... | `H:1 L:0 E:0 T:16 S:0` | Exact-match trap: 16 species with author epithets ("L.", "1805") + comma-joined German names |
| 42 | `6BKNRKUA` | ⚠️ | Untersuchung der xylobionten Käferfauna im NSG Kü... | `H:1 L:1 E:1 T:1 S:0` | Floodplain convention + taxa: saproxylic beetles labeled II/Fließende Gewässer; "-" instead of Coleoptera |
| 43 | `6CMGEZPD` | ⚠️ | Die Wasserrahmenrichtlinie – Gewässer in Deutschl... | `H:2 L:0 E:3 T:4 S:0` | Taxa missing scientific_name |
| 44 | `6D23L7B5` | ✅ | Kunststoffe in der Umwelt: Emissionen in landwirt... | `H:2 L:0 E:0 T:0 S:0` | No anomalies detected |
| 45 | `6DCMK7GA` | ⚠️ | Wiederansiedlung seltener und gefährdeter Ackerwi... | `H:1 L:0 E:0 T:1 S:0` | Taxa convention: taxa missing scientific_name |
| 46 | `6G2THNDX` | ⚠️ | Öekosystemforschung: Der Rhein und seine Auen | `H:1 L:2 E:1 T:8 S:1` | Taxa missing scientific_name; Has ecosystem_service (1) |
| 47 | `6JGB9KZ2` | ✅ | Bio-Lebensmittel: Marktentwicklung und freiwillig... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 48 | `6RLMM8BQ` | ✅ | Losers, winners, and opportunists: How grassland ... | `H:1 L:1 E:0 T:1 S:0` | No anomalies detected |
| 49 | `7AYVH7FI` | ⚠️ | The blue carbon wealth of nations | `H:1 L:0 E:0 T:3 S:0` | Missing label: Blue Carbon (CO2 sequestration), but no ecosystem_service |
| 50 | `7DSY6RMR` | ⚠️ | Eutrophication in the Baltic Sea – An integrated ... | `H:1 L:0 E:1 T:7 S:0` | Format noise: "Fucus spp" missing period |
| 51 | `7EUNQX3Z` | ✅ | Research on Dutch Elm Disease in Europe | `H:1 L:0 E:0 T:1 S:0` | No anomalies detected |
| 52 | `7T8NZA5Q` | ✅ | Environmental Suitability of Vibrio Infections in... | `H:1 L:0 E:1 T:4 S:0` | No anomalies detected |
| 53 | `84P9CN4J` | ✅ | Ausgestaltung der Ökoregelungen in Deutschland – ... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 54 | `84QQ9F5S` | ✅ | Natural Contaminants | `H:2 L:0 E:0 T:1 S:0` | No anomalies detected |
| 55 | `885FDL5Z` | ✅ | State of knowledge of soil biodiversity – Status,... | `H:2 L:2 E:0 T:0 S:0` | No anomalies detected |
| 56 | `8GEYSUXR` | ✅ | The marine macroalgae of Helgoland (North Sea): a... | `H:1 L:1 E:1 T:4 S:0` | No anomalies detected |
| 57 | `8HFL2MPL` | ✅ | "Gute Küste Niedersachsen": Reallabore für einen ... | `H:1 L:0 E:1 T:0 S:0` | No anomalies detected |
| 58 | `8PFJ2KHU` | ✅ | Global Epidemiology, Ecology and Control of Soil-... | `H:1 L:0 E:0 T:3 S:0` | No anomalies detected |
| 59 | `8Q86MQJE` | ✅ | Thünen-Baseline 2020 - 2030: Agrarökonomische Pro... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 60 | `8QKB2BJE` | ✅ | Strategische Lösungsansätze und Best-Practice-Bei... | `H:2 L:1 E:0 T:0 S:0` | No anomalies detected |
| 61 | `8UVBVFHS` | ⚠️ | Ecology of Central European Forests: Vegetation E... | `H:4 L:2 E:0 T:0 S:0` | Many habitats (4/6) |
| 62 | `8ZESW42A` | ✅ | Vogelparadies Wattenmeer | `H:1 L:2 E:1 T:1 S:0` | No anomalies detected |
| 63 | `ASEHVK97` | ✅ | Methane and sulfate profiles within the subsurfac... | `H:1 L:2 E:1 T:2 S:0` | No anomalies detected |
| 64 | `BBDCY7DW` | ✅ | Impact on Population Health of Baltic Shipping Em... | `H:1 L:0 E:0 T:1 S:0` | No anomalies detected |
| 65 | `BE56TLWA` | ✅ | Gardening as a mental health intervention: a review | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 66 | `BMTEN2FG` | ✅ | Genetic diversity and disease control in rice | `H:1 L:1 E:0 T:1 S:1` | Has ecosystem_service (1) |
| 67 | `DAI3VR9M` | ✅ | Longer and more frequent marine heatwaves over th... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 68 | `DFWI5DCP` | ✅ | Rewetting prolongs root growing season in minerot... | `H:1 L:2 E:1 T:3 S:1` | Has ecosystem_service (1) |
| 69 | `F582S9H8` | ⚠️ | Managing open habitats by wild ungulate browsing ... | `H:1 L:2 E:0 T:1 S:0` | Taxa missing scientific_name; Comma-combined German taxon name |
| 70 | `FG3YVDCZ` | ⚠️ | Unbalanced reduction of nutrient loads has create... | `H:1 L:0 E:1 T:7 S:0` | Taxa missing scientific_name |
| 71 | `G4VAFJZH` | ✅ | Reduktion der Ammoniak-Emissionen auf der Weide | `H:1 L:0 E:0 T:1 S:0` | No anomalies detected |
| 72 | `G7H698R6` | ✅ | Effects of forest management intensity on herb la... | `H:1 L:0 E:1 T:1 S:0` | No anomalies detected |
| 73 | `HNT6RCEA` | ✅ | Regional Conditions and Land-Use Alter the Potent... | `H:2 L:1 E:0 T:5 S:1` | Has ecosystem_service (1) |
| 74 | `KJ7JVAEW` | ⚠️ | The drivers of biogeochemistry in beach ecosystem... | `H:1 L:2 E:1 T:5 S:0` | Taxa missing scientific_name |
| 75 | `KXP8FP94` | ⚠️ | How will Ocean Acidification Affect Baltic Sea Ec... | `H:1 L:0 E:1 T:16 S:0` | Large taxa list (16 entries); Taxa missing scientific_name |
| 76 | `LY4P94Q4` | ✅ | An ecotope map of the trilateral Wadden Sea | `H:1 L:1 E:3 T:0 S:0` | No anomalies detected |
| 77 | `N82RZ7WL` | ✅ | Conspicuous decline of Fucus in Kiel Bay (Western... | `H:1 L:1 E:1 T:1 S:0` | No anomalies detected |
| 78 | `NGFBPJCG` | ✅ | Imprint of past and present environmental conditi... | `H:1 L:1 E:1 T:2 S:0` | No anomalies detected |
| 79 | `NZ7ETG2K` | ✅ | Seasonal dynamics and changing sea level as deter... | `H:1 L:1 E:1 T:6 S:0` | No anomalies detected |
| 80 | `PBV2BUX6` | 🔎 | Drought, windthrow and forest operations strongly... | `H:0 L:0 E:0 T:0 S:0` | habitat empty (None) |
| 81 | `PMJGMLUG` | ⚠️ | Global warming and plant–pollinator mismatches | `H:2 L:0 E:0 T:2 S:0` | Taxa missing scientific_name |
| 82 | `PW2PNF55` | ⚠️ | No increase in marine microplastic concentration ... | `H:1 L:0 E:1 T:3 S:0` | Taxa missing scientific_name |
| 83 | `QGF2WV3C` | ✅ | The impact of Natura 2000 designation on agricult... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 84 | `QR5QGMTI` | ✅ | Untersuchungen über den Durchzug der Wasservögel ... | `H:1 L:2 E:1 T:1 S:0` | No anomalies detected |
| 85 | `SVXF36H4` | ⚠️ | Langjährige Bestandsentwicklung häufiger Brutvoge... | `H:1 L:1 E:0 T:1 S:0` | Taxa missing scientific_name |
| 86 | `TAXBZ8G5` | ✅ | Preliminary indications for diverging heat and dr... | `H:1 L:1 E:1 T:4 S:1` | Has ecosystem_service (1) |
| 87 | `TTJ3S6P9` | ✅ | Synthesising the diversity of European agri-food ... | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 88 | `UGNPXL8B` | ✅ | Relationships between biodiversity and biological... | `H:1 L:1 E:0 T:2 S:0` | No anomalies detected |
| 89 | `V2BVMBPF` | ⚠️ | Zum Rückgang von Naß- und Feuchtbiotopen im Emstal | `H:1 L:2 E:1 T:1 S:0` | Very low extractable PDF text (0 chars) |
| 90 | `V477VCE6` | ✅ | Population dynamics of pikeperch (Sander lucioper... | `H:1 L:1 E:1 T:1 S:0` | No anomalies detected |
| 91 | `V6SVVBUH` | ✅ | Fauna der Nierdemoore | `H:1 L:1 E:1 T:17 S:0` | Large taxa list (17 entries) |
| 92 | `VQT78VA5` | ✅ | The soil macrofauna (Diplopoda, Isopoda, Lumbrici... | `H:2 L:1 E:1 T:5 S:0` | No anomalies detected |
| 93 | `WW9EZUVE` | ✅ | Ecosystem services and disservices provided by sm... | `H:1 L:1 E:0 T:2 S:1` | Has ecosystem_service (1) |
| 94 | `XG99Q5DJ` | ✅ | Widespread deoxygenation of temperate lakes | `H:1 L:0 E:1 T:0 S:0` | No anomalies detected |
| 95 | `XMIP837H` | ✅ | Invasion trajectory of Pacific oysters in the nor... | `H:1 L:0 E:1 T:2 S:0` | No anomalies detected |
| 96 | `XQ7DW4VS` | ✅ | Zur aktuellen Verbreitung des Mausohrs, Myotis my... | `H:1 L:1 E:2 T:1 S:0` | No anomalies detected |
| 97 | `XU5H8QJI` | ✅ | Der Kalikokrebs Orconectes immunis am nördlichen ... | `H:1 L:0 E:1 T:1 S:0` | No anomalies detected |
| 98 | `Z6CEAMK4` | ✅ | Effects of ecological flooding on the temporal an... | `H:1 L:2 E:1 T:2 S:0` | No anomalies detected |
| 99 | `ZQ329JQ6` | ✅ | The North Sea — A shelf sea in the Anthropocene | `H:1 L:0 E:0 T:0 S:0` | No anomalies detected |
| 100 | `ZS59ID9J` | ✅ | Wieviel weniger darf´s denn sein? Düstere Zukunft... | `H:1 L:0 E:1 T:10 S:1` | Large taxa list (10 entries); Has ecosystem_service (1) |
