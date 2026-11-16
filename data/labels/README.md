# Generated label files

Both files are generated from the training references by the scripts in `src/scripts/`
(see README.md, Quick Start, step 4). They are not committed, because they are derived
from course data. Check the printed reports by hand after generating them.

| File | Content | Used for |
|---|---|---|
| `term_to_category.json` | term → category for `ecosystem_type` and `ecosystem_service` | prompt (valid pairs) and verbatim checker (check C2) |
| `variable_vocabulary.json` | values of `biodiversity_variable` in the training references, with counts | prompt: preferred German terms for the free-text field |
