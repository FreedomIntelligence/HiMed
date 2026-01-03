# Data Generation (HiMed-Trad)

This directory implements the end-to-end construction of culture-grounded Hindi reasoning data from noisy, real-world scans.

```text
Data_code/02_data_generation/
├── 01_preprocessing/                    # PDF/MMD → passages (clean/cluster/combine/pick/calibrate/label)
├── 02_instance_generation_scoring/      # passages → training instances + optional judge scoring
└── README.md                            # (this file)
```

---

## 1) Passage Preparation & Cleaning (`01_preprocessing/`)

### Inputs
- `pdf/`: raw scanned PDFs, named like `001.pdf`, `002.pdf`, ...
- `mmd/`: DeepSeek-OCR outputs, named like `001.mmd`, `002.mmd`, ...
- (auto-generated) `pictures/`: page images aligned to PDF pages (e.g., `pictures/001/1.png`)

### Step-by-step scripts (recommended order)

| Step | Script | Input | Output | What it does |
|---|---|---|---|---|
| 0 | DeepSeek-OCR | `pdf/*.pdf` | `mmd/*.mmd` | OCR each PDF (kept in `mmd/`). |
| 1 | `split.py` | `pdf/*.pdf` | `pictures/<raw_id>/<page>.png` | Extract page images for later calibration / grounding. |
| 2 | `transform.py` | `mmd/*.mmd` | `transformed_json/*.json` | Convert OCR outputs into JSON entries. |
| 3 | `clean.py` | `transformed_json/*.json` | `cleaned_json/*.clean.json` | Normalize & clean OCR text (remove noise / broken fragments). |
| 4 | `cluster.py` | `cleaned_json/*.clean.json` | `clustered_json/*.cluster.json` | Merge adjacent fragments into coherent passages by “knowledge density”. |
| 5 | `filter1.py` | `clustered_json/*.cluster.json` | `analyzed_json/*.analyze.json` | Produce four decisions per passage (the “four-flag” analysis). |
| 6 | `combine.py` | `analyzed_json/*.analyze.json` | `combined_json/*.combine.json` | Merge passages based on the four-flag analysis results. |
| 7 | `pick.py` | `combined_json/*.combine.json` | `picked_json/*.pick.json` + `abandoned_json/*.abandon.json` | Select target four-flag pattern (e.g., **TTFF**) and discard the rest. |
| 8 | `calibrate.py` | `picked_json/*.pick.json` + `pictures/` | `calibrated_json/*.calibrate.json` | Repair OCR errors with page-image grounding. |
| 9 | `filter2.py` | `calibrated_json/*.calibrate.json` | `labeled_json/*.label.json` | Label into quality buckets (Good / Middle / Bad). |
| 10 | `distribute.py` | `labeled_json/*.label.json` | `good/*.json`, `middle/*.json`, `bad/*.json` | Split labeled outputs into three files for easy consumption. |

### Notes
- We treat all items derived from the same **source passage** as an **indivisible unit** for splitting (strict passage-level split) to avoid leakage between corpus and benchmark.
- Ensure `metadata` preserves alignment between `raw_id` ↔ PDF, and stores the merged `page_range` list for calibration/traceability.

---

## 2) Training Instance Generation & Scoring (`02_instance_generation_scoring/`)

This stage converts **Good passages** into instruction-style reasoning instances (Q/A/CoT), and optionally runs an audit model (“LLM-as-a-judge”) to score quality.

### Pipeline overview
1) **Prepare grounded passages**  
   Each entry contains `text` plus `metadata` that tracks `raw_id`, `text_id`, and merged `page_range`.

2) **Tagging: subject + question type**  
   Ask an LLM to predict up to **5 subjects** and up to **3 question types** (e.g., `MCQ / QA / Dialogue`).  
   If nothing fits, fall back to a general category such as `medical knowledge`.

3) **Expand multi-label entries into single-label instances**  
   Split one entry with `(subject_list, type_list)` into multiple entries, each with exactly **one** `subject` and **one** `type`.  
   Assign `entry_id` like `01, 02, 03, ...`.

4) **Add generation controls**  
   Add deterministic `id`, plus `few_shot` and `question_template` fields to control instance generation.

5) **Generate Q/A/CoT (grounded-only)**  
   Generate **question**, **answer**, and **cot** strictly based on the given `text`.  
   If the model cannot generate grounded content, it must output `<FAIL>`.  
   After generation, remove `few_shot` and keep the rest.

6) **LLM-as-a-judge scoring (optional)**  
   Score each instance with 0–1 values such as:
   - `grounded_in_context`
   - `medical_correctness`
   - `reasoning_clarity`
   - `language_quality`  
   Apply thresholds later to filter training instances.

