# <PROJECT-NAME> (HiMed-8B): Hindi Medical Reasoning with Cross-Lingual Transfer + DSR-RL

<div align="center">
<h3><PROJECT-NAME> / HiMed-8B</h3>
</div>

<p align="center">
📃 <a href="<PAPER_URL>" target="_blank">Paper</a> |
🤗 <a href="<HF_MODEL_URL>" target="_blank">Model (HiMed-8B)</a> |
📚 <a href="<HF_DATASET_URL>" target="_blank">Data (HiMed)</a> |
💻 <a href="<GITHUB_URL>" target="_blank">Code</a>
</p>

<div align="center">
<img src="assets/<HERO_IMAGE>.png" width="90%" alt="<PROJECT-NAME> overview" />
</div>

---

## ⚡ Introduction

Medical large language models hold promise for reducing healthcare disparities, yet **Hindi remains severely underrepresented**. While medical LLMs excel in high-resource languages, their performance degrades sharply in Hindi, particularly on **Indian systems of medicine**. We therefore argue that robust cross-lingual medical transfer requires **Hindi reasoning**.

To this end, we propose a **three-stage training framework** comprising **language adaptation**, **reasoning cold-start**, and **Decaying Scaffolding Reward Reinforcement Learning (DSR-RL)**, which gradually shifts optimization from reasoning behavior guidance to task-optimal objectives. We further introduce **HiMed**, a comprehensive Hindi medical dataset and benchmark suite covering both **Western** and **Indian medicine**. Experiments based on **LLaMA-3.1-8B-Instruct** yield **HiMed-8B**, which consistently improves Hindi medical reasoning performance and substantially reduces the English–Hindi accuracy gap. Ablation studies further validate the contribution of each training stage and the reward design.

**This repository releases:**
- ✅ **Data/**: all datasets & benchmark files (or download pointers)
- ✅ **Training code/**: stage1/2/3 training + RL + evaluation scripts
- ✅ **Data Code/**: data construction / translation / filtering / dedup pipelines

---

## 🔥 Highlights
- **HiMed**: Hindi medical dataset + benchmark suite spanning **Western** + **Indian** medicine.
- **HiMed-8B**: Hindi medical reasoning model trained from **LLaMA-3.1-8B-Instruct**.
- **DSR-RL**: a decaying scaffolding reward that transitions from guided reasoning to task-optimal objectives.
- **Cross-lingual gains**: consistently narrows the English–Hindi performance gap (see paper for details).

---

## 🧭 Repository Structure

> If you see broken tree formatting on GitHub, ensure it is inside a fenced code block (as below).

```text
.
├── Data/                  # all released datasets & benchmark files (or download pointers)
├── Training code/         # stage1/2/3 training + RL + evaluation scripts
├── Data Code/             # data construction / translation / filtering / dedup pipelines
└── assets/                # figures, diagrams, project images
```

**Recommended reading order:** `Data/ → Training code/ → Data Code/`.

---

## 👨‍⚕️ Models

### Model Access
| Model | Backbone | Languages | Description | Link |
|------|----------|-----------|-------------|------|
| **HiMed-8B** | LLaMA-3.1-8B-Instruct | Hindi & English | Hindi medical reasoning model | [HF Link](<HF_MODEL_URL>) |

> If you provide multiple checkpoints (e.g., stage1/stage2/stage3), add rows here.

---

## 🚀 Quickstart

### 1) Installation
```bash
git clone <GITHUB_URL>
cd <REPO_NAME>

# (recommended) create env
conda create -n himed python=3.10 -y
conda activate himed

pip install -r requirements.txt
```

### 2) Quick Inference (Transformers)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "<HF_MODEL_ID_OR_LOCAL_PATH>"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "हिंदी में खांसी को कैसे रोकें? संक्षेप में बताएं।"}]
inputs = tokenizer(
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3) Deploy (vLLM / SGLang)

**vLLM**
```bash
vllm serve <HF_MODEL_ID_OR_LOCAL_PATH> \
  --tensor-parallel-size <TP> \
  --port <PORT>
```

**SGLang**
```bash
python -m sglang.launch_server \
  --model-path <HF_MODEL_ID_OR_LOCAL_PATH> \
  --port <PORT> \
  --mem-fraction-static 0.8
```

---

## 📚 Data (HiMed)

HiMed is a Hindi medical dataset and benchmark suite covering both Western medicine and Indian systems of medicine. It consists of two parts: **HiMed-Trad** (traditional Indian medicine) and **HiMed-West** (Western medicine under Hindi prompts). We enforce strict data separation between training corpora and evaluation benchmarks to prevent leakage (see paper for details).

### Data Files
We release five JSON files under `Data/`:

```text
Data/
├── HiMed-Trad_Bench.json
├── HiMed-Trad_Corpus_sample.json
├── HiMed-West_Bench.json
├── HiMed-West_Corpus_sample.json
└── HiMed-West_Exam.json
```

### Statistics
- **HiMed-Trad Bench**: 6,010
- **HiMed-West Bench**: 1,784
- **HiMed-West Exam**: 470
- **HiMed-Trad Corpus (full)**: 286,657
- **HiMed-West Corpus (full)**: 116,859

### Note on Corpus Release
The full training corpora are large. In this repository, we provide **500-sample subsets** for:
- `HiMed-Trad_Corpus_sample.json`
- `HiMed-West_Corpus_sample.json`

The complete versions of **HiMed-Trad Corpus** and **HiMed-West Corpus** will be released upon paper acceptance.


## 🚀 Training

### Stage 1: Language Adaptation (LA)

Fine-tune the base model (**LLaMA-3.1-8B-Instruct**) on an **8×H200** setup with Accelerate + DeepSpeed. We use bf16 and ZeRO stage-2; see `Train_code/configs/ds_config.yaml` for details.

```bash
accelerate launch \
  --config_file Train_code/configs/ds_config.yaml \
  --num_processes 8 \
  Train_code/LA.py \
  --model_path <BASE_MODEL_PATH_OR_HF_ID> \
  --data_path <STAGE1_DATA_PATH> \
  --output_dir <OUTPUT_DIR> \
  --max_seq_len 4096 \
  --train_bsz_per_gpu 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --n_epochs 3 \
  --gradient_checkpointing
```

Notes:
- You can set `--experiment_name` / logging backends (e.g., SwanLab/W&B) if needed; we omit them here for clarity.



### Stage 2: Reasoning Cold-Start (RC)

Fine-tune the Stage-1 checkpoint for Hindi medical reasoning on an **8×H200** setup with Accelerate + DeepSpeed (bf16, ZeRO-2). The distributed/ZeRO configuration is defined in `Train_code/configs/ds_config.yaml`.

```bash
accelerate launch \
  --config_file Train_code/configs/ds_config.yaml \
  --num_processes 8 \
  Train_code/RC.py \
  --model_path <PATH_TO_STAGE1_CKPT> \
  --data_path <STAGE2_DATA_PATH> \
  --output_dir <OUTPUT_DIR> \
  --best_ckpt_dir <BEST_CKPT_DIR> \
  --max_seq_len 4096 \
  --train_bsz_per_gpu 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --n_epochs 3 \
  --gradient_checkpointing
```

Optional:
- `--weight_decay` (default: 0.01)
- `--warmup_rates` (default: 0.03)
- `--ckpt_per_epoch` / `--log_steps_per_epoch` (checkpointing/logging frequency)


### Stage 3 — Decaying Scaffolding Reward Reinforcement Learning (DSR-RL)
Goal: RL with a **decaying scaffolding** reward that shifts emphasis from guided reasoning behavior → task-optimal objective.

```bash
cd "Training code"
accelerate launch --config_file ./configs/<CONFIG_STAGE3>.yaml \
  S3_DSR_RL.py \
  --model_name_or_path <CKPT_FROM_STAGE2> \
  --dataset_name <RL_DATASET> \
  --reward_model <REWARD_OR_VERIFIER_PATH> \
  --output_dir <OUTPUT_DIR_STAGE3> \
  --run_name <RUN_NAME_STAGE3> \
  --total_episodes <N> \
  --learning_rate <LR> \
  --kl_coef <KL>
```

### Notes (fill as needed)
- **Reward design:** describe your scaffold reward components (format/consistency/faithfulness/answer correctness, etc.).
- **Decay schedule:** specify how scaffold weight decays over steps/epochs.
- **Answer extraction:** how final answer is parsed for verifiable reward/metrics.

---


## 🧩 Data Pipeline (Data_code)

This section corresponds to **`Data_code/`**, which contains scripts for **(1) OCR**, **(2) HiMed data generation**, and **(3) translation**.

```text
Data_code/
├── 01_ocr/                              # DeepSeek-OCR (official codebase, unmodified)
├── 02_data_generation/
│   ├── 01_preprocessing/                # PDF/MMD → passages (clean/cluster/combine/pick/calibrate/label)
│   └── 02_sft_generation_scoring/       # passage → Q/A/CoT instances + LLM-as-a-judge scoring
└── 03_translation/                      # lexicon-guided translation scripts (HiMed-West)
```

---

### 1) OCR (DeepSeek-OCR)

We use the **official DeepSeek-OCR** codebase **without modifications**. Please follow the original instructions in `Data_code/01_ocr/`.

> Recommendation: keep DeepSeek-OCR as a git submodule or a pinned snapshot, and preserve its LICENSE/NOTICE.

---

### 2) Data Generation (HiMed-Trad passage pipeline + instance construction)

This part implements the end-to-end construction of **culture-grounded Hindi reasoning data** from noisy, real-world scans (PDF → OCR → passages → calibrated passages → labeled quality splits → training instances).

#### 2.1 Passage Preparation & Cleaning (`02_data_generation/01_preprocessing/`)

**Inputs**
- `pdf/`: raw scanned PDFs, named like `001.pdf`, `002.pdf`, ...
- `mmd/`: DeepSeek-OCR outputs, named like `001.mmd`, `002.mmd`, ...
- (auto-generated) `pictures/`: page images aligned to PDF pages (e.g., `pictures/001/1.png`)

**Step-by-step scripts (in order)**
| Step | Script | Input | Output | What it does |
|---|---|---|---|---|
| 0 | DeepSeek-OCR | `pdf/*.pdf` | `mmd/*.mmd` | OCR each PDF (kept in `mmd/`). |
| 1 | `split.py` | `pdf/*.pdf` (or OCR artifacts) | `pictures/<raw_id>/<page>.png` | Extract page images for later calibration / grounding. |
| 2 | `transform.py` | `mmd/*.mmd` | `transformed_json/*.json` | Convert OCR outputs into JSON entries. |
| 3 | `clean.py` | `transformed_json/*.json` | `cleaned_json/*.clean.json` | Normalize & clean OCR text (remove noise / broken fragments). |
| 4 | `cluster.py` | `cleaned_json/*.clean.json` | `clustered_json/*.cluster.json` | Merge adjacent fragments into coherent passages by “knowledge density” (threshold is configurable; default ~0.65). |
| 5 | `filter1.py` | `clustered_json/*.cluster.json` | `analyzed_json/*.analyze.json` | Produce **four decisions** per passage to decide whether it should be merged / retained (the “four-flag” analysis). |
| 6 | `combine.py` | `analyzed_json/*.analyze.json` | `combined_json/*.combine.json` | Merge passages based on the four-flag analysis results. |
| 7 | `pick.py` | `combined_json/*.combine.json` | `picked_json/*.pick.json` + `abandoned_json/*.abandon.json` | Select passages that match the target four-flag pattern (e.g., **TTFF**) and discard the rest. |
| 8 | `calibrate.py` | `picked_json/*.pick.json` + `pictures/` | `calibrated_json/*.calibrate.json` | Use an LLM to **repair OCR errors** with page-image grounding, yielding coherent, self-contained Hindi passages. |
| 9 | `filter2.py` | `calibrated_json/*.calibrate.json` | `labeled_json/*.label.json` | LLM-based labeling into quality buckets (**Good / Middle / Bad**) for downstream usage. |
| 10 | `distribute.py` | `labeled_json/*.label.json` | `good/*.json`, `middle/*.json`, `bad/*.json` | Split labeled outputs into three files for easy consumption (Good can be used directly; Middle/Bad are optional for review). |

**Notes**
- We treat all items derived from the same **source passage** as an **indivisible unit** for splitting (strict passage-level split) to avoid leakage between corpus and benchmark.
- Ensure your metadata preserves alignment between `raw_id` ↔ PDF, and stores the merged `page_range` list for calibration/traceability.

#### 2.2 Training Instance Generation & Scoring (`02_data_generation/02_sft_generation_scoring/`)

This folder turns **Good passages** into **instruction-style reasoning instances** (Q/A/CoT), and optionally runs a second-model audit (“LLM as a judge”).

**Pipeline overview (aligned with the design doc)**
1) **Prepare grounded passages**  
   - Each entry contains `text` plus `metadata` that tracks `raw_id`, `text_id`, and merged `page_range`.

2) **Tagging: subject + question type**  
   - Ask an LLM to predict up to **5 subjects** and up to **3 question types** (e.g., `MCQ / QA / Dialogue`).  
   - If nothing fits, fall back to a general category such as `medical knowledge`.

3) **Expand multi-label entries into single-label instances**  
   - Split one entry with `(subject_list, type_list)` into multiple entries, each with exactly **one** `subject` and **one** `type`.  
   - Assign `entry_id` like `01, 02, 03, ...`.

4) **Add generation controls**  
   - Add deterministic `id`, plus `few_shot` and `question_template` fields to control instance generation.

5) **Generate Q/A/CoT (grounded-only)**  
   - Generate **question**, **answer**, and **cot** strictly based on the given `text`.  
   - If the model cannot generate grounded content, it must output `<FAIL>`.  
   - After generation, remove `few_shot` and keep the rest.

6) **LLM-as-a-judge scoring (optional)**  
   - Score each instance with 0–1 values such as:
     - `grounded_in_context`
     - `medical_correctness`
     - `reasoning_clarity`
     - `language_quality`  
   - Apply thresholds later to filter training instances.

**Suggested script naming (optional, but recommended)**
- `training data.py` → `build_instances.py`
- `cot.py` → `generate_qa_cot.py`
- `score.py` → `judge_and_score.py`

(You can keep the original names if you prefer; the above are just for readability.)

---

### 3) Translation (`Data_code/03_translation/`)

This directory includes scripts for translation and lexicon-guided term preservation used in **HiMed-West**.  
(Details to be filled once the translation pipeline section is finalized.)

---

## 📄 License
- Code: `<MIT / Apache-2.0 / ...>`
- Data: `<license + attribution + restrictions>`


## 📄 License
- Code: <MIT/Apache-2.0/...>
- Data: <license + attribution + restrictions>


## 📄 License
- Code: `<MIT/Apache-2.0/...>`
- Data: `<license + attribution + restrictions>`

---

## 📖 Citation
```bibtex
@misc{<YOUR_BIBKEY>,
  title        = {<TITLE>},
  author       = {<AUTHORS>},
  year         = {<YEAR>},
  eprint       = {<ARXIV_ID>},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {<URL>}
}
```

---

## 🙏 Acknowledgements
- Base model: LLaMA-3.1-8B-Instruct (Meta)
- Tooling: Hugging Face Transformers / Accelerate / <TRL> / vLLM / SGLang
- Contributors: <names/affiliations>

---

## 📬 Contact
- Maintainer: <name>
- Email: <email>
- Issues: please use GitHub Issues for bugs / requests
