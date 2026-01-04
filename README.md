# <PROJECT-NAME> HiMed: Incentivizing Hindi Reasoning via Decaying Scaffolding Reward Reinforcement Learning in Medical LLMs



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


```text
.
├── Data/                  # all released datasets & benchmark files (or download pointers)
├── Training code/         # stage1/2/3 training + RL + evaluation scripts
└── Data Code/             # data construction / translation / filtering / dedup pipelines
```

**Recommended reading order:** `Data/ → Training code/ → Data Code/`.

---

## 👨‍⚕️ Models

We do **not** release model checkpoints at this stage.

- `Models/` is intentionally left **empty** in this repository.
- Checkpoints and model weights **will be open-sourced once accepted**.

---

## 🚀 Quickstart

### 1) Installation

```bash
git clone https://github.com/FreedomIntelligence/HiMed.git
cd HiMed
```

This repo uses **two Conda environments**:
- **Stage 1 / Stage 2** share the same environment and dependencies (see `Train_code/requirements.txt`).
- **Stage 3 (DSR-RL)** uses a separate environment (see `Train_code/DSR-RL/requirements.txt`).

#### Env A — for Stage 1 / Stage 2 (LA + RC)

```bash
conda create -n himed-train python=3.10 -y
conda activate himed-train

pip install -r Train_code/requirements.txt
```

#### Env B — for Stage 3 (DSR-RL)

```bash
conda create -n himed-rl python=3.10 -y
conda activate himed-rl

pip install -r Train_code/DSR-RL/requirements.txt
```

---

### 2) Training (Stage 1 / Stage 2 / Stage 3)

> Training scripts are under `Train_code/`.  
> Our runs use **8×H200**, **bf16**, and **Accelerate + DeepSpeed (ZeRO-2)**.

(Optional but recommended for large-scale runs)
```bash
mkdir -p /data/tmp
export TMPDIR=/data/tmp
export TEMP=/data/tmp
export TMP=/data/tmp

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
```

#### Stage 1 — Language Adaptation (LA)
Fine-tune the base model (**LLaMA-3.1-8B-Instruct**) on an **8×H200** setup with Accelerate + DeepSpeed. We use bf16 and ZeRO stage-2; see `Train_code/configs/ds_config.yaml` for details.

- Script: `Train_code/LA.py`
- Accelerate/DeepSpeed config: `Train_code/configs/ds_config.yaml`

```bash
conda activate himed-train
cd Train_code

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

#### Stage 2 — Reasoning Cold-Start (RC)
Fine-tune the Stage-1 checkpoint for Hindi medical reasoning on an **8×H200** setup with Accelerate + DeepSpeed (bf16, ZeRO-2). The distributed/ZeRO configuration is defined in `Train_code/configs/ds_config.yaml`.

- Script: `Train_code/RC.py`
- Same config: `Train_code/configs/ds_config.yaml`
- `--model_path` points to the Stage 1 checkpoint (e.g., `best_checkpoint`)

```bash
conda activate himed-train
cd Train_code

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




#### Stage 3 — DSR-RL (Placeholder)
Fine-tune the Stage-2 checkpoint for overall medical reasoning on an **8×H200** setup with Accelerate. The configuration is defined in `Train_code/DSR-RL/config_lora.yaml`.

- Script: `Train_code/DSR-RL/run_grpo_lora.py`
- Config: `Train_code/DSR-RL/config_lora.yaml`
- `model, name:` points to the Stage 2 checkpoint (e.g., `best_checkpoint`)
- `reward_model, model_name:` points to our R<sub>1</sub>  Reward Model 
-  `dataset, path:` points to our RL training dataset.
- Before running, please fill in all the corresponding path in the `config_lora.yaml` file
```bash
conda activate himed-rl
cd Train_code/DSR-RL

accelerate launch run_grpo_lora.py --config config_lora.yaml
```



---

## 📚 Data (HiMed)

HiMed is a Hindi medical dataset and benchmark suite covering both Western medicine and Indian systems of medicine.  
It consists of two parts: **HiMed-Trad** (traditional Indian medicine) and **HiMed-West** (Western medicine under Hindi prompts).  
We enforce strict separation between training corpora and evaluation benchmarks to prevent leakage (see paper for details).

### Directory Structure

```text
Data/
├── HiMed-Trad_Bench/
│   └── HiMed-Trad_Bench.json
├── HiMed-Trad_Corpus/
│   ├── HiMed-Trad_Corpus.part0001.json
│   ├── HiMed-Trad_Corpus.part0002.json
│   ├── HiMed-Trad_Corpus.part0003.json
│   └── HiMed-Trad_Corpus.part0004.json
├── HiMed-West_Bench/
│   └── HiMed-West_Bench.json
├── HiMed-West_Corpus/
│   ├── HiMed-West_Corpus.part0001.json
│   ├── HiMed-West_Corpus.part0002.json
│   ├── HiMed-West_Corpus.part0003.json
│   ├── HiMed-West_Corpus.part0004.json
│   └── HiMed-West_Corpus.part0005.json
└── HiMed-West_Exam/
    └── HiMed-West_Exam.json
```

### Statistics
- **HiMed-Trad Bench**: 6,010
- **HiMed-West Bench**: 1,784
- **HiMed-West Exam**: 470
- **HiMed-Trad Corpus (full)**: 286,657
- **HiMed-West Corpus (full)**: 116,859

### Note on Corpus Sharding
The two training corpora are **sharded** into multiple `*.partXXXX.json` files for easier storage and transfer.  
All parts share the same schema and can be loaded/merged in order.


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


---

### 2) Data Generation (`Data_code/02_data_generation/`)

This directory contains the core construction pipeline for **HiMed-Trad**, including:
- **Passage Preparation & Cleaning** (PDF → OCR → calibrated passages → quality splits)
- **Training Instance Generation & Scoring** (passages → question/answer/reasoning instances + optional judge scoring)

For the full step-by-step workflow and script mapping, see:
- `Data_code/02_data_generation/README.md`


---

### 3) Translation (`Data_code/03_translation/`)

This folder provides the English→Hindi translation pipeline used for **HiMed-West**.

```text
Data_code/03_translation/
├── translation_api.py        # core API (NLLB + lexicon-guided term handling)
└── translate.py              # example: batch-translate a JSON dataset
```

> Note: the example script in this repo may be named `translate (1).py` locally. We recommend renaming it to `translate.py`.

---

#### 3.1 Configure `translation_api.py`

Edit the `_Config` class in `translation_api.py`:

- `MODEL_PATH` (**required**): NLLB model path or HuggingFace repo id  
  e.g., `/data/models/nllb-200-3.3B` or `facebook/nllb-200-3.3B`
- `LEXICON_PATH` (**required**): English–Hindi medical lexicon file (`.xlsx` or `.csv`)  
  Must contain columns **`English`** and **`Hindi`** (can be an empty table with only headers if you want to disable term rules).
- `SOURCE_LANG` (default: `eng_Latn`): NLLB source language code
- `TARGET_LANG` (default: `hin_Deva`): NLLB target language code
- `BATCH_SIZE` (default: `8`): translation batch size inside the API
- `USE_DYNAMIC_BATCHING` (default: `True`): enable length-aware batching for speed
- `LENGTH_BUCKET_SIZE` (default: `16`): bucket size used by dynamic batching

---

#### 3.2 Run the example translator (`translate.py`)

```bash
cd Data_code/03_translation
python translate.py
```

It will ask for:

- `Input JSON path`: path to your dataset JSON
- `Output JSON path`: output file path
- `Batch size` (default: `100`): how many segments to translate per call to `translate_paragraphs`
- `Save interval (batches)` (default: `10`): periodic saving frequency (for long runs)

**Input format**
- A JSON list: `[{"prompt": ..., "ground_truth": ..., "Complex_CoT": ...}, ...]`, or
- A dict container: `{"questions": [...]}`

**What it writes**
- Adds translated fields (if missing):
  - `prompt_hi`
  - `ground_truth_hi`
  - `Complex_CoT_hi` (only if `Complex_CoT` exists)

The script is resumable: if `*_hi` fields already exist, they will be skipped.

---

#### 3.3 Minimal API usage

```python
from translation_api import translate_paragraph

print(translate_paragraph("How to stop a cough?"))
```


---
