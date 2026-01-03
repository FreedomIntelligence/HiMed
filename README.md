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

### What’s Included
HiMed covers both **Western medicine** and **Indian systems of medicine**, and supports:
- language adaptation corpora (Hindi medical text / mixed corpora)
- instruction/reasoning data for cold-start
- benchmark suite with standardized evaluation formats

### Data Folder Layout (example)
```text
Data/
├── README.md                        # data index (recommended)
├── himed_corpus/                    # training corpora (if released locally)
├── himed_sft/                       # SFT / cold-start reasoning data
├── himed_bench/                     # benchmark JSON/JSONL
└── LICENSE / NOTICE                 # data license & attribution
```

### Download / Access
- If hosted on Hugging Face: [HiMed on HF](<HF_DATASET_URL>)
- If you keep large files out of git, provide:
  - `Data/README.md` with download scripts / pointers
  - checksums for integrity (optional but recommended)

### Benchmarks
Fill your benchmark table here (add/remove as needed):

| Benchmark | Domain | Language | Format | Split | Notes |
|---|---|---:|---|---|---|
| HiMed-West | Western Medicine | hi/en | JSON | test | MCQ / QA |
| HiMed-Trad | Indian Medicine | hi/en | JSON | test | Ayurveda / etc. |
| <...> | <...> | <...> | <...> | <...> | <...> |

---

## 🧪 Training (Three-Stage Framework)

> This section corresponds to **Training code/**. We recommend adding a `Training code/README.md` as well.

### Hardware / Environment
- GPUs: `<e.g., 8×H100 / 8×A100>`
- Precision: `<bf16/fp16>`
- Distributed: `<deepspeed/accelerate>`
- Key deps: `transformers`, `datasets`, `accelerate`, `<trl/peft/etc.>`

### Stage 1 — Language Adaptation (LAPT)
Goal: improve Hindi medical language competence (domain + language coverage).

```bash
cd "Training code"
accelerate launch --config_file ./configs/<CONFIG_STAGE1>.yaml \
  S1_language_adaptation.py \
  --model_path <BASE_MODEL_ID> \
  --data_path <DATA_PATH_STAGE1> \
  --output_dir <OUTPUT_DIR_STAGE1> \
  --run_name <RUN_NAME_STAGE1>
```

### Stage 2 — Reasoning Cold-Start (SFT)
Goal: bootstrap Hindi medical reasoning (instruction-following + rationale style, if used).

```bash
cd "Training code"
accelerate launch --config_file ./configs/<CONFIG_STAGE2>.yaml \
  S2_reasoning_coldstart.py \
  --model_path <CKPT_FROM_STAGE1> \
  --data_path <DATA_PATH_STAGE2> \
  --output_dir <OUTPUT_DIR_STAGE2> \
  --run_name <RUN_NAME_STAGE2>
```

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

## 🧐 Evaluation

> This section should map to scripts inside **Training code/** or `evaluation/`.

### 1) Launch Server
**SGLang example**
```bash
model_name="<HF_MODEL_ID_OR_LOCAL_PATH>"
port=<PORT>

python -m sglang.launch_server \
  --model-path "$model_name" \
  --port "$port" \
  --mem-fraction-static 0.8 \
  > sglang.log 2>&1 &
```

### 2) Run Evaluation
```bash
cd "Training code"
python evaluation/eval.py \
  --model_name "$model_name" \
  --eval_file "../Data/himed_bench/<EVAL_FILE>.json" \
  --port "$port" \
  --strict_prompt <True/False>
```

### 3) Stop Server
```bash
bash evaluation/kill_server.sh
```

### Metrics
- Primary: `<accuracy / exact match / F1 / etc.>`
- Optional: `<calibration, refusal rate, hallucination metrics, etc.>`

---

## 🧩 Data Pipeline (Data Code)

> This section corresponds to **Data Code/**. Keep it practical: “input → steps → output”.

### Typical Pipeline
1) Collect raw sources (Western + Indian medicine)
2) Cleaning & normalization (Unicode, punctuation, boilerplate removal)
3) Translation / bilingual alignment (if applicable)
4) Deduplication (exact + near-dup; doc + sample level)
5) Filtering / quality control (language ID, domain classifier, rules)
6) Construct SFT / reasoning data (templates, constraints)
7) Build benchmarks (standardized schema + answer checks)

### Example Commands (placeholders)
```bash
cd "Data Code"

# (1) preprocess / clean
python preprocess.py --input <RAW_DIR> --output <CLEAN_DIR>

# (2) translate / align (optional)
python translate.py --input <CLEAN_DIR> --output <HI_DIR> --engine <...>

# (3) dedup
python dedup.py --input <HI_DIR> --output <DEDUP_DIR> --method <minhash/exact>

# (4) build benchmark json
python build_bench.py --input <DEDUP_DIR> --output ../Data/himed_bench/
```

---

## ✅ Reproducibility
- Random seeds: `<seed>`
- Training configs: `Training code/configs/*.yaml`
- Checkpoint naming: `<convention>`
- Logs: `<wandb/swanlab/tensorboard>`
- Recommended: record `git commit`, `transformers` version, `cuda` version in logs

---

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
