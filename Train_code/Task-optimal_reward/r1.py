import os
import math
import json
import subprocess
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

try:
    from transformers import get_linear_schedule_with_warmup

    HAS_HF_SCHED = True
except Exception:
    HAS_HF_SCHED = False

from sklearn.metrics import accuracy_score, f1_score

import wandb

MODEL_PATH = os.environ.get("MODEL_PATH", "")
PATH_TRAIN = ""
PATH_TEST = ""
OUTPUT_DIR = ""

NUM_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 128
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 30
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 1
MAX_LENGTH = 2048

USE_WANDB = True
WANDB_PROJECT = ""
WANDB_RUN_NAME = ""

template = """<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True" . If it is incorrect or fails to select the correct option (if options are provided), output "False" . {}"""


def query_gpus_via_nvidia_smi() -> Optional[list]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        rows = []
        for line in out.strip().splitlines():
            idx_str, used_str, total_str = [x.strip() for x in line.split(",")]
            rows.append((int(idx_str), int(used_str), int(total_str)))
        return rows
    except Exception:
        return None


def pick_best_gpu() -> Optional[int]:
    rows = query_gpus_via_nvidia_smi()
    if not rows:
        return None
    rows_sorted = sorted(rows, key=lambda x: (x[1], -x[2]))
    return rows_sorted[0][0]


def bind_to_best_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    best = pick_best_gpu()
    if best is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best)


def load_any_json(path: str):
    assert path and os.path.exists(path)
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        return data
    raise ValueError("")


def build_hf_dataset(path: str, tokenizer) -> Dataset:
    raw = load_any_json(path)

    def _format(ex):
        text = template.format(ex["answer_llama"], ex["answer"], tokenizer.eos_token)
        label = 1 if (ex["label"] is True or str(ex["label"]).lower() == "true") else 0
        return {"text": text, "labels": label}

    return Dataset.from_list([_format(x) for x in raw])


def tokenize_batch(batch, tokenizer, max_length=2048):
    out = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    out["labels"] = batch["labels"]
    return out


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = F.softmax(logits, dim=-1)
            preds = (probs[:, 1] > 0.5).long()
            all_preds.append(preds.cpu())
            all_labels.append(batch["labels"].cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1


def load_model_sdpa_only(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype="auto",
        num_labels=2,
    )
    try:
        model.set_attn_implementation("sdpa")
        model.config.attn_implementation = "sdpa"
    except Exception:
        try:
            model.config.attn_implementation = "sdpa"
        except Exception:
            pass
    model.config.problem_type = "single_label_classification"
    model.config.id2label = {0: "False", 1: "True"}
    model.config.label2id = {"False": 0, "True": 1}
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model


def main():
    bind_to_best_gpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BF16_OK = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    USE_AMP = torch.cuda.is_available()
    AMP_DTYPE = torch.bfloat16 if BF16_OK else (torch.float16 if torch.cuda.is_available() else None)
    USE_SCALER = USE_AMP and (AMP_DTYPE == torch.float16)

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_raw = build_hf_dataset(PATH_TRAIN, tokenizer)
    test_raw = build_hf_dataset(PATH_TEST, tokenizer)

    train_ds = train_raw.map(lambda b: tokenize_batch(b, tokenizer, MAX_LENGTH), batched=True, remove_columns=["text"])
    test_ds = test_raw.map(lambda b: tokenize_batch(b, tokenizer, MAX_LENGTH), batched=True, remove_columns=["text"])

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(train_ds, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=PER_DEVICE_EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)

    model = load_model_sdpa_only(MODEL_PATH)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    t_total = NUM_EPOCHS * num_update_steps_per_epoch
    if HAS_HF_SCHED:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total
        )
    else:

        def lr_lambda(current_step):
            if current_step < WARMUP_STEPS:
                return float(current_step) / float(max(1, WARMUP_STEPS))
            return max(0.0, float(t_total - current_step) / float(max(1, t_total - WARMUP_STEPS)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_SCALER)

    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model_path": MODEL_PATH,
                "train_path": PATH_TRAIN,
                "test_path": PATH_TEST,
                "output_dir": OUTPUT_DIR,
                "epochs": NUM_EPOCHS,
                "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "learning_rate": LEARNING_RATE,
                "warmup_steps": WARMUP_STEPS,
                "weight_decay": WEIGHT_DECAY,
                "logging_steps": LOGGING_STEPS,
                "max_length": MAX_LENGTH,
                "use_amp": USE_AMP,
                "amp_dtype": str(AMP_DTYPE),
            },
        )
        wandb.watch(model, log="all", log_freq=LOGGING_STEPS)

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss

            running_loss += loss.item()
            loss = loss / GRAD_ACCUM_STEPS

            if USE_SCALER:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % GRAD_ACCUM_STEPS == 0:
                if USE_SCALER:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                avg_loss = running_loss / GRAD_ACCUM_STEPS
                current_lr = (
                    scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
                )

                if USE_WANDB and wandb.run is not None and (global_step % 10 == 0):
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        }
                    )

                running_loss = 0.0

        acc, f1 = evaluate(model, test_loader, device)
        if USE_WANDB and wandb.run is not None:
            wandb.log(
                {
                    "eval/accuracy": acc,
                    "eval/f1": f1,
                    "eval/epoch": epoch,
                }
            )

        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch-{epoch:02d}")
        os.makedirs(epoch_dir, exist_ok=True)

        try:
            model.config.attn_implementation = "sdpa"
            model.config.use_cache = True
        except Exception:
            pass
        model.config.id2label = {0: "False", 1: "True"}
        model.config.label2id = {"False": 0, "True": 1}

        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    acc, f1 = evaluate(model, test_loader, device)

    if USE_WANDB and wandb.run is not None:
        wandb.log(
            {
                "final/accuracy": acc,
                "final/f1": f1,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()