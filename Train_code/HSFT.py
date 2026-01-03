#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import shutil
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader

import swanlab
from accelerate import Accelerator
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from jinja2 import Template

# 避免 tokenizers fork warning（不影响训练行为）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.umask(0)
logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


# -----------------------------
# Distributed helpers
# -----------------------------
def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1

def broadcast_bool(accelerator: Accelerator, flag: bool) -> bool:
    """把 rank0 的 bool 决策广播到所有 rank，保证分支一致。"""
    if not (is_dist_initialized() and get_world_size() > 1):
        return flag
    if accelerator.is_main_process:
        t = torch.tensor([1 if flag else 0], device=accelerator.device, dtype=torch.int32)
    else:
        t = torch.tensor([0], device=accelerator.device, dtype=torch.int32)
    dist.broadcast(t, src=0)
    return bool(t.item())


# -----------------------------
# JSON utilities
# -----------------------------
def safe_json_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


# -----------------------------
# Dataset (Stage-2 CoT SFT)
# -----------------------------
class TrainDataset(torch.utils.data.Dataset):
    """
    Stage-2 CoT SFT（huatuo风格：Template.render + tokenizer.encode）
      - prompt 视为 user
      - assistant 输出为：
            ## Thinking
            {Complex_CoT}

            ## Response
            {ground_truth}
      - 只在 assistant tokens 上算 loss（mask prompt）
      - dynamic padding + attention_mask
      - tail truncation：保留末尾 max_seq_len
    """

    def __init__(self, config: argparse.Namespace, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_len = int(config.max_seq_len)

        if not os.path.exists(config.data_path):
            raise FileNotFoundError(f"Data file not found: {config.data_path}")

        raw = safe_json_load(config.data_path)
        if not isinstance(raw, list):
            raise ValueError("Training JSON must be a list of dict samples.")

        data: List[Dict[str, Any]] = []
        for da in raw:
            if not isinstance(da, dict):
                continue
            if da.get("prompt") and da.get("Complex_CoT") and da.get("ground_truth"):
                data.append(da)

        if get_rank() == 0:
            print(f"Original data: {len(raw)}, Filtered: {len(data)}")
        self.data = data

        # fallback chat template（llama3-instruct 形状）
        chat_template_llama3 = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + "
            "message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
        )
        if not getattr(self.tokenizer, "chat_template", None):
            self.tokenizer.chat_template = chat_template_llama3

        self.template = Template(self.tokenizer.chat_template)
        self.debug = 0

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index]

    @staticmethod
    def _clean_text(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = s.replace("(Option text extraction failed)", "").strip()
        while s.endswith("..") or s.endswith("。。") or s.endswith("。."):
            s = s[:-1].strip()
        return s

    def get_response(self, da: Dict[str, Any]) -> str:
        cot = self._clean_text(da["Complex_CoT"])
        ans = self._clean_text(da["ground_truth"])
        return "## Thinking\n\n{}\n\n## Response\n\n{}".format(cot, ans)

    def _render_text(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        return self.template.render(
            messages=messages,
            bos_token=self.tokenizer.bos_token or "",
            add_generation_prompt=add_generation_prompt,
        )

    def build_features(self, da: Dict[str, Any]) -> Dict[str, List[int]]:
        q = da["prompt"]
        a = self.get_response(da)

        full_text = self._render_text(
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            add_generation_prompt=False,
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        prompt_text = self._render_text(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # huatuo 风格：prompt_ids 是 full_ids 前缀；不匹配则 LCP fallback
        if len(prompt_ids) <= len(full_ids) and full_ids[:len(prompt_ids)] == prompt_ids:
            cut = len(prompt_ids)
        else:
            m = min(len(full_ids), len(prompt_ids))
            i = 0
            while i < m and full_ids[i] == prompt_ids[i]:
                i += 1
            cut = i
            if get_rank() == 0 and self.debug < 2:
                print(f"[WARN] prompt_ids not strict prefix; fallback cut by LCP={cut}")

        labels = [-100] * cut + full_ids[cut:]
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[-self.max_seq_len:]
            labels = labels[-self.max_seq_len:]

        if get_rank() == 0 and self.debug < 2:
            print("---- debug sample ----")
            print("prompt:", q[:120].replace("\n", "\\n"))
            print("decoded input:", self.tokenizer.decode(full_ids[:min(len(full_ids), 256)]))
            # labels 打印时把 -100 替换成 eos 方便 decode
            eos = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else 0
            lbs_show = [t if t != -100 else eos for t in labels]
            print("decoded labels:", self.tokenizer.decode(lbs_show[:256]))
            self.debug += 1

        return {"input_ids": full_ids, "labels": labels}

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        feats = [self.build_features(da) for da in batch]
        input_ids_list = [f["input_ids"] for f in feats]
        labels_list = [f["labels"] for f in feats]

        max_len = min(max(len(x) for x in input_ids_list), self.max_seq_len)

        # ✅ 改动(1)：不新增 pad token，直接用 eos padding（Huatuo 风格）
        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id is None; cannot do EOS-padding.")
        pad_id = int(self.tokenizer.eos_token_id)

        padded_input_ids, padded_labels, attention_masks = [], [], []
        for ids, lbs in zip(input_ids_list, labels_list):
            ids = ids[:max_len]
            lbs = lbs[:max_len]
            real_len = len(ids)
            pad_len = max_len - real_len

            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(lbs + [-100] * pad_len)
            attention_masks.append([1] * real_len + [0] * pad_len)

        return {
            "input_ids": torch.LongTensor(padded_input_ids),
            "labels": torch.LongTensor(padded_labels),
            "attention_mask": torch.LongTensor(attention_masks),
        }


# -----------------------------
# Metric (token-weighted)
# -----------------------------
class SFTMetric:
    """token-weighted loss + token-level acc (labels != -100)."""
    def __init__(self, device: torch.device):
        self.device = device
        self.use_distributed = is_dist_initialized() and get_world_size() > 1
        self.reset()

    def reset(self) -> None:
        self.correct = torch.zeros((), device=self.device, dtype=torch.float32)
        self.total_tokens = torch.zeros((), device=self.device, dtype=torch.float32)
        self.loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
        shift_preds = logits[..., :-1, :].argmax(dim=-1)
        shift_labels = labels[..., 1:]
        mask = shift_labels != -100
        token_count = mask.sum().to(dtype=torch.float32)

        if token_count.item() > 0:
            correct = ((shift_preds == shift_labels) & mask).sum().to(dtype=torch.float32)
            self.correct += correct
            self.total_tokens += token_count
            self.loss_sum += loss.detach().to(dtype=torch.float32) * token_count

    def get_metric(self, reset: bool = True) -> Tuple[float, float]:
        correct = self.correct.clone()
        total = self.total_tokens.clone()
        loss_sum = self.loss_sum.clone()

        if self.use_distributed:
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)

        denom = total + 1e-9
        acc = (correct / denom).item()
        loss = (loss_sum / denom).item()

        if reset:
            self.reset()
        return acc, loss


# -----------------------------
# Save helpers (ZeRO-2 safe) + Huatuo-style "补拷贝"
# -----------------------------
def get_zero_stage(accelerator: Accelerator) -> int:
    stage = 0
    ds_plugin = accelerator.state.deepspeed_plugin
    if ds_plugin is not None:
        try:
            stage = int(ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", 0))
        except Exception:
            stage = 0
    return stage

def rm_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

def atomic_rename_dir(src_dir: str, dst_dir: str) -> None:
    parent = os.path.dirname(dst_dir)
    os.makedirs(parent, exist_ok=True)

    old_dir = dst_dir + ".old"
    rm_dir(old_dir)

    if os.path.exists(dst_dir):
        os.rename(dst_dir, old_dir)
    os.rename(src_dir, dst_dir)
    rm_dir(old_dir)

def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def fast_copytree_hardlink(src_dir: str, dst_dir: str) -> None:
    if os.path.exists(dst_dir):
        rm_dir(dst_dir)
    shutil.copytree(src_dir, dst_dir, copy_function=_link_or_copy)

def _should_skip_copy_file(filename: str) -> bool:
    # 不拷贝权重类大文件，避免覆盖 finetune 权重
    if filename.startswith("pytorch_model") and filename.endswith(".bin"):
        return True
    if filename.endswith(".safetensors"):
        return True
    if filename.endswith(".index.json"):
        return True
    return False

def copy_missing_files_from_base(base_dir: str, dst_dir: str) -> List[str]:
    copied = []
    for item in os.listdir(base_dir):
        src = os.path.join(base_dir, item)
        dst = os.path.join(dst_dir, item)
        if not os.path.isfile(src):
            continue
        if os.path.exists(dst):
            continue
        if _should_skip_copy_file(item):
            continue
        _link_or_copy(src, dst)
        copied.append(item)
    return copied

def patch_tokenizer_config_chat_template(dst_dir: str, tokenizer: AutoTokenizer) -> None:
    # 不重写 tokenizer.json，只确保 tokenizer_config.json 的 chat_template 一致（可选）
    if not getattr(tokenizer, "chat_template", None):
        return
    path = os.path.join(dst_dir, "tokenizer_config.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict) and cfg.get("chat_template") != tokenizer.chat_template:
            cfg["chat_template"] = tokenizer.chat_template
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def save_hf_checkpoint_huatuo_copy(
    accelerator: Accelerator,
    model,
    tokenizer: AutoTokenizer,
    save_dir: str,
    base_model_dir: str,
) -> None:
    """
    ✅ 改动(3)：不 tokenizer.save_pretrained()，避免重新序列化 tokenizer.json
    - 保存 finetuned 权重：save_pretrained(state_dict=...)
    - 从 base_model_dir 补拷贝缺失文件（含 tokenizer.json/tokenizer_config.json 等）
    """
    stage = get_zero_stage(accelerator)
    if stage == 3:
        raise RuntimeError("Detected ZeRO-3, but this script only supports ZeRO-2 saving.")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        unwrapped.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        copied = copy_missing_files_from_base(base_model_dir, save_dir)
        patch_tokenizer_config_chat_template(save_dir, tokenizer)
        accelerator.print(f"[CKPT] Saved to {save_dir} | copied_from_base={len(copied)}")
    accelerator.wait_for_everyone()

def update_best_from_saved_ckpt(accelerator: Accelerator, saved_ckpt_dir: str, best_dir: str) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        staging = best_dir + ".tmp_replace"
        rm_dir(staging)
        fast_copytree_hardlink(saved_ckpt_dir, staging)
        atomic_rename_dir(staging, best_dir)
        accelerator.print(f"[BEST] Best checkpoint updated at: {best_dir}")
    accelerator.wait_for_everyone()


# -----------------------------
# Training
# -----------------------------
def train(args: argparse.Namespace) -> None:
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    ga = int(accelerator.gradient_accumulation_steps)
    world_size = int(accelerator.num_processes)

    if accelerator.is_main_process:
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment_name,
            config=vars(args),
        )

    accelerator.print(f"args:\n{args}")
    accelerator.print(f"num_processes={world_size}, gradient_accumulation_steps={ga}")

    # Update DS batch size (and GA for safety)
    if accelerator.state.deepspeed_plugin is not None:
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        ds_cfg["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
        ds_cfg["train_batch_size"] = args.train_bsz_per_gpu * world_size * ga
        ds_cfg["gradient_accumulation_steps"] = ga

    # tokenizer：不改 vocab（不加 pad token）
    # 同时加一层兼容：如果 transformers 支持 fix_mistral_regex，就打开（不改变 llama 的行为）
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # ✅ 与 eos padding 对齐
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is None; cannot set pad_token_id to eos.")
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = int(tokenizer.eos_token_id)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    # optimizer
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay_keywords)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay_keywords)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # data
    train_dataset = TrainDataset(args, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # total update steps（按 optimizer-step）
    steps_per_epoch = len(train_dataloader)  # micro-steps per rank
    num_update_steps_per_epoch = math.ceil(steps_per_epoch / ga)
    num_training_steps = num_update_steps_per_epoch * args.n_epochs
    warmup_steps = int(args.warmup_rates * num_training_steps)

    accelerator.print(
        f"steps_per_epoch={steps_per_epoch}, num_update_steps_per_epoch={num_update_steps_per_epoch}, "
        f"num_training_steps={num_training_steps}, warmup_steps={warmup_steps}"
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    step_log_interval = None if args.log_steps_per_epoch <= 0 else max(1, num_update_steps_per_epoch // args.log_steps_per_epoch)
    ckpt_save_interval = None if args.ckpt_per_epoch <= 0 else max(1, num_update_steps_per_epoch // args.ckpt_per_epoch)

    best_score = float("inf")
    best_root = os.path.join(args.best_ckpt_dir, args.experiment_name)
    best_dir = os.path.join(best_root, "best_checkpoint")
    ema_loss: Optional[float] = None

    global_step = 0  # optimizer-step
    model.train()

    for epoch in range(args.n_epochs):
        metric_step = SFTMetric(device=accelerator.device)
        metric_epoch = SFTMetric(device=accelerator.device)
        updates_in_epoch = 0

        it = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}") \
            if accelerator.is_main_process else enumerate(train_dataloader)

        for _, batch in it:
            with accelerator.accumulate(model):
                outputs = model(**batch, return_dict=True, use_cache=False)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            metric_step.update(outputs.logits, batch["labels"], loss)
            metric_epoch.update(outputs.logits, batch["labels"], loss)

            if accelerator.sync_gradients:
                global_step += 1
                updates_in_epoch += 1

                acc_step, loss_step = metric_step.get_metric(reset=True)

                if accelerator.is_main_process:
                    ema_loss = float(loss_step) if ema_loss is None else (args.ema_decay * float(ema_loss) + (1.0 - args.ema_decay) * float(loss_step))

                    it.set_postfix(
                        loss=round(loss_step, 4),
                        ema=round(ema_loss, 4),
                        acc=round(acc_step, 4),
                        lr=lr_scheduler.get_last_lr()[0],
                        up=updates_in_epoch,
                    )

                    if step_log_interval is not None and (
                        (updates_in_epoch % step_log_interval == 0) or (updates_in_epoch == num_update_steps_per_epoch)
                    ):
                        swanlab.log(
                            {
                                "step_loss": float(loss_step),
                                "step_acc": float(acc_step),
                                "ema_loss": float(ema_loss),
                                "lr": float(lr_scheduler.get_last_lr()[0]),
                                "epoch": epoch,
                                "updates_in_epoch": updates_in_epoch,
                            },
                            step=global_step,
                        )

                if ckpt_save_interval is not None and (
                    (updates_in_epoch % ckpt_save_interval == 0) or (updates_in_epoch == num_update_steps_per_epoch)
                ):
                    accelerator.wait_for_everyone()
                    ckpt_name = f"checkpoint-epoch-{epoch}-step-{updates_in_epoch}"
                    ckpt_dir = os.path.join(args.output_dir, ckpt_name)

                    # ✅ 使用 Huatuo “补拷贝”保存方式
                    save_hf_checkpoint_huatuo_copy(
                        accelerator=accelerator,
                        model=model,
                        tokenizer=tokenizer,
                        save_dir=ckpt_dir,
                        base_model_dir=args.model_path,
                    )

                    do_update_best = False
                    if accelerator.is_main_process:
                        score = float(ema_loss)
                        if score < best_score - args.best_improve_threshold:
                            best_score = score
                            do_update_best = True

                    do_update_best = broadcast_bool(accelerator, do_update_best)
                    if do_update_best:
                        update_best_from_saved_ckpt(accelerator, ckpt_dir, best_dir)
                        if accelerator.is_main_process:
                            swanlab.log({"best_score": float(best_score)}, step=global_step)

                    accelerator.wait_for_everyone()

        # end of epoch checkpoint
        epoch_acc, epoch_loss = metric_epoch.get_metric(reset=True)
        accelerator.wait_for_everyone()

        epoch_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
        save_hf_checkpoint_huatuo_copy(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            save_dir=epoch_dir,
            base_model_dir=args.model_path,
        )

        # ✅ 改动(4)：修正 epoch2/4/6...（按 1-based）
        epoch_num = epoch + 1
        if accelerator.is_main_process:
            accelerator.print(f"[CKPT] Epoch {epoch_num} saved to {epoch_dir}, epoch_loss={epoch_loss:.6f}")
            swanlab.log(
                {
                    "epoch": epoch_num,
                    "epoch_loss": float(epoch_loss),
                    "epoch_acc": float(epoch_acc),
                    "epoch_global_step": global_step,
                },
                step=global_step,
            )

            if epoch_num % 2 == 0:
                target_dir = os.path.join(best_root, f"epoch-{epoch_num}")
                staging = target_dir + ".tmp_replace"
                rm_dir(staging)
                fast_copytree_hardlink(epoch_dir, staging)
                atomic_rename_dir(staging, target_dir)
                accelerator.print(f"[SPECIFIC EPOCH] Epoch {epoch_num} checkpoint saved to {target_dir}")

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print("\n" + "=" * 60)
        accelerator.print("Training completed!")
        accelerator.print(f"Best score (lower is better): {best_score:.6f}")
        accelerator.print(f"Best checkpoint directory: {best_dir}")
        accelerator.print(f"Specific epochs saved under: {best_root}/epoch-2, epoch-4, ...")
        accelerator.print("=" * 60 + "\n")
        try:
            swanlab.finish()
        except AttributeError:
            pass


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HSFT Stage-2 (Huatuo-style template+encode, EOS padding, copy-base saving, warmup+cosine, ZeRO-2 safe)"
    )

    parser.add_argument("--experiment_name", type=str, default="llama3_stage2_cot_sft")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--output_dir", default="./ckpts", type=str)
    parser.add_argument("--best_ckpt_dir", default="./best_ckpt", type=str)
    parser.add_argument("--log_dir", default="./train_logs", type=str)

    parser.add_argument("--swanlab_project", type=str, default="Hindi_HSFT")
    parser.add_argument("--swanlab_experiment_name", type=str, default=None)

    parser.add_argument("--max_seq_len", default=4096, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--train_bsz_per_gpu", default=8, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--warmup_rates", default=0.03, type=float)
    parser.add_argument("--n_epochs", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--ckpt_per_epoch", default=10, type=int)
    parser.add_argument("--log_steps_per_epoch", default=10, type=int)

    parser.add_argument("--ema_decay", type=float, default=0.9)
    parser.add_argument("--best_improve_threshold", type=float, default=1e-4)

    args = parser.parse_args()

    if args.swanlab_experiment_name is None:
        args.swanlab_experiment_name = args.experiment_name

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.best_ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.best_ckpt_dir, args.experiment_name), exist_ok=True)

    set_seed(args.seed)
    train(args)
