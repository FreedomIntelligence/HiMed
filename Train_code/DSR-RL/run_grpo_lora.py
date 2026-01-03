#!/usr/bin/env python3
import sys
import os
import argparse
import math
import yaml
from typing import Any, Dict, List, Optional

import torch
from accelerate import PartialState
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig
import wandb

from grpo import RewardModelScorer, load_json_dataset, reward_model_reward_func
from hindi_reward import create_hindi_reward_func


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "-1")) in (-1, 0)


def load_config(config_path: str = "") -> dict:
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError("")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_base_model(m: Any) -> Any:
    if hasattr(m, "get_base_model"):
        try:
            return m.get_base_model()
        except Exception:
            pass
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        return m.base_model.model
    return m


def _infer_torch_dtype(training_cfg: Dict) -> torch.dtype:
    bf16 = bool(training_cfg.get("bf16", True))
    fp16 = bool(training_cfg.get("fp16", False))
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def _normalize_weights(ws: List[float]) -> List[float]:
    s = float(sum(ws))
    if s <= 0:
        n = len(ws)
        return [1.0 / n for _ in range(n)]
    return [float(w) / s for w in ws]


class ExtraWandbCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs is None:
            return
        logs = dict(logs)
        logs["step"] = state.global_step
        if wandb.run is not None:
            wandb.log(logs)


class RewardWeightsAnnealCallback(TrainerCallback):
    def __init__(
        self,
        start_weights: List[float],
        end_weights: List[float],
        schedule: str = "linear",
    ):
        self.start = _normalize_weights([float(x) for x in start_weights])
        self.end = _normalize_weights([float(x) for x in end_weights])
        self.schedule = str(schedule).lower()
        self.total_steps: Optional[int] = None
        self.current: List[float] = list(self.start)

    def on_train_begin(self, args, state, control, **kwargs):
        ms = getattr(state, "max_steps", None)
        if ms is None or ms <= 0:
            ms = 1
        self.total_steps = int(max(1, ms))
        self.current = self._compute(step=state.global_step)

    def on_step_begin(self, args, state, control, **kwargs):
        self.current = self._compute(step=state.global_step)

        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.args.reward_weights = list(self.current)
            if hasattr(trainer, "reward_weights"):
                try:
                    trainer.reward_weights = list(self.current)
                except Exception:
                    pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.args.reward_weights = list(self.current)
            if hasattr(trainer, "reward_weights"):
                try:
                    trainer.reward_weights = list(self.current)
                except Exception:
                    pass

        logs["reward_weights/w1"] = float(self.current[0])
        logs["reward_weights/w2"] = float(self.current[1]) if len(self.current) > 1 else 0.0
        logs["reward_weights/progress"] = float(self._progress(state.global_step))

    def _progress(self, step: int) -> float:
        if self.total_steps is None or self.total_steps <= 1:
            return 1.0
        t = float(step) / float(self.total_steps - 1)
        return max(0.0, min(1.0, t))

    def _compute(self, step: int) -> List[float]:
        t = self._progress(step)
        if self.schedule == "cosine":
            tt = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            tt = t

        ws = [(1.0 - tt) * s + tt * e for s, e in zip(self.start, self.end)]
        return _normalize_weights(ws)


def merge_lora_to_base_and_save_hf(
    base_model_name_or_path: str = "",
    adapter_dir: str = "",
    merged_dir: str = "",
    tokenizer: AutoTokenizer = None,
    torch_dtype: torch.dtype = torch.float32,
):
    os.makedirs(merged_dir, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else pad_id

    if hasattr(merged, "config") and merged.config is not None:
        merged.config.pad_token_id = pad_id
        merged.config.eos_token_id = eos_id
    if getattr(merged, "generation_config", None) is not None:
        merged.generation_config.pad_token_id = pad_id
        merged.generation_config.eos_token_id = eos_id

    merged.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    tokenizer.save_pretrained(merged_dir)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_path = config["dataset"]["path"]
    training_cfg: Dict = config["training"]
    model_name = config["model"]["name"]

    if not os.path.exists(dataset_path):
        sys.exit(1)

    distributed_state = PartialState()

    with distributed_state.local_main_process_first():
        reward_scorer = RewardModelScorer(config["reward_model"])
        rm_reward = reward_model_reward_func(reward_scorer)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_json_dataset(dataset_path)

    lora_cfg = config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        bias=lora_cfg.get("bias", "none"),
    )

    wandb_cfg = config.get("wandb", {})
    report_to = ["tensorboard"]
    if wandb_cfg.get("enabled", False):
        report_to = ["wandb"]
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "")
        if wandb_cfg.get("run_name"):
            os.environ["WANDB_NAME"] = str(wandb_cfg["run_name"])
        if wandb_cfg.get("tags"):
            os.environ["WANDB_TAGS"] = ",".join([str(x) for x in wandb_cfg["tags"]])
        if wandb_cfg.get("notes"):
            os.environ["WANDB_NOTES"] = str(wandb_cfg["notes"])

    reward_weights_start = training_cfg.get("reward_weights", [0.8, 0.2])
    reward_weights_end = training_cfg.get("reward_weights_end", [0.9, 0.1])
    reward_weights_schedule = training_cfg.get("reward_weights_schedule", "linear")

    gc = bool(training_cfg.get("gradient_checkpointing", True))
    gc_kwargs = training_cfg.get("gradient_checkpointing_kwargs", None)

    training_args = GRPOConfig(
        output_dir=training_cfg["output_dir"],
        ddp_find_unused_parameters=False,
        learning_rate=float(training_cfg["learning_rate"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 1)),
        logging_steps=int(training_cfg.get("logging_steps", 1)),
        save_steps=int(training_cfg.get("save_steps", 200)),
        max_prompt_length=int(training_cfg["max_prompt_length"]),
        max_completion_length=int(training_cfg["max_completion_length"]),
        num_generations=int(training_cfg.get("num_generations", 8)),
        beta=float(training_cfg.get("beta", 0.0)),
        epsilon=float(training_cfg.get("epsilon", 0.2)),
        reward_weights=list(reward_weights_start),
        bf16=bool(training_cfg.get("bf16", True)),
        fp16=bool(training_cfg.get("fp16", False)),
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs=gc_kwargs,
        use_vllm=bool(training_cfg.get("use_vllm", False)),
        vllm_mode=training_cfg.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=float(training_cfg.get("vllm_gpu_memory_utilization", 0.5)),
        report_to=report_to,
        run_name=wandb_cfg.get("run_name", None),
        model_init_kwargs={"device_map": None},
    )

    reward_funcs = [rm_reward]
    has_r2 = False
    if training_cfg.get("lexicon_config"):
        reward_funcs.append(create_hindi_reward_func(training_cfg["lexicon_config"]))
        has_r2 = True

    trainer = GRPOTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    base = _get_base_model(trainer.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else pad_id

    if hasattr(base, "config") and base.config is not None:
        base.config.pad_token_id = pad_id
        base.config.eos_token_id = eos_id
    if getattr(base, "generation_config", None) is not None:
        base.generation_config.pad_token_id = pad_id
        base.generation_config.eos_token_id = eos_id

    emb_vocab = base.get_input_embeddings().weight.size(0)
    tok_len = len(tokenizer)
    assert 0 <= pad_id < emb_vocab
    assert tok_len <= emb_vocab

    trainer.add_callback(ExtraWandbCallback())

    if has_r2:
        trainer.add_callback(
            RewardWeightsAnnealCallback(
                start_weights=list(reward_weights_start),
                end_weights=list(reward_weights_end),
                schedule=str(reward_weights_schedule),
            )
        )

    if gc:
        if hasattr(trainer.model, "config"):
            trainer.model.config.use_cache = False

        trainer.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        if hasattr(trainer.model, "enable_input_require_grads"):
            trainer.model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            trainer.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    if is_main_process():
        out_dir = training_cfg["output_dir"]

        adapter_dir = os.path.join(out_dir, "final_adapter")
        trainer.save_model(adapter_dir)

        merged_dir = os.path.join(out_dir, "final_merged_hf")
        torch_dtype = _infer_torch_dtype(training_cfg)

        merge_lora_to_base_and_save_hf(
            base_model_name_or_path=model_name,
            adapter_dir=adapter_dir,
            merged_dir=merged_dir,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
        )


if __name__ == "__main__":
    main()