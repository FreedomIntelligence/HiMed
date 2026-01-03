import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardModelScorer:
    def __init__(self, config: Dict):
        self.config = config

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        device_conf = config.get("device", "auto")
        if device_conf == "auto":
            self.device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device_conf)

        dtype_str = str(config.get("dtype", "")).lower()
        torch_dtype = None
        if dtype_str in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype_str in {"fp16", "float16"}:
            torch_dtype = torch.float16

        model_name = config.get("model_name", "")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        self.model.to(self.device)
        self.model.eval()

        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            self.model.config.pad_token_id = self.reward_tokenizer.eos_token_id

        self.temperature = float(config.get("temperature", 1.0))
        self.threshold = float(config.get("threshold", 0.4))
        self.max_length = int(config.get("max_length", 2048))
        self.eos_token = self.reward_tokenizer.eos_token or ""

        self._template = (
            "<Model Response>\n{gen}\n</Model Response>\n\n"
            "<Reference Answer>\n{ref}\n</Reference Answer>\n\n"
            "Your task is to evaluate the model response by comparing it to the reference answer. "
            'If the model response is correct and aligns with the reference answer, output "True". '
            'If it is incorrect or fails to select the correct option, output "False".\n\n'
            "{eos}"
        )

    def _format_input(self, gen_text: str, ref_text: str) -> str:
        return self._template.format(gen=gen_text.strip(), ref=ref_text.strip(), eos=self.eos_token)

    @torch.inference_mode()
    def score_batch(self, gen_texts: List[str], ref_texts: List[str]) -> List[float]:
        if not gen_texts or not ref_texts:
            return [0.0] * len(gen_texts)

        min_len = min(len(gen_texts), len(ref_texts))
        gen_texts = gen_texts[:min_len]
        ref_texts = ref_texts[:min_len]

        formatted_inputs = [self._format_input(g, r) for g, r in zip(gen_texts, ref_texts)]

        inputs = self.reward_tokenizer(
            formatted_inputs,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        ).to(self.device)

        logits = self.model(**inputs, return_dict=True).logits
        logits = logits / self.temperature
        probs = F.softmax(logits, dim=-1)
        p_true = probs[:, 1]

        rewards = torch.where(
            p_true > self.threshold,
            torch.ones_like(p_true),
            torch.full_like(p_true, 0.1),
        )

        del inputs, logits, probs
        return rewards.detach().cpu().tolist()


def load_json_dataset(json_file_path: str = "") -> Dataset:

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        if "prompt" not in item:
            continue

        cot = item.get("Complex_CoT", "")
        gt = item.get("ground_truth", "")

        final_gt = f"{cot}\n\n{gt}" if cot else gt
        item["ground_truth"] = final_gt
        processed_data.append(item)

    return Dataset.from_list(processed_data)


def reward_model_reward_func(reward_scorer: RewardModelScorer):
    def reward_func(prompts=None, completions=None, **kwargs):
        if completions is None:
            completions = kwargs.get("completions", [])

        ground_truth = kwargs.get("ground_truth", kwargs.get("reference", None))

        if not completions:
            return []

        if ground_truth is None:
            return [0.0] * len(completions)

        if len(ground_truth) != len(completions):
            num_gen = len(completions) // len(ground_truth)
            expanded_gt = []
            for gt in ground_truth:
                expanded_gt.extend([gt] * num_gen)
            ground_truth = expanded_gt

        return reward_scorer.score_batch(completions, ground_truth)

    return reward_func