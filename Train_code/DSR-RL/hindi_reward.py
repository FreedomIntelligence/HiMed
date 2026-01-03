import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    import pyahocorasick  

    AHO_ENABLED = True
except ImportError:
    AHO_ENABLED = False


def is_hindi_char(c: str) -> bool:
    return 0x0900 <= ord(c) <= 0x097F


def has_hindi_chars(s: str) -> bool:
    return any(is_hindi_char(c) for c in s)


class LexiconProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.english_column = config.get("english_column", "English")
        self.hindi_column = config.get("hindi_column", "Hindi")
        self.terms = self._load_and_clean_terms()
        self.automaton = None
        self.regex_pattern = None
        self._build_matcher()

    def _load_lexicon_table(self) -> pd.DataFrame:
        path = self.config["path"]
        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext in {".xlsx", ".xls"}:
            return pd.read_excel(path, engine="openpyxl")
        if ext == ".csv":
            return pd.read_csv(path)

        raise ValueError(f"Unsupported lexicon file format: {ext}")

    def _load_and_clean_terms(self) -> Set[str]:
        df = self._load_lexicon_table()

        normalized_columns = {col.lower(): col for col in df.columns}
        english_col = normalized_columns.get(self.english_column.lower())

        if not english_col or english_col not in df.columns:
            return set()

        en_terms = df[english_col].dropna().astype(str)

        en_terms = en_terms.str.lower().str.strip()
        en_terms = en_terms.str.replace(r"\s+", " ", regex=True)
        en_terms = en_terms.str.replace(r"[-_]+", "-", regex=True)

        min_en_len = int(self.config.get("min_en_length", 2))
        en_terms = en_terms[en_terms.str.len() >= min_en_len]

        return set(en_terms.tolist())

    def _build_matcher(self):
        if not self.terms:
            self.regex_pattern = None
            return

        sorted_terms = sorted(self.terms, key=len, reverse=True)
        escaped_terms = [rf"\b{re.escape(term)}\b" for term in sorted_terms]
        pattern_str = "|".join(escaped_terms)
        self.regex_pattern = re.compile(pattern_str, re.IGNORECASE)

    def _expand_span_to_parens(self, text: str, start: int, end: int) -> Tuple[int, int]:
        left = start
        while left > 0 and text[left - 1].isspace():
            left -= 1
        if left > 0 and text[left - 1] == "(":
            left -= 1

        right = end
        while right < len(text) and text[right].isspace():
            right += 1
        if right < len(text) and text[right] == ")":
            right += 1

        return left, right

    def find_term_spans(self, text: str) -> List[Tuple[int, int]]:
        if not self.regex_pattern:
            return []

        spans: List[Tuple[int, int]] = []
        for match in self.regex_pattern.finditer(text):
            s, e = match.span()
            s, e = self._expand_span_to_parens(text, s, e)
            spans.append((s, e))

        spans.sort()
        if not spans:
            return []

        merged = [spans[0]]
        for cur in spans[1:]:
            last_s, last_e = merged[-1]
            if cur[0] <= last_e:
                merged[-1] = (min(last_s, cur[0]), max(last_e, cur[1]))
            else:
                merged.append(cur)

        return merged

    def mask_english_terms(self, text: str) -> str:
        spans = self.find_term_spans(text)
        if not spans:
            return text

        result: List[str] = []
        last_end = 0

        for start, end in spans:
            segment = text[last_end:start].rstrip()
            result.append(segment)
            last_end = end

        remaining = text[last_end:].lstrip()
        result.append(remaining)

        masked_text = "".join(result)
        masked_text = re.sub(r"\s+", " ", masked_text).strip()
        return masked_text


def create_hindi_reward_func(lexicon_config: Dict):
    processor = LexiconProcessor(lexicon_config) if (lexicon_config and os.path.exists(lexicon_config.get("path", ""))) else None

    def _completion_to_text(comp: Any) -> str:
        if isinstance(comp, str):
            return comp

        if isinstance(comp, dict):
            return comp.get("content") or comp.get("text") or str(comp)

        if isinstance(comp, list) and len(comp) > 0:
            last = comp[-1]
            if isinstance(last, dict):
                return last.get("content") or last.get("text") or str(last)
            return str(last)

        return str(comp)

    def hindi_reward_func(
        prompts: Optional[List[Any]] = None,
        completions: Optional[List[Any]] = None,
        completions_ids: Optional[List[List[int]]] = None,
        trainer_state: Any = None,
        **kwargs,
    ) -> List[float]:
        if completions is None:
            completions = kwargs.get("completions", [])
        if completions is None:
            completions = []

        if completions_ids is None:
            completions_ids = kwargs.get("completion_ids", None)

        rewards: List[float] = []

        for comp in completions:
            text = _completion_to_text(comp)

            if not has_hindi_chars(text):
                rewards.append(0.0)
                continue

            masked_text = processor.mask_english_terms(text) if processor else text

            hindi_chars = sum(1 for c in masked_text if is_hindi_char(c))
            total_chars = sum(1 for c in masked_text if not c.isspace())

            rewards.append((hindi_chars / total_chars) if total_chars > 0 else 0.0)

        return rewards

    return hindi_reward_func