import os
import json
import time
import traceback
import threading
import random
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
MAX_RETRY = int(os.getenv("MAX_RETRY", "3"))
DEBUG = bool(int(os.getenv("DEBUG", "0")))

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "100"))
REPORT_BATCH_SIZE = int(os.getenv("REPORT_BATCH_SIZE", "100"))


class GetOpenAI:
    _session = None
    _session_lock = threading.Lock()

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:
                    cls._session = requests.Session()
                    retry_strategy = Retry(
                        total=3,
                        backoff_factor=0.3,
                        status_forcelist=[429, 500, 502, 503, 504],
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
                    cls._session.mount("http://", adapter)
                    cls._session.mount("https://", adapter)
        return cls._session

    @staticmethod
    def __gpt_api_stream(messages: list, model=MODEL_NAME):
        try:
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            data = {"model": model, "messages": messages}
            session = GetOpenAI._get_session()
            response = session.post(API_BASE_URL, headers=headers, json=data, timeout=(30, 120))
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
            try:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    return False, f"No content: {response.text[:200]}"
                return True, content
            except json.JSONDecodeError:
                return False, f"Invalid JSON: {response.text[:200]}"
        except requests.exceptions.Timeout as err:
            return False, f"Timeout: {err}"
        except requests.exceptions.ConnectionError as err:
            return False, f"Connection error: {err}"
        except requests.exceptions.RequestException as err:
            return False, f"Request error: {err}"
        except Exception as err:
            return False, f"Exception: {err}"

    def get_respons(self, messages, model=MODEL_NAME):
        out = ""
        for attempt in range(MAX_RETRY):
            ok, out = self.__gpt_api_stream(messages, model=model)
            if ok:
                return ok, (out or "").strip()
            if attempt < MAX_RETRY - 1:
                time.sleep(2 ** (attempt + 1))
        return ok, (out or "").strip()


openai_tool = GetOpenAI()


def parse_label_and_reason(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {"label": "DEFINITE_ISSUE", "reason": "Empty GPT response."}
    lines = raw.splitlines()
    first = (lines[0] if lines else "").strip()
    rest = "\n".join((l.strip() for l in lines[1:])).strip()
    valid = {"NO_PROBLEM", "POSSIBLE_ISSUE", "DEFINITE_ISSUE"}
    if first not in valid:
        up = first.upper()
        for v in valid:
            if v in up:
                first = v
                break
    if first not in valid:
        first = "DEFINITE_ISSUE"
    return {"label": first, "reason": rest or ""}


def call_gpt_classify_text(text: str) -> Dict[str, str]:
    prompt = f"""
You are an OCR text quality evaluator.

Your main goal is to determine whether the text contains serious, meaning-breaking OCR corruption.
Default to NO_PROBLEM unless there is clear evidence of major corruption.

Output format:
- FIRST LINE: ONLY one of:
  NO_PROBLEM
  POSSIBLE_ISSUE
  DEFINITE_ISSUE
- SECOND LINE: short English explanation.

Text:
{text}
""".strip()

    messages = [
        {"role": "system", "content": "You are a lenient OCR evaluator who flags only truly broken text."},
        {"role": "user", "content": prompt},
    ]
    ok, raw = openai_tool.get_respons(messages, model=MODEL_NAME)
    if ok:
        return parse_label_and_reason(raw)
    return {"label": "DEFINITE_ISSUE", "reason": f"GPT call failed: {raw}"}


def process_one_file(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return

    def _one(i_entry):
        i, entry = i_entry
        text = entry.get("text", "")
        lab = call_gpt_classify_text(text)
        new_e = dict(entry)
        new_e.update(lab)
        return i, new_e

    out = [None] * len(data)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = [ex.submit(_one, (i, data[i])) for i in range(len(data))]
        for fut in as_completed(futures):
            i, ne = fut.result()
            out[i] = ne

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def process_all(input_dir: str, output_dir: str, recursive: bool = False):
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    files = sorted(in_root.rglob("*.json")) if recursive else sorted(in_root.glob("*.json"))
    for fp in files:
        out_path = out_root / (fp.stem.replace(".calibrate", "") + ".label.json")
        process_one_file(fp, out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    process_all(args.input_dir, args.output_dir, recursive=args.recursive)
