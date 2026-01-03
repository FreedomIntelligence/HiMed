import os
import json
import time
import base64
import mimetypes
import traceback
import threading
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

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "50"))
REPORT_BATCH_SIZE = int(os.getenv("REPORT_BATCH_SIZE", "50"))


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


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def call_gpt_with_images(text: str, image_paths: List[Path]) -> str:
    content: List[Dict[str, Any]] = []
    content.append(
        {
            "type": "text",
            "text": (
                "You are an OCR post-correction assistant.\n"
                "Rules:\n"
                "- Only correct characters / punctuation / spacing / diacritics.\n"
                "- Do NOT add new content or delete meaningful content.\n"
                "- Do NOT translate.\n"
                "- Output ONLY the corrected text.\n\n"
                "OCR text:\n"
                "----------------\n"
                f"{text}\n"
                "----------------"
            ),
        }
    )

    for img in image_paths:
        if not img.exists():
            continue
        data_url = image_to_data_url(img)
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    if len(content) == 1:
        return text

    messages = [
        {"role": "system", "content": "You are a highly accurate OCR post-correction assistant."},
        {"role": "user", "content": content},
    ]
    ok, result = openai_tool.get_respons(messages, model=MODEL_NAME)
    if ok:
        return result
    return text


def process_single_entry(args):
    idx, entry, picture_root = args
    text = entry.get("text", "")
    meta = entry.get("metadata", {})
    raw_id = meta.get("raw_id")
    page_list = entry.get("page_idx", [])

    if not text or not raw_id or not page_list:
        return idx, entry

    pages = sorted(set([p for p in page_list if isinstance(p, int)]))
    image_paths = []
    for p in pages:
        image_paths.append(picture_root / str(raw_id) / f"{p}.png")

    corrected = call_gpt_with_images(text, image_paths)

    new_entry = dict(entry)
    new_entry["text"] = corrected
    return idx, new_entry


def process_one_file(input_path: Path, output_path: Path, picture_root: Path):
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return

    out = [None] * len(data)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = [ex.submit(process_single_entry, (i, data[i], picture_root)) for i in range(len(data))]
        for fut in as_completed(futures):
            i, e = fut.result()
            out[i] = e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def process_all(input_dir: str, picture_root: str, output_dir: str, recursive: bool = False):
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    pic_root = Path(picture_root)

    files = sorted(in_root.rglob("*.json")) if recursive else sorted(in_root.glob("*.json"))
    for fp in files:
        out_path = out_root / (fp.stem.replace(".label", "") + ".calibrate.json")
        process_one_file(fp, out_path, pic_root)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("picture_root")
    ap.add_argument("output_dir")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    process_all(args.input_dir, args.picture_root, args.output_dir, recursive=args.recursive)
