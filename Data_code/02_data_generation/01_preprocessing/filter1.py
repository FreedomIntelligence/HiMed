import os
import json
import time
import traceback
import threading
from pathlib import Path
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
API_KEY = os.getenv('API_KEY', '')
API_BASE_URL = os.getenv('API_BASE_URL', '')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
MAX_RETRY = int(os.getenv('MAX_RETRY', '3'))
DEBUG = bool(int(os.getenv('DEBUG', '0')))

class GetOpenAI:
    _session = None
    _session_lock = threading.Lock()

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:
                    cls._session = requests.Session()
                    retry_strategy = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
                    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
                    cls._session.mount('http://', adapter)
                    cls._session.mount('https://', adapter)
        return cls._session

    @staticmethod
    def __gpt_api_stream(messages: list, model=MODEL_NAME):
        try:
            headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
            data = {'model': model, 'messages': messages}
            session = GetOpenAI._get_session()
            response = session.post(API_BASE_URL, headers=headers, json=data, timeout=(30, 120))
            if response.status_code != 200:
                return (False, f'HTTP {response.status_code}: {response.text[:200]}')
            try:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not content:
                    return (False, f'No content: {response.text[:200]}')
                return (True, content)
            except json.JSONDecodeError:
                return (False, f'Invalid JSON: {response.text[:200]}')
        except requests.exceptions.Timeout as err:
            return (False, f'Timeout: {err}')
        except requests.exceptions.ConnectionError as err:
            return (False, f'Connection error: {err}')
        except requests.exceptions.RequestException as err:
            return (False, f'Request error: {err}')
        except Exception as err:
            return (False, f'Exception: {err}')

    def get_respons(self, messages, model=MODEL_NAME):
        out = ''
        for attempt in range(MAX_RETRY):
            ok, out = self.__gpt_api_stream(messages, model=model)
            if ok:
                return (ok, (out or '').strip())
            if attempt < MAX_RETRY - 1:
                time.sleep(2 ** (attempt + 1))
        return (ok, (out or '').strip())
openai_tool = GetOpenAI()

def build_prompt(text: str) -> str:
    return f'\nYou are a classification tool. Analyze the following text and output exactly 4 lines:\n\nis_hindi: True/False\nis_medical: True/False\nhas_ambiguity: True/False\nis_title_or_heading: True/False\n\nRules:\n- Output ONLY these 4 lines.\n- No explanations.\n- Use Python-style booleans True/False.\n\nText:\n"""{text}"""\n'.strip()

def parse_boolean_token(token: str) -> bool:
    t = token.strip().lower()
    if t in ['true', '1', 'yes']:
        return True
    if t in ['false', '0', 'no']:
        return False
    return False

def parse_gpt_response(raw: str) -> dict:
    out = {'is_hindi': False, 'is_medical': False, 'has_ambiguity': False, 'is_title_or_heading': False}
    if not raw:
        return out
    for line in raw.splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        k = k.strip()
        v = v.strip()
        if k in out:
            out[k] = parse_boolean_token(v)
    return out

def call_gpt(prompt: str) -> str:
    messages = [{'role': 'user', 'content': prompt}]
    ok, result = openai_tool.get_respons(messages, model=MODEL_NAME)
    if ok:
        return result
    raise Exception(f'GPT API call failed: {result}')

def load_json_file(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def save_json_file(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def process_one_file(input_path: Path, output_path: Path):
    data = load_json_file(input_path)
    if not isinstance(data, list):
        return
    out = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        text = entry.get('text', '')
        prompt = build_prompt(text)
        raw = call_gpt(prompt)
        flags = parse_gpt_response(raw)
        new_entry = dict(entry)
        new_entry.update(flags)
        out.append(new_entry)
    save_json_file(out, output_path)

def process_all(input_dir: str, output_dir: str, recursive: bool=False):
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    files = sorted(in_root.rglob('*.json')) if recursive else sorted(in_root.glob('*.json'))
    for fp in tqdm(files, desc='Files'):
        rel = fp.relative_to(in_root)
        out_path = out_root / rel.parent / (fp.stem + '.analyze.json')
        process_one_file(fp, out_path)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('input_dir')
    ap.add_argument('output_dir')
    ap.add_argument('--recursive', action='store_true')
    args = ap.parse_args()
    process_all(args.input_dir, args.output_dir, recursive=args.recursive)
