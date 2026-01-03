import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
OPEN_TO_CLOSE = {'(': ')', '（': '）', '[': ']', '【': '】', '{': '}', '｛': '｝'}
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}

def clean_edges_hash_and_spaces(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub('^[#\\s]+', '', s)
    s = re.sub('[#\\s]+$', '', s)
    return s

def remove_unmatched_brackets(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    chars = []
    stack = []
    for ch in s:
        if ch in OPEN_TO_CLOSE:
            stack.append((ch, len(chars)))
            chars.append(ch)
        elif ch in CLOSE_TO_OPEN:
            if stack and stack[-1][0] == CLOSE_TO_OPEN[ch]:
                stack.pop()
                chars.append(ch)
            else:
                continue
        else:
            chars.append(ch)
    if not stack:
        return ''.join(chars)
    remove_indices = {pos for _, pos in stack}
    return ''.join((ch for i, ch in enumerate(chars) if i not in remove_indices))

def fix_spaces_around_brackets(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    s = re.sub('([(\\[{\\（【｛])\\s+', '\\1', s)
    s = re.sub('\\s+([)\\]\\}）】｝])', '\\1', s)
    return s

def collapse_multiple_spaces(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    return re.sub(' {2,}', ' ', s)

def clean_text(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    s = clean_edges_hash_and_spaces(s)
    s = remove_unmatched_brackets(s)
    s = fix_spaces_around_brackets(s)
    s = collapse_multiple_spaces(s)
    return s

def flatten_entries(data: Any) -> List[Dict[str, Any]]:
    flat = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                for e in item:
                    if isinstance(e, dict):
                        flat.append(e)
            elif isinstance(item, dict):
                flat.append(item)
    elif isinstance(data, dict):
        flat.append(data)
    return flat

def process_one_json(src_file: Path, out_dir: Path, force_overwrite: bool, suffix: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_file = out_dir / f'{src_file.stem}.clean{suffix}'
    if dst_file.exists() and (not force_overwrite):
        print(f'[SKIP] {dst_file.name} exists')
        return True
    try:
        raw = src_file.read_text(encoding='utf-8')
        data = json.loads(raw)
    except Exception as e:
        print(f'[ERROR] Cannot read {src_file.name}: {e}')
        return False
    entries = flatten_entries(data)
    raw_id = src_file.stem.zfill(3)
    cleaned = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        new_entry.pop('type', None)
        new_entry['text'] = clean_text(entry.get('text', ''))
        new_entry['metadata'] = {'raw_id': raw_id}
        cleaned.append(new_entry)
    try:
        dst_file.write_text(json.dumps(cleaned, ensure_ascii=False, indent=4), encoding='utf-8')
        print(f'[OK] {src_file.name} → {dst_file.name} (entries={len(cleaned)})')
        return True
    except Exception as e:
        print(f'[ERROR] Cannot write {dst_file.name}: {e}')
        return False

def main():
    parser = argparse.ArgumentParser(description="Clean JSON files by removing 'type' field, cleaning text, and adding metadata")
    parser.add_argument('--input-dir', type=str, required=True, help='Path to input directory containing JSON files')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory for cleaned JSON files')
    parser.add_argument('--force-overwrite', action='store_true', help='Force overwrite existing output files')
    parser.add_argument('--suffix', type=str, default='.json', help='File suffix to filter (default: .json)')
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    force_overwrite = args.force_overwrite
    suffix = args.suffix
    json_files = sorted((p for p in input_dir.glob('*') if p.is_file() and p.suffix.lower() == suffix))
    print(f'[INFO] {len(json_files)} JSON files found')
    ok = 0
    fail = 0
    for i, jf in enumerate(json_files, 1):
        print(f'\n[FILE {i}/{len(json_files)}] {jf.name}')
        if process_one_json(jf, output_dir, force_overwrite, suffix):
            ok += 1
        else:
            fail += 1
    print('\n=== SUMMARY ===')
    print(f'Success : {ok}')
    print(f'Fail    : {fail}')
    print(f'Output  : {output_dir}')
if __name__ == '__main__':
    main()
