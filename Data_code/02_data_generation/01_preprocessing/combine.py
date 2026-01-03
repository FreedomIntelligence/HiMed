import json
import argparse
from pathlib import Path
from tqdm import tqdm
FIELD_IS_HINDI = 'is_hindi'
FIELD_IS_MEDICAL = 'is_medical'
FIELD_HAS_AMBIGUITY = 'has_ambiguity'
FIELD_IS_TITLE = 'is_title_or_heading'

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_group(entries):
    if not entries:
        return None
    first = entries[0].copy()
    texts = []
    for e in entries:
        t = (e.get('text') or '').strip()
        if t:
            texts.append(t)
    merged_text = ' '.join(texts)
    first['text'] = merged_text
    first[FIELD_IS_HINDI] = bool(first.get(FIELD_IS_HINDI, False))
    first[FIELD_IS_MEDICAL] = any((bool(e.get(FIELD_IS_MEDICAL, False)) for e in entries))
    first[FIELD_HAS_AMBIGUITY] = all((bool(e.get(FIELD_HAS_AMBIGUITY, False)) for e in entries))
    first[FIELD_IS_TITLE] = all((bool(e.get(FIELD_IS_TITLE, False)) for e in entries))
    page_idx = []
    for e in entries:
        if isinstance(e.get('page_idx'), list):
            page_idx.extend(e['page_idx'])
    if page_idx:
        first['page_idx'] = sorted(set(page_idx))
    return first

def process_file(input_path: Path, output_path: Path):
    data = load_json(input_path)
    if not isinstance(data, list):
        print(f'Skipping invalid file: {input_path}')
        return False
    entries = data
    n = len(entries)
    if n == 0:
        save_json([], output_path)
        return True
    merged_entries = []
    i = n - 1
    while i >= 0:
        cur = entries[i]
        if not bool(cur.get(FIELD_HAS_AMBIGUITY, False)):
            merged_entries.append(cur)
            i -= 1
        else:
            end_idx = i
            j = i - 1
            while j >= 0 and bool(entries[j].get(FIELD_HAS_AMBIGUITY, False)):
                j -= 1
            if j < 0:
                start_idx = 0
            else:
                start_idx = j
            group = entries[start_idx:end_idx + 1]
            merged = merge_group(group)
            merged_entries.append(merged)
            i = start_idx - 1
    merged_entries.reverse()
    save_json(merged_entries, output_path)
    return True

def process_all(input_folder: str, output_folder: str):
    input_root = Path(input_folder)
    output_root = Path(output_folder)
    if not input_root.is_dir():
        print('Input folder not found:', input_folder)
        return
    json_files = sorted(input_root.rglob('*.json'))
    if not json_files:
        print('No JSON files found.')
        return
    print(f'Found {len(json_files)} JSON files.\n')
    for input_file in tqdm(json_files, desc='Files'):
        rel = input_file.relative_to(input_root)
        stem1 = input_file.stem
        if stem1.endswith('.analyze'):
            base = stem1[:-8]
        else:
            base = stem1
        output_name = base + '.combine.json'
        output_file = output_root / rel.parent / output_name
        process_file(input_file, output_file)
    print('\nAll combining finished.')

def main():
    parser = argparse.ArgumentParser(description='Combine analyzed JSON files by merging ambiguous entries')
    parser.add_argument('--input-folder', type=str, required=True, help='Folder containing analyzed JSON files')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder for combined JSON files')
    args = parser.parse_args()
    process_all(args.input_folder, args.output_folder)
if __name__ == '__main__':
    main()
