import json
import argparse
from pathlib import Path
from tqdm import tqdm

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def is_ttff(entry: dict) -> bool:
    FIELD_IS_HINDI = 'is_hindi'
    FIELD_IS_MEDICAL = 'is_medical'
    FIELD_HAS_AMBIGUITY = 'has_ambiguity'
    FIELD_IS_TITLE = 'is_title_or_heading'
    is_hindi = bool(entry.get(FIELD_IS_HINDI, False))
    is_medical = bool(entry.get(FIELD_IS_MEDICAL, False))
    has_ambiguity = bool(entry.get(FIELD_HAS_AMBIGUITY, False))
    is_title = bool(entry.get(FIELD_IS_TITLE, False))
    return is_hindi is True and is_medical is True and (has_ambiguity is False) and (is_title is False)

def get_base_name(input_file: Path) -> str:
    stem = input_file.stem
    for suffix in ['.combine', '.analyze', '.filtered']:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem

def process_file(input_path: Path, picked_path: Path, abandoned_path: Path):
    data = load_json(input_path)
    if not isinstance(data, list):
        print(f'Skipping invalid file (top-level is not list): {input_path}')
        return False
    picked = []
    abandoned = []
    for entry in data:
        if is_ttff(entry):
            picked.append(entry)
        else:
            abandoned.append(entry)
    save_json(picked, picked_path)
    save_json(abandoned, abandoned_path)
    print(f'{input_path.name}: picked={len(picked)}, abandoned={len(abandoned)}, total={len(data)}')
    return True

def process_all(input_folder: str, picked_folder: str, abandoned_folder: str):
    input_root = Path(input_folder)
    picked_root = Path(picked_folder)
    abandoned_root = Path(abandoned_folder)
    if not input_root.is_dir():
        print('Input folder not found:', input_folder)
        return
    json_files = sorted(input_root.rglob('*.json'))
    if not json_files:
        print('No JSON files found in input folder.')
        return
    print(f'Found {len(json_files)} JSON files.')
    print('Starting picking...\n')
    for input_file in tqdm(json_files, desc='Files'):
        rel = input_file.relative_to(input_root)
        base = get_base_name(input_file)
        picked_name = base + '.pick.json'
        abandoned_name = base + '.abandon.json'
        picked_path = picked_root / rel.parent / picked_name
        abandoned_path = abandoned_root / rel.parent / abandoned_name
        process_file(input_file, picked_path, abandoned_path)
    print('\nAll picking done.')

def main():
    parser = argparse.ArgumentParser(description='Pick entries from JSON files based on boolean flags')
    parser.add_argument('--input-folder', type=str, required=True, help='Folder containing input JSON files')
    parser.add_argument('--picked-folder', type=str, required=True, help='Folder for picked entries (TTFF)')
    parser.add_argument('--abandoned-folder', type=str, required=True, help='Folder for abandoned entries (not TTFF)')
    args = parser.parse_args()
    process_all(input_folder=args.input_folder, picked_folder=args.picked_folder, abandoned_folder=args.abandoned_folder)
if __name__ == '__main__':
    main()
