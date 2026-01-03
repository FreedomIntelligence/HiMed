import json
from pathlib import Path
from typing import List, Dict, Any

def extract_core_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {'text': entry.get('text', ''), 'page_idx': entry.get('page_idx', []), 'metadata': entry.get('metadata', {})}

def process_one_label_file(label_path: Path, out_good_dir: Path, out_mid_dir: Path, out_bad_dir: Path) -> None:
    raw_id = label_path.name.replace('.label.json', '')
    data: List[Dict[str, Any]] = json.loads(label_path.read_text(encoding='utf-8'))
    good_items: List[Dict[str, Any]] = []
    mid_items: List[Dict[str, Any]] = []
    bad_items: List[Dict[str, Any]] = []
    for entry in data:
        label = entry.get('label', '').upper().strip()
        core = extract_core_fields(entry)
        if label == 'NO_PROBLEM':
            good_items.append(core)
        elif label == 'POSSIBLE_ISSUE':
            mid_items.append(core)
        elif label == 'DEFINITE_ISSUE':
            bad_items.append(core)
        else:
            bad_items.append(core)
    out_good_path = out_good_dir / f'{raw_id}.good.json'
    out_mid_path = out_mid_dir / f'{raw_id}.middle.json'
    out_bad_path = out_bad_dir / f'{raw_id}.bad.json'
    out_good_path.write_text(json.dumps(good_items, ensure_ascii=False, indent=2), encoding='utf-8')
    out_mid_path.write_text(json.dumps(mid_items, ensure_ascii=False, indent=2), encoding='utf-8')
    out_bad_path.write_text(json.dumps(bad_items, ensure_ascii=False, indent=2), encoding='utf-8')

def distribute_all(labeled_dir: Path, good_dir: Path, mid_dir: Path, bad_dir: Path, recursive: bool=False) -> None:
    good_dir.mkdir(parents=True, exist_ok=True)
    mid_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(labeled_dir.rglob('*.label.json')) if recursive else sorted(labeled_dir.glob('*.label.json'))
    for lf in label_files:
        process_one_label_file(lf, good_dir, mid_dir, bad_dir)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('labeled_dir')
    ap.add_argument('good_dir')
    ap.add_argument('middle_dir')
    ap.add_argument('bad_dir')
    ap.add_argument('--recursive', action='store_true')
    args = ap.parse_args()
    distribute_all(Path(args.labeled_dir), Path(args.good_dir), Path(args.middle_dir), Path(args.bad_dir), recursive=args.recursive)
