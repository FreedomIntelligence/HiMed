import json
import copy
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

def load_json_file(file_path: Path):
    return json.loads(file_path.read_text(encoding='utf-8'))

def save_json_file(data, file_path: Path) -> bool:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return True

def _merge_page_idx(idx1, idx2):
    pages = []
    for idx in (idx1, idx2):
        if isinstance(idx, list):
            pages.extend(idx)
        elif isinstance(idx, int):
            pages.append(idx)
    return sorted(set(pages))

def merge_similar_entries_in_document(entries, model, threshold=0.65):
    if not entries:
        return (entries, [])
    list_item_pattern = re.compile('^\\s*(\\d+\\s*\\.|\\(\\s*[a-zA-Z\\u0900-\\u097F]+\\s*\\)|[\\w\\s]+—)', re.UNICODE)
    items = []
    for i, item in enumerate(entries):
        text = (item.get('text') or '').strip()
        if text:
            items.append({'index': i, 'text': text, 'item': item})
    if len(items) < 2:
        return (entries, [])
    texts = [p['text'] for p in items]
    embeddings = model.encode(texts, convert_to_tensor=True)
    merge_logs = []
    result_data = copy.deepcopy(entries)
    current_indices = {p['index']: p['index'] for p in items}
    i = len(items) - 1
    while i > 0:
        current_item = items[i]
        prev_item = items[i - 1]
        current_text = current_item['text']
        prev_text = prev_item['text']
        is_list_item = list_item_pattern.match(current_text) or list_item_pattern.match(prev_text)
        if is_list_item:
            i -= 1
            continue
        current_idx_in_result = current_indices[current_item['index']]
        prev_idx_in_result = current_indices[prev_item['index']]
        similarity = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
        if similarity >= threshold:
            merged_text = ((result_data[prev_idx_in_result].get('text') or '').strip() + ' ' + (result_data[current_idx_in_result].get('text') or '').strip()).strip()
            prev_page_idx = result_data[prev_idx_in_result].get('page_idx')
            current_page_idx = result_data[current_idx_in_result].get('page_idx')
            merged_page_idx = _merge_page_idx(prev_page_idx, current_page_idx)
            merge_logs.append({'similarity': round(similarity, 3), 'from_index': current_item['index'], 'to_index': prev_item['index'], 'final_page_idx': merged_page_idx})
            result_data[prev_idx_in_result]['text'] = merged_text
            result_data[prev_idx_in_result]['page_idx'] = merged_page_idx
            del result_data[current_idx_in_result]
            for p in items:
                if p['index'] > current_item['index']:
                    current_indices[p['index']] -= 1
        i -= 1
    return (result_data, list(reversed(merge_logs)))

def process_one_file(input_path: Path, output_path: Path, threshold: float):
    data = load_json_file(input_path)
    if not isinstance(data, list):
        return
    model = SentenceTransformer('all-MiniLM-L6-v2')
    merged, _ = merge_similar_entries_in_document(data, model, threshold=threshold)
    save_json_file(merged, output_path)

def process_all(input_dir: str, output_dir: str, recursive: bool, threshold: float):
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    files = sorted(in_root.rglob('*.json')) if recursive else sorted(in_root.glob('*.json'))
    for fp in files:
        rel = fp.relative_to(in_root)
        out_name = fp.stem.replace('.clean', '') + '.cluster.json'
        out_path = out_root / rel.parent / out_name
        process_one_file(fp, out_path, threshold=threshold)
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('input_dir')
    ap.add_argument('output_dir')
    ap.add_argument('--recursive', action='store_true')
    ap.add_argument('--threshold', type=float, default=0.65)
    args = ap.parse_args()
    process_all(args.input_dir, args.output_dir, recursive=args.recursive, threshold=args.threshold)
