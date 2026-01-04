import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import sys
import pandas as pd
from cot import classify_subject_and_type, generate_qa_for_entry, openai_tool, MODEL_NAME, sanitize_output_text, TYPES

def load_results_good(path: Path) -> List[Dict]:
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, list):
        raise ValueError(f'Invalid results_good JSON (not list): {path}')
    return obj

def normalize_page_range(block: Dict) -> List[int]:
    p = block.get('page_idx')
    if p is None:
        return []
    if isinstance(p, int):
        return [p]
    if isinstance(p, list):
        return [int(x) for x in p if isinstance(x, (int, float))]
    return []

def build_raw_id_mapping(blocks: List[Dict]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for blk in blocks:
        meta = blk.get('metadata') or {}
        rid = meta.get('raw_id')
        if not rid:
            continue
        if rid not in mapping:
            mapping[rid] = f'{len(mapping) + 1:03d}'
    if not mapping:
        mapping['<default>'] = '001'
    return mapping

def get_numeric_raw_id(meta: Dict, mapping: Dict[str, str]) -> Tuple[str, str]:
    rid_raw = (meta or {}).get('raw_id') or '<default>'
    rid3 = mapping.get(rid_raw)
    if rid3 is None:
        rid3 = f'{len(mapping) + 1:03d}'
        mapping[rid_raw] = rid3
    return (rid_raw, rid3)

TOPICS = ['diagnosis', 'etiology', 'medical knowledge', 'prognosis', 'treatment']
TYPES = ['MCQ', 'QA', 'Dialogue']

def _norm_subject_name(s: str) -> str:
    t = str(s or '').strip().lower()
    mapping = {'diagnosis': 'diagnosis', 'dx': 'diagnosis', 'treatment': 'treatment', 'therapy': 'treatment', 'management': 'treatment', 'etiology': 'etiology', 'aetiology': 'etiology', 'prognosis': 'prognosis', 'medical knowledge': 'medical knowledge', 'knowledge': 'medical knowledge', 'general': 'medical knowledge'}
    return mapping.get(t, t)

def _norm_qtype_name(s: str) -> str:
    t = str(s or '').strip().upper()
    if t in {'MCQA', 'MCQ'}:
        return 'MCQ'
    if t in TYPES:
        return t
    return ''

def load_templates_pool(xlsx_path: Path) -> Dict[Tuple[str, str], List[str]]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f'Template Excel not found: {xlsx_path}')
    df = pd.read_excel(xlsx_path)
    required_cols = ['Category']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f'Missing required column in template sheet: {c}. Existing columns: {list(df.columns)}')
    if 'Hindi' not in df.columns and 'English' not in df.columns:
        raise ValueError(f'Missing Hindi/English text column in template sheet. Existing columns: {list(df.columns)}')
    pool: Dict[Tuple[str, str], List[str]] = {}
    for subj_raw, sub_df in df.groupby('Category', sort=False):
        if not subj_raw:
            continue
        subj = _norm_subject_name(subj_raw)
        rows = sub_df.reset_index(drop=True)
        for i in range(len(rows)):
            qtxt = ''
            if 'Hindi' in rows.columns:
                qtxt = str(rows.at[i, 'Hindi'] or '').strip()
            if not qtxt and 'English' in rows.columns:
                qtxt = str(rows.at[i, 'English'] or '').strip()
            if not qtxt:
                continue
            if 'Type' in rows.columns:
                type_raw = str(rows.at[i, 'Type'] or '').strip()
                if type_raw.upper() in {'MCQA', 'MCQ'}:
                    tname = 'MCQ'
                elif type_raw.upper() in {'QA', 'Q&A'}:
                    tname = 'QA'
                elif type_raw.upper() in {'DIALOGUE', 'DIALOG'}:
                    tname = 'Dialogue'
                else:
                    slot = i % 9
                    if slot < 3:
                        tname = 'QA'
                    elif slot < 6:
                        tname = 'Dialogue'
                    else:
                        tname = 'MCQ'
            else:
                slot = i % 9
                if slot < 3:
                    tname = 'QA'
                elif slot < 6:
                    tname = 'Dialogue'
                else:
                    tname = 'MCQ'
            key = (subj, tname)
            pool.setdefault(key, []).append(qtxt)
    if not pool:
        raise ValueError('Template sheet parsed to an empty pool. Please check inspool_HI.xlsx content.')
    return pool

def get_question_template(pool: Dict[Tuple[str, str], List[str]], subject: str, qtype: str) -> str:
    subj = _norm_subject_name(subject)
    tname = _norm_qtype_name(qtype)

    def _collect(keys):
        out: List[str] = []
        for k in keys:
            out.extend(pool.get(k, []))
        return out
    cand = _collect([(subj, tname)])
    if not cand:
        cand = _collect([(subj, '')])
    if not cand:
        cand = _collect([('medical knowledge', tname), ('medical knowledge', '')])
    if not cand:
        all_tpls: List[str] = []
        for lst in pool.values():
            all_tpls.extend(lst)
        cand = all_tpls
    return random.choice(cand) if cand else ''

def load_examples_pool(xlsx_path: Path) -> Dict[Tuple[str, str], List[Dict]]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f'Examples Excel not found: {xlsx_path}')
    df = pd.read_excel(xlsx_path)
    required_cols = ['Category', 'Question', 'Answer']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f'Missing required column in few-shot sheet: {c}. Existing columns: {list(df.columns)}')
    pool: Dict[Tuple[str, str], List[Dict]] = {}
    for i in range(len(df)):
        subj_raw = df.at[i, 'Category']
        if not subj_raw:
            continue
        subj = _norm_subject_name(subj_raw)
        qtxt = str(df.at[i, 'Question'] or '').strip()
        atxt = str(df.at[i, 'Answer'] or '').strip()
        if not (qtxt and atxt):
            continue
        para = str(df.at[i, 'Paragraph'] or '').strip() if 'Paragraph' in df.columns else ''
        cot = str(df.at[i, 'Reasoning'] or '').strip() if 'Reasoning' in df.columns else ''
        qtype_raw = str(df.at[i, 'Type'] or '').strip() if 'Type' in df.columns else ''
        tname = _norm_qtype_name(qtype_raw)
        key = (subj, tname)
        pool.setdefault(key, []).append({'paragraph': para, 'q': qtxt, 'a': atxt, 'cot': cot})
    return pool

def get_few_shot(pool: Dict[Tuple[str, str], List[Dict]], subject: str, qtype: str, k: int=3) -> List[Dict]:
    subj = _norm_subject_name(subject)
    tname = _norm_qtype_name(qtype)

    def _collect(keys):
        out: List[Dict] = []
        for key in keys:
            out.extend(pool.get(key, []))
        return out
    cand = _collect([(subj, tname)])
    if not cand:
        cand = _collect([(subj, '')])
    if not cand:
        cand = _collect([('medical knowledge', tname), ('medical knowledge', '')])
    random.shuffle(cand)
    return cand[:k]

def process_single_text_id(blk: Dict, idx: int, rid_map: Dict[str, str], processed_text_ids: set, text_id_entry_count: Dict[str, int], templ_pool: Dict[Tuple[str, str], List[str]], ex_pool: Dict[Tuple[str, str], List[Dict]]) -> Tuple[str, List[Dict], bool]:
    text = (blk.get('text') or '').strip()
    if not text:
        return (None, [], True)
    page_range = normalize_page_range(blk)
    meta_raw = blk.get('metadata') or {}
    raw_id_str, raw_id3 = get_numeric_raw_id(meta_raw, rid_map)
    text_id = f'{idx:05d}'
    subject_list, _type_list = classify_subject_and_type(text)
    type_list = TYPES[:]
    expected_entry_count = len(subject_list) * len(type_list) * 3
    if text_id in processed_text_ids:
        actual_count = text_id_entry_count.get(text_id, 0)
        if actual_count >= expected_entry_count:
            return (text_id, [], True)
    text_entries: List[Dict] = []
    entry_counter = 0
    for subject in subject_list:
        for qtype in type_list:
            qtpl = get_question_template(templ_pool, subject, qtype)
            fs = get_few_shot(ex_pool, subject, qtype)
            base_entry = {'text': text, 'subject': subject, 'type': qtype, 'question_templete': qtpl, 'few_shot': fs, 'metadata': {'raw_id': raw_id3, 'text_id': text_id, 'page_range': page_range, 'raw_id_str': raw_id_str}}
            qa_list = generate_qa_for_entry(base_entry)
            entry_counter += 1
            base_entry_id = f'{entry_counter:02d}'
            for qa_idx, qa in enumerate(qa_list):
                difficulty = qa.get('difficulty', '').lower() or 'unknown'
                entry_id = f'{base_entry_id}{qa_idx:01d}'
                _id = f'{raw_id3}{text_id}{entry_id}'
                entry = {'id': _id, **{k: v for k, v in base_entry.items() if k != 'few_shot'}, 'metadata': {**base_entry['metadata'], 'entry_id': entry_id, 'difficulty': difficulty}, 'question': sanitize_output_text(qa.get('question', '')), 'answer': sanitize_output_text(qa.get('answer', '')), 'cot': sanitize_output_text(qa.get('cot', ''))}
                entry.pop('few_shot', None)
                entry['llmjudge'] = {'grounded_in_context': 0.0, 'medical_correctness': 0.0, 'reasoning_clarity': 0.0, 'language_quality': 0.0}
                text_entries.append(entry)
    print(f'  - text_id={text_id}, subjects={subject_list}, types={type_list}, entries={entry_counter}, total_entries={len(text_entries)}')
    return (text_id, text_entries, False)

def process_results_good(src: Path, dst: Path, instr_xlsx_path: Path, examples_xlsx_path: Path, sleep_between_req: float=0.5, parallel_workers: int=100):
    file_name = src.stem.replace('.good', '')
    print(f'[FILE] {file_name}')
    print(f'[INFO] load good json: {src}')
    blocks = load_results_good(src)
    print(f'[INFO] found {len(blocks)} blocks')
    if not instr_xlsx_path.exists():
        raise FileNotFoundError(f'Template Excel not found: {instr_xlsx_path}')
    if not examples_xlsx_path.exists():
        raise FileNotFoundError(f'Examples Excel not found: {examples_xlsx_path}')
    templ_pool = load_templates_pool(instr_xlsx_path)
    ex_pool = load_examples_pool(examples_xlsx_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    existing_entries: List[Dict] = []
    processed_text_ids = set()
    text_id_entry_count = {}
    if dst.exists() and dst.stat().st_size > 0:
        try:
            existing_entries = json.loads(dst.read_text(encoding='utf-8'))
            for ent in existing_entries:
                tid = (ent.get('metadata') or {}).get('text_id') or ''
                if tid:
                    processed_text_ids.add(tid)
                    text_id_entry_count[tid] = text_id_entry_count.get(tid, 0) + 1
            print(f'[INFO] resume mode: loaded {len(existing_entries)} existing entries, processed text_id count={len(processed_text_ids)}')
        except Exception as e:
            print(f'[WARN] failed to load existing output {dst}, will regenerate from scratch. error={e}')
            existing_entries = []
            processed_text_ids = set()
            text_id_entry_count = {}
    rid_map = build_raw_id_mapping(blocks)
    print(f'[INFO] raw_id mapping: {rid_map}')
    all_entries: List[Dict] = list(existing_entries)
    write_lock = threading.Lock()
    text_id_subjects = {}
    for ent in existing_entries:
        tid = (ent.get('metadata') or {}).get('text_id') or ''
        if tid:
            subject = ent.get('subject', '')
            if subject:
                if tid not in text_id_subjects:
                    text_id_subjects[tid] = set()
                text_id_subjects[tid].add(subject)
    blocks_to_process = []
    skipped_count = 0
    incomplete_count = 0
    for idx, blk in enumerate(blocks, start=1):
        text = (blk.get('text') or '').strip()
        if not text:
            continue
        text_id = f'{idx:05d}'
        if text_id in processed_text_ids:
            actual_count = text_id_entry_count.get(text_id, 0)
            known_subjects = text_id_subjects.get(text_id, set())
            if known_subjects:
                expected_entry_count = len(known_subjects) * len(TYPES) * 3
                if actual_count >= expected_entry_count:
                    skipped_count += 1
                    continue
                else:
                    all_entries = [e for e in all_entries if (e.get('metadata') or {}).get('text_id') != text_id]
                    incomplete_count += 1
                    print(f'[INFO] [{file_name}] text_id {text_id} incomplete ({actual_count}/{expected_entry_count}), reprocessing...')
            else:
                min_expected = len(TYPES) * 3
                if actual_count >= min_expected and actual_count % 9 == 0:
                    pass
                elif actual_count > 0:
                    all_entries = [e for e in all_entries if (e.get('metadata') or {}).get('text_id') != text_id]
                    incomplete_count += 1
                    print(f'[INFO] [{file_name}] text_id {text_id} incomplete ({actual_count} entries, subject unknown), reprocessing...')
        blocks_to_process.append((idx, blk))
    if skipped_count > 0:
        print(f'[INFO] [{file_name}] skipped completed text_id: {skipped_count}')
    if incomplete_count > 0:
        print(f'[INFO] [{file_name}] found incomplete text_id: {incomplete_count}; will reprocess')
    print(f'[INFO] [{file_name}] to process: {len(blocks_to_process)} text_id')
    if parallel_workers > 1 and len(blocks_to_process) > 1:
        print(f'[INFO] [{file_name}] parallel mode, workers: {parallel_workers}')

        def process_and_save(block_data):
            idx, blk = block_data
            text_id, text_entries, should_skip = process_single_text_id(blk, idx, rid_map, processed_text_ids, text_id_entry_count, templ_pool, ex_pool)
            if should_skip or not text_entries:
                return (text_id, [], should_skip)
            with write_lock:
                all_entries[:] = [e for e in all_entries if (e.get('metadata') or {}).get('text_id') != text_id]
                all_entries.extend(text_entries)
                dst.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding='utf-8')
            return (text_id, text_entries, should_skip)

        completed_count = 0
        batch_results = []
        batch_number = 1
        REPORT_BATCH_SIZE = 100
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_block = {executor.submit(process_and_save, block_data): block_data for block_data in blocks_to_process}
            for future in as_completed(future_to_block):
                try:
                    text_id, text_entries, should_skip = future.result()
                    completed_count += 1
                    if not should_skip and text_entries:
                        batch_results.append((text_id, len(text_entries)))
                        if len(batch_results) >= REPORT_BATCH_SIZE:
                            batch_entries_sum = sum((entries for _, entries in batch_results))
                            print(f'[{file_name}] batch {batch_number}: done {len(batch_results)} text_id, total {batch_entries_sum} entries, progress {completed_count}/{len(blocks_to_process)}')
                            batch_results = []
                            batch_number += 1
                except Exception as e:
                    block_data = future_to_block[future]
                    print(f'[ERROR] [{file_name}] failed at block {block_data[0]}: {e}')
            if batch_results:
                batch_entries_sum = sum((entries for _, entries in batch_results))
                print(f'[{file_name}] batch {batch_number}: done {len(batch_results)} text_id, total {batch_entries_sum} entries, progress {completed_count}/{len(blocks_to_process)} (all done)')
    else:
        print(f'[INFO] [{file_name}] serial mode')
        completed_count = 0
        for idx, blk in blocks_to_process:
            text_id, text_entries, should_skip = process_single_text_id(blk, idx, rid_map, processed_text_ids, text_id_entry_count, templ_pool, ex_pool)
            if should_skip:
                continue
            completed_count += 1
            if text_id in processed_text_ids:
                all_entries = [e for e in all_entries if (e.get('metadata') or {}).get('text_id') != text_id]
            all_entries.extend(text_entries)
            dst.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'[PROGRESS] [{file_name}] {completed_count}/{len(blocks_to_process)} - text_id {text_id} done ({len(text_entries)} entries)')
        time.sleep(sleep_between_req)
    print(f'[OK] wrote step6 json: {dst} (entries: {len(all_entries)})')

def main():
    parser = argparse.ArgumentParser(description='Convert one results_good *.good.json into step6 JSON.')
    parser.add_argument('--input', type=str, required=True, help='Path to *.good.json')
    parser.add_argument('--output', type=str, required=True, help='Path to output step6 JSON')
    parser.add_argument('--instr-xlsx', type=str, required=True, help='Path to instruction template Excel file')
    parser.add_argument('--examples-xlsx', type=str, required=True, help='Path to few-shot examples Excel file')
    parser.add_argument('--sleep-between-req', type=float, default=0.5, help='Sleep time between requests (default: 0.5)')
    parser.add_argument('--parallel-workers', type=int, default=100, help='Number of parallel workers (default: 100)')
    args = parser.parse_args()
    input_json = Path(args.input)
    output_json = Path(args.output)
    instr_xlsx_path = Path(args.instr_xlsx)
    examples_xlsx_path = Path(args.examples_xlsx)
    process_results_good(src=input_json, dst=output_json, instr_xlsx_path=instr_xlsx_path, examples_xlsx_path=examples_xlsx_path, sleep_between_req=args.sleep_between_req, parallel_workers=args.parallel_workers)

if __name__ == '__main__':
    main()