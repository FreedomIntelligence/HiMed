import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Any

def strip_image_lines(text: str) -> str:
    img_line_pattern = '^\\s*!\\[[^\\]]*\\]\\(\\s*[^)]+\\s*\\)\\s*$'
    img_line_rx = re.compile(img_line_pattern, re.IGNORECASE)
    lines = text.splitlines()
    kept = [ln for ln in lines if not img_line_rx.match(ln)]
    return '\n'.join(kept) + '\n'

def split_pages(text: str) -> List[str]:
    page_split_mark = '\\s*<--- Page Split --->\\s*'
    page_split_rx = re.compile(page_split_mark, re.UNICODE)
    parts = page_split_rx.split(text)
    return parts if parts else ['']

def lines_to_paragraphs(lines: List[str]) -> List[str]:
    paras, buf = ([], [])
    for ln in lines:
        if ln.strip():
            buf.append(ln.strip())
        elif buf:
            paras.append(' '.join(buf).strip())
            buf = []
    if buf:
        paras.append(' '.join(buf).strip())
    return paras

def page_text_to_blocks(page_text: str, page_idx: int) -> List[dict]:
    cleaned = strip_image_lines(page_text)
    max_blank_lines = 3
    blanks_rx = re.compile('\\n{' + str(max_blank_lines) + ',}', re.UNICODE)
    cleaned = blanks_rx.sub('\n\n', cleaned)
    lines = cleaned.splitlines()
    paras = lines_to_paragraphs(lines)
    blocks = []
    for p in paras:
        if not p.strip():
            continue
        blocks.append({'type': 'paragraph', 'text': p, 'page_idx': [page_idx]})
    return blocks

def mmd_to_json_structure(raw_text: str) -> List[List[dict]]:
    pages_raw = split_pages(raw_text)
    pages_json: List[List[dict]] = []
    for page_num, pg_text in enumerate(pages_raw, 1):
        blocks = page_text_to_blocks(pg_text, page_idx=page_num)
        pages_json.append(blocks)
    return pages_json

def normalize_page_idx_structure(pages: List[List[dict]]) -> None:
    for page in pages:
        for block in page:
            if 'page_idx' not in block:
                continue
            v: Any = block['page_idx']
            if isinstance(v, list):
                new_list = []
                for elem in v:
                    try:
                        new_list.append(int(elem))
                    except Exception:
                        pass
                block['page_idx'] = new_list
            elif isinstance(v, int):
                block['page_idx'] = [v]
            else:
                try:
                    iv = int(v)
                    block['page_idx'] = [iv]
                except Exception:
                    block['page_idx'] = []

def get_root_mmd_files(input_dir: Path) -> List[Path]:
    mmd_files: List[Path] = []
    for p in input_dir.glob('*'):
        if p.is_file() and p.suffix.lower() == '.mmd' and (not p.name.lower().endswith('_det.mmd')):
            mmd_files.append(p)
    return sorted(mmd_files)

def get_output_path(src_file: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{src_file.stem}.json'

def process_one_mmd(src_file: Path, output_dir: Path, force_overwrite: bool) -> bool:
    dst_file = get_output_path(src_file, output_dir)
    if dst_file.exists() and (not force_overwrite):
        try:
            if dst_file.stat().st_size > 0:
                print(f'[SKIP] Existing result found: {dst_file.name}')
                return True
        except Exception as e:
            print(f'[WARN] Failed to stat existing file: {e}')
    try:
        raw_text = src_file.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raw_text = None
        fallback_encodings = ('utf-8-sig', 'utf-16', 'gb18030')
        for enc in fallback_encodings:
            try:
                raw_text = src_file.read_text(encoding=enc)
                print(f'[INFO] Read with encoding {enc}: {src_file.name}')
                break
            except Exception:
                continue
        if raw_text is None:
            print(f'[ERROR] Failed to read file with known encodings: {src_file.name}')
            return False
    try:
        pages_json = mmd_to_json_structure(raw_text)
        normalize_page_idx_structure(pages_json)
        print(f'[INFO] Parsed {src_file.name}: {len(pages_json)} pages (aligned with original PDF page count)')
        dst_file.write_text(json.dumps(pages_json, ensure_ascii=False, indent=4), encoding='utf-8')
        print(f'[OK] Processed: {src_file.name} -> {dst_file.name}')
        return True
    except Exception as e:
        print(f'[ERROR] Failed to process {src_file.name}: {e}')
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert MMD files to JSON structure')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing MMD files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for JSON files')
    parser.add_argument('--force-overwrite', action='store_true', help='Force overwrite existing output files')
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    force_overwrite = args.force_overwrite
    if not input_dir.exists() or not input_dir.is_dir():
        print(f'[FATAL] Input directory does not exist or is not a directory: {input_dir}')
        return
    mmd_files = get_root_mmd_files(input_dir)
    if not mmd_files:
        print(f'[INFO] No .mmd files found in {input_dir} (skipping *_det.mmd).')
        return
    print(f'[INFO] Found {len(mmd_files)} .mmd files to process')
    print('-' * 50)
    success_count = 0
    fail_count = 0
    for idx, file in enumerate(mmd_files, 1):
        print(f'\n[FILE {idx}/{len(mmd_files)}] Processing: {file.name}')
        if process_one_mmd(file, output_dir, force_overwrite):
            success_count += 1
        else:
            fail_count += 1
    print('\n' + '=' * 50)
    print('=== Summary ===')
    print(f'Total files : {len(mmd_files)}')
    print(f'Success     : {success_count}')
    print(f'Failed      : {fail_count}')
    print(f'Output dir  : {output_dir}')
    print('=' * 50)
if __name__ == '__main__':
    main()
