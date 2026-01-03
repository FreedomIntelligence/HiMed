#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import argparse
import threading
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from cot import openai_tool, MODEL_NAME

def judge_entry(entry: dict) -> dict[str, float]:
    """调用 GPT 对单条样本打分，返回 llmjudge dict。"""
    JUDGE_PROMPT_BASE = """You are an expert data quality rater for a medical Q-A-CoT dataset in traditional medicine.

You will be given one JSON object with fields:
- text: source paragraph (in Hindi)
- subject: one of ["diagnosis","etiology","medical knowledge","prognosis","treatment"]
- type: one of ["MCQ","QA","Dialogue"]
- question: generated question
- answer: generated answer
- cot: chain-of-thought reasoning

You must evaluate the question/answer/cot on four dimensions, each scored from 0.00 to 1.00:

1. **grounded_in_context** (生成的内容是否完全基于原文信息):
   - Score 1.00 if all information in question/answer/cot is directly derivable from the source text
   - Score lower if there is any hallucinated information or external knowledge not present in the text
   - Score 0.00 if the content is completely unrelated to the source text

2. **medical_correctness** (医学回答是否正确(基于原文)):
   - Score 1.00 if the medical information is correct according to the source text
   - Score lower if there are minor inaccuracies or misinterpretations
   - Score 0.00 if the medical information is incorrect or contradicts the source text

3. **reasoning_clarity** (推理步骤是否合理清晰):
   - Score 1.00 if the reasoning steps are logical, clear, and well-structured
   - Score lower if reasoning is somewhat unclear or has minor logical gaps
   - Score 0.00 if reasoning is illogical, confusing, or missing

4. **language_quality** (印地语是否流畅自然):
   - Score 1.00 if the Hindi language is fluent, natural, and grammatically correct
   - Score lower if there are minor grammatical errors or awkward phrasing
   - Score 0.00 if the language is severely broken or incomprehensible

Return ONLY a JSON object with 4 float scores in [0.00, 1.00]:
{
  "grounded_in_context": 0.00,
  "medical_correctness": 0.00,
  "reasoning_clarity": 0.00,
  "language_quality": 0.00
}

Do NOT add any explanations or extra keys. Return only the JSON object.
"""
    
    payload = {
        "text": entry.get("text", ""),
        "subject": entry.get("subject", ""),
        "type": entry.get("type", ""),
        "question": entry.get("question", ""),
        "answer": entry.get("answer", ""),
        "cot": entry.get("cot", ""),
    }
    
    prompt = JUDGE_PROMPT_BASE + "\n\nHere is the JSON object to rate:\n" + \
             json.dumps(payload, ensure_ascii=False, indent=2)
    
    ok, content = openai_tool.get_respons(prompt, model=MODEL_NAME)
    if not ok:
        print(f"    [warn] GPT judge failed for entry {entry.get('id', 'unknown')}, raw: {content[:200]}")
        return {
            "grounded_in_context": 0.01,
            "medical_correctness": 0.01,
            "reasoning_clarity": 0.01,
            "language_quality": 0.01,
        }
    
    try:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        elif content.startswith("```json"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        
        obj = json.loads(content)
    except Exception as e:
        print(f"    [warn] JSON parse error for entry {entry.get('id', 'unknown')}, error: {e}")
        print(f"    [warn] Raw content: {content[:300]}")
        return {
            "grounded_in_context": 0,
            "medical_correctness": 0,
            "reasoning_clarity": 0,
            "language_quality": 0,
        }
    
    def _clamp(x):
        try:
            v = float(x)
        except Exception:
            return 0.01
        return max(0.01, min(1.0, v))
    
    return {
        "grounded_in_context": _clamp(obj.get("grounded_in_context", 0.0)),
        "medical_correctness": _clamp(obj.get("medical_correctness", 0.0)),
        "reasoning_clarity": _clamp(obj.get("reasoning_clarity", 0.0)),
        "language_quality": _clamp(obj.get("language_quality", 0.0)),
    }


def main(input_json: Path, output_json: Path, sleep_between_req: float, parallel_workers: int):
    print("=" * 60)
    print("Scoring step6.json with GPT")
    print("=" * 60)
    
    if not input_json.exists():
        raise FileNotFoundError(f"Input file not found: {input_json}")
    
    print(f"\n[1] Loading {input_json}...")
    
    max_retries = 5
    retry_delay = 0.5
    
    entries = None
    for attempt in range(max_retries):
        try:
            with open(input_json, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[警告] 文件内容为空: {input_json}")
                        print("    跳过此文件")
                        return
                
                entries = json.loads(content)
                break
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"[警告] JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                print(f"    等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"[错误] JSON 解析失败，已重试 {max_retries} 次: {str(e)[:200]}")
                print("    跳过此文件")
                return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[警告] 读取文件失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"[错误] 读取文件失败，已重试 {max_retries} 次: {str(e)[:200]}")
                print("    跳过此文件")
                return
    
    if entries is None:
        print("[错误] 无法读取文件")
        return
    
    if not isinstance(entries, list):
        print(f"[警告] JSON 格式不正确，期望列表: {input_json}")
        print("    跳过此文件")
        return
    
    print(f"    Found {len(entries)} entries")
    
    text_id_entries = defaultdict(list)
    skipped_empty = 0
    
    for entry in entries:
        question = entry.get("question", "").strip()
        answer = entry.get("answer", "").strip()
        cot = entry.get("cot", "").strip()
        
        if not (question or answer or cot):
            skipped_empty += 1
            continue
        
        meta = entry.get("metadata", {})
        text_id = meta.get("text_id", "")
        if text_id:
            text_id_entries[text_id].append(entry)
        else:
            text_id_entries[""].append(entry)
    
    need_scoring = []
    already_scored_text_ids = set()
    already_scored_entries = []
    
    for text_id, text_entries in text_id_entries.items():
        if not text_id:
            for entry in text_entries:
                llmjudge = entry.get("llmjudge", {})
                if isinstance(llmjudge, dict) and llmjudge:
                    scores = [
                        llmjudge.get("grounded_in_context", 0.0),
                        llmjudge.get("medical_correctness", 0.0),
                        llmjudge.get("reasoning_clarity", 0.0),
                        llmjudge.get("language_quality", 0.0),
                    ]
                    if not all(s == 0.0 or s == 0.01 for s in scores):
                        already_scored_entries.append(entry)
                        continue
                need_scoring.append(entry)
            continue
        
        all_scored = True
        for entry in text_entries:
            llmjudge = entry.get("llmjudge", {})
            if isinstance(llmjudge, dict) and llmjudge:
                scores = [
                    llmjudge.get("grounded_in_context", 0.0),
                    llmjudge.get("medical_correctness", 0.0),
                    llmjudge.get("reasoning_clarity", 0.0),
                    llmjudge.get("language_quality", 0.0),
                ]
                if all(s == 0.0 or s == 0.01 for s in scores):
                    all_scored = False
                    break
            else:
                all_scored = False
                break
        
        if all_scored:
            already_scored_text_ids.add(text_id)
            already_scored_entries.extend(text_entries)
        else:
            for entry in text_entries:
                llmjudge = entry.get("llmjudge", {})
                if isinstance(llmjudge, dict) and llmjudge:
                    scores = [
                        llmjudge.get("grounded_in_context", 0.0),
                        llmjudge.get("medical_correctness", 0.0),
                        llmjudge.get("reasoning_clarity", 0.0),
                        llmjudge.get("language_quality", 0.0),
                    ]
                    if all(s == 0.0 or s == 0.01 for s in scores):
                        need_scoring.append(entry)
                    else:
                        already_scored_entries.append(entry)
                else:
                    need_scoring.append(entry)
    
    print(f"\n[2] Classification:")
    print(f"    Entries with empty Q/A/CoT (skipped): {skipped_empty}")
    print(f"    Text IDs already fully scored (skipped): {len(already_scored_text_ids)}")
    print(f"    Entries already scored: {len(already_scored_entries)}")
    print(f"    Entries need scoring (llmjudge all 0.0/0.01 or missing): {len(need_scoring)}")
    
    if not need_scoring:
        print("\n✓ All entries are already scored (no all-zero llmjudge). Nothing to do.")
        return
    
    print(f"\n[3] Scoring entries...")
    
    file_name = input_json.stem
    write_lock = threading.Lock()
    scored_count = 0
    failed_count = 0
    completed_count = 0
    REPORT_BATCH_SIZE = 1000
    
    def score_and_save(entry_data):
        nonlocal scored_count, failed_count, completed_count
        
        entry, entry_idx = entry_data
        entry_id = entry.get("id", "unknown")
        
        time.sleep(random.uniform(0, 0.1))
        
        scores = judge_entry(entry)
        entry["llmjudge"] = scores
        
        success = any(s > 0.0 for s in scores.values())
        
        with write_lock:
            if success:
                scored_count += 1
            else:
                failed_count += 1
            
            completed_count += 1
            
            save_interval = 100
            if completed_count % save_interval == 0 or completed_count % REPORT_BATCH_SIZE == 0:
                temp_file = output_json.with_suffix('.tmp')
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(entries, f, ensure_ascii=False, indent=2)
                    temp_file.replace(output_json)
                except Exception as e:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except:
                        pass
                
                if completed_count % REPORT_BATCH_SIZE == 0:
                    percentage = (completed_count / len(need_scoring)) * 100 if len(need_scoring) > 0 else 0
                    print(f"[{file_name}] 批次 {completed_count // REPORT_BATCH_SIZE}: "
                          f"完成 {completed_count}/{len(need_scoring)} entry "
                          f"({percentage:.1f}%) | "
                          f"成功: {scored_count}, 失败: {failed_count}")
        
        return entry_id, scores, success
    
    if parallel_workers > 1 and len(need_scoring) > 1:
        print(f"[{file_name}] 使用并行处理，并行度: {parallel_workers}, 总任务数: {len(need_scoring)}, 汇报间隔: {REPORT_BATCH_SIZE} entry")
        
        entry_with_idx = [(entry, i) for i, entry in enumerate(need_scoring)]
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_entry = {executor.submit(score_and_save, entry_data): entry_data 
                              for entry_data in entry_with_idx}
            
            for future in as_completed(future_to_entry):
                try:
                    entry_id, scores, success = future.result()
                    if not success:
                        print(f"    [warn] Entry {entry_id} scoring failed (all 0.0)")
                except Exception as e:
                    entry_data = future_to_entry[future]
                    print(f"    [error] Entry {entry_data[0].get('id', 'unknown')} error: {e}")
    else:
        print(f"[{file_name}] 使用串行处理")
        for i, entry in enumerate(need_scoring, 1):
            entry_id = entry.get("id", "unknown")
            print(f"    [{i}/{len(need_scoring)}] Scoring entry {entry_id}...", end=" ", flush=True)
            
            scores = judge_entry(entry)
            entry["llmjudge"] = scores
            
            if any(s > 0.0 for s in scores.values()):
                scored_count += 1
                print(f"✓ {scores}")
            else:
                failed_count += 1
                print(f"✗ (all 0.0)")
            
            if i % 10 == 0:
                temp_file = output_json.with_suffix('.tmp')
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(entries, f, ensure_ascii=False, indent=2)
                    temp_file.replace(output_json)
                    print(f"    [checkpoint] Saved progress at {i}/{len(need_scoring)}")
                except Exception as e:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except:
                        pass
            
            time.sleep(sleep_between_req)
    
    print(f"\n[4] Saving results to {output_json}...")
    temp_file = output_json.with_suffix('.tmp')
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        temp_file.replace(output_json)
    except Exception as e:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except:
            pass
        raise
    
    print(f"\n" + "=" * 60)
    print("Final statistics:")
    print(f"  Total entries: {len(entries)}")
    print(f"  Entries newly scored (was all-zero or missing): {scored_count}")
    print(f"  Entries failed: {failed_count}")
    print(f"  Entries skipped (empty Q/A/CoT): {skipped_empty}")
    print(f"  Text IDs fully scored (skipped): {len(already_scored_text_ids)}")
    print(f"  Entries kept old scores: {len(already_scored_entries)}")
    print("=" * 60)
    print("✓ Scoring complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score step6.json files with GPT-based llmjudge"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input step6 JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: same as input, in-place update).",
    )
    parser.add_argument(
        "--sleep-between-req",
        type=float,
        default=0.5,
        help="Sleep time between requests (default: 0.5)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=100,
        help="Number of parallel workers (default: 100)",
    )
    args = parser.parse_args()

    input_json = Path(args.input)
    output_json = Path(args.output) if args.output else input_json

    main(input_json, output_json, args.sleep_between_req, args.parallel_workers)