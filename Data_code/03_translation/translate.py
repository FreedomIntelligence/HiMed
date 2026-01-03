import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
try:
    from translation_api import translate_paragraphs
except ImportError:
    print("⚠️ Warning: 'translation_api' not found. Using dummy translation for testing.")

    def translate_paragraphs(texts):
        return [f'[HI] {t}' for t in texts]

class StructureParser:
    OPTION_PATTERN = re.compile('(?m)^([A-Z])\\.\\s+(.*)$')
    GT_PATTERN = re.compile('^([A-Z])\\.\\s+(.*)$')

    @staticmethod
    def parse_prompt(text: str) -> Dict[str, Any]:
        if not text:
            return {'question': '', 'options': {}, 'order': []}
        matches = list(StructureParser.OPTION_PATTERN.finditer(text))
        if not matches:
            return {'question': text.strip(), 'options': {}, 'order': []}
        first_match_start = matches[0].start()
        question_text = text[:first_match_start].strip()
        options, order = ({}, [])
        for match in matches:
            label = match.group(1)
            content = match.group(2).strip()
            options[label] = content
            order.append(label)
        return {'question': question_text, 'options': options, 'order': order}

    @staticmethod
    def parse_gt(text: str) -> Tuple[str, str]:
        if not text:
            return ('', '')
        match = StructureParser.GT_PATTERN.match(text.strip())
        if match:
            return (match.group(1), match.group(2).strip())
        return ('', text.strip())

    @staticmethod
    def reconstruct_prompt(hi_question: str, hi_options: Dict[str, str], order: List[str]) -> str:
        parts = [hi_question]
        for label in order:
            parts.append(f"{label}. {hi_options.get(label, '')}")
        return '\n'.join(parts)

    @staticmethod
    def reconstruct_gt(label: str, hi_content: str) -> str:
        return f'{label}. {hi_content}' if label else hi_content

class QuestionTranslator:

    def __init__(self, batch_size: int=100, save_interval: int=10):
        self.batch_size = batch_size
        self.save_interval = save_interval

    def load_data(self, path: Path) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else data['questions']

    def save_data(self, data: List[Dict], path: Path):
        temp_path = path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        if path.exists():
            path.unlink()
        temp_path.rename(path)

    def process_file(self, input_path: str, output_path: str):
        in_p, out_p = (Path(input_path), Path(output_path))
        questions = self.load_data(in_p)
        tasks = []
        buffer = {}
        for idx, item in enumerate(questions):
            need_prompt = 'prompt_hi' not in item
            need_gt = 'ground_truth_hi' not in item
            cot_raw = item.get('Complex_CoT', None)
            cot_hi_exist = item.get('Complex_CoT_hi')
            need_cot = cot_raw not in [None, ''] and cot_hi_exist is None
            prompt_en = item.get('prompt', '')
            gt_en = item.get('ground_truth', '')
            p_struct = StructureParser.parse_prompt(prompt_en)
            gt_label, gt_text = StructureParser.parse_gt(gt_en)
            buffer[idx] = {'p_struct': p_struct, 'gt_label': gt_label, 'gt_text_raw': gt_text, 'trans_q': '', 'trans_opts': {}, 'trans_gt': '', 'cot_raw': cot_raw, 'trans_cot': '', 'need_prompt': need_prompt, 'need_gt': need_gt, 'need_cot': need_cot}
            if need_prompt and p_struct['question']:
                tasks.append((idx, 'q', None, p_struct['question']))
                for label, txt in p_struct['options'].items():
                    if txt:
                        tasks.append((idx, 'opt', label, txt))
            if need_gt and gt_text:
                tasks.append((idx, 'gt', None, gt_text))
            if need_cot:
                tasks.append((idx, 'cot', None, cot_raw))
        with tqdm(total=len(tasks), unit='seg') as pbar:
            for i in range(0, len(tasks), self.batch_size):
                batch = tasks[i:i + self.batch_size]
                batch_texts = [t[3] for t in batch]
                translations = translate_paragraphs(batch_texts)
                for j, trans_text in enumerate(translations):
                    d_idx, t_type, t_key, _ = batch[j]
                    if t_type == 'q':
                        buffer[d_idx]['trans_q'] = trans_text
                    elif t_type == 'opt':
                        buffer[d_idx]['trans_opts'][t_key] = trans_text
                    elif t_type == 'gt':
                        buffer[d_idx]['trans_gt'] = trans_text
                    elif t_type == 'cot':
                        buffer[d_idx]['trans_cot'] = trans_text
                for d_idx in {b[0] for b in batch}:
                    buf = buffer[d_idx]
                    struct = buf['p_struct']
                    if not buf['need_prompt'] and (not buf['need_gt']) and (not buf['need_cot']):
                        continue
                    if buf['need_prompt']:
                        hi_q = buf['trans_q'] or struct['question']
                        hi_opts = {lbl: buf['trans_opts'].get(lbl, struct['options'][lbl]) for lbl in struct['order']}
                        questions[d_idx]['prompt_hi'] = StructureParser.reconstruct_prompt(hi_q, hi_opts, struct['order'])
                    if buf['need_gt']:
                        hi_gt_text = buf['trans_gt'] or buf['gt_text_raw']
                        questions[d_idx]['ground_truth_hi'] = StructureParser.reconstruct_gt(buf['gt_label'], hi_gt_text)
                    if buf['need_cot'] and buf['trans_cot']:
                        questions[d_idx]['Complex_CoT_hi'] = buf['trans_cot']
                pbar.update(len(batch))
                if self.save_interval and (i // self.batch_size + 1) % self.save_interval == 0:
                    self.save_data(questions, out_p)
        self.save_data(questions, out_p)
        print(f'\n✅ Done! Saved to: {out_p}')

def main():
    input_file = input('📂 Input JSON path: ').strip().strip('"').strip("'")
    output_file = input('📂 Output JSON path: ').strip().strip('"').strip("'")
    batch_size = int(input('📦 Batch size [Enter for 100]: ') or 100)
    save_interval = int(input('💾 Save interval (batches) [Enter for 10]: ') or 10)
    translator = QuestionTranslator(batch_size=batch_size, save_interval=save_interval)
    translator.process_file(input_file, output_file)
if __name__ == '__main__':
    main()
