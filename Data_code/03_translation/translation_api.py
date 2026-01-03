import sys
import re
from typing import Tuple, Dict, List
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd

class _Config:
    MODEL_PATH = '/data/models/nllb-200-3.3B'
    LEXICON_PATH = '/data/bufan/machenze/Translation/Stage_2/final_lexicon_table.xlsx'
    SOURCE_LANG = 'eng_Latn'
    TARGET_LANG = 'hin_Deva'
    BATCH_SIZE = 8
    USE_DYNAMIC_BATCHING = True
    LENGTH_BUCKET_SIZE = 16
    DEBUG = False
    DEBUG_INTERVAL = 10

class _NLTKManager:

    def __init__(self):
        try:
            from blingfire import text_to_sentences
            self._text_to_sentences = text_to_sentences
        except ImportError:
            print('✗ FATAL ERROR: blingfire not installed. Please `pip install blingfire`.')
            sys.exit(1)

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        s_block = self._text_to_sentences(text)
        sents = [s.strip() for s in s_block.split('\n') if s.strip()]
        return sents

class _PreTranslationLexicon:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.pattern = None
        self.translation_map: Dict[str, str] = {}
        self._load()

    def _build_term_regex(self, eng: str) -> str:
        escaped = re.escape(eng)
        escaped = escaped.replace('\\ ', '[\\s\\-]+')
        return escaped

    def _load(self):
        if _Config.DEBUG:
            print(f"\n{'=' * 80}")
            print('LOADING PRE-TRANSLATION LEXICON')
            print(f"{'=' * 80}")
            print(f'File: {self.file_path}')
        try:
            if self.file_path.endswith('.xlsx'):
                df = pd.read_excel(self.file_path)
            else:
                df = pd.read_csv(self.file_path)
            if 'English' not in df.columns or 'Hindi' not in df.columns:
                print("✗ ERROR: Lexicon must contain 'English' and 'Hindi' columns")
                sys.exit(1)
            df.dropna(subset=['English', 'Hindi'], inplace=True)
            df['English'] = df['English'].astype(str).str.strip()
            df['Hindi'] = df['Hindi'].astype(str).str.strip()
            df = df[(df['English'] != '') & (df['Hindi'] != '')]
            pairs = sorted(list(zip(df['English'], df['Hindi'])), key=lambda x: len(x[0]), reverse=True)
            self.translation_map = {eng.lower(): hin for eng, hin in pairs}
            term_patterns = [self._build_term_regex(eng) for eng, _ in pairs]
            if term_patterns:
                alternation = '(?:' + '|'.join(term_patterns) + ')'
                left_boundary = '(?<![A-Za-z0-9_])'
                right_boundary = "(?=(?:'s|'s)?(?![A-Za-z0-9_]))"
                full = left_boundary + alternation + right_boundary
                self.pattern = re.compile(full, re.IGNORECASE)
            else:
                self.pattern = re.compile('^\\b\\B$')
            if _Config.DEBUG:
                print(f'✓ Pre-translation lexicon loaded successfully')
                print(f'  - Active terms: {len(pairs)}')
        except FileNotFoundError:
            print(f"✗ ERROR: Lexicon file not found at '{self.file_path}'")
            sys.exit(1)
        except Exception as e:
            print(f'✗ Error reading lexicon: {e}')
            sys.exit(1)

    def apply(self, text: str) -> Tuple[str, Dict[str, Tuple[str, str]]]:
        if not self.pattern:
            return (text, {})
        used: Dict[str, Tuple[str, str]] = {}

        def repl(m):
            original = m.group(0)
            norm = re.sub('[\\s\\-]+', ' ', original).strip().lower()
            hin = self.translation_map.get(norm)
            if hin:
                used[norm] = (original, hin)
                return hin
            return original
        out = self.pattern.sub(repl, text)
        return (out, used)

    @staticmethod
    def annotate(hin_text: str, used: Dict[str, Tuple[str, str]]) -> str:
        if not used:
            return hin_text
        text = hin_text
        items = sorted(used.items(), key=lambda x: len(x[1][1]), reverse=True)
        for _, (eng_original, hin) in items:
            pattern = re.compile(re.escape(hin) + '(?!\\s*\\()', flags=0)
            text = pattern.sub(f'{hin} ({eng_original})', text)
        return text

class _Preprocessor:

    def __init__(self, lex: _PreTranslationLexicon):
        self.lex = lex

    def run(self, text: str) -> Tuple[str, Dict[str, Tuple[str, str]]]:
        return self.lex.apply(text)

    def run_batch(self, texts: List[str]) -> List[Tuple[str, Dict[str, Tuple[str, str]]]]:
        return [self.run(text) for text in texts]

class _Postprocessor:

    def __init__(self, lex: _PreTranslationLexicon):
        self.lex = lex

    def run(self, hin_text: str, used: Dict[str, Tuple[str, str]]) -> str:
        return self.lex.annotate(hin_text, used)

    def run_batch(self, hin_texts: List[str], used_list: List[Dict[str, Tuple[str, str]]]) -> List[str]:
        return [self.run(hin, used) for hin, used in zip(hin_texts, used_list)]

class _NLLBTranslator:

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length = None
        self.target_lang_id = None
        self._load()

    def _load(self):
        if _Config.DEBUG:
            print(f"\n{'=' * 80}")
            print('MODEL INITIALIZATION (OPTIMIZED)')
            print(f"{'=' * 80}")
            print(f'Model path: {self.model_path}')
            print(f'Batch size: {_Config.BATCH_SIZE}')
            print(f'Dynamic batching: {_Config.USE_DYNAMIC_BATCHING}')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, device_map='auto', torch_dtype=dtype)
            self.model.eval()
            self.device = next(self.model.parameters()).device
            device_map = getattr(self.model, 'hf_device_map', None)
            if device_map is not None:
                print('HF device map:', device_map)
            else:
                print('HF device map not found; model main device:', self.device)
            self.max_length = getattr(self.model.config, 'max_position_embeddings', 1024)
            self.target_lang_id = self._get_target_lang_id()
            if _Config.DEBUG:
                print('✓ Model loaded successfully')
                print(f'✓ Device: {self.device}')
                print(f'✓ Data type: {dtype}')
                print(f'✓ Max input length: {self.max_length} tokens')
                print(f'✓ Target language ID: {self.target_lang_id}')
                if torch.cuda.is_available():
                    print(f'✓ GPU Memory allocated: {torch.cuda.memory_allocated() / 1000000000.0:.2f} GB')
                    print(f'✓ GPU Memory reserved: {torch.cuda.memory_reserved() / 1000000000.0:.2f} GB')
                print(f"{'=' * 80}\n")
        except Exception as e:
            print(f'✗ Error loading model: {e}')
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _get_target_lang_id(self):
        try:
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                return self.tokenizer.lang_code_to_id.get(_Config.TARGET_LANG)
            if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                token_id = self.tokenizer.convert_tokens_to_ids(_Config.TARGET_LANG)
                if token_id != self.tokenizer.unk_token_id:
                    return token_id
            vocab = self.tokenizer.get_vocab()
            if _Config.TARGET_LANG in vocab:
                return vocab[_Config.TARGET_LANG]
            for token, idx in vocab.items():
                if _Config.TARGET_LANG in token:
                    return idx
            if _Config.DEBUG:
                print(f'⚠ Warning: Could not find language token for {_Config.TARGET_LANG}')
            return None
        except Exception as e:
            if _Config.DEBUG:
                print(f'⚠ Warning: Error getting target language ID: {e}')
            return None

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        try:
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                if self.target_lang_id is not None:
                    generated = self.model.generate(**inputs, forced_bos_token_id=self.target_lang_id, max_length=self.max_length, num_beams=5, early_stopping=True)
                else:
                    generated = self.model.generate(**inputs, max_length=self.max_length, num_beams=5, early_stopping=True)
            translations = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            return translations
        except Exception as e:
            print(f'✗ Batch translation error: {e}')
            if _Config.DEBUG:
                import traceback
                traceback.print_exc()
            return [f'[TRANSLATION_ERROR: {e}]'] * len(texts)

class _SentenceTranslator:

    def __init__(self, model: _NLLBTranslator, pre: _Preprocessor, post: _Postprocessor, nltk_mgr: _NLTKManager):
        self.model = model
        self.pre = pre
        self.post = post
        self.nltk_mgr = nltk_mgr

    def _create_batches(self, sentences: List[str]) -> List[List[int]]:
        if not _Config.USE_DYNAMIC_BATCHING:
            return [list(range(i, min(i + _Config.BATCH_SIZE, len(sentences)))) for i in range(0, len(sentences), _Config.BATCH_SIZE)]
        tokenizer = self.model.tokenizer
        with torch.no_grad():
            enc = tokenizer(sentences, add_special_tokens=False, padding=False, truncation=True, return_attention_mask=False)
        lengths = [len(ids) for ids in enc['input_ids']]
        indexed_lengths = sorted(enumerate(lengths), key=lambda x: x[1])
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_tokens = 0
        for idx, length in indexed_lengths:
            if len(current_batch) >= _Config.BATCH_SIZE or (current_batch and length > current_tokens + _Config.LENGTH_BUCKET_SIZE):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(idx)
            current_tokens = max(current_tokens, length)
        if current_batch:
            batches.append(current_batch)
        return batches

    def translate(self, text: str, item_idx: int=0) -> str:
        sents = self.nltk_mgr.tokenize(text)
        if _Config.DEBUG:
            print(f"\n{'=' * 80}")
            print(f'Item #{item_idx + 1} - Translating {len(sents)} sentence(s) with batching')
            print(f"{'=' * 80}")
        preprocess_results = self.pre.run_batch(sents)
        prepped_sents = [r[0] for r in preprocess_results]
        used_dicts = [r[1] for r in preprocess_results]
        batches = self._create_batches(prepped_sents)
        if _Config.DEBUG:
            print(f'Created {len(batches)} batches')
        all_translations = [None] * len(sents)
        for batch_idx, indices in enumerate(batches):
            batch_texts = [prepped_sents[i] for i in indices]
            if _Config.DEBUG:
                print(f'\n[Batch {batch_idx + 1}/{len(batches)}] Processing {len(indices)} sentences')
            translations = self.model.translate_batch(batch_texts)
            for i, translation in zip(indices, translations):
                all_translations[i] = translation
        final_outputs = self.post.run_batch(all_translations, used_dicts)
        if _Config.DEBUG:
            print(f"\n{'=' * 80}")
            print('TRANSLATION RESULTS')
            print(f"{'=' * 80}")
            for i, (eng, hin) in enumerate(zip(sents, final_outputs)):
                print(f'\n[{i + 1}] English: {eng}')
                print(f'    Hindi:   {hin}')
            print(f"{'=' * 80}\n")
        return ' '.join(final_outputs)

class _Service:
    _inst: '_Service' = None

    def __init__(self):
        if _Config.DEBUG:
            print('Initializing optimized translation service...')
        self.nltk_mgr = _NLTKManager()
        self.lex = _PreTranslationLexicon(_Config.LEXICON_PATH)
        self.pre = _Preprocessor(self.lex)
        self.post = _Postprocessor(self.lex)
        self.model = _NLLBTranslator(_Config.MODEL_PATH)
        self.engine = _SentenceTranslator(self.model, self.pre, self.post, self.nltk_mgr)

    @classmethod
    def get(cls) -> '_Service':
        if cls._inst is None:
            cls._inst = _Service()
        return cls._inst

def translate_paragraph(text: str) -> str:
    service = _Service.get()
    return service.engine.translate(text, item_idx=0)

def translate_paragraphs(texts: List[str]) -> List[str]:
    if not texts:
        return []
    service = _Service.get()
    results = []
    for idx, text in enumerate(texts):
        try:
            translated = service.engine.translate(text, item_idx=idx)
            results.append(translated)
        except Exception as e:
            print(f'✗ Error translating paragraph {idx + 1}: {e}')
            results.append(f'[ERROR: {e}]')
    return results
__all__ = ['translate_paragraph', 'translate_paragraphs']
if __name__ == '__main__':
    import time
    example = "A 73-year-old man presents to the emergency department complaining of abdominal pain with nausea and vomiting, stating that he 'cannot keep anything down'. He states that the pain has been gradually getting worse over the past 2 months, saying that, at first, it was present only an hour after he ate but now is constant."
    print('=' * 80)
    print('PERFORMANCE TEST: Single paragraph')
    print('=' * 80)
    start = time.time()
    result = translate_paragraph(example)
    elapsed = time.time() - start
    print(f"\n{'=' * 80}")
    print('TRANSLATED OUTPUT')
    print(f"{'=' * 80}")
    print(result)
    print(f'\n⏱️  Translation time: {elapsed:.2f} seconds')
