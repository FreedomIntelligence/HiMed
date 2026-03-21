import os
import re
import json
import time
import random
import traceback
import threading
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
INPUT_DIR = Path('')
OUTPUT_DIR = Path('')
RECURSIVE = False
FORCE_OVERWRITE = False
INSTR_XLSX_PATH = Path('')
EXAMPLES_XLSX_PATH = Path('')
API_KEY = ''
API_BASE_URL = ''
MODEL_NAME = 'gpt-4o'
MAX_RETRY = 3
DEBUG = False
SLEEP_BETWEEN_REQ = 0.1
TOPICS = ['diagnosis', 'etiology', 'medical knowledge', 'prognosis', 'treatment']
TYPES = ['MCQ', 'QA', 'Dialogue']

class GetOpenAI:
    _session = None
    _session_lock = threading.Lock()

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            with cls._session_lock:
                if cls._session is None:
                    cls._session = requests.Session()
                    retry_strategy = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
                    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
                    cls._session.mount('http://', adapter)
                    cls._session.mount('https://', adapter)
        return cls._session

    @staticmethod
    def __gpt_api_stream(messages: list, model=MODEL_NAME):
        try:
            headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
            data = {'model': model, 'messages': messages}
            session = GetOpenAI._get_session()
            response = session.post(API_BASE_URL, headers=headers, json=data, timeout=(30, 120))
            if response.status_code != 200:
                return (False, f'OpenAI API Exception: HTTP code {response.status_code} from API: {response.text[:200]}')
            try:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if not content:
                    return (False, f'OpenAI API Exception: No content in response: {response.text[:200]}')
                if content and (content.strip().startswith('<!doctype') or content.strip().startswith('<!DOCTYPE') or content.strip().startswith('<html')):
                    return (False, f'OpenAI API Exception: HTTP code 200 from API (HTML response detected: {content[:200]})')
                return (True, content)
            except json.JSONDecodeError:
                text = response.text[:200]
                return (False, f'OpenAI API Exception: Invalid JSON response: {text}')
        except requests.exceptions.Timeout as err:
            if DEBUG:
                print(traceback.format_exc())
            return (False, f'OpenAI API Exception: Request timeout: {err}')
        except requests.exceptions.ConnectionError as err:
            if DEBUG:
                print(traceback.format_exc())
            return (False, f'OpenAI API Exception: Connection error (DNS/network issue): {err}')
        except requests.exceptions.RequestException as err:
            if DEBUG:
                print(traceback.format_exc())
            return (False, f'OpenAI API Exception: Request error: {err}')
        except Exception as err:
            if DEBUG:
                print(traceback.format_exc())
            return (False, f'OpenAI API Exception: {err}')

    def get_respons(self, input_msg, model=MODEL_NAME):
        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': input_msg}]
        out = ''
        ok = False
        for attempt in range(MAX_RETRY):
            ok, out = self.__gpt_api_stream(msgs, model=model)
            if ok:
                return (ok, (out or '').strip())
            if attempt < MAX_RETRY - 1:
                sleep_time = 2 ** (attempt + 1)
                if DEBUG:
                    print(f'Retry {attempt + 1}/{MAX_RETRY} after {sleep_time}s...')
                time.sleep(sleep_time)
        return (ok, (out or '').strip())
openai_tool = GetOpenAI()
DEV_RX = re.compile('[\\u0900-\\u097F]')

def detect_lang(text: str) -> str:
    return 'hi' if DEV_RX.search(text or '') else 'en'
CLASSIFY_PROMPT = 'You are a medical annotation expert.\n\nIgnore all the possible names orpersonal identifiers.\nGiven a paragraph of text, you MUST decide:\n1) All applicable subjects from this closed set:\n   ["diagnosis", "etiology", "medical knowledge", "prognosis", "treatment"]\n2) All applicable question/text types from this closed set:\n   ["MCQ", "QA", "Dialogue"]\n\nRules:\n\n[For subjects]\n- Choose EVERY subject that is clearly supported by the content.\n- Be conservative: if a subject is not clearly supported, do NOT include it.\n- If NONE of the 5 subjects clearly fit, return ONLY ["medical knowledge"].\n\n[For types]\n\n"Dialogue": the text contains a multi-turn User–Assistant conversation. It should have two or more turns, and the last line is a new question posed by the User.\n(e.g., User: … Assistant: … User: … ?)\n\n"MCQ": the text is or contains a multiple-choice style question (options like\nA./B./C./D., or "choose the correct option", "from the following options", etc.).\n\n"QA": the text is or contains a single question with an answer, or is suitable for\ngenerating QA pairs, but is NOT clearly MCQ or Dialogue.\n\nA paragraph may correspond to multiple types.\n\nYou must always return at least ONE type.\n[OUTPUT FORMAT]\nReturn ONLY a valid JSON object with this exact schema, nothing else:\n\n{\n  "subject_list": ["diagnosis", "medical knowledge"],\n  "type_list": ["QA", "Dialogue"]\n}\n\nDo NOT add any extra keys. Do NOT add comments. Do NOT wrap the JSON in markdown.\n\nNow analyze the following paragraph:\n\n'
CLASS_CACHE_PATH = Path('.subject_type_cache.json')

def _class_cache_load() -> Dict[str, dict]:
    if CLASS_CACHE_PATH.exists():
        try:
            return json.loads(CLASS_CACHE_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}

def _class_cache_save(cache: Dict[str, dict]):
    try:
        CLASS_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass

def _norm_subject_name(s: str) -> str:
    t = str(s or '').strip().lower()
    mapping = {'diagnosis': 'diagnosis', 'dx': 'diagnosis', 'etiology': 'etiology', 'aetiology': 'etiology', 'medicalknowledge': 'medical knowledge', 'knowledge': 'medical knowledge', 'general': 'medical knowledge', 'prognosis': 'prognosis', 'treatment': 'treatment', 'therapy': 'treatment', 'management': 'treatment'}
    t = re.sub('[\\s_]+', '', t)
    return mapping.get(t, s)

def classify_subject_and_type(paragraph: str) -> Tuple[List[str], List[str]]:
    key = (paragraph or '').strip()
    cache = _class_cache_load()
    if key in cache:
        data = cache[key]
        subj_cached = data.get('subject_list', ['medical knowledge'])
        type_cached = data.get('type_list', None)
        if not type_cached:
            type_cached = TYPES[:]
        return (subj_cached, type_cached)
    prompt = CLASSIFY_PROMPT + key
    ok, content = openai_tool.get_respons(prompt, model=MODEL_NAME)
    if not ok:
        if DEBUG:
            print('[warn] GPT classify failed, fallback medical knowledge + all types. Raw:', content[:200])
        return (['medical knowledge'], TYPES[:])
    try:
        obj = json.loads(content)
    except Exception:
        if DEBUG:
            print('[warn] JSON parse error in classify, raw content:', content)
        return (['medical knowledge'], TYPES[:])
    subjects_raw = obj.get('subject_list') or []
    subject_list: List[str] = []
    for s in subjects_raw:
        s_norm = _norm_subject_name(s)
        if s_norm in TOPICS and s_norm not in subject_list:
            subject_list.append(s_norm)
    if not subject_list:
        subject_list = ['medical knowledge']
    types_raw = obj.get('type_list') or []
    type_list: List[str] = []
    for t in types_raw:
        t = str(t).strip()
        if t in TYPES and t not in type_list:
            type_list.append(t)
    if not type_list:
        type_list = TYPES[:]
    cache[key] = {'subject_list': subject_list, 'type_list': type_list}
    _class_cache_save(cache)
    return (subject_list, type_list)

def load_merged_json(path: Path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, list):
        raise ValueError(f'Invalid merged JSON: {path}')
    return obj

def flatten_paragraph_blocks(obj) -> List[Dict]:
    out: List[Dict] = []
    if obj and all((isinstance(x, dict) for x in obj)):
        for b in obj:
            if b.get('type') == 'paragraph':
                out.append({'text': b.get('text', ''), 'page_idx': b.get('page_idx')})
        return out
    for page in obj:
        if not isinstance(page, list):
            continue
        for block in page:
            if not isinstance(block, dict):
                continue
            if block.get('type') != 'paragraph':
                continue
            out.append({'text': block.get('text', ''), 'page_idx': block.get('page_idx')})
    return out

def _canon(s: str) -> str:
    s = str(s or '').strip().lower()
    s = re.sub('[\\s\\-_/]+', '', s)
    s = s.replace('：', '').replace(':', '')
    return s

def _norm_qtype_to_type_name(s: str) -> str:
    t = _canon(s)
    if t in {'qa', 'q', 'shortanswer'}:
        return 'QA'
    if t in {'dia', 'dialog', 'dialogue'}:
        return 'Dialogue'
    if t in {'mcq', 'mcqa', 'choice', 'singlechoice', 'multiplechoice'}:
        return 'MCQ'
    return ''
_TEMPL_POOL: Dict[Tuple[str, str], List[str]] = None

def load_templates_pool(xlsx_path: Path) -> Dict[Tuple[str, str], List[str]]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f'Template Excel not found: {xlsx_path}')
    df = pd.read_excel(xlsx_path)

    def _find_col(cands):
        for c in df.columns:
            if any((_canon(x) in _canon(c) for x in cands)):
                return c
        return None
    has_hindi = _find_col(['hindi']) is not None
    has_english = _find_col(['english']) is not None
    col_qtype = _find_col(['qtype', 'type'])
    if has_hindi or has_english:
        col_subj = _find_col(['category', 'theme', 'subject'])
        pool: Dict[Tuple[str, str], List[str]] = {}
        for subj_raw, sub_df in df.groupby(col_subj, sort=False):
            if pd.isna(subj_raw):
                continue
            subj = _norm_subject_name(str(subj_raw))
            rows = sub_df.reset_index(drop=True)
            for i in range(len(rows)):
                qtxt = ''
                if has_hindi:
                    qtxt = str(rows.at[i, _find_col(['hindi'])] or '').strip()
                if not qtxt and has_english:
                    qtxt = str(rows.at[i, _find_col(['english'])] or '').strip()
                if not qtxt:
                    continue
                if col_qtype:
                    type_raw = str(rows.at[i, col_qtype] or '').strip()
                    tname = _norm_qtype_to_type_name(type_raw)
                    if not tname:
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
        return pool
    col_subj = _find_col(['category', 'theme', 'subject'])
    col_qtpl = _find_col(['question_templete', 'questiontemplate', 'question'])
    col_atpl = _find_col(['answer(template)', 'answertemplate'])
    pool: Dict[Tuple[str, str], List[str]] = {}
    for i in range(len(df)):
        subj_raw = str(df.at[i, col_subj]).strip()
        if not subj_raw:
            continue
        subj = _norm_subject_name(subj_raw)
        qtype_raw = str(df.at[i, col_qtype]).strip()
        tname = _norm_qtype_to_type_name(qtype_raw)
        if not tname:
            continue
        qtxt = str(df.at[i, col_qtpl]).strip()
        atxt = str(df.at[i, col_atpl]).strip() if col_atpl in df.columns else ''
        tpl = '\n\n'.join([x for x in [qtxt, atxt] if x])
        if not tpl:
            continue
        key = (subj, tname)
        pool.setdefault(key, []).append(tpl)
    return pool

def get_question_template(subject: str, qtype: str) -> str:
    global _TEMPL_POOL
    if _TEMPL_POOL is None:
        _TEMPL_POOL = load_templates_pool(INSTR_XLSX_PATH)
    key = (subject, qtype)
    if key in _TEMPL_POOL:
        return random.choice(_TEMPL_POOL[key])
    candidates = []
    for (subj, tname), lst in _TEMPL_POOL.items():
        if subj == subject:
            candidates.extend(lst)
    if candidates:
        return random.choice(candidates)
    candidates = []
    for (subj, tname), lst in _TEMPL_POOL.items():
        if subj == 'medical knowledge':
            candidates.extend(lst)
    if candidates:
        return random.choice(candidates)
    return ''
_EX_POOL: Dict[Tuple[str, str], List[Dict]] = None

def load_examples_pool(xlsx_path: Path) -> Dict[Tuple[str, str], List[Dict]]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f'Examples Excel not found: {xlsx_path}')
    df = pd.read_excel(xlsx_path)

    def _find_col(cands):
        for c in df.columns:
            if any((_canon(x) in _canon(c) for x in cands)):
                return c
        return None
    col_subj = _find_col(['subject', 'theme', 'category'])
    col_q = _find_col(['q', 'question'])
    col_a = _find_col(['a', 'answer'])
    col_cot = _find_col(['cot', 'reasoning'])
    col_para = _find_col(['paragraph', 'context', 'text'])
    col_qtype = _find_col(['qtype', 'type'])
    pool: Dict[Tuple[str, str], List[Dict]] = {}
    for i in range(len(df)):
        subj_raw = str(df.at[i, col_subj]).strip()
        if not subj_raw:
            continue
        subj = _norm_subject_name(subj_raw)
        qtxt = str(df.at[i, col_q]).strip()
        atxt = str(df.at[i, col_a]).strip()
        if not (qtxt and atxt):
            continue
        cot = str(df.at[i, col_cot]).strip() if col_cot else ''
        para = str(df.at[i, col_para]).strip() if col_para else ''
        qtype_raw = str(df.at[i, col_qtype]).strip() if col_qtype else ''
        tname = _norm_qtype_to_type_name(qtype_raw) if qtype_raw else ''
        key = (subj, tname)
        pool.setdefault(key, []).append({'paragraph': para, 'q': qtxt, 'a': atxt, 'cot': cot})
    return pool

def get_few_shot(subject: str, qtype: str, k: int=3) -> List[Dict]:
    global _EX_POOL
    if _EX_POOL is None:
        _EX_POOL = load_examples_pool(EXAMPLES_XLSX_PATH)

    def _collect_keys(keys):
        out = []
        for key in keys:
            out.extend(_EX_POOL.get(key, []))
        return out
    cand = _collect_keys([(subject, qtype)])
    if not cand:
        cand = _collect_keys([(subject, '')])
    if not cand:
        cand = _collect_keys([('medical knowledge', qtype), ('medical knowledge', '')])
    random.shuffle(cand)
    return cand[:k]
Q_RX = re.compile('<q>(.*?)</q>', re.S | re.I)
A_RX = re.compile('<a>(.*?)</a>', re.S | re.I)
COT_RX = re.compile('<cot>(.*?)</cot>', re.S | re.I)
DIFF_BLOCK_RX = re.compile('<(EASY|MEDIUM|HARD)>(.*?)</\\\\1>', re.S | re.I)
MCQ_FEW_SHOT = '\nExample (MCQ - Easy):\n<q>पारंपरिक भारतीय चिकित्सा में, रोगों से स्वस्थ होने में मुख्य सहायता किस तत्व को माना जाता है? \nA. तीव्र औषधीय मिश्रण\nB. प्रकृति की उपचार क्षमता\nC. रक्त-निर्गमन (bloodletting)\nD. फफोले उत्पन्न करने वाली चिकित्सा\nE. शक्तिशाली बहु-औषधि उपचार</q>\n<a>B</a>\n<cot>पारंपरिक भारतीय दृष्टिकोण में माना जाता है कि रोग से उबरने में शरीर की प्राकृतिक उपचार-शक्ति सबसे अधिक योगदान देती है। अत्यधिक तीव्र औषधियाँ या कठोर प्रक्रियाएँ हानि पहुँचा सकती हैं। इसलिए सही विकल्प है B—प्राकृतिक उपचार-प्रक्रिया。</cot>\n\nExample (MCQ - Medium):\n<q>पारंपरिक भारतीय चिकित्सा में तीव्र औषधीय उपचारों को सावधानी से क्यों देखा जाता है?\nA. क्योंकि वे आध्यात्मिक असंतुलन बढ़ाते हैं\nB. क्योंकि उनका उपयोग केवल शीत ऋतु में किया जा सकता है\nC. क्योंकि वे हानि पहुँचा सकते हैं और प्रायः रोग के परिणाम पर विशेष प्रभाव नहीं डालते\nD. क्योंकि वे पाचन शक्ति को कम करते हैं\nE. क्योंकि इन्हें प्रभावी बनाने के लिए जटिल अनुष्ठान की आवश्यकता होती है</q>\n<a>C</a>\n<cot>अत्यधिक तीव्र औषधियाँ शरीर को अतिरिक्त हानि पहुँचा सकती हैं, और जब शरीर स्वयं रोग से लड़ सकता है तो उनका वास्तविक लाभ सीमित होता है। इसलिए हानि की संभावना + सीमित प्रभाव—इन दोनों कारणों को समेटने वाला विकल्प C सही है。</cot>\n\nExample (MCQ - Hard):\n<q>एक पारंपरिक भारतीय वैद्य ज्वर से पीड़ित रोगी को देखते हैं, परंतु वे शक्तिशाली औषधियों का प्रयोग नहीं करते। इसके बजाय, वे विश्राम और शरीर की प्राकृतिक उपचार-शक्ति पर भरोसा करने की सलाह देते हैं。\nइस निर्णय के पीछे सबसे उपयुक्त तर्क क्या हो सकता है?\nA. शक्तिशाली औषधियाँ सभी प्रकार के ज्वर को अनिवार्य रूप से बढ़ा देती हैं\nB. प्राकृतिक उपचार सामान्यतः अधिक प्रभावी होता है, जबकि औषधियों का लाभ तभी है जब वे कोई हानि न पहुँचाएँ\nC. पारंपरिक मतानुसार ज्वर में औषधियों का प्रयोग पूर्णतः वर्जित है\nD. हल्का ज्वर कभी भी चिकित्सा की आवश्यकता नहीं रखता\nE. बिना औषधि के उपचार से हमेशा सबसे तेज़ स्वस्थता प्राप्त होती है</q>\n<a>B</a>\n<cot>पारंपरिक चिकित्सा शरीर की नैसर्गिक संतुलन-स्थापना क्षमता को अत्यंत महत्वपूर्ण मानती है। तीव्र औषधियों को तभी उपयोगी माना जाता है जब वे अतिरिक्त हानि न पहुँचाएँ। इसलिए प्राकृतिक उपचार को प्राथमिकता देना और संभावित हानिकारक हस्तक्षेपों को कम करना—यह सिद्धांत विकल्प B में सटीक रूप से व्यक्त है。</cot>\n'
QA_FEW_SHOT = '\nExample (QA - Easy):\n<q>पारंपरिक भारतीय चिकित्सा में, रोग से उबरने की मुख्य शक्ति आमतौर पर किससे मानी जाती है?</q>\n<a>शरीर की स्वाभाविक पुनर्स्थापन (प्राकृतिक उपचार) क्षमता से。</a>\n<cot>पारंपरिक विचारधारा शरीर की जन्मजात संतुलन-बनाए-रखने और मरम्मत करने की क्षमता को अत्यंत महत्वपूर्ण मानती है। बाहरी हस्तक्षेप सहायक होते हैं, मुख्य शक्ति शरीर की स्वाभाविक पुनर्स्थापन क्षमता से आती है。</cot>\n\nExample (QA - Medium):\n<q>पारंपरिक भारतीय चिकित्सा में, प्रबल औषधियों का महत्व क्यों सीमित माना जाता है, जबकि प्राकृतिक पुनर्स्थापन प्रक्रिया को अधिक महत्व दिया जाता है?</q>\n<a>क्योंकि प्रबल हस्तक्षेप शरीर पर अतिरिक्त बोझ डाल सकते हैं, जबकि प्राकृतिक प्रक्रिया अधिक सौम्य, स्थिर और शरीर की अपनी लय के अनुरूप मानी जाती है。</a>\n<cot>तीव्र उत्तेजना जोखिम ला सकती है और संतुलन-शक्ति को प्रभावित कर सकती है। प्राकृतिक पुनर्स्थापन शरीर की लय में रहता है, इसलिए अधिक स्थिर माना जाता है。 जीवन-शक्ति की रक्षा करने वाली परंपरा अतिरिक्त बोझ से बचने को प्राथमिकता देती है。</cot>\n\nExample (QA - Hard):\n<q>एक शिष्य पूछता है: “गुरुजी, शक्तिशाली औषधियाँ तो तेजी से प्रभाव दिखाती हैं, फिर आप ज्वर या दुर्बलता में अक्सर विश्राम और शरीर की प्राकृतिक शक्ति पर भरोसा क्यों करते हैं?” पारंपरिक चिकित्सा-दर्शन के अनुसार, गुरु ऐसा निर्णय क्यों लेते हैं?</q>\n<a>क्योंकि इस विचार-प्रणाली में मुख्य स्वास्थ्य-लाभ शरीर की अपनी शक्ति से आता है; यदि प्रबल औषधियाँ इस शक्ति को कमजोर कर सकती हैं या अतिरिक्त बोझ डालती हैं, तो संयमित हस्तक्षेप जीवन-शक्ति की रक्षा के सिद्धांत से अधिक मेल खाता है。</a>\n<cot>पारंपरिक चिकित्सा स्व-नियमन व स्व-पुनर्स्थापन को मूल मानती है। प्रबल औषधियाँ इस आंतरिक क्षमता में व्यवधान डाल सकती हैं। जीवन-शक्ति की रक्षा के लिए अनावश्यक हस्तक्षेप से बचना उपयुक्त है, इसलिए रोगी को स्वयं पुनर्स्थापित होने का अवसर देना बेहतर माना जाता है。</cot>\n'
DIALOGUE_FEW_SHOT = '\nExample (Dialogue - Easy):\n<q>User: गुरुजी, कुछ लोग थोड़ा-सा खाने पर भी भारीपन क्यों महसूस करते हैं?\nAssistant: पारंपरिक चिकित्सा में इसका संबंध इस बात से होता है कि शरीर की अग्नि संतुलित है या नहीं。\nUser: तो अग्नि के कमजोर होने का सीधा कारण क्या माना जाता है?</q>\n<a>अनुचित आहार, असंतुलित जीवनशैली और अस्थिर भावनाएँ तीनों दोषों को विचलित करती हैं, और यही अग्नि को कमजोर कर देती हैं。</a>\n<cot>दोषों का विचलन पाचन-अग्नि पर सीधा प्रभाव डालता है। आहार, दिनचर्या और भावनाएँ इन विचलनों के मुख्य कारण हैं。</cot>\n\nExample (Dialogue - Medium):\n<q>User: एक रोगी हर भोजन के बाद पेट में भारीपन महसूस कर रहा है。\nAssistant: ऐसा अनुभव प्रायः संकेत देता है कि उसकी पाचन-अग्नि स्थिर नहीं है。\nUser: यदि अग्नि जन्म से कमजोर न हो, तो उसमें समस्या कैसे आती है?\nAssistant: जब तीनों दोष अस्थिर होते हैं, तो वे सबसे पहले पाचन-अग्नि को प्रभावित करते हैं。\nUser: तो दोषों के असंतुलित होने का मूल कारण क्या हो सकता है?</q>\n<a>अनियमित भोजन, अस्वस्थ दैनिक आदतें और नकारात्मक भावनाएँ दोषों को असंतुलित करती हैं, और यही पाचन-अग्नि के कमजोर होने की जड़ बनती हैं。</a>\n<cot>दोषों की स्थिरता जीवनशैली, भोजन और मानसिक अवस्था पर निर्भर करती है। असंतुलन से अग्नि कमजोर होकर भोजन नहीं तोड़ पाती, इसलिए ये कारक एटिऑलॉजी की शुरुआत बनते हैं。</cot>\n\nExample (Dialogue - Hard):\n<q>User: गुरुजी, मेरे पास एक रोगी है। वह अधिक नहीं खाता, पर भोजन करते ही पेट भारी हो जाता है और मन भी थका हुआ लगता है。\nAssistant: यदि भोजन मात्रा समस्या नहीं है, तो शरीर की आंतरिक अग्नि-शक्ति में अवरोध की संभावना देखनी चाहिए。\nUser: उसने बताया कि हाल में उसकी दिनचर्या बिगड़ गई है और मन भी अक्सर उदास रहता है。\nAssistant: ये दोनों बातें तीनों दोषों के संतुलन को बदल सकती हैं。\nUser: यदि दोष विचलित हों, तो शरीर का कौन सा तंत्र सबसे पहले प्रभावित होता है?\nAssistant: प्रायः वही अग्नि, जो भोजन के परिवर्तन, विघटन और पोषण की मुख्य जिम्मेदारी निभाती है。\nUser: तो इस परिस्थिति में भोजन का ठीक से न टूट पाना किस गहरी एटिऑलॉजी से संबंधित है?</q>\n<a>जीवनशैली और भावनाओं से उत्पन्न दोष-असंतुलन ने अग्नि को कमजोर किया और भोजन-परिवर्तन की सामान्य क्षमता बाधित की—यही मूल एटिऑलॉजी है。</a>\n<cot>अव्यवस्थित दिनचर्या शरीर की प्राकृतिक लय को बाधित करती है, नकारात्मक भावनाएँ दोषों को अस्थिर करती हैं, और दोष-असंतुलन सबसे पहले अग्नि को प्रभावित करता है। अग्नि-क्षीणता से भोजन विघटन और अवशोषण घटता है, जिससे भारीपन और थकावट होती है。</cot>\n'
DIFFICULTY_BLOCK = '\nDifficulty requirements:\n- EASY: single fact, no scenario, no reasoning required.\n- MEDIUM: 2-step reasoning, combine ~2 points, brief scenario optional.\n- HARD: 3-5 step reasoning, realistic scenario REQUIRED in traditional medicine context.\nAlways output EASY, MEDIUM, HARD in this order, all three must exist.\n'
TYPE_PROMPTS = {'MCQ': f'You are an expert question writer for traditional medicine.\n\nGenerate THREE multiple-choice items (EASY, MEDIUM, HARD). Each item MUST have exactly five options labeled A., B., C., D., E. with ONE correct option. Provide correct option letter and reasoning.\n\nCRITICAL: MCQ is NOT a dialogue. Do NOT use any "User:" or "Assistant:" labels. Do NOT write multi-turn conversations. Each item should be a single self-contained question plus options.\n\nAvoid any mention of "text/paragraph/source". Keep content self-contained but grounded in the given paragraph. If details are missing, choose the most plausible option consistent with the paragraph without inventing unrelated facts.\n{DIFFICULTY_BLOCK}\n\nUse these few-shot examples as style references (do NOT copy sentences):\n{MCQ_FEW_SHOT}\n', 'QA': f'You are an expert question writer for traditional medicine.\n\nGenerate THREE short-answer items (EASY, MEDIUM, HARD). Provide concise question, short correct answer, and reasoning.\n\nCRITICAL: QA type means a SINGLE direct question followed by a direct answer. \n- DO NOT use dialogue format (no "User:" or "Assistant:" labels)\n- DO NOT create multi-turn conversations\n- Format: Just a question, then an answer\n- If you see "User:" or "Assistant:" in your output, you have made an error - QA should be simple Q&A pairs only\n\nAvoid any mention of "text/paragraph/source". Keep content self-contained but grounded in the given paragraph. If details are limited, provide the best plausible question/answer consistent with it.\n{DIFFICULTY_BLOCK}\n\nUse these few-shot examples as style references (do NOT copy sentences):\n{QA_FEW_SHOT}\n', 'Dialogue': f'You are an expert question writer for traditional medicine.\n\nGenerate THREE multi-turn User–Assistant dialogues (EASY, MEDIUM, HARD). Each dialogue must have two or more turns and end with a question from User. Provide the dialogue (up to the final user question), the correct answer, and reasoning. The setting must clearly be traditional medicine.\nAvoid any mention of "text/paragraph/source". Keep content self-contained but grounded in the given paragraph. If details are limited, provide the best plausible dialogue consistent with it.\n{DIFFICULTY_BLOCK}\n\nUse these few-shot examples as style references (do NOT copy sentences):\n{DIALOGUE_FEW_SHOT}\n'}
OUTPUT_FORMAT = 'Return ONLY in this exact structure (no explanations, no markdown):\n<EASY><q>...</q><a>...</a><cot>...</cot></EASY>\n<MEDIUM><q>...</q><a>...</a><cot>...</cot></MEDIUM>\n<HARD><q>...</q><a>...</a><cot>...</cot></HARD>\n- MCQ: question must include options A. B. C. D. E.\n- QA: MUST be a single direct question (NO "User:" or "Assistant:" labels, NO dialogue format)\n- Dialogue: MUST include \'User:\' and \'Assistant:\' turns (multi-turn conversation format)\nIf absolutely impossible, return <FAIL>.\n'
_FORBIDDEN_SOURCE_PATTERNS = ['paragraph', 'source text', 'the text', 'the paragraph']

def sanitize_output_text(text: str) -> str:
    if not text:
        return ''
    if len(text.strip()) < 10:
        return text.strip()
    sanitize_prompt = f'You are a text editor. Your task is to clean the following text by removing any references to source materials (like "the text", "the paragraph", "according to the source", etc.) while keeping the sentence grammatically correct and coherent.\n\nRules:\n1. Remove phrases like "according to the text", "the paragraph mentions", "in the source", etc.\n2. Keep the meaning and flow of the sentence intact\n3. If removing a phrase makes the sentence incomplete, rewrite it to be complete\n4. Do NOT add new information\n5. Return ONLY the cleaned text, no explanations\n\nText to clean:\n{text}\n\nCleaned text:'
    ok, cleaned = openai_tool.get_respons(sanitize_prompt, model=MODEL_NAME)
    if not ok:
        cleaned = text
        for pat in _FORBIDDEN_SOURCE_PATTERNS:
            cleaned = re.sub(pat, '', cleaned, flags=re.I)
        return ' '.join(cleaned.split()).strip()
    return cleaned.strip()

def build_gen_prompt(text: str, subject: str, qtype: str, question_template: str, few_shot: List[Dict], relaxed: bool=False) -> str:
    lang = detect_lang(text)
    lang_note = 'The paragraph is in Hindi (Devanagari script); use Hindi for Q/A/CoT.' if lang == 'hi' else 'The paragraph is not in Devanagari; use English for Q/A/CoT.'
    prompt_head = TYPE_PROMPTS.get(qtype, TYPE_PROMPTS['QA'])
    tpl_text = question_template.strip() if question_template else 'No fixed template; craft your own wording.'
    dyn_fs_block = ''
    if few_shot:
        parts = []
        for i, ex in enumerate(few_shot, 1):
            para = ex.get('paragraph', '').strip()
            q = ex.get('q', '').strip()
            a = ex.get('a', '').strip()
            cot = ex.get('cot', '').strip()
            parts.append(f'Example {i} (from pool):\n' + (f'Paragraph:\n{para}\n' if para else '') + f"<q>{q}</q>\n<a>{a}</a>\n<cot>{cot or 'Reasoning consistent with the answer.'}</cot>\n")
        dyn_fs_block = '\nAdditional few-shot examples:\n----------------------------------------\n' + '\n'.join(parts) + '----------------------------------------\n'
    relax_note = ''
    if relaxed:
        relax_note = '\nRelaxed mode: If the paragraph lacks detail, still propose plausible traditional medicine items without contradicting the given text.'
    prompt = prompt_head + relax_note + '\n\nSubject focus: ' + subject + '\n' + lang_note + '\nTemplate hint (optional, do not mention template in output):\n' + tpl_text + dyn_fs_block + '\n\nSource paragraph (treat as the only knowledge, do NOT mention it explicitly):\n' + text.strip() + '\n\n' + OUTPUT_FORMAT
    return prompt

def parse_multi_qa_cot(text: str) -> List[Dict]:
    t = (text or '').strip()
    if t.upper() == '<FAIL>':
        return []
    results: List[Dict] = []
    for diff in ['EASY', 'MEDIUM', 'HARD']:
        m = re.search(f'<{diff}>(.*?)</{diff}>', t, re.S | re.I)
        if not m:
            continue
        block = m.group(1)
        q = Q_RX.search(block)
        a = A_RX.search(block)
        c = COT_RX.search(block)
        if q and a and c:
            results.append({'difficulty': diff.lower(), 'question': q.group(1).strip(), 'answer': a.group(1).strip(), 'cot': c.group(1).strip()})
    if not results:
        q = Q_RX.search(t)
        a = A_RX.search(t)
        c = COT_RX.search(t)
        if q and a and c:
            results.append({'difficulty': '', 'question': q.group(1).strip(), 'answer': a.group(1).strip(), 'cot': c.group(1).strip()})
    return results

def generate_qa_for_entry(entry: Dict) -> List[Dict]:
    text = entry['text']
    subject = entry['subject']
    qtype = entry['type']
    question_template = entry.get('question_templete', '')
    few_shot = entry.get('few_shot', [])
    prompt = build_gen_prompt(text, subject, qtype, question_template, few_shot)
    ok, content = openai_tool.get_respons(prompt, model=MODEL_NAME)
    if not ok:
        if DEBUG:
            print('[warn] QA generation failed, raw:', content[:200])
        ok, content = openai_tool.get_respons(build_gen_prompt(text, subject, qtype, question_template, few_shot, relaxed=True), model=MODEL_NAME)
        if not ok:
            return []
    qa_list = parse_multi_qa_cot(content)
    if not qa_list:
        ok_relax, content_relax = openai_tool.get_respons(build_gen_prompt(text, subject, qtype, question_template, few_shot, relaxed=True), model=MODEL_NAME)
        if ok_relax:
            qa_list = parse_multi_qa_cot(content_relax)
    for qa in qa_list:
        qa['question'] = sanitize_output_text(qa.get('question', ''))
        qa['answer'] = sanitize_output_text(qa.get('answer', ''))
        qa['cot'] = sanitize_output_text(qa.get('cot', ''))
    if not qa_list and DEBUG:
        print('[warn] QA parse failed after relaxed retry.')
    return qa_list

def process_one_file(src: Path, out_dir: Path, raw_id: str) -> bool:
    out_path = out_dir / (src.stem + '.qa.json')
    if out_path.exists() and (not FORCE_OVERWRITE):
        try:
            if out_path.stat().st_size > 0:
                print(f'[skip] exists: {out_path.name}')
                return True
        except Exception:
            pass
    try:
        raw_obj = load_merged_json(src)
        blocks = flatten_paragraph_blocks(raw_obj)
    except Exception as e:
        print(f'[error] load json failed: {src} ({e})')
        return False
    print(f'[info] {src.name}: found {len(blocks)} paragraphs')
    all_entries: List[Dict] = []
    for idx, blk in enumerate(blocks, start=1):
        text = (blk.get('text') or '').strip()
        if not text:
            continue
        page_idx = blk.get('page_idx')
        text_id = f'{idx:05d}'
        subject_list, type_list = classify_subject_and_type(text)
        entry_counter = 0
        for subject in subject_list:
            for qtype in type_list:
                entry_counter += 1
                entry_id = f'{entry_counter:02d}'
                _id = f'{raw_id}{text_id}{entry_id}'
                qtpl = get_question_template(subject, qtype)
                fs = get_few_shot(subject, qtype)
                entry = {'id': _id, 'text': text, 'subject': subject, 'type': qtype, 'question_templete': qtpl, 'few_shot': fs, 'metadata': {'raw_id': raw_id, 'text_id': text_id, 'entry_id': entry_id, 'page_idx': page_idx}}
                generate_qa_for_entry(entry)
                entry.pop('few_shot', None)
                all_entries.append(entry)
        print(f'  - text_id={text_id}, subjects={subject_list}, types={type_list}, entries={entry_counter}')
        time.sleep(SLEEP_BETWEEN_REQ)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[OK] {src.name} -> {out_path.name}  (entries: {len(all_entries)})')
    return True

def iter_input_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return [p for p in root.rglob('*.json') if p.is_file()]
    return [p for p in root.glob('*.json') if p.is_file()]

def main():
    in_dir, out_dir = (INPUT_DIR.resolve(), OUTPUT_DIR.resolve())
    if not in_dir.exists() or not in_dir.is_dir():
        print(f'[error] input dir invalid: {in_dir}')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = load_templates_pool(INSTR_XLSX_PATH)
    _ = load_examples_pool(EXAMPLES_XLSX_PATH)
    files = iter_input_files(in_dir, RECURSIVE)
    print(f'[info] found {len(files)} json files in {in_dir}')
    cnt = 0
    for i, src in enumerate(files, 1):
        raw_id = f'{i:03d}'
        print(f'\n[{i}/{len(files)}] {src.name} (raw_id={raw_id})')
        if process_one_file(src, out_dir, raw_id):
            cnt += 1
    print(f'\n== done == processed {cnt}/{len(files)} files → {out_dir}')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Q-A-CoT JSON from merged paragraph JSON files.')
    parser.add_argument('--input-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--instr-xlsx', required=True, type=str)
    parser.add_argument('--examples-xlsx', default='', type=str)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--force-overwrite', action='store_true')
    parser.add_argument('--cache', default='.subject_type_cache.json', type=str)
    parser.add_argument('--api-key', default=os.getenv('API_KEY', ''), type=str)
    parser.add_argument('--api-base-url', default=os.getenv('API_BASE_URL', ''), type=str)
    parser.add_argument('--model', default=os.getenv('MODEL_NAME', 'gpt-4o'), type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    INSTR_XLSX_PATH = Path(args.instr_xlsx)
    EXAMPLES_XLSX_PATH = Path(args.examples_xlsx) if args.examples_xlsx else Path('')
    RECURSIVE = bool(args.recursive)
    FORCE_OVERWRITE = bool(args.force_overwrite)
    CLASS_CACHE_PATH = Path(args.cache)
    API_KEY = args.api_key
    API_BASE_URL = args.api_base_url
    MODEL_NAME = args.model
    DEBUG = bool(args.debug)
    main()
