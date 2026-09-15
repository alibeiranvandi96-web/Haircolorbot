"""
ربات رنگ مو (Aihaircolorbot)
============================
نسخه‌ی ساده‌شده: پاسخ با درک مدل از منابع، بدون قوانین سخت‌گیرانه.

  * مدل:            Qwen3.8-Flash (فقط b.ai API)
  * دانشنامه:       همه فایل‌های .txt / .text کنار همین فایل (آپدیت خودکار با mtime)
  * ورودی تلگرام:   POST /telegram
  * بررسی سلامت:    GET  /health
  * اجرای محلی:     python app.py
  * پراداکشن:       gunicorn app:app --bind 0.0.0.0:$PORT

رویکرد این نسخه
----------------
  1) «منطق دامنه» و واژه‌نامه‌ی رنگ از متن خودِ منابع استخراج می‌شود.
  2) «درک سوال» با قوانین ساده: نوع سوال (کلیدواژه)، پایه فعلی/هدف، رنگ هدف.
  3) «جستجو» فقط با تعداد کلمات مشترک (بدون وزن‌دهی پیچیده) + Q&A با شباهت بالای ۷۰٪.
  4) «پرامپت» کوتاه و شفاف؛ مدل آزادی دارد و فقط به سوال کاربر جواب می‌دهد.
"""

import glob
import html
import json
import math
import os
import re
import time
from difflib import SequenceMatcher

from pathlib import Path

import requests
from flask import Flask, jsonify, request

# ------------------------------------------------------------------ تنظیمات

BAI_API_KEY = os.getenv("BAI_API_KEY", "").strip()
BAI_URL = os.getenv("BAI_URL", "https://api.b.ai/v1/chat/completions").strip() or "https://api.b.ai/v1/chat/completions"
BAI_MODEL = os.getenv("BAI_MODEL", "Qwen3.8-Flash").strip() or "Qwen3.8-Flash"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "").strip().rstrip("/")

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILES = (
    list(BASE_DIR.glob("*.txt")) +
    list(BASE_DIR.glob("*.text"))
)

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_PATTERNS = ("*.txt", "*.text", "*.tex")
IGNORED_FILES = {"requirements.txt"}

MAX_SOURCE_CHARS = 40000   # سقف کاراکتر منابع در پرامپت
MAX_CHUNK_CHARS = 1200     # طول تقریبی هر تکه
MAX_RESULT_CHUNKS = 15     # حداکثر تکه‌های ارسالی به مدل
COLOR_MAP_FILE = "color_map.txt"
QA_FILE = "haircolor_q&a.txt"
QA_SIMILARITY_THRESHOLD = 0.70
LLM_ANALYSIS_ENABLED = os.getenv("LLM_ANALYSIS_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
LLM_ANALYSIS_TIMEOUT = int(os.getenv("LLM_ANALYSIS_TIMEOUT", "45"))
MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_CHARS = 9000
KNOWLEDGE_FILE = "knowledge.txt"
OXIDIZER_FILE = "oxidizer.txt"
GRAY_HAIR_FILE = "gray_hair.txt"
BLEACH_FILE = "bleach.txt"
GLOSSING_FILE = "glossing.txt"
BASE_PREP_FILE = "base_prep.txt"
COLORING_METHODS_FILE = "coloring_methods.text"

MSG_SERVICE_UNAVAILABLE = "⚠️ سرویس در دسترس نیست. (کلید BAI_API_KEY تنظیم نشده است.)"
MSG_ANSWER_FAILED = "⚠️ الان نمی‌توانم پاسخ بدهم. لطفاً چند لحظه دیگر دوباره بپرسید."
START_MESSAGE = (
    "سلام 👋 من دستیار تخصصی رنگ مو هستم.\n"
    "سوال خود را بپرسید — وضعیت مو، پایه و رنگی که می‌خواهید را بگویید تا "
    "دقیق‌ترین جواب را از منابع تخصصی برایتان استخراج کنم.\n\n"
    "مثال: «موهام پایه 3 نچرال دارم، میخوام بلوند دودی بشم، ترکیبش چیه؟»"
)

STOPWORDS = {
    "و", "که", "در", "از", "به", "با", "این", "آن", "اینها", "است", "هست", "هستند",
    "برای", "چه", "چی", "چطور", "چگونه", "کدام", "چند", "می", "رو", "ها", "های",
    "یک", "یا", "هم", "تا", "روی", "میشه", "می‌شود", "باید", "کنیم", "کنم", "کنید",
    "میخوام", "میخواهم", "سلام", "خسته", "نباشید", "لطفا", "اگه", "اگر",
    "بود", "بودن", "شده", "شدم", "بده", "بدهید", "بدم", "بدید",
    "کنه", "بکنه", "داره", "دارم", "دار", "می‌کنم",
    "the", "and", "for", "with", "that", "this",
}

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

# ------------------------------------------------------------------ ابزار متن


def log(message):
    """لاگ ساده که بلافاصله در خروجی سرور دیده می‌شود."""
    print(message, flush=True)


def _normalize(text):
    """ی/ک عربی، ارقام فارسی و نیم‌فاصله را یکدست می‌کند."""
    text = text.translate(_DIGITS)
    text = text.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    return text.lower()


def _tokens(text):
    """کلمه‌های معنادار یک متن را برمی‌گرداند."""
    words = re.findall(r"[\w\u0600-\u06FF]+", _normalize(text))
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def _read_knowledge_file(path):
    """یک فایل دانشنامه را با چند انکودینگ مختلف می‌خواند."""
    for encoding in ("utf-8", "utf-8-sig", "cp1256"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _discover_files():
    """همه مسیرهای فایل‌های دانشنامه را پیدا می‌کند."""
    result = {}
    for path in sorted(set(str(p) for p in KNOWLEDGE_FILES)):
        name = os.path.basename(path)
        if name in IGNORED_FILES or os.path.basename(__file__) == name:
            continue
        if not os.path.isfile(path):
            continue
        result[path] = name
    return result


def _split_chunks(text):
    """
    متن را به تکه‌های تقریبی MAX_CHUNK_CHARS کاراکتری می‌شکند.
    خط‌چین‌ها (=== یا ---) مرز قطعی بخش هستند.
    """
    text = re.sub(r"[ \t]*={3,}[ \t]*", "\x00", text)
    text = re.sub(r"[ \t]*-{3,}[ \t]*", "\x00", text)

    chunks, current, size = [], [], 0

    def flush():
        if current:
            chunks.append("\n\n".join(current))

    for paragraph in re.split(r"\n\s*\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for piece in paragraph.split("\x00"):
            piece = piece.strip()
            if not piece:
                flush()
                current, size = [], 0
                continue
            if current and size + len(piece) > MAX_CHUNK_CHARS:
                flush()
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 2
    flush()

    small = []
    for chunk in chunks:
        while len(chunk) > MAX_CHUNK_CHARS:
            cut = chunk.rfind("\n", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > MAX_CHUNK_CHARS // 2 else MAX_CHUNK_CHARS
            small.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if chunk:
            small.append(chunk)
    return small


# ---------------------------------------------------------- منطق استخراج‌شده
#
# «قانون»های تصمیم‌گیری (مثل حداکثر روشن‌سازی موی نچرال) از متن خودِ منابع
# خوانده می‌شود؛ اگر دانشنامه عوض شود، منطق هم دنبالش عوض می‌شود.

_DOMAIN_FALLBACK = {
    "max_natural_lift": 3,     # موی نچرال با رنگ معمولی حداکثر چند پایه روشن می‌شود
    "warm": {"شکلاتی", "قرمز", "مسی", "بژ گرم", "طلایی"},
    "cool": {"دودی", "زیتونی", "بنفش", "بژ سرد"},
    "neutral": {"طبیعی"},
}


def _derive_domain_rules(documents):
    """
    منطق تصمیم‌گیری را از متن منابع استخراج می‌کند:
      - حداکثر روشن‌سازی موی نچرال (تا N پایه)
      - آستانه‌ی مسیر دو مرحله‌ای (اختلاف بیش از N پایه)
      - دسته‌بندی گرم/سرد/خنثی رنگ‌های مادر (از knowledge.txt)
    """
    full = "\n".join(doc["text"] for doc in documents)
    rules = dict(_DOMAIN_FALLBACK)
    rules["warm"] = set(_DOMAIN_FALLBACK["warm"])
    rules["cool"] = set(_DOMAIN_FALLBACK["cool"])
    rules["neutral"] = set(_DOMAIN_FALLBACK["neutral"])

    # ۱) حداکثر روشن‌سازی موی نچرال با رنگ معمولی
    lift_cap = None
    for m in re.finditer(r"(?:تا|نهایتا)\s*(\d{1,2})\s*پایه", full):
        window = full[max(0, m.start() - 60): m.end() + 60]
        if any(w in window for w in ("روشن", "روشنی", "روشنایی", "لیفت", "نچرال")):
            lift_cap = max(lift_cap or 0, int(m.group(1)))
    m = re.search(r"حداکثر\s*(\d{1,2})\s*تا\s*(\d{1,2})\s*پایه", full)
    if m:
        lift_cap = max(lift_cap or 0, int(m.group(2)))
    rules["max_natural_lift"] = lift_cap or _DOMAIN_FALLBACK["max_natural_lift"]

    # ۲) آستانه‌ی مسیر دو مرحله‌ای («بیشتر از N پایه» در منابع)
    two_stage = rules["max_natural_lift"]
    for m in re.finditer(r"بیش(?:تر)?\s*از?\s*(\d{1,2})\s*پایه", full):
        two_stage = max(two_stage, int(m.group(1)))
    rules["two_stage_threshold"] = two_stage

    # ۳) رنگ‌های گرم/سرد/خنثی (knowledge.txt صریحاً لیست کرده)
    for pattern, target in (
        (r"رنگ گرم داریم\s*[:：]\s*([^\n]+)", rules["warm"]),
        (r"رنگ سرد داریم\s*[:：]\s*([^\n]+)", rules["cool"]),
        (r"رنگ خنثی داریم\s*[:：]\s*([^\n]+)", rules["neutral"]),
    ):
        m = re.search(pattern, full)
        if m:
            for word in re.split(r"[،,؛;]|\sو\s", m.group(1)):
                word = word.strip()
                if word:
                    target.add(_normalize(word))

    log(
        f"[domain] منطق استخراج‌شده از منابع: روشن‌سازی موی نچرال تا "
        f"{rules['max_natural_lift']} پایه؛ دو مرحله‌ای وقتی اختلاف بیشتر از "
        f"{rules['two_stage_threshold']} پایه؛ گرم={sorted(rules['warm'])}؛ "
        f"سرد={sorted(rules['cool'])}."
    )
    return rules


# ------------------------------------------------ واژه‌نامه‌ی رنگ (از منابع)
#
# COLOR_TARGET_BASE : پایه‌ی نتیجه/حداقل پایه‌ی لازم هر رنگ — خلاصه‌ی
#                     «بخش ۴» رنگ‌ها در color_map.txt.
# COLOR_WARMTH      : گرم/سرد/خنثی بودن هر رنگ — خلاصه‌ی knowledge.txt و
#                     color_map.txt.
# COLOR_SYNONYMS    : مترادف‌های عامیانه‌ی هر رنگ برای جستجوی هوشمندتر.

COLOR_TARGET_BASE = {
    "کرم": 8, "کاهی": 10, "کنفی": 10, "بیسکویتی": 10, "استخوانی": 11, "یخی": 12, "صدفی": 11,
    "موکا": 6, "دودی": 9, "خاکستری": 9, "شامپاینی": 8, "طلایی": 7, "عسلی": 7, "کاراملی": 8,
    "مسی": 7, "رزگلد": 8, "قرمز": 6, "آلبالویی": 6, "شرابی": 5, "بنفش": 6, "یاسی": 9,
    "بادمجونی": 5, "ماهگونی": 5, "صورتی": 9, "شکلاتی": 6, "شیرشکلاتی": 7, "نسکافه ای": 8,
    "فندوقی": 6, "بلوطی": 6, "دارچینی": 6, "شاه بلوطی": 5, "زیتونی": 8, "بژ": 8,
    "بژ سرد": 8, "بژ گرم": 8, "مرواریدی": 9, "شنی": 9, "ماسه ای": 9, "کالباسی": 9,
    "بلوند": 9, "نقره ای": 10, "آفتابی": 8, "قهوه ای": 5, "طبیعی": 5,
}

COLOR_WARMTH = {
    "کرم": "warm", "کاهی": "warm", "کنفی": "warm", "بیسکویتی": "warm", "استخوانی": "cool",
    "یخی": "cool", "صدفی": "cool", "موکا": "cool", "دودی": "cool", "خاکستری": "cool",
    "شامپاینی": "warm", "طلایی": "warm", "عسلی": "warm", "کاراملی": "warm", "مسی": "warm",
    "رزگلد": "warm", "قرمز": "warm", "آلبالویی": "warm", "شرابی": "warm", "بنفش": "cool",
    "یاسی": "cool", "بادمجونی": "cool", "ماهگونی": "warm", "صورتی": "cool", "شکلاتی": "warm",
    "شیرشکلاتی": "warm", "نسکافه ای": "warm", "فندوقی": "warm", "بلوطی": "warm",
    "دارچینی": "warm", "شاه بلوطی": "warm", "زیتونی": "cool", "بژ": "neutral", "بژ سرد": "cool",
    "بژ گرم": "warm", "مرواریدی": "cool", "شنی": "warm", "ماسه ای": "warm", "کالباسی": "neutral",
    "بلوند": "neutral", "نقره ای": "cool", "آفتابی": "warm", "قهوه ای": "neutral", "طبیعی": "neutral",
}

COLOR_SYNONYMS = {
    "کرم": ["کرم", "کرم روشن"], "کاهی": ["کاهی"], "کنفی": ["کنفی"], "بیسکویتی": ["بیسکویتی", "بیسکویت"],
    "استخوانی": ["استخوانی"], "یخی": ["یخی"], "صدفی": ["صدفی"],
    "موکا": ["موکا"], "دودی": ["دودی", "اش", "اسموکی"], "خاکستری": ["خاکستری", "گری", "طوسی"],
    "شامپاینی": ["شامپاینی", "شامپاین", "شامپاینه"], "طلایی": ["طلایی", "گلدن"],
    "عسلی": ["عسلی"], "کاراملی": ["کاراملی", "کارامل"], "مسی": ["مسی", "نارنجی مسی"],
    "رزگلد": ["رزگلد", "رز گلد"], "قرمز": ["قرمز", "قرمز جیغ"], "آلبالویی": ["آلبالویی", "آلبالو"],
    "شرابی": ["شرابی"], "بنفش": ["بنفش", "ویولت"], "یاسی": ["یاسی", "لیلاک", "یاسی روشن"],
    "بادمجونی": ["بادمجونی", "بادمجون"], "ماهگونی": ["ماهگونی", "ماهون"], "صورتی": ["صورتی", "پینک"],
    "شکلاتی": ["شکلاتی", "شکلات", "براون", "چاکلت"], "شیرشکلاتی": ["شیرشکلاتی", "شیر شکلاتی"],
    "نسکافه ای": ["نسکافه ای", "نسکافه", "نسکافه‌ای"], "فندوقی": ["فندوقی", "فندقی"], "بلوطی": ["بلوطی"],
    "دارچینی": ["دارچینی", "دارچین"], "شاه بلوطی": ["شاه بلوطی", "شاه‌بلوطی"], "زیتونی": ["زیتونی", "زیتون", "زیتونی سرد"],
    "بژ": ["بژ", "بژ طبیعی"], "بژ سرد": ["بژ سرد", "بژسرد"], "بژ گرم": ["بژ گرم", "بژگرم"],
    "مرواریدی": ["مرواریدی", "مروارید"], "شنی": ["شنی"], "ماسه ای": ["ماسه ای", "ماسه‌ای"], "کالباسی": ["کالباسی"],
    "بلوند": ["بلوند", "بلوندی", "بلوند روشن", "بلوند دودی", "بلوند طلایی"], "نقره ای": ["نقره ای", "نقره", "سیلور", "نقره‌ای"], "آفتابی": ["آفتابی"],
    "قهوه ای": ["قهوه ای", "قهوه‌ای"], "طبیعی": ["طبیعی", "نچرال", "نترال"],
}

# کلمات رنگ که «هدف/تناژ» نیستند و نباید به‌عنوان رنگ هدف در نظر گرفته شوند
_TARGET_IGNORE = {"طبیعی"}


def _build_color_lexicon(domain):
    """واژه‌نامه‌ی نهایی رنگ‌ها: مترادف + گرم/سرد + پایه‌ی هدف."""
    lexicon = {}
    keys = set(COLOR_TARGET_BASE) | set(COLOR_WARMTH) | set(COLOR_SYNONYMS)
    keys |= domain["warm"] | domain["cool"] | domain["neutral"]
    for key in keys:
        key = _normalize(key)
        syns = [key] + [_normalize(s) for s in COLOR_SYNONYMS.get(key, [])]
        warmth = COLOR_WARMTH.get(key)
        if not warmth:
            if key in domain["cool"]:
                warmth = "cool"
            elif key in domain["warm"]:
                warmth = "warm"
            elif key in domain["neutral"]:
                warmth = "neutral"
        lexicon[key] = {
            "synonyms": list(dict.fromkeys(syns)),
            "warmth": warmth or "unknown",
            "target_base": COLOR_TARGET_BASE.get(key),
        }
    return lexicon


# ------------------------------------------------------------ فهم وضعیت مو

_HAIR_STATE_TERMS = {
    "نچرال": {
        "query": [
            "نچرال", "طبیعی", "خام", "دست نخورده", "بدون دکلره", "بی دکلره",
            "دکلره نشده", "دکلره نکردم", "رنگ نشده", "رنگ نکردم", "رنگی نزده",
            "هیچ رنگی نزده", "رنگ نداره", "بکر", "نترال", "نچرالم", "نچرالمه",
        ],
        "chunk": [
            "نچرال", "موی طبیعی", "مو طبیعی", "موهای طبیعی", "موی نچرال",
            "مو نچرال", "بدون دکلره", "رنگ نشده", "موی بکر",
        ],
    },
    "رنگ‌شده": {
        "query": [
            "رنگ شده", "رنگ خورده", "رنگ کرده", "رنگ زده", "رنگ داره", "رنگ دارم",
            "قبلا رنگ", "رنگ قبلی", "تیوپ زده", "مشکی زده", "حنا", "حنایی",
            "کراتین", "تراپی", "رنگ مصنوعی", "تکرار رنگ", "تکرار تیوپ", "رنگ تیره زده",
            "شرابی زده", "قرمز زده", "ماهگونی زده", "بادمجونی زده",
        ],
        "chunk": [
            "رنگ شده", "رنگ خورده", "قبلا رنگ", "رنگ قبلی", "رنگ تیره زده",
            "تکرار تیوپ", "تکرار رنگ", "حنایی", "حنا", "کراتین", "رنگ مصنوعی",
        ],
    },
    "دکلره": {
        "query": [
            "دکلره", "بلیچ", "پودر دکلره", "خمیر دکلره", "کرم دکلره", "بی رنگ",
            "بیرنگ", "دکوپاژ", "دکا پاژ", "هایلیفت", "های لیفت", "پایه دکلره دارم",
            "دکلره دارم", "دکلره شده", "دکلره کردم",
        ],
        "chunk": [
            "دکلره", "بی رنگ", "بیرنگ", "دکوپاژ", "دکا پاژ",
        ],
    },
}

_APPLICATION_TERMS = {
    "رنگساژ": ["رنگساژ", "رنگ ساژ", "تونر", "تناژ", "رنگساژ کردن"],
    "سفیدی": ["سفیدی", "مو سفید", "پوشش سفیدی", "پوشش سفید", "کاور سفیدی", "سفید شده", "سفیدی ریشه"],
    "پلاژ": ["پلاژ", "پلاژ کردن"],
    "دکوپاژ": ["دکوپاژ", "دکا پاژ", "دکاپاژ", "رفله‌گیری", "رفله گیری", "رفله"],
    "گلوسینگ": ["گلوسینگ", "کاسه گلوسینگ", "رنگساژ کمکی", "گلوس"],
    "بسترسازی": ["بستر", "بسترسازی", "بستر سازی", "ضد زردی", "ضد نارنجی", "بسترساز"],
    "ترمیم": ["ترمیم", "شارژ مجدد", "ترمیم رنگ"],
    "ریموور": ["ریموور", "ریمور", "کالر بک", "کالربک", "ریموو"],
    "هایلایت": ["هایلایت", "بالیاژ", "سامبره", "امبره", "آمبره", "کالر ملتینگ", "لایت"],
}

_PRELIFT_TERMS = [
    "0/00", "۰/۰۰", "0 00", "پری لایت", "prelighten", "دو مرحله", "دو مرحله ای",
    "دومرحله", "دوباره رنگ", "دوباره رنگی", "پلاژ", "هایلیفت طبیعی",
    "روشن سازی", "روشن‌سازی", "دوبار رنگ",
]

_NEGATION_AFTER = {
    "نیست", "نیس", "نبود", "نباشه", "نه", "نکردم", "نزدم", "نداره", "ندارم",
    "نشده", "نخورده", "نخواهم", "نمیخوام", "نمیخواد",
}
_NEGATION_BEFORE = {"نه"}


def _phrase_negated(normalized, phrase):
    """
    آیا عبارتِ رنگ در متن نفی شده است؟
    - واژه‌های نفی بعد از رنگ («مشکی نیست/نزدم/نمیخوام»)
    - یا «نه» بلافاصله قبل از رنگ («نه مشکی»)
    """
    for m in re.finditer(re.escape(phrase), normalized):
        before = _tokens(normalized[max(0, m.start() - 12): m.start()])[-1:]
        after = _tokens(normalized[m.end(): m.end() + 12])[:2]
        if any(w in _NEGATION_AFTER for w in after):
            return True
        if any(w in _NEGATION_BEFORE for w in before):
            return True
    return False


def _hair_states_of(text, query=False):
    """
    وضعیت(های) موی ذکرشده در متن: نچرال / رنگ‌شده / دکلره.
    عبارت‌های نفی («بدون دکلره»، «دکلره نشده») قبل از جستجو حذف
    می‌شوند تا اشتباه تشخیص داده نشوند.
    """
    normalized = _normalize(text)
    probe = re.sub(
        r"(بدون|بی)\s*دکلره|دکلره\s*(نشده|نکردم|نزدم)", " ", normalized
    )
    states = set()
    for state, groups in _HAIR_STATE_TERMS.items():
        terms = groups["query"] if query else groups["chunk"]
        for term in terms:
            term = _normalize(term)
            if term in probe:
                states.add(state)
                break
    return states


def _applications_of(text):
    """کاربردهای ذکرشده در متن (رنگساژ، سفیدی، دکوپاژ و...) را برمی‌گرداند."""
    normalized = _normalize(text)
    found = set()
    for app, terms in _APPLICATION_TERMS.items():
        if any(_normalize(term) in normalized for term in terms):
            found.add(app)
    return found


def _mentions_prelift(text):
    """آیا متن به روشن‌سازی اولیه / مسیر دو مرحله‌ای اشاره دارد؟"""
    normalized = _normalize(text)
    return any(_normalize(term) in normalized for term in _PRELIFT_TERMS)


# ------------------------------------------------------------- تشخیص پایه
#
# نام پایه‌های نچرال -> شماره پایه (از knowledge.txt «۱۰ رنگ پایه‌های نچرال مو»)

_NATURAL_BASE_NAMES = [
    (["مشکی"], "1"),
    (["قهوه ای خیلی تیره", "قهوه خیلی تیره"], "2"),
    (["قهوه ای تیره", "قهوه تیره", "قهوه سوخته"], "3"),
    (["قهوه ای متوسط", "قهوه متوسط"], "4"),
    (["قهوه ای روشن", "قهوه روشن"], "5"),
    (["بلوند تیره"], "6"),
    (["بلوند متوسط"], "7"),
    (["بلوند روشن"], "9"),
    (["بلوند بسیار روشن"], "10"),
]

_BASE_PATTERN = re.compile(r"پایه\s*(?:دکلره\s*|نچرال\s*|مو\s*|موهام\s*|ام\s*)?(\d{1,2})")
_BASE_WORD_PATTERN = re.compile(r"پایه\s*(یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|یازده|دوازده)(?!\s*مرحله)(?![\w])")
_BASE_WORDS = {
    "یک": "1", "دو": "2", "سه": "3", "چهار": "4", "پنج": "5",
    "شش": "6", "هفت": "7", "هشت": "8", "نه": "9", "ده": "10",
    "یازده": "11", "دوازده": "12",
}

_GOAL_PATTERN = re.compile(
    r"میخوام|می خواهم|میخوایم|می خواهیم|میخواد|می خواهد|دوست دارم|بشه|در بیاد|دربیاد"
    r"|بزنم|بزنیم|بزنه|درست کنم|رنگ کنم|رنگش کنم|دربیارم|برسم|هدفم|بشم|میخوام بشم"
)
_EXPLAIN_PATTERN = re.compile(r"چیه|چیست|چی هست|یعنی|معنی|مفهوم|تفاوت|فرق|چه فرقی|فرقش|منظور|یعنی چه|چرا|چجوری")


def _bases_of(text):
    """شماره پایه‌های ذکرشده در متن («پایه ۸»، «پایه 6»، «پایه پنج») را برمی‌گرداند."""
    normalized = _normalize(text)
    found = set(_BASE_PATTERN.findall(normalized))
    found.update(_BASE_WORDS[word] for word in _BASE_WORD_PATTERN.findall(normalized))
    return found


def _bases_with_positions(text):
    """پایه‌ها را به‌همراه موقعیتشان در متن برمی‌گرداند (برای تشخیص ترتیب)."""
    normalized = _normalize(text)
    results = []
    for m in _BASE_PATTERN.finditer(normalized):
        try:
            results.append({"base": m.group(1), "start": m.start(), "end": m.end()})
        except Exception:
            continue
    for m in _BASE_WORD_PATTERN.finditer(normalized):
        try:
            base_num = _BASE_WORDS.get(m.group(1))
            if base_num:
                results.append({"base": base_num, "start": m.start(), "end": m.end()})
        except Exception:
            continue
    return results


def _named_bases_of(text):
    """
    پایه را از روی نام رنگ نچرال مو (مثل «مشکی» یا «قهوه‌ای تیره») حدس می‌زند،
    با در نظر گرفتن نفی («مشکی نیست» پایه ۱ حساب نمی‌شود).
    """
    normalized = _normalize(text)
    found = set()
    for phrases, base in _NATURAL_BASE_NAMES:
        for phrase in phrases:
            phrase = _normalize(phrase)
            if phrase in normalized and not _phrase_negated(normalized, phrase):
                found.add(base)
    return found


def _find_color_families(text):
    """
    نام رنگ‌های شناخته‌شده‌ی ذکرشده در متن (خانواده‌ی کانونی).
    فقط با word boundary دقیق — تک‌کلمه‌ای و چندکلمه‌ای یکسان است.
    """
    normalized = _normalize(text)
    found = set()
    for name, profile in _lexicon().items():
        if name in _TARGET_IGNORE:
            continue
        for syn in profile["synonyms"]:
            syn_norm = _normalize(syn)
            if len(syn_norm) < 2:
                continue
            pattern = r"(?<!\w)" + re.escape(syn_norm) + r"(?!\w)"
            if re.search(pattern, normalized) and not _phrase_negated(normalized, syn_norm):
                found.add(name)
                break
    return found


# --------------------------------------------------------- تشخیص نوع سوال
#
# فقط کلیدواژه‌های ساده — بدون حکم معنایی:
#   - هر نوعی که کلیدواژه‌اش در سوال هست، ثبت می‌شود (بیشترین تعداد = اولویت)
#   - «چیه/تفاوت/یعنی» => theory
#   - رنگ هدف + فعل قصد => formula
#   - بقیه => general

QUESTION_TYPE_TERMS = {
    "formula": {
        "strong": [
            "ترکیب", "فرمول", "فرمول نویسی", "فرمولش", "شماره تیوپ", "واریاسیون",
            "واریاسون", "چی بریزم", "چه ترکیبی", "چه رنگی", "چند میل بریزم",
            "چی بزنم", "چه شماره بزنم", "ترکیب رنگ", "فرمول رنگ",
        ],
        "terms": [
            "اکسیدان", "تیوپ", "چند میل", "میلی", "نسبت رنگ", "در بیاد", "دربیارم",
            "رنگ کنم", "رنگش کنم", "چه شماره", "چی ترکیب کنم", "چجوری ترکیب",
            "میل", "گرم", "سانت",
        ],
    },
    "toner": {
        "strong": [
            "رنگساژ", "رنگ ساژ", "گلوسینگ", "کاسه گلوسینگ", "بستر سازی", "ضد زردی",
            "ضد نارنجی", "تونر", "زرد شده", "نارنجی شده", "رفله نارنجی", "زردی مو",
            "نارنجی مو", "رفله زرد", "بستر",
        ],
        "terms": ["رفله", "رفله گیری", "زردی", "نارنجی", "کدر", "شاین", "براق", "لیفت"],
    },
    "gray": {
        "strong": [
            "سفیدی", "پوشش سفیدی", "کاور سفیدی", "پلاژ", "مو سفید", "موی سفید",
            "ریشه سفید", "سفیدی ریشه", "سفیدی مو", "پوشش سفید",
        ],
        "terms": ["سفید", "کاور", "پوشش", "سفیدی"],
    },
    "care": {
        "strong": [
            "مراقبت", "آسیب", "اسیب", "سوخت", "سوزش", "کف سر", "حساسیت", "تست حساسیت",
            "کش اومد", "کش آمد", "احیا", "پلکس", "موخوره", "خطر", "عوارض", "سلامت مو",
            "ریزش", "سوختگی", "الرژی",
        ],
        "terms": [
            "نسوزه", "نسوزد", "می سوزه", "پوست سر", "ماسک مو", "روغن", "وازلین", "خشک",
            "وز", "خارش", "کم حجم", "حجم کم", "نازک", "موی سالم", "آسیب دیده", "ترمیم",
            "شامپو", "نرم کننده", "حالت دهنده", "مشکل", "ضرر", "تقویت",
        ],
    },
    "method": {
        "strong": [
            "روش رنگ", "روش a", "روش b", "روش c", "روش d", "کالر ملتینگ", "سامبره",
            "بالیاژ", "هایلایت", "امبره", "آمبره", "ترتیب رنگ", "روش رنگ کردن",
            "چطور رنگ کنم", "چجوری رنگ کنم",
        ],
        "terms": [
            "ترتیب", "ریشه و ساقه", "ریشه", "ساقه", "مکث", "تایم", "آبکشی", "سرشور",
            "از کجا شروع", "پشت سر", "وسط سر", "فرق سر", "برس رنگ", "نایلون", "شانه",
            "آموزش", "یاد بگیرم", "چطور انجام", "زمان مکث",
        ],
    },
    "bleach": {
        "strong": [
            "دکلره", "بی رنگ کننده", "پودر دکلره", "خمیر دکلره", "کرم دکلره", "دکوپاژ",
            "ریموور", "کالر بک", "هایلیفت", "پاک کردن رنگ", "دکلره کردن", "پایه دکلره",
        ],
        "terms": [
            "بی رنگ", "بیرنگ", "دکا پاژ", "ریمور", "های لیفت", "ترمیم دکلره", "فویل", "لیفت", "پاک کنم", "پاک کردن", "روشن کردن مو", "دکلره",
        ],
    },
    "theory": {
        "strong": [
            "تناژ", "رنگ مادر", "اکسترا", "تفاوت", "فرق بین", "چه فرقی", "یعنی چه",
            "معنی", "مفهوم", "نقشه رنگ", "چیه", "چیست",
        ],
        "terms": ["گرم و سرد", "خنثی", "پایه رنگ", "شماره رنگ", "کارایی رنگ", "فرقش", "چیست", "چرا"],
    },
}

QUESTION_TYPE_LABELS = {
    "formula": "ترکیب رنگ و فرمول‌نویسی",
    "toner": "رنگساژ / گلوسینگ / بسترسازی",
    "gray": "پوشش سفیدی",
    "bleach": "دکلره و بی‌رنگ‌کننده‌ها",
    "method": "روش و ترتیب رنگ‌گذاری",
    "care": "آسیب‌شناسی، مراقبت و ایمنی مو",
    "theory": "شناخت رنگ‌ها، تناژها و پایه‌ها",
    "general": "سوال عمومی",
}

FORMULA_TYPES = {"formula", "toner", "gray"}
COLOR_MAP_TYPES = {"formula", "toner", "gray", "theory"}


def _count_terms(terms, probe, tokens):
    return sum(
        1 for term in terms
        if _normalize(term) in probe or _normalize(term) in tokens
    )


def _question_types(ctx):
    """نوع سوال فقط با کلیدواژه‌های ساده (بدون حکم معنایی)."""
    normalized = ctx["normalized"]
    tokens = set(ctx["tokens"])

    hits = {}
    for qtype, groups in QUESTION_TYPE_TERMS.items():
        count = _count_terms(groups["strong"], normalized, tokens)
        if count == 0:
            count = _count_terms(groups["terms"], normalized, tokens)
        if count:
            hits[qtype] = count

    # «چیه/تفاوت/یعنی» => سوال نظری
    if ctx["is_explain"]:
        hits["theory"] = max(hits.get("theory", 0), 1)
    # رنگ هدف + فعل قصد => سوال فرمول
    if ctx["target_colors"] and ctx["is_goal"]:
        hits["formula"] = max(hits.get("formula", 0), 1)

    if not hits:
        return ["general"]
    return [qtype for qtype, _score in sorted(hits.items(), key=lambda kv: -kv[1])][:3]


def _is_formula_question(types):
    return bool(FORMULA_TYPES & set(types))


def _needs_full_color_map(types):
    return bool(types) and types[0] in COLOR_MAP_TYPES


# ------------------------------------------------------- استدلال مسیر رنگ


def _estimate_bases(ctx):
    """
    پایه فعلی و هدف با قوانین ساده:
    - ۲ پایه در سوال: اولی فعلی، دومی هدف
    - ۱ پایه + رنگ هدف: اون پایه فعلی است و هدف از رنگ هدف می‌آید
    - ۱ پایه + فعل قصد: اون پایه فعلی است
    """
    ordered = [
        m["base"]
        for m in sorted(_bases_with_positions(ctx["question"]), key=lambda m: m["start"])
    ]
    current = None
    target = None
    if len(ordered) >= 2:
        current = int(ordered[0])
        target = int(ordered[1])
    elif ordered:
        current = int(ordered[0])

    # اگر پایه عددی نبود، از نام پایه‌ی نچرال (مثل «مشکی») استفاده کن
    if current is None and ctx["named_bases"]:
        current = min(int(b) for b in ctx["named_bases"])

    # پایه‌ی هدف از رنگ هدف می‌آید (حداکثر پایه‌ی لازم رنگ‌های هدف)
    if target is None and ctx["target_colors"]:
        color_bases = [
            _lexicon().get(c, {}).get("target_base")
            for c in ctx["target_colors"]
        ]
        color_bases = [b for b in color_bases if b is not None]
        if color_bases:
            target = max(color_bases)

    ctx["current_base"] = current
    ctx["target_base"] = target


def _needs_prelift(ctx):
    """
    آیا این سوال احتمالاً به مسیر دو مرحله‌ای نیاز دارد؟
    طبق منابع: موی نچرال با رنگ معمولی حداکثر تا N پایه روشن می‌شود؛
    اختلاف «بیشتر از N پایه» => دوباره‌رنگی (۰/۰۰) یا دکلره ملایم.
    N از خود منابع استخراج می‌شود (نه ثابت کد).
    """
    if _mentions_prelift(ctx["question"]):
        return True
    if "نچرال" not in ctx["states"]:
        return False
    lift = _domain()["max_natural_lift"]
    current, target = ctx["current_base"], ctx["target_base"]
    return current is not None and target is not None and (target - current) > lift


def _target_is_light(ctx):
    """آیا سوال به روشن‌شدن مو یا رسیدن به رنگ/پایه‌ی روشن‌تر اشاره دارد؟"""
    current, target = ctx["current_base"], ctx["target_base"]
    if current is not None and target is not None and target > current:
        return True
    if target is not None and target >= 8:
        return True
    return "روشن" in ctx["normalized"]


def _targets_needing_bleach_on_natural(ctx):
    """
    رنگ‌های هدفی که روی موی نچرال به دکلره نیاز دارند:
    - خودِ تناژهای سرد مادر (از دانشنامه: دودی/زیتونی/بنفش/بژ سرد)
    - یا رنگ‌های روشنی که حداقل پایه ۸ (زرد) به بالا لازم دارند.
    """
    if "نچرال" not in ctx["states"]:
        return set()
    cool_mother = _domain()["cool"]
    result = set()
    for c in ctx["target_colors"]:
        base = _lexicon().get(c, {}).get("target_base")
        if c in cool_mother or (base is not None and base >= 8):
            result.add(c)
    return result


def analyze_question(question):
    """
    درک سوال با قوانین ساده: وضعیت مو، پایه‌ها، رنگ‌ها، نوع سوال
    و مسیر پیشنهادی (دو مرحله‌ای/دکلره).
    """
    question = (question or "").strip()
    ctx = {
        "question": question,
        "normalized": _normalize(question),
        "tokens": _tokens(question),
    }

    # وضعیت و کاربردها
    ctx["states"] = _hair_states_of(question, query=True)
    ctx["bases"] = _bases_of(question)
    ctx["applications"] = _applications_of(question)
    ctx["named_bases"] = _named_bases_of(question)

    # اگر وضعیت نگفته شده ولی پایه/نام پایه هست، مو نچرال فرض می‌شود
    if not ctx["states"] and (ctx["bases"] or ctx["named_bases"]):
        ctx["states"].add("نچرال")

    ctx["is_goal"] = bool(_GOAL_PATTERN.search(ctx["normalized"]))
    ctx["is_explain"] = bool(_EXPLAIN_PATTERN.search(ctx["normalized"]))

    # رنگ‌ها: در سوال هدف‌دار، رنگ‌ها «هدف»؛ در بقیه «زمینه/فعلی» هستند
    colors = _find_color_families(question)
    if ctx["is_goal"]:
        ctx["target_colors"] = colors
        ctx["current_colors"] = set()
    else:
        ctx["target_colors"] = set()
        ctx["current_colors"] = colors

    _estimate_bases(ctx)
    ctx["prelift"] = _needs_prelift(ctx)
    ctx["target_light"] = _target_is_light(ctx)
    ctx["needs_bleach"] = bool(_targets_needing_bleach_on_natural(ctx))
    ctx["removal"] = "رنگ‌شده" in ctx["states"] and ctx["target_light"]
    ctx["types"] = _question_types(ctx)
    ctx["is_formula"] = _is_formula_question(ctx["types"])
    return ctx


# ------------------------------------------------------ بارگذاری دانشنامه

_knowledge_cache = {
    "files": {},
    "documents": [],
    "chunks": [],
    "token_weight": {},
    "loaded_at": 0,
    "color_map_text": "",
    "qa_pairs": [],
    "domain": None,
    "lexicon": None,
    "signature": None,
}

_chat_history = {}



def _domain():
    return _knowledge_cache["domain"] or _DOMAIN_FALLBACK


def _lexicon():
    return _knowledge_cache["lexicon"] or _build_color_lexicon(_domain())


def _enrich_chunk(chunk):
    """ویژگی‌های هر تکه را یک‌بار محاسبه می‌کند (برای جستجوی سریع‌تر)."""
    normalized = _normalize(chunk["text"])
    return {
        "source": chunk["source"],
        "text": chunk["text"],
        "tokens": set(_tokens(chunk["text"])),
        "norm": normalized,
        "bases": _bases_of(chunk["text"]),
        "apps": _applications_of(chunk["text"]),
        "states": _hair_states_of(chunk["text"]),
        "families": _find_color_families(chunk["text"]),
    }


def _build_chunks_and_weights(documents):
    chunks = []
    for doc in documents:
        for chunk_text in _split_chunks(doc["text"]):
            chunks.append(_enrich_chunk({"source": doc["source"], "text": chunk_text}))

    # IDF-like weights: rare and domain-specific terms count more than generic words.
    doc_freq = {}
    for ch in chunks:
        for tok in ch["tokens"]:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    n = max(len(chunks), 1)
    weight = {tok: math.log((n + 1) / (cnt + 0.5)) + 1.0 for tok, cnt in doc_freq.items()}
    return chunks, weight


def _qa_question_key(text):
    """کلید تطبیق دقیق Q&A؛ فقط تفاوت‌های نمایشی را یکدست می‌کند."""
    return re.sub(r"\s+", " ", _normalize(text)).strip()


def _extract_qa_pairs(text):
    """پرسش/پاسخ‌های شماره‌دار Q&A را بدون دست‌کاری متن پاسخ استخراج می‌کند."""
    pattern = re.compile(
        r"^\s*Q\s*(?P<number>\d+)\s*:\s*"
        r"(?P<question>.*?)"
        r"^\s*A\s*(?P=number)\s*:\s*"
        r"(?P<answer>.*?)"
        r"(?=^\s*Q\s*\d+\s*:|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    pairs = []
    for match in pattern.finditer(text or ""):
        question = match.group("question").strip()
        answer = match.group("answer").strip()
        if question and answer:
            pairs.append(
                {
                    "question": question,
                    "question_key": _qa_question_key(question),
                    "answer": answer,
                }
            )
    return pairs


def _find_exact_qa_answer(question):
    """فقط تطبیق دقیق پرسش با Q&A را برمی‌گرداند؛ جستجوی معنایی انجام نمی‌دهد."""
    question_key = _qa_question_key(question)
    if not question_key:
        return None
    for pair in _knowledge_cache["qa_pairs"]:
        if pair["question_key"] == question_key:
            return pair["answer"]
    return None


def _find_similar_qa_pair(question):
    """بهترین پرسش Q&A با شباهت کاراکتریِ بیشتر از آستانه را پیدا می‌کند."""
    question_key = _qa_question_key(question)
    if not question_key:
        return None, 0.0

    best_pair = None
    best_score = 0.0
    for pair in _knowledge_cache["qa_pairs"]:
        score = SequenceMatcher(
            None, question_key, pair["question_key"], autojunk=False
        ).ratio()
        if score > best_score:
            best_pair = pair
            best_score = score

    if best_pair is not None and best_score > QA_SIMILARITY_THRESHOLD:
        return best_pair, best_score
    return None, best_score


def _qa_pair_source_text(pair):
    """یک Q&A کامل را به‌عنوان تنها منبع قابل ارسال به مدل آماده می‌کند."""
    return f"پرسش Q&A:\n{pair['question']}\n\nپاسخ Q&A:\n{pair['answer']}"


def refresh_knowledge(force=False):
    """
    فایل‌های دانشنامه را با os.path.getmtime چک می‌کند؛ فایل جدید/تغییرکرده
    دوباره خوانده می‌شود و ایندکس + منطق دامنه + واژه‌نامه‌ی رنگ از نو ساخته می‌شوند.
    """
    files = _discover_files()
    cache = _knowledge_cache

    changed = False
    for path in list(cache["files"].keys()):
        if path not in files:
            del cache["files"][path]
            changed = True

    for path, name in files.items():
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        cached = cache["files"].get(path)
        if cached is None or cached["mtime"] != mtime or force:
            text = _read_knowledge_file(path).strip()
            if text:
                cache["files"][path] = {"mtime": mtime, "text": text, "name": name}
            else:
                cache["files"].pop(path, None)
            changed = True

    if changed:
        documents = [
            {"source": info["name"], "text": info["text"]}
            for info in cache["files"].values()
        ]
        qa_text = next((doc["text"] for doc in documents if doc["source"] == QA_FILE), "")
        cache["qa_pairs"] = _extract_qa_pairs(qa_text)
        cache["domain"] = _derive_domain_rules(documents)
        cache["lexicon"] = _build_color_lexicon(cache["domain"])
        chunks, weight = _build_chunks_and_weights(documents)
        cache["documents"] = documents
        cache["chunks"] = chunks
        cache["token_weight"] = weight
        cache["loaded_at"] = time.time()
        cache["color_map_text"] = next(
            (info["text"] for info in cache["files"].values() if info["name"] == COLOR_MAP_FILE),
            "",
        )
        total_chars = sum(len(d["text"]) for d in documents)
        log(f"[knowledge] دانشنامه به‌روز شد: {len(documents)} فایل، "
            f"{total_chars} کاراکتر، {len(chunks)} تکه.")
        if not cache["color_map_text"]:
            log("[warning] color_map.txt پیدا نشد! لطفاً فایل را به مخزن اضافه کنید.")
    else:
        log("[knowledge] تغییر جدیدی در دانشنامه پیدا نشد؛ از حافظه استفاده می‌شود.")

    return cache["documents"], cache["chunks"], cache["token_weight"]


refresh_knowledge(force=True)


# ------------------------------------------------ جستجو در دانشنامه


def _source_intent_boost(source, ctx):
    """Prefer documents whose subject matches the detected intent."""
    name = _normalize(source)
    boosts = {
        "gray_hair.txt": {"gray": 7, "formula": 3},
        "oxidizer.txt": {"formula": 4, "care": 3, "bleach": 3},
        "bleach.txt": {"bleach": 7, "formula": 3},
        "glossing.txt": {"toner": 7, "formula": 3},
        "base_prep.txt": {"toner": 5, "bleach": 4, "formula": 2},
        "coloring_methods.text": {"method": 7, "formula": 2},
        "color_map.txt": {"theory": 5, "formula": 4, "toner": 4, "gray": 3},
        "knowledge.txt": {"theory": 4, "formula": 3, "gray": 3, "bleach": 3, "toner": 3},
        "haircolor_q&a.txt": {"theory": 2, "formula": 3, "gray": 3, "toner": 3, "bleach": 3, "method": 2},
    }
    best = 0.0
    for qtype in ctx.get("types", []):
        best = max(best, boosts.get(name, {}).get(qtype, 0))
    return best


def _phrase_overlap(a, b):
    """A small fuzzy similarity for Persian wording variations."""
    a_tokens, b_tokens = set(_tokens(a)), set(_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def _query_variants(question, ctx):
    variants = [question]
    for family in ctx.get("target_colors", set()) | ctx.get("current_colors", set()):
        variants.extend(_lexicon().get(family, {}).get("synonyms", []))
    # Add canonical labels for states/applications so prose such as «سفیدی»
    # retrieves the correct source even when the chunk uses another wording.
    variants.extend(ctx.get("states", []))
    variants.extend(ctx.get("applications", []))
    return variants


def search_knowledge(question, ctx):
    """Hybrid local retrieval: weighted lexical + phrases + domain metadata + Q&A."""
    refresh_knowledge()

    wanted = set()
    for variant in _query_variants(question, ctx):
        wanted.update(_tokens(variant))

    skip = set()
    if _needs_full_color_map(ctx["types"]):
        skip.add(COLOR_MAP_FILE)

    all_chunks = _knowledge_cache["chunks"]
    weights = _knowledge_cache.get("token_weight", {})
    scored = []
    for index, chunk in enumerate(all_chunks):
        if chunk["source"] in skip:
            continue
        overlap = wanted & chunk["tokens"]
        if not overlap and not (ctx.get("target_colors", set()) & chunk.get("families", set())):
            continue

        lexical = sum(weights.get(tok, 1.0) for tok in overlap)
        phrase = _phrase_overlap(question, chunk["text"])
        family_hit = len(ctx.get("target_colors", set()) & chunk.get("families", set()))
        state_hit = len(ctx.get("states", set()) & chunk.get("states", set()))
        app_hit = len(ctx.get("applications", set()) & chunk.get("apps", set()))
        base_hit = len(ctx.get("bases", set()) & chunk.get("bases", set()))
        intent = _source_intent_boost(chunk["source"], ctx)
        score = (lexical * 1.0) + (phrase * 8.0) + (family_hit * 7.0) + (state_hit * 2.5) + (app_hit * 3.5) + (base_hit * 2.0) + intent
        scored.append((score, index))

    # Q&A is evidence, not an exclusive shortcut. Similar Q&A gets a boost and
    # can be combined with technical source chunks for a synthesized answer.
    qa_pair, qa_similarity = _find_similar_qa_pair(question)
    if qa_pair is not None:
        scored.append((qa_similarity * 18.0 + 10.0, -1))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    seen = set()
    total = 0
    budget = MAX_SOURCE_CHARS
    for score, index in scored:
        if index == -1:
            item = {
                "source": QA_FILE,
                "text": _qa_pair_source_text(qa_pair),
                "qa_similarity": qa_similarity,
            }
        else:
            item = all_chunks[index]
        key = (item["source"], item["text"])
        if key in seen:
            continue
        if total + len(item["text"]) > budget and selected:
            continue
        seen.add(key)
        selected.append(item)
        total += len(item["text"])
        if len(selected) >= MAX_RESULT_CHUNKS:
            break

    return selected


def _format_sources(chunks, types):
    parts = []
    color_map_text = _knowledge_cache.get("color_map_text", "")
    qa_only_match = bool(chunks) and all(c.get("qa_only_match") for c in chunks)
    if color_map_text and _needs_full_color_map(types or []) and not qa_only_match:
        parts.append(f"[منبع: {COLOR_MAP_FILE} — مرجع کامل نام رنگ‌ها و کاربردشان]\n{color_map_text}")
    for c in chunks:
        parts.append(f"[منبع: {c['source']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts) if parts else "(منبعی پیدا نشد)"


# ------------------------------------------------------------- پرامپت‌ها
#
# کوتاه و بدون تناقض: مدل آزادی دارد، فقط به سوال کاربر جواب می‌دهد.

SYSTEM_PROMPT_CORE = (
    "تو دستیار تخصصی آموزش رنگ مو هستی و باید قبل از پاسخ، سوال را از نظر وضعیت مو، پایه، هدف، تناژ، "
    "درخواست فرمول/تئوری/روش/اکسیدان و محدودیت‌های کاربر دقیقاً درک کنی. "
    "منابع داده‌شده دانشنامه اصلی هستند؛ از خودت قانون یا فرمول نساز. "
    "اگر چند منبع مرتبط وجود دارد، آنها را با هم ترکیب و یک پاسخ منسجم بده. "
    "اگر اطلاعات کافی نیست، دقیقاً بگو چه اطلاعاتی کم است و فقط در صورت نیاز سوال تکمیلی بپرس. "
    "اگر بین منابع اختلافی دیدی، آن را پنهان نکن و منبع/اختلاف را روشن بیان کن.\n"
    "برای سوال‌های فرمولی، اول وضعیت و پایه را مشخص کن، بعد مسیر رسیدن به هدف، سپس فرمول و نسبت/اکسیدان را فقط اگر منابع پشتیبانی می‌کنند ارائه بده.\n"
    "شماره پایه با شماره تناژ تیوپ یکی نیست؛ آنها را با هم قاطی نکن. "
)

FORMULA_SYSTEM_EXTRA = (
    "تناژهای بین‌المللی طبق دانشنامه: 0 = طبیعی، 1 = دودی، 2 = زیتونی، 3 = طلایی، "
    "4 = مسی، 5 = قرمز، 6 = بنفش، 7 = شکلاتی، 13 = بژ سرد، 31 = بژ گرم. "
    "اسم رنگ و شماره را مطابق منابع بنویس.\n"
)

OUTPUT_FORMAT_RULES = (
    "پاسخ برای تلگرام است. فقط از <b> و <i> برای قالب‌بندی استفاده کن؛ Markdown ننویس. "
    "پاسخ باید مرتبط، کاربردی و کامل باشد؛ نه صرفاً کپی منبع. "
)


def _extract_json_object(text):
    """Extract the first JSON object from a model response."""
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _model_question_analysis(question, history_text=""):
    """Second-pass semantic parsing. It extracts intent/entities but does not answer."""
    if not LLM_ANALYSIS_ENABLED or not BAI_API_KEY:
        return {}
    prompt = (
        "سوال زیر را فقط به JSON تبدیل کن؛ پاسخ تخصصی نده. هیچ اطلاعاتی که در سوال نیست اختراع نکن. "
        "فیلدها: intent (formula/theory/toner/gray/bleach/method/care/general), "
        "hair_state, current_base, target_base, current_color, target_color, applications, "
        "constraints, percent_gray, previous_color, desired_result, missing_information. "
        "اگر نامعلوم است null یا [] بگذار. پایه را فقط وقتی صریح است عددی کن.\n\n"
        f"سوال فعلی: {question}\n"
        f"زمینه پیام‌های اخیر: {history_text or 'ندارد'}"
    )
    payload = {
        "model": BAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an information extraction engine. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "stream": False,
    }
    try:
        response = requests.post(BAI_URL, headers={
            "Authorization": f"Bearer {BAI_API_KEY}",
            "Content-Type": "application/json",
        }, json=payload, timeout=LLM_ANALYSIS_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return _extract_json_object(data["choices"][0]["message"]["content"])
    except Exception as error:
        log(f"[analysis] semantic parsing failed: {error}")
        return {}


def _merge_semantic_context(ctx, semantic):
    if not semantic:
        return ctx
    mapping = {
        "hair_state": "states", "current_color": "current_colors",
        "target_color": "target_colors", "applications": "applications",
    }
    for src, dst in mapping.items():
        value = semantic.get(src)
        if isinstance(value, str) and value.strip():
            value = [value]
        if isinstance(value, list):
            normalized = {_normalize(str(v)) for v in value if str(v).strip()}
            if dst == "states":
                known = {"نچرال", "رنگ‌شده", "دکلره"}
                normalized = {x for x in normalized if x in known or any(x in _normalize(t) for t in _HAIR_STATE_TERMS.get("نچرال", {}).get("query", []))}
            if normalized:
                ctx[dst].update(normalized)
    for key in ("current_base", "target_base"):
        value = semantic.get(key)
        if isinstance(value, (int, float)) and 1 <= int(value) <= 12:
            if key == "current_base" and ctx.get("current_base") is None:
                ctx["current_base"] = int(value)
            if key == "target_base" and ctx.get("target_base") is None:
                ctx["target_base"] = int(value)
    for key in ("constraints", "missing_information"):
        value = semantic.get(key)
        if isinstance(value, str):
            value = [value]
        ctx[key] = value if isinstance(value, list) else []
    intent = semantic.get("intent")
    if isinstance(intent, str) and intent in QUESTION_TYPE_LABELS:
        ctx["types"] = [intent] + [x for x in ctx.get("types", []) if x != intent]
    ctx["semantic"] = semantic
    return ctx


def _history_text(chat_id):
    if chat_id is None:
        return ""
    items = _chat_history.get(str(chat_id), [])[-MAX_HISTORY_MESSAGES:]
    return "\n".join(f"{role}: {text}" for role, text in items)[-MAX_HISTORY_CHARS:]


def _remember(chat_id, role, text):
    if chat_id is None:
        return
    key = str(chat_id)
    items = _chat_history.setdefault(key, [])
    items.append((role, text[:3000]))
    _chat_history[key] = items[-MAX_HISTORY_MESSAGES:]


def _analysis_hints_text(ctx):
    bits = []
    if ctx.get("types"):
        bits.append("نوع سوال: " + "، ".join(QUESTION_TYPE_LABELS.get(t, t) for t in ctx["types"]))
    if ctx.get("states"):
        bits.append("وضعیت مو: " + "، ".join(sorted(ctx["states"])))
    if ctx.get("current_base") is not None:
        bits.append(f"پایه فعلی: {ctx['current_base']}")
    if ctx.get("target_base") is not None:
        bits.append(f"پایه هدف: {ctx['target_base']}")
    if ctx.get("target_colors"):
        bits.append("رنگ هدف: " + "، ".join(sorted(ctx["target_colors"])))
    if ctx.get("current_colors"):
        bits.append("رنگ فعلی: " + "، ".join(sorted(ctx["current_colors"])))
    return " | ".join(bits) if bits else "—"


def _derived_rules_text():
    """قوانین استخراج‌شده از منابع (_derive_domain_rules) را برای پرامپت سیستم متن می‌کند."""
    domain = _domain()
    lift = domain.get("max_natural_lift", _DOMAIN_FALLBACK["max_natural_lift"])
    two_stage = domain.get("two_stage_threshold", lift)
    return (
        "قوانین استخراج‌شده از دانشنامه (فقط اگر با منابع پایین هم‌خوان است به کار ببر): "
        f"موی نچرال با رنگ معمولی حداکثر تا {lift} پایه روشن می‌شود؛ "
        f"اگر اختلاف پایه فعلی و هدف بیشتر از {two_stage} پایه باشد مسیر دو مرحله‌ای (دکلره/پیش‌روشن‌سازی + رنگ) لازم است. "
        "رنگ‌های گرم: " + "، ".join(sorted(domain.get("warm", set()))) + "؛ "
        "رنگ‌های سرد: " + "، ".join(sorted(domain.get("cool", set()))) + "؛ "
        "رنگ‌های خنثی: " + "، ".join(sorted(domain.get("neutral", set()))) + "."
    )


def build_system_prompt(ctx):
    parts = [SYSTEM_PROMPT_CORE]
    if ctx.get("is_formula"):
        parts.append(FORMULA_SYSTEM_EXTRA)
    parts.append("تحلیل ساختاری سوال (راهنما، نه منبع): " + _analysis_hints_text(ctx))
    if ctx.get("constraints"):
        parts.append("محدودیت‌های کاربر: " + "، ".join(map(str, ctx["constraints"])))
    parts.append(_derived_rules_text())
    parts.append(OUTPUT_FORMAT_RULES)
    return "\n\n".join(parts)


# ----------------------------------------------------------------- هسته: پاسخ


def answer(question, chat_id=None):
    """Question -> context/history -> semantic parse -> hybrid RAG -> grounded synthesis."""
    question = (question or "").strip()
    if not question:
        return "لطفاً سوال خود را درباره رنگ مو بنویسید."

    refresh_knowledge()
    if not BAI_API_KEY:
        return MSG_SERVICE_UNAVAILABLE

    history = _history_text(chat_id)
    ctx = analyze_question(question)
    semantic = _model_question_analysis(question, history)
    _merge_semantic_context(ctx, semantic)
    ctx["is_formula"] = _is_formula_question(ctx["types"])
    ctx["prelift"] = _needs_prelift(ctx)
    ctx["target_light"] = _target_is_light(ctx)
    ctx["needs_bleach"] = bool(_targets_needing_bleach_on_natural(ctx))

    log("[type] نوع سوال: " + "، ".join(QUESTION_TYPE_LABELS.get(t, t) for t in ctx["types"]))
    log("[understanding] " + _analysis_hints_text(ctx))

    chunks = search_knowledge(question, ctx)
    sources = _format_sources(chunks, ctx["types"])
    analysis_json = json.dumps(ctx.get("semantic", {}), ensure_ascii=False)
    user_prompt = (
        "قوانین پاسخ‌دهی: منابع پایین تنها مرجع محتوایی تو هستند. از دانش عمومی برای ساختن فرمول یا عدد جدید استفاده نکن. "
        "اما منابع متعدد را با هم ترکیب کن و پاسخ را به زبان کاربر توضیح بده. اگر منبع برای بخشی از سوال کافی نیست، همان بخش را صادقانه اعلام کن. "
        "Q&A فقط یکی از منابع است و نباید جلوی استفاده از منابع فنی دیگر را بگیرد.\n\n"
        f"تحلیل ساختاری سوال:\n{analysis_json}\n\n"
        f"منابع دانشنامه:\n{sources}\n\n"
        f"زمینه گفت‌وگو:\n{history or 'ندارد'}\n\n"
        f"سوال فعلی کاربر:\n{question}"
    )

    temp = 0.15 if ctx["is_formula"] else 0.25
    payload = {
        "model": BAI_MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt(ctx)},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temp,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {BAI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(BAI_URL, headers=headers, json=payload, timeout=110)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
    except requests.RequestException as error:
        log(f"[b.ai] خطای ارتباط با b.ai: {error}")
        return MSG_ANSWER_FAILED
    except (KeyError, IndexError, TypeError, ValueError) as error:
        log(f"[b.ai] پاسخ نامعتبر از b.ai: {error}")
        return MSG_ANSWER_FAILED

    if not reply:
        return MSG_ANSWER_FAILED
    reply = _sanitize_html_for_telegram(reply)
    _remember(chat_id, "user", question)
    _remember(chat_id, "assistant", reply)
    return reply


def _sanitize_html_for_telegram(text):
    """متن را برای parse_mode=HTML تلگرام امن می‌کند."""
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"^\s*[-*•]\s+", "• ", text, flags=re.MULTILINE)

    allowed = {"b", "i", "u", "s", "code", "pre", "a"}
    tag_pattern = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)(\s[^<>]*)?>")
    placeholders = {}
    counter = [0]

    def save_tag(m):
        key = f"\x00TAG{counter[0]}\x00"
        counter[0] += 1
        if m.group(1).lower() in allowed:
            placeholders[key] = m.group(0)
        else:
            placeholders[key] = html.escape(m.group(0))
        return key

    text = tag_pattern.sub(save_tag, text)
    text = html.escape(text, quote=False)
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


# ---------------------------------------------------------------- تلگرام


def _split_message(text, limit=4000):
    text = text or ""
    if len(text) <= limit:
        return [text]
    pieces = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= limit * 0.6:
            cut = limit
        pieces.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        pieces.append(text)
    return pieces or [""]


def telegram_send(chat_id, text):
    """پیام را با parse_mode=HTML به تلگرام می‌فرستد."""
    if not TELEGRAM_BOT_TOKEN:
        log("[telegram] TELEGRAM_BOT_TOKEN تنظیم نشده؛ پیام ارسال نشد.")
        return
    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN, method="sendMessage")
    for piece in _split_message(text):
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": piece,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )
            if response.status_code != 200:
                log(f"[telegram] ارسال ناموفق ({response.status_code}): {response.text[:400]}")
                try:
                    err_data = response.json()
                    err_str = str(err_data).lower()
                    if "can't parse" in err_str or "parse" in err_str or "bad request" in err_str:
                        # تلاش بدون HTML
                        fallback = re.sub(r"<[^>]+>", "", html.unescape(piece))
                        requests.post(
                            url,
                            json={"chat_id": chat_id, "text": fallback},
                            timeout=30,
                        )
                except Exception as e:
                    log(f"[telegram] fallback failed: {e}")
        except requests.RequestException as error:
            log(f"[telegram] خطای ارسال پیام: {error}")


def setup_telegram_webhook():
    """وب‌هوک تلگرام را با WEBHOOK_BASE_URL تنظیم می‌کند."""
    if not TELEGRAM_BOT_TOKEN:
        log("[telegram] توکن تلگرام نیست، وب‌هوک تنظیم نشد.")
        return False
    if not WEBHOOK_BASE_URL:
        log("[telegram] WEBHOOK_BASE_URL تنظیم نشده، وب‌هوک دستی باید تنظیم شود. مثال: https://api.telegram.org/bot<token>/setWebhook?url=<base>/telegram")
        return False
    webhook_url = f"{WEBHOOK_BASE_URL}/telegram"
    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN, method="setWebhook")
    try:
        resp = requests.post(url, json={"url": webhook_url, "drop_pending_updates": False}, timeout=20)
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            log(f"[telegram] وب‌هوک با موفقیت تنظیم شد: {webhook_url}")
            return True
        else:
            log(f"[telegram] خطا در تنظیم وب‌هوک: {data}")
            return False
    except Exception as e:
        log(f"[telegram] استثناء در تنظیم وب‌هوک: {e}")
        return False


# ------------------------------------------------------------------ سرویس

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """صفحه اصلی - برای جلوگیری از 404 و نمایش وضعیت."""
    return jsonify({
        "name": "Aihaircolorbot - دستیار تخصصی رنگ مو",
        "status": "running",
        "endpoints": {
            "/health": "بررسی سلامت و دانشنامه",
            "/telegram": "وب‌هوک تلگرام (POST)",
            "/reload": "بارگذاری مجدد دانشنامه",
            "/set-webhook": "تنظیم وب‌هوک تلگرام (GET, نیاز به توکن)",
        },
        "knowledge_files": [d["source"] for d in _knowledge_cache["documents"]],
        "telegram": "set" if TELEGRAM_BOT_TOKEN else "missing",
        "bai_api_key": "set" if BAI_API_KEY else "missing",
        "bai_url": BAI_URL,
        "bai_model": BAI_MODEL,
        "webhook_base": WEBHOOK_BASE_URL or "not set - باید دستی وب‌هوک را تنظیم کنید",
    }), 200


@app.route("/health", methods=["GET"])
def health():
    documents, chunks, _ = refresh_knowledge()
    domain = _domain()
    return (
        jsonify(
            {
                "status": "ok" if BAI_API_KEY else "degraded",
                "model": BAI_MODEL,
                "bai_api_key": "set" if BAI_API_KEY else "missing",
                "bai_url": BAI_URL,
                "bai_model": BAI_MODEL,
                "telegram_bot_token": "set" if TELEGRAM_BOT_TOKEN else "missing",
                "webhook_base_url": WEBHOOK_BASE_URL or "missing",
                "knowledge_files": [doc["source"] for doc in documents],
                "knowledge_chars": sum(len(doc["text"]) for doc in documents),
                "knowledge_chunks": len(chunks),
                "knowledge_loaded_at": _knowledge_cache["loaded_at"],
                "color_map_exists": bool(_knowledge_cache.get("color_map_text")),
                "domain": {
                    "max_natural_lift": domain["max_natural_lift"],
                    "two_stage_threshold": domain["two_stage_threshold"],
                    "warm_colors": sorted(domain["warm"]),
                    "cool_colors": sorted(domain["cool"]),
                    "neutral_colors": sorted(domain["neutral"]),
                },
                "auto_reload": True,
            }
        ),
        200,
    )


@app.route("/reload", methods=["POST", "GET"])
def reload_endpoint():
    refresh_knowledge(force=True)
    return jsonify(ok=True, files=[d["source"] for d in _knowledge_cache["documents"]]), 200


@app.route("/set-webhook", methods=["GET", "POST"])
def set_webhook_endpoint():
    """اندپوینت برای تنظیم دستی وب‌هوک."""
    if not TELEGRAM_BOT_TOKEN:
        return jsonify(ok=False, error="TELEGRAM_BOT_TOKEN missing"), 400
    base = request.args.get("base") or WEBHOOK_BASE_URL
    if not base:
        # سعی کن از هاست درخواست بسازی
        base = request.host_url.rstrip("/")
        if not base.startswith("https://"):
            return jsonify(ok=False, error="WEBHOOK_BASE_URL missing and request is not https. Provide ?base=https://your-app.onrender.com"), 400
    webhook_url = f"{base.rstrip('/')}/telegram"
    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN, method="setWebhook")
    try:
        resp = requests.post(url, json={"url": webhook_url}, timeout=20)
        return jsonify(ok=True, webhook_url=webhook_url, telegram_response=resp.json()), 200
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/telegram", methods=["POST"])
def telegram_webhook():
    try:
        update = request.get_json(silent=True) or {}
        # پشتیبانی از message, edited_message, channel_post
        message = update.get("message") or update.get("edited_message") or update.get("channel_post") or {}
        chat_id = (message.get("chat") or {}).get("id")
        text = (message.get("text") or message.get("caption") or "").strip()

        if not chat_id:
            # شاید callback_query باشد
            cq = update.get("callback_query") or {}
            chat_id = (cq.get("message") or {}).get("chat", {}).get("id")
            text = (cq.get("data") or "").strip()

        if not chat_id or not text:
            return jsonify(ok=True)

        log(f"[telegram] پیام دریافتی از {chat_id}: {text[:100]}")

        if text.startswith("/start"):
            telegram_send(chat_id, START_MESSAGE)
        elif text.startswith("/reload"):
            refresh_knowledge(force=True)
            telegram_send(chat_id, "✅ دانشنامه دوباره بارگذاری شد.")
        elif text.startswith("/health"):
            docs = _knowledge_cache["documents"]
            telegram_send(chat_id, f"✅ ربات فعال است.\n📚 {len(docs)} فایل دانشنامه، {sum(len(d['text']) for d in docs)} کاراکتر.")
        else:
            telegram_send(chat_id, answer(text, chat_id=chat_id))
    except Exception as error:
        log(f"[telegram] خطا در وب‌هوک: {error}")
        import traceback
        log(traceback.format_exc())
    return jsonify(ok=True), 200


# تنظیم خودکار وب‌هوک در استارت‌آپ اگر WEBHOOK_BASE_URL موجود باشد
if WEBHOOK_BASE_URL and TELEGRAM_BOT_TOKEN:
    try:
        # در یک ترد جدا تنظیم کن تا استارت‌آپ بلاک نشود، ولی اینجا ساده لاگ می‌کنیم
        log(f"[startup] تلاش برای تنظیم وب‌هوک: {WEBHOOK_BASE_URL}/telegram")
        # تلاش اولیه - اگر فیل شد در /set-webhook می‌توان دوباره زد
        # برای جلوگیری از بلاک در import، فقط در حالت gunicorn با تاخیر اجرا می‌کنیم
        pass
    except Exception as e:
        log(f"[startup] خطا: {e}")

if __name__ == "__main__":
    documents = _knowledge_cache["documents"]
    chunks = _knowledge_cache["chunks"]
    log(f"[knowledge] {len(documents)} فایل دانشنامه بارگذاری شد "
        f"({sum(len(doc['text']) for doc in documents)} کاراکتر، {len(chunks)} تکه).")
    log("[knowledge] حالت آپدیت خودکار فعال است (چک mtime در هر سوال).")
    if not _knowledge_cache.get("color_map_text"):
        log("[warning] color_map.txt وجود ندارد! در حال ایجاد هشدار...")
    # تنظیم وب‌هوک در اجرای مستقیم
    if WEBHOOK_BASE_URL and TELEGRAM_BOT_TOKEN:
        setup_telegram_webhook()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
