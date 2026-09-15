import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import requests
from flask import Flask, request, jsonify


# =========================================================
# تنظیمات اصلی
# =========================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# توکن ربات تلگرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# تنظیمات BAI / Qwen
BAI_API_KEY = os.getenv("BAI_API_KEY", "").strip()
BAI_URL = os.getenv(
    "BAI_URL",
    "https://api.b.ai/v1/chat/completions"
).strip()

BAI_MODEL = os.getenv(
    "BAI_MODEL",
    "Qwen3.8-Flash"
).strip()

# آدرس عمومی Render برای Webhook
WEBHOOK_BASE_URL = os.getenv(
    "WEBHOOK_BASE_URL",
    ""
).strip().rstrip("/")

# محدودیت‌ها
MAX_TELEGRAM_MESSAGE = 3900
MAX_CONTEXT_CHARS = 18000
MAX_HISTORY_MESSAGES = 6

# مسیر فایل‌های دانشنامه
BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_FILES = [
    BASE_DIR / "knowledge.txt",
    BASE_DIR / "haircolor_q&a.txt",
]


# =========================================================
# حافظه کوتاه‌مدت گفتگو
# =========================================================

conversation_history: Dict[int, List[Dict[str, str]]] = {}


# =========================================================
# خواندن و آماده‌سازی دانشنامه
# =========================================================

def read_text_file(path: Path) -> str:
    """
    خواندن فایل متنی با پشتیبانی از UTF-8 و UTF-8 BOM.
    """
    if not path.exists():
        logger.warning("Knowledge file not found: %s", path)
        return ""

    try:
        return path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        logger.exception("Could not read %s: %s", path, exc)
        return ""


def normalize_text(text: str) -> str:
    """
    نرمال‌سازی سبک برای جست‌وجو.
    متن اصلی را تغییر نمی‌دهد؛ فقط نسخه جست‌وجو را می‌سازد.
    """
    text = text.lower()

    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "‌": " ",
        "\u200c": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # یکسان‌سازی فاصله‌ها
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_into_chunks(text: str) -> List[str]:
    """
    دانشنامه را به قطعه‌های معنی‌دار تقسیم می‌کند.

    اولویت:
    1. پرسش و پاسخ‌های Q&A
    2. بخش‌های شماره‌گذاری‌شده
    3. پاراگراف‌های معمولی
    """

    if not text.strip():
        return []

    # اگر فایل ساختار Q1 / A1 دارد، هر پرسش و پاسخ یک قطعه باشد.
    qa_matches = re.split(
        r"(?=Q\d+\s*:)",
        text,
        flags=re.IGNORECASE
    )

    chunks = []

    for part in qa_matches:
        part = part.strip()

        if not part:
            continue

        # قطعه‌های خیلی بزرگ را به پاراگراف تقسیم کن
        if len(part) <= 4500:
            chunks.append(part)
        else:
            paragraphs = re.split(r"\n\s*\n", part)

            current = ""

            for paragraph in paragraphs:
                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                if len(current) + len(paragraph) + 2 <= 4500:
                    current += ("\n\n" if current else "") + paragraph
                else:
                    if current:
                        chunks.append(current)

                    current = paragraph

            if current:
                chunks.append(current)

    return chunks


def load_knowledge() -> List[Dict[str, Any]]:
    """
    همه فایل‌های دانشنامه را می‌خواند و قطعه‌بندی می‌کند.
    """

    all_chunks = []

    for file_path in KNOWLEDGE_FILES:
        text = read_text_file(file_path)

        if not text:
            continue

        chunks = split_into_chunks(text)

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "source": file_path.name,
                "index": index,
                "text": chunk,
                "normalized": normalize_text(chunk),
            })

    logger.info(
        "Knowledge loaded: %s chunks from %s files",
        len(all_chunks),
        len(KNOWLEDGE_FILES)
    )

    return all_chunks


KNOWLEDGE = load_knowledge()


# =========================================================
# جست‌وجوی هوشمند دانشنامه
# =========================================================

def extract_search_terms(query: str) -> List[str]:
    """
    کلمات مهم سؤال را استخراج می‌کند.
    """

    normalized = normalize_text(query)

    # کلمات خیلی عمومی که ارزش جست‌وجوی کمی دارند
    stop_words = {
        "از", "به", "در", "با", "برای", "که", "و", "یا",
        "را", "روی", "چه", "چطور", "چگونه", "چی", "می", "شود",
        "کنم", "کنه", "داره", "دارد", "است", "هست", "این",
        "آن", "اگر", "من", "مو", "رنگ", "رنگی", "یک", "تا",
        "the", "is", "a", "an", "of", "to", "and"
    }

    words = re.findall(
        r"[a-zA-Z0-9آ-ی]+(?:[./-][a-zA-Z0-9آ-ی]+)*",
        normalized
    )

    terms = []

    for word in words:
        if len(word) >= 2 and word not in stop_words:
            terms.append(word)

    # عبارت‌های مهم رنگ و پایه را هم حفظ کن
    patterns = [
        r"\b\d+\s*/\s*\d+\b",
        r"\bپایه\s*\d+\b",
        r"\bاکسیدان\s*%?\s*\d+\b",
        r"\b\d+\s*درصد\b",
        r"\bq\d+\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        terms.extend(matches)

    # حذف تکراری‌ها
    unique_terms = []

    for term in terms:
        if term not in unique_terms:
            unique_terms.append(term)

    return unique_terms


def score_chunk(query: str, chunk: Dict[str, Any]) -> float:
    """
    امتیازدهی ترکیبی:
    - تطابق عبارت کامل
    - تعداد کلمات مشترک
    - تطابق اصطلاحات تخصصی
    """

    q = normalize_text(query)
    text = chunk["normalized"]

    if not q or not text:
        return 0.0

    score = 0.0

    # تطابق عبارت کامل
    if q in text:
        score += 20

    terms = extract_search_terms(query)

    for term in terms:
        if term in text:
            score += 2

        # اصطلاحات دقیق‌تر وزن بیشتری دارند
        if "/" in term or "%" in term or "پایه" in term:
            if term in text:
                score += 4

    # تطابق کلمه‌ای
    text_words = set(re.findall(r"[a-zA-Z0-9آ-ی]+", text))
    query_words = set(terms)

    common_words = text_words.intersection(query_words)

    score += min(len(common_words) * 1.5, 25)

    return score


def search_knowledge(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    مرتبط‌ترین قطعه‌های دانشنامه را برمی‌گرداند.
    """

    if not KNOWLEDGE:
        return []

    scored = []

    for chunk in KNOWLEDGE:
        score = score_chunk(query, chunk)

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            **chunk,
            "score": round(score, 2)
        }
        for score, chunk in scored[:top_k]
    ]


def build_context(results: List[Dict[str, Any]]) -> str:
    """
    منابع پیدا شده را برای مدل آماده می‌کند.
    """

    if not results:
        return "هیچ بخش مرتبطی از دانشنامه پیدا نشد."

    parts = []

    for i, result in enumerate(results, start=1):
        parts.append(
            f"--- منبع {i}: {result['source']} ---\n"
            f"{result['text']}"
        )

    context = "\n\n".join(parts)

    return context[:MAX_CONTEXT_CHARS]


# =========================================================
# پرامپت تخصصی مدل
# =========================================================

SYSTEM_PROMPT = """
تو دستیار هوشمند و تخصصی دانشنامه آموزش رنگ مو و تراپی مو هستی.

وظیفه تو این است که به پرسش‌های کاربر درباره رنگ مو، دکلره،
رنگساژ، پایه‌شناسی، تناژها، واریاسیون، اکسیدان، فرمول‌نویسی،
ترمیم دکلره و موضوعات موجود در دانشنامه پاسخ بدهی.

قوانین بسیار مهم:

1. منابعی که در پیام کاربر به تو داده می‌شوند، دانشنامه اصلی هستند.
2. پاسخ را تا حد امکان دقیقاً بر اساس همان منابع بنویس.
3. اگر پاسخ در منابع وجود ندارد، صادقانه بگو:
   «این مورد در دانشنامه فعلی به‌صورت مشخص توضیح داده نشده است.»
   سپس فقط اگر لازم بود، بگو برای پاسخ قطعی به اطلاعات بیشتری نیاز است.
4. هرگز فرمول رنگ، شماره رنگ، پایه، مقدار واریاسیون یا درصد اکسیدان
   را از خودت اختراع نکن.
5. اگر سؤال چند حالت دارد، پاسخ را مرحله‌به‌مرحله و دسته‌بندی‌شده بده.
6. اگر سؤال درباره فرمول است، شماره رنگ‌ها، مقدارها، اکسیدان و زمان
   را واضح و جداگانه بنویس.
7. تفاوت «موی نچرال»، «موی دکلره‌شده» و «رنگساژ» را حفظ کن.
8. اصطلاحات تخصصی فارسی کاربر را حفظ کن؛ مثل:
   پایه، تناژ، رفله، کاور، لیفت، بستر سازی، رنگساژ، دکوپاژ،
   واریاسیون، اکسترا، نچرال و هایلیفت.
9. اگر منابع با هم تفاوت یا ابهام دارند، آن را پنهان نکن.
10. پاسخ را فارسی، روان، دقیق و کاربردی بنویس.
11. از جواب‌های کلی و غیرمرتبط خودداری کن.
12. در مسائل شیمیایی و کار با مواد، نکات ایمنی را نادیده نگیر.
13. اگر سؤال پزشکی، حساسیت شدید یا آسیب جدی مو و پوست سر است،
    پاسخ را به‌عنوان تشخیص پزشکی ارائه نکن و توصیه به مراجعه به متخصص بده.
14. اگر کاربر فقط سلام کرد، کوتاه و دوستانه پاسخ بده.
15. اگر کاربر فرمولی را اشتباه فهمیده، محترمانه اصلاح کن و دلیل بیاور.
16. هیچ‌وقت ادعا نکن که چیزی در دانشنامه هست، مگر اینکه در منابع
    ارائه‌شده وجود داشته باشد.

ساختار پیشنهادی پاسخ:
- جواب مستقیم
- توضیح علمی یا دلیل
- فرمول / مراحل، در صورت نیاز
- نکته مهم یا هشدار، در صورت نیاز

منابع دانشنامه در پیام بعدی قرار دارند.
"""


# =========================================================
# ارتباط با BAI API
# =========================================================

def call_bai(
    user_question: str,
    context: str,
    history: List[Dict[str, str]]
) -> str:
    """
    ارسال سؤال به API سازگار با OpenAI Chat Completions.
    """

    if not BAI_API_KEY:
        raise RuntimeError("BAI_API_KEY is not configured.")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # حافظه کوتاه‌مدت گفتگو
    for message in history[-MAX_HISTORY_MESSAGES:]:
        messages.append(message)

    # سؤال فعلی + منابع
    user_content = f"""
سؤال کاربر:
{user_question}

متن مرتبط استخراج‌شده از دانشنامه:
{context}

اکنون فقط بر اساس سؤال و منابع بالا پاسخ بده.
اگر منابع برای پاسخ کافی نیستند، این موضوع را شفاف اعلام کن.
"""

    messages.append({
        "role": "user",
        "content": user_content
    })

    payload = {
        "model": BAI_MODEL,
        "messages": messages,
        "temperature": 0.15,
        "top_p": 0.85,
        "max_tokens": 1800,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {BAI_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        BAI_URL,
        headers=headers,
        json=payload,
        timeout=90
    )

    if not response.ok:
        logger.error(
            "BAI API error: %s | %s",
            response.status_code,
            response.text[:1000]
        )

        raise RuntimeError(
            f"BAI API returned HTTP {response.status_code}"
        )

    data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.error("Unexpected BAI response: %s", data)
        raise RuntimeError("Invalid response from BAI API.")

    if not answer or not answer.strip():
        raise RuntimeError("Empty response from BAI API.")

    return answer.strip()


# =========================================================
# توابع تلگرام
# =========================================================

def telegram_request(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    ارسال درخواست به Telegram Bot API.
    """

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/{method}"
    )

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def send_telegram_message(
    chat_id: int,
    text: str,
    reply_to_message_id: int = None
) -> None:
    """
    ارسال پیام با تقسیم خودکار متن‌های طولانی.
    """

    if not text:
        return

    chunks = [
        text[i:i + MAX_TELEGRAM_MESSAGE]
        for i in range(0, len(text), MAX_TELEGRAM_MESSAGE)
    ]

    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_web_page_preview": True
        }

        if reply_to_message_id:
            payload["reply_parameters"] = {
                "message_id": reply_to_message_id
            }

        telegram_request("sendMessage", payload)


def send_typing_action(chat_id: int) -> None:
    """
    نمایش حالت در حال نوشتن در تلگرام.
    """

    try:
        telegram_request(
            "sendChatAction",
            {
                "chat_id": chat_id,
                "action": "typing"
            }
        )
    except Exception:
        # خطای typing نباید پاسخ اصلی را متوقف کند
        pass


# =========================================================
# پردازش پیام کاربر
# =========================================================

def process_user_question(
    chat_id: int,
    user_question: str
) -> str:
    """
    مسیر اصلی:
    سؤال → جست‌وجوی دانشنامه → مدل → پاسخ
    """

    question = user_question.strip()

    if not question:
        return "لطفاً سؤال خودت را درباره رنگ مو یا تراپی بنویس."

    # دستورات ساده
    if question in ["/start", "/help"]:
        return (
            "سلام 🌹\n\n"
            "من دستیار هوشمند دانشنامه رنگ مو و تراپی هستم.\n"
            "سؤالت را بفرست؛ مثلاً:\n\n"
            "• روی پایه ۸ چطور رنگساژ دودی بزنم؟\n"
            "• تفاوت فرمول رنگساژ معمولی و پیشرفته چیست؟\n"
            "• برای موی چندپایه چه رنگساژی انتخاب کنم؟"
        )

    if question == "/reset":
        conversation_history.pop(chat_id, None)
        return "حافظه گفت‌وگوی این چت پاک شد. 🌿"

    # جست‌وجوی منابع
    results = search_knowledge(question, top_k=5)
    context = build_context(results)

    logger.info(
        "Question: %s | Retrieved chunks: %s",
        question[:100],
        len(results)
    )

    history = conversation_history.get(chat_id, [])

    answer = call_bai(
        user_question=question,
        context=context,
        history=history
    )

    # ذخیره فقط سؤال و جواب، نه متن کامل دانشنامه
    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    conversation_history[chat_id] = history[-MAX_HISTORY_MESSAGES:]

    return answer


# =========================================================
# Webhook تلگرام
# =========================================================

@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    """
    دریافت پیام‌های تلگرام از طریق Webhook.
    """

    try:
        update = request.get_json(silent=True) or {}

        message = update.get("message")

        if not message:
            return jsonify({"ok": True})

        chat = message.get("chat", {})
        chat_id = chat.get("id")

        text = message.get("text", "")

        if not chat_id or not text:
            return jsonify({"ok": True})

        send_typing_action(chat_id)

        try:
            answer = process_user_question(
                chat_id=chat_id,
                user_question=text
            )

        except Exception as exc:
            logger.exception("Error processing question: %s", exc)

            answer = (
                "متأسفانه در پردازش سؤال مشکلی پیش آمد. "
                "لطفاً چند لحظه بعد دوباره تلاش کن."
            )

        send_telegram_message(
            chat_id=chat_id,
            text=answer,
            reply_to_message_id=message.get("message_id")
        )

        return jsonify({"ok": True})

    except Exception as exc:
        logger.exception("Webhook error: %s", exc)
        return jsonify({"ok": True})


# =========================================================
# سلامت سرویس
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "knowledge_chunks": len(KNOWLEDGE),
        "model": BAI_MODEL,
        "knowledge_files": [
            path.name for path in KNOWLEDGE_FILES
            if path.exists()
        ]
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Hair Color AI Bot",
        "status": "running"
    })


# =========================================================
# تنظیم Webhook هنگام اجرای برنامه
# =========================================================

def setup_webhook():
    """
    اگر WEBHOOK_BASE_URL تنظیم شده باشد،
    آدرس Webhook ربات را روی Render ثبت می‌کند.
    """

    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN is missing.")
        return

    if not WEBHOOK_BASE_URL:
        logger.warning("WEBHOOK_BASE_URL is missing.")
        return

    webhook_url = f"{WEBHOOK_BASE_URL}/webhook"

    try:
        result = telegram_request(
            "setWebhook",
            {
                "url": webhook_url,
                "drop_pending_updates": False
            }
        )

        logger.info("Webhook setup result: %s", result)

    except Exception as exc:
        logger.exception("Could not set Telegram webhook: %s", exc)


# =========================================================
# اجرای برنامه
# =========================================================

if __name__ == "__main__":
    setup_webhook()

    port = int(os.getenv("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
          )
