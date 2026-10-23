import io
import statistics
import os
import re
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

APP_VERSION = "4.0.0"
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(15 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "10"))
OCR_DPI = int(os.getenv("OCR_DPI", "220"))
OCR_LANG = os.getenv("OCR_LANG", "ara+eng")
# auto  = يكتشف اتجاه الكلمات العربية لكل سطر (الافتراضي، الأدق)
# true  = يعكس كل المقاطع العربية دائماً (السلوك القديم)
# false = بدون أي عكس
REVERSE_ARABIC_DEFAULT = os.getenv("REVERSE_ARABIC_DEFAULT", "auto").lower()
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("umrah_track_ocr")

app = FastAPI(
    title="VANT // Umrah Track Pro OCR API",
    version=APP_VERSION,
    description="Smart OCR and field extraction for Umrah visas, passports, and travel tickets.",
)

origins_setting = os.getenv("ALLOWED_ORIGINS", "*").strip()
allowed_origins = ["*"] if origins_setting == "*" else [
    origin.strip().rstrip("/") for origin in origins_setting.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get("/")
def home():
    index_file = Path(__file__).with_name("index.html")
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return {
        "status": "ok",
        "service": "Yemenia Ticket OCR API",
        "version": APP_VERSION,
        "message": "Yemenia ticket OCR API is running.",
    }


@app.get("/health")
def health():
    try:
        languages = sorted(pytesseract.get_languages(config=""))
    except Exception:
        languages = []
    required = {lang.strip() for lang in OCR_LANG.split("+") if lang.strip()}
    available = required.issubset(set(languages))
    return {
        "status": "healthy" if available else "degraded",
        "version": APP_VERSION,
        "ocr_lang": OCR_LANG,
        "available_languages": languages,
        "missing_languages": sorted(required - set(languages)),
    }


NA = "غير متاح"
AR_RE = re.compile(r"[\u0600-\u06FF]")
_AR_LETTERS = "\u0600-\u06FF"


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def has_arabic(text: str) -> bool:
    return bool(AR_RE.search(text or ""))


def normalize_digits(text: str) -> str:
    """Arabic-Indic / Persian digits -> ASCII digits."""
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return text.translate(table)


_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\u061c\ufeff\x00]")


def normalize_text(text: str) -> str:
    # NFKC converts Arabic presentation forms (e.g. the lam-alef ligature) to plain letters
    text = unicodedata.normalize("NFKC", text or "")
    text = normalize_digits(text)
    text = _INVISIBLE.sub("", text)
    for dash in ("‐", "‑", "–", "—"):
        text = text.replace(dash, "-")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


# PDF text layers often scramble the lam-alef ligature ("لا" comes out as "ال").
# We repair the known label words safely. Personal names are not touched because
# "سالم/سلام" or "صالح/صلاح" cannot be told apart without context.
LAM_ALEF_FIXES = {
    "االسم": "الاسم",
    "الميالد": "الميلاد",
    "اإلقامة": "الإقامة",
    "االقامة": "الاقامة",
    "اإلصدار": "الإصدار",
    "االصدار": "الاصدار",
    "اإلمارات": "الإمارات",
    "األردن": "الأردن",
    "اإلسالم": "الإسلام",
    "خالل": "خلال",
    "هالل": "هلال",
    "بالل": "بلال",
    "جالل": "جلال",
}


def repair_lam_alef(text: str) -> str:
    for bad, good in LAM_ALEF_FIXES.items():
        text = re.sub(
            rf"(?<![{_AR_LETTERS}]){re.escape(bad)}(?![{_AR_LETTERS}])", good, text
        )
    return text


def first_match(patterns, text: str, flags=re.IGNORECASE | re.MULTILINE) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" \t:;|")
            if value:
                return value
    return None


# ---------------------------------------------------------------------------
# Arabic direction handling (per line)
#
# In bilingual PDFs the extractor emits two kinds of lines:
#   Arabic-first  : "<AR label> <value> <EN label reversed>"  -> Arabic is already in order
#   English-first : "<EN label> <value> <AR runs reversed>"   -> Arabic words come out reversed
# A blind global reversal (the old behaviour) breaks the first kind. Here we decide
# per Arabic run, using known labels as evidence.
# ---------------------------------------------------------------------------

FWD_PHRASES = [
    "رقم التأشيرة", "صالحة اعتبارا من", "صالحة لغاية", "مدة الإقامة", "مدة الاقامة",
    "رقم جواز السفر", "مصدر التأشيرة", "تاريخ الميلاد", "نوع التأشيرة", "مكتب العمرة",
    "الوكيل الخارجي", "رقم الطلب", "اسم المسافر", "اسم المعتمر", "الاسم الكامل",
    "رقم التذكرة", "رقم الحجز", "تاريخ الإصدار", "تاريخ السفر",
]
REV_PHRASES = [" ".join(reversed(p.split())) for p in FWD_PHRASES]


def _is_ar_tok(tok: str) -> bool:
    return bool(AR_RE.search(tok))


def arabic_runs(tokens: List[str]):
    """Maximal runs of Arabic tokens (a bare number between two Arabic tokens stays inside the run)."""
    runs = []
    i, n = 0, len(tokens)
    while i < n:
        if _is_ar_tok(tokens[i]):
            j = i
            while True:
                k = j + 1
                if k < n and _is_ar_tok(tokens[k]):
                    j = k
                    continue
                if k + 1 < n and re.fullmatch(r"[\d.,/:-]+", tokens[k]) and _is_ar_tok(tokens[k + 1]):
                    j = k + 1
                    continue
                break
            runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return runs


def _evidence(run_text: str):
    fw = sum(run_text.count(p) for p in FWD_PHRASES)
    rv = sum(run_text.count(p) for p in REV_PHRASES)
    return fw, rv


def english_first_reversed(lines: List[str]) -> bool:
    fw_total = rv_total = 0
    for line in lines:
        if not has_arabic(line):
            continue
        tokens = line.split()
        if not tokens or _is_ar_tok(tokens[0]):
            continue
        for s, e in arabic_runs(tokens):
            fw, rv = _evidence(" ".join(tokens[s:e]))
            fw_total += fw
            rv_total += rv
    return rv_total > fw_total


def fix_line_direction(line: str, mode: str, ef_reversed: bool) -> str:
    if mode == "false" or not has_arabic(line):
        return line
    tokens = line.split()
    if not tokens:
        return line
    first_is_ar = _is_ar_tok(tokens[0])
    for s, e in arabic_runs(tokens):
        if mode == "true":
            reverse = True
        else:
            fw, rv = _evidence(" ".join(tokens[s:e]))
            if rv > fw:
                reverse = True
            elif fw > rv:
                reverse = False
            else:
                reverse = (not first_is_ar) and ef_reversed
        if reverse:
            tokens[s:e] = tokens[s:e][::-1]
    return " ".join(tokens)


def normalize_mode(value) -> str:
    v = str(value if value is not None else "auto").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return "true"
    if v in ("0", "false", "no", "off"):
        return "false"
    return "auto"


def prepare_lines(raw_text: str, mode: str):
    text = repair_lam_alef(normalize_text(raw_text))
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.split("\n")]
    lines = [l for l in lines if l]
    ef_rev = english_first_reversed(lines) if mode == "auto" else False
    lines = [fix_line_direction(l, mode, ef_rev) for l in lines]
    return text, lines


# ---------------------------------------------------------------------------
# Label based field extraction (works for Arabic-first, English-first and stacked layouts)
# ---------------------------------------------------------------------------

LABEL_PHRASES: Dict[str, List[str]] = {
    "visa_no": [r"Visa\s+No\.?", r"\.?No\.?\s+Visa", r"Visa\s+Number", r"رقم\s+التأشيرة", r"التأشيرة\s+رقم", r"رقم\s+الفيزا"],
    "issue_date": [r"Date\s+of\s+Issue", r"Issue\s+of\s+Date", r"Issue\s+Date", r"صالحة\s+اعتبارا\s+من",
                   r"من\s+اعتبارا\s+صالحة", r"تاريخ\s+الإصدار", r"تاريخ\s+الاصدار"],
    "valid_until": [r"Valid\s+Until", r"Until\s+Valid", r"Valid\s+To", r"صالحة\s+لغاية", r"لغاية\s+صالحة"],
    "duration": [r"Duration\s+of\s+Stay", r"Stay\s+of\s+Duration", r"مدة\s+الإقامة", r"مدة\s+الاقامة",
                 r"الإقامة\s+مدة", r"الاقامة\s+مدة"],
    "passport": [r"Passport\s+No\.?", r"\.?No\.?\s+Passport", r"Passport\s+Number", r"رقم\s+جواز\s+السفر",
                 r"السفر\s+جواز\s+رقم", r"رقم\s+الجواز"],
    "place_of_issue": [r"Place\s+of\s+issue", r"issue\s+of\s+Place", r"مصدر\s+التأشيرة", r"التأشيرة\s+مصدر"],
    "name": [r"Full\s+Name", r"Passenger\s+Name", r"Name", r"الاسم\s+الكامل", r"اسم\s+المعتمر", r"اسم\s+المسافر",
             r"الاسم", r"الإسم", r"السم"],
    "birth": [r"Birth\s+Date", r"Date\s+of\s+Birth", r"Date\s+Birth", r"تاريخ\s+الميلاد", r"الميلاد\s+تاريخ"],
    "nationality": [r"Nationality", r"Citizenship", r"الجنسية", r"جنسية\s+المعتمر"],
    "visa_type": [r"Type\s+Of\s+Visa", r"Visa\s+Of\s+Type", r"نوع\s+التأشيرة", r"التأشيرة\s+نوع"],
    "operator": [r"Umrah\s+Operator", r"Operator\s+Umrah", r"مكتب\s+العمرة", r"العمرة\s+مكتب", r"شركة\s+العمرة"],
    "agent": [r"External\s+Agent", r"Agent\s+External", r"الوكيل\s+الخارجي", r"الخارجي\s+الوكيل"],
    "application": [r"Application\s+No\.?", r"\.?No\.?\s+Application", r"رقم\s+الطلب", r"الطلب\s+رقم"],
    "reference": [r"Booking\s+Ref\.?", r"Ticket\s+Number", r"Ticket\s+No\.?", r"Reference", r"رقم\s+التذكرة", r"رقم\s+الحجز"],
    "travel_date": [r"Travel\s+Date", r"Departure\s+Date", r"تاريخ\s+السفر", r"موعد\s+الرحلة", r"التاريخ"],
}


def _compile_labels(phrases: List[str]):
    ordered = sorted(phrases, key=len, reverse=True)
    return re.compile(r"(?<!\S)(?:" + "|".join(ordered) + r")(?![^\s:])", re.IGNORECASE)


LABEL_RES = {key: _compile_labels(vals) for key, vals in LABEL_PHRASES.items()}
ANY_LABEL_RE = _compile_labels([p for vals in LABEL_PHRASES.values() for p in vals])

NAME_STOP_TOKENS = {"السم", "الاسم", "الإسم", "اسم", "إسم", "name", "fullname"}


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip(" :;|,-_/.")


def strip_labels(line: str) -> str:
    return _clean_value(ANY_LABEL_RE.sub(" ", line))


def get_field(lines: List[str], key: str, value_re=None, allow_prev: bool = True) -> Optional[str]:
    label_re = LABEL_RES[key]
    for i, line in enumerate(lines):
        if not label_re.search(line):
            continue
        cands = [strip_labels(line)]
        if i + 1 < len(lines) and not ANY_LABEL_RE.search(lines[i + 1]):
            cands.append(strip_labels(lines[i + 1]))
        if value_re is not None and allow_prev and i > 0 and not ANY_LABEL_RE.search(lines[i - 1]):
            cands.append(strip_labels(lines[i - 1]))
        for cand in cands:
            if not cand:
                continue
            if value_re is None:
                return cand
            m = value_re.search(cand)
            if m:
                return (m.group(1) if m.groups() else m.group(0)).strip()
    return None


VISA_NO_RE = re.compile(r"(?<!\d)(\d{8,12})(?!\d)")
PASSPORT_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{0,2}\d{6,9})(?![A-Za-z0-9])")
APPLICATION_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]?\d{8,11})(?![A-Za-z0-9])")
DATE_VAL_RE = re.compile(r"(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4})")
DURATION_RE = re.compile(r"(\d{1,3})\s*(?:Days?|يوم|أيام)", re.IGNORECASE)


def parse_date_any(s: str) -> str:
    s = (s or "").strip()
    m = re.fullmatch(r"(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", s)
        if not m:
            return ""
        d, mo, y = map(int, m.groups())
        if y < 100:
            y += 2000
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def clean_person_name(raw: str) -> str:
    s = _clean_value(raw)
    tokens = [t for t in s.split(" ") if t and re.sub(r"[:\-|_/\\.,]", "", t).lower() not in NAME_STOP_TOKENS]
    s = " ".join(tokens).strip()
    if not s or len(s) > 100 or re.fullmatch(r"[\d\s./-]+", s):
        return ""
    return s


def reverse_arabic_field_if_marked(raw: str, field: str) -> str:
    """Reverse only values with an explicit marker of reversed OCR order."""
    value = _clean_value(raw)
    if not value or not has_arabic(value):
        return value
    tokens = value.split()
    if len(tokens) < 2:
        return value
    normalized = [t.strip(" :;|,-_/.") for t in tokens]
    name_marker = field == "name" and normalized[-1] in {"السم", "الاسم", "الإسم"}
    company_marker = field in {"operator", "agent"} and (
        normalized[-1] in {"شركة", "مؤسسة", "وكالة"}
        or normalized[0] in {"المحدودة", "المساهمة"}
    )
    return " ".join(reversed(tokens)) if name_marker or company_marker else value


# ---------------------------------------------------------------------------
# Nationality
# ---------------------------------------------------------------------------

NATIONALITY_PATTERNS = [
    (r"(?:يمني|اليمن|الجمهورية\s+اليمنية|YEMENI|YEMEN|REPUBLIC\s+OF\s+YEMEN|\bYEM\b)", "اليمن"),
    (r"(?:مصري|مصر|جمهورية\s+مصر|EGYPTIAN|EGYPT|\bEGY\b)", "مصر"),
    (r"(?:سوداني|السودان|SUDANESE|SUDAN|\bSDN\b)", "السودان"),
    (r"(?:أردني|اردني|الأردن|الاردن|JORDANIAN|JORDAN|\bJOR\b)", "الأردن"),
    (r"(?:سوري|سوريا|الجمهورية\s+العربية\s+السورية|SYRIAN|SYRIA|\bSYR\b)", "سوريا"),
    (r"(?:عراقي|العراق|IRAQI|IRAQ|\bIRQ\b)", "العراق"),
    (r"(?:فلسطيني|فلسطين|PALESTINIAN|PALESTINE|\bPSE\b|\bPLT\b)", "فلسطين"),
    (r"(?:لبناني|لبنان|LEBANESE|LEBANON|\bLBN\b)", "لبنان"),
    (r"(?:مغربي|المغرب|MOROCCAN|MOROCCO|\bMAR\b)", "المغرب"),
    (r"(?:جزائري|الجزائر|ALGERIAN|ALGERIA|\bDZA\b|\bALG\b)", "الجزائر"),
    (r"(?:تونسي|تونس|TUNISIAN|TUNISIA|\bTUN\b)", "تونس"),
    (r"(?:ليبي|ليبيا|LIBYAN|LIBYA|\bLBY\b)", "ليبيا"),
    (r"(?:عماني|سلطنة\s+عمان|OMANI|OMAN|\bOMN\b)", "عمان"),
    (r"(?:كويتي|الكويت|KUWAITI|KUWAIT|\bKWT\b)", "الكويت"),
    (r"(?:إماراتي|اماراتي|الإمارات|الامارات|EMIRATI|UAE|\bARE\b)", "الإمارات"),
    (r"(?:قطري|قطر|QATARI|QATAR|\bQAT\b)", "قطر"),
    (r"(?:بحريني|البحرين|BAHRAINI|BAHRAIN|\bBHR\b)", "البحرين"),
    (r"(?:باكستاني|باكستان|PAKISTANI|PAKISTAN|\bPAK\b)", "باكستان"),
    (r"(?:هندي|الهند|INDIAN|INDIA|\bIND\b)", "الهند"),
    (r"(?:بنغلاديشي|بنغلاديش|بنجلاديش|BANGLADESHI|BANGLADESH|\bBGD\b)", "بنغلاديش"),
    (r"(?:إندونيسي|اندونيسي|إندونيسيا|اندونيسيا|INDONESIAN|INDONESIA|\bIDN\b)", "إندونيسيا"),
    (r"(?:ماليزي|ماليزيا|MALAYSIAN|MALAYSIA|\bMYS\b)", "ماليزيا"),
    (r"(?:تركي|تركيا|TURKISH|TURKEY|TÜRKIYE|\bTUR\b)", "تركيا"),
    (r"(?:صومالي|الصومال|SOMALI|SOMALIA|\bSOM\b)", "الصومال"),
    (r"(?:جيبوتي|DJIBOUTI|\bDJI\b)", "جيبوتي"),
    (r"(?:موريتاني|موريتانيا|MAURITANIAN|MAURITANIA|\bMRT\b)", "موريتانيا"),
]
# Latin guard so "Yemenia" (the airline) or "WOMAN" never count as a nationality
COMPILED_NATIONALITIES = [
    (re.compile(r"(?<![A-Za-z])(?:%s)(?![A-Za-z])" % p, re.IGNORECASE), c) for p, c in NATIONALITY_PATTERNS
]


def normalize_country_label(raw_val: str) -> Optional[str]:
    if not raw_val:
        return None
    cleaned = re.sub(
        r"(?:الجنسية|جنسية|Nationality|Nat\.?|Citizenship|Country)\s*[:\-–—_|/\\]*", "", raw_val, flags=re.IGNORECASE
    ).strip(" :;|,-_/")
    if not cleaned:
        return None
    for pattern, canonical in COMPILED_NATIONALITIES:
        if pattern.search(cleaned):
            return canonical
    if has_arabic(cleaned):
        words = cleaned.split(" ")
        if 1 < len(words) <= 4:
            reversed_try = " ".join(reversed(words))
            for pattern, canonical in COMPILED_NATIONALITIES:
                if pattern.search(reversed_try):
                    return canonical
    return cleaned[:40]


# ---------------------------------------------------------------------------
# MRZ (machine readable zone) – a reliable cross-check for passport no / nationality / Latin name
# ---------------------------------------------------------------------------

def parse_mrz(lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    compact = [re.sub(r"\s+", "", l).replace("«", "<") for l in lines]
    for i, l in enumerate(compact):
        if len(l) < 30 or "<<" not in l or not re.fullmatch(r"[A-Z0-9<]+", l):
            continue
        l = l.lstrip("<")
        m = re.match(r"^[A-Z0-9][A-Z0-9<]?([A-Z]{3})([A-Z]+(?:<[A-Z]+)*)<<([A-Z]+(?:<[A-Z]+)*)", l)
        if m:
            surname = m.group(2).replace("<", " ").strip()
            given = m.group(3).replace("<", " ").strip()
            out["name_latin"] = f"{given} {surname}".strip()
        if i + 1 < len(compact) and re.fullmatch(r"[A-Z0-9<]{30,}", compact[i + 1]):
            m2 = re.match(r"^([A-Z0-9<]{9})(\d)([A-Z]{3})", compact[i + 1])
            if m2:
                out["passport"] = m2.group(1).replace("<", "")
                out["nationality_code"] = m2.group(3)
        if out:
            break
    return out


# ---------------------------------------------------------------------------
# Flight ticket parsing
# ---------------------------------------------------------------------------

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}

CITY_TO_IATA = {
    "aden": "ADE", "sanaa": "SAH", "sana'a": "SAH", "seiyun": "GXF", "say'un": "GXF", "mukalla": "RIY",
    "riyan": "RIY", "taiz": "TAI", "hodeidah": "HOD", "socotra": "SCT", "amman": "AMM", "cairo": "CAI",
    "jeddah": "JED", "riyadh": "RUH", "medina": "MED", "madinah": "MED", "dubai": "DXB", "sharjah": "SHJ",
    "abu dhabi": "AUH", "doha": "DOH", "kuwait": "KWI", "muscat": "MCT", "khartoum": "KRT",
    "addis ababa": "ADD", "djibouti": "JIB", "istanbul": "IST", "beirut": "BEY", "kuala lumpur": "KUL",
    "jakarta": "CGK", "karachi": "KHI", "lahore": "LHE", "dhaka": "DAC", "delhi": "DEL", "new delhi": "DEL",
    "mumbai": "BOM", "casablanca": "CMN", "algiers": "ALG", "tunis": "TUN", "tripoli": "TIP",
    "baghdad": "BGW", "damascus": "DAM", "mogadishu": "MGQ", "hargeisa": "HGA", "salalah": "SLL",
    "taif": "TIF", "dammam": "DMM", "bahrain": "BAH", "hurghada": "HRG", "alexandria": "HBE",
    "luxor": "LXR", "aswan": "ASW", "port sudan": "PZU", "asmara": "ASM", "nairobi": "NBO",
}

AIRLINES = {
    "IY": "الخطوط الجوية اليمنية (Yemenia)", "SV": "الخطوط السعودية (Saudia)", "MS": "مصر للطيران (EgyptAir)",
    "RJ": "الملكية الأردنية (Royal Jordanian)", "QR": "القطرية (Qatar Airways)", "EK": "طيران الإمارات (Emirates)",
    "EY": "الاتحاد للطيران (Etihad)", "FZ": "فلاي دبي (flydubai)", "G9": "العربية للطيران (Air Arabia)",
    "XY": "طيران ناس (flynas)", "F3": "طيران أديل (flyadeal)", "KU": "الكويتية (Kuwait Airways)",
    "WY": "الطيران العُماني (Oman Air)", "TK": "التركية (Turkish Airlines)", "ME": "طيران الشرق الأوسط (MEA)",
}

_CITY = r"[A-Za-z][A-Za-z.'\-]*(?: [A-Za-z][A-Za-z.'\-]*)?"
_STATUS = r"OK|HK|RQ|NS|SA|LL|TK|UN|UC|KK|KL"
# carrier + flight + class + (date | status): independent of how the PDF engine orders the city names
ROW_RE = re.compile(
    r"(?<![\w/.])([A-Z0-9]{2})\s+(OPEN|\d{1,4})\s+([A-Z])\b"
    r"(?:\s+(\d{1,2}[.\- ]?[A-Za-z]{3}[.\- ]?\d{2,4}))?"
    r"(?:\s+(\d{1,2}:\d{2})\s+(\d{1,2}:\d{2}))?"
    r"(?:\s+(" + _STATUS + r")\b)?"
)
# "Aden / Amman IY 648 C" (only some extractors keep the cities next to the row)
CITY_PAIR_RE = re.compile(
    r"(" + _CITY + r")\s*/\s*(" + _CITY + r")\s+[A-Z0-9]{2}\s+(?:OPEN|\d{1,4})\s+[A-Z]\b"
)


def parse_flight_date(s: str):
    m = re.match(r"^(\d{1,2})[.\- ]?([A-Za-z]{3})[.\- ]?(\d{2,4})$", (s or "").strip())
    if not m:
        return "", ""
    d, mon, y = int(m.group(1)), m.group(2).upper(), int(m.group(3))
    if mon not in MONTHS:
        return "", ""
    if y < 100:
        y += 2000
    try:
        iso = datetime(y, MONTHS[mon], d).strftime("%Y-%m-%d")
    except ValueError:
        return "", ""
    return iso, f"{d:02d}{mon.lower()}{y % 100:02d}"


def parse_passenger(raw: str):
    raw = re.sub(r"\s+", " ", raw or "").strip()
    if not raw:
        return "", "", ""
    title = ""
    if "/" in raw:
        surname, given = [p.strip() for p in raw.split("/", 1)]
    else:
        surname, given = "", raw
    m = re.search(r"\s+(MRS|MR|MS|MISS|MSTR|DR|CHD|INF)$", given, re.IGNORECASE)
    if m:
        title = m.group(1).upper()
        given = given[: m.start()].strip()
    full = f"{given} {surname}".strip()
    return full, title, raw


def parse_flight_ticket(text: str, data: dict) -> None:
    flat = re.sub(r"\s+", " ", text)
    data["document_type"] = "تذكرة طيران دولية"

    raw_name = first_match([
        r"PASSENGER\s+NAME\s*:?\s*([^\n]+)",
        r"NAME\s+OF\s+PASSENGER\s*:?\s*([^\n]+)",
    ], text)
    if not raw_name:
        raw_name = first_match([r"\b([A-Z][A-Z' -]{2,}/[A-Z][A-Z' -]{2,})\b"], text, flags=re.MULTILINE)
    full, title, raw = parse_passenger(raw_name or "")
    if full:
        data["full_name"] = full
        data["name_latin"] = full
        data["title"] = title

    pnr = re.search(r"BOOKING\s+REF\.?\s*:?\s*(?!TICKET\b|ISSUED\b|PASSENGER\b)([A-Z0-9]{5,8})\b", text)
    if pnr:
        data["pnr"] = pnr.group(1)

    tkt = re.search(r"TICKET\s+NUMBER\s*:?\s*(\d{3}\s?\d{10})", text, re.IGNORECASE)
    if not tkt:
        tkt = re.search(r"(?<!\d)(\d{3}\s?\d{10})(?!\d)", text)
    if tkt:
        data["ticket_number"] = re.sub(r"\s+", "", tkt.group(1))
        data["reference_number"] = data["ticket_number"]

    exch = re.search(r"ISSUED\s+IN\s+EXCHANGE\s+FOR\s*:?\s*(\d{3}\s?\d{10})", text, re.IGNORECASE)
    if exch:
        data["exchanged_from_ticket"] = re.sub(r"\s+", "", exch.group(1))

    issued = re.search(
        r"ISO:\s*[A-Z]{2}\s+(\d{1,2}[A-Z]{3}\d{2,4})\s+(\d{1,2}:\d{2})\s+(\d{6,10})\s+([A-Z][A-Z0-9]{3,40})", flat)
    if issued:
        data["issue_date"] = parse_flight_date(issued.group(1))[0]
        data["issued_by"] = f"{issued.group(4)} ({issued.group(3)})"

    endorsement = first_match([r"ENDORSEMENT\s*/\s*RESTRICTION\s*:?\s*([^\n]+)"], text) or ""

    # ----- coupons / segments -----
    segs = []
    seen = set()
    matches = [m for m in ROW_RE.finditer(flat) if m.group(4) or m.group(7)]
    city_pairs = [(m.group(1), m.group(2)) for m in CITY_PAIR_RE.finditer(flat)]
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else m.end() + 120
        tail = flat[m.end(): min(end, m.end() + 120)]
        flight_no = m.group(2)
        date_iso, date_disp = parse_flight_date(m.group(4) or "")
        basis = re.match(r"\s*([A-Z0-9]{4,10})\b", tail)
        if basis and basis.group(1) in {"FARE", "CALCULATION", "AREA", "END"}:
            basis = None
        bag = re.search(r"(?<![\d/])(\d{1,3})\s*(KG|PCS|PC)\b", tail)
        nva = re.findall(r"\d{2}/\d{2}/\d{4}", tail)
        valid_iso = parse_date_any(nva[-1]) if nva else ""
        pair = city_pairs[len(segs)] if len(city_pairs) >= len(matches) and len(segs) < len(city_pairs) else ("", "")
        seg = {
            "from_city": pair[0], "to_city": pair[1],
            "airline": m.group(1), "flight": flight_no, "class": m.group(3),
            "date": date_iso, "date_display": date_disp,
            "dep_time": m.group(5) or "", "arr_time": m.group(6) or "",
            "status": m.group(7) or "",
            "fare_basis": basis.group(1) if basis else "",
            "baggage": f"{bag.group(1)} {bag.group(2)}" if bag else "",
            "valid_until": valid_iso,
            "open": flight_no == "OPEN" or not date_iso,
        }
        key = (seg["airline"], seg["flight"], seg["date"], seg["from_city"])
        if key in seen:
            continue
        seen.add(key)
        segs.append(seg)

    # ----- IATA codes from the fare calculation line (e.g. ADE IY AMM405.30 IY ADE223.04NUC628.34END) -----
    fare_codes: List[str] = []
    fc = re.search(r"FARE\s+CALCULATION\s+AREA\s*:?\s*([^\n]+)", text, re.IGNORECASE)
    if fc:
        body = re.split(r"NUC\s*\d|END\b", fc.group(1))[0]
        fare_codes = re.findall(r"(?:^|\s)(?:X/)?([A-Z]{3})(?=\s|\d)", body)

    def city_code(name: str) -> str:
        return CITY_TO_IATA.get((name or "").strip().lower(), "")

    if segs:
        if len(fare_codes) == len(segs) + 1:
            for i, seg in enumerate(segs):
                seg["from"], seg["to"] = fare_codes[i], fare_codes[i + 1]
        else:
            for seg in segs:
                seg["from"], seg["to"] = city_code(seg["from_city"]), city_code(seg["to_city"])

    codes: List[str] = []
    if segs and all(s.get("from") and s.get("to") for s in segs):
        codes = [segs[0]["from"]]
        for s in segs:
            if codes[-1] != s["from"]:
                codes.append(s["from"])
            codes.append(s["to"])
    elif len(fare_codes) >= 2:
        codes = fare_codes
    if codes:
        data["route"] = " -> ".join(c.lower() for c in codes)

    data["segments"] = segs
    one_way = bool(re.search(r"ONE\s*WAY", endorsement or text, re.IGNORECASE))
    round_trip = bool(re.search(r"ROUND\s*TRIP", endorsement or text, re.IGNORECASE))

    if segs:
        first = segs[0]
        data["airline_code"] = first["airline"]
        outbound_arr = segs[-1]["arr_time"] if (one_way and len(segs) > 1) else first["arr_time"]
        data["outbound"] = {
            "date": first["date_display"], "dep": first["dep_time"], "arr": outbound_arr,
            "flight": f'{first["airline"]}{first["flight"]}', "baggage": first["baggage"],
        }
        data["date"] = (
            datetime.strptime(first["date"], "%Y-%m-%d").strftime("%d/%m/%Y") if first["date"] else NA)
        if one_way or len(segs) == 1:
            data["trip_type"] = "ذهاب فقط"
            data["return_trip"] = {"status": "one_way", "date": "ذهاب فقط (One Way)", "dep": "-", "arr": "-"}
        else:
            last = segs[-1]
            data["trip_type"] = "ذهاب وعودة"
            if last["open"]:
                until = ""
                if last["valid_until"]:
                    y, mo, d = last["valid_until"].split("-")
                    until = f" (حتى {d}{list(MONTHS)[int(mo) - 1].lower()}{y[2:]})"
                data["return_trip"] = {
                    "status": "open", "date": "OPEN" + until, "dep": "مفتوح", "arr": "مفتوح",
                    "flight": f'{last["airline"]}{last["flight"]}', "baggage": last["baggage"],
                }
            else:
                data["return_trip"] = {
                    "status": "dated", "date": last["date_display"], "dep": last["dep_time"],
                    "arr": last["arr_time"], "flight": f'{last["airline"]}{last["flight"]}',
                    "baggage": last["baggage"],
                }

    # ----- airline / company -----
    low = text.lower()
    if data.get("airline_code") and data["airline_code"] in AIRLINES:
        data["extra1"] = AIRLINES[data["airline_code"]]
    elif "yemenia" in low or "yemen airways" in low:
        data["extra1"] = AIRLINES["IY"]


# ---------------------------------------------------------------------------
# Visa parsing
# ---------------------------------------------------------------------------

def parse_visa(lines: List[str], text: str, data: dict) -> None:
    mrz = parse_mrz(lines)

    is_umrah = bool(re.search(r"umrah|عمرة|المعتمر", text, re.IGNORECASE))
    data["document_type"] = "تأشيرة عمرة" if is_umrah else "تأشيرة"

    name_raw = get_field(lines, "name") or ""
    # get_field strips known labels, including the OCR noise token "السم".
    # Recover that marker from the original labelled line before deciding to reverse.
    if any(LABEL_RES["name"].search(line) and re.search(r"\bالسم\s*$", line) for line in lines):
        name_raw = f"{name_raw} السم"
    name = clean_person_name(reverse_arabic_field_if_marked(name_raw, "name"))
    if name:
        data["full_name"] = name
    elif mrz.get("name_latin"):
        data["full_name"] = mrz["name_latin"]
    if mrz.get("name_latin"):
        data["name_latin"] = mrz["name_latin"]

    passport = get_field(lines, "passport", PASSPORT_RE)
    if passport:
        data["passport_number"] = passport.upper()
        if mrz.get("passport") and mrz["passport"] != data["passport_number"]:
            data["warnings"].append("رقم الجواز في النص لا يطابق رقم الجواز في سطر MRZ، راجعه يدوياً")
    elif mrz.get("passport"):
        data["passport_number"] = mrz["passport"]

    visa_no = get_field(lines, "visa_no", VISA_NO_RE)
    if not visa_no:
        m = re.search(r"(?<!\d)(\d{10})(?!\d)", text)
        visa_no = m.group(1) if m else None
    if visa_no:
        data["reference_number"] = visa_no

    issue = get_field(lines, "issue_date", DATE_VAL_RE)
    until = get_field(lines, "valid_until", DATE_VAL_RE)
    birth = get_field(lines, "birth", DATE_VAL_RE)
    if issue:
        data["date"] = issue
        data["issue_date"] = parse_date_any(issue)
    if until:
        data["valid_until"] = parse_date_any(until)
        data["valid_until_text"] = until
    if birth:
        data["birth_date"] = birth

    duration_line = get_field(lines, "duration")
    if duration_line:
        m = DURATION_RE.search(duration_line) or re.search(r"(\d{1,3})", duration_line)
        if m:
            data["duration_days"] = int(m.group(1))

    app_no = get_field(lines, "application", APPLICATION_RE)
    if app_no:
        data["application_number"] = app_no.upper()

    place = get_field(lines, "place_of_issue")
    if place:
        data["place_of_issue"] = place
    vtype = get_field(lines, "visa_type")
    if vtype:
        data["visa_type"] = vtype

    operator = reverse_arabic_field_if_marked(get_field(lines, "operator") or "", "operator")
    agent = reverse_arabic_field_if_marked(get_field(lines, "agent") or "", "agent")
    if operator:
        data["extra1"] = _clean_value(operator)
    if agent:
        data["extra2"] = _clean_value(agent)
        if not operator:
            data["extra1"] = data["extra2"]

    nat = get_field(lines, "nationality")
    if nat:
        data["nationality"] = normalize_country_label(nat) or NA
    if data["nationality"] == NA and mrz.get("nationality_code"):
        data["nationality"] = normalize_country_label(mrz["nationality_code"]) or NA


# ---------------------------------------------------------------------------
# Generic / local tickets
# ---------------------------------------------------------------------------

def parse_generic(lines: List[str], text: str, data: dict, local: bool) -> None:
    data["document_type"] = "تذكرة نقل / سفر" if local else "مستند غير معروف"
    name = clean_person_name(get_field(lines, "name") or "")
    if name:
        data["full_name"] = name
    passport = get_field(lines, "passport", PASSPORT_RE)
    if passport:
        data["passport_number"] = passport.upper()
    ref = get_field(lines, "reference", re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9-]{5,20})(?![A-Za-z0-9])"))
    if ref:
        data["reference_number"] = ref
    dt = get_field(lines, "travel_date", DATE_VAL_RE) or first_match(
        [r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", r"\b(\d{4}/\d{2}/\d{2})\b"], text)
    if dt:
        data["date"] = dt
    nat = get_field(lines, "nationality")
    if nat:
        data["nationality"] = normalize_country_label(nat) or NA
    for needle, label in (("البركة", "شركة البركة للنقل"), ("سماء النخبة", "سماء النخبة للسفريات")):
        if needle in text:
            data["extra1"] = label
            break


def new_result() -> dict:
    return {
        "full_name": NA, "name_latin": "", "title": "",
        "passport_number": NA, "reference_number": NA, "date": NA,
        "nationality": NA, "extra1": NA, "extra2": "",
        "document_type": "مستند غير معروف",
        "issue_date": "", "valid_until": "", "valid_until_text": "", "duration_days": "",
        "birth_date": "", "application_number": "", "visa_type": "", "place_of_issue": "",
        "pnr": "", "ticket_number": "", "exchanged_from_ticket": "", "issued_by": "",
        "airline_code": "", "route": "", "trip_type": "",
        "segments": [], "outbound": {}, "return_trip": {},
        "warnings": [],
    }


def parse_universal_document(raw_text: str, reverse_arabic="auto") -> dict:
    mode = normalize_mode(reverse_arabic)
    data = new_result()
    text, lines = prepare_lines(raw_text, mode)
    if not text:
        return data

    low = text.lower()
    strong_ticket = (
        "itinerary receipt" in low or "ticket number" in low or bool(re.search(r"PASSENGER\s+NAME", text))
    )
    visa_markers = bool(re.search(
        r"visa\s*(?:no|number)|no\.?\s*visa|umrah\s+operator|operator\s+umrah|ksa\s*visa|evisa", low
    )) or any(k in text for k in ("تأشيرة", "التأشيرة", "نسك", "وزارة الحج"))
    weak_ticket = any(k in low for k in ("e-ticket", "eticket", "booking ref", "yemenia", "yemen airways"))
    local_ticket = any(k in text for k in ("البركة", "سماء النخبة", "رقم التذكرة"))

    if strong_ticket:
        parse_flight_ticket(text, data)
    elif visa_markers:
        parse_visa(lines, text, data)
    elif weak_ticket:
        parse_flight_ticket(text, data)
    else:
        parse_generic(lines, text, data, local=local_ticket)
    return data


# ---------------------------------------------------------------------------
# OCR + endpoint
# ---------------------------------------------------------------------------

PDF_ENGLISH_LABELS = {
    "Visa No.": "Visa No",
    "Date of Issue": "Date of Issue",
    "Valid Until": "Valid Until",
    "Duration of Stay": "Duration of Stay",
    "Passport No.": "Passport No",
    "Place of issue": "Place of issue",
    "Name": "Name",
    "Birth Date": "Birth Date",
    "Nationality": "Nationality",
    "Type Of Visa": "Type Of Visa",
    "Umrah Operator": "Umrah Operator",
    "External Agent": "External Agent",
    "Application No.": "Application No",
}


def _is_latin_token(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token or ""))


def _is_arabic_token(token: str) -> bool:
    return bool(AR_RE.search(token or ""))


def _order_pdf_value(words) -> str:
    """Order value words while respecting RTL Arabic runs inside LTR values."""
    ordered = sorted(words, key=lambda w: w[0])
    out = []
    i = 0
    while i < len(ordered):
        j = i + 1
        kind = _is_arabic_token(ordered[i][4])
        while j < len(ordered) and _is_arabic_token(ordered[j][4]) == kind:
            j += 1
        run = ordered[i:j]
        if kind:
            run = list(reversed(run))
        out.extend(w[4] for w in run)
        i = j
    return " ".join(out)


def extract_pdf_layout_text(page) -> str:
    """Rebuild labelled PDF rows from word coordinates.

    Chromium-generated Arabic visa PDFs expose words in visual RTL order.
    Reading the plain text stream therefore produces values such as
    ``عقيلName علي``. Coordinates preserve the actual three-column layout:
    English label on the left, value in the middle, Arabic label on the right.
    """
    words = page.get_text("words")
    if not words:
        return ""
    rows = []
    for word in sorted(words, key=lambda w: (w[1], w[0])):
        if not rows or abs(word[1] - rows[-1][0]) > 7:
            rows.append([word[1], [word]])
        else:
            rows[-1][1].append(word)

    lines = []
    pending_label = None
    for _, row in rows:
        row = sorted(row, key=lambda w: w[0])
        latin_left = [w for w in row if w[0] < 300 and _is_latin_token(w[4])]
        latin_label = " ".join(w[4] for w in latin_left)
        matched = None
        for label, canonical in PDF_ENGLISH_LABELS.items():
            if label.lower() in latin_label.lower():
                matched = canonical
                break

        if matched:
            label_words = latin_left
            value_words = [w for w in row if w[0] > (max(w[2] for w in label_words) if label_words else 0) and w[0] < 460]
            # In the duplicate lower section, the Arabic label is immediately
            # beside the English label and there is no value on that row. Keep
            # the label pending so the numeric value on the next row attaches
            # to it instead of becoming "Visa No: رقم التأشيرة".
            if value_words and all(_is_arabic_token(w[4]) for w in value_words) and min(w[0] for w in value_words) > 380:
                value_words = []
            value = _order_pdf_value(value_words)
            if value:
                lines.append(f"{matched}: {value}")
                pending_label = None
            else:
                pending_label = matched
            continue

        if pending_label:
            value_words = [w for w in row if 120 < w[0] < 460]
            value = _order_pdf_value(value_words)
            if value:
                lines.append(f"{pending_label}: {value}")
                pending_label = None

    return "\n".join(lines)

def ocr_image(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width < 1400:
        scale = min(2.0, 1400 / max(image.width, 1))
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    try:
        candidates = []
        variants = [image, ImageOps.autocontrast(ImageOps.grayscale(image))]
        for variant in variants:
            for psm in (6, 11):
                data = pytesseract.image_to_data(
                    variant,
                    lang=OCR_LANG,
                    config=f"--oem 3 --psm {psm}",
                    output_type=pytesseract.Output.DICT,
                )
                words = [w.strip() for w in data.get("text", []) if w and w.strip()]
                text = "\n".join(words)
                confidences = [
                    float(c) for c in data.get("conf", [])
                    if str(c).strip() not in {"", "-1"} and float(c) >= 0
                ]
                if text:
                    score = (statistics.mean(confidences) if confidences else 0.0, len(text))
                    candidates.append((score, text))
        return max(candidates, key=lambda item: item[0])[1] if candidates else ""
    except pytesseract.TesseractError as exc:
        logger.warning("Configured OCR language unavailable; retrying English OCR.")
        try:
            return pytesseract.image_to_string(image, lang="eng", config="--oem 3 --psm 6")
        except pytesseract.TesseractError:
            raise HTTPException(
                status_code=503,
                detail="محرك OCR أو ملفات اللغة غير متاحة على الخادم. تحقق من تثبيت Tesseract وبيانات ara/eng."
            ) from exc


async def read_limited_upload(file: UploadFile) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"حجم الملف يتجاوز الحد المسموح ({MAX_FILE_BYTES // (1024 * 1024)} MB).")
        chunks.append(chunk)
    if not total:
        raise HTTPException(status_code=400, detail="الملف فارغ.")
    return b"".join(chunks)


@app.post("/extract-document/")
async def extract_document(
    file: UploadFile = File(...),
    reverse_arabic: str = Query(default=REVERSE_ARABIC_DEFAULT, description="auto | true | false — معالجة اتجاه الكلمات العربية")
):
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="نوع الملف غير مدعوم. ارفع PDF أو PNG أو JPG/JPEG.")

    mode = normalize_mode(reverse_arabic)
    extracted_parts = []
    try:
        file_bytes = await read_limited_upload(file)
        if suffix == ".pdf":
            if not file_bytes.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="الملف لا يبدو مستند PDF صالحًا.")
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    if doc.is_encrypted:
                        raise HTTPException(status_code=400, detail="ملفات PDF المحمية بكلمة مرور غير مدعومة.")
                    if len(doc) == 0:
                        raise HTTPException(status_code=400, detail="ملف PDF لا يحتوي على صفحات.")
                    if len(doc) > MAX_PDF_PAGES:
                        raise HTTPException(status_code=413, detail=f"عدد صفحات PDF يتجاوز الحد ({MAX_PDF_PAGES} صفحة).")
                    for page in doc:
                        page_text = page.get_text("text") or ""
                        if len(page_text.strip()) > 30:
                            layout_text = extract_pdf_layout_text(page)
                            extracted_parts.append(layout_text or page_text)
                        else:
                            pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
                            image = Image.open(io.BytesIO(pix.tobytes("png")))
                            extracted_parts.append(ocr_image(image))
                            pix = None
            except HTTPException:
                raise
            except Exception as exc:
                logger.info("PDF parsing failed: %s", type(exc).__name__)
                raise HTTPException(status_code=400, detail="تعذر قراءة PDF. تأكد أن الملف سليم وغير تالف.") from exc
        else:
            try:
                with Image.open(io.BytesIO(file_bytes)) as image:
                    image.verify()
                with Image.open(io.BytesIO(file_bytes)) as image:
                    extracted_parts.append(ocr_image(image))
            except UnidentifiedImageError as exc:
                raise HTTPException(status_code=400, detail="الصورة غير صالحة أو تنسيقها غير مدعوم.") from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.info("Image parsing failed: %s", type(exc).__name__)
                raise HTTPException(status_code=400, detail="تعذر قراءة الصورة. جرّب حفظها بصيغة PNG أو JPG.") from exc

        extracted_text = "\n".join(part for part in extracted_parts if part).strip()
        if not extracted_text:
            return {
                "filename": filename,
                "success": True,
                "server_normalized": True,
                "reverse_mode": mode,
                "parsed_data": parse_universal_document("", reverse_arabic=mode),
                "raw_text": "",
                "extracted_text": "",
                "warning": "لم يتم استخراج نص واضح. جرّب صورة أوضح أو راجع المستند يدويًا."
            }

        parsed_result = parse_universal_document(extracted_text, reverse_arabic=mode)
        return {
            "filename": filename,
            "success": True,
            "server_normalized": True,
            "reverse_mode": mode,
            "parsed_data": parsed_result,
            "raw_text": extracted_text,
            "extracted_text": extracted_text,
            "warning": "النتيجة استخراج آلي وليست تحققًا من صحة الوثيقة. راجع البيانات يدويًا."
        }
    finally:
        await file.close()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled API error on %s", request.url.path)
    return JSONResponse(status_code=500, content={
        "success": False,
        "error": "حدث خطأ داخلي أثناء معالجة المستند. حاول مرة أخرى أو راجع سجلات الخادم."
    })
