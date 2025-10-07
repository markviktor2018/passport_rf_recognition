#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fms_units.py

Нормализация и мэппинг кода подразделения ↔ варианты названий из справочника.

- Поддерживает коды "010-002", "010 002", "010002" -> "010-002".
- Для одного кода хранит ВСЕ варианты названий (нужны для матчинга с шумным OCR).
- Ранжирование кандидатов учитывает:
    * max(WRatio, token_set_ratio) из rapidfuzz
    * бонусы за «редкие» токены из OCR (топонимы/специфичные слова)
    * штраф за слишком общие названия, когда в OCR есть редкие токены
- Без rapidfuzz — фолбэк по приоритетам/длине.

API:
    from fms_units import init_fms_units, normalize_unit_code, resolve_issued_by
    init_fms_units()  # путь к CSV зашит ниже
    norm_code, best_name, meta = resolve_issued_by("610041",
        "ОТДЕЛЕНИЕМ ... БОЛЬШАЯ МАРТЫНОВКА ... КОНСТАНТИНОВСКЕ")
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import csv
import re

# --- жёсткий путь к CSV (поменять при необходимости) ---
DEFAULT_CSV_PATH = Path(
    "fms_unit.csv"
)

# --- rapidfuzz (опционально) ---
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
except Exception:
    rf_process = None
    rf_fuzz = None

# --- глобальные структуры ---
# code -> список ВСЕХ нормализованных вариантов названий (UPPER + одинарные пробелы)
_CODE2NAMES_ALL: Dict[str, List[str]] = {}
_INITIALIZED = False

# Слишком «общие» заголовки (штрафуем, если в OCR есть редкие токены)
_GENERIC_RE = re.compile(r"^(ГУ МВД|МВД|ГУВМ|УВМ)\b")

# Приоритет, если rapidfuzz недоступен
_PRIORITY = [
    r"\bГУ МВД\b",
    r"\bГУВМ\b",
    r"\bУВМ\b",
    r"\bМВД\b",
    r"\bУФМС\b",
    r"\bОФМС\b",
    r"\bТП ОФМС\b",
    r"\bОТДЕЛЕНИЕМ\b",
]

# Частые «стоп-слова» в названиях (не считаем их «редкими»)
_STOPWORDS = {
    "РОССИИ", "РФ", "РОССИЙСКОЙ", "ФЕДЕРАЦИИ", "ПО", "ОБЛАСТИ", "ОБЛ.",
    "ГОРОДЕ", "Г.", "РАЙОНЕ", "Р-НЕ", "СЛ.", "ПОС.", "РП", "УЛИЦЕ", "УМВД",
    "МВД", "ГУ", "ГУМВД", "ГУ МВД", "ГУВМ", "УВМ", "УФМС", "ОФМС", "МРО",
    "ТП", "ОТДЕЛЕНИЕМ", "ОТДЕЛОМ", "ОТДЕЛ", "ОТДЕЛЕНИЕ", "ОТД-ЕМ",
}

# --- нормализация ---

def normalize_unit_code(s: str) -> Optional[str]:
    """ '010-002' / '010 002' / '010002' -> '010-002' """
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) != 6:
        return None
    return f"{digits[:3]}-{digits[3:]}"

def _norm_name(s: str) -> str:
    """UPPER + одинарные пробелы + лёгкая замена похожих символов."""
    t = (s or "").upper()
    t = t.replace("Ё", "Е")
    t = re.sub(r"[–—−]", "-", t)         # любые «длинные тире» -> -
    t = re.sub(r"[·•∙·]", ".", t)        # редкие точки -> .
    t = re.sub(r"\s+", " ", t.strip())
    return t

def _split_tokens(s: str) -> List[str]:
    """Грубое разбиение на токены (кириллица/латиница/цифры)."""
    s = _norm_name(s)
    tokens = re.findall(r"[A-ZА-ЯЁ0-9\-]+", s)
    return tokens

def _rare_tokens(s: str) -> List[str]:
    """
    Редкие/содержательные токены OCR-текста:
    - длина >= 5
    - не в стоп-словах
    - только кириллица/латиница/цифры/-
    """
    toks = _split_tokens(s)
    out = []
    for w in toks:
        if len(w) >= 5 and w not in _STOPWORDS and re.match(r"^[A-ZА-ЯЁ0-9\-]+$", w):
            out.append(w)
    # удалим дубликаты, сохраняя порядок
    seen = set()
    uniq = []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq

# --- инициализация словаря ---

def init_fms_units(csv_path: Optional[str | Path] = None) -> None:
    global _INITIALIZED, _CODE2NAMES_ALL
    p = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    if not p.exists():
        raise FileNotFoundError(f"fms_unit.csv не найден: {p}")

    _CODE2NAMES_ALL.clear()

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code_raw = (row.get("code") or row.get("Code") or "").strip()
            name_raw = (row.get("name") or row.get("Name") or "").strip()
            if not code_raw or not name_raw:
                continue
            code = normalize_unit_code(code_raw)
            if not code:
                continue
            name = _norm_name(name_raw)
            if name:
                _CODE2NAMES_ALL.setdefault(code, []).append(name)

    # уникализируем и сортируем
    for code, names in list(_CODE2NAMES_ALL.items()):
        _CODE2NAMES_ALL[code] = sorted(set(names))

    _INITIALIZED = True

# --- скоринг кандидатов ---

def _rf_score(query: str, candidates: List[str], limit: int = 10) -> List[Tuple[str, int]]:
    """
    Базовый скоринг через rapidfuzz:
      score = max(WRatio, token_set_ratio)
    """
    if rf_process is None or rf_fuzz is None or not candidates:
        return []
    q = _norm_name(query)
    if not q:
        return []

    # Сразу два набора оценок, потом берём максимум по каждому кандидату
    wr = rf_process.extract(q, candidates, scorer=rf_fuzz.WRatio, processor=lambda s: s, limit=limit)
    ts = rf_process.extract(q, candidates, scorer=rf_fuzz.token_set_ratio, processor=lambda s: s, limit=limit)

    # Превратим в словарь: option -> score
    score_map = {}
    for opt, sc, _ in wr:
        score_map[opt] = max(score_map.get(opt, 0), int(sc))
    for opt, sc, _ in ts:
        score_map[opt] = max(score_map.get(opt, 0), int(sc))

    # Вернём убывающе по score
    items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return items[:limit]

def _boost_by_rare_tokens(query: str, option: str) -> int:
    """
    Бонус за наличие редких токенов из OCR в кандидатe.
    За каждый редкий токен вхождения в опцию +5 (макс +20).
    Если опция «слишком общая» и при этом есть редкие токены — штраф -10.
    """
    toks = _rare_tokens(query)
    if not toks:
        return 0
    opt = _norm_name(option)
    bonus = 0
    for t in toks:
        if t in opt:
            bonus += 5
            if bonus >= 20:
                break
    if _GENERIC_RE.search(opt) and bonus > 0:
        bonus -= 10  # штрафуем «ГУ МВД ...», если OCR явно содержит топонимы
    return bonus

def _rank_with_context(query: str, options: List[str]) -> List[Tuple[str, int]]:
    """
    Композитный скоринг: max(WRatio, token_set_ratio) + контекстные бонусы/штрафы.
    """
    base = _rf_score(query, options, limit=min(20, len(options))) if (rf_process and rf_fuzz) else []
    if not base:
        # фолбэк без rapidfuzz: больше приоритет, короче строка, лексикографически
        # при наличии редких токенов — отдадим предпочтение опциям, где они встречаются
        toks = _rare_tokens(query)
        if toks:
            def has_any(s: str) -> int:
                s2 = _norm_name(s)
                return sum(1 for t in toks if t in s2)
            scored = [(opt, has_any(opt)) for opt in options]
            scored.sort(key=lambda x: (-(x[1]), len(_norm_name(x[0])), _norm_name(x[0])))
            return [(opt, 50 + 10*cnt) for opt, cnt in scored]  # условные баллы
        else:
            def score_no_rf(s: str) -> tuple[int, int, str]:
                pri = min((i for i, pat in enumerate(_PRIORITY) if re.search(pat, _norm_name(s))), default=len(_PRIORITY))
                return (pri, len(_norm_name(s)), _norm_name(s))
            opts = sorted(set(options), key=score_no_rf)
            return [(opt, 60 - i) for i, opt in enumerate(opts)]

    # есть rapidfuzz — добавим контекстный бонус
    ranked = []
    for opt, sc in base:
        bonus = _boost_by_rare_tokens(query, opt)
        ranked.append((opt, sc + bonus))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

# --- основной резолвер ---

def resolve_issued_by(issued_code_text: Optional[str],
                      issued_by_text: Optional[str]) -> Tuple[Optional[str], Optional[str], dict]:
    """
    Возвращает:
      (norm_code, best_name, meta)

    meta:
      {
        'source': 'dictionary'|'ocr',
        'note': str,
        'normalized_code': 'NNN-NNN'|None,
        'candidates': [('NAME', score), ...],  # после контекстного ранжирования
        'picked': 'NAME'|''
      }
    """
    if not _INITIALIZED:
        norm = normalize_unit_code(issued_code_text or "")
        return (norm, None, {
            "source": "ocr",
            "note": "FMS units dictionary is not initialized",
            "normalized_code": norm,
            "candidates": [],
            "picked": ""
        })

    norm = normalize_unit_code(issued_code_text or "")
    if not norm:
        return (None, None, {
            "source": "ocr",
            "note": "cannot normalize issued_code",
            "normalized_code": None,
            "candidates": [],
            "picked": ""
        })

    options = _CODE2NAMES_ALL.get(norm)
    if not options:
        return (norm, None, {
            "source": "ocr",
            "note": "code not found in dictionary",
            "normalized_code": norm,
            "candidates": [],
            "picked": ""
        })

    picked = None
    candidates: List[Tuple[str, int]] = []

    if issued_by_text:
        candidates = _rank_with_context(issued_by_text, options)
        if candidates:
            picked = candidates[0][0]

    if not picked:
        # фолбэк: приоритеты/длина/лексикографически
        def score_no_ctx(s: str) -> tuple[int, int, str]:
            pri = min((i for i, pat in enumerate(_PRIORITY) if re.search(pat, _norm_name(s))), default=len(_PRIORITY))
            return (pri, len(_norm_name(s)), _norm_name(s))
        picked = sorted(set(options), key=score_no_ctx)[0]

    return (norm, picked or None, {
        "source": "dictionary",
        "note": ("matched with context" if issued_by_text else "picked by priority"),
        "normalized_code": norm,
        "candidates": candidates,
        "picked": picked or ""
    })

