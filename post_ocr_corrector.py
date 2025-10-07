#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
post_ocr_corrector.py

Лёгкий пост-корректор ФИО:
- Для полей name (имя), fam (фамилия), sername (отчество) ищет ближайшие совпадения
  в CSV-словарях (names_only.csv / surnames_only.csv / patronymics_only.csv) через RapidFuzz.
- Учитывает «подсказку пола» из контекста (sex/окончания) и спец-правила для частых OCR-ошибок.
- Остальные поля возвращаются как есть.

Как подключить словари:
  * вызови init_dictionaries(base_dir="/abs/path/to/russiannames_csv")
  * либо выставь переменную окружения RUSSIANNAMES_DIR
  * либо задай DEFAULT_DIR ниже

CSV: в первой колонке — слово (разделители ; , или таб).

Зависимости:  pip install rapidfuzz
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple, List, Optional, Iterable
from pathlib import Path
import os
import re
import sys
import logging

# -------------------- ЛОГИ --------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -------------------- ЗАВИСИМОСТИ --------------------

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
except Exception:
    rf_process = None
    rf_fuzz = None
    logger.warning("RapidFuzz не найден.  pip install rapidfuzz — иначе коррекция ФИО будет отключена.")

# -------------------- КОНФИГ --------------------

DEFAULT_DIR = None  # можно указать строкой путь по умолчанию, если удобно
DEFAULT_NAME_FILE = "helper_files/names_only.csv"
DEFAULT_SURNAME_FILE = "helper_files/surnames_only.csv"
DEFAULT_PATR_FILE = "helper_files/patronymics_only.csv"

TARGET_FIELDS = {"name", "fam", "sername"}

# Глобальные кэши словарей (в нормализованном виде)
_NAMES: Optional[List[str]] = None
_SURNAMES: Optional[List[str]] = None
_PATRONYMICS: Optional[List[str]] = None

# -------------------- УТИЛИТЫ --------------------

def _normalize_token(s: str) -> str:
    if s is None:
        return ""
    x = s.strip().lower()
    x = x.replace("ё", "е")
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\s*-\s*", "-", x)
    return x

# «Сглаживание» часто путаемых символов (О/0, I/1, й/и и т.п.)
_CONFUSION_MAP = str.maketrans({
    "0": "о", "o": "о",
    "1": "и", "l": "и", "i": "и",
    "3": "з",
    "6": "б",
    "й": "и",
})
def _confusion_normalize(s: str) -> str:
    return _normalize_token(s).translate(_CONFUSION_MAP)

def _title_russian(s: str) -> str:
    """Title Case для русских ФИО (с сохранением дефисов)."""
    def cap(part: str) -> str:
        return part[:1].upper() + part[1:].lower() if part else part
    return " ".join("-".join(cap(p) for p in chunk.split("-")) for chunk in s.split())

def _read_first_column_many(file_list: List[Path]) -> Optional[List[str]]:
    """
    Пытается открыть файлы по очереди и вернуть первую колонку в виде списка.
    Печатает полный путь каждого файла, который пробует открыть.
    """
    for p in file_list:
        try:
            rp = p if isinstance(p, Path) else Path(p)
            try:
                abs_ = rp.resolve(strict=False)
            except Exception:
                abs_ = rp
            logger.info(f"[TRY] Открываю файл: {abs_}")
            with rp.open("r", encoding="utf-8") as f:
                out: List[str] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = re.split(r"[;,\t]", line)
                    token = parts[0].strip().strip('"').strip("'")
                    if token:
                        out.append(token)
                if out:
                    logger.info(f"[OK] Прочитано строк: {len(out)} из {abs_}")
                    return out
                else:
                    logger.warning(f"[WARN] Пустой файл или нет валидных строк: {abs_}")
        except Exception as e:
            logger.warning(f"[WARN] Не удалось открыть {p}: {e}")
            continue
    return None

def _safe_len(x: Optional[Iterable]) -> int:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return 0

# -------------------- ЗАГРУЗКА СЛОВАРЕЙ --------------------

def init_dictionaries(base_dir: Optional[str | Path] = None,
                      name_file: str = DEFAULT_NAME_FILE,
                      surname_file: str = DEFAULT_SURNAME_FILE,
                      patronymic_file: str = DEFAULT_PATR_FILE) -> None:
    """
    Инициализирует глобальные словари имен/фамилий/отчеств.
    Порядок поиска:
      1) base_dir (если задан)
      2) переменная окружения RUSSIANNAMES_DIR
      3) DEFAULT_DIR (если задан глобально)
      4) текущая директория — крайний случай
    """
    global _NAMES, _SURNAMES, _PATRONYMICS

    search_dirs: List[Path] = []
    if base_dir is not None:
        search_dirs.append(Path(base_dir))
    env_dir = os.environ.get("RUSSIANNAMES_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))
    if DEFAULT_DIR:
        search_dirs.append(Path(DEFAULT_DIR))
    if not search_dirs:
        search_dirs.append(Path.cwd())

    logger.info("[DICT] Ищу CSV в директориях (по порядку):")
    for d in search_dirs:
        logger.info(f"       - {d.resolve(strict=False)}")

    name_candidates: List[Path] = [d / name_file for d in search_dirs]
    surname_candidates: List[Path] = [d / surname_file for d in search_dirs]
    patr_candidates: List[Path] = [d / patronymic_file for d in search_dirs]

    names = _read_first_column_many(name_candidates) or []
    surnames = _read_first_column_many(surname_candidates) or []
    pats = _read_first_column_many(patr_candidates) or []

    _NAMES = sorted(set(_normalize_token(x) for x in names)) if names else []
    _SURNAMES = sorted(set(_normalize_token(x) for x in surnames)) if surnames else []
    _PATRONYMICS = sorted(set(_normalize_token(x) for x in pats)) if pats else []

    logger.info(f"[DICT] Имена: {_safe_len(_NAMES)}, Фамилии: {_safe_len(_SURNAMES)}, Отчества: {_safe_len(_PATRONYMICS)}")

@lru_cache(maxsize=1)
def _load_dicts_cached() -> Tuple[List[str], List[str], List[str]]:
    """
    Возвращает кэшированные списки (names, surnames, pats).
    Если init_dictionaries ещё не вызывали — автоинициализация через окружение/DEFAULT_DIR.
    """
    global _NAMES, _SURNAMES, _PATRONYMICS
    if _NAMES is None or _SURNAMES is None or _PATRONYMICS is None:
        logger.info("[DICT] Авто-инициализация словарей (init_dictionaries() не вызывали).")
        init_dictionaries()
    return _NAMES or [], _SURNAMES or [], _PATRONYMICS or []

# -------------------- ПОДСКАЗКА ПОЛА --------------------

def _gender_hint_from_context(ctx: Optional[dict]) -> Optional[str]:
    """Возвращает 'f' / 'm' / None по sex/фамилии/отчеству в ctx."""
    if not ctx:
        return None

    # 1) явная метка пола
    sex = _normalize_token(str(ctx.get("sex", "")))
    if "жен" in sex:
        return "f"
    if "муж" in sex:
        return "m"

    # 2) отчество
    ser = _normalize_token(str(ctx.get("sername", "")))
    if ser.endswith(("овна", "евна")):
        return "f"
    if ser.endswith(("ович", "евич")):
        return "m"

    # 3) фамилия по окончаниям
    fam = _normalize_token(str(ctx.get("fam", "")))
    if fam.endswith(("ова", "ева", "ина", "ына", "ая")):
        return "f"
    if fam.endswith(("ов", "ев", "ин", "ын", "ский", "цкий")):
        return "m"

    return None

# -------------------- СПЕЦ-КАНДИДАТЫ --------------------

def _name_special_candidates(qn: str, corpus: List[str]) -> List[str]:
    """
    Спец-починки имён (ШЛИЯ -> ЮЛИЯ и т.п.), qn — уже _normalize_token().
    Возвращает список кандидатов, которые реально есть в корпусе.
    """
    cands: List[str] = []

    # Ю часто превращается в «шл», «ио/йо»
    repls = [
        ("шл", "ю"),
        ("ио", "ю"),
        ("йо", "ю"),
    ]
    for bad, good in repls:
        if bad in qn:
            cand = qn.replace(bad, good)
            if cand in corpus:
                cands.append(cand)

    # Обломанное мужское окончание: ...олп -> ...олий
    if qn.endswith("олп"):
        cand = qn[:-3] + "олий"
        if cand in corpus:
            cands.append(cand)

    # Частые огрехи с 'й'/'и': анатолй -> анатолий
    if qn.endswith("лй"):
        cand = qn[:-2] + "лий"
        if cand in corpus:
            cands.append(cand)

    # Юлия — популярный фикс для женского
    if ("юли" in qn or qn.startswith(("юл", "ул", "йул"))) and "юлия" in corpus:
        cands.append("юлия")

    # Удаляем дубликаты, сохраняем порядок
    out, seen = [], set()
    for w in cands:
        if w not in seen:
            out.append(w); seen.add(w)
    return out

def _patronymic_special_candidates(qn: str, corpus: List[str]) -> List[str]:
    """Спец-починка для отчества: 'влдимир...' -> 'владимир...'."""
    cands: List[str] = []
    if qn.startswith("влдимир"):
        cand = "владимир" + qn[len("влдимир"):]
        if cand in corpus:
            cands.append(cand)
    return cands

# -------------------- ФУЗЗИ-ПОИСК И СКОРИНГ --------------------

def _best_match(token: str, corpus: List[str], *, limit: int = 8) -> List[Tuple[str, int]]:
    """Топ-N кандидатов (кандидат, base_score 0..100)."""
    if rf_process is None or rf_fuzz is None or not corpus:
        return []
    q = _normalize_token(token)
    if not q:
        return []
    results = rf_process.extract(
        q,
        corpus,
        scorer=rf_fuzz.WRatio,
        processor=lambda s: s,  # corpus уже нормализован
        limit=limit
    )
    return [(w, int(s)) for (w, s, _) in results]

def _composite_score(q: str, cand: str, base_score: int, field: str,
                     gender_hint: Optional[str]) -> float:
    """
    Пересчёт итогового скора: base + эвристики по префиксам, длине, суффиксам и полу.
    """
    score = float(base_score)
    qn, cn = _confusion_normalize(q), _confusion_normalize(cand)

    # Сильный префикс/суффикс
    for k in (4, 3):
        if qn[:k] and qn[:k] == cn[:k]:
            score += 4.5 if k == 4 else 2.5
        if qn[-k:] and qn[-k:] == cn[-k:]:
            score += 4.0 if k == 4 else 2.0

    # Близость длины
    dlen = abs(len(qn) - len(cn))
    if dlen == 0:
        score += 2.5
    elif dlen == 1:
        score += 1.0
    elif dlen >= 3:
        score -= 2.0

    # Отчества — приоритет суффиксов
    if field == "sername":
        if cn.endswith(("ович", "евич", "овна", "евна")):
            score += 6.0
        else:
            score -= 1.5

    # Имя — мягкий приоритет популярных строк
    if field == "name" and cn in {"юлия", "анатолий", "александр", "алексей"}:
        score += 1.5

    # Буст по полу
    if gender_hint and field in ("name", "sername"):
        is_feminine = cn.endswith(("а", "я", "ия", "лия", "юлия", "овна", "евна"))
        is_masculine = cn.endswith(("ий", "й", "ович", "евич"))
        if gender_hint == "f":
            if is_feminine: score += 8.0
            if is_masculine: score -= 4.0
        elif gender_hint == "m":
            if is_masculine: score += 6.0
            if is_feminine: score -= 3.0

    return score

# -------------------- ПУБЛИЧНЫЙ API --------------------

def correct_field(
    field_name: str,
    value: str,
    topk: int = 5,
    auto_threshold: int = 85,
    aggressive: bool = True,
    context: Optional[dict] = None,   # сюда можно передать all_fields из main.py
) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Возвращает (suggested_text, candidates):
      * Если поле name/fam/sername — ищем кандидатов.
        - При score >= auto_threshold — подменяем строку (Title Case).
        - Иначе — оставляем исходное, возвращаем кандидатов для UI.
      * Для остальных полей возвращаем исходное и [].

    Пример:
        fixed, cand = correct_field("name", "шлия", context={"fam": "назарова", "sername": "владимировна"})
        -> ("Юлия", [("юлия", ...), ...])
    """
    if value is None:
        value = ""
    fld = (field_name or "").strip().lower()
    if fld not in TARGET_FIELDS:
        return value, []

    names, surnames, pats = _load_dicts_cached()
    if fld == "name":
        corpus = names
    elif fld == "fam":
        corpus = surnames
    else:  # sername
        corpus = pats

    if not corpus:
        logger.warning(f"[DICT] Пустой словарь для '{fld}'. Проверьте CSV и init_dictionaries().")
        return value, []

    gender_hint = _gender_hint_from_context(context)
    q = _normalize_token(value)

    # 1) базовые кандидаты из RapidFuzz
    base = _best_match(q, corpus, limit=max(topk, 12))

    # 2) спец-кандидаты
    extra: List[str] = []
    if fld == "name":
        extra += _name_special_candidates(q, corpus)
    elif fld == "sername":
        extra += _patronymic_special_candidates(q, corpus)

    # добавим extra как «высоковероятные»
    for w in extra:
        if w not in (bw for bw, _ in base):
            base.append((w, 94))

    # 3) рескоринг
    rescored = [(w, _composite_score(q, w, s, fld, gender_hint)) for (w, s) in base]
    rescored.sort(key=lambda x: x[1], reverse=True)

    # 4) кламп/округление и список для UI
    cand_ui: List[Tuple[str, int]] = []
    for w, sc in rescored[:max(topk, 5)]:
        z = int(max(0, min(100, round(sc))))
        cand_ui.append((w, z))

    # 5) авто-подмена?
    if cand_ui:
        best_w, best_z = cand_ui[0]
        if aggressive:
            if best_z >= auto_threshold:
                return _title_russian(best_w), cand_ui
        else:
            # консервативный режим: подменяем только при очень высоком score и коротких строках
            if best_z >= max(auto_threshold, 92) and len(q) <= max(6, len(best_w) + 1):
                return _title_russian(best_w), cand_ui

    # иначе — без изменений
    return value, cand_ui


# -------------------- быстрый локальный тест --------------------
if __name__ == "__main__":
    # Пример ручного запуска: python post_ocr_corrector.py /path/to/csv_dir "ШЛИЯ" name
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir", type=str, nargs="?")
    ap.add_argument("value", type=str, nargs="?")
    ap.add_argument("field", type=str, nargs="?", default="name")
    args = ap.parse_args()

    if args.csv_dir:
        init_dictionaries(args.csv_dir)
    else:
        init_dictionaries()

    fixed, cand = correct_field(args.field, args.value or "шлия",
                                context={"fam": "назарова", "sername": "владимировна"})
    print("INPUT :", args.value)
    print("FIXED :", fixed)
    print("CANDID:", cand)

