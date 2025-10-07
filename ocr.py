#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import re
from typing import Tuple, List, Dict, Any, Optional
import logging
from functools import lru_cache
from pathlib import Path

try:
    # Верификатор может и не быть под рукой — тогда просто не используем.
    from verifier import Verifier
except ImportError:
    Verifier = None

# -----------------------
# ЛОГИРОВАНИЕ
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# EASYOCR: КОНФИГ
# -----------------------
import easyocr

# === ВАЖНО: перенос кастомной модели в папку проекта ===
# Вместо домашней директории используем локальные пути от текущего файла.
# Структура:
#   <project_root>/helper_files/easyocr_models/
#     └─ user_network/
#         └─ passport_ru_g2.*   (твои веса/конфиг)
_THIS_DIR = Path(__file__).resolve().parent
_EASY_BASE = (_THIS_DIR / "helper_files" / "easyocr_models").resolve()
_USER_NET  = (_EASY_BASE / "user_network").resolve()

# Эти пути скармливаем в EasyOCR
EASYOCR_MODEL_DIR = str(_EASY_BASE)    # где хранить/искать веса (включая стандартные)
EASYOCR_USER_NET_DIR = str(_USER_NET)  # где лежат кастомные user_network модели

# Названия сеток (имена должны совпадать с твоими файлами/регистрацией в EasyOCR).
CUSTOM_PASSPORT_MODEL = "passport_ru_g2"   # твоя обученная сетка для паспортных полей
CUSTOM_LATIN_MODEL    = "english_g2"       # для MRZ проще так

# GPU можно выключить, если не нужен
USE_GPU = True

# Какие поля гоним через «рус+eng» кастомную сетку; MRZ — отдельной.
PASSPORT_FIELDS = {
    "birth_place", "issued_by", "fam", "name", "sername",
    "series_number", "birth_date", "issue_date", "issued_code", "sex"
}
MRZ_FIELDS = {"mrz"}

# Быстрая проверка, что папки с моделью на месте. Не валим процесс,
# просто предупредим — EasyOCR тогда попытается скачать дефолтные веса,
# но для кастомной сети это, понятно, не поможет.
def _check_local_models():
    if not _EASY_BASE.exists():
        logger.warning(f"[EasyOCR] База моделей не найдена: {_EASY_BASE}")
    if not _USER_NET.exists():
        logger.warning(f"[EasyOCR] Папка user_network не найдена: {_USER_NET}")
    else:
        # проверим, что внутри есть что-то похожее на нашу сетку
        hits = list(_USER_NET.glob(f"{CUSTOM_PASSPORT_MODEL}*"))
        if not hits:
            logger.warning(f"[EasyOCR] В user_network не найдено файлов для '{CUSTOM_PASSPORT_MODEL}'. "
                           f"Проверь размещение: {_USER_NET}")

_check_local_models()

@lru_cache(maxsize=4)
def _build_reader(lang_tuple: Tuple[str, ...], recog_network: str):
    """
    Конструируем EasyOCR.Reader с нужной сетью и папками.
    download_enabled=False — чтобы он не лез в интернет,
    всё берём из локальных путей (и твоей user_network).
    """
    logger.info(
        "Init EasyOCR.Reader: langs=%s, recog_network=%s, model_dir=%s, user_net_dir=%s",
        lang_tuple, recog_network, EASYOCR_MODEL_DIR, EASYOCR_USER_NET_DIR
    )
    return easyocr.Reader(
        list(lang_tuple),
        gpu=USE_GPU,
        model_storage_directory=EASYOCR_MODEL_DIR,
        user_network_directory=EASYOCR_USER_NET_DIR,
        recog_network=recog_network,
        detector=True,
        recognizer=True,
        download_enabled=False  # <- ключевое: работаем с локальными весами
    )

def get_reader_for_field(field_name: str):
    """
    MRZ — латиница (english_g2), остальное — наша кастомная сеть passport_ru_g2.
    """
    if field_name in MRZ_FIELDS:
        return _build_reader(("en",), CUSTOM_LATIN_MODEL)
    else:
        return _build_reader(("ru", "en"), CUSTOM_PASSPORT_MODEL)

# -----------------------
# ПРЕДОБРАБОТКА
# -----------------------
def preprocess_for_ocr(bgr: np.ndarray, field_name: str = None) -> List[np.ndarray]:
    """
    Готовим несколько «версий» одного и того же ROI.
    Берём ту, где уверенность выше.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    original_gray = gray.copy()

    # Вариант 1: базовая нормализация яркости/контраста.
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2 + 10:
        gray1 = cv2.convertScaleAbs(gray, alpha=255.0 / (p98 - p2), beta=-p2 * 255.0 / (p98 - p2))
    else:
        gray1 = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

    gray1 = np.where(gray1 < 120, cv2.convertScaleAbs(gray1, alpha=0.85, beta=-15), gray1)

    if field_name in ("name", "sername", "birth_place", "fam", "issued_by", None):
        gray1 = cv2.bilateralFilter(gray1, d=5, sigmaColor=60, sigmaSpace=60)
    else:
        gray1 = cv2.fastNlMeansDenoising(gray1, h=12, templateWindowSize=7, searchWindowSize=21)

    clip_limit = 3.0 if field_name in ("mrz", "series_number") else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray1 = clahe.apply(gray1)

    if field_name in ("name", "sername", "birth_place", "fam", "issued_by") and np.std(gray1) < 30:
        gray1 = cv2.equalizeHist(gray1)

    if field_name in ("series_number", "birth_date", "issue_date", "issued_code", "sex", None):
        kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]], dtype=np.float32)
        gray1 = cv2.filter2D(gray1, -1, kernel)
    elif field_name in ("name", "sername", "birth_place", "fam", "issued_by"):
        kernel = np.array([[-0.2, -0.2, -0.2], [-0.2, 2.6, -0.2], [-0.2, -0.2, -0.2]], dtype=np.float32)
        gray1 = cv2.filter2D(gray1, -1, kernel)

    if field_name in ("series_number", "birth_date", "issue_date", "issued_code", "sex"):
        h, w = gray1.shape
        if h < 100 or w < 100:
            gray1 = cv2.resize(gray1, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]], dtype=np.float32)
            gray1 = cv2.filter2D(gray1, -1, kernel)

    # Вариант 2: инверсия + морфология + адаптивный порог.
    gray2 = original_gray.copy()
    gray2 = cv2.bitwise_not(gray2)
    gray2 = cv2.equalizeHist(gray2)
    kernel = np.ones((2, 2), np.uint8)
    gray2 = cv2.morphologyEx(gray2, cv2.MORPH_CLOSE, kernel)
    gray2 = cv2.bilateralFilter(gray2, d=7, sigmaColor=50, sigmaSpace=50)
    gray2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

    # Вариант 3: Оцу.
    gray3 = original_gray.copy()
    _, gray3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [gray1, gray2, gray3]

# -----------------------
# ПАРАМЕТРЫ EASYOCR ПО ПОЛЯМ
# -----------------------
def easyocr_lang_and_params(name: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Подбираем allowlist и пару порогов под конкретное поле.
    Reader — отдельно, через get_reader_for_field().
    """
    if name in ("birth_place", "issued_by", "fam", "name", "sername"):
        langs = ['ru', 'en']
        params = {
            'allowlist': "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя-0123456789",
            'text_threshold': 0.4,
            'low_text': 0.3,
            'link_threshold': 0.4,
        }
    elif name == "series_number":
        langs = ['en']
        params = {'allowlist': "0123456789", 'text_threshold': 0.6, 'low_text': 0.4}
    elif name == "issued_code":
        langs = ['en']
        params = {'allowlist': "0123456789-", 'text_threshold': 0.6}
    elif name in ("birth_date", "issue_date"):
        langs = ['en']
        params = {'allowlist': "0123456789.", 'text_threshold': 0.6}
    elif name == "sex":
        langs = ['ru', 'en']
        params = {'allowlist': "МУЖЕН.мужен", 'text_threshold': 0.7}
    elif name == "mrz":
        langs = ['en']
        params = {'allowlist': "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<", 'text_threshold': 0.5}
    else:
        langs = ['ru', 'en']
        params = {'allowlist': "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-0123456789", 'text_threshold': 0.5}

    params.update({
        'decoder': 'beamsearch',
        'beamWidth': 5,
        'batch_size': 1,
        'workers': 0,
        'detail': 1,
        'paragraph': False,
        'min_size': 20,
        'contrast_ths': 0.1,
        'adjust_contrast': 0.5,
        'filter_ths': 0.003,
        'canvas_size': 2560,
        'mag_ratio': 1.5,
    })
    return langs, params

# -----------------------
# OCR НА НЕСК. ПРЕПРОЦЕССИНГАХ
# -----------------------
def ocr_text_and_conf(
    gray_variants: List[np.ndarray],
    name: str,
    reader_override=None
) -> Tuple[str, float]:
    """
    Гоним каждый вариант предобработки в OCR, копим лучший результат.
    Возвращаем текст и среднюю уверенность (в процентах).
    """
    _, params = easyocr_lang_and_params(name)
    reader = reader_override if reader_override is not None else get_reader_for_field(name)

    best_text = ""
    best_conf = 0.0

    for i, gray in enumerate(gray_variants):
        try:
            bgr_variant = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray
            results = reader.readtext(bgr_variant, **params)
            if not results:
                continue

            text = " ".join([res[1] for res in results]).strip()
            confidences = [res[2] for res in results if res[2] is not None]
            if confidences:
                conf = float(np.mean(confidences))
                logger.info(f"Вариант {i+1} для поля {name}: текст='{text}', уверенность={conf:.2f}")
                if conf > best_conf:
                    best_conf = conf
                    best_text = text

        except Exception as e:
            logger.error(f"Ошибка при распознавании варианта {i+1} для поля {name}: {e}")
            continue

    if best_conf < 0.5:
        logger.info(f"Низкая уверенность ({best_conf:.2f}) для поля {name}, пробуем без whitelist")
        try:
            for i, gray in enumerate(gray_variants):
                bgr_variant = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray
                params_no = params.copy()
                params_no.pop('allowlist', None)
                results = reader.readtext(bgr_variant, **params_no)
                if results:
                    text = " ".join([res[1] for res in results]).strip()
                    confidences = [res[2] for res in results if res[2] is not None]
                    if confidences:
                        conf = float(np.mean(confidences))
                        if conf > best_conf:
                            best_conf = conf
                            best_text = text
        except Exception as e:
            logger.error(f"Ошибка при распознавании без whitelist для поля {name}: {e}")

    return best_text, best_conf * 100.0

# -----------------------
# ПОСТОБРАБОТКА
# -----------------------
def postprocess_field(name: str, text: str, all_fields: Optional[Dict[str, str]] = None) -> str:
    """
    Чистим пробелы/переводы строк. Если подключён Verifier — даём ему слово.
    На выходе — строка для записи в финальный JSON.
    """
    t = text.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()

    if Verifier is not None and all_fields is not None:
        verifier = Verifier()
        corrected = verifier.verify_all_fields(all_fields)
        return corrected.get(name, t)

    return t

# -----------------------
# ОБРАБОТКА ПОЛЯ С ПОВТОРАМИ
# -----------------------
def process_field_with_retry(
    roi: np.ndarray,
    name: str,
    reader=None,
    max_retries: int = 3,
    all_fields: Optional[Dict[str, str]] = None
) -> Tuple[str, float]:
    """
    Пробуем распознать поле несколько раз (разные препроцессы),
    и берём лучший текст по уверенности.
    """
    if name in ("passport", "page1", "page2", "photo"):
        logger.info(f"Пропуск поля {name}")
        return "", 0.0

    best_text = ""
    best_conf = 0.0

    for attempt in range(max_retries):
        try:
            gray_variants = preprocess_for_ocr(roi, name)
            text, conf = ocr_text_and_conf(gray_variants, name, reader_override=reader)
            processed_text = postprocess_field(name, text, all_fields=all_fields)
            logger.info(f"Попытка {attempt+1} для '{name}': '{text}' -> '{processed_text}', conf={conf:.2f}%")
            if conf > best_conf:
                best_conf = conf
                best_text = processed_text
            if conf > 80.0:
                break
        except Exception as e:
            logger.error(f"Ошибка при обработке поля {name} (попытка {attempt+1}): {e}")

    return best_text, best_conf

