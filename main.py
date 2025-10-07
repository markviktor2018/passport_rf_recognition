#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Два режима:
1) CLI (по умолчанию): обработка одиночного изображения, сохранение .json и т.д.
2) SERVICE (--service): воркер RabbitMQ — слушает очередь задач, принимает base64-изображения,
   прогоняет пайплайн, отдаёт ответ в reply_to/ reply.<correlation_id>.

Зависимости доп. для сервиса:
  pip install aio-pika

Очереди:
  - вход:  passport.tasks
  - выход: reply.<correlation_id> (если в сообщении не задано reply_to)

Ожидаемый запрос в задаче (пример):
{
  "image_b64": "<BASE64>",
  "params": {
    "model": "runs/segment/train5/weights/best.pt",
    "imgsz": 1536,
    "conf": 0.25,
    "iou": 0.6
  }
}

Ответ (упрощённый вид):
{
  "status": "ok",
  "result": {
    "image": "<source>",
    "used_stage": "stage2|stage3",
    "fields": {
      "<field>": { "value": "...", "confidence": 0.99, "candidates": [...]? }
      ...
    },
    "images": {
      "final_b64": "...",
      "annotated_b64": "...",
      "final_path": "results/....jpg",
      "annotated_path": "results/....jpg"
    }
  }
}
"""

import sys
import json
import argparse
from pathlib import Path
import base64
import cv2
import time
import uuid
import os
import logging
from typing import Dict, Any, Optional, Tuple, List

# ML
from ultralytics import YOLO
import easyocr

# твои модули
from stages import (
    choose_best_angle_by_sweep,
    normalize_without_mirror,
    choose_best_k90_layout,
    choose_best_k90_photo_mode,
    draw_oriented_and_collect
)
from ocr import preprocess_for_ocr, easyocr_lang_and_params, ocr_text_and_conf, postprocess_field
from utils import crop_from_xyxy
from post_ocr_corrector import correct_field
from fms_units import init_fms_units, resolve_issued_by  # словарь подразделений

# ----------------- ЛОГ -----------------
# Логгер тут простой, в stdout. Удобно для докера и сервисного режима.
log = logging.getLogger("passport.main")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

Verifier = None  # пока не подключаем (зарезервировано под smart-верификатор/LLM и т.п.)

# --- Defaults ---
# Если ничего не передали — берём готовые веса детектора (YOLOv8-seг) отсюда.
DEFAULT_MODEL = "runs/segment/train5/weights/best.pt"
# Размер инференса. 1536 — компромисс по качеству/скорости. Работает стабильно на A100/3080.
DEFAULT_IMGSZ = 1536
# Конфиденс для детектора. 0.25 — не слишком строгий, чтоб не терять поля.
DEFAULT_CONF = 0.25
# IOU-NMS. 0.6 — норм для наших боксов/масок, без агрессивного отсева.
DEFAULT_IOU = 0.6
# По умолчанию GPU #0. Можно передать "cpu", если без видеокарты.
DEFAULT_DEV = "0"
# Куда собираем кропы под EasyOCR-тюнинг (датасет копится «на лету»).
DEFAULT_DATASET_ROOT = (
    "dataset_for_easyocr"
)

# путь к словарю подразделений (CSV) — если есть, нормализуем issued_code/issued_by
DEFAULT_FMS_CSV = "helper_files/fms_unit.csv"

# Папка для финальных и размеченных кадров (для UI/отладки)
RESULTS_DIR = Path("results")

# Порядок для сортировки полей (чтобы в json всё шло опр. логикой и OCR делался стабильно)
ALL_FIELD_ORDER = [
    "passport", "page1", "page2", "photo", "mrz", "series_number",
    "birth_date", "birth_place", "fam", "name", "sername",
    "issue_date", "issued_by", "issued_code", "sex"
]

def ensure_dir(p: Path):
    # Мелкая утилита — просто mkdir -p
    p.mkdir(parents=True, exist_ok=True)

def make_unique_filename(field_name: str, ext: str = ".png") -> str:
    # Для датасета OCR-кропов: делаем уникальное имя файла, чтобы не перезаписывалось
    ts_ms = int(time.time() * 1000)
    short_uuid = uuid.uuid4().hex[:8]
    return f"{field_name}_{ts_ms}_{short_uuid}{ext}"

def write_crop_to_dataset(dataset_root: Path, field_name: str, crop_bgr, text_candidate: str) -> Path:
    # Сохраняем кроп + в gt.txt кидаем строку filename<TAB>text
    field_dir = dataset_root / field_name
    ensure_dir(field_dir)

    fname = make_unique_filename(field_name, ".png")
    out_img_path = field_dir / fname

    cv2.imwrite(str(out_img_path), crop_bgr)

    clean_text = (text_candidate or "").replace("\r", " ").replace("\n", " ")
    clean_text = clean_text.replace("\t", " ").strip()

    gt_path = field_dir / "gt.txt"
    with open(gt_path, "a", encoding="utf-8") as gt:
        gt.write(f"{fname}\t{clean_text}\n")

    return out_img_path

# ----------- ВСПОМОГАТЕЛЬНО: сохранение и b64 для фронта -----------

def _bgr_to_b64_jpeg(bgr, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")

def _save_debug_renders(final_img, final_dets, used_stage: str) -> dict:
    """
    Сохраняет:
      - «чистый» повёрнутый кадр
      - размеченную версию (полигоны/подписи)
    Возвращает словарь с путями и base64 обеих картинок.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)

    # «чистая» финальная картинка (после всех поворотов/ориентаций)
    final_path = RESULTS_DIR / f"{ts}_{used_stage}_final.jpg"
    cv2.imwrite(str(final_path), final_img)

    # размеченная версия
    annotated_path = RESULTS_DIR / f"{ts}_{used_stage}_annotated.jpg"
    _ = draw_oriented_and_collect(final_img, final_dets, annotated_path)

    # base64 для фронта
    final_b64 = _bgr_to_b64_jpeg(final_img)

    annotated_bgr = cv2.imread(str(annotated_path))
    annotated_b64 = _bgr_to_b64_jpeg(annotated_bgr) if annotated_bgr is not None else ""

    return {
        "final_path": str(final_path),
        "annotated_path": str(annotated_path),
        "final_b64": final_b64,
        "annotated_b64": annotated_b64,
    }

# ----------------- CORE PIPELINE -----------------
# Тут весь путь: выровнять документ, выбрать правильную ориентацию, OCR, пост-обработка полей.

def run_pipeline_on_bgr(
    bgr0,
    *,
    model_path: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    dataset_root: Path
) -> Dict[str, Any]:
    """Запускает Stage1–4 и возвращает финальный JSON-словарь (как раньше .ocr.json)."""
    H0, W0 = bgr0.shape[:2]

    # Инициализация моделей
    # YOLO — только для детекта полей, OCR отдельно (EasyOCR)
    model = YOLO(model_path)
    reader = easyocr.Reader(['ru', 'en'], gpu=(device != "cpu"))

    ensure_dir(dataset_root)

    # Stage 1 — небольшая «разморозка»: пробуем несколько углов и забираем лучший (MRZ внизу, и т.п.)
    best_deg, best_score, dets_best = choose_best_angle_by_sweep(
        model, bgr0, imgsz, conf, iou, device
    )
    bgr1 = bgr0 if abs(best_deg) <= 0 else cv2.warpAffine(
        bgr0, cv2.getRotationMatrix2D((W0/2, H0/2), best_deg, 1.0), (W0, H0),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    # без зеркала, MRZ вниз — тут немного хардкода, но работает по паспорту РФ
    bgr2, dets2 = normalize_without_mirror(bgr1, dets_best)

    # Stage 2 — оцениваем четыре поворота (k*90) и берём тот, где лейаут «на месте»
    score_k, k, bgr3, dets3, W3, H3 = choose_best_k90_layout(bgr2, dets2)

    # Stage 3 — спец. режим, когда нет page1, но есть фото (разворот разворота)
    final_img = bgr3
    final_dets = dets3
    used_stage = "stage2"
    if not any(d["name"] == "page1" for d in dets3) and any(d["name"] == "photo" for d in dets3):
        score_p, kp, bgr4, dets4, W4, H4 = choose_best_k90_photo_mode(bgr3, dets3)
        final_img = bgr4
        final_dets = dets4
        used_stage = "stage3"

    # >>> сохраняем повёрнутый кадр и размеченную версию, готовим base64 для фронта
    dbg = _save_debug_renders(final_img, final_dets, used_stage)

    # Stage 4 — OCR (EasyOCR) по кропам, немного предобработки (усиление контраста, морфология, и т.д.)
    ocr_results = {}
    all_fields = {}
    order = {n: i for i, n in enumerate(ALL_FIELD_ORDER)}
    dets_sorted = sorted(final_dets, key=lambda d: order.get(d["name"], 999))

    for d in dets_sorted:
        field = d["name"]
        # пропускаем «служебные» классы (сам паспорт/страницы/фото)
        if field not in ALL_FIELD_ORDER or field in ("passport", "page1", "page2", "photo"):
            continue

        # небольшой expansion по некоторым «длинным» текстам (issued_by, birth_place)
        crop = crop_from_xyxy(
            final_img, d["xyxy"],
            expand_frac=(0.16 if field in ("issued_by", "birth_place") else 0.10)
        )
        if crop is None:
            ocr_results[field] = {"text": "", "conf_ocr": None, "conf_det": round(float(d["conf"]), 3)}
            continue

        # серия-номер у нас вертикально — разворачиваем
        if field == "series_number":
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # подготавливаем несколько вариантов (градации серого/бинаризации), прогоняем, берём лучший
        gray_variants = preprocess_for_ocr(crop, field)
        text_raw, conf_ocr = ocr_text_and_conf(gray_variants, field, reader)
        text_post = postprocess_field(field, text_raw)

        # Подсказки для ФИО (не автозаменяем)
        _, suggestions = correct_field(field, text_post, topk=5, auto_threshold=999, aggressive=True, context=all_fields)

        all_fields[field] = text_post
        # складируем кроп в датасет — потом пригодится для дообучения EasyOCR
        saved_path = write_crop_to_dataset(dataset_root, field, crop, text_post)

        ocr_results[field] = {
            "text": text_post,
            "conf_ocr": (None if conf_ocr is None else round(float(conf_ocr), 3)),
            "conf_det": round(float(d["conf"]), 3),
            "bbox_xyxy": list(map(int, d["xyxy"])),
            "crop_path": str(saved_path)
        }
        # кандидаты подсовываем только для ФИО — для UI/ручной проверки
        if field in {"fam", "name", "sername"}:
            ocr_results[field]["candidates"] = suggestions

    # Нормализация связки issued_code -> issued_by по словарю (если есть CSV)
    issued_code_text = ocr_results.get("issued_code", {}).get("text")
    issued_by_text = ocr_results.get("issued_by", {}).get("text")
    norm_code, dict_name, meta = resolve_issued_by(issued_code_text, issued_by_text)
    if norm_code:
        # приводим код к NNN-NNN (если получилось нормализовать)
        if "issued_code" in ocr_results:
            ocr_results["issued_code"]["text"] = norm_code
    if dict_name:
        # если словарь знает этот код — заменяем «выдавший орган» на каноническое наименование
        if "issued_by" not in ocr_results:
            ocr_results["issued_by"] = {}
        ocr_results["issued_by"]["text"] = dict_name
        ocr_results["issued_by"]["dict_meta"] = meta
    elif meta:
        # даже если не нашли — оставим метаданные (полезно для дебага)
        if "issued_by" not in ocr_results:
            ocr_results["issued_by"] = {}
        ocr_results["issued_by"]["dict_meta"] = meta

    result = {
        "used_stage": used_stage,
        "fields_raw": ocr_results,  # детальная форма (как раньше)
        # дебаг-инфо с картинками (и путь, и base64)
        "debug": {
            "final_image_path": dbg["final_path"],
            "annotated_image_path": dbg["annotated_path"],
            "final_image_b64": dbg["final_b64"],
            "annotated_image_b64": dbg["annotated_b64"],
        },
    }
    return result

def format_fields_for_api(fields_raw: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Преобразует в схему API:
    { field: {value: str, confidence: float, candidates?: [...] } }
    где confidence берём из conf_ocr (если есть), иначе из conf_det.

    Важный момент: conf_ocr у нас обычно 0..100, поэтому тут делим/скейлим до 0..1.
    Если conf_ocr нет — подставляем conf_det, чтобы пользователь видел хоть какую-то уверенность.
    """
    out = {}
    for k, v in fields_raw.items():
        val = v.get("text", "")
        conf = v.get("conf_ocr")
        if conf is None:
            conf = v.get("conf_det")
        # переведём в 0..1
        if isinstance(conf, (int, float)) and conf > 1.0:
            conf = round(float(conf) / 100.0, 3)
        out[k] = {
            "value": val,
            "confidence": (None if conf is None else float(conf))
        }
        if "candidates" in v:
            out[k]["candidates"] = v["candidates"]
        if "dict_meta" in v:
            out[k]["dict_meta"] = v["dict_meta"]
    return out

# ----------------- CLI -----------------
# Обычный одноразовый запуск: передали путь к картинке, получили result.json рядом

def run_cli(args) -> None:
    img_path = Path(args.image)
    bgr0 = cv2.imread(str(img_path))
    if bgr0 is None:
        print(f"ERROR: can't read image: {img_path}")
        sys.exit(1)

    # init словаря подразделений (если файла нет — просто предупреждение, не критично)
    try:
        init_fms_units(args.fms_csv)
    except Exception as e:
        log.warning(f"FMS units dictionary init failed: {e}")

    result = run_pipeline_on_bgr(
        bgr0,
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        dataset_root=Path(args.dataset_root),
    )

    fields_api = format_fields_for_api(result["fields_raw"])

    final = {
        "image": str(img_path),
        "used_stage": result["used_stage"],
        "fields": fields_api,
        # добавим пути к сохранённым картинкам, чтоб видно было в CLI
        "images": {
            "final_path": result.get("debug", {}).get("final_image_path"),
            "annotated_path": result.get("debug", {}).get("annotated_image_path"),
        }
    }

    # сохраняем рядом с картинкой *.ocr.json
    ocr_json = img_path.with_suffix(".ocr.json")
    with open(ocr_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "ocr_json": str(ocr_json),
        "used_stage": result["used_stage"],
        "final_image": final["images"]["final_path"],
        "annotated_image": final["images"]["annotated_path"],
    }, ensure_ascii=False))

# ----------------- SERVICE (RabbitMQ) -----------------
# Режим «воркер»: слушает очередь, принимает base64, гоняет пайплайн и отвечает в reply-очередь

async def service_run(amqp_url: str, task_queue: str,
                      model_path: str, imgsz: int, conf: float, iou: float,
                      device: str, dataset_root: str, fms_csv: str) -> None:
    import aio_pika
    from aio_pika.abc import AbstractIncomingMessage

    # init словаря подразделений один раз на старте
    try:
        init_fms_units(fms_csv)
    except Exception as e:
        log.warning(f"FMS units dictionary init failed: {e}")

    async def publish_reply(channel: aio_pika.RobustChannel, correlation_id: str,
                            payload: dict, reply_to: Optional[str]) -> None:
        # отвечаем либо в reply_to, либо в fallback-очередь reply.<corr_id>
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        routing_key = reply_to or f"reply.{correlation_id}"
        if not reply_to:
            await channel.declare_queue(routing_key, durable=True)
        msg = aio_pika.Message(
            body=body,
            content_type="application/json",
            correlation_id=correlation_id,
        )
        await channel.default_exchange.publish(msg, routing_key=routing_key)
        log.info(f"→ reply sent to '{routing_key}' (correlation_id={correlation_id})")

    def decode_image_from_payload(payload: Dict[str, Any]) -> Tuple[Optional[Any], str]:
        """
        Возвращает (bgr, source_label). Ожидает payload["image_b64"] или ["image_path"].

        На практике base64 — основной сценарий. image_path оставлен как запасной, когда очередь
        гоняют внутри одного хоста и доступ к файловой системе общий.
        """
        if "image_b64" in payload and payload["image_b64"]:
            try:
                raw = base64.b64decode(payload["image_b64"], validate=True)
                arr = np.frombuffer(raw, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    return bgr, "image_b64"
            except Exception as e:
                log.warning(f"bad image_b64: {e}")
        p = payload.get("image_path")
        if isinstance(p, str) and p:
            bgr = cv2.imread(p)
            if bgr is not None:
                return bgr, p
        return None, "<none>"

    async def handle_message(message: AbstractIncomingMessage, channel: aio_pika.RobustChannel) -> None:
        # каждое сообщение оборачиваем message.process — чтобы ack шел корректно
        async with message.process(ignore_processed=True):
            corr_id = message.correlation_id or ""
            reply_to = message.reply_to
            try:
                payload = json.loads(message.body.decode("utf-8", errors="ignore"))
            except Exception:
                payload = {}

            log.info(f"← task corr_id={corr_id!r}, reply_to={reply_to!r}")

            if not corr_id:
                # без corr_id отвечать некуда — ack и в лог
                log.warning("message without correlation_id — ACK w/o reply")
                return

            # параметры можно переопределить в payload["params"], это удобно для A/B и отладки
            params = payload.get("params") or {}
            _model = params.get("model", model_path)
            _imgsz = int(params.get("imgsz", imgsz))
            _conf = float(params.get("conf", conf))
            _iou  = float(params.get("iou", iou))
            _device = params.get("device", device)

            # декодим картинку
            bgr, src = decode_image_from_payload(payload)
            if bgr is None:
                resp = {"status": "error", "error": "bad_or_missing_image", "detail": "Provide 'image_b64' or 'image_path'."}
                await publish_reply(channel, corr_id, resp, reply_to)
                return

            # запускаем пайплайн
            try:
                result = run_pipeline_on_bgr(
                    bgr,
                    model_path=_model,
                    imgsz=_imgsz,
                    conf=_conf,
                    iou=_iou,
                    device=_device,
                    dataset_root=Path(dataset_root),
                )
                fields_api = format_fields_for_api(result["fields_raw"])
                resp = {
                    "status": "ok",
                    "result": {
                        "image": src,
                        "used_stage": result["used_stage"],
                        "fields": fields_api,
                        # добавим и картинки для фронта/UI
                        "images": {
                            "final_b64": result.get("debug", {}).get("final_image_b64"),
                            "annotated_b64": result.get("debug", {}).get("annotated_image_b64"),
                            "final_path": result.get("debug", {}).get("final_image_path"),
                            "annotated_path": result.get("debug", {}).get("annotated_image_path"),
                        }
                    }
                }
            except Exception as e:
                # не душим исключение — пишем стек, чтобы понять где упало (OCR/детект/IO)
                log.exception("pipeline failed")
                resp = {"status": "error", "error": "pipeline_failed", "detail": str(e)}

            await publish_reply(channel, corr_id, resp, reply_to)

    # Подключаемся к Rabbit
    log.info(f"Connecting to {amqp_url}")
    connection = await aio_pika.connect_robust(amqp_url)
    async with connection:
        channel: aio_pika.RobustChannel = await connection.channel()
        await channel.set_qos(prefetch_count=8)
        queue = await channel.declare_queue(task_queue, durable=True)
        log.info(f"Waiting for messages in queue: {task_queue}")

        await queue.consume(lambda msg: handle_message(msg, channel), no_ack=False)

        # простой «вечный» цикл — работаем пока нас не стопнут
        try:
            while True:
                await asyncio.sleep(3600)
        except (asyncio.CancelledError, KeyboardInterrupt):
            log.info("Shutdown requested")

# ----------------- ARGS & ENTRY -----------------
# Тут парсим аргументы. Если указан --service — уходим в асинхронный воркер.
# Иначе — классический CLI: main.py <image> …

def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8-seg + EasyOCR pipeline | CLI & Service modes")
    ap.add_argument("--service", action="store_true", help="Run as RabbitMQ worker service")
    ap.add_argument("--amqp_url", type=str, default=os.getenv("AMQP_URL", "amqp://guest:guest@localhost/"))
    ap.add_argument("--task_queue", type=str, default=os.getenv("TASK_QUEUE", "passport.tasks"))

    # CLI params
    ap.add_argument("image", nargs="?", help="Path to image (CLI mode)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--device", type=str, default=DEFAULT_DEV)
    ap.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--fms_csv", type=str, default=DEFAULT_FMS_CSV, help="Путь к словарю подразделений fms_unit.csv")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.service:
        # lazy import asyncio & numpy (для b64 decode)
        import asyncio
        import numpy as np  # noqa: F401  (нужно для imdecode)
        try:
            asyncio.run(service_run(
                amqp_url=args.amqp_url,
                task_queue=args.task_queue,
                model_path=args.model,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                dataset_root=args.dataset_root,
                fms_csv=args.fms_csv,
            ))
        except KeyboardInterrupt:
            log.info("Exiting…")
    else:
        if not args.image:
            print("Usage (CLI): main.py <image> [--model ...]")
            sys.exit(2)
        run_cli(args)

