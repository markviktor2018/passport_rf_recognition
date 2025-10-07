#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
worker_stub.py
Простой воркер-эхо для очереди OCR.
- Слушает очередь ocr.tasks (durable).
- Читает correlation_id и reply_to из свойств сообщения.
- Отвечает в reply_to, либо в очередь reply.<correlation_id>, сохраняя correlation_id.
- Отправляет JSON-заглушку (MVP-ответ).
"""

import asyncio
import json
import os
import sys
import logging
from typing import Optional

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

# ---------- конфиг ----------
AMQP_URL = os.getenv("AMQP_URL", "amqp://guest:guest@localhost/")
TASK_QUEUE = os.getenv("TASK_QUEUE", "passport.tasks")
# Ответ всегда application/json
CONTENT_TYPE = "application/json"

# Заглушка-результат
STUB_RESULT = {
    "image": "passport_123.jpg",
    "used_stage": "stage3",
    "fields": {
        "series_number": {"text": "60 08 123456", "conf_ocr": 99.9},
        "fam":           {"text": "ПЕТРОВ",       "conf_ocr": 98.1},
        "name":          {"text": "АЛЕКСЕЙ",      "conf_ocr": 97.2},
        "sername":       {"text": "НИКОЛАЕВИЧ",   "conf_ocr": 96.7},
        "birth_date":    {"text": "15 07 1985",   "conf_ocr": 95.3},
        "issued_code":   {"text": "610-041",      "conf_ocr": 94.5},
        "issued_by":     {"text": "ГУ МВД РОССИИ ПО РОСТОВСКОЙ ОБЛ.", "conf_ocr": 92.0},
    },
}

# ---------- логирование ----------
log = logging.getLogger("worker_stub")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


async def publish_reply(channel: aio_pika.RobustChannel,
                        correlation_id: str,
                        payload: dict,
                        reply_to: Optional[str]) -> None:
    """Публикует ответ в reply_to или в reply.<correlation_id>."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    routing_key = reply_to or f"reply.{correlation_id}"

    # если reply_to не задан — создадим временно/на всякий случай очередь
    if not reply_to:
        await channel.declare_queue(routing_key, durable=True)

    msg = aio_pika.Message(
        body=body,
        content_type=CONTENT_TYPE,
        correlation_id=correlation_id,
    )
    await channel.default_exchange.publish(msg, routing_key=routing_key)
    log.info(f"→ reply sent to '{routing_key}' (correlation_id={correlation_id})")


async def handle_message(message: AbstractIncomingMessage, channel: aio_pika.RobustChannel) -> None:
    """Обработка одного входящего сообщения."""
    async with message.process(ignore_processed=True):
        corr_id = message.correlation_id or ""
        reply_to = message.reply_to

        try:
            raw = message.body.decode("utf-8", errors="ignore")
            log.info(f"← got task (corr_id={corr_id!r}, reply_to={reply_to!r}) payload_len={len(raw)}")
        except Exception:
            log.info(f"← got task (corr_id={corr_id!r}, reply_to={reply_to!r}) payload: <binary>")

        if not corr_id:
            # Без correlation_id не сможем корректно ответить — просто ACK и лог
            log.warning("message without correlation_id — skipping reply")
            return

        # Тут мог бы быть вызов твоего пайплайна OCR, но сейчас — заглушка:
        result_envelope = {
            "status": "ok",
            "result": STUB_RESULT,
        }

        await publish_reply(channel, corr_id, result_envelope, reply_to)


async def main() -> None:
    log.info(f"Connecting to {AMQP_URL}")
    connection = await aio_pika.connect_robust(AMQP_URL)
    async with connection:
        channel: aio_pika.RobustChannel = await connection.channel()
        await channel.set_qos(prefetch_count=16)

        # Объявляем очередь задач
        queue = await channel.declare_queue(TASK_QUEUE, durable=True)
        log.info(f"Waiting for messages in queue: {TASK_QUEUE}")

        await queue.consume(lambda msg: handle_message(msg, channel), no_ack=False)

        # Бесконечное ожидание
        try:
            while True:
                await asyncio.sleep(3600)
        except (asyncio.CancelledError, KeyboardInterrupt):
            log.info("Shutdown requested")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Exiting…")

