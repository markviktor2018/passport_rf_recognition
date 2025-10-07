#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uuid
import json
import base64
import asyncio
from typing import Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# CORS-мидлварь: чтобы фронтенд с любого домена мог дёргать этот API
from fastapi.middleware.cors import CORSMiddleware

import aio_pika
from aio_pika.abc import AbstractConnection, AbstractChannel, AbstractQueue

# ----------------------------
# Конфигурация через ENV
# ----------------------------
# тут ключик для простой защиты эндпоинта. без особых претензий на безопасность,
# но лучше чем ничего. в проде прокидывать секретом, не хардкодить.
API_KEY = os.getenv("API_KEY", "changeme")  # ключ для заголовка X-API-Key

# строка подключения к раббиту. по умолчанию — локалхост, гость-гость.
# в кубере/докере перегружается через env.
RABBIT_URL = os.getenv(
    "RABBIT_URL",
    "amqp://guest:guest@localhost:5672/"
)
# очередь, куда будут падать задачи от апи к воркеру
TASK_QUEUE = os.getenv("TASK_QUEUE", "passport.tasks")  # входная очередь задач в ваш воркер
# сколько ждём синхронного ответа. если не успели — отдадим timeout.
RPC_TIMEOUT_SEC = int(os.getenv("RPC_TIMEOUT_SEC", "30"))
# квос для канала, чтобы не завалить консьюмера
PREFETCH_COUNT = int(os.getenv("PREFETCH_COUNT", "10"))

# Порт 80 (обычно за reverse-proxy). Можно поменять через ENV, если нужно.
# По факту тут выставлен 8085, т.к. 80 часто занят/нужны root-права. Смотри docker/proxy.
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8085"))

# ----------------------------
# Валидация входа
# ----------------------------

class ProcessRequest(BaseModel):
    # просто одна картинка в base64. без multipart — проще для брокера.
    image_b64: str = Field(..., description="Base64 изображения (JPEG/PNG)")
    # опциональные метки на будущее (ид клиента/заказа/и т.п.)
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Произвольные поля/настройки")
    # Можно добавить конкретные поля паспорта, если хотите валидировать заранее:
    # fields: Optional[Dict[str, Any]] = None

    @validator("image_b64")
    def validate_b64(cls, v: str) -> str:
        # на всякий случай проверяем что это действительно валидный base64,
        # иначе воркер упадёт на декоде. тут быстро, без записи на диск.
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("image_b64 is not valid base64")
        return v


class ProcessResponse(BaseModel):
    # request_id — это наш correlation_id. удобно искать в логах/очередях.
    request_id: str
    status: str  # "ok" | "timeout" | "error"
    result: Optional[Dict[str, Any]] = None
    detail: Optional[str] = None


# ----------------------------
# Приложение и соединение с RabbitMQ
# ----------------------------
app = FastAPI(title="Passport OCR API", version="1.0.0")

# CORS: максимально открыто — пускаем запросы с любых источников.
# если понадобится сузить, позже подставишь список доменов в allow_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # любой Origin
    allow_credentials=False,    # с "*" нельзя включать креды; если понадобится — перечисли домены явно
    allow_methods=["*"],        # все методы (GET/POST/…)
    allow_headers=["*"],        # любые заголовки (в т.ч. кастомные)
)

# держим одно соединение и канал — экономим на коннектах
_rabbit_connection: Optional[AbstractConnection] = None
_channel: Optional[AbstractChannel] = None
_task_queue: Optional[AbstractQueue] = None


async def _ensure_rabbit() -> None:
    """Ленивая/повторная инициализация соединения с RabbitMQ."""
    # жизнь показывает: коннект может отвалиться. robust-клиент поднимет обратно.
    # тут просто проверяем и поднимаем если нужно.
    global _rabbit_connection, _channel, _task_queue

    if _rabbit_connection and not _rabbit_connection.is_closed:
        return

    _rabbit_connection = await aio_pika.connect_robust(RABBIT_URL)
    _channel = await _rabbit_connection.channel()
    await _channel.set_qos(prefetch_count=PREFETCH_COUNT)

    # Гарантируем существование входной очереди задач (durable),
    # чтобы публикации не терялись если воркер стартанёт позже
    _task_queue = await _channel.declare_queue(
        TASK_QUEUE,
        durable=True,
        auto_delete=False
    )


@app.on_event("startup")
async def on_startup():
    # на старте сразу поднимем коннект, чтобы первый запрос не ждал
    await _ensure_rabbit()


@app.on_event("shutdown")
async def on_shutdown():
    # аккуратно закрываем соединение, чтоб не висели TCP-сессии
    global _rabbit_connection
    if _rabbit_connection and not _rabbit_connection.is_closed:
        await _rabbit_connection.close()


# ----------------------------
# Хелперы
# ----------------------------
def _check_api_key(x_api_key: Optional[str]) -> None:
    # сверка ключа. сейчас возвращает True (в обход), чтобы не мешало отладке.
    # в бою надо раскомментить проверку ниже.
    return True
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")


async def _rpc_publish_and_wait(payload: Dict[str, Any], timeout_sec: int) -> Optional[Dict[str, Any]]:
    """
    Публикуем сообщение в TASK_QUEUE и ждём ответ в уникальной reply-очереди.
    Возвращает dict ответа или None, если таймаут.

    Схема простая RPC:
      - генерим correlation_id (uuid4)
      - создаём эксклюзивную автоудаляемую очередь reply.<correlation_id>
      - кладём задачу с reply_to и correlation_id
      - слушаем reply.<correlation_id> до первого ответа или таймаута
    """
    assert _channel is not None

    correlation_id = str(uuid.uuid4())
    reply_queue_name = f"reply.{correlation_id}"

    # Эксклюзивная автоудаляемая очередь для ответа
    # (никто кроме нас её не слушает, удалится сама)
    reply_q = await _channel.declare_queue(
        reply_queue_name,
        durable=False,
        auto_delete=True,
        exclusive=True
    )

    # Добавим тех.метаданные. Воркеру это не обязательно, но удобно для трассировки.
    payload["_meta"] = {
        "request_id": correlation_id,
    }

    # Публикация задачи в основную очередь
    await _channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            correlation_id=correlation_id,   # важно: воркер должен вернуть этот же corr_id
            reply_to=reply_queue_name,       # отвечать сюда
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # переживёт ребут брокера
        ),
        routing_key=TASK_QUEUE,
    )

    # Ожидание первого ответа с совпадающим correlation_id
    # Тут без фанатизма: берём первое, не копим пачки.
    future: asyncio.Future = asyncio.get_event_loop().create_future()

    async def on_message(message: aio_pika.IncomingMessage):
        nonlocal future
        async with message.process():
            if message.correlation_id == correlation_id and not future.done():
                try:
                    body = json.loads(message.body.decode("utf-8"))
                except Exception:
                    # если воркер прислал не JSON — всё равно отдадим как есть
                    body = {"raw": message.body.decode("utf-8", errors="ignore")}
                future.set_result(body)

    # подписались на нашу временную очередь
    consume_tag = await reply_q.consume(on_message)

    try:
        # ждём до timeout_sec, потом возвращаем None (позже фронт увидит статус=timeout)
        return await asyncio.wait_for(future, timeout=timeout_sec)
    except asyncio.TimeoutError:
        return None
    finally:
        # Отписываемся от очереди и позволяем ей автоудалиться
        try:
            await reply_q.cancel(consume_tag)
        except Exception:
            # бывает уже удалена — ничего страшного
            pass


# ----------------------------
# Маршрут API
# ----------------------------
@app.post("/v1/process", response_model=ProcessResponse)
async def process_endpoint(
    req: ProcessRequest,
    request: Request,
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
):
    """
    Принимает JSON с base64-картинкой и опциональными метаданными.
    Паблишит задачу в RabbitMQ и ждёт ответ до RPC_TIMEOUT_SEC.

    По-хорошему сюда можно ещё добавить лимиты (размер картинки/частота запросов),
    но пока лёгкий вариант для MVP.
    """
    _check_api_key(x_api_key)
    await _ensure_rabbit()

    # Готовим нагрузку для воркера (вашего пайплайна)
    task_payload = {
        "type": "passport_ocr",      # для будущего роутинга, если появятся разные типы задач
        "image_b64": req.image_b64,  # воркер сам декодит
        "meta": req.meta or {},      # просто пробросим вглубь
        # сюда можно прокинуть доп флаги: какая модель, нужен ли дебаг, и пр.
    }

    result = await _rpc_publish_and_wait(task_payload, timeout_sec=RPC_TIMEOUT_SEC)
    request_id = task_payload.get("_meta", {}).get("request_id", "")

    if result is None:
        # таймаут — не ошибка сервера, а просто «не успели»
        # фронт/клиент могут по request_id потом подтянуть результат из БД, если вы туда пишете
        return JSONResponse(
            status_code=200,
            content=ProcessResponse(
                request_id=request_id,
                status="timeout",
                detail=f"No reply within {RPC_TIMEOUT_SEC}s"
            ).dict()
        )

    # Ожидаем, что воркер вернёт JSON (например, что собирает main.py в сервисном режиме)
    return JSONResponse(
        status_code=200,
        content=ProcessResponse(
            request_id=request_id,
            status="ok",
            result=result
        ).dict()
    )


# ----------------------------
# Точка входа (локальный запуск)
# ----------------------------
if __name__ == "__main__":
    # локально поднимаем uvicorn без autoreload (нам не нужен в проде)
    import uvicorn
    uvicorn.run("api_server:app", host=API_HOST, port=API_PORT, reload=False)

