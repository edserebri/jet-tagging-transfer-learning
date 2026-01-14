"""
Minimal FastAPI app for inference.

- есть HTTP-ручка для получения предсказаний
- есть health-check
- есть валидация входа, чтобы не принимать что угодно

Сейчас используется детерминированная mock student-модель.
В будущем, при необходимости, ее можно легко заменить на реальную модель,
не меняя API и внешний контракт.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException

from .config import settings
from .model import load_model, normalize_jets
from .schemas import HealthResponse, MetadataResponse, PredictRequest, PredictResponse

app = FastAPI(title="Jet Tagging Inference", version="0.1.0")

# Держим модель в памяти на уровне приложения
# На данном этапе этого более чем достаточно
_model = load_model()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Простой health-check для мониторинга и контейнеров."""
    return HealthResponse(status="ok")


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Метаданные сервиса: что ожидаем на вход и что сейчас используем."""
    return MetadataResponse(
        expected_constituents=settings.n_constituents,
        expected_features=settings.n_features,
        supported_formats=["constituents", "flat"],
        model=_model.name,
        notes=(
            "Minimal inference service for jet tagging. "
            "Uses a mock student FCNN for fast CPU inference. "
            "Constituent-level inputs are padded or truncated to a fixed size."
        ),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Инференс для батча джетов."""

    # Не даем клиенту прислать слишком большой батч
    if len(req.jets) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large: {len(req.jets)} > max_batch_size={settings.max_batch_size}",
        )

    t0 = time.perf_counter()

    try:
        # Приводим вход к формату, который ожидает модель
        x = normalize_jets(
            req.jets,
            fmt=req.format,
            n_constituents=settings.n_constituents,
            n_features=settings.n_features,
        )
    except ValueError as e:
        # Ошибки формата превращаем в понятный HTTP-ответ
        raise HTTPException(status_code=422, detail=str(e))

    probs, logits = _model.predict_proba(x)

    # Пока никуда не пишем, но тайминг полезен для будущего логирования
    _ = time.perf_counter() - t0

    return PredictResponse(
        probs=probs.tolist(),
        logits=logits.tolist() if req.return_logits else None,
        model=_model.name,
        batch_size=int(x.shape[0]),
    )
