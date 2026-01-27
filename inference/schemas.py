"""
Pydantic-схемы для инференс-сервиса.

Сервис принимает данные о джетах в двух форматах:

1) constituents
   Наиболее близкий к физическому представлению.
   Один джет — это список конституентов, каждый конституент —
   список признаков фиксированной длины.
   Форма: [batch, n_constituents, n_features]

2) flat
   Упрощённый формат для FCNN-подобных моделей.
   Каждый джет - это плоский вектор длины n_constituents * n_features.

Поддержка обоих форматов позволяет:
- работать с семантически понятным вводом,
- при необходимости использовать компактный JSON и прямой FCNN-вход.

В формате constituents сервис автоматически делает padding / truncation
до ожидаемого числа конституентов.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# Допустимые формы входных данных:
#  - flat:         List[jet], jet = List[float]
#  - constituents: List[jet], jet = List[List[float]]
JetsPayload = Union[List[List[float]], List[List[List[float]]]]


class PredictRequest(BaseModel):
    # Батч джетов в одном из поддерживаемых форматов
    jets: JetsPayload = Field(..., description="Batch of jets in flat or constituent-level format")

    # Явно указываем формат или даем сервису определить его автоматически
    format: Literal["auto", "flat", "constituents"] = Field(
        "auto", description="How to interpret the `jets` field"
    )

    # Посмотреть логиты до softmax (например, для отладки)
    return_logits: bool = Field(False, description="Whether to include raw logits in the response")

    @field_validator("jets")
    @classmethod
    def non_empty(cls, v: JetsPayload) -> JetsPayload:
        # Не принимаем пустые запросы
        if not v:
            raise ValueError("`jets` must be a non-empty list")
        return v


class PredictResponse(BaseModel):
    # Вероятности классов для каждого джета
    probs: List[List[float]]

    # Логиты (опционально, если запрошены)
    logits: Optional[List[List[float]]] = None

    # Идентификатор используемой модели
    model: str

    # Размер обработанного батча
    batch_size: int


class MetadataResponse(BaseModel):
    # Ожидаемое число конституентов на джет
    expected_constituents: int

    # Число признаков на конституент
    expected_features: int

    # Поддерживаемые форматы входа
    supported_formats: List[str]

    # Название текущей модели
    model: str

    # Короткое текстовое описание сервиса
    notes: str


class HealthResponse(BaseModel):
    # health-check
    status: Literal["ok"]
