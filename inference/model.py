"""
Model and input normalization used by the inference service.

В проекте джет представлен как набор конституентов:
- до 200 конституентов на джет,
- по 7 признаков на каждый после preprocessing,
- для FCNN это обычно сводится к flatten: [200, 7] -> [1400].

Здесь используется детерминированная mock student-модель:
- интерфейс совпадает с student FCNN,
- работает быстро (numpy),
- даёт воспроизводимый результат.

При появлении реальной student-модели ее просто можно подставить здесь,
не меняя остальной код и REST API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from .config import settings


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _infer_format(jets) -> Literal["flat", "constituents"]:
    """Пытаемся угадать формат входа по вложенности списков."""
    first = jets[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        return "constituents"
    return "flat"


def normalize_jets(
    jets,
    *,
    fmt: Literal["auto", "flat", "constituents"] = "auto",
    n_constituents: int = settings.n_constituents,
    n_features: int = settings.n_features,
) -> np.ndarray:
    """
    Приводит входной payload к 2D-массиву формы
    [batch, n_constituents * n_features].

    Правила простые:
    - если конституентов меньше, то дополняем нулями,
    - если больше, то обрезаем.

    Это позволяет тестировать сервис с маленькими примерами,
    не ломая контракт, который ожидает модель.
    """

    if fmt == "auto":
        fmt = _infer_format(jets)

    if fmt == "flat":
        x = np.asarray(jets, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Flat format expects [batch, features], got shape {x.shape}")

        expected = n_constituents * n_features
        if x.shape[1] != expected:
            raise ValueError(
                f"Flat jets must have length {expected} (= {n_constituents}*{n_features}); "
                f"got {x.shape[1]}"
            )
        return x

    # constituents format: [batch][constituent][feature]
    batch = len(jets)
    out = np.zeros((batch, n_constituents, n_features), dtype=np.float32)

    for i, jet in enumerate(jets):
        # базовая проверка структуры
        if not isinstance(jet, list):
            raise ValueError(f"Jet #{i} must be a list, got {type(jet)}")

        take = min(len(jet), n_constituents)
        for j in range(take):
            feat = jet[j]
            if not isinstance(feat, list):
                raise ValueError(f"Jet #{i}, constituent #{j} must be a list, got {type(feat)}")
            if len(feat) != n_features:
                raise ValueError(
                    f"Jet #{i}, constituent #{j}: "
                    f"expected {n_features} features, got {len(feat)}"
                )
            out[i, j, :] = np.asarray(feat, dtype=np.float32)

    return out.reshape(batch, n_constituents * n_features)


@dataclass(frozen=True)
class MockStudentFCNN:
    """
    Небольшая детерминированная FCNN.

    Ззадача имитировать поведение сжатой student-модели:
    - быстрый CPU-инференс,
    - стабильные выходы,
    - корректный I/O контракт.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @property
    def name(self) -> str:
        return "mock-student-fcnn-v0"

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: [batch, input_dim]
        h = _relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def predict_proba(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logits = self.forward(x)
        probs = _softmax(logits)
        return probs, logits


def load_model(
    *,
    input_dim: int = settings.n_constituents * settings.n_features,
    hidden_dim: int = settings.hidden_dim,
    n_classes: int = settings.n_classes,
    seed: int = settings.seed,
) -> MockStudentFCNN:
    """
    Создает mock-модель с фиксированной инициализацией.

    Фиксированный seed нужен для воспроизводимости,
    что удобно для отладки и нагрузочного тестирования.
    """
    rng = np.random.default_rng(seed)

    # Небольшая дисперсия весов
    w1 = rng.normal(0, 0.05, size=(input_dim, hidden_dim)).astype(np.float32)
    b1 = rng.normal(0, 0.02, size=(hidden_dim,)).astype(np.float32)
    w2 = rng.normal(0, 0.05, size=(hidden_dim, n_classes)).astype(np.float32)
    b2 = rng.normal(0, 0.02, size=(n_classes,)).astype(np.float32)

    return MockStudentFCNN(w1=w1, b1=b1, w2=w2, b2=b2)
