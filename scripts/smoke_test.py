"""
Небольшой smoke-тест для инференс-сервиса.

Как использовать:
1. В одном терминале запустить сервис:
       uvicorn inference.app:app --reload
2. В другом терминале выполнить:
       python scripts/smoke_test.py

Скрипт отправляет небольшой батч джетов в формате constituents
и печатает ответ сервиса. Нужен для быстрой проверки, что
эндпоинт /predict работает и возвращает осмысленный результат.
"""

from __future__ import annotations

import json
import random
import sys
from typing import List

import requests


def rand_constituent(n_features: int = 7) -> List[float]:
    """Генерирует один конституент с n_features признаками.

    Значения случайные, физического смысла не несут.
    """
    return [random.uniform(-1.0, 1.0) for _ in range(n_features)]


def main() -> None:
    url = "http://127.0.0.1:8000/predict"

    # Небольшой тестовый payload:
    # два джета, каждый с несколькими конституентами
    payload = {
        "format": "constituents",
        "return_logits": True,
        "jets": [
            [rand_constituent() for _ in range(3)],
            [rand_constituent() for _ in range(5)],
        ],
    }

    r = requests.post(url, json=payload, timeout=10)
    print("Status:", r.status_code)

    try:
        # Ограничиваем вывод, чтобы не засорять терминал
        print(json.dumps(r.json(), indent=2)[:2000])
    except Exception:
        print(r.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
