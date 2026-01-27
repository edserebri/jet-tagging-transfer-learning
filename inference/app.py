"""
Inference service for jet tagging (FastAPI).
"""
# ruff: noqa: E501

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .config import settings
from .model import load_model, normalize_jets
from .schemas import HealthResponse, MetadataResponse, PredictRequest, PredictResponse

# ----------------------------
# OpenAPI
# ----------------------------
tags_metadata = [
    {
        "name": "Inference",
        "description": (
            "Основной inference API. "
            "Используется для классификации джетов с помощью student-модели."
        ),
    },
    {
        "name": "Service",
        "description": "Технические эндпоинты: health-check и метаданные сервиса.",
    },
]

DESCRIPTION = """
**Inference service** для задачи **jet tagging** в физике высоких энергий.

Сервис предоставляет HTTP API для применения обученной **student-модели**
к новым данным о джетах и предназначен для демонстрации production-подхода
к использованию ML-моделей вне обучающих ноутбуков.

---

### Как это работает

1. Клиент (batch-пайплайн, trigger или аналитический сервис) отправляет
   батч джетов в формате JSON.
2. Сервис выполняет:
   - валидацию входных данных,
   - preprocessing (padding / truncation до фиксированного размера),
   - inference с помощью **student-модели**.
3. В ответ возвращаются вероятности классов для каждого джета.

Модель загружается один раз при старте сервиса и переиспользуется
для всех запросов, что обеспечивает быстрый CPU inference.

---

### Что здесь можно делать

- Выполнять **inference джет-теггинга** через эндпоинт `POST /predict`;
- Работать с двумя форматами входа:
  - `constituents` — близкий к физическому представлению,
  - `flat` — плоский вектор для FCNN-подобных моделей;
- Проверять состояние сервиса (`GET /health`);
- Получать информацию о текущей конфигурации и модели (`GET /metadata`).

---

### Зачем нужен этот сервис

- Отделить **обучение модели** от ее использования;
- Зафиксировать стабильный **API-контракт** для upstream-систем;
- Показать, как student-модель может быть использована
  в offline, online и near-real-time сценариях;
- Обеспечить возможность замены модели без изменения клиентского кода.

---

### Примечание

В текущей версии используется детерминированная **mock student-модель**.
Она повторяет интерфейс реальной модели и позволяет протестировать
инференс-контур независимо от этапа обучения.
"""


app = FastAPI(
    title="Jet Tagging Inference Service",
    version="0.2.0",
    description=DESCRIPTION,
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "docExpansion": "none",
        "displayRequestDuration": True,
        "defaultModelsExpandDepth": -1,
    },
)

_model = load_model()

# Пример для Swagger
PREDICT_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "constituents_minimal": {
        "summary": "Пример: формат constituents (короткий, padding делается автоматически)",
        "description": (
            "Если конституентов меньше 200, сервис сделает padding нулями. "
            "Если больше — truncation."
        ),
        "value": {
            "format": "constituents",
            "return_logits": False,
            "jets": [
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    [0.0, 0.1, 0.0, 0.2, 0.0, 0.3, 0.0],
                ],
                [[1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1]],
            ],
        },
    }
}


# ----------------------------
# DEMO UI на /
# ----------------------------
UI_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Jet Tagging Inference Service</title>
  <style>
    :root {
      --bg: #0b1020;
      --card: #121a33;
      --text: #e7e9ee;
      --muted: #aab1c5;
      --border: rgba(255,255,255,0.12);
      --ok: #35d07f;
      --bad: #ff5d5d;
      --btn: rgba(255,255,255,0.08);
      --btnHover: rgba(255,255,255,0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(1200px 600px at 10% 0%, rgba(120,140,255,0.25), transparent 55%),
                  radial-gradient(900px 500px at 90% 20%, rgba(53,208,127,0.18), transparent 55%),
                  var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
      line-height: 1.35;
    }
    a { color: var(--text); }
    .wrap { max-width: 1100px; margin: 28px auto; padding: 0 16px 40px; }
    .top {
      display: grid;
      gap: 12px;
      padding: 18px 18px;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
      border-radius: 16px;
    }
    .titleRow { display:flex; gap: 12px; align-items: center; justify-content: space-between; flex-wrap: wrap; }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0.2px; }
    .badges { display:flex; gap: 10px; align-items:center; flex-wrap: wrap; }
    .pill {
      display:inline-flex; gap: 8px; align-items:center;
      padding: 6px 10px; border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot { width: 9px; height: 9px; border-radius: 99px; background: var(--muted); }
    .dot.ok { background: var(--ok); }
    .dot.bad { background: var(--bad); }
    .subtitle { color: var(--muted); font-size: 14px; margin: 0; }
    .links { display:flex; gap: 12px; flex-wrap: wrap; }
    .links a {
      display:inline-flex; align-items:center; gap:8px;
      padding: 8px 10px; border-radius: 10px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      text-decoration: none;
    }
    .grid { display:grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px; }
    @media (min-width: 960px) { .grid { grid-template-columns: 1.1fr 0.9fr; } }
    .card {
      border: 1px solid var(--border);
      background: rgba(18,26,51,0.75);
      border-radius: 16px;
      padding: 16px;
      backdrop-filter: blur(6px);
    }
    h2 { margin: 0 0 10px; font-size: 16px; }
    .row { display:flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    button {
      cursor: pointer;
      border: 1px solid var(--border);
      background: var(--btn);
      color: var(--text);
      padding: 9px 12px;
      border-radius: 12px;
      font-size: 14px;
    }
    button:hover { background: var(--btnHover); }
    button.primary { background: rgba(53,208,127,0.18); border-color: rgba(53,208,127,0.35); }
    button.ghost { background: transparent; }
    textarea {
      width: 100%;
      min-height: 250px;
      margin-top: 10px;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.25);
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 13px;
      line-height: 1.35;
      outline: none;
    }
    .hint { margin-top: 10px; color: var(--muted); font-size: 13px; }
    pre.code {
      margin: 10px 0 0;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.28);
      overflow: auto;
      min-height: 230px;
      white-space: pre;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 13px;
      line-height: 1.35;
    }
    .status {
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      color: var(--muted);
      background: rgba(255,255,255,0.04);
      font-size: 13px;
    }
    .status.ok { border-color: rgba(53,208,127,0.35); }
    .status.bad { border-color: rgba(255,93,93,0.35); }
    .k { display:inline-block; padding: 2px 8px; border-radius: 8px; border: 1px solid var(--border); background: rgba(255,255,255,0.04); font-family: ui-monospace, monospace; font-size: 12px; color: var(--text); }
    .split { display:grid; grid-template-columns: 1fr; gap: 14px; }
    @media (min-width: 960px) { .split { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="titleRow">
        <h1>Jet Tagging Inference Service</h1>
        <div class="badges">
          <div class="pill"><span class="dot" id="healthDot"></span><span id="healthText">health: …</span></div>
          <div class="pill"><span>model:</span> <strong id="modelName">…</strong></div>
        </div>
      </div>

      <p class="subtitle">
        Демо-страница, чтобы за несколько секунд стало понятно: ввод → <span class="k">POST /predict</span> → вероятности классов.
      </p>

      <div class="links">
        <a href="/docs" target="_blank" rel="noreferrer">Swagger UI (/docs)</a>
        <a href="/redoc" target="_blank" rel="noreferrer">ReDoc (/redoc)</a>
        <a href="/metadata" target="_blank" rel="noreferrer">/metadata</a>
        <a href="/health" target="_blank" rel="noreferrer">/health</a>
      </div>

      <div class="hint" id="metaLine">Загружаю метаданные…</div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>1) Ввод (JSON)</h2>
        <div class="row">
          <button class="primary" onclick="loadExampleConstituents()">Загрузить пример constituents</button>
          <button onclick="loadExampleFlat()">Загрузить пример flat</button>
          <button class="ghost" onclick="clearAll()">Очистить</button>
        </div>

        <textarea id="payload" spellcheck="false"></textarea>

        <div class="hint">
          Поддерживаемые форматы: <span class="k">constituents</span> и <span class="k">flat</span>.
          Для <span class="k">constituents</span> сервис делает padding / truncation до фиксированного размера.
        </div>

        <div class="status" id="statusBox">
          Шаги: нажми <span class="k">Загрузить пример</span> → затем <span class="k">POST /predict</span> → смотри вывод справа.
        </div>

        <div class="row" style="margin-top: 10px;">
          <button class="primary" onclick="runPredict()">POST /predict</button>
          <button onclick="runMetadata()">GET /metadata</button>
          <button onclick="runHealth()">GET /health</button>
        </div>
      </div>

      <div class="card">
        <h2>2) Вывод</h2>
        <div class="split">
          <div>
            <div class="hint">Response (JSON)</div>
            <pre class="code" id="response"></pre>
          </div>
          <div>
            <div class="hint">cURL (для вставки в терминал)</div>
            <pre class="code" id="curl"></pre>
            <div class="row" style="margin-top: 10px;">
              <button class="ghost" onclick="copyCurl()">Copy cURL</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="hint" style="margin-top: 14px;">
      Примечание: сейчас используется детерминированная mock student-модель (для воспроизводимых демо и тестов).
    </div>
  </div>

<script>
  const payloadEl = document.getElementById('payload');
  const responseEl = document.getElementById('response');
  const curlEl = document.getElementById('curl');
  const statusBox = document.getElementById('statusBox');
  const metaLine = document.getElementById('metaLine');
  const modelName = document.getElementById('modelName');
  const healthText = document.getElementById('healthText');
  const healthDot = document.getElementById('healthDot');

  function pretty(obj) { return JSON.stringify(obj, null, 2); }

  function setStatus(msg, ok=null) {
    statusBox.textContent = msg;
    statusBox.classList.remove('ok', 'bad');
    if (ok === true) statusBox.classList.add('ok');
    if (ok === false) statusBox.classList.add('bad');
  }

  function escapeForSingleQuotes(s) {
    return s.replace(/'/g, "'\\''");
  }

  function updateCurl() {
    const body = payloadEl.value.trim() || "{}";
    const url = window.location.origin + "/predict";
    curlEl.textContent =
`curl -X POST ${url} \\
  -H "Content-Type: application/json" \\
  -d '${escapeForSingleQuotes(body)}'`;
  }

  async function fetchJson(url, options={}) {
    const t0 = performance.now();
    const res = await fetch(url, Object.assign({
      headers: { "Content-Type": "application/json" }
    }, options));
    const t1 = performance.now();
    const text = await res.text();
    let data = text;
    try { data = JSON.parse(text); } catch (e) {}
    return { res, data, ms: Math.round((t1 - t0) * 10) / 10 };
  }

  function loadExampleConstituents() {
    const ex = {
      "format": "constituents",
      "return_logits": false,
      "jets": [
        [
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
          [0.0, 0.1, 0.0, 0.2, 0.0, 0.3, 0.0]
        ],
        [
          [1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1]
        ]
      ]
    };
    payloadEl.value = pretty(ex);
    responseEl.textContent = "";
    updateCurl();
    setStatus("Загружен пример constituents. Нажми «POST /predict».");
  }

  function loadExampleFlat() {
    const n = 200 * 7; // 1400
    const jet0 = new Array(n).fill(0);
    jet0[0] = 0.1; jet0[1] = 0.2; jet0[2] = 0.3; jet0[6] = 0.7;
    const jet1 = new Array(n).fill(0);
    jet1[0] = 1.0; jet1[4] = 0.1; jet1[6] = 0.1;

    const ex = { "format": "flat", "return_logits": false, "jets": [jet0, jet1] };
    payloadEl.value = pretty(ex);
    responseEl.textContent = "";
    updateCurl();
    setStatus("Загружен пример flat (длина 1400). Нажми «POST /predict».");
  }

  function clearAll() {
    payloadEl.value = "";
    responseEl.textContent = "";
    curlEl.textContent = "";
    setStatus("Очищено. Нажми «Загрузить пример», чтобы начать.");
  }

  async function runPredict() {
    let payload;
    try { payload = JSON.parse(payloadEl.value); }
    catch (e) { setStatus("Ошибка JSON: " + e.message, false); return; }

    setStatus("Отправляю POST /predict …");
    const { res, data, ms } = await fetchJson("/predict", {
      method: "POST",
      body: JSON.stringify(payload)
    });

    responseEl.textContent = (typeof data === "string") ? data : pretty(data);
    setStatus(`${res.ok ? "OK" : "Ошибка"}: HTTP ${res.status} · ${ms} ms`, res.ok);
  }

  async function runMetadata() {
    setStatus("Запрашиваю GET /metadata …");
    const { res, data, ms } = await fetchJson("/metadata", { method: "GET" });
    responseEl.textContent = (typeof data === "string") ? data : pretty(data);
    setStatus(`${res.ok ? "OK" : "Ошибка"}: HTTP ${res.status} · ${ms} ms`, res.ok);
    if (res.ok && data && data.model) modelName.textContent = data.model;
  }

  async function runHealth() {
    const { res, data } = await fetchJson("/health", { method: "GET" });
    const ok = res.ok && data && data.status === "ok";
    healthText.textContent = ok ? "health: ok" : "health: error";
    healthDot.classList.remove("ok", "bad");
    healthDot.classList.add(ok ? "ok" : "bad");
    return ok;
  }

  async function init() {
    loadExampleConstituents();
    await runHealth();

    const { res, data } = await fetchJson("/metadata", { method: "GET" });
    if (res.ok && data) {
      modelName.textContent = data.model || "unknown";
      metaLine.textContent =
        `Ожидаемый shape: [batch, ${data.expected_constituents}, ${data.expected_features}] (constituents) или [batch, ${data.expected_constituents * data.expected_features}] (flat).`;
    } else {
      metaLine.textContent = "Не удалось загрузить /metadata.";
    }
  }

  async function copyCurl() {
    try {
      await navigator.clipboard.writeText(curlEl.textContent);
      setStatus("cURL скопирован в буфер обмена ✅", true);
    } catch (e) {
      setStatus("Не получилось скопировать cURL (браузер запретил clipboard).", false);
    }
  }

  payloadEl.addEventListener("input", updateCurl);
  window.addEventListener("load", init);
</script>
</body>
</html>
""".strip()


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def ui() -> HTMLResponse:
    """Простая web-страница для понятного демо сервиса."""
    return HTMLResponse(UI_HTML)


# ----------------------------
# API эндпоинты
# ----------------------------
@app.get("/health", response_model=HealthResponse, tags=["Service"], summary="Health-check")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get(
    "/metadata", response_model=MetadataResponse, tags=["Service"], summary="Метаданные сервиса"
)
def metadata() -> MetadataResponse:
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


@app.post(
    "/predict",
    response_model=PredictResponse,
    response_model_exclude_none=True,
    tags=["Inference"],
    summary="Inference: классификация батча джетов",
    description=(
        "Принимает батч джетов в формате `constituents` или `flat` и возвращает "
        "вероятности классов (и логиты — опционально)."
    ),
)
def predict(req: PredictRequest = Body(..., examples=PREDICT_EXAMPLES)) -> PredictResponse:
    # защита от слишком больших батчей
    if len(req.jets) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large: {len(req.jets)} > max_batch_size={settings.max_batch_size}",
        )

    t0 = time.perf_counter()

    try:
        x = normalize_jets(
            req.jets,
            fmt=req.format,
            n_constituents=settings.n_constituents,
            n_features=settings.n_features,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    probs, logits = _model.predict_proba(x)
    _ = time.perf_counter() - t0

    return PredictResponse(
        probs=probs.tolist(),
        logits=logits.tolist() if req.return_logits else None,
        model=_model.name,
        batch_size=int(x.shape[0]),
    )
