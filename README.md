# Qdrant Product Search

Production-сервіс пошуку товарів на Qdrant: семантика + BM25 + точний і fuzzy-пошук за кодами.

Один пошуковий ендпоінт розумно обробляє всі типи запитів:

| Запит | Гілка | Як працює |
|---|---|---|
| `бездротовий пилосос для дому` | `hybrid` | dense (multilingual-e5-large) + sparse (BM25) з серверним RRF-фьюжном Qdrant Query API |
| `акумулятор 18V 5Ah Li-Ion` | `hybrid` | одиниці виміру (18V, 5Ah) розпізнаються і НЕ вважаються кодами |
| `GSB-13-RE` (артикул) | `exact` | нормалізація (регістр, роздільники, кирилично-латинські гомогліфи) + O(1) lookup |
| `НВ-1234` (кирилицею) | `exact` | Н→H, В→B — код, набраний в українській розкладці, знаходиться |
| `BQSCH-O6` (OCR-плутанина) | `exact_normalized` | лосі OCR-скелет: O→0, I→1, S→5, B→8 … |
| `4006381333932` (зіпсований EAN) | `ean_corrected` | checksum-aware генерація кандидатів: одна невірна цифра / транспозиція сусідніх |
| `GSB-13-RF` (одрук) | `fuzzy` | RapidFuzz: OSA + Jaro-Winkler по in-memory корпусу всіх кодів |
| `дриль GSB13RE з кейсом` | `mixed` | точні код-хіти піняться першими, решта — гібридний пошук |

Кожен hit містить `match: {branch, matched_field, code_score}` — видно, чому товар у видачі.

## Швидкий старт

```bash
cp .env.example .env          # задайте API_KEYS
docker compose up -d --build  # перший старт завантажує ~2.2 ГБ моделей у volume
# чекаємо healthy:  curl http://localhost:8000/ready

# демо: інжест 38 зразків + усі типи запитів
uv run python scripts/smoke_search.py
```

Qdrant dashboard: http://localhost:6333/dashboard

## API

Автентифікація: заголовок `X-API-Key` (список ключів у `API_KEYS`, comma-separated; порожній = вимкнено, тільки для dev). `/health`, `/ready` — відкриті.

| Маршрут | Призначення |
|---|---|
| `POST /api/v1/products` | upsert одного товару |
| `POST /api/v1/products:batch` | upsert до 1000 товарів |
| `PUT /api/v1/products/{external_id}` | повна заміна |
| `DELETE /api/v1/products/{external_id}` | видалення |
| `POST /api/v1/imports` | файловий імпорт CSV/JSON/XLSX (multipart; опційно `column_mapping`) |
| `GET /api/v1/imports/{job_id}` | прогрес імпорту |
| `POST /api/v1/search` | пошук |
| `GET /health`, `GET /ready` | liveness / readiness |

OpenAPI/Swagger: http://localhost:8000/docs

### Приклад: інжест

```bash
curl -X POST http://localhost:8000/api/v1/products \
  -H "X-API-Key: change-me-secret-key" -H "Content-Type: application/json" \
  -d '{
    "external_id": "tool-001",
    "name": "Дриль ударний Bosch GSB 13 RE",
    "brand": "Bosch",
    "category": "Електроінструмент",
    "article": "GSB-13-RE",
    "product_code": "060114E600",
    "ean13": "4006381333931",
    "attributes": {"Потужність": "600 Вт", "Патрон": "ШЗП 13 мм"},
    "price": 3299.0
  }'
```

### Приклад: пошук

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: change-me-secret-key" -H "Content-Type: application/json" \
  -d '{
    "query": "ударний дриль 600 Вт",
    "limit": 10,
    "rerank": false,
    "filters": {"brand": "Bosch", "price_max": 5000, "in_stock": true}
  }'
```

Відповідь:

```json
{
  "query_kind": "text",
  "took_ms": 42.1,
  "total": 7,
  "items": [
    {
      "product": {"external_id": "tool-001", "name": "Дриль ударний Bosch GSB 13 RE", "...": "..."},
      "score": 0.87,
      "match": {"branch": "hybrid", "matched_field": null, "code_score": null, "reranked": false}
    }
  ]
}
```

## Архітектура

```
запит ──> classify()                        інжест ──> нормалізація ──> ембединги ──> Qdrant
   │       │                                              │                              │
   │       ├─ CODE_ONLY ─> CodeIndex (in-memory)          └─> CodeIndex.add (інкрементно)┘
   │       │    exact → skeleton → EAN-fix → RapidFuzz
   │       ├─ TEXT ──────> Qdrant query_points:
   │       │    prefetch[dense, sparse] + RRF fusion (+опційний cross-encoder rerank)
   │       └─ MIXED ─────> обидві гілки, exact-хіти піняться першими
   └──────> SearchResponse з поясненням match.branch для кожного hit
```

- **Dense**: `intfloat/multilingual-e5-large` (FastEmbed/ONNX, CPU, 1024-dim, uk/ru/en).
- **Sparse**: `Qdrant/bm25` (стемер russian; коди індексуються raw + нормалізовано).
- **Фьюжн**: нативний RRF Qdrant (`query_points` + `prefetch`) — на боці БД.
- **Коди**: dense-вектор коди не бачить (не шумлять у семантиці); sparse і CodeIndex — бачать.
- **CodeIndex**: in-memory (≤100k товарів ×3 поля — десятки мс на повний fuzzy-прохід RapidFuzz);
  відбудовується на старті payload-only scroll'ом, оновлюється інкрементно. Кожна репліка API
  тримає власну копію — прийнятний компроміс для цього масштабу; JobStore імпорту теж in-memory
  (single-instance).
- **Версіонування схеми**: колекція `service_meta` зберігає модель/розмірність; при зміні конфігу
  сервіс відмовляється стартувати і просить реіндексацію (видалити колекції та переінжестити).

## Reranking (опційно)

```bash
uv sync --group rerank            # sentence-transformers + torch (CPU)
RERANK_ENABLED=true               # у .env
# у запиті: {"query": "...", "rerank": true}
```

Cross-encoder `BAAI/bge-reranker-v2-m3` вантажиться ліниво при першому запиті. Дає найкращу
якість на складних фразах (+50–200 мс). У FastEmbed цієї моделі немає, тому окрема група залежностей.

## Розробка

```bash
uv sync --group dev
uv run pytest -q                  # unit-тести (Qdrant/моделі не потрібні)
uv run pytest -m integration -q   # e2e проти піднятого compose
uv run ruff check app tests scripts
```

Файловий імпорт: колонки мапляться на поля `ProductIn` (aliases: `sku`/`артикул`→article,
`ean`/`штрихкод`→ean13, `назва`→name …), невідомі колонки потрапляють у `attributes`.
Явний мапінг: form-поле `column_mapping` = `{"Колонка з файла": "article"}`.

### Відомі особливості

- Перший інжест великого каталогу повільний: e5-large на CPU ≈ 10–30 товарів/с (100k ≈ 1–3 год,
  одноразово). Прогрів моделей у volume: `uv run python scripts/download_models.py`.
- Український стемінг у BM25 відсутній (використовується russian) — компенсується dense-моделлю
  через RRF.
- Windows-хост для локального запуску embedding-стека нестабільний (onnxruntime DLL) — цільовий
  рантайм: Docker. Unit-тести від ONNX не залежать.
