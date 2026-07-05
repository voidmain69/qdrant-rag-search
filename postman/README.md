# Postman collection

- `qdrant-product-search.postman_collection.json` — all endpoints (Health, Products, Imports, Search).
- `qdrant-product-search.postman_environment.json` — local environment (`baseUrl`, `apiKey`, `jobId`).

## Use

1. In Postman: **Import** → drop both files.
2. Select the **Qdrant Product Search — local** environment (top-right).
3. Set `apiKey` to one of the service's `API_KEYS` (default `change-me-secret-key`); adjust `baseUrl` if not on `localhost:8000`.
4. Run **Products → Upsert product** (or **Upsert batch**), then anything under **Search**.

Auth is collection-level: every `/api/v1/**` request sends `X-API-Key: {{apiKey}}`; `/health`, `/ready`, `/metrics` are open. **Start file import** captures the returned `job_id` into the `jobId` variable so **Get import job** works without editing the URL.

The whole collection can also run headless with [newman](https://github.com/postmanlabs/newman):

```bash
newman run postman/qdrant-product-search.postman_collection.json \
  -e postman/qdrant-product-search.postman_environment.json
```
