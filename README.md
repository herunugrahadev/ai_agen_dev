# OpenMausBot + 9Router Autonomous AI Agent Platform

Platform agen AI otonom yang menghubungkan **OpenMausBot** (orkestrasi multi-agen, *tool calling*, *state & memory*) dengan **9Router** (*unified OpenAI-compatible AI gateway*, *dynamic tier routing*, dan *auto-fallback* saat HTTP 429).

## Arsitektur & Alur Kerja

```text
[REST API / Webhook] ──▶ [Redis Task Queue] ──▶ [OpenMausBot Worker]
                                                        │
                     ┌──────────────────────────────────┼──────────────────────────────────┐
                     ▼                                  ▼                                  ▼
           [1. PlannerAgent]                  [2. ExecutorAgent]                 [3. ReviewerAgent]
          model: agent-planner               model: agent-executor              model: agent-reviewer
           (JSON Plan Output)             (web_scraper + database_writer)       (QA & Final Verdict)
                     │                                  │                                  │
                     └──────────────────────────────────┼──────────────────────────────────┘
                                                        ▼
                                            [9Router (:20128/v1)]
                                       (Auto-Fallback pada HTTP 429)
```

## Cara Menjalankan (Docker Compose)

1. **Salin konfigurasi environment:**
   ```bash
   cp .env.example .env
   ```
   Isi API key provider LLM (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`) pada file `.env`. OpenMausBot tidak pernah membaca key provider secara langsung; seluruh panggilan LLM diarahkan ke `http://9router:20128/v1`.

2. **Jalankan seluruh layanan:**
   ```bash
   docker compose up -d --build
   ```

3. **Verifikasi kesehatan layanan:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:20128/api/health
   ```

## Menguji Alur End-to-End

1. **Kirim tugas via REST API (asinkron via Redis Worker):**
   ```bash
   curl -X POST http://localhost:8000/api/v1/tasks \
     -H "Content-Type: application/json" \
     -d '{
       "objective": "Scrape halaman https://example.com dan simpan ringkasan serta judulnya ke database.",
       "target_url": "https://example.com"
     }'
   ```

2. **Atau kirim via Webhook Trigger:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/webhook/trigger \
     -H "Content-Type: application/json" \
     -d '{
       "event_type": "task.requested",
       "objective": "Ekstrak informasi utama dari https://example.com dan simpan ke PostgreSQL.",
       "target_url": "https://example.com"
     }'
   ```

3. **Cek hasil eksekusi & QA Review:**
   ```bash
   curl http://localhost:8000/api/v1/tasks/<TASK_ID>
   curl http://localhost:8000/api/v1/tasks/<TASK_ID>/records
   ```

## Menjalankan Unit & Integration Tests (Lokal)

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -v
```
