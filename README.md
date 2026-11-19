# Synthetic retail-policy FAQ API

1. **OCR reads the PDF.** Explicit ingestion uses hosted Mistral OCR 4.1 on an approved synthetic scanned PDF; reviewed extraction fixtures work without hosted access.
2. **Chroma finds relevant passages.** Local Ollama embeds pages and questions; local Docker Chroma returns up to three passages.
3. **Mistral answers with a page citation.** Local Mistral selects an exact supporting quote. Python validates it and returns the stored document title and page as structured citations.

**Milestone 1 is document-backed FAQ over HTTP.** The demo entry point is `POST /v1/faq`. Browser UI and banking actions are deferred. It answers public questions from synthetic documents only. Riverside is fictional; its Saturday schedule is on page 1 of the Synthetic Riverside Branch FAQ, not verified HDFC information. There are no balance/card actions or live branch-directory handlers.

See the short [demo and document-onboarding runbook](docs/DEMO_RUNBOOK.md) for startup, explicit OCR/ingestion, approval metadata, one matching regression case and API examples.

## Set up and start

For EKS, AKS or on-prem Kubernetes, use the [minimal container/deployment runbook](deploy/k8s/README.md). It includes a Dockerfile, Deployment/Service, health probes, autoscaling and a deploy script with private-service configuration.

Requires Python 3.14, `uv`, installed Docker with a running daemon, and local Ollama. Run from the repository root:

```sh
uv sync --locked
# In another terminal if Ollama is not running:
ollama serve
# Back in the original terminal:
ollama pull nomic-embed-text:latest
ollama pull ministral-3:14b
docker compose up -d
# Explicitly index the reviewed synthetic extraction fixtures (15 sentence-level chunks):
uv run hdfc-mistral-usecase ingest
```

Start the app with one command:

```sh
uv run uvicorn hdfc_mistral_mvp.api:app --host 127.0.0.1 --port 8020
```

OpenAPI and interactive requests: <http://127.0.0.1:8020/docs>.

Answer found — HTTP 200:

```sh
curl -sS -X POST http://127.0.0.1:8020/v1/faq \
  -H 'Content-Type: application/json' \
  -d '{"question":"Is Riverside open on Saturdays?"}'
```

Example response (exact supporting quote and chunk ID may vary with reviewed extraction):

```json
{
  "status": "answered",
  "answer": "Riverside is open on Saturdays from 10 AM to 2 PM under this synthetic demo schedule.",
  "citations": [
    {"title": "Synthetic Riverside Branch FAQ", "page": 1, "chunk_id": "riverside-p1-s0-..."}
  ]
}
```

No supporting passage — HTTP 200:

```sh
curl -sS -X POST http://127.0.0.1:8020/v1/faq \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the home loan interest rate?"}'
```

```json
{"status":"not_found","message":"I couldn’t find that in the approved documents.","citations":[]}
```

If Chroma or Mistral is unavailable, the app returns **HTTP 503** with `Retry-After: 5`:

```json
{"status":"unavailable","message":"FAQ is temporarily unavailable. Please try again shortly."}
```

Invalid, empty, oversized or sensitive questions return a safe HTTP 422 without echoing submitted content. Questions are screened before connecting to dependencies or sending embeddings/generation requests. The query endpoint only opens the existing collection and reads passages; it never creates/indexes documents, runs OCR or reads `mistral_keys.env`. Missing/uninitialized collections return HTTP 503; run explicit ingestion first.

## Explicit hosted OCR ingestion

The hosted adapter alone reads the ignored repo-root `mistral_keys.env`, without loading it into process environment. Create/edit that file privately:

```dotenv
MISTRAL_API_KEY=your-key
```

To run OCR 4.1 on the approved scanned Riverside PDF and then index the reviewed result:

```sh
uv run hdfc-mistral-usecase ocr --document riverside --allow-hosted-ocr
# Review artifacts/riverside.json, then explicitly ingest:
uv run hdfc-mistral-usecase ingest --extraction artifacts/riverside.json
```

Only PDFs matching `demo/approved.json` SHA-256 hashes may be sent to `https://api.mistral.ai/v1/ocr`, using exactly `mistral-ocr-4-1`. Missing files/keys or authentication errors produce safe errors. There is no implicit hosted call or model fallback. Outputs under `artifacts/` are ignored. The latest explicit OCR closeout attempt returned HTTP 429 with no numeric Retry-After (earlier attempts returned HTTP 401), so successful hosted extraction remains unverified; the HTTP demo uses reviewed synthetic transcription fixtures rather than claiming hosted OCR output.

For a separate approval list, `ocr` and `ingest --extraction` also accept `--manifest PATH`; default approval stays `demo/approved.json`. This allows review/testing a new document without editing the demo corpus or API code. See the runbook for the JSON schemas and regression-question format.

Four image-only PDFs have headers, graphics, text blocks and footers; statements has two pages, complaints, accessibility and Riverside have one each. `uv run python scripts/make_demo_documents.py --document riverside` rebuilds just the Riverside PDF/fixture/approval hash. Omitting `--document` rebuilds all synthetic documents and hashes.

## Tests and checks

```sh
# Entirely offline: HTTP fixture tests and actual-SDK wire-format tests:
uv run pytest -q
# Real HTTP RAG + isolated real dependency failure (Chroma/Ollama required):
uv run python scripts/check_faq_http.py
# Add a temporary approved document through the real CLI/Chroma/API path:
uv run python scripts/check_document_onboarding.py
# Separate opt-in Docker check (restarts only this demo's Chroma service):
uv run python scripts/check_docker.py
```

The [reviewable scenario table](tests/FAQ_SCENARIOS.md) and [JSON expectations](tests/faq_scenarios.json) contain 26 cases, kept outside the indexed corpus: direct questions/paraphrases across all four documents, statement page 2, insufficient/unsupported/ambiguous questions, screening, outages and an isolated document update.

`check_faq_http.py` starts temporary FastAPI processes and checks actual local RAG answers against reviewed facts and the cited stored text. It uses a temporary collection for the revised statement and reserved non-listening ports for outages. It cleans up its processes/collection and verifies that the demo collection and approved files are unchanged. No hosted OCR, expected-answer ingestion, or mock inference occurs in this live check. Run explicit fixture ingestion once before it. The JSON report defaults to `artifacts/faq-scenarios-live.json`; use `--report PATH` to save elsewhere or `--case CASE_ID` for diagnosis.

**Milestone 1 API-only RAG is closed:** latest closeout has 77 offline tests passing; all 26 live scenarios and the corpus-integrity check pass (27/27). Initial relevance/ambiguity failures and an older-statement paraphrase regression were fixed and are recorded in the plan and [final report](docs/evidence/scenarios-final.json). Offline tests use explicit model/retrieval doubles and do not establish live model accuracy. The live checker requires the expected fact in the answer and its cited source, not just a valid citation ID.

Observed timings are HTTP smoke samples, not latency SLOs: regular RAG 0.326–1.287 s, isolated outages about 8 s. Hosted OCR remains separately unverified (last HTTP 429; earlier HTTP 401); these scenarios do not load the key or call OCR. Browser UI, balance and card actions remain deferred.

The retained CLI `ocr`/`ingest` commands are ingestion tools. `uv run hdfc-mistral-usecase demo --offline` remains a labelled simulated troubleshooting check with no services or key; use the HTTP requests above for the demo.

## Configuration and limits

The supervisor uses the standard **OpenAI-compatible Chat Completions protocol** through the Mistral Agents SDK, so its code does not depend on an Ollama-specific API. Set `SUPPORT_INFERENCE_URL` to a local server origin (without `/v1`) and `SUPPORT_AGENT_MODEL` to that server's model ID. For example, to use a compatible server on port 8000:

```sh
SUPPORT_INFERENCE_URL=http://127.0.0.1:8000 \
SUPPORT_AGENT_MODEL=your-served-model \
OLLAMA_CHAT_MODEL=ministral-3:14b \
uv run uvicorn hdfc_mistral_mvp.api:app --host 127.0.0.1 --port 8021
```

This changes supervisor inference; the example keeps FAQ generation/embeddings on the existing local server. `INFERENCE_URL` changes the shared inference origin, while the legacy `OLLAMA_URL` remains a fallback. A shared server must support both Chat Completions and the configured embedding model. Servers must support streaming with token usage (`stream_options.include_usage`) and `max_tokens`. Ollama is live-tested; vLLM, llama.cpp, and other compatible servers still require their own smoke test. This demo permits loopback HTTP endpoints with a placeholder credential; authenticated/hosted inference is not configured. OCR credentials remain ingestion-only.

`NativeChatCompletionConfig` reuses the pinned Mistral SDK Chat Completions adapter with an explicit 400-token ceiling per native generation. The older Phase 1 classifier configuration remains only for regression diagnostics. Both use the same standard protocol without an additional AI SDK.

`CHROMA_HOST` defaults to `127.0.0.1`, `CHROMA_PORT` to `8015` (also used by Compose), `OLLAMA_URL` to `http://127.0.0.1:11434` (an origin without `/v1`), and `OLLAMA_EMBED_MODEL` to `nomic-embed-text:latest`. `SUPPORT_AGENT_MODEL` defaults to `ministral-3:14b`; FAQ answers use the same model by default and can be overridden with `OLLAMA_CHAT_MODEL`. Chroma is pinned and bound only to loopback, with data in the named `faq_chroma` volume. Re-ingestion is idempotent. The same embedding model must be used for indexing and questions; a mismatched model is rejected.

The direct `/v1/faq` path makes one query-embedding call and at most one generation call, with retries disabled. Answers must quote retrieved text exactly; citations come from stored metadata. Exact quotes prove provenance, not semantic relevance. Keyword/number screening is a small demo guardrail, not full PII detection. No chat history, raw diagnostic bodies or credentials are logged. A 60-second local dependency timeout allows cold starts; production latency/load targets remain pending.

`docker compose down` stops Chroma while retaining data. `docker compose down -v` explicitly deletes the demo volume; use only for an intended reset.

See [Implementation_Plan.md](Implementation_Plan.md) and the [verification record](docs/evidence/faq-milestone.md) for completed checks and pending milestones.

## Runtime configuration

All application environment settings are listed in the version-controlled [.env.example](.env.example). For a new checkout, copy it to `.env` and fill in the required database, Studio and synthetic-auth/webhook values locally. The existing local `.env` already retains its database configuration and has the missing template defaults added. Actual `.env` files are ignored by Git; only templates with empty secret values belong in version control. Keep the OCR key in the separate `mistral_keys.env` file, using [mistral_keys.env.example](mistral_keys.env.example) as its template.

```sh
uv run --env-file .env uvicorn hdfc_mistral_mvp.api:app --host 127.0.0.1 --port 8021
```

`uv --env-file` loads the chosen file into the process environment before startup, so existing `os.getenv()` calls read those values. API, worker and demo commands can use the same runtime file. Set `INFERENCE_URL` for direct FAQ/embeddings and `SUPPORT_INFERENCE_URL` for agents; legacy `OLLAMA_URL` is the fallback when explicit origins are absent. Keep origins without `/v1` and choose matching local model tags. Explicit variables in the template take precedence over fallback aliases.

All model HTTP operations now use the official `mistralai` client: agent/safety/supervisor/child completions use `Mistral.chat.stream_async`, FAQ completion uses `chat.complete`, embeddings use `embeddings.create`, and explicit hosted OCR uses `ocr.process`. The Agents SDK still manages native orchestration and its incremental tool-call parser. Its transport adapter delegates outbound requests to `mistralai`; the `OpenAICompletionProvider` base is a wire-format helper within the official Mistral Agents distribution, and no OpenAI client SDK is used. The small HTTP compatibility shim adds `stream_options.include_usage`, which the pinned `mistralai==3.7.0` method does not expose as an argument. SDK retries stay disabled and the existing request budgets/cleanup remain enforced.

## Milestone 2 Phase 2: native specialist demo

`POST /v1/customer-support` submits a Studio workflow that runs a Mistral Agents SDK supervisor and **native named subagents**. The model chooses one specialist; Python checks the choice, supplies the original screened question and validates the specialist's tool result. All specialists run asynchronously.

Use the short [code walkthrough](docs/SUPPORT_WALKTHROUGH.md) when presenting the demo.

For business and technology stakeholders, read the [architecture summary](ARCHITECTURE_SUMMARY.md): customer flow, Mistral platform capabilities, safety, human approval and deployment choices.

### Safety before routing

The flow is now **Python regex scan → redacted question → Studio worker → SafetyGuardrail agent → supervisor → one specialist**. `safety.py:scan_banking_input` is the mandatory Python scan tool. It runs before any model session or Studio submission, and before storing an authenticated support request. It masks detected PAN/Aadhaar, account/card/customer IDs, phone/email/UPI details, labelled credentials/OTP/PIN/CVV, balances, receipt references and labelled personal details with `[REDACTED_…]` placeholders. It keeps no map back to the original values. For example, `PAN ABCDE1234F. Are electronic statements free?` becomes `PAN [REDACTED_PAN]. Are electronic statements free?`.

`sdk.py:build_safety_guardrail` defines a separate tool-free Agents SDK node using the existing local model. It semantically checks for adversarial instructions, inappropriate content, unrelated requests and urgent banking incidents. For example, “Someone else is spending with my card” can return `urgent_support`: Python returns its configured support link before any supervisor, specialist, embedding or action. Python validates the typed verdict; only `allow` starts the supervisor. Urgency shares the existing safety call within the six-generation/twenty-second limit. The scan runs directly in Python before raw input reaches a model session. Regex-detected urgency and `GET /v1/support/urgent` remain available without inference; semantic-only detection depends on the model and Studio workflow. Neither urgent path calls an external bank API or blocks a card.

Obvious injection and input with no question left after redaction return HTTP 422 before SSE/Studio starts. Semantic rejection returns JSON HTTP 422 with `status: "rejected"` and a fixed safe message; an open SSE stream ends with a validated `final` containing that rejection. Existing authenticated requests keep their 202/polling flow: the workflow stores a terminal rejection without preparing an action. Scanner/model errors fail closed with 503 or an SSE `error`. Urgent lost-card/fraud support remains reachable without inference. Direct `POST /v1/faq` keeps its existing strict screening and JSON contract.

```sh
# Offline guardrail cases and pinned-SDK protocol fixtures:
uv run pytest tests/test_safety.py -q
# Explicit local Ollama + read-only Chroma smoke; no Studio or OCR credential:
uv run python scripts/check_safety_agent.py --live
```

Regex coverage is conservative and finite: unlabelled names/addresses, novel encodings and unsupported formats can evade it, and the semantic model can misclassify requests. This remains a synthetic demo; comprehensive production PII detection and adversarial evaluation remain open gates.

| Specialist | Its single tool | Demo behavior |
| --- | --- | --- |
| FAQ | `faq_lookup` | Existing read-only RAG, approved quote and page citation |
| CardsGuidance | `card_next_step` | Secure next step for temporary deactivation |
| AccountGuidance | `account_next_step` | Secure balance access or statement download |
| PaymentsGuidance | `payment_next_step` | Secure beneficiary change or money transfer |
| ProfileGuidance | `profile_next_step` | Secure contact update or card limit change |
| HighStakeTransactions | `transaction_next_step` | Typed semantic preparation; Python threshold triggers workflow review |

The original banking guidance specialists return secure next steps. HighStakeTransactions can request synthetic preparation through a validated Python adapter; its workflow triggers human review from the prepared action policy. Authentication, account data, verification, confirmation, approval and execution remain in the bank service. Urgent loss/fraud/permanent-block cases bypass the model. Ambiguous requests receive clarification.

Flow: scan/redact → urgent check → Studio workflow → local worker activity → SafetyGuardrail → SDK supervisor → one named child → one typed async tool → validated JSON/SSE result. `sdk.py:build_support_supervisor` declares the graph and `support_models.py` defines its Pydantic schemas. A short-lived in-memory request guard enforces call limits and carries the validated result between SDK callbacks; it is deleted when the worker activity finishes. It is not a session transcript or checkpoint. The graph uses `agents.Subagent` and native `agents.spawn`, not legacy `handoffs`/`RunContext` syntax. The SDK requires its TypeScript tool runtime for native spawning; hooks restrict it to one allowlisted child at depth one. Children have no child catalog or processor tools.

The user-approved Phase 2 demo limit is **six total generations, one embedding, twenty seconds**, including child turns, FAQ answering and supervisor continuations. Transport opening retries are disabled; internal SDK broken-stream retries consume the same six-generation budget at the transport boundary. The API returns the validated tool result and cancels the remaining session tree instead of generating a summary. The 5-second p95 completed-FAQ target remains unverified. Cancellation closes async model transport; a blocking FAQ request may finish up to its remaining transport deadline. Stopping Ollama compute after disconnect remains unverified.

Start the API with a dedicated `MISTRAL_STUDIO_API_KEY` in its environment:

```sh
# Set MISTRAL_STUDIO_API_KEY in the environment, then:
uv run uvicorn hdfc_mistral_mvp.api:app --host 127.0.0.1 --port 8021
```

JSON is the default; only `?stream=true` enables SSE:

```sh
curl -sS http://127.0.0.1:8021/v1/customer-support \
  -H 'Content-Type: application/json' -d '{"question":"Is Riverside open on Saturdays?"}'
curl -N 'http://127.0.0.1:8021/v1/customer-support?stream=true' \
  -H 'Content-Type: application/json' -d '{"question":"Is Riverside open on Saturdays?"}'
curl -sS http://127.0.0.1:8021/v1/customer-support \
  -H 'Content-Type: application/json' -d '{"question":"I want to add a beneficiary"}'
curl -sS http://127.0.0.1:8021/v1/customer-support \
  -H 'Content-Type: application/json' -d '{"question":"Help please"}'
curl -sS http://127.0.0.1:8021/v1/support/urgent
```

Support responses add `orchestration: "mistral_agents_native"`, `specialist`, `handoffs` and `model_calls` to the validated result. `/v1/faq` keeps its existing contract. SSE emits safe progress and one validated final result or typed error; raw tokens and tool payloads stay private. Sensitive input fails before a session or stream opens. `SUPPORT_INFERENCE_DISABLED=1` leaves urgent support reachable. `URGENT_SUPPORT_URL` configures its HTTPS destination.

```sh
# Offline HTTP and real-SDK protocol fixtures:
uv run pytest -q
# Real HTTP native specialists, FAQ citations, fallback and urgent outage:
uv run python scripts/check_support_http.py
# Existing direct-FAQ regression suite:
uv run python scripts/check_faq_http.py
```

Historical direct-local results before mandatory Workflows: **134 offline tests**, **23/23 native HTTP scenarios**, and **27/27 direct-FAQ regression checks** pass. The [native HTTP evidence](docs/evidence/support-phase2-http.json) records actual calls: each FAQ used four generations plus one embedding; guidance used three generations; clarification used one. Screening and urgent outage used zero. FAQ answers took about **6.7–8.0 seconds** locally, so the proposed five-second target remains unmet in these smoke samples. Earlier failures and the full accounting are retained in [the implementation plan](Implementation_Plan.md). The current workflow HTTP suite writes `artifacts/support-workflow-http.json`; it requires Studio credentials and a reachable Studio service. Its in-process API test starts the worker through FastAPI lifespan. The previous evidence is in `docs/evidence/support-phase2-http.json`; these smoke results are not load or reliability certification.

## Required Studio workflow worker

Support chat runs inside `CustomerSupportWorkflow` through `mistralai-workflows==3.20.0`. The supervisor has six named specialists and selects one per request inside its local activity. `SyntheticTransactionWorkflow` uses the same graph, then orchestrates review and synthetic dispatch. Ollama, Chroma and tools run on the worker host.

Provide a dedicated `MISTRAL_STUDIO_API_KEY` in the API process environment. FastAPI starts one Studio workflow worker per API process during application startup and stops it during shutdown. Run Uvicorn with its default single worker to avoid registering multiple workers. There is no separate worker command or direct-agent HTTP fallback. Missing credentials, an unavailable Studio service or worker timeouts return safe HTTP 503 / SSE error responses. Urgent support remains available locally. Only screened synthetic public text is permitted; the previous three-question restriction is removed.

Run `uv run python scripts/check_support_http.py` for the full HTTP suite, or `uv run python scripts/check_studio_support.py` for one Riverside workflow execution. See [the runbook](docs/HYBRID_SUPPORT.md).

Live Studio registration and recovery remain **unverified**. Offline tests verify the default API submits a workflow and never falls back to direct agents. Studio checkpoints the activity result; individual agent-turn recovery and detailed child traces require separate verification. The 20-second workflow limit includes queueing; submission waits at most 18 seconds, then requests cancellation with a one-second bound. The existing API deadline stays 20 seconds. Worker activity retries are disabled to preserve the six-generation limit.

## Synthetic transactions with human approval

An authenticated question at `POST /v1/customer-support` starts the Studio workflow. The `HighStakeTransactions` child semantically extracts a synthetic transfer and its Python tool prepares it. Above INR 10,000 (configurable), the subagent requests `human_review`; Studio conditionally notifies the reviewer webhook and waits for its authenticated approval callback; no separate action creation or HITL request is needed. Execution requires exact customer confirmation and an authorized staff decision. Python rechecks approval before committing a simulated transfer; agent tools cannot execute banking writes. See [setup, API calls and approval walkthrough](docs/TRANSACTION_HITL.md). This demo uses fixed synthetic identities and a PostgreSQL action ledger; live Studio approval/resume remains unverified.

The action ledger uses the existing local Docker PostgreSQL server through `psycopg[binary]==3.3.9`. Set `DEMO_ACTION_DATABASE_URL` in your server environment or ignored repo-root `.env`; for example, use a dedicated application role and database at `127.0.0.1:5432/hdfc_actions`. The local setup has its own `hdfc_actions` database and role, isolated from the other application's data. API and worker must use the same connection configuration. Credentials stay on the server. Initialize the tables once before using authenticated actions:

```sh
uv run --env-file .env python -m hdfc_mistral_mvp.actions
uv run --env-file .env uvicorn hdfc_mistral_mvp.api:app --host 127.0.0.1 --port 8021
# Offline checks; database cases are explicitly skipped:
uv run pytest -q
# Full suite, including real PostgreSQL in disposable schemas:
uv run --env-file .env pytest -q --postgres
```

Initialization is repeatable. The default schema is `bank_actions`; `DEMO_ACTION_DB_SCHEMA` overrides it for isolated checks. Confirmations, staff decisions, action acceptance and outbox writes use PostgreSQL transactions and action row locks. No database connection is held while waiting for a reviewer or sending the webhook. Missing/unavailable storage returns a sanitized 503. `DEMO_ACTION_DB` and SQLite storage are no longer used.
