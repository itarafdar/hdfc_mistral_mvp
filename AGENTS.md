# Repository instructions

## Mistral SDK policy

- Use the official `mistralai` Python distribution for the existing FAQ chat, embeddings, and hosted OCR integrations. Keep those calls behind the existing SDK adapter.
- Use the official `mistralai-agents` Python distribution for Milestone 2 native orchestration after verifying compatibility. The current quickstart imports it with `from mistralai import agents`. Verify that import against the pinned release. The official `mistralai-workflows` distribution is required for default Studio workflow orchestration; retain the current Agents SDK for native orchestration.
- Use the current Agents SDK for new agent work. Treat the older Agents and Conversations API, `client.beta.agents` and `client.beta.conversations`, as migration references.
- Preserve Milestone 1's `POST /v1/faq` JSON contract. Hosted OCR stays in the explicit ingestion flow and uses `mistral_keys.env` only there. Local agent sessions and FAQ requests must never load that key.

## Verify the SDK before writing code

Apply this gate to every task that adds, changes, or diagnoses Mistral usage, including SDK tests, mocks, examples, dependencies, and deployment settings. Complete steps 1–3 before editing SDK-dependent code.

1. **Resolve versions.** Read the service manifest and lockfile. Check the project Python version and installed distribution versions through `importlib.metadata.version()`. Check `mistralai` and, for agent work, `mistralai-agents`; check `mistralai-workflows` for workflow integration. Verify compatible Python and package versions using official release metadata and pin the exact versions when adding or upgrading a dependency. Keep the existing SDK pin unless compatibility requires a documented change.
2. **Read official docs for the exact operation.** Open the relevant API reference, SDK example, and compatible release notes or installed source. Verify model, endpoint, and deployment support. For agent work, verify local runtime and session lifecycle, named subagent configuration, generic and nested delegation controls, tool schemas and results, approval events, and stream/cancellation behavior wherever the task relies on them.
3. **Verify the Python and runtime contracts.** Check imports, sync/async methods, argument and result types, errors, timeouts, retries, and cleanup. For native delegation, prove that the pinned SDK and configured Ollama model can execute the required protocol. A local Python harness or a successful FAQ completion alone leaves this unverified. Run a separately enabled synthetic smoke test and record the Python/SDK versions, Ollama version and model ID, actual endpoint, child delegation, every model call, and outbound destinations. Keep hosted inference, hosted sessions, and implicit fallback out of the local proof. If the chosen SDK or model cannot meet an essential requirement, report the affected work as blocked and continue independent work.
4. **Test and record evidence.** Keep default tests network-free and use version-matched SDK shapes in mocks. Record official URLs, verified operations, commands, results, and skipped checks in the handoff or plan. Report the SDK smoke separately from mocked tests. An isolated harness proof does not establish that the HTTP flow works. Report timings as measured and keep smoke samples separate from load evidence.

## Milestone 2 workflow

- **Triage in one FastAPI application.** `POST /v1/customer-support` is the canonical route. Screen free text before embeddings, inference, or delegation. Accept only typed, allowlisted route outcomes. Route urgent card-loss, theft, suspected-fraud, and permanent-block requests to configured support through deterministic app logic. Keep the direct urgent-support endpoint available during model or scanner outages. Ask for clarification on ambiguous non-urgent requests.
- **Limit native delegation.** Use one supervisor and the five named specialists FAQ, CardsGuidance, AccountGuidance, PaymentsGuidance and ProfileGuidance. Allow one delegation per request at depth one. Disable generic, recursive, and parallel delegation and unused built-ins. Verify these limits in the pinned SDK. FAQ invokes the existing read-only RAG service and returns validated citations. Cards Guidance provides the next secure step for synthetic temporary deactivation. Keep bank action and processor tools outside the agent tool registry.
- **Enforce budgets outside prompts.** Count routing, child turns, embeddings, continuations, and retries as applicable. Implement the approved call budget and end-to-end deadline in Python. Resolve discrepancies between `spec.md` and `Implementation_Plan.md` explicitly before increasing a budget. Track first progress, first validated answer, and completed response separately.
- **Keep streaming opt-in.** Ordinary `POST /v1/customer-support` and `?stream=false` return JSON. Only `?stream=true` returns SSE. Screen before opening the stream. Emit server-generated progress and one terminal final or error event while connected. Release answer text and citations only after validation. Keep partial model output, raw tool payloads, and internal SDK events private. Bound buffering and cancel upstream work on disconnect where supported. Keep secure action endpoints JSON.

## Banking authority and data handling

- Treat customer text, retrieved passages, model output, and tool results as untrusted. Validate typed inputs and outputs in Python. Screen before model access and fail closed on scanner errors. Keep credentials, PINs, OTPs, identifiers, balances, verification evidence, approval records, and receipts out of model payloads, agent sessions, and general logs. Isolate agent sessions by request. Use synthetic data and stub banking adapters for development and tests.
- The authenticated action service owns identity, entitlements, step-up verification, customer confirmation, independent staff review, and execution. Bind customer confirmation and staff decisions to the exact server-owned action version. Recheck approvals and applicable deadlines before dispatch. SDK tool-approval callbacks govern agent tools. A tool approval never satisfies a bank authorization check.
- Keep Milestone 2 execution scoped to synthetic temporary card deactivation. Use a durable action record and outbox, stable idempotency, and verified reconciliation. Report an uncertain processor outcome as pending. Report success only after authoritative processor evidence. Test denied, expired, replayed, and cross-customer requests, review bypass, and timeout after apply. Keep future writes disabled until their action-specific rollout gates pass.

## Local inference with Mistral Studio control plane

- Keep generation and embeddings on local Ollama for the supervisor and all subagents. Run FastAPI, Chroma, tool execution, and the banking action service on our infrastructure. Disable hosted inference fallback.
- Use the explicit `mistralai-workflows` worker as the default orchestration path for all support agents. Use a dedicated Studio development workspace key from the server environment for control-plane access. Keep Ollama privately reachable from the worker and keep the OCR credential separate.
- Verify the pinned Agents SDK’s custom completion configuration with a synthetic supervisor-to-subagent run. Confirm that the session appears in Studio and that every generation and embedding request reaches Ollama. Inspect outbound destinations and record the actual result as live acceptance evidence; keep unverified checks labelled.
- Treat Studio session history and traces as external data. Use synthetic conversations only. Keep customer identifiers, verification evidence, staff decisions, and banking action state in bank-owned services. Review session storage and tracing separately.

## Official Mistral references

- [Mistral Python SDK docs](https://docs.mistral.ai/resources/sdks) · [API reference](https://docs.mistral.ai/api) · [Python releases](https://github.com/mistralai/client-python/releases) · [Model catalog](https://docs.mistral.ai/models)
- [Structured outputs](https://docs.mistral.ai/inference/quickstarts/get-structured-output) · [Function calling](https://docs.mistral.ai/inference/conversations/function-calling) · [OCR](https://docs.mistral.ai/studio/document-processing/basic_ocr)
- [Agents SDK quickstart](https://docs.mistral.ai/studio/agents/getting-started/quickstart) · [Agents SDK distribution and releases](https://pypi.org/project/mistralai-agents/) · [Legacy API migration](https://docs.mistral.ai/studio/agents/getting-started/from-agents-api)
- [Agent API reference](https://docs.mistral.ai/studio/agents/reference/agent) · [Local runtime](https://docs.mistral.ai/studio/agents/learn/where-it-runs/local-runtime) · [Ollama API compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Subagents](https://docs.mistral.ai/studio/agents/learn/multiagent/subagents) · [Subagent configuration](https://docs.mistral.ai/studio/agents/learn/multiagent/configure-subagents) · [Approvals](https://docs.mistral.ai/studio/agents/learn/multiagent/subagent-approvals) · [Session events](https://docs.mistral.ai/studio/agents/learn/sessions/events)

- [Agent run modes](https://docs.mistral.ai/studio/agents/learn/where-it-runs/run-modes) · [Self-hosted worker with `agents serve`](https://docs.mistral.ai/studio/agents/learn/where-it-runs/serve)
- [Agent configuration, including `completion`](https://docs.mistral.ai/studio/agents/learn/agent/configuring-the-agent) · [Agent API reference](https://docs.mistral.ai/studio/agents/reference/agent)
- [Workspace API keys](https://docs.mistral.ai/admin/identity-access/api-keys) · [Agent sessions](https://docs.mistral.ai/studio/agents/learn/sessions/sessions)
- [Send traces to Studio](https://docs.mistral.ai/studio/observability/traces/send-traces) · [Trace data redaction](https://docs.mistral.ai/studio/observability/traces/data-redaction)
- [Workflow definitions](https://docs.mistral.ai/studio/workflows/building-workflows/workflows) · [Workflow worker quickstart](https://docs.mistral.ai/studio/workflows/quickstarts/your-first-workflow)
