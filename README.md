# Microsoft Foundry TPM quota tester

A Python CLI for testing an **Azure OpenAI chat deployment in Microsoft Foundry** using **Microsoft Entra ID**. No Azure resources or quotas are changed.

- Configure the deployment, total input tokens, request size, output cap, concurrency, and dispatch window.
- See every response status, latency, throttling body, retry-after value, request ID, and returned rate-limit headers.
- No automatic retries or backoff: each planned request is attempted at most once, so HTTP 429 responses stay visible.
- Flushed JSONL reports preserve completed responses even if a later request fails.
- Dry-run by default; real requests require `--live` and can incur charges.

## Setup (PowerShell)

Requires Python 3.11+ and Azure CLI for local sign-in.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
az login

$env:AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "YOUR-DEPLOYMENT-NAME"
```

Your identity needs **Cognitive Services OpenAI User**, or an equivalent inference role, on the target resource. Subscription Reader/Contributor alone does not necessarily grant inference access. The suite uses `DefaultAzureCredential`: Azure CLI credentials work locally; managed identity and service-principal environments are also supported. No API key is used.

Use the **resource endpoint**, not a Foundry project URL. A resource root, `/openai/v1/` base URL, or full `/openai/v1/chat/completions` URL is accepted. This suite uses the v1 chat API, without a dated `api-version`. The deployment must support chat completions.

## Try your 30,000-TPM deployment

First inspect the plan; this does not acquire credentials or call a model:

```powershell
.\.venv\Scripts\python.exe -m tpm_tester --dry-run
```

Default: **45,000 offered prompt-text tokens**, spread across **23 requests in 60 seconds**, with concurrency up to 4. Each prompt is at most 2,000 tokens. The output cap is 64 tokens per request.

Run the experiments separately. Let capacity recover between runs (at least a minute, or longer if indicated by the service), and pause other clients using the deployment:

```powershell
# Below quota: roughly 15K input TPM, plus message overhead and generated output.
.\.venv\Scripts\python.exe -m tpm_tester --live --target-tokens 15000

# Near quota: 30K input TPM; overhead/output may push this above the limit.
.\.venv\Scripts\python.exe -m tpm_tester --live --target-tokens 30000

# Above quota: 45K offered input TPM; assert that a 429 is observed.
.\.venv\Scripts\python.exe -m tpm_tester --live --target-tokens 45000 --expect-throttle
```

A shorter window increases offered load without increasing the total input token target:

```powershell
.\.venv\Scripts\python.exe -m tpm_tester --live `
    --quota-tpm 30000 --target-tokens 60000 --duration-seconds 30 `
    --input-tokens 4000 --max-output-tokens 64 --concurrency 4 `
    --expect-throttle --output .\results\over-quota.jsonl
```

Here, 60,000 input tokens are offered in 30 seconds: **120,000 average offered input TPM**, not 60,000 TPM. Start with smaller experiments and use a nonproduction deployment.

## Configuration

Copy [config.example.json](config.example.json) to `config.local.json`, edit it, and run:

```powershell
.\.venv\Scripts\python.exe -m tpm_tester --config .\config.local.json --dry-run
.\.venv\Scripts\python.exe -m tpm_tester --config .\config.local.json --live
.\.venv\Scripts\python.exe -m tpm_tester --help
```

Precedence: **CLI > environment > JSON file > defaults**. The endpoint/deployment environment variables are shown above. JSON fields match the example; CLI flags use hyphens instead of underscores. Unknown fields and invalid values are rejected.

| Setting | Meaning |
|---|---|
| `quota_tpm` | Your assigned deployment TPM, for comparison only; never updates Azure. |
| `target_tokens` | Exact total prompt-text tokens to offer across the whole dispatch window, including rejected attempts. |
| `duration_seconds` | Window for starting requests; in-flight requests may finish afterward. |
| `input_tokens` | Maximum prompt-text tokens per request; the target is distributed evenly, without rounding it up. Minimum 64. |
| `max_output_tokens` | Per-request output cap, **not included in the input target**. The prompt asks for a short `OK` response to limit output cost. |
| `concurrency` | Maximum in-flight requests. Slow responses can prevent the full workload from being dispatched. |
| `timeout_seconds` | Total wall-clock timeout for each HTTP request; timeout errors are recorded. |
| `max_requests` | Safety ceiling, default 1,000. Oversized plans are rejected before authentication. |
| `encoding` | `tiktoken` encoding, default `o200k_base`; choose an encoding appropriate for your model, e.g. `cl100k_base` for older models. |
| `output_token_parameter` | `max_completion_tokens` by default, including reasoning models. Select `max_tokens` if required by an older deployment. |
| `token_scope` | Defaults to `https://ai.azure.com/.default` for public Azure v1. Override only if your Azure environment requires another audience. |

No temperature or reasoning-specific parameters are forced. For reasoning models, the output cap also includes reasoning tokens; raise it if your model needs a larger budget.

## Reading the results

Example throttling output (illustrative; messages and headers vary):

```text
#0017 throttled HTTP=429 t=41.740s latency=0.120s prompt_text_tokens=1956
  headers: {"retry-after": "18", "x-ratelimit-remaining-tokens": "0"}
  error: {"error":{"code":"429","message":"...original service message..."}}
```

Reports are new files under `results` by default. `--output` selects a path, but existing files are never overwritten. Each JSONL line is a `plan`, completed `request`, `fatal` error, or final `summary`. Authentication headers, access tokens, and generated answers are not recorded. Raw service error bodies are preserved; review reports before sharing.

Important distinctions:

- **Offered is not accepted or billed.** `target_tokens` counts locally tokenized prompt text, including requests that Azure rejects. It excludes chat-message framing and output. Accepted prompt/completion totals come from response `usage`; missing usage is explicitly reported, not silently counted as zero.
- **Output caps do not force output generation.** The suite drives its target with input alone, rather than assuming a maximum output budget was consumed or reserved.
- **Tokenizer counts are local.** Azure's actual model tokenizer and quota enforcement can differ. The unique early request marker reduces cross-request prefix caching; this is not a measurement of cache effectiveness.
- **429 does not prove TPM exhaustion.** RPM limits, short evaluation windows, shared-service capacity, or temporarily adjusted limits can also produce 429. Inspect raw errors, remaining token/request headers, and effective limit headers. Headers may be absent.
- **Check achieved load.** `peak_60s_offered_prompt_tokens` measures completed attempts' submitted input in a rolling 60-second window, not response-completion throughput. Scheduling lag and `not_completed_requests` expose insufficient concurrency. Incomplete runs fail rather than claiming the target was reached.
- **No guaranteed threshold.** A 45K run against a 30K allocation might not produce 429 because of window alignment or service behavior. No 429 is an observation, not proof quota is unenforced. Adjust the bounded workload, repeat after recovery, and correlate with Azure metrics.
- The dispatch window ends new submissions; existing calls finish or time out. Ctrl+C produces a partial report when possible; interrupted in-flight requests might have reached Azure without a recorded result.

Exit codes: `0` complete observation run (429s are expected test results); `1` non-429/transport/fatal errors or incomplete workload; `2` invalid arguments/configuration; `3` complete run with no 429 when `--expect-throttle` was requested; `130` interrupted.

## Offline automated tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests use mocked HTTP transports and a deterministic tokenizer: no Azure sign-in, model calls, or charges. They cover exact workload totals, bearer authentication, request parameters, 429 visibility without retries, other HTTP errors, timeouts, scheduling deadlines, concurrency, reports, and CLI safety. Live prompt construction additionally checks exact round-trip token counts with the selected real tokenizer. The first live use of a `tiktoken` encoding may download its public encoding data.

References: [Microsoft quota and rate-limit guidance](https://learn.microsoft.com/azure/foundry/openai/how-to/quota), [v1 API and Entra authentication](https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle), [Azure Identity credential chains](https://learn.microsoft.com/azure/developer/python/sdk/authentication/credential-chains).
