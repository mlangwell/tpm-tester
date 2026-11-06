"""Observable, bounded Azure OpenAI TPM experiments with no automatic retries."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TextIO, TypedDict
from urllib.parse import urlsplit

import httpx
import tiktoken
from azure.core.exceptions import AzureError
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

MIN_INPUT_TOKENS = 64
TokenProvider = Callable[[], Awaitable[str]]
Outcome = Literal["success", "throttled", "http_error", "transport_error", "invalid_response"]


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


class ResponseDetails(TypedDict):
    outcome: Outcome
    status_code: int | None
    headers: dict[str, str]
    usage: NotRequired[dict[str, int]]
    usage_warning: NotRequired[str]
    error: NotRequired[str]


@dataclass(frozen=True)
class Config:
    endpoint: str = ""
    deployment: str = ""
    quota_tpm: int = 30_000
    target_tokens: int = 45_000
    duration_seconds: float = 60.0
    input_tokens: int = 2_000
    max_output_tokens: int = 64
    concurrency: int = 4
    timeout_seconds: float = 60.0
    max_requests: int = 1_000
    encoding: str = "o200k_base"
    output_token_parameter: str = "max_completion_tokens"
    token_scope: str = "https://ai.azure.com/.default"

    def validate(self, *, live: bool) -> None:
        for name in (
            "quota_tpm", "target_tokens", "input_tokens", "max_output_tokens",
            "concurrency", "max_requests",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("duration_seconds", "timeout_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number.")
        for name in (
            "endpoint", "deployment", "encoding", "output_token_parameter", "token_scope",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string.")
        if self.input_tokens < MIN_INPUT_TOKENS or self.target_tokens < MIN_INPUT_TOKENS:
            raise ValueError(f"input_tokens and target_tokens must be >= {MIN_INPUT_TOKENS}.")
        if self.encoding not in tiktoken.list_encoding_names():
            raise ValueError(f"Unknown encoding: {self.encoding}.")
        if self.output_token_parameter not in ("max_completion_tokens", "max_tokens"):
            raise ValueError("output_token_parameter must be max_completion_tokens or max_tokens.")
        if not self.token_scope.startswith("https://") or not self.token_scope.endswith("/.default"):
            raise ValueError("token_scope must be an HTTPS Microsoft Entra .default scope.")
        if live and (not self.endpoint.strip() or not self.deployment.strip()):
            raise ValueError(
                "Live tests require endpoint and deployment. Set AZURE_OPENAI_ENDPOINT "
                "and AZURE_OPENAI_DEPLOYMENT, or configure them in the file/CLI."
            )
        if self.endpoint:
            self.request_url()

    def request_url(self) -> str:
        url = urlsplit(self.endpoint)
        allowed_paths = ("", "/openai/v1", "/openai/v1/chat/completions")
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or url.path.rstrip("/") not in allowed_paths
            or any(character.isspace() for character in self.endpoint)
        ):
            raise ValueError(
                "endpoint must be an HTTPS resource URL, /openai/v1 base URL, or "
                "/openai/v1/chat/completions URL without credentials, query, or fragment. "
                "Do not use a Foundry project endpoint."
            )
        path = url.path.rstrip("/")
        suffix = "/openai/v1/chat/completions" if not path else "/chat/completions"
        return self.endpoint.rstrip("/") if path.endswith("/chat/completions") else (
            self.endpoint.rstrip("/") + suffix
        )


@dataclass(frozen=True)
class Plan:
    prompt_tokens_per_request: tuple[int, ...]
    request_count: int
    total_prompt_tokens: int
    maximum_generated_tokens: int
    offered_prompt_tpm: float
    offered_rpm: float


@dataclass(frozen=True)
class RequestResult:
    request_number: int
    scheduled_seconds: float
    started_seconds: float
    latency_seconds: float
    prompt_text_tokens: int
    max_output_tokens: int
    outcome: Outcome
    status_code: int | None
    headers: dict[str, str]
    usage: dict[str, int] | None = None
    usage_warning: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Summary:
    planned_requests: int
    recorded_requests: int
    not_completed_requests: int
    succeeded: int
    throttled: int
    failed: int
    peak_60s_offered_prompt_tokens: int
    first_throttle_seconds: float | None
    maximum_schedule_lag_seconds: float
    elapsed_seconds: float
    usage_reported_requests: int
    usage_missing_requests: int
    reported_usage: dict[str, int] | None


def build_plan(config: Config) -> Plan:
    count = (config.target_tokens + config.input_tokens - 1) // config.input_tokens
    if count > config.max_requests:
        raise ValueError(
            f"Plan requires {count} requests, exceeding max_requests={config.max_requests}. "
            "Increase input_tokens or explicitly increase max_requests."
        )
    size, remainder = divmod(config.target_tokens, count)
    if size < MIN_INPUT_TOKENS:
        raise ValueError(
            f"Cannot split this target into requests of {MIN_INPUT_TOKENS} to "
            f"{config.input_tokens} input tokens. Increase target_tokens or input_tokens."
        )
    sizes = tuple(size + (index < remainder) for index in range(count))
    return Plan(
        prompt_tokens_per_request=sizes,
        request_count=count,
        total_prompt_tokens=sum(sizes),
        maximum_generated_tokens=count * config.max_output_tokens,
        offered_prompt_tpm=config.target_tokens * 60 / config.duration_seconds,
        offered_rpm=count * 60 / config.duration_seconds,
    )


def make_prompt(encoder: Tokenizer, token_count: int, marker: str) -> str:
    prefix = f"Quota test {marker}. Reply only OK.\n"
    if len(encoder.encode(prefix)) > token_count:
        raise ValueError("The request marker exceeds the configured input token count.")
    tokens = encoder.encode(prefix + " data" * token_count)[:token_count]
    prompt = encoder.decode(tokens)
    if len(encoder.encode(prompt)) != token_count:
        raise ValueError("The configured tokenizer could not produce the exact prompt token count.")
    return prompt


def diagnostic_headers(headers: Mapping[str, str]) -> dict[str, str]:
    allowed = {
        "retry-after", "retry-after-ms", "x-ms-retry-after-ms",
        "apim-request-id", "x-request-id", "x-ms-request-id", "x-ms-region",
    }
    return {
        name.lower(): value
        for name, value in headers.items()
        if name.lower().startswith("x-ratelimit-") or name.lower() in allowed
    }


def response_details(response: httpx.Response) -> ResponseDetails:
    details: ResponseDetails = {
        "outcome": "success",
        "status_code": response.status_code,
        "headers": diagnostic_headers(response.headers),
    }
    if not response.is_success:
        return {
            **details,
            "outcome": "throttled" if response.status_code == 429 else "http_error",
            "error": response.text,
        }
    try:
        data = response.json()
    except ValueError:
        return {**details, "outcome": "invalid_response", "error": response.text}
    if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
        return {
            **details,
            "outcome": "invalid_response",
            "error": f"No chat-completion choices in successful HTTP response: {response.text}",
        }
    raw_usage = data.get("usage")
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage: dict[str, int] = {}
    if isinstance(raw_usage, dict):
        for key in keys:
            value = raw_usage.get(key)
            if type(value) is not int or value < 0:
                break
            usage[key] = value
    if len(usage) == len(keys):
        return {**details, "usage": usage}
    return {
        **details,
        "outcome": "success",
        "usage_warning": "Response usage is missing or incomplete; billed token totals are unknown.",
    }


async def run_load(
    config: Config,
    plan: Plan,
    encoder: Tokenizer,
    client: httpx.AsyncClient,
    token_provider: TokenProvider,
    results: list[RequestResult],
    on_result: Callable[[RequestResult], None] | None = None,
) -> None:
    await token_provider()
    loop = asyncio.get_running_loop()
    started = loop.time()
    jobs = iter(enumerate(plan.prompt_tokens_per_request, start=1))
    run_marker = uuid.uuid4().hex[:12]

    async def worker() -> None:
        for number, prompt_tokens in jobs:
            scheduled = (number - 1) * config.duration_seconds / plan.request_count
            await asyncio.sleep(max(0, started + scheduled - loop.time()))
            if loop.time() - started >= config.duration_seconds:
                return
            token = await token_provider()
            prompt = make_prompt(encoder, prompt_tokens, f"{run_marker}/{number}")
            sent_at = loop.time()
            if sent_at - started >= config.duration_seconds:
                return
            payload = {
                "model": config.deployment,
                "messages": [{"role": "user", "content": prompt}],
                config.output_token_parameter: config.max_output_tokens,
                "stream": False,
            }
            details: ResponseDetails
            try:
                async with asyncio.timeout(config.timeout_seconds):
                    response = await client.post(
                        config.request_url(),
                        headers={"Authorization": f"Bearer {token}"},
                        json=payload,
                    )
                details = response_details(response)
            except TimeoutError:
                details = {
                    "outcome": "transport_error", "status_code": None, "headers": {},
                    "error": f"Total request timeout after {config.timeout_seconds} seconds.",
                }
            except httpx.HTTPError as error:
                details = {
                    "outcome": "transport_error", "status_code": None, "headers": {},
                    "error": f"{type(error).__name__}: {error}",
                }
            result = RequestResult(
                request_number=number,
                scheduled_seconds=scheduled,
                started_seconds=sent_at - started,
                latency_seconds=loop.time() - sent_at,
                prompt_text_tokens=prompt_tokens,
                max_output_tokens=config.max_output_tokens,
                **details,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)

    tasks = [
        asyncio.create_task(worker())
        for _ in range(min(config.concurrency, plan.request_count))
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def summarize(plan: Plan, results: Sequence[RequestResult]) -> Summary:
    ordered = sorted(results, key=lambda result: result.started_seconds)
    left = running = peak = 0
    for result in ordered:
        running += result.prompt_text_tokens
        while result.started_seconds - ordered[left].started_seconds >= 60:
            running -= ordered[left].prompt_text_tokens
            left += 1
        peak = max(peak, running)
    successful = [result for result in results if result.outcome == "success"]
    usages = [result.usage for result in successful if result.usage is not None]
    throttled = [result for result in results if result.outcome == "throttled"]
    return Summary(
        planned_requests=plan.request_count,
        recorded_requests=len(results),
        not_completed_requests=plan.request_count - len(results),
        succeeded=len(successful),
        throttled=len(throttled),
        failed=len(results) - len(successful) - len(throttled),
        peak_60s_offered_prompt_tokens=peak,
        first_throttle_seconds=min((result.started_seconds for result in throttled), default=None),
        maximum_schedule_lag_seconds=max(
            (max(0, result.started_seconds - result.scheduled_seconds) for result in results),
            default=0,
        ),
        elapsed_seconds=max(
            (result.started_seconds + result.latency_seconds for result in results), default=0,
        ),
        usage_reported_requests=len(usages),
        usage_missing_requests=len(successful) - len(usages),
        reported_usage={
            key: sum(usage[key] for usage in usages)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        } if usages else None,
    )


def result_exit_code(summary: Summary, *, expect_throttle: bool) -> int:
    if summary.failed or summary.not_completed_requests:
        return 1
    if expect_throttle and not summary.throttled:
        return 3
    return 0


def write_event(report: TextIO, event_type: str, data: dict[str, object]) -> None:
    report.write(json.dumps({"type": event_type, **data}, allow_nan=False) + "\n")
    report.flush()


def print_result(result: RequestResult) -> None:
    print(
        f"#{result.request_number:04d} {result.outcome} HTTP={result.status_code} "
        f"t={result.started_seconds:.3f}s latency={result.latency_seconds:.3f}s "
        f"prompt_text_tokens={result.prompt_text_tokens}",
        flush=True,
    )
    if result.headers:
        print(f"  headers: {json.dumps(result.headers)}", flush=True)
    if result.error is not None:
        print(f"  error: {result.error}", flush=True)
    if result.usage_warning:
        print(f"  WARNING: {result.usage_warning}", flush=True)


def parser_for_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration file.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Send real, billable model requests.")
    mode.add_argument("--dry-run", action="store_true", help="Print the plan only (the default).")
    parser.add_argument("--output", type=Path, help="New JSONL report path; never overwrites a file.")
    parser.add_argument(
        "--expect-throttle", action="store_true",
        help="Exit 3 if a complete live run observes no HTTP 429.",
    )
    options = (
        ("endpoint", str, "Resource endpoint or OpenAI v1 base URL."),
        ("deployment", str, "Azure deployment name, not the underlying model ID."),
        ("quota_tpm", int, "Assigned deployment TPM for comparison; default 30000."),
        ("target_tokens", int, "Total offered prompt-text tokens; default 45000."),
        ("duration_seconds", float, "Dispatch window; default 60 seconds."),
        ("input_tokens", int, "Maximum prompt-text tokens per request; default 2000."),
        ("max_output_tokens", int, "Output cap per request, NOT part of target_tokens; default 64."),
        ("concurrency", int, "Maximum simultaneous requests; default 4."),
        ("timeout_seconds", float, "Total timeout per request; default 60 seconds."),
        ("max_requests", int, "Safety ceiling on the planned request count; default 1000."),
        ("encoding", str, "tiktoken encoding; default o200k_base."),
        ("output_token_parameter", str, "max_completion_tokens (default) or max_tokens."),
        ("token_scope", str, "Entra scope; default https://ai.azure.com/.default."),
    )
    for name, value_type, help_text in options:
        parser.add_argument("--" + name.replace("_", "-"), type=value_type, help=help_text)
    return parser


def load_config(arguments: argparse.Namespace) -> Config:
    values = asdict(Config())
    if arguments.config is not None:
        with arguments.config.open(encoding="utf-8-sig") as file:
            supplied = json.load(file)
        if not isinstance(supplied, dict):
            raise ValueError("The configuration file must contain a JSON object.")
        unknown = set(supplied) - set(values)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}.")
        values.update(supplied)
    for name, environment_variable in (
        ("endpoint", "AZURE_OPENAI_ENDPOINT"),
        ("deployment", "AZURE_OPENAI_DEPLOYMENT"),
    ):
        if environment_variable in os.environ:
            values[name] = os.environ[environment_variable]
    for field in fields(Config):
        override = getattr(arguments, field.name)
        if override is not None:
            values[field.name] = override
    config = Config(**values)
    config.validate(live=arguments.live)
    return config


async def execute_live(
    config: Config,
    plan: Plan,
    results: list[RequestResult],
    on_result: Callable[[RequestResult], None],
) -> None:
    encoder = tiktoken.get_encoding(config.encoding)
    async with DefaultAzureCredential(exclude_interactive_browser_credential=True) as credential:
        provider = get_bearer_token_provider(credential, config.token_scope)
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=0),
            timeout=config.timeout_seconds,
            follow_redirects=False,
        ) as client:
            await run_load(config, plan, encoder, client, provider, results, on_result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = parser_for_cli()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments)
        plan = build_plan(config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(
        f"Configured quota: {config.quota_tpm:,} TPM (comparison only; quota is not changed)\n"
        f"Target: {plan.total_prompt_tokens:,} offered prompt-text tokens across "
        f"{plan.request_count} requests in {config.duration_seconds:g}s\n"
        f"Offered average: {plan.offered_prompt_tpm:,.1f} prompt TPM; "
        f"{plan.offered_rpm:,.1f} RPM; concurrency <= {config.concurrency}\n"
        f"Output cap: {config.max_output_tokens} per request "
        f"({plan.maximum_generated_tokens:,} maximum, not guaranteed or counted toward target)"
    )
    if not arguments.live:
        print("DRY RUN: no credentials acquired and no model requests sent. Use --live to execute.")
        return 0

    output = arguments.output or Path("results") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:8] + ".jsonl"
    )
    results: list[RequestResult] = []
    fatal_error: str | None = None
    interrupted = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as report:
            write_event(report, "plan", {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config), "plan": asdict(plan),
            })

            def on_result(result: RequestResult) -> None:
                write_event(report, "request", asdict(result))
                print_result(result)

            try:
                asyncio.run(execute_live(config, plan, results, on_result))
            except KeyboardInterrupt:
                interrupted = True
                fatal_error = "Interrupted; in-flight requests may have been sent without a recorded response."
            except (AzureError, httpx.HTTPError, OSError, ValueError) as error:
                fatal_error = f"{type(error).__name__}: {error}"
            if fatal_error:
                write_event(report, "fatal", {"error": fatal_error})
                print(f"ERROR: {fatal_error}", file=sys.stderr)
            summary = summarize(plan, results)
            write_event(report, "summary", asdict(summary))
    except OSError as error:
        print(f"ERROR: could not write report {output}: {error}", file=sys.stderr)
        return 1

    print("Summary:\n" + json.dumps(asdict(summary), indent=2))
    print(f"Report: {output.resolve()}")
    print("Offered tokens include rejected attempts. Reported usage covers successful responses only.")
    if summary.not_completed_requests:
        print(
            "WARNING: incomplete workload; the dispatch window expired or the run failed. "
            "Increase concurrency/window or reduce the workload."
        )
    if summary.throttled:
        print("HTTP 429 observed. Inspect error bodies and headers: TPM, RPM, or capacity may be responsible.")
    elif arguments.expect_throttle:
        print("FAIL: expected an HTTP 429, but none was observed.", file=sys.stderr)
    if interrupted:
        return 130
    return 1 if fatal_error else result_exit_code(summary, expect_throttle=arguments.expect_throttle)


if __name__ == "__main__":
    raise SystemExit(main())
