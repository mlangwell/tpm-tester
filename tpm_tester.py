"""Test how an existing Azure OpenAI deployment responds to bounded token load.

The program reads configuration, plans synthetic prompts, and prints that plan.
Only --live starts authentication and sends potentially billable model requests.
It never creates deployments, changes Azure quota, or retries rejected requests.

Input-token targets describe attempted load, not accepted or billed usage.
HTTP 429 responses are preserved as observations; their error bodies and headers
help distinguish token limits, request limits, and service-capacity throttling.
Start reading at Config for settings and main for the complete execution flow.
"""

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
from dotenv import dotenv_values

# Leave room for the instruction and unique marker inside every synthetic prompt.
MIN_INPUT_TOKENS = 64
# Calling this asynchronous provider obtains a bearer token, refreshing it if needed.
TokenProvider = Callable[[], Awaitable[str]]
Outcome = Literal["success", "throttled", "http_error", "transport_error", "invalid_response"]


class Tokenizer(Protocol):
    """Describe the tokenizer operations required to construct sized prompts.

    A Protocol describes an interface rather than a concrete tokenizer.
    Live runs use tiktoken; offline tests can supply a deterministic substitute.
    """

    def encode(self, text: str) -> list[int]:
        """Split text into the token IDs understood by this tokenizer.

        Args:
            text: Prompt text whose local token count is needed.

        Returns:
            Token IDs; their list length is the locally measured token count.
        """
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Reconstruct prompt text from a chosen sequence of token IDs.

        Args:
            tokens: Token IDs, typically trimmed to the desired prompt size.

        Returns:
            Text that make_prompt will encode again to verify its exact size.
        """
        ...


class ResponseDetails(TypedDict):
    """Describe the dictionary extracted from an HTTP response.

    TypedDict documents dictionary keys for type checking; it is not an Azure
    response class. The request's timing and planned size are added separately.

    Attributes:
        outcome: success, throttled (HTTP 429), http_error, transport_error,
            or invalid_response (an unusable HTTP-success response).
        status_code: HTTP status, or None when no HTTP response was received.
        headers: Lowercase diagnostic headers only, excluding credentials/cookies.
        usage: Optional service-reported prompt_tokens, completion_tokens, and
            total_tokens; present only when all three are nonnegative integers.
        usage_warning: Optional explanation of missing or incomplete usage.
        error: Optional original HTTP error body or transport/parsing description.
    """

    outcome: Outcome
    status_code: int | None
    headers: dict[str, str]
    usage: NotRequired[dict[str, int]]
    usage_warning: NotRequired[str]
    error: NotRequired[str]


@dataclass(frozen=True)
class Config:
    """Hold the effective settings for one experiment.

    This frozen dataclass is an immutable settings container. Its fields also
    describe the arguments accepted by its generated constructor. These values
    configure this client only; none of them update your deployment in Azure.

    Attributes:
        endpoint: HTTPS resource root, OpenAI v1 base URL, or full chat URL.
            Read from AZURE_OPENAI_ENDPOINT; not a Foundry project URL.
        deployment: Your Azure deployment name, not necessarily its model ID.
            Read from AZURE_OPENAI_DEPLOYMENT.
        quota_tpm: Assigned tokens-per-minute limit used for display/comparison.
            The app neither retrieves nor changes the actual Azure allocation.
        target_tokens: Total prompt-text tokens to attempt during the whole run.
            Includes rejected attempts, but excludes output and message framing.
        duration_seconds: Time window for starting requests, in seconds.
            In-flight requests can finish after this window closes.
        input_tokens: Maximum local prompt-text tokens in each request.
            The planner balances request sizes to match target_tokens exactly.
        max_output_tokens: Maximum output budget sent with each request.
            It is not guaranteed output and is not included in target_tokens.
        concurrency: Maximum number of requests that can be in flight together.
            This is a ceiling, not the number of requests started each second.
        timeout_seconds: Wall-clock timeout for an individual HTTP attempt.
        max_requests: Safety ceiling for the total number of planned attempts.
        encoding: tiktoken encoding used for local prompt construction/counting.
            It should match your model; Azure's measured usage can still differ.
        output_token_parameter: API field carrying the output cap:
            max_completion_tokens, or max_tokens for compatible older models.
        token_scope: Entra token audience, normally https://ai.azure.com/.default.
            This is a scope string, not an access token or API key.
    """

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
        """Reject invalid settings before workload planning or network activity.

        Args:
            live: Whether real model calls were requested. Endpoint/deployment
                may be omitted for a dry run, but are required for a live run.

        Raises:
            ValueError: A value has an unsupported type, size, encoding, API
                parameter name, scope, or endpoint format.
        """
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
                "and AZURE_OPENAI_DEPLOYMENT in .env or your environment, "
                "or configure them in the JSON file/CLI."
            )
        if self.endpoint:
            self.request_url()

    def request_url(self) -> str:
        """Validate endpoint syntax and produce the chat-completions URL.

        A resource root gains /openai/v1/chat/completions, a v1 base gains
        /chat/completions, and an existing full chat URL is kept. This method
        does not check whether the resource exists or whether you can access it.

        Returns:
            The HTTPS URL to which model inference requests will be posted.

        Raises:
            ValueError: The URL uses an unsupported scheme/path, embeds
                credentials, or contains query parameters/fragments/whitespace.
        """
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
    """Describe a proposed workload without sending requests.

    Attributes:
        prompt_tokens_per_request: Ordered local prompt sizes, one per attempt.
        request_count: Number of model calls the run will try to dispatch.
        total_prompt_tokens: Sum of planned input sizes; equals target_tokens.
        maximum_generated_tokens: Sum of per-request output caps. This is only
            an upper budget, not expected output, billed usage, or reserved TPM.
        offered_prompt_tpm: Planned input tokens * 60 / dispatch-window seconds.
            This is an average offered rate, not Azure's enforcement counter.
        offered_rpm: Planned requests * 60 / dispatch-window seconds.
            Azure also enforces request limits, including over short windows.
    """

    prompt_tokens_per_request: tuple[int, ...]
    request_count: int
    total_prompt_tokens: int
    maximum_generated_tokens: int
    offered_prompt_tpm: float
    offered_rpm: float


@dataclass(frozen=True)
class RequestResult:
    """Record one HTTP attempt after it has returned or failed.

    Completed attempts can arrive out of request-number order. Requests that
    were never dispatched, or were interrupted without a result, have no record.

    Attributes:
        request_number: One-based position in the original workload plan.
        scheduled_seconds: Intended start offset from the load-test clock.
        started_seconds: Actual start offset, just before the HTTP attempt.
        latency_seconds: Elapsed time for the HTTP attempt and response parsing.
        prompt_text_tokens: Local input-text token count, even if Azure rejects it.
        max_output_tokens: Output cap requested, not actual generated tokens.
        outcome: Classified result, such as success, throttled, or transport_error.
        status_code: HTTP response status, or None for connection/timeout failures.
        headers: Diagnostic rate-limit, retry, region, and request-ID headers.
        usage: Valid service-reported token counts, or None when unavailable.
        usage_warning: Why usage cannot be counted for an otherwise valid response.
        error: Original HTTP failure body or a transport/parsing error description.
    """

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
    """Summarize observed attempts without treating offered load as billed usage.

    Attributes:
        planned_requests: Number of attempts in the initial plan.
        recorded_requests: Completed attempts, including throttling and failures.
        not_completed_requests: Planned attempts without recorded results.
            They may be unsent, or interrupted after reaching Azure.
        succeeded: HTTP-success responses with a usable chat-completion envelope.
        throttled: Attempts returning HTTP 429, regardless of its underlying cause.
        failed: Other recorded HTTP, transport, or response-format failures.
            Fatal startup/run errors are reported separately by main.
        peak_60s_offered_prompt_tokens: Largest sum of recorded attempts' input
            sizes started within any rolling 60-second window; includes rejections.
        first_throttle_seconds: Start offset of the earliest attempt returning
            HTTP 429, or None if none did; not the time its response arrived.
        maximum_schedule_lag_seconds: Largest positive actual-minus-planned
            start delay, showing whether the client fell behind its schedule.
        elapsed_seconds: Latest recorded completion offset from the load clock;
            excludes initial authentication/tokenizer setup and final reporting.
        usage_reported_requests: Successful attempts with complete token usage.
        usage_missing_requests: Successful attempts without complete token usage.
        reported_usage: Sum of available prompt/completion/total usage, or None
            if none was reported. Partial coverage is not the whole run's cost.
    """

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
    """Divide the input target into bounded, nearly equal prompt sizes.

    Request count is rounded up from target_tokens / input_tokens, but the
    token total is not rounded up. Extra tokens are distributed one at a time
    across the first requests. This performs arithmetic only, with no Azure calls.

    Args:
        config: Validated settings supplying the target, per-request size,
            dispatch window, output cap, and request-count safety ceiling.

    Returns:
        A plan whose input sizes sum exactly to config.target_tokens.

    Raises:
        ValueError: The request ceiling would be exceeded, or the target cannot
            be split while respecting both the minimum and maximum prompt size.
    """
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
    """Construct synthetic input containing exactly the requested local tokens.

    The text starts with a unique marker and asks for only "OK", followed by
    repetitive filler. The marker reduces shared-prefix caching across requests;
    the instruction reduces expected output, but does not guarantee its length.
    Azure can count tokens differently, including chat-message overhead.

    Args:
        encoder: Tokenizer that encodes/decodes the model's chosen local encoding.
        token_count: Exact prompt-text size, including the marker and instruction.
        marker: Unique run/request identifier placed early in the prompt.

    Returns:
        Prompt text verified by encoding it again after trimming/decoding.

    Raises:
        ValueError: The marker/instruction cannot fit, or a token round trip
            cannot preserve the requested size with this tokenizer.
    """
    prefix = f"Quota test {marker}. Reply only OK.\n"
    if len(encoder.encode(prefix)) > token_count:
        raise ValueError("The request marker exceeds the configured input token count.")
    tokens = encoder.encode(prefix + " data" * token_count)[:token_count]
    prompt = encoder.decode(tokens)
    if len(encoder.encode(prompt)) != token_count:
        raise ValueError("The configured tokenizer could not produce the exact prompt token count.")
    return prompt


def diagnostic_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Keep useful throttling diagnostics without copying every HTTP header.

    Args:
        headers: Response-header names and values, usually from httpx.

    Returns:
        Lowercase rate-limit, retry-after, request-ID, and region headers.
        Authorization and cookie headers are not included. Missing diagnostic
        headers remain absent rather than being invented.
    """
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
    """Classify an HTTP response and extract diagnostics without retrying it.

    HTTP 429 becomes throttled; other non-success statuses become http_error.
    HTTP-success responses must contain nonempty chat choices. Missing usage
    produces an explicit warning, not fabricated zero-token usage. Successful
    generated answers are not retained, while error bodies are kept verbatim.

    Args:
        response: Completed HTTP response, whether successful or unsuccessful.

    Returns:
        Fields ready to include in RequestResult, without request timing/size.
        A throttled result proves a 429 was observed, not that TPM caused it.
    """
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
    """Dispatch paced requests with bounded concurrency and no application retries.

    Authentication is checked before starting the timing clock. Each worker
    claims one planned request at a time, waits for its intended start, builds
    the prompt, and submits it once. Nothing new is submitted after the dispatch
    window expires; existing HTTP calls can finish or reach their timeout.
    Timeouts/network errors are recorded per attempt. Authentication or callback
    failures propagate after outstanding worker tasks are cancelled and awaited.

    Args:
        config: Validated destination, timing, authentication scope, and limits.
        plan: Prompt sizes and request count computed by build_plan.
        encoder: Tokenizer used to create exactly sized synthetic prompt text.
        client: Caller-owned asynchronous HTTP client; this function does not
            close it. execute_live supplies the no-retry, no-redirect client.
        token_provider: Awaitable bearer-token factory from Azure Identity.
            Checked initially and called for each request to allow token refresh.
        results: Caller-owned list appended to as attempts finish. Retains
            completed observations even if a later attempt stops the run.
        on_result: Optional synchronous callback for each appended observation;
            the CLI uses it to write the report and print immediate diagnostics.
    """
    await token_provider()
    loop = asyncio.get_running_loop()
    started = loop.time()
    jobs = iter(enumerate(plan.prompt_tokens_per_request, start=1))
    run_marker = uuid.uuid4().hex[:12]

    async def worker() -> None:
        """Consume the shared plan iterator using the enclosing run's resources.

        This closure takes no arguments: config, plan, encoder, client, token
        provider, results, and callback come from run_load. Every claimed job is
        attempted at most once; a 429 is recorded rather than backed off/retried.
        """
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
    """Compute completion counts, achieved offered load, timing, and known usage.

    The rolling-window calculation uses actual request start offsets, not
    response arrival times. Rejected requests count toward offered load, but
    only successful responses with valid usage contribute to reported usage.

    Args:
        plan: Original workload, used to identify missing attempt records.
        results: Completed observations, possibly out of order or incomplete.
            This function does not modify their order or add missing records.

    Returns:
        Run totals that distinguish a complete experiment from an incomplete one.
    """
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
    """Choose the outcome code for a run without a separate fatal/interruption.

    Args:
        summary: Counts of completed, failed, missing, and throttled attempts.
        expect_throttle: Whether observing at least one HTTP 429 is required.
            This does not require the service to identify TPM as the cause.

    Returns:
        1 for failed/missing attempts; otherwise 3 if a required 429 was absent;
        otherwise 0. A 429 alone is an experimental result, not a program failure.
        main handles configuration errors, fatal errors, and interruptions.
    """
    if summary.failed or summary.not_completed_requests:
        return 1
    if expect_throttle and not summary.throttled:
        return 3
    return 0


def write_event(report: TextIO, event_type: str, data: dict[str, object]) -> None:
    """Append and flush one JSONL record so completed progress is preserved.

    Args:
        report: Writable UTF-8 text report opened by main; remains open afterward.
        event_type: Record category: plan, request, fatal, or summary.
        data: JSON-serializable fields for the record, without a type key.
            Request errors may contain service-provided details; review before sharing.

    Raises:
        OSError: Writing/flushing the report fails.
        TypeError: A field cannot be serialized as JSON.
        ValueError: Data contains nonfinite numbers that JSON must not accept.
    """
    report.write(json.dumps({"type": event_type, **data}, allow_nan=False) + "\n")
    report.flush()


def print_result(result: RequestResult) -> None:
    """Display one observation immediately, including any raw throttling error.

    Args:
        result: Completed attempt to print with its HTTP status, timing, local
            prompt size, diagnostic headers, raw error, and optional usage warning.
            Successful model answers are not part of this record.
    """
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
    """Define the command-line interface without reading config or calling Azure.

    Config fields use matching hyphenated flags, such as --target-tokens.
    --config selects JSON settings; --output selects a new report path.
    --live explicitly authorizes model calls; --dry-run is the safe default.
    --expect-throttle makes a complete run fail if no HTTP 429 was observed.

    Returns:
        The parser used by main; --help describes each CLI setting and exits.
    """
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
    """Combine local settings in precedence order and validate the final values.

    Highest to lowest priority: CLI, process environment, current directory's
    .env, JSON file, built-in defaults. Endpoint/deployment use AZURE_OPENAI_
    variable names; all other fields use TPM_ plus their uppercase field name.
    Reading .env does not change os.environ, search parent folders, or acquire
    Azure credentials. Numeric strings are converted to their setting's type.

    Args:
        arguments: Namespace produced by parser_for_cli, including an optional
            config path, the live/dry-run choice, and explicit setting overrides.

    Returns:
        An immutable Config containing the effective values for this invocation.

    Raises:
        OSError: A requested JSON file or an existing .env cannot be read.
        ValueError: JSON contains unknown fields, a selected environment value
            is malformed, or the final configuration fails validation.
    """
    defaults = asdict(Config())
    values = defaults.copy()
    if arguments.config is not None:
        with arguments.config.open(encoding="utf-8-sig") as file:
            supplied = json.load(file)
        if not isinstance(supplied, dict):
            raise ValueError("The configuration file must contain a JSON object.")
        unknown = set(supplied) - set(values)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}.")
        values.update(supplied)
    file_environment: Mapping[str, str | None] = {}
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        with env_path.open(encoding="utf-8-sig") as file:
            file_environment = dotenv_values(stream=file, interpolate=False)
    for field in fields(Config):
        override = getattr(arguments, field.name)
        if override is not None:
            values[field.name] = override
            continue
        prefix = "AZURE_OPENAI_" if field.name in ("endpoint", "deployment") else "TPM_"
        environment_variable = prefix + field.name.upper()
        if environment_variable in os.environ:
            raw_value = os.environ[environment_variable]
        elif environment_variable in file_environment:
            raw_value = file_environment[environment_variable]
            if raw_value is None:
                raise ValueError(f"{environment_variable} in .env must use NAME=value syntax.")
        else:
            continue
        converter = type(defaults[field.name])
        try:
            values[field.name] = converter(raw_value)
        except ValueError as error:
            raise ValueError(
                f"{environment_variable} must be a valid {converter.__name__}."
            ) from error
    config = Config(**values)
    config.validate(live=arguments.live)
    return config


async def execute_live(
    config: Config,
    plan: Plan,
    results: list[RequestResult],
    on_result: Callable[[RequestResult], None],
) -> None:
    """Open live-run resources, execute the load, and close them afterward.

    This is the model-network path that main enters only with --live.
    DefaultAzureCredential supplies Entra credentials (for example, az login),
    not an API key. The HTTP transport has zero automatic retries and will not
    follow redirects. First use of an encoding may download public tokenizer
    data separately from model inference. Real inference can incur charges.

    Args:
        config: Validated destination, tokenizer, scope, and workload limits.
        plan: Bounded prompt workload to pass to run_load.
        results: Mutable list that receives completed attempts during the run.
        on_result: Synchronous observer used to persist and display each result.
            Errors from this callback are not silently ignored.
    """
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
    """Run the CLI from safe planning through optional live calls and reporting.

    Always print the workload before any live operation. Dry runs stop there.
    Live runs create a new report without overwriting existing files, collect
    observations, and write a summary even after handled failures/interruption.
    No Azure resource or quota-management operation is performed.

    Args:
        argv: CLI arguments without the executable name, such as ["--dry-run"].
            None reads the actual process command line; useful for normal use,
            while an explicit list lets offline tests exercise the same CLI.

    Returns:
        0 for a complete observation run; 1 for failures or missing results;
        3 when --expect-throttle was requested but no 429 occurred; or 130 for
        an interrupted live run. Live calls may already have reached Azure
        before an interrupt, even if their response was not recorded.

    Raises:
        SystemExit: argparse handles --help and invalid arguments/configuration.
            Invalid input exits with code 2 before any model calls are attempted.
    """
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
                """Persist and display an attempt already collected by run_load.

                Args:
                    result: Completed observation to add to the currently open
                        report and print. The enclosing main supplies the report.
                """
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
