import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from azure.core.exceptions import ClientAuthenticationError

import tpm_tester as suite


class CharacterTokenizer:
    """Offline deterministic stand-in; real tokenizer round trips are checked at runtime."""

    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def successful_response():
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 70, "completion_tokens": 1, "total_tokens": 71},
        },
        headers={"x-ratelimit-remaining-tokens": "29929"},
    )


class PlanningTests(unittest.TestCase):
    def test_default_target_exceeds_30k_with_input_alone(self):
        plan = suite.build_plan(suite.Config())
        self.assertEqual(plan.total_prompt_tokens, 45_000)
        self.assertEqual(sum(plan.prompt_tokens_per_request), 45_000)
        self.assertGreater(plan.total_prompt_tokens, 30_000)
        self.assertEqual(plan.request_count, 23)
        self.assertEqual(plan.offered_prompt_tpm, 45_000)
        self.assertTrue(all(64 <= count <= 2_000 for count in plan.prompt_tokens_per_request))

    def test_uneven_target_is_exact_and_output_caps_are_not_counted(self):
        config = suite.Config(target_tokens=45_001, max_output_tokens=10_000)
        plan = suite.build_plan(config)
        self.assertEqual(sum(plan.prompt_tokens_per_request), 45_001)
        self.assertEqual(plan.maximum_generated_tokens, plan.request_count * 10_000)

    def test_shorter_window_changes_rate_not_total_tokens(self):
        plan = suite.build_plan(suite.Config(duration_seconds=30))
        self.assertEqual(plan.total_prompt_tokens, 45_000)
        self.assertEqual(plan.offered_prompt_tpm, 90_000)

    def test_request_ceiling_prevents_large_run(self):
        with self.assertRaisesRegex(ValueError, "exceeding max_requests"):
            suite.build_plan(suite.Config(target_tokens=100_000, max_requests=2))

    def test_unrepresentable_small_target_is_rejected_not_rounded(self):
        with self.assertRaisesRegex(ValueError, "Cannot split"):
            suite.build_plan(suite.Config(target_tokens=65, input_tokens=64))

    def test_invalid_configuration_values(self):
        for changes in (
            {"target_tokens": 0}, {"input_tokens": 63}, {"concurrency": True},
            {"duration_seconds": float("nan")}, {"timeout_seconds": float("inf")},
            {"max_output_tokens": "64"}, {"encoding": "not-an-encoding"},
            {"output_token_parameter": "invented"}, {"deployment": None},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(suite.Config(), **changes).validate(live=False)

    def test_endpoint_normalization(self):
        expected = "https://example.openai.azure.com/openai/v1/chat/completions"
        for endpoint in (
            "https://example.openai.azure.com/",
            "https://example.openai.azure.com/openai/v1/",
            expected,
        ):
            self.assertEqual(suite.Config(endpoint=endpoint).request_url(), expected)

    def test_invalid_endpoints(self):
        for endpoint in (
            "http://example.openai.azure.com",
            "https://user:password@example.openai.azure.com",
            "https://example.openai.azure.com/?api-key=secret",
            "https://example.services.ai.azure.com/api/projects/project",
            "https://example.openai.azure.com/#fragment",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                suite.Config(endpoint=endpoint).validate(live=False)

    def test_live_requires_model_configuration(self):
        with self.assertRaisesRegex(ValueError, "Live tests require"):
            suite.Config().validate(live=True)

    def test_prompts_have_exact_sizes_and_unique_prefixes(self):
        encoder = CharacterTokenizer()
        first = suite.make_prompt(encoder, 128, "run/1")
        second = suite.make_prompt(encoder, 128, "run/2")
        self.assertEqual(len(encoder.encode(first)), 128)
        self.assertEqual(len(encoder.encode(second)), 128)
        self.assertNotEqual(first[:64], second[:64])
        self.assertIn("Reply only OK", first)


class ResponseTests(unittest.TestCase):
    def test_raw_429_body_and_retry_headers_preserved(self):
        body = '{"error":{"code":"429","message":"Token rate limit exceeded. Retry in 42 seconds."}}'
        details = suite.response_details(httpx.Response(
            429,
            text=body,
            headers={
                "retry-after": "42", "retry-after-ms": "42000",
                "x-ratelimit-remaining-tokens": "0", "apim-request-id": "request-123",
                "authorization": "must-not-be-recorded", "set-cookie": "must-not-be-recorded",
            },
        ))
        self.assertEqual(details["outcome"], "throttled")
        self.assertEqual(details["error"], body)
        self.assertEqual(details["headers"]["retry-after-ms"], "42000")
        self.assertNotIn("must-not-be-recorded", json.dumps(details))

    def test_other_http_errors_are_not_classified_as_throttling(self):
        for code in (400, 401, 403, 404, 500, 503):
            with self.subTest(code=code):
                details = suite.response_details(httpx.Response(code, text="original error"))
                self.assertEqual(details["outcome"], "http_error")
                self.assertEqual(details["error"], "original error")

    def test_missing_usage_is_explicit(self):
        details = suite.response_details(httpx.Response(200, json={"choices": [{}]}))
        self.assertEqual(details["outcome"], "success")
        self.assertIn("unknown", details["usage_warning"])
        self.assertNotIn("usage", details)

    def test_invalid_or_partial_usage_does_not_inflate_reported_totals(self):
        for usage in (
            {"prompt_tokens": 70},
            {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 71},
            {"prompt_tokens": 70, "completion_tokens": -1, "total_tokens": 69},
        ):
            with self.subTest(usage=usage):
                details = suite.response_details(httpx.Response(
                    200, json={"choices": [{}], "usage": usage},
                ))
                self.assertIn("usage_warning", details)
                self.assertNotIn("usage", details)

    def test_malformed_success_is_an_error(self):
        for response in (
            httpx.Response(200, text="not JSON"),
            httpx.Response(200, json={"error": "unexpected envelope"}),
            httpx.Response(200, json={"choices": []}),
        ):
            self.assertEqual(suite.response_details(response)["outcome"], "invalid_response")

    def test_success_records_usage_but_not_generated_content(self):
        details = suite.response_details(successful_response())
        self.assertEqual(details["usage"]["total_tokens"], 71)
        self.assertNotIn("choices", details)


class LoadTests(unittest.IsolatedAsyncioTestCase):
    async def run_mock(self, handler, **changes):
        config = replace(
            suite.Config(
                endpoint="https://example.openai.azure.com", deployment="test-deployment",
                target_tokens=256, input_tokens=128, duration_seconds=0.15,
            ),
            **changes,
        )
        plan = suite.build_plan(config)
        results = []
        provider = AsyncMock(return_value="fake-entra-token")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await suite.run_load(
                config, plan, CharacterTokenizer(), client, provider, results,
            )
        return config, plan, results

    async def test_real_request_shape_and_bearer_authentication(self):
        requests = []

        def handler(request):
            requests.append(request)
            return successful_response()

        _, plan, results = await self.run_mock(handler)
        self.assertEqual(len(requests), plan.request_count)
        self.assertEqual(len(results), plan.request_count)
        for request in requests:
            body = json.loads(request.content)
            self.assertEqual(request.headers["authorization"], "Bearer fake-entra-token")
            self.assertEqual(body["model"], "test-deployment")
            self.assertEqual(body["max_completion_tokens"], 64)
            self.assertFalse(body["stream"])
            self.assertEqual(len(body["messages"][0]["content"]), 128)
            self.assertEqual(
                str(request.url), "https://example.openai.azure.com/openai/v1/chat/completions",
            )
        self.assertNotIn("fake-entra-token", json.dumps([asdict(result) for result in results]))

    async def test_throttling_is_not_retried_or_hidden(self):
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(429, text="quota exceeded", headers={"retry-after": "60"})

        _, plan, results = await self.run_mock(handler)
        self.assertEqual(calls, plan.request_count)
        self.assertTrue(all(result.status_code == 429 for result in results))
        summary = suite.summarize(plan, results)
        self.assertEqual(summary.throttled, plan.request_count)
        self.assertIsNotNone(summary.first_throttle_seconds)
        self.assertEqual(suite.result_exit_code(summary, expect_throttle=True), 0)

    async def test_legacy_output_parameter_is_configurable(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["max_tokens"], 64)
            self.assertNotIn("max_completion_tokens", body)
            return successful_response()

        await self.run_mock(handler, output_token_parameter="max_tokens")

    async def test_transport_failure_is_recorded_and_fails_run(self):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        _, plan, results = await self.run_mock(handler)
        self.assertTrue(all(result.outcome == "transport_error" for result in results))
        self.assertIn("connection refused", results[0].error)
        self.assertEqual(suite.result_exit_code(suite.summarize(plan, results), expect_throttle=False), 1)

    async def test_total_request_timeout_is_visible(self):
        async def handler(request):
            await asyncio.sleep(0.2)
            return successful_response()

        _, _, results = await self.run_mock(handler, timeout_seconds=0.01)
        self.assertTrue(all(result.outcome == "transport_error" for result in results))
        self.assertIn("Total request timeout", results[0].error)

    async def test_dispatch_deadline_marks_unfinished_workload_as_failure(self):
        async def handler(request):
            await asyncio.sleep(0.25)
            return successful_response()

        _, plan, results = await self.run_mock(handler, concurrency=1)
        summary = suite.summarize(plan, results)
        self.assertEqual(summary.recorded_requests, 1)
        self.assertEqual(summary.not_completed_requests, 1)
        self.assertEqual(suite.result_exit_code(summary, expect_throttle=False), 1)

    async def test_concurrency_is_bounded(self):
        active = peak = 0
        sleep = asyncio.sleep

        async def yield_without_pacing(delay):
            await sleep(0)

        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await sleep(0)
            active -= 1
            return successful_response()

        for concurrency in (1, 2):
            with self.subTest(concurrency=concurrency):
                active = peak = 0
                with patch.object(suite.asyncio, "sleep", yield_without_pacing):
                    _, _, results = await self.run_mock(
                        handler, concurrency=concurrency, duration_seconds=60,
                    )
                self.assertEqual(peak, concurrency)
                self.assertEqual(len(results), 2)

    async def test_authentication_fails_before_sending_requests(self):
        provider = AsyncMock(side_effect=ClientAuthenticationError("Sign in with az login."))
        handler = AsyncMock(return_value=successful_response())
        config = suite.Config()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ClientAuthenticationError):
                await suite.run_load(
                    config, suite.build_plan(config), CharacterTokenizer(), client, provider, [],
                )
        handler.assert_not_called()

    async def test_live_entrypoint_wires_entra_scope_and_disables_retries(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(429, text="original throttling response")

        config = suite.Config(
            endpoint="https://example.openai.azure.com", deployment="test",
            target_tokens=128, input_tokens=128,
        )
        plan = suite.build_plan(config)
        results = []
        credential = AsyncMock()
        credential.__aenter__.return_value = credential
        provider = AsyncMock(return_value="fake-entra-token")
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with (
            patch.object(suite, "DefaultAzureCredential", return_value=credential) as credential_type,
            patch.object(suite, "get_bearer_token_provider", return_value=provider) as provider_factory,
            patch.object(suite.tiktoken, "get_encoding", return_value=CharacterTokenizer()),
            patch.object(suite.httpx, "AsyncHTTPTransport") as transport_type,
            patch.object(suite.httpx, "AsyncClient", return_value=client) as client_type,
        ):
            await suite.execute_live(config, plan, results, lambda result: None)
        credential_type.assert_called_once_with(exclude_interactive_browser_credential=True)
        provider_factory.assert_called_once_with(credential, "https://ai.azure.com/.default")
        transport_type.assert_called_once_with(retries=0)
        self.assertFalse(client_type.call_args.kwargs["follow_redirects"])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["authorization"], "Bearer fake-entra-token")
        self.assertEqual(results[0].error, "original throttling response")
        self.assertTrue(client.is_closed)


class SummaryAndCliTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(contextlib.chdir(directory))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def result(self, number, started, **changes):
        values = {
            "request_number": number, "scheduled_seconds": started, "started_seconds": started,
            "latency_seconds": 0.1, "prompt_text_tokens": 128, "max_output_tokens": 64,
            "outcome": "success", "status_code": 200, "headers": {},
        }
        values.update(changes)
        return suite.RequestResult(**values)

    def test_peak_60s_window_does_not_count_boundary_twice(self):
        plan = suite.build_plan(suite.Config(target_tokens=384, input_tokens=128))
        results = [self.result(1, 0), self.result(2, 59), self.result(3, 60)]
        summary = suite.summarize(plan, results)
        self.assertEqual(summary.peak_60s_offered_prompt_tokens, 256)
        self.assertEqual(summary.usage_missing_requests, 3)
        self.assertIsNone(summary.reported_usage)

    def test_missing_expected_throttle_has_distinct_exit_status(self):
        plan = suite.build_plan(suite.Config(target_tokens=128, input_tokens=128))
        summary = suite.summarize(plan, [self.result(1, 0)])
        self.assertEqual(suite.result_exit_code(summary, expect_throttle=True), 3)
        self.assertEqual(suite.result_exit_code(summary, expect_throttle=False), 0)

    def test_dry_run_never_executes_or_acquires_credentials(self):
        with patch.object(suite, "execute_live") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(suite.main(["--dry-run"]), 0)
        execute.assert_not_called()

    def test_configuration_precedence_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"deployment": "file-value", "target_tokens": 40_000}))
            arguments = suite.parser_for_cli().parse_args([
                "--config", str(path), "--deployment", "cli-value",
            ])
            with patch.dict(os.environ, {"AZURE_OPENAI_DEPLOYMENT": "environment-value"}):
                config = suite.load_config(arguments)
            self.assertEqual(config.deployment, "cli-value")
            self.assertEqual(config.target_tokens, 40_000)
            path.write_text('{"unknown_option": 1}')
            with self.assertRaisesRegex(ValueError, "Unknown configuration fields"):
                suite.load_config(arguments)

    def test_environment_file_provides_every_setting_without_modifying_process_environment(self):
        Path(".env").write_text(
            "\ufeff# Local configuration\n"
            'AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com"\n'
            "AZURE_OPENAI_DEPLOYMENT='env-file-deployment'\n"
            "TPM_QUOTA_TPM=30000\n"
            "TPM_TARGET_TOKENS=60000\n"
            "TPM_DURATION_SECONDS=30.5\n"
            "TPM_INPUT_TOKENS=4000\n"
            "TPM_MAX_OUTPUT_TOKENS=128\n"
            "TPM_CONCURRENCY=2\n"
            "TPM_TIMEOUT_SECONDS=12.5\n"
            "TPM_MAX_REQUESTS=50\n"
            "TPM_ENCODING=cl100k_base\n"
            "TPM_OUTPUT_TOKEN_PARAMETER=max_tokens\n"
            "TPM_TOKEN_SCOPE=https://cognitiveservices.azure.com/.default\n",
            encoding="utf-8",
        )
        arguments = suite.parser_for_cli().parse_args(["--live"])
        config = suite.load_config(arguments)
        self.assertEqual(config, suite.Config(
            endpoint="https://example.openai.azure.com",
            deployment="env-file-deployment",
            quota_tpm=30000,
            target_tokens=60000,
            duration_seconds=30.5,
            input_tokens=4000,
            max_output_tokens=128,
            concurrency=2,
            timeout_seconds=12.5,
            max_requests=50,
            encoding="cl100k_base",
            output_token_parameter="max_tokens",
            token_scope="https://cognitiveservices.azure.com/.default",
        ))
        self.assertNotIn("AZURE_OPENAI_ENDPOINT", os.environ)
        self.assertNotIn("TPM_TARGET_TOKENS", os.environ)

    def test_cli_process_environment_dotenv_and_json_precedence(self):
        Path(".env").write_text(
            "AZURE_OPENAI_DEPLOYMENT=dotenv-deployment\nTPM_TARGET_TOKENS=45000\n",
        )
        Path("config.json").write_text(json.dumps({
            "deployment": "json-deployment", "target_tokens": 20000, "concurrency": 2,
        }))
        arguments = suite.parser_for_cli().parse_args(["--config", "config.json"])
        config = suite.load_config(arguments)
        self.assertEqual(config.deployment, "dotenv-deployment")
        self.assertEqual(config.target_tokens, 45000)
        self.assertEqual(config.concurrency, 2)
        with patch.dict(os.environ, {"TPM_TARGET_TOKENS": "60000"}):
            self.assertEqual(suite.load_config(arguments).target_tokens, 60000)
            arguments = suite.parser_for_cli().parse_args([
                "--config", "config.json", "--target-tokens", "75000",
            ])
            self.assertEqual(suite.load_config(arguments).target_tokens, 75000)

    def test_invalid_numeric_environment_values_fail_explicitly(self):
        arguments = suite.parser_for_cli().parse_args([])
        for value in ("not-a-number", "", "45.5"):
            with self.subTest(value=value), patch.dict(os.environ, {"TPM_TARGET_TOKENS": value}):
                with self.assertRaisesRegex(ValueError, "TPM_TARGET_TOKENS must be a valid int"):
                    suite.load_config(arguments)
        Path(".env").write_text("TPM_TARGET_TOKENS=not-a-number\n")
        with self.assertRaisesRegex(ValueError, "TPM_TARGET_TOKENS must be a valid int"):
            suite.load_config(arguments)
        override = suite.parser_for_cli().parse_args(["--target-tokens", "60000"])
        self.assertEqual(suite.load_config(override).target_tokens, 60000)

    def test_valueless_environment_file_setting_is_rejected(self):
        Path(".env").write_text("TPM_TARGET_TOKENS\n")
        arguments = suite.parser_for_cli().parse_args([])
        with self.assertRaisesRegex(ValueError, "TPM_TARGET_TOKENS in .env must use NAME=value"):
            suite.load_config(arguments)

    def test_unreadable_environment_path_is_a_cli_error(self):
        Path(".env").mkdir()
        with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as result:
            suite.main(["--dry-run"])
        self.assertEqual(result.exception.code, 2)
        self.assertIn(".env", errors.getvalue())

    def test_dry_run_uses_environment_file_without_model_calls(self):
        Path(".env").write_text("TPM_TARGET_TOKENS=60000\n")
        with (
            patch.object(suite, "execute_live") as execute,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(suite.main(["--dry-run"]), 0)
        execute.assert_not_called()
        self.assertIn("60,000 offered prompt-text tokens", output.getvalue())

    def test_environment_example_covers_all_configuration_fields(self):
        template = Path(suite.__file__).with_name(".env.example").read_text(encoding="utf-8")
        Path(".env").write_text(template, encoding="utf-8")
        config = suite.load_config(suite.parser_for_cli().parse_args(["--live"]))
        expected = replace(
            suite.Config(),
            endpoint="https://YOUR-RESOURCE.openai.azure.com",
            deployment="YOUR-DEPLOYMENT-NAME",
        )
        self.assertEqual(config, expected)
        file_values = suite.dotenv_values(".env")
        self.assertEqual(len(file_values), len(asdict(config)))

    def test_jsonl_report_preserves_raw_errors_and_partial_progress(self):
        report = io.StringIO()
        body = '{"error": "quota exceeded"}\nsecond line'
        result = self.result(1, 0, outcome="throttled", status_code=429, error=body)
        suite.write_event(report, "request", asdict(result))
        records = [json.loads(line) for line in report.getvalue().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["error"], body)
        self.assertEqual(records[0]["type"], "request")

    def test_live_report_contains_plan_requests_and_summary(self):
        async def execute(config, plan, results, on_result):
            result = self.result(
                1, 0, outcome="throttled", status_code=429, error="original quota error",
            )
            results.append(result)
            on_result(result)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.jsonl"
            argv = [
                "--live", "--endpoint", "https://example.openai.azure.com",
                "--deployment", "test", "--target-tokens", "128", "--output", str(output),
                "--expect-throttle",
            ]
            with (
                patch.object(suite, "execute_live", execute),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(suite.main(argv), 0)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([record["type"] for record in records], ["plan", "request", "summary"])
            self.assertEqual(records[1]["error"], "original quota error")
            self.assertEqual(records[-1]["throttled"], 1)
            with (
                patch.object(suite, "execute_live") as again,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(suite.main(argv), 1)
            again.assert_not_called()

    def test_fatal_authentication_error_leaves_a_report(self):
        async def execute(config, plan, results, on_result):
            raise ClientAuthenticationError("Sign in with az login.")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed-run.jsonl"
            with (
                patch.object(suite, "execute_live", execute),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = suite.main([
                    "--live", "--endpoint", "https://example.openai.azure.com",
                    "--deployment", "test", "--target-tokens", "128", "--output", str(output),
                ])
            self.assertEqual(code, 1)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([record["type"] for record in records], ["plan", "fatal", "summary"])
            self.assertIn("Sign in with az login", records[1]["error"])
            self.assertEqual(records[-1]["not_completed_requests"], 1)


if __name__ == "__main__":
    unittest.main()
