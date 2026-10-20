"""Pin the gateway_hint contract: an additive note beside the upstream error.

The hint must never replace or rewrite the upstream message (clients keep
parsing the same envelope), must only appear when the gateway actually
classified the failure, and unknown failures must stay hint-less rather than
guess. Run with the bundled Python:

    python tests/_test_gateway_hint.py
"""
import io
import json
import os
import sys
import types
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _isolated_dirs import isolated_data_dirs

_TMP = isolated_data_dirs("wb-context-errors-")

import wb_proxy as proxy


class HintClassificationTests(unittest.TestCase):
    def test_known_shapes_get_a_hint(self):
        cases = [
            (503, "concurrent chat limit reached", "concurrency"),
            (400, "11133 model_param_invalid: bad field", "parameters"),
            (400, "11135 invalid_image_data", "image"),
            (503, "no usable account for realm 'intl'", "no healthy account"),
            (400, "maximum context length exceeded", "context"),
            (429, "rate limit reached for model", "rate limited"),
            (402, "insufficient credit balance", "credits"),
            (401, "session not found, please login", "session expired"),
            (400, "content rejected by policy", "content policy"),
            (404, "no such model: foo-bar", "no such model"),
            (403, "request blocked by WAF", "WAF"),
        ]
        for status, message, needle in cases:
            hint = proxy.gateway_hint(status, message)
            self.assertIn(needle, hint, (status, message, hint))

    def test_unknown_failures_get_no_hint(self):
        for status, message in ((500, "internal error"), (400, ""),
                                (400, "something we have never seen")):
            self.assertEqual("", proxy.gateway_hint(status, message),
                             (status, message))

    def test_hint_never_replaces_the_message(self):
        # The hint is additive; the upstream wording stays verbatim.
        message = "rate limit reached for model deepseek-v4.1-flash"
        self.assertEqual(message, str(message))
        hint = proxy.gateway_hint(429, message)
        self.assertTrue(hint)
        self.assertNotIn(hint, message)


class _Stub(object):
    _error = proxy.Handler._error
    _handle_responses = proxy.Handler._handle_responses
    _dispatch_chat_post = proxy.Handler._dispatch_chat_post

    def __init__(self):
        self.status = None
        self.payload = None
        self.headers = {}

    def _request_realm(self):
        return "intl"

    def _cross_realm_error(self, *_):
        return ""

    _banned_model_error = _cross_realm_error
    _key_model_error = _cross_realm_error

    def _key_id(self):
        return None

    def _handle_expect_continue(self):
        pass

    def _discard_body(self):
        pass

    def _json(self, code, obj):
        self.status = code
        self.payload = obj
        return code, obj


class ErrorEnvelopeTests(unittest.TestCase):
    def test_envelope_carries_the_hint_when_present(self):
        stub = _Stub()
        stub._error(429, "rate limit reached for model x", "rate_limit_error")
        self.assertEqual(429, stub.status)
        error = stub.payload["error"]
        self.assertEqual("rate limit reached for model x", error["message"])
        self.assertEqual("rate_limit_error", error["type"])
        self.assertEqual(429, error["code"])
        self.assertEqual("rate limited by upstream; retry after reset",
                         error["gateway_hint"])

    def test_envelope_omits_the_hint_when_unknown(self):
        stub = _Stub()
        stub._error(500, "internal error")
        error = stub.payload["error"]
        self.assertNotIn("gateway_hint", error)
        self.assertEqual("internal error", error["message"])

    def test_semantic_code_does_not_change_http_status_or_message(self):
        stub = _Stub()
        stub._error(400, "original", "invalid_request_error",
                    error_code="context_length_exceeded")
        self.assertEqual(400, stub.status)
        self.assertEqual({"message": "original", "type": "invalid_request_error",
                          "code": "context_length_exceeded"}, stub.payload["error"])


class ContextErrorTests(unittest.TestCase):
    def reply(self, route, body, status=400, enabled="1", stream=True):
        stub = _Stub()
        failure = urllib.error.HTTPError("https://up", status, "Bad Request", {},
                                         io.BytesIO(body))
        payload = {"model": "deepseek-v4.1-flash", "stream": stream,
                   "input": "hello", "messages": [{"role": "user", "content": "hello"}]}
        with mock.patch.dict(os.environ, {"WB_CONTEXT_ERROR_NORMALIZE": enabled}), \
                mock.patch.object(proxy, "open_upstream", side_effect=failure) as upstream, \
                mock.patch.object(proxy, "record_error") as recorded, \
                mock.patch.object(proxy, "log"), \
                mock.patch.object(proxy, "BLOCK_BACKGROUND_REQUESTS", False):
            stub._dispatch_chat_post(route, payload)
        failure.close()
        self.assertEqual(1, upstream.call_count)
        self.assertEqual(body[:600].decode("utf-8", "replace"), recorded.call_args[0][2])
        self.assertEqual("upstream %d: %s" % (status, body[:600].decode("utf-8", "replace")),
                         stub.payload["error"]["message"])
        return stub

    def test_routes_normalize_the_observed_error_only_when_enabled(self):
        body = b'{"code":11115,"msg":"input exceeds limit"}'
        for route in ("/v1/responses", "/v1/chat/completions"):
            for stream in (True, False):
                for enabled in ("", "0", "1"):
                    with self.subTest(route=route, stream=stream, enabled=enabled):
                        stub = self.reply(route, body, enabled=enabled, stream=stream)
                        self.assertEqual(400, stub.status)
                        self.assertEqual("context_length_exceeded" if enabled == "1" else 400,
                                         stub.payload["error"]["code"])
                        self.assertEqual("invalid_request_error" if enabled == "1" else "server_error",
                                         stub.payload["error"]["type"])

    def test_structural_lookalikes_and_non_400_stay_unchanged(self):
        bodies = [b'{"msg":"11115"}', b'{"code":111150}', b'{"code":11115.9}',
                  b'{"code":true}', b'{"error":{"code":11115}}', b'11115',
                  b'not JSON 11115', b'{"code":11115', b'{"code":400}']
        for route in ("/v1/responses", "/v1/chat/completions"):
            for body in bodies:
                with self.subTest(route=route, body=body):
                    stub = self.reply(route, body)
                    self.assertEqual(400, stub.payload["error"]["code"])
                    self.assertEqual("server_error", stub.payload["error"]["type"])
            stub = self.reply(route, b'{"code":11115}', status=500)
            self.assertEqual(500, stub.status)
            self.assertEqual(500, stub.payload["error"]["code"])

    def test_larger_body_is_inspected_without_expanding_display(self):
        body = json.dumps({"msg": "x" * 700, "code": "11115"}).encode()
        for route in ("/v1/responses", "/v1/chat/completions"):
            stub = self.reply(route, body)
            self.assertEqual("context_length_exceeded", stub.payload["error"]["code"])
            oversized = b'{"code":11115,"msg":"' + b'x' * 9000 + b'"}'
            stub = self.reply(route, oversized)
            self.assertEqual(400, stub.payload["error"]["code"])

    def test_hint_requires_the_structured_code(self):
        with mock.patch.dict(os.environ, {"WB_CONTEXT_ERROR_NORMALIZE": "1"}):
            for body in ('{"code":11115}', 'upstream 400: {"code":"11115"}'):
                self.assertIn("context", proxy.gateway_hint(400, body))
            for body in ('{"msg":"11115"}', '11115', '{"code":111150}',
                         '{"code":11115.9}', '{"error":{"code":11115}}'):
                self.assertEqual("", proxy.gateway_hint(400, body))
            self.assertEqual("", proxy.gateway_hint(500, '{"code":11115}'))

    def test_context_400_does_not_retry_or_penalize_the_account(self):
        account = proxy.wb_accounts.Account({"uid": "synthetic-context-account",
                                            "accessToken": "x.y.z", "enabled": True,
                                            "realm": "intl"})
        pool = types.SimpleNamespace(
            accounts=[account], count_ready=lambda *a, **kw: 1,
            apply_remaining_weights=lambda *a: None,
            pick_for_session=lambda *a, **kw: account,
            affinity=types.SimpleNamespace(demote=mock.Mock()))
        failure = urllib.error.HTTPError("https://up", 400, "Bad Request", {},
                                         io.BytesIO(b'{"code":11115,"msg":"too long"}'))
        with mock.patch.object(proxy, "POOL", pool), \
                mock.patch.object(proxy, "apply_daily_guards"), \
                mock.patch.object(proxy.wb_accounts, "urlopen", side_effect=failure) as upstream:
            with self.assertRaises(urllib.error.HTTPError):
                proxy.open_upstream({"model": "deepseek-v4.1-flash", "messages": [
                    {"role": "user", "content": "hello"}]}, target_realm="intl")
        failure.close()
        self.assertEqual(1, upstream.call_count)
        self.assertEqual(0, account.fails)
        self.assertEqual(0, account.degrade_count)
        self.assertEqual(0, account.throttle_wait())
        pool.affinity.demote.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
