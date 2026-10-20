"""Unit tests for per-key concurrency limits (#359)."""
import unittest
import ast
import inspect
import sys
import os
import threading
from unittest.mock import Mock, patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wb_settings
import wb_proxy


class PostHandler:
    """Run the real route/gate logic with only the upstream work stubbed."""
    do_POST = wb_proxy.Handler.do_POST

    def __init__(self, path, key=None, dispatch=None):
        self.path = path
        self.key_entry = key
        self.dispatch = dispatch or (lambda: "ok")

    def _apply_pinned_route(self):
        return True

    def _is_panel_route(self, path):
        return False

    def _key_ok(self):
        return True

    _authorized = _key_ok

    def _payload_or_error(self, **kwargs):
        return {"model": "test", "stream": True}

    def _key_id(self):
        return (self.key_entry or {}).get("id")

    def _error(self, code, message, error_type=None):
        return code, error_type

    _anthropic_error = _error

    def _dispatch_chat_post(self, path, payload):
        return self.dispatch()

    def _handle_messages(self, payload):
        return self.dispatch()


class KeyConcurrencyTests(unittest.TestCase):
    def test_clean_key_entry_defaults_to_zero(self):
        entry = wb_settings._clean_key_entry({"name": "test", "key": "k-123"})
        self.assertEqual(entry["concurrency_limit"], 0)

        entry2 = wb_settings._clean_key_entry({"name": "test", "key": "k-456", "concurrency_limit": "5"})
        self.assertEqual(entry2["concurrency_limit"], 5)

        entry3 = wb_settings._clean_key_entry({"name": "test", "key": "k-789", "concurrency_limit": "-2"})
        self.assertEqual(entry3["concurrency_limit"], 0)

    def test_acquire_and_release_key_slot(self):
        key = {"id": "key-test-1", "concurrency_limit": 2}
        
        ok1, cur1, lim1 = wb_proxy._acquire_key_slot(key)
        self.assertTrue(ok1)
        self.assertEqual(cur1, 1)

        ok2, cur2, lim2 = wb_proxy._acquire_key_slot(key)
        self.assertTrue(ok2)
        self.assertEqual(cur2, 2)

        ok3, cur3, lim3 = wb_proxy._acquire_key_slot(key)
        self.assertFalse(ok3)

        wb_proxy._release_key_slot(key)
        ok4, cur4, lim4 = wb_proxy._acquire_key_slot(key)
        self.assertTrue(ok4)
        self.assertEqual(cur4, 2)

        wb_proxy._release_key_slot(key)
        wb_proxy._release_key_slot(key)
        self.assertNotIn("key-test-1", wb_proxy._key_active_concurrency)

    def test_unlimited_key_never_blocks(self):
        key = {"id": "key-test-0", "concurrency_limit": 0}
        for _ in range(10):
            ok, cur, lim = wb_proxy._acquire_key_slot(key)
            self.assertTrue(ok)
        wb_proxy._release_key_slot(key)
        self.assertNotIn("key-test-0", wb_proxy._key_active_concurrency)


class GlobalConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(wb_proxy, "_anthropic_messages_enabled", return_value=True),
            patch.object(wb_proxy, "_apply_background_model_alias"),
            patch.object(wb_proxy, "record_error"),
            patch.object(wb_proxy, "CHAT_SLOT_WAIT_SECONDS", 0),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_slot_environment_defaults_zero_auto_and_negative(self):
        # Execute only the actual import-time configuration statements; avoid
        # importing a second gateway or touching any deployment state.
        tree = ast.parse(inspect.getsource(wb_proxy))
        start = next(i for i, n in enumerate(tree.body)
                     if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_CHAT_SLOTS_ENV"
                             for t in n.targets))
        finish = next(i for i, n in enumerate(tree.body[start:], start)
                      if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "CHAT_SLOT_WAIT_SECONDS"
                              for t in n.targets))
        code = compile(ast.Module(body=tree.body[start:finish], type_ignores=[]),
                       "<global slot configuration>", "exec")
        for value, maximum, automatic in ((None, 32, False), ("0", 0, False),
                                          ("-1", 32, False), ("bad", 32, False),
                                          ("auto", 32, True), ("7", 7, False)):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop("WB_MAX_CONCURRENT_CHAT", None)
                else:
                    os.environ["WB_MAX_CONCURRENT_CHAT"] = value
                namespace = {"os": os}
                exec(code, namespace)
                self.assertEqual(namespace["MAX_CONCURRENT_CHAT"], maximum)
                self.assertEqual(namespace["CHAT_SLOTS_AUTO"], automatic)
                threading.BoundedSemaphore(namespace["MAX_CONCURRENT_CHAT"])

    def test_unlimited_global_never_touches_gate_for_both_protocols(self):
        gate = Mock()
        gate.acquire.side_effect = AssertionError("unlimited mode acquired gate")
        gate.release.side_effect = AssertionError("unlimited mode released gate")
        with patch.object(wb_proxy, "MAX_CONCURRENT_CHAT", 0), \
                patch.object(wb_proxy, "_chat_slots", gate):
            for path in ("/v1/responses", "/v1/chat/completions", "/v1/messages"):
                with self.subTest(path=path):
                    self.assertEqual(PostHandler(path).do_POST(), "ok")
            gate.acquire.assert_not_called()
            gate.release.assert_not_called()

    def test_finite_saturation_returns_503_without_releasing_unheld_gate(self):
        for path in ("/v1/responses", "/v1/messages"):
            gate = Mock()
            gate.acquire.return_value = False
            dispatch = Mock()
            with self.subTest(path=path), patch.object(wb_proxy, "MAX_CONCURRENT_CHAT", 1), \
                    patch.object(wb_proxy, "_chat_slots", gate):
                result = PostHandler(path, dispatch=dispatch).do_POST()
                self.assertEqual(result[0], 503)
                gate.acquire.assert_called_once_with(timeout=0)
                gate.release.assert_not_called()
                dispatch.assert_not_called()

    def test_acquired_gate_and_key_are_released_after_success_or_exception(self):
        for path in ("/v1/responses", "/v1/messages"):
            for fail in (False, True):
                key = {"id": "global-release-test", "concurrency_limit": 1}
                gate = threading.BoundedSemaphore(1)
                replacement = Mock()

                def dispatch():
                    # Replacing the global cannot change which semaphore this
                    # request must release when its work finishes.
                    wb_proxy._chat_slots = replacement
                    if fail:
                        raise ValueError("upstream failed")
                    return "finished"

                with self.subTest(path=path, fail=fail), \
                        patch.object(wb_proxy, "MAX_CONCURRENT_CHAT", 1), \
                        patch.object(wb_proxy, "_chat_slots", gate):
                    handler = PostHandler(path, key=key, dispatch=dispatch)
                    if fail:
                        with self.assertRaisesRegex(ValueError, "upstream failed"):
                            handler.do_POST()
                    else:
                        self.assertEqual(handler.do_POST(), "finished")
                    self.assertTrue(gate.acquire(blocking=False))
                    gate.release()
                    replacement.release.assert_not_called()
                    self.assertNotIn(key["id"], wb_proxy._key_active_concurrency)

    def test_unlimited_global_retains_independent_key_limits(self):
        first = {"id": "global-off-first", "concurrency_limit": 1}
        second = {"id": "global-off-second", "concurrency_limit": 1}
        for path in ("/v1/responses", "/v1/messages"):
            self.assertTrue(wb_proxy._acquire_key_slot(first)[0])
            try:
                with self.subTest(path=path), patch.object(wb_proxy, "MAX_CONCURRENT_CHAT", 0):
                    self.assertEqual(PostHandler(path, key=first).do_POST()[0], 429)
                    self.assertEqual(PostHandler(path, key=second).do_POST(), "ok")
                    self.assertNotIn(second["id"], wb_proxy._key_active_concurrency)
            finally:
                wb_proxy._release_key_slot(first)

if __name__ == "__main__":
    unittest.main()
