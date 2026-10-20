"""Affinity length cap, pool pages, and pool-sized chat concurrency.

Two independent limits that both used to be fixed constants:

* ``derive_affinity_key`` pinned every conversation to one upstream account so
  the account-level prompt cache keeps hitting. That is right for normal
  conversations, but a conversation's body grows monotonically, and past a few
  hundred messages the oversized request makes the upstream drop the
  connection. ``WB_AFFINITY_MAX_MSGS`` releases the pin once a conversation is
  that long, trading its prefix cache for not being retried.

  What it is released *into* is ``WB_AFFINITY_PAGE_SIZE``: 0 sends the
  conversation back to the whole pool (the 1.6.x behaviour), 1 keeps it pinned
  (the cap switched off), and anything higher rotates it over one page of the
  pool so every account of that page keeps a warm copy of the prefix. The page
  tests below pin the boundary; the pool side is in ``_test_affinity_pages.py``.

* ``MAX_CONCURRENT_CHAT`` was a fixed 32 regardless of pool size.
  ``WB_MAX_CONCURRENT_CHAT=auto`` sizes it from the ready-account count.

Run with: python _test_affinity_length_cap.py
No upstream credentials or outbound network are used.
"""
import os
import sys
import tempfile
import threading
import unittest

_startup_dir = tempfile.TemporaryDirectory(prefix="affinity-cap-")
os.environ["ACCOUNTS_DIR"] = _startup_dir.name
os.environ["WB_PROXY_USAGE_DIR"] = _startup_dir.name
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wb_proxy


class FlatUserAffinityTests(unittest.TestCase):
    """Opt-in stable routing for flattened, growing single-user prompts."""

    def setUp(self):
        self._saved_prefix_chars = wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS

    def tearDown(self):
        wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS = self._saved_prefix_chars

    @staticmethod
    def messages(content):
        return [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": content},
        ]

    def legacy_key(self, messages):
        blob = wb_proxy.json.dumps(
            messages[:2], ensure_ascii=False, sort_keys=True).encode("utf-8")
        return "pfx-" + wb_proxy.hashlib.sha256(blob).hexdigest()[:16]

    def test_default_off_preserves_the_legacy_key(self):
        wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS = 0
        messages = self.messages("opening" + "x" * 40)
        self.assertEqual(self.legacy_key(messages),
                         wb_proxy.derive_affinity_key(messages))

    def test_appended_tail_keeps_the_key_after_the_gate(self):
        wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS = 8
        first = self.messages("opening! first history")
        later = self.messages("opening! first history plus later turns")
        self.assertEqual(wb_proxy.derive_affinity_key(first),
                         wb_proxy.derive_affinity_key(later))

    def test_prefix_changes_and_distinct_openings_get_distinct_keys(self):
        wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS = 8
        long_tail = "x" * 20
        a = self.messages("opening!" + long_tail)
        changed_prefix = self.messages("openinG!" + long_tail)
        separate = self.messages("separate" + long_tail)
        self.assertNotEqual(wb_proxy.derive_affinity_key(a),
                            wb_proxy.derive_affinity_key(changed_prefix))
        self.assertNotEqual(wb_proxy.derive_affinity_key(a),
                            wb_proxy.derive_affinity_key(separate))

    def test_short_prompt_and_other_shapes_keep_the_legacy_hash(self):
        wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS = 8
        short = self.messages("12345678")
        self.assertEqual(self.legacy_key(short), wb_proxy.derive_affinity_key(short))
        other_shapes = [
            [{"role": "user", "content": "x" * 30}],
            [{"role": "system", "content": "s"},
             {"role": "assistant", "content": "x" * 30}],
            self.messages("x" * 30) + [{"role": "assistant", "content": "y"}],
            # No text to hash: an image-only turn stays on the legacy key.
            [{"role": "system", "content": "s"},
             {"role": "user", "content": [{"type": "image_url",
                                           "image_url": {"url": "data:image/png;base64,AA"}}]}],
        ]
        for messages in other_shapes:
            with self.subTest(messages=messages):
                self.assertEqual(self.legacy_key(messages),
                                 wb_proxy.derive_affinity_key(messages))

    def test_content_parts_are_flat_prompts_too(self):
        """A user message in OpenAI/Responses parts must take the same fallback.

        Several SDKs switch a long prompt from `content: "..."` to parts. While
        the fallback tested `isinstance(content, str)` those turns fell back to
        hashing the whole ever-growing message, so every turn derived a new key
        and the pool rotated the conversation away - 5 identical requests over
        5 accounts with `cached_tokens: 0` each, measured on the live gateway.
        """
        wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS = 8
        as_string = self.messages("opening!" + "x" * 40)
        as_parts = self.messages([{"type": "text", "text": "opening!" + "x" * 40}])
        self.assertEqual(wb_proxy.derive_affinity_key(as_string),
                         wb_proxy.derive_affinity_key(as_parts))
        # The fallback has to apply to parts too, not just hash them whole.
        later = self.messages([{"type": "text", "text": "opening!" + "x" * 40 + " more turns"}])
        self.assertEqual(wb_proxy.derive_affinity_key(as_parts),
                         wb_proxy.derive_affinity_key(later))

    def test_parts_text_is_joined_in_order(self):
        self.assertEqual("abcdef",
                         wb_proxy._affinity_user_text(
                             [{"type": "text", "text": "abc"}, "def"]))
        self.assertEqual("", wb_proxy._affinity_user_text(None))
        self.assertEqual("", wb_proxy._affinity_user_text({"not": "a list"}))

    def test_prefix_fallback_does_not_mutate_payload(self):
        wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS = 8
        messages = self.messages("opening!" + "x" * 30)
        original = [dict(message) for message in messages]
        wb_proxy.derive_affinity_key(messages)
        self.assertEqual(original, messages)

    def test_explicit_session_key_bypasses_derived_key(self):
        wb_proxy.AFFINITY_FLAT_USER_PREFIX_CHARS = 8
        key, page = wb_proxy.affinity_route(
            self.messages("opening!" + "x" * 30),
            explicit_session_key="client-session")
        self.assertEqual(("client-session", None), (key, page))

    def test_nonpositive_and_malformed_settings_disable_fallback(self):
        self.assertEqual(0, wb_proxy._parse_nonnegative_int("-12"))
        self.assertEqual(0, wb_proxy._parse_nonnegative_int("bad"))
        self.assertEqual(0, wb_proxy._parse_nonnegative_int(None))
        self.assertEqual(7, wb_proxy._parse_nonnegative_int("bad", default=7))


def conversation(length):
    """A conversation of exactly `length` messages, stable prefix first."""
    msgs = [{"role": "system", "content": "you are a helpful assistant"}]
    for i in range(max(0, length - 1)):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                     "content": "turn %d" % i})
    return msgs


class AffinityLengthCapTests(unittest.TestCase):
    """The cap must only touch conversations past the threshold."""

    def setUp(self):
        self._saved = wb_proxy.AFFINITY_MAX_MSGS
        self._saved_page = wb_proxy.AFFINITY_PAGE_SIZE
        # The 1.6.x behaviour the release tests pin: no page, back to the pool.
        wb_proxy.AFFINITY_PAGE_SIZE = 0

    def tearDown(self):
        wb_proxy.AFFINITY_MAX_MSGS = self._saved
        wb_proxy.AFFINITY_PAGE_SIZE = self._saved_page

    def test_short_conversations_still_bind(self):
        wb_proxy.AFFINITY_MAX_MSGS = 400
        for length in (1, 2, 10, 399, 400):
            key = wb_proxy.derive_affinity_key(conversation(length))
            self.assertTrue(key, "msgs=%d should keep its affinity key" % length)
            self.assertTrue(key.startswith("pfx-"))

    def test_long_conversations_are_released_when_the_page_is_off(self):
        wb_proxy.AFFINITY_MAX_MSGS = 400
        for length in (401, 500, 1245):
            self.assertIsNone(
                wb_proxy.derive_affinity_key(conversation(length)),
                "msgs=%d should be released so the pool can rotate" % length)

    def test_threshold_is_inclusive(self):
        """msgs == cap binds; the cap is a limit, not an off-by-one."""
        wb_proxy.AFFINITY_MAX_MSGS = 3
        self.assertIsNotNone(wb_proxy.derive_affinity_key(conversation(3)))
        self.assertIsNone(wb_proxy.derive_affinity_key(conversation(4)))

    def test_zero_disables_the_cap(self):
        """0 keeps the pre-1.6.x behaviour: always bind."""
        wb_proxy.AFFINITY_MAX_MSGS = 0
        for length in (1, 400, 5000):
            self.assertIsNotNone(wb_proxy.derive_affinity_key(conversation(length)))

    def test_bound_conversations_keep_a_stable_key(self):
        """The whole point: later turns of one conversation share one account."""
        wb_proxy.AFFINITY_MAX_MSGS = 400
        first = wb_proxy.derive_affinity_key(conversation(10))
        later = wb_proxy.derive_affinity_key(conversation(300))
        self.assertEqual(first, later)

    def test_distinct_conversations_do_not_collide(self):
        wb_proxy.AFFINITY_MAX_MSGS = 400
        a = conversation(10)
        b = conversation(10)
        b[1] = {"role": "user", "content": "a different opening turn"}
        self.assertNotEqual(wb_proxy.derive_affinity_key(a),
                            wb_proxy.derive_affinity_key(b))

    def test_degenerate_inputs_are_safe(self):
        wb_proxy.AFFINITY_MAX_MSGS = 400
        self.assertIsNone(wb_proxy.derive_affinity_key(None))
        self.assertIsNone(wb_proxy.derive_affinity_key([]))


class AffinityPageTests(unittest.TestCase):
    """Past the cap a conversation rotates over one page, not the whole pool."""

    def setUp(self):
        self._saved_msgs = wb_proxy.AFFINITY_MAX_MSGS
        self._saved_page = wb_proxy.AFFINITY_PAGE_SIZE
        self._saved_prefix = wb_proxy.AFFINITY_BY_PREFIX
        wb_proxy.AFFINITY_MAX_MSGS = 400

    def tearDown(self):
        wb_proxy.AFFINITY_MAX_MSGS = self._saved_msgs
        wb_proxy.AFFINITY_PAGE_SIZE = self._saved_page
        wb_proxy.AFFINITY_BY_PREFIX = self._saved_prefix

    def test_a_long_conversation_gets_a_key_and_a_page(self):
        wb_proxy.AFFINITY_PAGE_SIZE = 3
        key, page = wb_proxy.affinity_route(conversation(500))
        self.assertTrue(key and key.startswith("pfx-"))
        self.assertIsNotNone(page, "500 msgs must rotate over a page")
        index, size = page
        self.assertEqual(size, 3)
        self.assertTrue(0 <= index < size, "page index out of range: %r" % (page,))

    def test_the_key_survives_crossing_the_cap(self):
        """One conversation, one key: the binding carries over the boundary."""
        wb_proxy.AFFINITY_PAGE_SIZE = 3
        short = wb_proxy.derive_affinity_key(conversation(400))
        long_key = wb_proxy.derive_affinity_key(conversation(500))
        self.assertEqual(short, long_key)

    def test_the_page_is_stable_across_turns(self):
        wb_proxy.AFFINITY_PAGE_SIZE = 3
        first = wb_proxy.affinity_route(conversation(401))[1]
        later = wb_proxy.affinity_route(conversation(900))[1]
        self.assertEqual(first, later)

    def test_distinct_conversations_spread_over_pages(self):
        wb_proxy.AFFINITY_PAGE_SIZE = 3
        pages = set()
        for index in range(60):
            msgs = conversation(500)
            msgs[1] = {"role": "user", "content": "conversation %d" % index}
            pages.add(wb_proxy.affinity_route(msgs)[1][0])
        self.assertGreater(len(pages), 1,
                           "every conversation landing on one page would be a "
                           "pin with extra steps")

    def test_page_size_one_pins_instead(self):
        wb_proxy.AFFINITY_PAGE_SIZE = 1
        key, page = wb_proxy.affinity_route(conversation(500))
        self.assertTrue(key, "1 means pinned, so the conversation keeps its key")
        self.assertIsNone(page)

    def test_page_size_zero_keeps_the_pool_rotation(self):
        wb_proxy.AFFINITY_PAGE_SIZE = 0
        key, page = wb_proxy.affinity_route(conversation(500))
        self.assertIsNone(key)
        self.assertIsNone(page)

    def test_the_route_and_the_key_agree(self):
        wb_proxy.AFFINITY_PAGE_SIZE = 3
        for length in (10, 400, 500):
            msgs = conversation(length)
            self.assertEqual(wb_proxy.derive_affinity_key(msgs),
                             wb_proxy.affinity_route(msgs)[0])

    def test_switching_prefix_affinity_off_releases_everything(self):
        wb_proxy.AFFINITY_PAGE_SIZE = 3
        wb_proxy.AFFINITY_BY_PREFIX = False
        for length in (10, 500):
            self.assertEqual(wb_proxy.affinity_route(conversation(length)),
                             (None, None))

    def test_degenerate_inputs_are_safe(self):
        wb_proxy.AFFINITY_PAGE_SIZE = 3
        for msgs in (None, []):
            self.assertEqual(wb_proxy.affinity_route(msgs), (None, None))


class ChatSlotResizeTests(unittest.TestCase):
    """resize_chat_slots grows the ceiling, never shrinks it."""

    def setUp(self):
        self._auto = wb_proxy.CHAT_SLOTS_AUTO
        self._max = wb_proxy.MAX_CONCURRENT_CHAT
        self._slots = wb_proxy._chat_slots

    def tearDown(self):
        wb_proxy.CHAT_SLOTS_AUTO = self._auto
        wb_proxy.MAX_CONCURRENT_CHAT = self._max
        wb_proxy._chat_slots = self._slots

    def test_auto_sizes_from_the_pool(self):
        wb_proxy.CHAT_SLOTS_AUTO = True
        wb_proxy.MAX_CONCURRENT_CHAT = wb_proxy.CHAT_SLOTS_FLOOR
        self.assertEqual(wb_proxy.resize_chat_slots(110), 110)
        self.assertEqual(wb_proxy.MAX_CONCURRENT_CHAT, 110)
        # The new semaphore really does carry that many permits.
        acquired = 0
        while wb_proxy._chat_slots.acquire(blocking=False):
            acquired += 1
        self.assertEqual(acquired, 110)
        for _ in range(acquired):
            wb_proxy._chat_slots.release()

    def test_small_pool_falls_back_to_the_floor(self):
        wb_proxy.CHAT_SLOTS_AUTO = True
        wb_proxy.MAX_CONCURRENT_CHAT = wb_proxy.CHAT_SLOTS_FLOOR
        self.assertEqual(wb_proxy.resize_chat_slots(3), wb_proxy.CHAT_SLOTS_FLOOR)

    def test_never_shrinks(self):
        """A shrink would let releases exceed the ceiling and raise ValueError."""
        wb_proxy.CHAT_SLOTS_AUTO = True
        wb_proxy.MAX_CONCURRENT_CHAT = wb_proxy.CHAT_SLOTS_FLOOR
        wb_proxy.resize_chat_slots(200)
        slots = wb_proxy._chat_slots
        self.assertEqual(wb_proxy.resize_chat_slots(5), 200)
        self.assertIs(wb_proxy._chat_slots, slots)

    def test_fixed_setting_is_a_no_op(self):
        """A numeric WB_MAX_CONCURRENT_CHAT keeps the 1.6.x behaviour."""
        wb_proxy.CHAT_SLOTS_AUTO = False
        wb_proxy.MAX_CONCURRENT_CHAT = 32
        slots = wb_proxy._chat_slots
        self.assertEqual(wb_proxy.resize_chat_slots(500), 32)
        self.assertEqual(wb_proxy.MAX_CONCURRENT_CHAT, 32)
        self.assertIs(wb_proxy._chat_slots, slots)

    def test_garbage_input_is_ignored(self):
        wb_proxy.CHAT_SLOTS_AUTO = True
        wb_proxy.MAX_CONCURRENT_CHAT = wb_proxy.CHAT_SLOTS_FLOOR
        self.assertEqual(wb_proxy.resize_chat_slots(None), wb_proxy.CHAT_SLOTS_FLOOR)
        self.assertEqual(wb_proxy.resize_chat_slots("abc"), wb_proxy.CHAT_SLOTS_FLOOR)


class SlotAccountingTests(unittest.TestCase):
    """A resize must not corrupt the in-flight count."""

    def test_resize_preserves_held_slots(self):
        saved_auto = wb_proxy.CHAT_SLOTS_AUTO
        saved_max = wb_proxy.MAX_CONCURRENT_CHAT
        saved_slots = wb_proxy._chat_slots
        try:
            wb_proxy.CHAT_SLOTS_AUTO = True
            wb_proxy.MAX_CONCURRENT_CHAT = wb_proxy.CHAT_SLOTS_FLOOR
            wb_proxy._chat_slots = threading.BoundedSemaphore(wb_proxy.CHAT_SLOTS_FLOOR)
            self.assertTrue(wb_proxy._chat_slots.acquire(blocking=False))
            wb_proxy.resize_chat_slots(64)
            # The permit taken before the resize belongs to the old semaphore;
            # releasing it there must not raise on the new one.
            for _ in range(64):
                self.assertTrue(wb_proxy._chat_slots.acquire(blocking=False))
            for _ in range(64):
                wb_proxy._chat_slots.release()
        finally:
            wb_proxy.CHAT_SLOTS_AUTO = saved_auto
            wb_proxy.MAX_CONCURRENT_CHAT = saved_max
            wb_proxy._chat_slots = saved_slots


if __name__ == "__main__":
    unittest.main(verbosity=2)
