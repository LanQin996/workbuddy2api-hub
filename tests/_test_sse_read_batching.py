"""Block-buffered SSE reading must be byte-identical to plain line iteration.

`for line in upstream` on an http.client.HTTPResponse costs one read() syscall
per SSE event, and every syscall releases and re-acquires the GIL. A streaming
turn emits thousands of events, so the process spends most of its time handing
the GIL between handler threads instead of working. Measured on the 4 vCPU box
running v1.7.2 at 15-20 req/s:

    10s syscall profile of wb-proxy (strace -c -f -p <pid>)
      71.44%   59.43s    559us/call   106214  futex     <- GIL handoff
      20.31%   16.89s  42659us/call      396  poll
       1.80%    1.50s     41us/call    36366  read      <- one per SSE line
       0.54%    0.45s     88us/call     5043  sendto

`iter_sse_lines()` reads with read1() (at most one underlying syscall per call)
and splits in memory, which is why the equivalence pinned here matters: the
whole point is to change the *number* of syscalls without changing a single
byte the caller sees.

The reference is io.BytesIO iteration, not a hand-rolled splitter, because that
is the same readline() semantics the replaced loop had: lines carry their
terminator, a final line without one is still yielded, and only "\\n" splits.
"""
import ast
import inspect
import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _isolated_dirs import isolated_data_dirs  # noqa: E402  (its own data root)
# Its own directories, removed when this process exits.
_TMP = isolated_data_dirs("wb-sse-read-batching-")

import wb_proxy as P  # noqa: E402

PASS = 0
FAIL = 0


def check(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("  [PASS] " + label)
    else:
        FAIL += 1
        print("  [FAIL] " + label + ("  " + str(detail) if detail else ""))


class FakeStream:
    """The wire as a list of chunks, answered through read1().

    `chunks` is handed over in order: read1() returns at most `n` bytes from the
    front of the current chunk and b"" once the script is exhausted. Chunks are
    deliberately not aligned to any block size, so lines straddle boundaries -
    the case the reader has to get right.
    """

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.pending = b""
        self.read1_calls = 0

    def read1(self, n=-1):
        self.read1_calls += 1
        if not self.pending:
            if not self.chunks:
                return b""
            self.pending = self.chunks.pop(0)
        if n is None or n < 0:
            out, self.pending = self.pending, b""
            return out
        out, self.pending = self.pending[:n], self.pending[n:]
        return out


def same_as_plain_iteration(data, chunk_size):
    """Read `data` in `chunk_size` chunks; return (yielded, read1_calls)."""
    wire = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)] or [b""]
    src = FakeStream(wire)
    got = list(P.iter_sse_lines(src))
    return got, src.read1_calls, len(wire)


print("-- equivalence with io.BytesIO iteration --")

CASES = [
    ("single event", b"data: {\"a\":1}\n\n"),
    ("two events, one block", b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n"),
    ("no trailing newline", b"data: {\"a\":1}\n\ndata: {\"b\":2}"),
    ("CRLF terminators", b"data: {\"a\":1}\r\n\r\ndata: {\"b\":2}\r\n\r\n"),
    ("blank lines and comments", b"\n: keepalive\n\ndata: x\n\n"),
    ("empty stream", b""),
    ("only a newline", b"\n"),
    ("no newline at all", b"data: {"),
    ("long single event", b"data: " + b"x" * 5000 + b"\n\n"),
    ("many events", b"".join(b"data: {\"i\":%d}\n\n" % i for i in range(300))),
]

for label, data in CASES:
    for chunk_size in (1, 7, 64, 4096):
        got, calls, wire = same_as_plain_iteration(data, chunk_size)
        ref = list(io.BytesIO(data))
        check("%s (chunk=%d) byte-identical" % (label, chunk_size),
              got == ref,
              "got=%r ref=%r" % (got[:4], ref[:4]))

print("-- the same bytes through the real io.BytesIO read1() --")
for label, data in CASES:
    got = list(P.iter_sse_lines(io.BytesIO(data)))
    check("%s via BytesIO" % label, got == list(io.BytesIO(data)),
          "got=%r" % (got[:4],))

print("-- a line is emitted as soon as it is complete, not when a block fills --")
src = FakeStream([b"data: one\n"])
gen = P.iter_sse_lines(src)
check("first complete line arrives on the first read1",
      next(gen) == b"data: one\n")
check("no second read1 was needed to emit it", src.read1_calls == 1,
      "%d calls" % src.read1_calls)

src = FakeStream([b"data: par", b"tial\n"])
got = list(P.iter_sse_lines(src))
check("a line straddling two chunks is reassembled", got == [b"data: partial\n"],
      repr(got))

print("-- syscall count: chunks, not lines --")

lines = b"".join(b"data: {\"i\":%d}\n\n" % i for i in range(2000))
ref = list(io.BytesIO(lines))
for chunk_size in (8192, 65536):
    got, calls, wire = same_as_plain_iteration(lines, chunk_size)
    check("%d-byte wire chunks stay byte-identical" % chunk_size, got == ref)
    check("%d-byte wire chunks: read1 calls bounded by wire chunks" % chunk_size,
          calls <= wire + 2,
          "%d calls for %d lines over %d chunks" % (calls, len(ref), wire))

# The headline number: what the old loop cost per line versus what it costs now.
_, calls, wire = same_as_plain_iteration(lines, 65536)
print("    2000 events / 64 KiB blocks: %d read1 calls vs %d lines (%.0fx fewer)"
      % (calls, len(ref), float(len(ref)) / max(1, calls)))
check("at least an order of magnitude fewer syscalls",
      len(ref) >= 10 * calls, "%d lines, %d calls" % (len(ref), calls))

print("-- objects without read1 fall back to plain iteration --")
check("a list iterates unchanged",
      list(P.iter_sse_lines([b"a\n", b"b\n"])) == [b"a\n", b"b\n"])
check("an iterator iterates unchanged",
      list(P.iter_sse_lines(iter([b"a\n", b"b"]))) == [b"a\n", b"b"])

src = FakeStream([b"y" * 100])
list(P.iter_sse_lines(src, block_size=16))
check("a custom block_size is passed to read1",
      src.read1_calls >= 7, "%d calls for 100 bytes at 16" % src.read1_calls)

print("-- strip_data_prefix still sees the shape it expects --")
got = list(P.iter_sse_lines(FakeStream([b"data: {\"a\": 1}\n\n", b"data: [DONE]\n\n"])))
check("prefix stripping works on the yielded lines",
      [P.strip_data_prefix(l.decode("utf-8", "replace")) for l in got]
      == ['{"a": 1}', "", "[DONE]", ""],
      repr(got))

print("-- before_read: the hook that lets a caller batch its writes --")

calls = [0]
src = FakeStream([b"a\n", b"b\n", b"c\n"])
got = list(P.iter_sse_lines(src, before_read=lambda: calls.__setitem__(0, calls[0] + 1)))
check("lines still come out intact with a hook attached",
      got == [b"a\n", b"b\n", b"c\n"], repr(got))
check("the hook fires exactly once per read1",
      calls[0] == src.read1_calls,
      "%d hook calls vs %d read1 calls" % (calls[0], src.read1_calls))

src = FakeStream([b"a\n", b"b\n"])
list(P.iter_sse_lines(src))
check("read1_calls is 3 for 2 chunks (the last one returns b'')",
      src.read1_calls == 3, src.read1_calls)

# The invariant that makes batching latency-free: whenever the generator is
# about to block on the socket, the caller's buffer has already been drained.
# Nothing sits in memory waiting for data that has not arrived.
drained = []
written = []
src = FakeStream([b"data: a\n", b"data: b\n", b"data: c\n"])
buf = []


def flush_out():
    if buf:
        written.append(b"".join(buf))
        del buf[:]


def before_read():
    # The hook runs immediately before read1(). Draining here is exactly what
    # makes batching latency-free, so record the size *after* the drain.
    flush_out()
    drained.append(len(buf))


for line in P.iter_sse_lines(src, before_read=before_read):
    buf.append(line)
flush_out()
check("the write buffer is empty whenever a blocking read begins",
      drained and all(n == 0 for n in drained), drained)
check("every event still reached the wire",
      b"".join(written) == b"data: a\ndata: b\ndata: c\n", written)


def batched(data, chunk_size):
    """Mimic _chat_stream_response: buffer events, flush on before_read."""
    wire = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)] or [b""]
    src = FakeStream(wire)
    writes = []
    pending = []

    def flush():
        if pending:
            writes.append(b"".join(pending))
            del pending[:]

    for line in P.iter_sse_lines(src, before_read=flush):
        pending.append(b"data: " + line)
    flush()
    return writes, len(wire)


print("-- batching: one write per read block, not one per event --")

events = b"".join(b"data: {\"i\":%d}\n\n" % i for i in range(2000))
expected = b"".join(b"data: " + l for l in io.BytesIO(events))
for chunk_size in (8192, 65536):
    writes, wire = batched(events, chunk_size)
    check("%d-byte blocks: batched output is byte-identical" % chunk_size,
          b"".join(writes) == expected)
    check("%d-byte blocks: at most one write per block" % chunk_size,
          len(writes) <= wire + 1,
          "%d writes over %d blocks" % (len(writes), wire))

writes, wire = batched(events, 65536)
n_events = len(list(io.BytesIO(events)))
print("    4000 SSE lines / 64 KiB blocks: %d writes vs %d unbuffered writes (%.0fx fewer)"
      % (len(writes), n_events, float(n_events) / max(1, len(writes))))
check("batching removes at least an order of magnitude of writes",
      n_events >= 10 * len(writes),
      "%d lines, %d writes" % (n_events, len(writes)))

print("-- Responses: block reads and writes preserve frames and latency --")


def chat_event(delta, finish=None):
    return ("data: " + json.dumps({
        "choices": [{"delta": delta, "finish_reason": finish}],
    }) + "\n\n").encode("utf-8")


class ResponseStream(FakeStream):
    def __init__(self, chunks, on_read=None, fail=False):
        super().__init__(chunks)
        self.on_read = on_read
        self.fail = fail
        self.closed = False

    def __iter__(self):
        raise AssertionError("Responses must use read1, not line iteration")

    def read1(self, n=-1):
        if self.on_read:
            self.on_read(self.read1_calls)
        if self.fail and not self.chunks and not self.pending:
            raise ValueError("scripted upstream failure")
        return super().read1(n)

    def close(self):
        self.closed = True


class ResponseWriter:
    def __init__(self):
        self.writes = []

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        pass

    def value(self):
        return b"".join(self.writes)


class ResponseHandler:
    _responses_stream_response = P.Handler._responses_stream_response
    path = "/v1/responses"

    def __init__(self):
        self.wfile = ResponseWriter()

    def send_response(self, code):
        self.status = code

    def send_header(self, *args):
        pass

    def end_headers(self):
        pass

    def _key_id(self):
        return None


def run_response(handler, source, base_body=None):
    handler._responses_stream_response(
        source, "m", set(), {}, None, SimpleNamespace(uid="test"), 1000,
        base_body=base_body)


wire = (b"".join(chat_event({"content": "piece %d" % i}) for i in range(200))
        + chat_event({}, "stop") + b"data: [DONE]\n\n")
with patch.object(P, "_new_id", side_effect=lambda prefix: prefix + "test"), \
        patch.object(P.time, "time", return_value=1000), \
        patch.object(P, "record_upstream_usage"), patch.object(P, "record_error"):
    reference = b"".join(P.clean_responses_frame(frame) for frame in
                         P.stream_responses_events(list(io.BytesIO(wire)), "m", {}))
    for label, chunks in (("DONE", [wire]), ("EOF", [wire.replace(b"data: [DONE]\n\n", b"")]),
                          ("split blocks", [wire[:6000], wire[6000:]])):
        handler = ResponseHandler()
        read_observations = []

        def observe_read(index):
            read_observations.append((index, handler.wfile.value()))

        source = ResponseStream(chunks, on_read=observe_read)
        run_response(handler, source)
        check("Responses %s: bytes unchanged" % label,
              handler.wfile.value() == reference)
        check("Responses %s: lifecycle sent before first read" % label,
              b"response.in_progress" in read_observations[0][1])
        check("Responses %s: deltas drained before next read" % label,
              b"response.output_text.delta" in read_observations[-1][1])
        check("Responses %s: finalized and closed" % label,
              b"response.completed" in handler.wfile.writes[-1] and source.closed)
        check("Responses %s: writes bounded by blocks plus lifecycle/final" % label,
              len(handler.wfile.writes) <= len(chunks) + 2,
              len(handler.wfile.writes))

    late_usage = b'data: {"choices":[],"usage":{"total_tokens":7}}\n\n'
    holder = {}
    list(P.stream_responses_events(ResponseStream([wire, late_usage]), "m", holder))
    check("Responses keeps reading usage after DONE until EOF",
          holder.get("usage", {}).get("total_tokens") == 7)

    handler = ResponseHandler()
    source = ResponseStream([chat_event({"content": "before failure"})], fail=True)
    run_response(handler, source)
    check("Responses error: buffered text precedes DONE",
          b"before failure" in handler.wfile.value()
          and handler.wfile.value().endswith(b"data: [DONE]\n\n"))
    check("Responses error: upstream closed", source.closed)

    # Without read1, obtaining the next item can itself block. Lifecycle frames
    # must retain immediate writes rather than waiting for the iterator hook.
    handler = ResponseHandler()

    class PlainStream:
        def __iter__(self):
            check("Responses plain iterator: lifecycle already sent",
                  b"response.in_progress" in handler.wfile.value())
            yield chat_event({"content": "plain"})
            check("Responses plain iterator: delta already sent",
                  b"plain" in handler.wfile.value())
            yield b"data: [DONE]\n\n"

        def close(self):
            pass

    run_response(handler, PlainStream())

    handler = ResponseHandler()
    first = ResponseStream([chat_event({"tool_calls": [{
        "index": 0, "id": "call_test", "type": "function", "function": {
            "name": P.wb_webtools.WEB_SEARCH_NAME, "arguments": '{"query":"test"}',
        },
    }]}, "tool_calls") + b"data: [DONE]\n\n"])
    second = ResponseStream([chat_event({"content": "after tool"})
                             + b"data: [DONE]\n\n"])

    def follow_up(*args, **kwargs):
        check("Responses follow-up: tool lifecycle flushed before next round",
              b"response.web_search_call.completed" in handler.wfile.value())
        return second, SimpleNamespace(uid="test"), None

    with patch.object(P, "follow_up_with_tool_results", side_effect=follow_up) as follow:
        run_response(handler, first, base_body={"_web_tools": True})
    check("Responses follow-up: exactly one new round", follow.call_count == 1)
    check("Responses follow-up: opening lifecycle emitted once",
          handler.wfile.value().count(b"event: response.created\n") == 1)
    check("Responses follow-up: final response flushed and both streams closed",
          b"response.completed" in handler.wfile.writes[-1]
          and first.closed and second.closed)

print("-- Responses cleanup: fast deltas and escaped JSON keys --")
plain_frame = b'event: response.output_text.delta\ndata: {"delta":"hello"}\n\n'
with patch.object(P.json, "loads", side_effect=AssertionError("unexpected parse")):
    check("plain delta skips JSON parsing", P.clean_responses_frame(plain_frame) is plain_frame)
for key in ("function_call", "function\\u005fcall", "\\u0066unction_call"):
    frame = ('event: test\ndata: {"nested":{"%s":{"name":""}},"delta":"hello"}\n\n'
             % key).encode("utf-8")
    cleaned = P.clean_responses_frame(frame)
    payload = json.loads(cleaned.split(b"data: ", 1)[1])
    check("empty function call removed with key %s" % key,
          payload == {"nested": {}, "delta": "hello"})
frame = b'event: test\ndata: {"tool_calls":[],"function_call":{"name":"real"}}\n\n'
check("nonempty function call survives cleanup",
      b'"name": "real"' in P.clean_responses_frame(frame))
check("malformed and non-bytes frames preserve fallback behavior",
      P.clean_responses_frame(b"data: broken\n\n") == b"data: broken\n\n"
      and P.clean_responses_frame("not bytes") == "not bytes")

print("-- Responses flat delta encoding: exact bytes and generic fallbacks --")

# Extract the production closure so unusual payloads can exercise its guards;
# normal upstream traffic only constructs the canonical payload shapes.
stream_tree = ast.parse(inspect.getsource(P.stream_responses_events))
stream_node = stream_tree.body[0]
emitter_nodes = []
for statement in stream_node.body:
    if (isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id in ("encode_json", "seq", "delta_shapes")
                    for target in statement.targets)):
        emitter_nodes.append(statement)
    elif (isinstance(statement, ast.For) and isinstance(statement.target, ast.Tuple)
          and [item.id for item in statement.target.elts] == ["event_type", "index_key"]):
        emitter_nodes.append(statement)
    elif isinstance(statement, ast.FunctionDef) and statement.name == "ev":
        emitter_nodes.append(statement)
factory = ast.parse("def make_emitter():\n    pass\n").body[0]
factory.body = emitter_nodes + [ast.Return(value=ast.Name(id="ev", ctx=ast.Load()))]
factory_namespace = {"json": json}
exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
             "<production Responses emitter>", "exec"), factory_namespace)


def reference_frame(etype, payload, sequence):
    data = {"type": etype, "sequence_number": sequence}
    data.update(payload)
    return ("event: " + etype + "\ndata: "
            + json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8")


encoding_calls = []
original_encoder = json.JSONEncoder


class RecordingEncoder(original_encoder):
    def encode(self, value):
        encoding_calls.append(type(value))
        return super().encode(value)


with patch.object(P.json, "JSONEncoder", RecordingEncoder):
    emit = factory_namespace["make_emitter"]()
sequence = 0
for event_type, index_key in (("response.output_text.delta", "content_index"),
                              ("response.reasoning_summary_text.delta", "summary_index")):
    for text in ("plain", "中文🙂", 'quote " slash \\ newline\n\t\r\x00', "", "x" * 10000):
        payload = {"item_id": 'id"\\中文', "output_index": -2, index_key: 3, "delta": text}
        encoding_calls.clear()
        sequence += 1
        result = emit(event_type, payload)
        check("%s text length %d: byte-identical" % (event_type, len(text)),
              result == reference_frame(event_type, payload, sequence))
        check("%s: only string values need encoding" % event_type,
              encoding_calls == [str, str], encoding_calls)

    canonical = {"item_id": "id", "output_index": 0, index_key: 0, "delta": "value"}
    fallbacks = [
        ("extra field", dict(canonical, extra=True)),
        ("field order", dict(reversed(list(canonical.items())))),
        ("boolean index", dict(canonical, output_index=True)),
        ("float index", dict(canonical, **{index_key: 1.5})),
        ("nonstring delta", dict(canonical, delta={"nested": [None, False]})),
        ("nonstring item", dict(canonical, item_id=123)),
        ("sequence override", dict(canonical, sequence_number=99)),
        ("type override", dict(canonical, type="different")),
        ("dict subclass", type("Payload", (dict,), {})(canonical)),
    ]
    for label, payload in fallbacks:
        encoding_calls.clear()
        sequence += 1
        result = emit(event_type, payload)
        check("%s %s: generic bytes preserved" % (event_type, label),
              result == reference_frame(event_type, payload, sequence)
              and encoding_calls == [dict], encoding_calls)
sequence += 1
payload = {"item_id": "id", "output_index": 0, "content_index": 0, "delta": "value"}
encoding_calls.clear()
check("other event types retain generic encoding",
      emit("response.output_text.done", payload)
      == reference_frame("response.output_text.done", payload, sequence)
      and encoding_calls == [dict])

print("-- Responses complete-stream comparison with original JSON paths --")
# Restore only the old nested emitter and parser in a source-derived reference
# converter. This compares every lifecycle, delta, final frame and sequence.
reference_ev = ast.parse('''
def ev(etype, payload):
    nonlocal seq
    seq += 1
    data = {"type": etype, "sequence_number": seq}
    data.update(payload)
    body = json.dumps(data, ensure_ascii=False)
    return ("event: " + etype + "\\ndata: " + body + "\\n\\n").encode("utf-8")
''').body[0]


class OriginalJSONPaths(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "ev":
            return reference_ev
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "decode_json":
            node.func = ast.Attribute(value=ast.Name(id="json", ctx=ast.Load()),
                                      attr="loads", ctx=ast.Load())
        return self.generic_visit(node)


reference_tree = OriginalJSONPaths().visit(stream_tree)
reference_tree.body[0].name = "reference_responses_events"
reference_namespace = dict(P.__dict__)
reference_namespace["_new_id"] = lambda prefix: prefix + "test"
exec(compile(ast.fix_missing_locations(reference_tree), "<original JSON paths>", "exec"),
     reference_namespace)
edge_wire = (
    b': heartbeat\n\ndata: {broken json}\n\ndata: \xef\xbb\xbf{}\n\n'
    + chat_event({"reasoning_content": 'thinking 中文🙂 " \\ \n\t\x00'})
    + b'data: data:  {"choices":[{"delta":{"content":"whitespace"}}]}  \n\n'
    + chat_event({"content": 'text 中文🙂 " \\ \n\t\x00'})
    + b'data: {"choices":[{"delta":{"content":"invalid utf8 \xff"}}]}\n\n'
    + chat_event({}, "stop") + b'data: [DONE]\n\n')
for chunk_size in (1, 7, 65536):
    chunks = [edge_wire[i:i + chunk_size] for i in range(0, len(edge_wire), chunk_size)]
    with patch.object(P, "_new_id", side_effect=lambda prefix: prefix + "test"), \
            patch.object(P.time, "time", return_value=1000):
        actual = list(P.stream_responses_events(ResponseStream(chunks), "m", {}))
        expected = list(reference_namespace["reference_responses_events"](
            ResponseStream(chunks), "m", {}))
    check("JSON paths match original bytes across %d-byte chunks" % chunk_size,
          actual == expected)

print("PASS=%d FAIL=%d" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
