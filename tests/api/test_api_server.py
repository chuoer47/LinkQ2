"""Service surface (qslab.api.server) — fast tier, no GPU, no weights.

The engine is faked: everything under test is the wire contract (routing,
request validation, SSE framing, usage accounting) and the pump's per-step
diffing, none of which needs a model. The real-engine path is exercised by the
smoke run recorded in benchmarks/11-service-surface/README.md.
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.request
from urllib.error import HTTPError

import pytest

from qslab.api.server import create_app

WORDS = [" The", " capital", " of", " Germany", " is", " Berlin", " and", " Rome"]
EOS_AT = 4                      # the fake emits an EOS after this many tokens


class FakeSeq:
    def __init__(self, prompt, max_tokens):
        self.num_prompt_tokens = len(prompt)
        self.completion_token_ids: list[int] = []
        self.max_tokens = max_tokens
        self.is_finished = False

    def finish(self):
        self.is_finished = True


class FakeTokenizer:
    def encode(self, text):
        return list(range(min(len(text), 100)))

    def decode(self, ids):
        return "".join(WORDS[i % len(WORDS)] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)


class FakeEngine:
    """Prefill is one step, then one token per step, EOS at EOS_AT."""

    def __init__(self):
        self.seqs: list[FakeSeq] = []
        self.tokenizer = FakeTokenizer()
        self.scheduler = type("S", (), {"max_model_len": 64})()
        self.max_inflight = 0

    def add_request(self, prompt, sampling_params):
        seq = FakeSeq(prompt, sampling_params.max_tokens)
        self.seqs.append(seq)
        return seq

    def is_finished(self):
        return all(s.is_finished for s in self.seqs)

    def step(self):
        self.max_inflight = max(self.max_inflight,
                                sum(1 for s in self.seqs if not s.is_finished))
        for s in self.seqs:
            if s.is_finished:
                continue
            s.completion_token_ids.append(len(s.completion_token_ids))
            if len(s.completion_token_ids) >= min(s.max_tokens, EOS_AT):
                s.finish()
        return [], -1


ENGINE: FakeEngine | None = None


@pytest.fixture(scope="module")
def base_url():
    from aiohttp import web

    global ENGINE
    ENGINE = FakeEngine()
    app = create_app(ENGINE, model_name="fake-qslab")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    runner = web.AppRunner(app)
    ready = threading.Event()

    def serve():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", port)
        loop.run_until_complete(site.start())
        ready.set()
        loop.run_forever()

    threading.Thread(target=serve, daemon=True).start()
    assert ready.wait(10), "server thread did not come up"
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                break
        except Exception:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    app["pump"].close()


def _post(url, path, body):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status, json.loads(r.read())


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=5) as r:
        return json.loads(r.read())


def test_health_and_models(base_url):
    health = _get(base_url, "/health")
    assert health["status"] == "ok" and health["model"] == "fake-qslab"
    assert _get(base_url, "/v1/models")["data"][0]["id"] == "fake-qslab"


def test_budget_cutoff_reports_length(base_url):
    status, reply = _post(base_url, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
    assert status == 200
    ch = reply["choices"][0]
    assert ch["message"]["role"] == "assistant" and ch["message"]["content"]
    assert ch["finish_reason"] == "length"
    u = reply["usage"]
    assert u["completion_tokens"] == 3
    assert u["total_tokens"] == u["prompt_tokens"] + 3


def test_eos_reports_stop(base_url):
    status, reply = _post(base_url, "/v1/completions", {"prompt": "abc", "max_tokens": 8})
    assert status == 200
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert reply["usage"]["completion_tokens"] == EOS_AT


def test_stream_frames_sse(base_url):
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps({"prompt": "abc", "max_tokens": 4, "stream": True}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    deltas, saw_done = [], False
    with urllib.request.urlopen(req, timeout=20) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        for raw in r:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                saw_done = True
                break
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                assert chunk["object"] == "chat.completion.chunk"
                deltas.append(chunk["choices"][0]["delta"]["content"])
    assert saw_done, "stream never terminated with [DONE]"
    assert any(deltas), "no content deltas on the wire"


def test_validation_rejects_bad_bodies(base_url):
    for bad in ({}, {"prompt": ""}, {"prompt": "x", "max_tokens": 0},
                {"prompt": "x", "max_tokens": "abc"}, {"prompt": "t" * 500}):
        with pytest.raises(HTTPError) as e:
            _post(base_url, "/v1/completions", bad)
        assert e.value.code == 400, bad


def test_two_clients_both_complete(base_url):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(_post, base_url, "/v1/completions",
                          {"prompt": p, "max_tokens": 4})
                for p in ("first prompt", "second prompt")]
        replies = [f.result() for f in futs]
    assert all(r[0] == 200 for r in replies)
    assert all(r[1]["usage"]["completion_tokens"] == 4 for r in replies)


class GateEngine(FakeEngine):
    """Blocks inside its first step() so a second submit necessarily arrives
    while the pump is mid-run — the deterministic version of "the pump does
    not serialize clients". It says nothing about how the real scheduler
    batches; that is benchmarks/09's territory."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def step(self):
        if not self.entered.is_set():
            self.entered.set()
            self.release.wait(5)
        return super().step()


def test_pump_accepts_a_second_request_mid_step():
    from qslab.api.server import EnginePump, Job

    eng = GateEngine()
    pump = EnginePump(eng)
    try:
        pump.submit(Job(job_id="a", prompt=[1, 2, 3], max_tokens=4))
        assert eng.entered.wait(5), "pump never drove a step"
        pump.submit(Job(job_id="b", prompt=[1, 2, 3, 4, 5], max_tokens=4))
        eng.release.set()          # unblock step #1; b lands on the next drain
        for _ in range(200):
            if eng.is_finished():
                break
            time.sleep(0.02)
        assert eng.is_finished(), "the second request never drained"
        assert len(eng.seqs) == 2
        assert eng.max_inflight >= 2, "the two requests never shared a step"
    finally:
        eng.release.set()
        pump.close()
