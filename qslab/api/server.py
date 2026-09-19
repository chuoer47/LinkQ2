"""L4 service surface: an OpenAI-compatible HTTP front end over the runtime.

Scope is deliberately minimal — /health, /v1/models, /v1/chat/completions and
the raw /v1/completions over one engine, streaming and non-streaming. What is
fixed here is the *shape*, so the measurement work (benchmarks/09, benchmarks/10)
has something a caller can point at.

Design notes
------------
* One pump thread owns the engine. ``LLMEngine.step()`` is synchronous and
  single-threaded and the scheduler inside it is the batching point, so the
  server does not create a thread per request: clients hand their request to
  ``EnginePump`` and read deltas from an ``asyncio`` queue the pump feeds with
  ``call_soon_threadsafe``.
* Token-level streaming needs per-step visibility, which ``step()`` does not
  give (it reports only finished sequences). The pump therefore keeps the
  ``Sequence`` that ``LLMEngine.add_request`` returns and diffs
  ``seq.completion_token_ids`` after every step. A preempted, re-prefilled
  sequence keeps its token list, so the diff stays correct.
* Client disconnects are not wired to cancellation: an abandoned request keeps
  generating until its token budget runs out and the pump drops the deltas.
  Cancelling mid-run needs a scheduler API that does not exist yet, so it is
  registered as a limit instead of being half-implemented.
* aiohttp is an optional extra (``pip install -e .[serve]``); the engine itself
  keeps no web dependency.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass

DEFAULT_MAX_TOKENS = 128
GREEDY = 1e-6          # qslab.api.llm convention: the runtime has no greedy branch


@dataclass
class Job:
    """One in-flight completion, as the pump sees it."""
    job_id: str
    prompt: list[int]
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = GREEDY
    seq: object = None
    sent: int = 0                       # completion tokens already handed out
    out_q: asyncio.Queue | None = None
    loop: asyncio.AbstractEventLoop | None = None

    def publish(self, item) -> None:
        if self.out_q is None:
            return
        self.loop.call_soon_threadsafe(self.out_q.put_nowait, item)


class EnginePump:
    """Drives one LLMEngine on a private thread and fans out token deltas."""

    def __init__(self, engine):
        self.eng = engine
        self.jobs: dict[str, Job] = {}
        self._inbox: queue.Queue[Job | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="qslab-pump")
        self._thread.start()

    # ------------------------------------------------------------------ callers
    def attach(self, job: Job) -> asyncio.Queue:
        """Bind a job to the calling event loop before submitting it."""
        q: asyncio.Queue = asyncio.Queue()
        job.out_q = q
        job.loop = asyncio.get_running_loop()
        return q

    def submit(self, job: Job) -> Job:
        self._inbox.put(job)
        return job

    def close(self) -> None:
        self._stop.set()
        self._inbox.put(None)

    # ------------------------------------------------------------------ pump thread
    def _register(self, job: Job) -> None:
        from qslab.runtime.sampling_params import SamplingParams
        job.seq = self.eng.add_request(
            prompt=job.prompt,
            sampling_params=SamplingParams(temperature=job.temperature,
                                           max_tokens=job.max_tokens))
        self.jobs[job.job_id] = job

    def _drain(self, blocking: bool = False) -> bool:
        """Submit everything waiting in the inbox; False when asked to stop."""
        while True:
            try:
                job = self._inbox.get() if blocking else self._inbox.get_nowait()
            except queue.Empty:
                return True
            if job is None:
                return False
            self._register(job)
            blocking = False

    def _fanout(self) -> None:
        outputs, _ = self.eng.step()
        for job_id in list(self.jobs):
            job = self.jobs[job_id]
            n = len(job.seq.completion_token_ids)
            if n > job.sent:
                delta = self.eng.tokenizer.decode(
                    job.seq.completion_token_ids[job.sent:n])
                job.publish({"delta": delta, "tokens": n - job.sent,
                             "finish": None})
                job.sent = n
            if job.seq.is_finished:
                # `length` when the budget cut it off, `stop` when EOS did
                job.publish({"delta": "", "tokens": 0,
                             "finish": "length" if job.sent >= job.max_tokens
                                       else "stop",
                             "completion_tokens": job.sent,
                             "prompt_tokens": job.seq.num_prompt_tokens})
                del self.jobs[job_id]

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._drain():
                return
            if self.eng.is_finished():
                # nothing running: block until the next request arrives rather
                # than spinning (the engine owns the GPU; a busy loop would
                # just burn a core)
                if not self._drain(blocking=True):
                    return
                continue
            self._fanout()


# ---------------------------------------------------------------------- HTTP wire
def _reply(model: str, text: str, finish: str, n_prompt: int, n_completion: int) -> dict:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:8],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                  "total_tokens": n_prompt + n_completion},
    }


def _chunk(model: str, cid: str, delta: str, finish: str | None) -> dict:
    return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta},
                         "finish_reason": finish}]}


def _bad_request(msg: str):
    from aiohttp import web
    return web.json_response({"error": {"message": msg, "type": "invalid_request"}},
                             status=400)


def create_app(llm, *, model_name: str | None = None, max_tokens_cap: int = 4096):
    """Build the aiohttp app over an ``LLM`` facade or a raw ``LLMEngine`` — or
    anything with the same add_request / step / is_finished / tokenizer surface,
    which is what the fast-tier tests hand it."""
    from aiohttp import web

    engine = getattr(llm, "engine", llm)
    pump = EnginePump(engine)
    tok = engine.tokenizer
    name = model_name or getattr(llm, "model_path", "qslab")
    # the scheduler clamps max_model_len against the rope ceiling, so it is the
    # only honest ceiling to validate input against
    ceiling = getattr(getattr(engine, "scheduler", None), "max_model_len", None)

    def render(body: dict) -> tuple[list[int], int, float, bool] | str:
        if not isinstance(body, dict):
            return "body must be a JSON object"
        try:
            raw_mt = body.get("max_tokens")
            # explicit None test: `or DEFAULT` would swallow a literal 0
            max_tokens = DEFAULT_MAX_TOKENS if raw_mt is None else int(raw_mt)
        except (TypeError, ValueError):
            return "max_tokens must be an integer"
        if not 1 <= max_tokens <= max_tokens_cap:
            return f"max_tokens must be in 1..{max_tokens_cap}"
        temp = body.get("temperature")
        temperature = GREEDY if temp is None or float(temp) <= 0 else float(temp)
        stream = bool(body.get("stream"))
        if "messages" in body:
            if not isinstance(body["messages"], list) or not body["messages"]:
                return "messages must be a non-empty list"
            text = tok.apply_chat_template(body["messages"], tokenize=False,
                                           add_generation_prompt=True)
            ids = tok.encode(text)
        elif "prompt" in body:
            raw = body["prompt"]
            ids = tok.encode(raw) if isinstance(raw, str) else list(raw)
        else:
            return "needs either 'messages' or 'prompt'"
        if not ids:
            return "empty prompt"
        if ceiling and len(ids) + 1 >= ceiling:
            return f"prompt of {len(ids)} tokens leaves no room under max_model_len={ceiling}"
        return ids, max_tokens, temperature, stream

    async def completions(request):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _bad_request("body is not JSON")
        rendered = render(body)
        if isinstance(rendered, str):
            return _bad_request(rendered)
        ids, max_tokens, temperature, stream = rendered

        job = Job(job_id=uuid.uuid4().hex, prompt=ids, max_tokens=max_tokens,
                  temperature=temperature)
        q = pump.attach(job)
        pump.submit(job)
        if not stream:
            # both modes read the same event stream; non-streaming just
            # collects it before answering
            parts: list[str] = []
            finish, n_prompt, n_completion = "stop", 0, 0
            while True:
                ev = await q.get()
                parts.append(ev["delta"])
                if ev["finish"]:
                    finish = ev["finish"]
                    n_completion = ev["completion_tokens"]
                    n_prompt = ev["prompt_tokens"]
                    break
            return web.json_response(
                _reply(name, "".join(parts), finish, n_prompt, n_completion))

        cid = "chatcmpl-" + uuid.uuid4().hex[:8]
        resp = web.StreamResponse()
        resp.headers["Content-Type"] = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        await resp.prepare(request)
        await resp.write(b"data: " + json.dumps(
            _chunk(name, cid, "", None)).encode() + b"\n\n")
        while True:
            ev = await q.get()
            if ev is None:
                break
            await resp.write(b"data: " + json.dumps(
                _chunk(name, cid, ev["delta"], ev["finish"])).encode() + b"\n\n")
            if ev["finish"]:
                break
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def health(request):
        return web.json_response({"status": "ok", "model": name,
                                  "active_requests": len(pump.jobs)})

    async def models(request):
        return web.json_response({"object": "list",
                                  "data": [{"id": name, "object": "model",
                                            "owned_by": "qslab"}]})

    app = web.Application()
    app["pump"] = pump
    app["engine"] = engine
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_post("/v1/completions", completions)
    app.on_cleanup.append(lambda _app: pump.close())
    return app


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m qslab.api.server",
                                description="OpenAI-compatible server over the qslab runtime")
    p.add_argument("--model", required=True)
    p.add_argument("--w4", default=None)
    p.add_argument("--w4-backend", default="w4.auto")
    p.add_argument("--smooth-kv", default=None)
    p.add_argument("--spec", default=None, choices=[None, "ngram", "lookahead", "draft"])
    p.add_argument("--spec-gamma", type=int, default=4)
    p.add_argument("--draft", default=None)
    p.add_argument("--device", default=None, help="e.g. cuda:3 (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--util", type=float, default=0.9)
    args = p.parse_args(argv)

    if args.device:
        # must happen before torch touches the GPU — after that the device list
        # is already frozen (same reason api/cli.py puts it before its imports)
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.rsplit(":", 1)[-1]
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        raise SystemExit("the service surface needs the optional extra: pip install -e .[serve]")

    from aiohttp import web
    from qslab.api.llm import LLM
    llm = LLM(args.model, w4=args.w4, w4_backend=args.w4_backend,
              smooth_kv=args.smooth_kv, spec=args.spec, spec_gamma=args.spec_gamma,
              draft=args.draft, max_num_seqs=args.max_num_seqs,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.util)
    web.run_app(create_app(llm, model_name=args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
