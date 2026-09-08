"""FarField console: a research question in, a written protocol out.

stdlib `http.server`, Server-Sent Events, `webui/index.html` plus `i18n.js`.
Settings (which model, how many tries) sit behind the question. The page
reports what a colleague would say out loud, not internal stage names.

Every run uses a live model the caller named. Credentials travel in the
POST body, never in the URL, and are never written to an artifact. If the
machine already has `$FARFIELD_LLM_*` set, the form can stay empty.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from farfield.extras.mission import (  # noqa: E402
    brief_from_dict,
    card_from_dict,
    default_client,
    load_assets,
    polish_idea,
    run_research,
    works_from_dicts,
)
from farfield.extras.llm import LLMUnavailable, configured_status  # noqa: E402
from farfield.extras.workspace import (  # noqa: E402
    iter_recorded_events,
    list_recorded_missions,
    mission_dir,
    resolve_recorded_mission,
    var_dir,
)

PAGE = ROOT / "webui" / "index.html"
PORT = 8765


def _runtime_stores() -> dict[str, Path]:
    """Optional FARFIELD_RUNTIME isolates archive/policy from other missions."""
    override = str(os.environ.get("FARFIELD_RUNTIME") or "").strip()
    runtime = Path(override).expanduser() if override else var_dir(ROOT)
    runtime.mkdir(parents=True, exist_ok=True)
    return {
        "state_store": runtime / "research_state.json",
        "policy_log": runtime / "policy_log.json",
        "policy_file": runtime / "research_policy.json",
        "judges_file": runtime / "research_judges.json",
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/i18n.js":
            body = (ROOT / "webui" / "i18n.js").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/status":
            payload = json.dumps(configured_status(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/missions":
            payload = json.dumps(
                {"missions": list_recorded_missions(ROOT)},
                ensure_ascii=False,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/replay":
            query = urllib.parse.parse_qs(parsed.query)
            mission_id = str((query.get("id") or [""])[0]).strip()
            try:
                pace = int((query.get("pace") or ["90"])[0])
            except (TypeError, ValueError):
                pace = 90
            self.stream_replay(mission_id, pace_ms=max(0, min(pace, 800)))
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/mission":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            self.stream_mission(body)
            return
        if parsed.path == "/api/polish":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            self.stream_polish(body)
            return
        self.send_error(404)

    def _begin_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _emit(self, event: dict) -> None:
        payload = json.dumps(event, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()

    def stream_replay(self, mission_id: str, *, pace_ms: int = 90) -> None:
        self._begin_sse()
        dest = resolve_recorded_mission(ROOT, mission_id)
        if dest is None:
            self._emit(
                {
                    "stage": "blocked",
                    "missing_capability": "recorded_mission",
                    "unlock_condition": "没有这场可回放的记录，或编号不合法",
                }
            )
            return
        events = iter_recorded_events(dest)
        if not events:
            self._emit(
                {
                    "stage": "blocked",
                    "missing_capability": "mission_ndjson",
                    "unlock_condition": (
                        f"{dest.name} 没有 mission.ndjson，无法在前端回放"
                    ),
                }
            )
            return
        try:
            for event in events:
                if not isinstance(event, dict):
                    continue
                event = dict(event)
                event["replay"] = True
                self._emit(event)
                stage = str(event.get("stage") or "")
                if not pace_ms or stage == "parallel":
                    continue
                factor = 3 if stage in {
                    "jump",
                    "verdict",
                    "idea_refining",
                    "world_incompatible",
                    "world_constructed",
                    "world_path",
                    "world_scout",
                    "world_forecast",
                    "world_surrogate",
                    "world_placebo",
                    "evidence",
                    "experiment_decision",
                    "explore_decision",
                    "program",
                    "done",
                    "brief",
                    "host_skipped",
                    "host_execute",
                    "diagnosis",
                    "probe",
                } else 1
                time.sleep(pace_ms * factor / 1000.0)
        except BrokenPipeError:
            return

    def stream_mission(self, body: dict) -> None:
        topic = str(body.get("topic") or "").strip()
        try:
            jumps = int(body.get("jumps") or 4)
        except (TypeError, ValueError):
            jumps = 0
        try:
            candidates = int(body.get("candidates") or 1)
        except (TypeError, ValueError):
            candidates = 0
        try:
            polish_rounds = int(
                body.get("polish_rounds") if body.get("polish_rounds") is not None else 0
            )
        except (TypeError, ValueError):
            polish_rounds = -2
        explore = str(body.get("explore") or "auto").strip().lower()
        try:
            experiment_rounds = int(
                body.get("experiment_rounds")
                if body.get("experiment_rounds") is not None
                else -1
            )
        except (TypeError, ValueError):
            experiment_rounds = -2
        try:
            idea_rounds = int(
                body.get("idea_rounds")
                if body.get("idea_rounds") is not None
                else -1
            )
        except (TypeError, ValueError):
            idea_rounds = -2
        try:
            reframes = int(body.get("reframes") if body.get("reframes") is not None else 1)
        except (TypeError, ValueError):
            reframes = -1
        model = str(body.get("model") or "").strip()
        base_url = str(body.get("base_url") or "").strip()
        api_key = str(body.get("api_key") or "").strip()
        world = str(body.get("world") if body.get("world") is not None else "auto").strip()
        if not world:
            world = "auto"
        host_execute = body.get("host_execute")
        if host_execute is None:
            host_execute = True
        else:
            host_execute = bool(host_execute)

        self._begin_sse()
        emit = self._emit

        if not topic:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "mission_topic",
                    "unlock_condition": "写一句你想做的方向",
                }
            )
            return
        if not 1 <= jumps <= 12:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_jump_count",
                    "unlock_condition": "探索方向数请放在 1 到 12 之间",
                }
            )
            return
        if not 1 <= candidates <= 3:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_candidate_count",
                    "unlock_condition": "每个方向起草 1 到 3 稿",
                }
            )
            return
        if explore not in ("auto", "fixed"):
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_explore_mode",
                    "unlock_condition": "探索模式请选 auto 或 fixed",
                }
            )
            return
        if not -1 <= polish_rounds <= 4:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_polish_count",
                    "unlock_condition": "打磨轮数请放在 -1（按探索）到 4 之间",
                }
            )
            return
        if not -1 <= experiment_rounds <= 4:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_experiment_count",
                    "unlock_condition": "实验迭代次数请放在 -1（按证据）到 4 之间",
                }
            )
            return
        if not -1 <= idea_rounds <= 4:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_idea_count",
                    "unlock_condition": "想法迭代次数请放在 -1（按门控）到 4 之间",
                }
            )
            return
        if not 0 <= reframes <= 3:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_reframe_count",
                    "unlock_condition": "重问方向数请放在 0 到 3 之间",
                }
            )
            return
        try:
            parallel = int(body.get("parallel") if body.get("parallel") is not None else 8)
        except (TypeError, ValueError):
            parallel = -1
        if not 1 <= parallel <= 16:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_parallel_count",
                    "unlock_condition": "并行路数请放在 1 到 16 之间",
                }
            )
            return
        try:
            dest = mission_dir(ROOT, topic)
            dest.mkdir(parents=True, exist_ok=True)
            stores = _runtime_stores()
            for event in run_research(
                topic,
                horizon=jumps,
                candidates=candidates,
                polish_rounds=polish_rounds,
                explore=explore,
                experiment_rounds=experiment_rounds,
                idea_rounds=idea_rounds,
                world=world if world not in {"auto", "none", ""} else None,
                host_execute=host_execute,
                reframes=reframes,
                parallel=parallel,
                model=model,
                base_url=base_url,
                api_key=api_key,
                workspace=dest,
                **stores,
            ):
                emit(event)
        except BrokenPipeError:
            return
        except Exception as error:  # surfaced, never swallowed
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "mission_crashed",
                    "unlock_condition": f"{type(error).__name__}: {error}",
                }
            )


    def stream_polish(self, body: dict) -> None:
        topic = str(body.get("topic") or "").strip()
        model = str(body.get("model") or "").strip()
        base_url = str(body.get("base_url") or "").strip()
        api_key = str(body.get("api_key") or "").strip()
        try:
            rounds = int(body.get("rounds") or 1)
        except (TypeError, ValueError):
            rounds = 0
        self._begin_sse()
        emit = self._emit

        card_row = body.get("card") if isinstance(body.get("card"), dict) else {}
        brief_row = body.get("brief") if isinstance(body.get("brief"), dict) else {}
        papers = body.get("papers") if isinstance(body.get("papers"), list) else []
        if not topic or not card_row or not brief_row:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "polish_payload",
                    "unlock_condition": "再打磨需要题目、当前假设和调研备忘",
                }
            )
            return
        if not 1 <= rounds <= 3:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "sane_polish_count",
                    "unlock_condition": "一次再打磨 1 到 3 轮",
                }
            )
            return
        try:
            from farfield.extras.mission import default_backend

            client = default_client(
                float(2 * rounds + 1),
                backend=default_backend(
                    model=model, base_url=base_url, api_key=api_key
                ),
            )
            works = works_from_dicts(papers)
            card = card_from_dict(card_row)
            brief = brief_from_dict(brief_row, works)
            for event in polish_idea(
                client, card, brief, topic, works, rounds=rounds
            ):
                emit(event)
            emit({"stage": "done", "card_id": card.card_id})
        except LLMUnavailable as error:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": error.record.missing_capability,
                    "unlock_condition": error.record.unlock_condition,
                }
            )
        except BrokenPipeError:
            return
        except Exception as error:
            emit(
                {
                    "stage": "blocked",
                    "missing_capability": "polish_crashed",
                    "unlock_condition": f"{type(error).__name__}: {error}",
                }
            )


def main() -> int:
    status = configured_status()
    if status["ready"]:
        print(f"LLM ready: {status['model']} @ {status['base_url']}", flush=True)
    else:
        print(f"LLM not configured yet: {status['unlock']}", flush=True)
        print("pick a model in the console; a key can stay on this machine", flush=True)
    print("loading production corpus (one-time)...", flush=True)
    assets = load_assets()
    print(
        f"ready: {assets.corpus_uid}, {len(assets.graph.nodes)} concepts,"
        f" knowledge through {assets.fresh_until}",
        flush=True,
    )
    # Keys travel in POST bodies over plain HTTP; default to loopback so a
    # shared machine's LAN cannot read them or start missions. Opt out with
    # FARFIELD_HOST=0.0.0.0 when the console must be reachable remotely.
    host = str(os.environ.get("FARFIELD_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    server = ThreadingHTTPServer((host, PORT), Handler)
    print(f"FarField -> http://localhost:{PORT}/ (bound to {host})", flush=True)
    if host in ("127.0.0.1", "localhost", "::1"):
        print("local only; tunnel this port to open the page from another machine", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
