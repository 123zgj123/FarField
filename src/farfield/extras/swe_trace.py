"""Deterministic adapters for attested software-agent trajectory archives.

The adapter keeps source observations as data.  It does not ask a model to
label a run, infer a counterfactual, or manufacture a CWM predicted state.
Fields under ``derived`` are mechanical projections whose rules are recorded
here; native source fields remain separately identifiable.
"""

from __future__ import annotations

import io
import hashlib
import json
import re
import zipfile
from pathlib import PurePosixPath
from typing import Any


class TraceFormatError(ValueError):
    """The supplied bytes are not a supported trajectory artifact."""


_RETURN_CODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.I)
_OUTPUT = re.compile(r"<output>\n?(.*?)\n?</output>", re.I | re.S)
_BASH = re.compile(r"```bash\s*\n(.*?)\n```", re.I | re.S)
_ACTION_LIMIT = 4096
_OUTPUT_LIMIT = 1024


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _excerpt(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _model_name(info: dict[str, Any], messages: list[Any], fallback: str) -> str:
    config = _as_dict(info.get("config"))
    model = config.get("model") or config.get("model_name")
    if isinstance(model, str) and model.strip():
        return model.strip()
    for raw in messages:
        message = _as_dict(raw)
        extra = _as_dict(message.get("extra"))
        response = _as_dict(extra.get("response"))
        candidate = response.get("model")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return fallback


def _result_sets(archive: zipfile.ZipFile) -> dict[str, dict[str, set[str]]]:
    results: dict[str, dict[str, set[str]]] = {}
    for name in sorted(archive.namelist()):
        if not name.endswith("/eval_result.json"):
            continue
        parts = PurePosixPath(name).parts
        if len(parts) < 3:
            continue
        model = parts[-2]
        payload = json.loads(archive.read(name))
        if not isinstance(payload, dict):
            continue
        results[model] = {
            "resolved": {str(v) for v in _as_list(payload.get("resolved_ids"))},
            "unresolved": {str(v) for v in _as_list(payload.get("unresolved_ids"))},
            "error": {str(v) for v in _as_list(payload.get("error_ids"))},
            "empty_patch": {str(v) for v in _as_list(payload.get("empty_patch_ids"))},
        }
    return results


def _created_tools(
    archive: zipfile.ZipFile, trajectory_member: str
) -> list[dict[str, Any]]:
    member = trajectory_member.rsplit("/", 1)[0] + "/created_tools.json"
    try:
        payload = json.loads(archive.read(member))
    except KeyError:
        return []
    if not isinstance(payload, dict):
        return []
    return [
        {
            "name": str(name),
            "chars": len(str(script)),
            "sha256": _digest(str(script)),
        }
        for name, script in sorted(payload.items())
    ]


def _steps(messages: list[Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pending = ""
    for raw in messages[2:]:
        message = _as_dict(raw)
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "assistant":
            pending = content
            continue
        if role != "user" or not pending:
            continue
        code = _RETURN_CODE.search(content)
        output = _OUTPUT.search(content)
        command = _BASH.search(pending)
        action = command.group(1) if command else pending
        observed = output.group(1) if output else content
        steps.append(
            {
                "t": len(steps),
                "action": _excerpt(action, _ACTION_LIMIT),
                "action_chars": len(action),
                "action_sha256": _digest(action),
                "action_truncated": len(action) > _ACTION_LIMIT,
                "returncode": int(code.group(1)) if code else None,
                "output_excerpt": _excerpt(observed, _OUTPUT_LIMIT),
                "output_chars": len(observed),
                "output_sha256": _digest(observed),
                "output_truncated": len(observed) > _OUTPUT_LIMIT,
            }
        )
        pending = ""
    return steps


def _label(instance_id: str, model: str, results: dict[str, dict[str, set[str]]]) -> str:
    model_results = results.get(model, {})
    for label in ("resolved", "unresolved", "error", "empty_patch"):
        if instance_id in model_results.get(label, set()):
            return label
    return "unknown"


def parse_live_swe_zip(payload: bytes) -> list[dict[str, Any]]:
    """Normalize the official Live-SWE-agent release archive in member order."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise TraceFormatError("labeled trace archive is not a valid zip") from exc
    results = _result_sets(archive)
    members = sorted(
        name for name in archive.namelist() if name.endswith(".traj.json")
    )
    if not members:
        raise TraceFormatError("zip contained no .traj.json members")
    traces: list[dict[str, Any]] = []
    for member in members:
        raw = json.loads(archive.read(member))
        if not isinstance(raw, dict):
            raise TraceFormatError(f"{member} is not a JSON object")
        path = PurePosixPath(member)
        fallback_model = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        instance_id = str(raw.get("instance_id") or path.stem.removesuffix(".traj"))
        messages = _as_list(raw.get("messages"))
        info = _as_dict(raw.get("info"))
        model = _model_name(info, messages, fallback_model)
        # Eval files use the release folder name, while API metadata may include
        # a dated provider version.  Prefer folder identity for the join.
        eval_model = fallback_model if fallback_model in results else model
        model_stats = _as_dict(info.get("model_stats"))
        steps = _steps(messages)
        repo = instance_id.split("__", 1)[0] if "__" in instance_id else instance_id
        tools = _created_tools(archive, member)
        submission = str(info.get("submission") or "")
        traces.append(
            {
                "id": f"{fallback_model}/{instance_id}",
                "label": _label(instance_id, eval_model, results),
                "instance_id": instance_id,
                "template_id": repo,
                "policy_version": {
                    "release_model": fallback_model,
                    "served_model": model,
                    "mini_version": str(info.get("mini_version") or ""),
                },
                "steps": steps,
                "created_tools": tools,
                "native": {
                    "trajectory_format": str(raw.get("trajectory_format") or ""),
                    "exit_status": str(info.get("exit_status") or ""),
                    "instance_cost": _number(model_stats.get("instance_cost")),
                    "api_calls": _number(model_stats.get("api_calls")),
                    "source_member": member,
                    "submission_chars": len(submission),
                    "submission_sha256": _digest(submission),
                },
                "derived": {
                    "repo": repo,
                    "step_count": len(steps),
                    "failed_step_count": sum(
                        1 for step in steps if step.get("returncode") not in (None, 0)
                    ),
                    "created_tool_count": len(tools),
                    "has_structured_predicted_state": False,
                    "normalization": {
                        "action": f"bash block, first {_ACTION_LIMIT} chars",
                        "output": f"output tag, first {_OUTPUT_LIMIT} chars",
                        "full_source": "attested release zip cache",
                    },
                },
            }
        )
    return traces


def parse_json_traces(text: str) -> list[dict[str, Any]]:
    """Read canonical ``{\"traces\": [...]}``, a JSON list, or JSONL objects."""
    stripped = text.strip()
    if not stripped:
        raise TraceFormatError("labeled trace source is empty")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        rows = []
        for number, line in enumerate(stripped.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TraceFormatError(f"invalid JSONL at line {number}") from exc
        payload = rows
    if isinstance(payload, dict):
        payload = payload.get("traces")
    if not isinstance(payload, list):
        raise TraceFormatError("labeled trace JSON must contain a traces list")
    traces = [dict(row) for row in payload if isinstance(row, dict)]
    if len(traces) != len(payload):
        raise TraceFormatError("every trace must be a JSON object")
    return traces


def validate_traces(traces: list[dict[str, Any]]) -> None:
    if len(traces) < 2:
        raise TraceFormatError("labeled trace world needs at least two traces")
    ids: set[str] = set()
    labels: set[str] = set()
    for index, trace in enumerate(traces):
        trace_id = str(trace.get("id") or "")
        label = str(trace.get("label") or "")
        steps = trace.get("steps")
        if not trace_id or trace_id in ids:
            raise TraceFormatError(f"trace {index} has a missing or duplicate id")
        if not label:
            raise TraceFormatError(f"trace {trace_id} has no label")
        if not isinstance(steps, list) or not steps:
            raise TraceFormatError(f"trace {trace_id} has no execution steps")
        ids.add(trace_id)
        labels.add(label)
    if len(labels) < 2:
        raise TraceFormatError("labeled trace world needs at least two labels")


def parse_labeled_trace_source(retrieved: bytes, text: str = "") -> tuple[list[dict[str, Any]], str]:
    if retrieved.startswith(b"PK\x03\x04"):
        traces = parse_live_swe_zip(retrieved)
        source_format = "live_swe_agent_release_zip"
    else:
        traces = parse_json_traces(text or retrieved.decode("utf-8"))
        source_format = "json_or_jsonl"
    validate_traces(traces)
    return traces, source_format

