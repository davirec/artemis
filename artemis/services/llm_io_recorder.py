# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-request Gemini I/O recorder.

Every ``generate_content`` HTTP request sent through the ``google-genai`` SDK
(LangChain ``ChatGoogleGenerativeAI`` as well as the native Explorer / Video
Analyzer calls) is written to the active session's trace directory, mirroring
what Google AI Studio shows for a logged request: System Instructions, Input
and Output.

Layout, under ``<traces>/<session_id>/llm_calls/``::

    0001_operator.json   full request/response (machine readable)
    0001_operator.md     System Instructions / Input / Output (human readable)
    index.jsonl          one summary line per call
    media/<sha>.<ext>    inline images / blobs, deduplicated by content hash

The SDK's private ``_generate_content`` / ``_generate_content_stream`` methods
are patched (not the public ones) because they map one-to-one to HTTP
requests: automatic function calling loops produce one record per round trip.
Recording is best-effort and can never fail or alter the model call.
Disable it with ``ARTEMIS_LLM_IO_LOG=false``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import functools
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

from artemis.data_engine.context_vars import CURRENT_NODE_NAME, CURRENT_TRACE_ID

logger = logging.getLogger(__name__)

LLM_CALLS_DIRNAME = "llm_calls"
MEDIA_DIRNAME = "media"

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_SEQ_LOCK = threading.Lock()
_SEQ_COUNTERS: dict[str, int] = {}

_MIME_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "video/mp4": "mp4",
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "application/pdf": "pdf",
    "text/plain": "txt",
}


# --------------------------------------------------------------------------- #
# Enablement / installation
# --------------------------------------------------------------------------- #


def is_enabled() -> bool:
    try:
        from artemis.config import settings

        return bool(getattr(settings, "ARTEMIS_LLM_IO_LOG", True))
    except Exception:  # pylint: disable=broad-exception-caught
        return True


def install() -> bool:
    """Patch the google-genai SDK once per process. Returns True when active."""
    global _INSTALLED
    if not is_enabled():
        return False
    with _INSTALL_LOCK:
        # Idempotent per method (checks the wrapper marker) rather than via a
        # global flag, so a method replaced after install gets wrapped again.
        try:
            from google.genai import models as genai_models
        except ImportError:
            logger.debug("google-genai not installed; LLM I/O recorder inactive.")
            return False

        patches = (
            (genai_models.Models, "_generate_content", _wrap_sync),
            (genai_models.Models, "_generate_content_stream", _wrap_sync_stream),
            (genai_models.AsyncModels, "_generate_content", _wrap_async),
            (genai_models.AsyncModels, "_generate_content_stream", _wrap_async_stream),
        )
        for cls, attr, wrapper_factory in patches:
            original = getattr(cls, attr, None)
            if original is None or getattr(original, "_artemis_io_wrapped", False):
                continue
            wrapped = wrapper_factory(original)
            wrapped._artemis_io_wrapped = True  # type: ignore[attr-defined]
            wrapped._artemis_io_original = original  # type: ignore[attr-defined]
            setattr(cls, attr, wrapped)
        if not _INSTALLED:
            logger.info("LLM I/O recorder installed (google-genai generate_content).")
        _INSTALLED = True
        return True


def uninstall() -> None:
    """Restore the original SDK methods (used by tests)."""
    global _INSTALLED
    with _INSTALL_LOCK:
        try:
            from google.genai import models as genai_models
        except ImportError:
            return
        for cls in (genai_models.Models, genai_models.AsyncModels):
            for attr in ("_generate_content", "_generate_content_stream"):
                current = cls.__dict__.get(attr)
                original = getattr(current, "_artemis_io_original", None)
                if original is not None:
                    setattr(cls, attr, original)
        _INSTALLED = False


# --------------------------------------------------------------------------- #
# SDK wrappers
# --------------------------------------------------------------------------- #


def _wrap_sync(original):
    @functools.wraps(original)
    def wrapper(self, *, model, contents, config=None):
        rec = _CallRecord.begin(model, contents, config, streamed=False)
        try:
            response = original(self, model=model, contents=contents, config=config)
        except BaseException as exc:
            _safe(rec, "fail", exc)
            raise
        _safe(rec, "finish", [response])
        return response

    return wrapper


def _wrap_sync_stream(original):
    @functools.wraps(original)
    def wrapper(self, *, model, contents, config=None):
        rec = _CallRecord.begin(model, contents, config, streamed=True)
        chunks: list[Any] = []
        try:
            for chunk in original(self, model=model, contents=contents, config=config):
                chunks.append(chunk)
                yield chunk
        except GeneratorExit:
            _safe(rec, "finish", chunks, status="incomplete")
            raise
        except BaseException as exc:
            _safe(rec, "fail", exc, chunks)
            raise
        _safe(rec, "finish", chunks)

    return wrapper


def _wrap_async(original):
    @functools.wraps(original)
    async def wrapper(self, *, model, contents, config=None):
        rec = _CallRecord.begin(model, contents, config, streamed=False)
        try:
            response = await original(self, model=model, contents=contents, config=config)
        except BaseException as exc:
            _safe(rec, "fail", exc)
            raise
        _safe(rec, "finish", [response])
        return response

    return wrapper


def _wrap_async_stream(original):
    @functools.wraps(original)
    async def wrapper(self, *, model, contents, config=None):
        rec = _CallRecord.begin(model, contents, config, streamed=True)
        try:
            stream = await original(self, model=model, contents=contents, config=config)
        except BaseException as exc:
            _safe(rec, "fail", exc)
            raise
        return _recording_async_iter(stream, rec)

    return wrapper


async def _recording_async_iter(stream, rec: _CallRecord | None):
    chunks: list[Any] = []
    try:
        async for chunk in stream:
            chunks.append(chunk)
            yield chunk
    except GeneratorExit:
        _safe(rec, "finish", chunks, status="incomplete")
        raise
    except BaseException as exc:
        _safe(rec, "fail", exc, chunks)
        raise
    _safe(rec, "finish", chunks)


def _safe(rec: _CallRecord | None, method: str, *args, **kwargs) -> None:
    if rec is None:
        return
    try:
        getattr(rec, method)(*args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("LLM I/O recorder could not write call record: %s", exc)


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #


def _current_engine():
    engine_module = sys.modules.get("artemis.data_engine.engine")
    return getattr(engine_module, "_CURRENT_DATA_ENGINE", None) if engine_module else None


class _CallRecord:
    """Request side of one model call, captured synchronously at call time."""

    def __init__(self, out_dir: Path, meta: dict[str, Any], model, contents, config):
        self.out_dir = out_dir
        self.meta = meta
        self.model = model
        self.contents = contents
        self.config = config
        self.started = time.time()

    @classmethod
    def begin(cls, model, contents, config, *, streamed: bool) -> _CallRecord | None:
        try:
            if not is_enabled():
                return None
            engine = _current_engine()
            if engine is None or not getattr(engine, "current_session_id", None):
                return None
            out_dir = Path(engine.base_dir) / LLM_CALLS_DIRNAME
            step_id = getattr(engine, "current_step_id", None)
            step_number = None
            try:
                step_number = engine.get_step_number(step_id) if step_id else None
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            trace_id = CURRENT_TRACE_ID.get()
            meta = {
                "session_id": str(engine.current_session_id),
                "agent": CURRENT_NODE_NAME.get() or "llm",
                "step_id": str(step_id) if step_id else None,
                "step_number": step_number,
                "parent_trace_id": str(trace_id) if trace_id else None,
                "model": str(model),
                "streamed": streamed,
            }
            return cls(out_dir, meta, model, contents, config)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("LLM I/O recorder skipped a call: %s", exc, exc_info=True)
            return None

    # -- completion ------------------------------------------------------- #

    def finish(self, responses: list[Any], *, status: str = "success") -> None:
        self._write(status=status, output=_merge_responses(responses), error=None)

    def fail(self, error: BaseException, partial: list[Any] | None = None) -> None:
        status = "cancelled" if type(error).__name__ == "CancelledError" else "error"
        output = _merge_responses(partial) if partial else None
        self._write(status=status, output=output, error=f"{type(error).__name__}: {error}")

    def _write(self, *, status: str, output: dict[str, Any] | None, error: str | None) -> None:
        duration = time.time() - self.started
        self.out_dir.mkdir(parents=True, exist_ok=True)
        media = _MediaStore(self.out_dir / MEDIA_DIRNAME)

        config = _config_dict(self.config)
        system_instruction = config.pop("system_instruction", None)
        tools = config.pop("tools", None)

        seq = _next_seq(self.out_dir)
        stem = f"{seq:04d}_{_slug(self.meta['agent'])}"
        record = {
            "seq": seq,
            "timestamp": datetime.fromtimestamp(self.started).astimezone().isoformat(),
            "duration_seconds": round(duration, 3),
            "status": status,
            **self.meta,
            "system_instruction": media.externalize(_content_dump(system_instruction)),
            "input": media.externalize(_contents_dump(self.contents)),
            "output": media.externalize(output),
            "config": media.externalize(config),
            "tools": media.externalize(tools),
            "error": error,
        }

        (self.out_dir / f"{stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (self.out_dir / f"{stem}.md").write_text(_render_markdown(record), encoding="utf-8")

        usage = (output or {}).get("usage_metadata") or {}
        summary = {
            "seq": seq,
            "file": f"{stem}.json",
            "timestamp": record["timestamp"],
            "agent": self.meta["agent"],
            "step_number": self.meta["step_number"],
            "model": self.meta["model"],
            "status": status,
            "duration_seconds": record["duration_seconds"],
            "prompt_tokens": usage.get("prompt_token_count"),
            "output_tokens": usage.get("candidates_token_count"),
            "thought_tokens": usage.get("thoughts_token_count"),
            "cached_tokens": usage.get("cached_content_token_count"),
        }
        with _SEQ_LOCK, (self.out_dir / "index.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")


def _next_seq(out_dir: Path) -> int:
    key = str(out_dir)
    with _SEQ_LOCK:
        if key not in _SEQ_COUNTERS:
            # Resumed sessions keep numbering after the files already there.
            existing = [
                int(m.group(1))
                for p in out_dir.glob("*.json")
                if (m := re.match(r"^(\d+)_", p.name))
            ]
            _SEQ_COUNTERS[key] = max(existing, default=0)
        _SEQ_COUNTERS[key] += 1
        return _SEQ_COUNTERS[key]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:60] or "llm"


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def _dump(obj: Any) -> Any:
    """Pydantic-aware recursive dump that keeps bytes as bytes."""
    if hasattr(obj, "model_dump"):
        try:
            return _dump(obj.model_dump(mode="python", exclude_none=True))
        except Exception:  # pylint: disable=broad-exception-caught
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _dump(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_dump(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (str, int, float, bool, bytes)) or obj is None:
        return obj
    return str(obj)


def _contents_dump(contents: Any) -> Any:
    try:
        from google.genai import _transformers as t

        return [_dump(c) for c in t.t_contents(contents)]
    except Exception:  # pylint: disable=broad-exception-caught
        return _dump(contents)


def _content_dump(content: Any) -> Any:
    if content is None:
        return None
    try:
        from google.genai import _transformers as t

        return _dump(t.t_content(content))
    except Exception:  # pylint: disable=broad-exception-caught
        return _dump(content)


def _config_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        try:
            from google.genai import types

            config = types.GenerateContentConfig.model_validate(config)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    raw = config.system_instruction if hasattr(config, "system_instruction") else None
    dumped = _dump(config)
    if not isinstance(dumped, dict):
        return {"raw": dumped}
    # Keep the original object so _content_dump can normalize str/Part/list forms.
    if raw is not None:
        dumped["system_instruction"] = raw
    dumped.pop("http_options", None)
    return dumped


def _merge_responses(responses: list[Any] | None) -> dict[str, Any] | None:
    """Collapse one response or a list of stream chunks into one output."""
    if not responses:
        return None
    candidates: dict[int, dict[str, Any]] = {}
    result: dict[str, Any] = {}
    for response in responses:
        data = _dump(response)
        if not isinstance(data, dict):
            continue
        for key in ("usage_metadata", "model_version", "response_id", "prompt_feedback"):
            if data.get(key):
                result[key] = data[key]
        for position, cand in enumerate(data.get("candidates") or []):
            index = cand.get("index", position)
            merged = candidates.setdefault(index, {"index": index, "parts": []})
            content = cand.get("content") or {}
            if content.get("role"):
                merged["role"] = content["role"]
            for part in content.get("parts") or []:
                _append_part(merged["parts"], part)
            for key in ("finish_reason", "finish_message", "safety_ratings", "grounding_metadata"):
                if cand.get(key):
                    merged[key] = cand[key]
    result["candidates"] = [candidates[i] for i in sorted(candidates)]
    if len(responses) > 1:
        result["stream_chunks"] = len(responses)
    return result


def _append_part(parts: list[dict[str, Any]], part: dict[str, Any]) -> None:
    """Concatenate consecutive streamed text parts of the same kind."""
    text = part.get("text")
    if text is not None and parts:
        last = parts[-1]
        if (
            "text" in last
            and bool(last.get("thought")) == bool(part.get("thought"))
            and set(last) - {"thought_signature"} <= {"text", "thought"}
        ):
            last["text"] += text
            if part.get("thought_signature"):
                last["thought_signature"] = part["thought_signature"]
            return
    parts.append(dict(part))


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #


class _MediaStore:
    """Writes inline blobs to media/ and swaps them for relative references."""

    def __init__(self, media_dir: Path):
        self.media_dir = media_dir

    def externalize(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            data, mime = obj.get("data"), obj.get("mime_type")
            if isinstance(data, (bytes, bytearray)) and mime is not None:
                return {
                    **{k: v for k, v in obj.items() if k != "data"},
                    "file": self._save(data, mime),
                }
            return {k: self.externalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.externalize(v) for v in obj]
        if isinstance(obj, (bytes, bytearray)):
            return f"<bytes length={len(obj)}>"
        return obj

    def _save(self, data: bytes, mime: str) -> str:
        digest = hashlib.sha256(data).hexdigest()[:24]
        ext = _MIME_EXTENSIONS.get(str(mime).lower(), "bin")
        name = f"{digest}.{ext}"
        path = self.media_dir / name
        if not path.exists():
            self.media_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(data))
        return f"{MEDIA_DIRNAME}/{name}"


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def _fence(text: str, lang: str = "") -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _render_part(part: dict[str, Any]) -> str:
    if "text" in part:
        label = "**Thought**\n\n" if part.get("thought") else ""
        return label + _fence(str(part["text"]))
    if "inline_data" in part:
        blob = part["inline_data"]
        ref = blob.get("file", "")
        mime = blob.get("mime_type", "")
        if str(mime).startswith("image/"):
            return f"![{mime}]({ref})"
        return f"[{mime} attachment]({ref})"
    if "file_data" in part:
        fd = part["file_data"]
        return f"File: `{fd.get('file_uri')}` ({fd.get('mime_type', '')})"
    if "function_call" in part:
        fc = part["function_call"]
        return f"**Function call** `{fc.get('name')}`\n\n" + _fence(
            json.dumps(fc.get("args", {}), ensure_ascii=False, indent=2, default=str), "json"
        )
    if "function_response" in part:
        fr = part["function_response"]
        return f"**Function response** `{fr.get('name')}`\n\n" + _fence(
            json.dumps(fr.get("response", {}), ensure_ascii=False, indent=2, default=str), "json"
        )
    shown = {k: v for k, v in part.items() if k != "thought_signature"}
    if not shown:
        return ""
    return _fence(json.dumps(shown, ensure_ascii=False, indent=2, default=str), "json")


def _render_content(content: Any) -> str:
    if not isinstance(content, dict):
        return _fence(json.dumps(content, ensure_ascii=False, indent=2, default=str), "json")
    rendered = [_render_part(p) for p in content.get("parts") or [] if isinstance(p, dict)]
    return "\n\n".join(r for r in rendered if r)


def _render_markdown(record: dict[str, Any]) -> str:
    lines = [
        f"# LLM call {record['seq']:04d} — {record['agent']}",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in (
        "timestamp",
        "model",
        "status",
        "duration_seconds",
        "step_number",
        "streamed",
        "session_id",
        "step_id",
        "parent_trace_id",
    ):
        if record.get(key) is not None:
            lines.append(f"| {key} | `{record[key]}` |")
    lines.append("")

    if record.get("error"):
        lines += ["## Error", "", _fence(str(record["error"])), ""]

    lines += ["## System Instructions", ""]
    lines.append(
        _render_content(record["system_instruction"])
        if record.get("system_instruction")
        else "_(none)_"
    )
    if record.get("config", {}).get("cached_content"):
        lines += [
            "",
            f"_Uses cached content `{record['config']['cached_content']}`: part of the prompt lives in the cache._",
        ]
    lines.append("")

    lines += ["## Input", ""]
    for i, content in enumerate(record.get("input") or [], start=1):
        role = content.get("role", "user") if isinstance(content, dict) else "user"
        lines += [f"### [{i}] {role}", "", _render_content(content), ""]

    lines += ["## Output", ""]
    output = record.get("output") or {}
    candidates = output.get("candidates") or []
    if not candidates:
        lines += ["_(no output)_", ""]
    for cand in candidates:
        if len(candidates) > 1:
            lines += [f"### Candidate {cand.get('index', 0)}", ""]
        lines += [_render_content(cand), ""]
        if cand.get("finish_reason"):
            lines += [f"_finish_reason: `{cand['finish_reason']}`_", ""]

    if output.get("usage_metadata"):
        lines += [
            "## Usage",
            "",
            _fence(
                json.dumps(output["usage_metadata"], ensure_ascii=False, indent=2, default=str),
                "json",
            ),
            "",
        ]
    if record.get("config"):
        lines += [
            "## Config",
            "",
            _fence(json.dumps(record["config"], ensure_ascii=False, indent=2, default=str), "json"),
            "",
        ]
    if record.get("tools"):
        lines += ["_Tool declarations are in the .json file (`tools`)._", ""]
    return "\n".join(lines)
