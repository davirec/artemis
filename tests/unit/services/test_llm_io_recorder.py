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

import json
from pathlib import Path
import uuid

from google.genai import models as genai_models
from google.genai import types
import pytest

import artemis.data_engine.engine as engine_module
from artemis.data_engine.context_vars import CURRENT_NODE_NAME
from artemis.services import llm_io_recorder


class _FakeEngine:
    def __init__(self, root: Path):
        self.current_session_id = uuid.uuid4()
        self.current_step_id = uuid.uuid4()
        self._root = root

    @property
    def base_dir(self) -> Path:
        return self._root / str(self.current_session_id)

    def get_step_number(self, step_id):
        return 3


def _response(text: str, *, thought: bool = False, usage: bool = False):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(role="model", parts=[types.Part(text=text, thought=thought)]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=(
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10, candidates_token_count=4
            )
            if usage
            else None
        ),
    )


@pytest.fixture
def engine(tmp_path, monkeypatch):
    fake = _FakeEngine(tmp_path)
    monkeypatch.setattr(engine_module, "_CURRENT_DATA_ENGINE", fake)
    monkeypatch.setattr(llm_io_recorder, "_SEQ_COUNTERS", {})
    yield fake
    llm_io_recorder.uninstall()


PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
CONFIG = {"system_instruction": "You are the operator.", "temperature": 0.0}
CONTENTS = [
    types.Content(
        role="user",
        parts=[types.Part(text="Tap OK"), types.Part.from_bytes(data=PNG, mime_type="image/png")],
    )
]


@pytest.mark.asyncio
async def test_async_call_writes_json_markdown_media_and_index(engine, monkeypatch):
    async def fake_generate(self, *, model, contents, config=None):
        return _response("clicked OK", usage=True)

    monkeypatch.setattr(genai_models.AsyncModels, "_generate_content", fake_generate)
    assert llm_io_recorder.install()

    token = CURRENT_NODE_NAME.set("operator")
    try:
        result = await genai_models.AsyncModels._generate_content(
            None, model="gemini-3-flash", contents=CONTENTS, config=CONFIG
        )
    finally:
        CURRENT_NODE_NAME.reset(token)

    assert result.text == "clicked OK"
    out = engine.base_dir / "llm_calls"
    record = json.loads((out / "0001_operator.json").read_text(encoding="utf-8"))
    assert record["status"] == "success"
    assert record["agent"] == "operator"
    assert record["step_number"] == 3
    assert record["system_instruction"]["parts"][0]["text"] == "You are the operator."
    image_part = record["input"][0]["parts"][1]["inline_data"]
    assert (out / image_part["file"]).read_bytes() == PNG
    assert record["output"]["candidates"][0]["parts"][0]["text"] == "clicked OK"
    assert record["config"]["temperature"] == 0.0

    markdown = (out / "0001_operator.md").read_text(encoding="utf-8")
    for section in ("## System Instructions", "## Input", "## Output", "You are the operator."):
        assert section in markdown
    assert f"]({image_part['file']})" in markdown

    index = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()]
    assert index[0]["prompt_tokens"] == 10 and index[0]["output_tokens"] == 4


@pytest.mark.asyncio
async def test_async_stream_merges_chunks(engine, monkeypatch):
    async def fake_stream(self, *, model, contents, config=None):
        async def gen():
            yield _response("think ", thought=True)
            yield _response("hard", thought=True)
            yield _response("done", usage=True)

        return gen()

    monkeypatch.setattr(genai_models.AsyncModels, "_generate_content_stream", fake_stream)
    llm_io_recorder.install()

    stream = await genai_models.AsyncModels._generate_content_stream(
        None, model="gemini-3-flash", contents="hi", config=None
    )
    chunks = [c async for c in stream]
    assert len(chunks) == 3

    record = json.loads((engine.base_dir / "llm_calls" / "0001_llm.json").read_text("utf-8"))
    parts = record["output"]["candidates"][0]["parts"]
    assert parts[0] == {"text": "think hard", "thought": True}
    assert parts[1]["text"] == "done"
    assert record["output"]["stream_chunks"] == 3
    assert record["system_instruction"] is None


def test_sync_error_is_recorded_and_reraised(engine, monkeypatch):
    def fake_generate(self, *, model, contents, config=None):
        raise RuntimeError("429 quota")

    monkeypatch.setattr(genai_models.Models, "_generate_content", fake_generate)
    llm_io_recorder.install()

    with pytest.raises(RuntimeError, match="429 quota"):
        genai_models.Models._generate_content(None, model="m", contents="hi", config=CONFIG)

    record = json.loads((engine.base_dir / "llm_calls" / "0001_llm.json").read_text("utf-8"))
    assert record["status"] == "error"
    assert "429 quota" in record["error"]


def test_no_session_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "_CURRENT_DATA_ENGINE", None)

    def fake_generate(self, *, model, contents, config=None):
        return _response("ok")

    monkeypatch.setattr(genai_models.Models, "_generate_content", fake_generate)
    llm_io_recorder.install()
    try:
        assert genai_models.Models._generate_content(None, model="m", contents="x").text == "ok"
    finally:
        llm_io_recorder.uninstall()
    assert not list(tmp_path.rglob("*.json"))
