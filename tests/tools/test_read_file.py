from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.tools.base import ToolError
from vibe.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileState,
    ReadFileToolConfig,
)


@pytest.mark.asyncio
async def test_read_file_rejects_absolute_outside(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("classified", encoding="utf-8")

    tool = ReadFile(
        config=ReadFileToolConfig(workdir=workspace),
        state=ReadFileState(),
    )

    with pytest.raises(ToolError, match="outside of the project directory"):
        await tool.run(ReadFileArgs(path=str(secret_file)))


@pytest.mark.asyncio
async def test_read_file_rejects_relative_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "elsewhere.txt"
    target.write_text("nope", encoding="utf-8")

    tool = ReadFile(
        config=ReadFileToolConfig(workdir=workspace),
        state=ReadFileState(),
    )

    escape_path = workspace / ".." / target.name
    with pytest.raises(ToolError, match="outside of the project directory"):
        await tool.run(ReadFileArgs(path=str(escape_path)))
