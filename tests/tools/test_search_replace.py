from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.tools.base import ToolError
from vibe.core.tools.builtins.search_replace import (
    SearchReplace,
    SearchReplaceArgs,
    SearchReplaceConfig,
    SearchReplaceState,
)

_CONTENT_BLOCK = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"


@pytest.mark.asyncio
async def test_search_replace_rejects_absolute_outside(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "sensitive.txt"
    target.write_text("old", encoding="utf-8")

    tool = SearchReplace(
        config=SearchReplaceConfig(workdir=workspace),
        state=SearchReplaceState(),
    )

    with pytest.raises(ToolError, match="Cannot edit"):
        await tool.run(SearchReplaceArgs(file_path=str(target), content=_CONTENT_BLOCK))


@pytest.mark.asyncio
async def test_search_replace_rejects_relative_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "secret.txt"
    target.write_text("old", encoding="utf-8")

    tool = SearchReplace(
        config=SearchReplaceConfig(workdir=workspace),
        state=SearchReplaceState(),
    )

    relative_escape = workspace / ".." / target.name

    with pytest.raises(ToolError, match="Cannot edit"):
        await tool.run(
            SearchReplaceArgs(file_path=str(relative_escape), content=_CONTENT_BLOCK)
        )
