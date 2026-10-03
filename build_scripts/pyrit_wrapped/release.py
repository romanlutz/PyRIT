# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import AwareDatetime, BaseModel

from build_scripts.pyrit_wrapped.churn import parse_numstat
from build_scripts.pyrit_wrapped.models import ReleaseBoundary, ReleaseRange, WrappedError

if TYPE_CHECKING:
    from build_scripts.pyrit_wrapped.github_client import GitHubClient


class ApiRelease(BaseModel):
    tag_name: str
    published_at: AwareDatetime
    html_url: str
    draft: bool
    prerelease: bool = False


class ApiGitObject(BaseModel):
    type: str
    sha: str


class ApiGitRef(BaseModel):
    object: ApiGitObject


class ReleaseResolver:
    def __init__(self, client: GitHubClient) -> None:
        self.client = client
        self.root = Path(__file__).resolve().parents[2]

    async def resolve_async(self, *, base_tag: str | None, head_tag: str) -> ReleaseRange:
        if base_tag is None:
            records = [
                ApiRelease.model_validate(item)
                for item in await self.client.list_async(path="repos/microsoft/PyRIT/releases")
            ]
            target = next((item for item in records if item.tag_name == head_tag and not item.draft), None)
            if target is None:
                raise WrappedError("Requested release is not a published repository release.")
            earlier = [
                item
                for item in records
                if not item.draft
                and item.published_at < target.published_at
                and (target.prerelease or not item.prerelease)
            ]
            if not earlier:
                raise WrappedError("No previous release exists; provide --since-release.")
            base_tag = max(earlier, key=lambda item: item.published_at).tag_name
        base = await self._boundary_async(base_tag)
        head = await self._boundary_async(head_tag)
        if base.published_at >= head.published_at or base.commit == head.commit:
            raise WrappedError("Release tags must identify different commits and ordered publication dates.")
        for boundary in (base, head):
            await self._ensure_commit_async(boundary.commit)
        _, diff = await self._git_async(
            args=[
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--find-renames",
                "--numstat",
                "-z",
                base.commit,
                head.commit,
                "--",
            ]
        )
        _, commits = await self._git_async(args=["rev-list", "--reverse", f"{base.commit}..{head.commit}"])
        _, first_parent = await self._git_async(
            args=["rev-list", "--first-parent", "--reverse", f"{base.commit}..{head.commit}"]
        )
        ancestry, _ = await self._git_async(
            args=["merge-base", "--is-ancestor", base.commit, head.commit], accepted={0, 1}
        )
        return ReleaseRange(
            base=base,
            head=head,
            files=parse_numstat(diff),
            commit_ids=commits.decode().splitlines(),
            first_parent_commits=first_parent.decode().splitlines(),
            base_is_ancestor=ancestry == 0,
        )

    async def _boundary_async(self, tag: str) -> ReleaseBoundary:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", tag) or ".." in tag or tag.endswith("/"):
            raise WrappedError("Enter a release tag, not a Git option, SHA expression, or URL.")
        escaped = quote(tag, safe="")
        release = ApiRelease.model_validate(
            (await self.client.get_async(path=f"repos/microsoft/PyRIT/releases/tags/{escaped}")).data
        )
        if release.draft or release.tag_name != tag:
            raise WrappedError("Only exact published release tags are supported.")
        current = ApiGitRef.model_validate(
            (await self.client.get_async(path=f"repos/microsoft/PyRIT/git/ref/tags/{escaped}")).data
        ).object
        for _ in range(8):
            if current.type == "commit":
                return ReleaseBoundary(
                    tag=tag, commit=current.sha, published_at=release.published_at, html_url=release.html_url
                )
            if current.type != "tag" or not re.fullmatch(r"[0-9a-f]{40}", current.sha):
                raise WrappedError("Release tag does not resolve to a commit.")
            current = ApiGitRef.model_validate(
                (await self.client.get_async(path=f"repos/microsoft/PyRIT/git/tags/{current.sha}")).data
            ).object
        raise WrappedError("Release tag has an excessive annotated-tag chain.")

    async def _ensure_commit_async(self, sha: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise WrappedError("GitHub returned an invalid release commit.")
        code, _ = await self._git_async(args=["cat-file", "-e", f"{sha}^{{commit}}"], accepted={0, 1, 128})
        if code:
            await self._git_async(args=["fetch", "--no-tags", "--filter=blob:none", "origin", sha])

    async def _git_async(self, *, args: list[str], accepted: set[int] | None = None) -> tuple[int, bytes]:
        process = await asyncio.create_subprocess_exec(
            "git", *args, cwd=self.root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            output, error = await asyncio.wait_for(process.communicate(), timeout=300)
        except TimeoutError as failure:
            raise WrappedError(
                "Git release inspection timed out; retry once required objects are available."
            ) from failure
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        code = process.returncode or 0
        if code not in (accepted or {0}):
            raise WrappedError(f"Git release inspection failed: {error.decode('utf-8', errors='replace').strip()}")
        return code, output
