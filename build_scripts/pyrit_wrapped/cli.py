# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from build_scripts.pyrit_wrapped.github_client import GitHubClient, write_json_atomic
from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import (
    CollectionSession,
    Period,
    ReleaseRange,
    Snapshot,
    TaxonomyConfig,
    WrappedError,
    parse_contributor,
)
from build_scripts.pyrit_wrapped.release import ReleaseResolver
from build_scripts.pyrit_wrapped.render import write_reports
from build_scripts.pyrit_wrapped.snapshot import Collector
from build_scripts.pyrit_wrapped.story import StoryBuilder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact, evidence-backed PyRIT contributor and release summaries.")
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize", help="Generate facts and summaries, stopping before HTML and music.")
    source = summarize.add_mutually_exclusive_group(required=True)
    source.add_argument("--contributor", help="GitHub username, @username, or profile URL.")
    source.add_argument("--snapshot", type=Path, help="Replay a saved complete snapshot offline.")
    source.add_argument("--release", help="Published release tag to wrap across all contributors.")
    summarize.add_argument(
        "--since-release", help="Base release tag; defaults to the previous published stable release."
    )
    summarize.add_argument("--year", type=int, help="Calendar year, defaulting to the current UTC year.")
    summarize.add_argument(
        "--output-dir", type=Path, help="New report directory; existing directories are not overwritten."
    )
    summarize.add_argument("--cache-dir", type=Path, default=Path(".cache") / "pyrit_wrapped")
    summarize.add_argument(
        "--refresh", action="store_true", help="Bypass cached requests and refresh live GitHub data."
    )
    summarize.add_argument("--taxonomy", type=Path, help="Custom versioned taxonomy JSON, for live collection.")
    return parser


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def _collect_async(
    *, login: str | None, cache_dir: Path, taxonomy: TaxonomyConfig, period: Period, release: ReleaseRange | None = None
) -> Snapshot:
    client = GitHubClient(cache_dir=cache_dir, progress=_progress)
    collector = Collector(client=client, progress=_progress)
    return await collector.collect_async(login=login, period=period, taxonomy=taxonomy, release=release)


def _load_snapshot(args: argparse.Namespace) -> Snapshot:
    if args.snapshot is not None:
        if args.refresh or args.taxonomy is not None or args.since_release is not None:
            raise WrappedError(
                "--snapshot is offline and retains its taxonomy; do not combine it with --refresh/--taxonomy."
            )
        snapshot = Snapshot.model_validate_json(args.snapshot.read_text(encoding="utf-8"))
        if args.year is not None and args.year != snapshot.period.year:
            raise WrappedError("--year must match the offline snapshot's year.")
        return snapshot
    now = datetime.now(UTC)
    taxonomy_path = args.taxonomy or Path(__file__).with_name("taxonomy.json")
    taxonomy = TaxonomyConfig.model_validate_json(taxonomy_path.read_text(encoding="utf-8"))
    if args.release is not None:
        if args.year is not None:
            raise WrappedError("Release mode uses publication dates, not --year.")
        client = GitHubClient(cache_dir=args.cache_dir / "release-resolution", refresh=args.refresh, progress=_progress)
        release = asyncio.run(ReleaseResolver(client).resolve_async(base_tag=args.since_release, head_tag=args.release))
        period = Period.for_release(start=release.base.published_at, end=release.head.published_at)
        return _load_live_snapshot(args=args, period=period, taxonomy=taxonomy, release=release)
    if args.since_release is not None:
        raise WrappedError("--since-release requires --release.")
    period = Period.for_year(year=args.year if args.year is not None else now.year, now=now)
    return _load_live_snapshot(args=args, period=period, taxonomy=taxonomy)


def _load_live_snapshot(
    *, args: argparse.Namespace, period: Period, taxonomy: TaxonomyConfig, release: ReleaseRange | None = None
) -> Snapshot:
    login = parse_contributor(args.contributor) if args.contributor is not None else None
    profile_key = (
        f"{login.lower()}-{period.year}"
        if login
        else f"release-{release.base.commit}-{release.head.commit}"
        if release
        else ""
    )
    profile_dir = args.cache_dir / profile_key
    manifest = profile_dir / "session.json"
    session = _collection_session(
        path=manifest, period=period, taxonomy=taxonomy, refresh=args.refresh, release=release
    )
    run_dir = profile_dir / session.identifier
    snapshot_path = run_dir / "snapshot.json"
    if session.complete:
        snapshot = Snapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
        if (
            snapshot.period != session.period
            or snapshot.taxonomy != session.taxonomy
            or snapshot.release != session.release
        ):
            raise WrappedError("Cached snapshot disagrees with its collection session; use --refresh.")
        _progress(
            f"Replaying completed snapshot with cutoff {snapshot.period.cutoff.isoformat()}; "
            "use --refresh for live data."
        )
        return snapshot
    _progress(f"Collecting/resuming with fixed cutoff {session.period.cutoff.isoformat()}.")
    snapshot = asyncio.run(
        _collect_async(
            login=login,
            cache_dir=run_dir / "requests",
            taxonomy=taxonomy,
            period=session.period,
            release=session.release,
        )
    )
    if snapshot.period != session.period or snapshot.taxonomy != session.taxonomy:
        raise WrappedError("Collected snapshot disagrees with its requested period/taxonomy.")
    write_json_atomic(path=snapshot_path, content=snapshot.model_dump_json())
    session.complete = True
    write_json_atomic(path=manifest, content=session.model_dump_json())
    return snapshot


def _collection_session(
    *, path: Path, period: Period, taxonomy: TaxonomyConfig, refresh: bool, release: ReleaseRange | None = None
) -> CollectionSession:
    now = datetime.now(UTC)
    if path.exists() and not refresh:
        session = CollectionSession.model_validate_json(path.read_text(encoding="utf-8"))
        if session.period.year != period.year:
            raise WrappedError("Collection session has a different reporting year; use --refresh.")
        age = now - session.started_at
        if age < timedelta(0):
            raise WrappedError("Collection session has a future timestamp; use --refresh.")
        same_release = (
            session.release.model_dump(exclude={"shipped_pr_numbers"})
            == release.model_dump(exclude={"shipped_pr_numbers"})
            if session.release is not None and release is not None
            else session.release is release
        )
        if age < timedelta(hours=24) and session.taxonomy == taxonomy and session.schema_version == 2 and same_release:
            return session
    session = CollectionSession(
        identifier=uuid.uuid4().hex, started_at=now, period=period, taxonomy=taxonomy, release=release
    )
    write_json_atomic(path=path, content=session.model_dump_json())
    return session


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        snapshot = _load_snapshot(args)
        stats = Metrics(snapshot).calculate()
        story = StoryBuilder(stats).build()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        name = (
            f"{snapshot.contributor.login}-{snapshot.period.year}"
            if snapshot.contributor is not None
            else f"release-{snapshot.release.head.tag.replace('/', '-')}"
            if snapshot.release
            else "wrapped"
        )
        output = args.output_dir or Path("results") / "wrapped" / f"{name}-{stamp}"
        destination = write_reports(snapshot=snapshot, stats=stats, story=story, output_dir=output)
    except (WrappedError, ValidationError, ValueError, OSError) as error:
        print(f"PyRIT Wrapped: {error}", file=sys.stderr)
        return 1
    print(f"Summaries written to {destination}")
    print("Review summary.md, story.json, and songs.md. Song candidates are references, not licensed audio.")
    return 0
