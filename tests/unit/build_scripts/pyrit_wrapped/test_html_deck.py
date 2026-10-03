# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from build_scripts.pyrit_wrapped.html_deck import HtmlDeck, script_json
from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import Snapshot, SongCandidate, WorkItem, WrappedError
from build_scripts.pyrit_wrapped.preview import PreviewFiles, make_handler
from build_scripts.pyrit_wrapped.story import StoryBuilder


def test_deck_has_navigation_selected_tracks_and_inline_assets(snapshot: Snapshot) -> None:
    stats = Metrics(snapshot).calculate()
    story = StoryBuilder(stats).build()
    output = HtmlDeck(stats=stats, story=story).render()
    assert "Previous" in output and "Next" in output
    assert "Start recording mode" in output
    assert "Find this song on Spotify" in output
    assert "Download cue sheet" in output
    assert "With A Little Help" not in output
    assert "wrapped-data" in output and "WrappedRecording" in output
    assert "<iframe" not in output
    assert "iframe_api" not in output
    assert "data:image/png;base64," in output
    assert "confetti-fall" in output
    assert "prefers-reduced-motion" in output
    assert output.count('id="countdown"') == 1
    assert output.count('<aside class="mascot"') == 1
    assert "strict-origin-when-cross-origin" in output
    assert "transcript" in output.lower()


def test_json_cannot_close_its_script() -> None:
    value = {"title": "</script><script>bad()</script>&\u2028"}
    encoded = script_json(value)
    assert "</script" not in encoded
    assert "<" not in encoded
    assert json.loads(encoded) == value


def test_user_text_is_escaped(*, snapshot: Snapshot, item: WorkItem) -> None:
    changed = snapshot.model_copy(
        update={
            "contributor": snapshot.contributor.model_copy(update={"login": "<script>bad()</script>"}),
            "items": [item],
        }
    )
    stats = Metrics(changed).calculate()
    output = HtmlDeck(stats=stats, story=StoryBuilder(stats).build()).render()
    assert "<script>bad()" not in output
    assert "&lt;script&gt;" in output


def test_mismatched_story_is_rejected(snapshot: Snapshot) -> None:
    stats = Metrics(snapshot).calculate()
    story = StoryBuilder(stats).build().model_copy(update={"contributor": None})
    with pytest.raises(WrappedError, match="same snapshot"):
        HtmlDeck(stats=stats, story=story)


@pytest.mark.parametrize("video", ["short", "https://youtube.com/watch", "<script>abc</script>"])
def test_video_ids_are_not_arbitrary_urls(video: str) -> None:
    with pytest.raises(ValidationError):
        SongCandidate(title="Track", artist="Artist", rationale="Cue", youtube_id=video)


def test_clip_bounds_are_validated() -> None:
    with pytest.raises(ValidationError, match="end after"):
        SongCandidate(title="Track", artist="Artist", rationale="Cue", start_seconds=10, end_seconds=5)


def test_only_report_files_are_resolved(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("Deck", encoding="utf-8")
    (tmp_path / "secret.env").write_text("not served", encoding="utf-8")
    files = PreviewFiles(tmp_path)
    assert files.resolve("/") == tmp_path / "index.html"
    assert files.resolve("/index.html?ignored=1") == tmp_path / "index.html"
    assert files.resolve("/secret.env") is None
    assert files.resolve("/../secret.env") is None
    assert files.resolve("/%2e%2e/secret.env") is None


def test_preview_is_readonly_local_and_preserves_referrer(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("Deck", encoding="utf-8")
    files = PreviewFiles(tmp_path)
    with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(files)) as server:
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200
            assert response.getheader("Referrer-Policy") == "strict-origin-when-cross-origin"
            assert response.getheader("Access-Control-Allow-Origin") is None
            assert "connect-src 'none'" in (response.getheader("Content-Security-Policy") or "")
            assert response.read() == b"Deck"
            connection.request("GET", "/", headers={"Host": "evil.example"})
            response = connection.getresponse()
            assert response.status == 403
            response.read()
            connection.request("POST", "/")
            response = connection.getresponse()
            assert response.status == 501
            response.read()
        finally:
            connection.close()
            server.shutdown()
            worker.join()


def test_javascript_controller_behaviors() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the actual JavaScript behavior tests.")
    script = Path(__file__).with_name("recording.test.cjs")
    result = subprocess.run([node, "--test", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
