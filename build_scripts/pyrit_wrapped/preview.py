# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from build_scripts.pyrit_wrapped.models import WrappedError

if TYPE_CHECKING:
    from pathlib import Path


class PreviewFiles:
    TYPES = {
        "index.html": "text/html; charset=utf-8",
        "summary.md": "text/plain; charset=utf-8",
        "activity.md": "text/plain; charset=utf-8",
        "songs.md": "text/plain; charset=utf-8",
        "stats.json": "application/json",
        "story.json": "application/json",
        "snapshot.json": "application/json",
    }

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        if not (self.directory / "index.html").is_file():
            raise WrappedError("Report directory has no index.html. Regenerate it with summarize first.")

    def resolve(self, request: str) -> Path | None:
        path = urlparse(request).path
        name = "index.html" if path == "/" else path.removeprefix("/")
        if name not in self.TYPES:
            return None
        candidate = (self.directory / name).resolve()
        if candidate.parent != self.directory or not candidate.is_file():
            return None
        return candidate


def make_handler(files: PreviewFiles) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._respond(head=False)

        def do_HEAD(self) -> None:  # noqa: N802
            self._respond(head=True)

        def _respond(self, *, head: bool) -> None:
            if not re.fullmatch(r"(?:127\.0\.0\.1|localhost)(?::\d{1,5})?", self.headers.get("Host", "")):
                self.send_error(403, "Local preview requires a loopback Host.")
                return
            file = files.resolve(self.path)
            if file is None:
                self.send_error(404, "Only generated report files are served.")
                return
            try:
                content = file.read_bytes()
            except OSError as error:
                self.log_error("Cannot read generated report file %s: %s", file.name, error)
                self.send_error(500, "Cannot read generated report. Regenerate it and retry.")
                return
            self.send_response(200)
            self.send_header("Content-Type", files.TYPES[file.name])
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline' https://www.youtube.com https://s.ytimg.com; "
                "style-src 'unsafe-inline'; frame-src https://www.youtube.com https://www.youtube-nocookie.com; "
                "img-src data:; connect-src https://www.youtube.com; base-uri 'none'; form-action 'none'",
            )
            self.end_headers()
            if not head:
                self.wfile.write(content)

    return Handler


def serve_preview(*, report_dir: Path, port: int) -> int:
    if not 0 <= port <= 65535:
        raise WrappedError("Preview port must be between 0 and 65535.")
    files = PreviewFiles(report_dir)
    with ThreadingHTTPServer(("127.0.0.1", port), make_handler(files)) as server:
        print(f"PyRIT Wrapped preview: http://127.0.0.1:{server.server_address[1]}/", flush=True)
        print("Only this generated report is served. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
    return 0
