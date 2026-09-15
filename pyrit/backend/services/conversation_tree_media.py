# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scoped local tree thumbnails; no remote downloads or audio/video decoding."""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pyrit.backend.models._media import build_filename, infer_mime_type
from pyrit.backend.routes.media import _validate_media_path
from pyrit.models import MEDIA_PATH_DATA_TYPES

if TYPE_CHECKING:
    from pyrit.memory.conversation_tree import TreePreviewPiece

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TreeMediaDescriptor:
    """Optional safe URLs and inexpensive display metadata."""

    media_url: str | None
    thumbnail_url: str | None
    mime_type: str | None
    filename: str | None


class ConversationTreeMedia:
    """Bound local image work separately from topology and text projection."""

    MAX_SOURCE_BYTES = 8 * 1024 * 1024
    MAX_DECODE_PIXELS = 16_000_000
    MAX_THUMBNAIL_EDGE = 256
    _IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff"})
    _IMAGE_FORMATS = ("PNG", "JPEG", "GIF", "WEBP", "BMP", "ICO", "TIFF")
    _MAX_LOCATOR_LENGTH = 8192

    def __init__(self, *, results_path: str | None) -> None:
        """Initialize without filesystem access; call methods on a worker thread."""
        self._results_path = results_path

    def describe(self, *, attack_result_id: str, piece: TreePreviewPiece, full: bool) -> TreeMediaDescriptor | None:
        """
        Describe a media piece without decoding it or downloading remote content.

        Returns:
            TreeMediaDescriptor | None: Safe media presentation, or an unavailable placeholder.
        """
        value = piece.media_value
        if piece.data_type not in MEDIA_PATH_DATA_TYPES or not value or len(value) > self._MAX_LOCATOR_LENGTH:
            return None
        parsed = urlparse(value)
        is_remote = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
        if value.startswith("data:"):
            return None
        path = None if is_remote else self._local_path(value)
        if not is_remote and path is None:
            return None
        thumbnail_url = None
        if (
            piece.data_type == "image_path"
            and path is not None
            and path.suffix.lower() in self._IMAGE_EXTENSIONS
            and path.stat().st_size <= self.MAX_SOURCE_BYTES
        ):
            thumbnail_url = f"/api/attacks/{attack_result_id}/conversation-tree/pieces/{piece.piece_id}/thumbnail"
        media_url = None
        if full:
            from pyrit.backend.mappers.attack_mappers import _resolve_media_url

            media_url = _resolve_media_url(value=value, data_type=piece.data_type)
        return TreeMediaDescriptor(
            media_url=media_url,
            thumbnail_url=thumbnail_url,
            mime_type=infer_mime_type(value=value, data_type=piece.data_type),
            filename=build_filename(
                data_type=piece.data_type, sha256=piece.converted_hash or piece.piece_id.hex, value=value
            ),
        )

    def render_thumbnail(self, piece: TreePreviewPiece) -> bytes:
        """
        Decode only a bounded local still image and return a small PNG.

        GIFs use the first frame. Video/audio and remote assets deliberately have
        no generated thumbnail. No files are written to canonical media storage.

        Returns:
            bytes: Encoded PNG with both dimensions at most 256 pixels.

        Raises:
            FileNotFoundError: If an image is unavailable or outside the decoding budget.
        """
        from PIL import Image, ImageOps, UnidentifiedImageError

        value = piece.media_value
        if piece.data_type != "image_path" or not value or len(value) > self._MAX_LOCATOR_LENGTH:
            raise FileNotFoundError("A thumbnail is not available for this piece")
        if urlparse(value).scheme in {"http", "https", "data"}:
            raise FileNotFoundError("Remote and inline media do not have server-generated thumbnails")
        path = self._local_path(value)
        if path is None or path.suffix.lower() not in self._IMAGE_EXTENSIONS:
            raise FileNotFoundError("A local image thumbnail is not available")
        try:
            with path.open("rb") as source:
                data = source.read(self.MAX_SOURCE_BYTES + 1)
            if len(data) > self.MAX_SOURCE_BYTES:
                raise ValueError("Image exceeds the thumbnail file-size budget")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data), formats=self._IMAGE_FORMATS) as image:
                    if image.width * image.height > self.MAX_DECODE_PIXELS:
                        raise ValueError("Image exceeds the thumbnail pixel budget")
                    image.seek(0)
                    ImageOps.exif_transpose(image, in_place=True)
                    image.thumbnail((self.MAX_THUMBNAIL_EDGE, self.MAX_THUMBNAIL_EDGE))
                    with io.BytesIO() as output:
                        image.convert("RGBA").save(output, format="PNG")
                        return output.getvalue()
        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            logger.warning("Thumbnail unavailable for piece %s: %s", piece.piece_id, exc)
            raise FileNotFoundError("Image is unavailable or exceeds the thumbnail decoding budget") from exc

    def _local_path(self, value: str) -> Path | None:
        # Reject UNC paths before resolve/is_file can contact a remote filesystem.
        if value.startswith(("\\\\", "//")) or "://" in value:
            raise PermissionError("Tree media must use managed local storage or an HTTP media URL")
        if not self._results_path:
            raise RuntimeError("Memory results_path is not configured")
        if "://" in self._results_path:
            return None
        allowed_root = Path(self._results_path).resolve(strict=False)
        path = _validate_media_path(path=value, allowed_root=allowed_root)
        if not path.is_file():
            logger.info("Stored tree media is unavailable at %s", path)
            return None
        return path
