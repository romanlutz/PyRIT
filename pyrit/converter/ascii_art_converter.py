# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import cast

from art import ASCII_FONTS, text2art
from art.utils import indirect_font

from pyrit.converter.converter import Converter, ConverterResult
from pyrit.models import ComponentIdentifier, PromptDataType

logger = logging.getLogger(__name__)

# These public ASCII fonts are excluded by art 6.5's built-in "rand" mode.
_ART_RANDOM_EXCLUDED_ASCII_FONTS = {
    "5x8",
    "binary",
    "decimal",
    "dwhistled",
    "flyn_sh",
    "gauntlet",
    "high_noo",
    "hills",
    "katakana",
    "morse",
    "moscow",
    "nfi1",
    "octal",
    "rot13",
    "smtengwar",
    "tengwar",
    "tsalagi",
}
_ART_RANDOM_FONTS = sorted(set(ASCII_FONTS) - _ART_RANDOM_EXCLUDED_ASCII_FONTS)

# art's plain-random aliases, which it resolves over the pool mirrored above.
# PyRIT intercepts them to make the pick seeded and prompt-aware.
_ART_RANDOM_SELECTORS = frozenset({"rand", "random", "rnd"})


# Bounded so prompts with many distinct unrenderable characters cannot grow the
# cache without limit. Printable ASCII tops out near 34k entries across the font
# pool, so ASCII workloads never evict.
@lru_cache(maxsize=100_000)
def _font_renders_character(character: str, font: str) -> bool:
    """
    Return whether ``font`` has a glyph for ``character``.

    ``text2art`` silently omits characters without a glyph, so renderability
    is detected by rendering the single character and checking for any
    non-whitespace output. ``" "``, ``"\\n"`` and ``"\\r"`` are layout rather
    than content: the first two are laid out by ``text2art``, and ``"\\r"`` is
    line-ending encoding, so passing it through would splice a control
    character into the art block. Other whitespace (tab, no-break space) has
    no glyph and would be dropped, so it goes through the same probe.

    Args:
        character (str): The character to check.
        font (str): The art font to render with.

    Returns:
        bool: True if the font renders the character, or it is layout rather than content.
    """
    if character in " \n\r":
        return True
    return bool(text2art(character, font=font).strip())


def _fonts_that_render(text: str, fonts: Sequence[str]) -> list[str]:
    """
    Return the fonts of ``fonts`` that render every character in ``text``.

    Args:
        text (str): The text the font must render fully.
        fonts (Sequence[str]): The font names to filter.

    Returns:
        list[str]: The font names whose glyphs cover every character of ``text``.
    """
    characters = set(text)
    return [font for font in fonts if all(_font_renders_character(character, font) for character in characters)]


def _segment_by_renderability(line: str, font: str) -> list[tuple[str, bool]]:
    """
    Split ``line`` into maximal runs of renderable text and single unrenderable characters.

    Renderable characters are grouped so each run is laid out by ``text2art`` in one
    call, which preserves the font's native spacing and letter joining.

    Args:
        line (str): A single line of the prompt, without newlines.
        font (str): The art font the segments are rendered with.

    Returns:
        list[tuple[str, bool]]: ``(segment, renderable)`` pairs in input order.
    """
    segments: list[tuple[str, bool]] = []
    for character in line:
        renderable = _font_renders_character(character, font)
        if renderable and segments and segments[-1][1]:
            segments[-1] = (segments[-1][0] + character, True)
        else:
            segments.append((character, renderable))
    return segments


def _render_line_with_passthrough(line: str, font: str) -> str:
    """
    Render one line as ASCII art, splicing unrenderable characters in as literal text.

    Each unrenderable character occupies a one-character column centered against the
    art rows, so it reads as part of the line instead of being dropped.

    Args:
        line (str): A single line of the prompt, without newlines.
        font (str): The art font to render with.

    Returns:
        str: The rendered line, wrapped in newlines to match ``text2art`` output.
    """
    if not line:
        return "\n"

    columns = [
        text2art(segment, font=font).strip("\n").splitlines() if renderable else [segment]
        for segment, renderable in _segment_by_renderability(line, font)
    ]

    height = max(len(column) for column in columns)
    grid = []
    for column in columns:
        width = max((len(row) for row in column), default=0)
        top = (height - len(column)) // 2
        padded = [""] * top + column + [""] * (height - len(column) - top)
        grid.append([row.ljust(width) for row in padded])

    rows = ["".join(column[index] for column in grid) for index in range(height)]
    return "\n" + "\n".join(rows) + "\n"


def _render_with_passthrough(text: str, font: str) -> str:
    """
    Render ``text`` as ASCII art, passing characters the font cannot render through as text.

    ``text2art`` lays out each line as its own art block, so lines are rendered
    independently and concatenated the same way.

    Args:
        text (str): The prompt to render.
        font (str): The art font to render with.

    Returns:
        str: The ASCII art representation with unrenderable characters preserved.
    """
    return "".join(_render_line_with_passthrough(line, font) for line in text.split("\n"))


class AsciiArtConverter(Converter):
    """
    Uses the `art` package to convert text into ASCII art.

    ``art`` silently omits characters the selected font has no glyph for. Those
    characters are instead passed through as literal text and a warning is logged,
    so no part of the prompt is lost.
    """

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)

    def __init__(self, *, font: str = "rand") -> None:
        """
        Initialize the converter with a specified font.

        Args:
            font (str): The font to use for ASCII art. Defaults to "rand" which selects a random
                font. Indirect selectors such as "wizard" or "rnd-small" are resolved to a single
                concrete font per conversion.
        """
        self._font = font

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the converter identifier with font parameter.

        Returns:
            ComponentIdentifier: The identifier for this converter.
        """
        return self._create_identifier(
            params={
                "font": self._font,
            },
        )

    def _resolve_font(self, *, prompt: str) -> str:
        """
        Resolve the configured font to a concrete font name.

        ``art`` re-resolves indirect selectors such as ``"wizard"`` or ``"rnd-small"``
        on every ``text2art`` call, so leaving one unresolved would give each
        pass-through segment its own font and make the glyph probe non-deterministic.
        Resolving once keeps the whole conversion in a single font.

        Args:
            prompt (str): The prompt being converted, which art's size and wizard
                selectors take into account.

        Returns:
            str: A concrete font name.
        """
        if self._font in _ART_RANDOM_SELECTORS:
            # Pool fonts freely use blank glyphs, e.g. for punctuation, so a
            # random pick can fail to render a prompt on some draws only.
            # Prefer the fonts that render the whole prompt, and fall back to
            # the full pool when no single font covers it.
            candidates = _fonts_that_render(prompt, _ART_RANDOM_FONTS) or _ART_RANDOM_FONTS
            return self._get_random_generator(stream="font").choice(candidates)
        return cast("str", indirect_font(self._font, prompt))

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Convert the given prompt into ASCII art.

        Args:
            prompt (str): The prompt to be converted.
            input_type (PromptDataType): The type of input data.

        Returns:
            ConverterResult: The result containing the ASCII art representation of the prompt.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        font = self._resolve_font(prompt=prompt)

        unrenderable = [
            character for character in dict.fromkeys(prompt) if not _font_renders_character(character, font)
        ]
        if not unrenderable:
            return ConverterResult(output_text=text2art(prompt, font=font), output_type="text")

        characters = "".join(sorted(unrenderable))
        logger.warning(
            f"Font {font!r} has no glyph for {len(unrenderable)} character(s): {characters!r}. "
            "They are passed through as literal text rather than dropped, so the output is "
            "not entirely ASCII art. Transliterate them or pick another font to avoid this."
        )
        return ConverterResult(output_text=_render_with_passthrough(prompt, font), output_type="text")
