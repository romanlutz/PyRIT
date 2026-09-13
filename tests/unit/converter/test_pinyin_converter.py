# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.converter import ConverterResult, PinyinConverter


def is_pypinyin_installed():
    try:
        import pypinyin  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


# Conversion needs the optional 'pinyin' extra; the constructor/validation tests below do not.
requires_pypinyin = pytest.mark.skipif(not is_pypinyin_installed(), reason="pypinyin is not installed")


@requires_pypinyin
async def test_pinyin_full_mode_romanizes_every_hanzi():
    converter = PinyinConverter(mode="full")
    result = await converter.convert_async(prompt="中心", input_type="text")
    assert isinstance(result, ConverterResult)
    assert result.output_text == "zhongxin"
    assert result.output_type == "text"


@requires_pypinyin
async def test_pinyin_full_mode_with_separator_keeps_syllables_readable():
    converter = PinyinConverter(mode="full", separator=" ")
    result = await converter.convert_async(prompt="中心", input_type="text")
    # A separator is inserted between syllables but not left trailing.
    assert result.output_text == "zhong xin"


@requires_pypinyin
async def test_pinyin_initial_mode_uses_first_letters():
    converter = PinyinConverter(mode="initial")
    result = await converter.convert_async(prompt="中心", input_type="text")
    assert result.output_text == "zx"


@requires_pypinyin
async def test_pinyin_leaves_non_hanzi_untouched():
    converter = PinyinConverter(mode="full")
    result = await converter.convert_async(prompt="你好world! 123", input_type="text")
    # Latin letters, punctuation, spaces, and digits pass through unchanged.
    assert result.output_text == "nihaoworld! 123"


@requires_pypinyin
async def test_pinyin_separator_only_wraps_romanized_characters():
    converter = PinyinConverter(mode="full", separator="-")
    result = await converter.convert_async(prompt="中a好", input_type="text")
    # The non-Hanzi "a" gets no separator; the trailing separator after 好 is dropped.
    assert result.output_text == "zhong-ahao"


@requires_pypinyin
async def test_pinyin_prompt_without_hanzi_is_identity():
    converter = PinyinConverter(mode="full")
    prompt = "the quick brown fox"
    result = await converter.convert_async(prompt=prompt, input_type="text")
    assert result.output_text == prompt


@requires_pypinyin
async def test_pinyin_zero_proportion_is_identity():
    converter = PinyinConverter(mode="full", proportion=0.0)
    result = await converter.convert_async(prompt="你好世界", input_type="text")
    assert result.output_text == "你好世界"


@requires_pypinyin
async def test_pinyin_partial_proportion_converts_expected_count():
    # 4 Hanzi at proportion 0.5 -> exactly 2 romanized, 2 kept as Hanzi.
    converter = PinyinConverter(mode="full", proportion=0.5, seed=42)
    result = await converter.convert_async(prompt="你好世界", input_type="text")
    output = result.output_text
    remaining_hanzi = [c for c in output if c in "你好世界"]
    assert len(remaining_hanzi) == 2


@requires_pypinyin
async def test_pinyin_seed_makes_partial_selection_reproducible():
    prompt = "今天天气很好我们出去玩"
    first = (await PinyinConverter(mode="full", proportion=0.5, seed=7).convert_async(prompt=prompt)).output_text
    second = (await PinyinConverter(mode="full", proportion=0.5, seed=7).convert_async(prompt=prompt)).output_text
    assert first == second


@requires_pypinyin
async def test_pinyin_mixed_mode_is_reproducible_with_seed():
    prompt = "今天天气很好"
    first = (await PinyinConverter(mode="mixed", seed=13).convert_async(prompt=prompt)).output_text
    second = (await PinyinConverter(mode="mixed", seed=13).convert_async(prompt=prompt)).output_text
    assert first == second
    # Mixed mode still romanizes everything at proportion 1.0, so no Hanzi remains.
    assert not any("一" <= c <= "鿿" for c in first)


async def test_pinyin_rejects_unsupported_input_type():
    # The input-type guard runs before pypinyin is imported, so this needs no extra.
    converter = PinyinConverter()
    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt="你好", input_type="image_path")


def test_pinyin_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        PinyinConverter(mode="tone")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_pinyin_rejects_out_of_range_proportion(bad):
    with pytest.raises(ValueError, match="proportion must be between 0.0 and 1.0"):
        PinyinConverter(proportion=bad)


def test_pinyin_identifier_includes_parameters():
    converter = PinyinConverter(mode="initial", proportion=0.25, separator=" ", seed=99)
    identifier = converter.get_identifier()
    assert identifier.class_name == "PinyinConverter"
    assert identifier.params["mode"] == "initial"
    assert identifier.params["proportion"] == 0.25
    assert identifier.params["separator"] == " "
    assert identifier.params["seed"] == 99
