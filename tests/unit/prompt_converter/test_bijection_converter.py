# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import string
import pytest
from pyrit.prompt_converter import BijectionConverter


def test_mapping_generated():
    converter = BijectionConverter()
    assert converter.mapping is not None
    assert len(converter.mapping) == 26


def test_all_letters_mapped():
    converter = BijectionConverter()
    for letter in string.ascii_lowercase:
        assert letter in converter.mapping


def test_mapping_is_bijection():
    converter = BijectionConverter()
    values = list(converter.mapping.values())
    assert len(values) == len(set(values))


def test_inverse_mapping_generated():
    converter = BijectionConverter()
    for k, v in converter.mapping.items():
        assert converter.inverse_mapping[v] == k


def test_fixed_size_zero():
    converter = BijectionConverter(fixed_size=0)
    changed = sum(1 for k, v in converter.mapping.items() if k != v)
    assert changed > 0


def test_fixed_size_keeps_letters():
    converter = BijectionConverter(fixed_size=5)
    letters = list(string.ascii_lowercase)
    for letter in letters[:5]:
        assert converter.mapping[letter] == letter


async def test_encode_prompt():
    converter = BijectionConverter(fixed_size=0)
    result = await converter.convert_async(prompt="hello world")
    assert result.output_text != "hello world"


async def test_decode_reverses_encoding():
    converter = BijectionConverter()
    original = "hello world"
    encoded = await converter.convert_async(prompt=original)
    decoded = converter.decode(encoded.output_text)
    assert decoded == original


async def test_spaces_preserved():
    converter = BijectionConverter()
    result = await converter.convert_async(prompt="hello world")
    assert " " in result.output_text


async def test_uppercase_preserved():
    converter = BijectionConverter()
    result = await converter.convert_async(prompt="Hello World")
    assert result.output_text[0].isupper()


async def test_unsupported_input_type():
    converter = BijectionConverter()
    with pytest.raises(ValueError):
        await converter.convert_async(prompt="hello", input_type="image")