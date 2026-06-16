# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import string

import pytest

from pyrit.prompt_converter import DigitBijectionConverter, LetterBijectionConverter


def test_mapping_generated():
    converter = LetterBijectionConverter()
    assert converter.mapping is not None
    assert len(converter.mapping) == 26


def test_all_letters_mapped():
    converter = LetterBijectionConverter()
    for letter in string.ascii_lowercase:
        assert letter in converter.mapping


def test_mapping_is_bijection():
    converter = LetterBijectionConverter()
    values = list(converter.mapping.values())
    assert len(values) == len(set(values))


def test_inverse_mapping_generated():
    converter = LetterBijectionConverter()
    for k, v in converter.mapping.items():
        assert converter.inverse_mapping[v] == k


def test_fixed_size_zero():
    converter = LetterBijectionConverter(fixed_size=0)
    changed = sum(1 for k, v in converter.mapping.items() if k != v)
    assert changed > 0


def test_fixed_size_keeps_letters():
    converter = LetterBijectionConverter(fixed_size=5)
    letters = list(string.ascii_lowercase)
    for letter in letters[:5]:
        assert converter.mapping[letter] == letter


def test_seed_reproducibility():
    converter1 = LetterBijectionConverter(seed=42)
    converter2 = LetterBijectionConverter(seed=42)
    assert converter1.mapping == converter2.mapping


def test_explicit_mapping():
    custom_mapping = {chr(ord("a") + i): chr(ord("z") - i) for i in range(26)}
    converter = LetterBijectionConverter(mapping=custom_mapping)
    assert converter.mapping == custom_mapping


def test_digit_converter_mapping():
    converter = DigitBijectionConverter()
    # maps all 26 letters to digit strings
    assert len(converter.mapping) == 26
    for letter in string.ascii_lowercase:
        assert letter in converter.mapping
        # each value should be a digit string
        assert converter.mapping[letter].isdigit()


def test_digit_converter_encodes_letters():
    converter = DigitBijectionConverter(num_digits=2)
    # encoding "hello" should produce digit strings
    result_chars = "".join(converter.mapping.get(c, c) for c in "hello")
    assert result_chars != "hello"
    assert result_chars.replace(" ", "").isdigit()


def test_digit_converter_invalid_num_digits():
    with pytest.raises(ValueError, match="num_digits must be between 1 and 4"):
        DigitBijectionConverter(num_digits=5)


def test_digit_converter_is_bijection():
    converter = DigitBijectionConverter()
    values = list(converter.mapping.values())
    assert len(values) == len(set(values))


async def test_encode_prompt():
    converter = LetterBijectionConverter(fixed_size=0)
    result = await converter.convert_async(prompt="hello world")
    assert result.output_text != "hello world"


async def test_decode_reverses_encoding():
    converter = LetterBijectionConverter()
    original = "hello world"
    encoded = await converter.convert_async(prompt=original)
    decoded = converter.decode(encoded.output_text)
    assert decoded == original


async def test_spaces_preserved():
    converter = LetterBijectionConverter()
    result = await converter.convert_async(prompt="hello world")
    assert " " in result.output_text


async def test_uppercase_preserved():
    converter = LetterBijectionConverter()
    result = await converter.convert_async(prompt="Hello World")
    assert result.output_text[0].isupper()


async def test_unsupported_input_type():
    converter = LetterBijectionConverter()
    with pytest.raises(ValueError):
        await converter.convert_async(prompt="hello", input_type="image")
