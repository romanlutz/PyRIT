# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from pyrit.prompt_converter import BijectionConverter


class TestBijectionConverter:
    """Tests for BijectionConverter."""

    def test_mapping_generated(self):
        """Test that a mapping is generated on initialization."""
        converter = BijectionConverter()
        assert converter.mapping is not None
        assert len(converter.mapping) == 26

    def test_all_letters_mapped(self):
        """Test that all 26 letters are in the mapping."""
        converter = BijectionConverter()
        import string
        for letter in string.ascii_lowercase:
            assert letter in converter.mapping

    def test_mapping_is_bijection(self):
        """Test that the mapping is one-to-one (no two letters map to same letter)."""
        converter = BijectionConverter()
        values = list(converter.mapping.values())
        assert len(values) == len(set(values))

    def test_inverse_mapping_generated(self):
        """Test that inverse mapping is generated correctly."""
        converter = BijectionConverter()
        for k, v in converter.mapping.items():
            assert converter.inverse_mapping[v] == k

    def test_fixed_size_zero(self):
        """Test that fixed_size=0 shuffles all letters."""
        converter = BijectionConverter(fixed_size=0)
        # with fixed_size=0, at least some letters should be different
        changed = sum(1 for k, v in converter.mapping.items() if k != v)
        assert changed > 0

    def test_fixed_size_keeps_letters(self):
        """Test that fixed_size keeps first N letters unchanged."""
        converter = BijectionConverter(fixed_size=5)
        import string
        letters = list(string.ascii_lowercase)
        for letter in letters[:5]:
            assert converter.mapping[letter] == letter

    @pytest.mark.asyncio
    async def test_encode_prompt(self):
        """Test that encoding a prompt produces different text."""
        converter = BijectionConverter(fixed_size=0)
        result = await converter.convert_async(prompt="hello world")
        assert result.output_text != "hello world"

    @pytest.mark.asyncio
    async def test_decode_reverses_encoding(self):
        """Test that decoding an encoded prompt gives back original."""
        converter = BijectionConverter()
        original = "hello world"
        encoded = await converter.convert_async(prompt=original)
        decoded = converter.decode(encoded.output_text)
        assert decoded == original

    @pytest.mark.asyncio
    async def test_spaces_preserved(self):
        """Test that spaces are not encoded."""
        converter = BijectionConverter()
        result = await converter.convert_async(prompt="hello world")
        assert " " in result.output_text

    @pytest.mark.asyncio
    async def test_uppercase_preserved(self):
        """Test that uppercase letters stay uppercase after encoding."""
        converter = BijectionConverter()
        result = await converter.convert_async(prompt="Hello World")
        assert result.output_text[0].isupper()

    @pytest.mark.asyncio
    async def test_unsupported_input_type(self):
        """Test that unsupported input type raises ValueError."""
        converter = BijectionConverter()
        with pytest.raises(ValueError):
            await converter.convert_async(prompt="hello", input_type="image")