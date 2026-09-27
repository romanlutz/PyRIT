# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hold original SSE frames while checking for a host credential echoed by a model."""

from collections.abc import Iterator


class CredentialEchoError(Exception):
    """A provider attempted to return the host-owned credential."""


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _values(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _values(item)


class SecretFrameGuard:
    """Delay a possible credential prefix across SSE events without rewriting bytes."""

    _OUTPUT_FIELDS = frozenset(
        {
            "delta",
            "text",
            "arguments",
            "input",
            "output",
            "refusal",
            "encrypted_content",
            "partial_json",
            "thinking",
            "comment",
            "signature",
            "message",
            "data",
            "id",
        }
    )

    def __init__(self, *, token: str) -> None:
        """Bind a host-only credential before checking any provider output."""
        self._token = token
        self._token_bytes = token.encode()
        self._pending: list[bytes] = []
        self._prefix_table = self._build_prefix_table(token=token)
        self._matched_prefix = 0

    def accept_payload(self, *, frame: bytes, data: object) -> list[bytes]:
        """
        Check original and decoded data before releasing frames to the caller.

        Args:
            frame (bytes): One complete original SSE frame.
            data (object): Parsed SSE payload or comment to scan for an echoed credential.

        Returns:
            list[bytes]: Original frames whose credential prefix can no longer complete.

        Raises:
            CredentialEchoError: If the host credential appeared in this event.
        """
        if self._token_bytes in frame or any(self._token in value for value in _strings(data)):
            raise CredentialEchoError
        for text in self._output_strings(value=data):
            self._check_text(text=text)
        self._pending.append(frame)
        if self._matched_prefix:
            return []
        return self.finish()

    def finish(self) -> list[bytes]:
        """
        Release held frames after genuine stream termination.

        Returns:
            list[bytes]: The withheld original frames.
        """
        frames = self._pending
        self._pending = []
        self._matched_prefix = 0
        return frames

    def check_body(self, *, raw: bytes, parsed: object) -> None:
        """
        Check a complete non-streaming response for the configured credential.

        Args:
            raw (bytes): Original JSON response or error body.
            parsed (object): Decoded JSON to inspect escaped representations.

        Raises:
            CredentialEchoError: If the host credential appeared in raw or decoded JSON.
        """
        if (
            self._token_bytes in raw
            or any(self._token in value for value in _strings(parsed))
            or self._token in "".join(_values(parsed))
        ):
            raise CredentialEchoError

    def _output_strings(self, *, value: object) -> Iterator[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in self._OUTPUT_FIELDS and isinstance(child, str):
                    yield child
                else:
                    yield from self._output_strings(value=child)
        elif isinstance(value, list):
            for child in value:
                yield from self._output_strings(value=child)

    def _check_text(self, *, text: str) -> None:
        for character in text:
            while self._matched_prefix and character != self._token[self._matched_prefix]:
                self._matched_prefix = self._prefix_table[self._matched_prefix - 1]
            if character == self._token[self._matched_prefix]:
                self._matched_prefix += 1
            if self._matched_prefix == len(self._token):
                raise CredentialEchoError

    @staticmethod
    def _build_prefix_table(*, token: str) -> list[int]:
        prefixes = [0] * len(token)
        matched = 0
        for index in range(1, len(token)):
            while matched and token[index] != token[matched]:
                matched = prefixes[matched - 1]
            if token[index] == token[matched]:
                matched += 1
                prefixes[index] = matched
        return prefixes
