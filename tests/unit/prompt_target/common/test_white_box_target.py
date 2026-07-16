# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Structural tests for the ``WhiteBoxTarget`` Protocol and ``WhiteBoxInputs``.

These are torch-free: they use lightweight stand-ins for tensors so the module can
be verified on installs without the heavy ML stack.
"""

from __future__ import annotations

from pyrit.prompt_target.common.white_box_target import WhiteBoxInputs, WhiteBoxTarget


class _StubImage:
    pass


class _CompleteTarget:
    """A structurally-complete WhiteBoxTarget stand-in (no torch required)."""

    vlm_id = "stub/vlm"
    device = "cpu"

    def preprocess(self, *, behavior, image):
        return WhiteBoxInputs(pixel_values=object(), model_inputs={"input_ids": [1, 2, 3]})

    def compute_loss(self, *, inputs, target_text):
        return 0.0

    def to_pil(self, *, inputs):
        return _StubImage()

    def release_white_box_resources(self):
        return None


class _MissingComputeLoss:
    vlm_id = "stub/vlm"
    device = "cpu"

    def preprocess(self, *, behavior, image):
        return WhiteBoxInputs(pixel_values=object())

    def to_pil(self, *, inputs):
        return _StubImage()

    def release_white_box_resources(self):
        return None


def test_complete_target_satisfies_protocol() -> None:
    assert isinstance(_CompleteTarget(), WhiteBoxTarget)


def test_incomplete_target_fails_isinstance() -> None:
    assert not isinstance(_MissingComputeLoss(), WhiteBoxTarget)


def test_with_pixel_values_replaces_and_copies_model_inputs() -> None:
    original = WhiteBoxInputs(pixel_values="base", model_inputs={"input_ids": [1]})
    swapped = original.with_pixel_values("perturbed")

    assert swapped.pixel_values == "perturbed"
    assert original.pixel_values == "base"
    assert swapped.model_inputs == {"input_ids": [1]}
    # model_inputs must be a distinct dict so loop mutation cannot leak back.
    assert swapped.model_inputs is not original.model_inputs


def test_model_inputs_defaults_to_empty_dict() -> None:
    inputs = WhiteBoxInputs(pixel_values="x")
    assert inputs.model_inputs == {}
