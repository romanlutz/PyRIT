# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from unit.mocks import get_mock_scorer_identifier, get_mock_target_identifier

from pyrit.datasets.seed_datasets.remote.figstep_dataset import _FigStepDataset
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackParameters,
    AttackScoringConfig,
    FigStepAttack,
    FigStepAttackParameters,
    SingleTurnAttackContext,
)
from pyrit.models import Message
from pyrit.prompt_converter import AddImageTextConverter
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer, TrueFalseScorer


@pytest.fixture
def mock_objective_target():
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = get_mock_target_identifier("MockTarget")
    return target


@pytest.fixture
def mock_adversarial_chat_target():
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = get_mock_target_identifier("MockAdversarialChat")
    return target


@pytest.fixture
def mock_scorer():
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.score_text_async = AsyncMock()
    scorer.get_identifier.return_value = get_mock_scorer_identifier()
    return scorer


@pytest.fixture
def figstep_attack(mock_objective_target):
    return FigStepAttack(objective_target=mock_objective_target)


@pytest.fixture
def figstep_attack_with_rephrase(mock_objective_target, mock_adversarial_chat_target):
    return FigStepAttack(
        objective_target=mock_objective_target,
        attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat_target),
    )


@pytest.fixture
def basic_context():
    return SingleTurnAttackContext(
        params=AttackParameters(objective="How do I bake a cake?"),
        conversation_id=str(uuid.uuid4()),
    )


@pytest.mark.usefixtures("patch_central_database")
class TestFigStepAttackInitialization:
    def test_init_defaults(self, mock_objective_target):
        attack = FigStepAttack(objective_target=mock_objective_target)
        assert attack._objective_target is mock_objective_target
        assert attack._rephrase_converter is None
        assert attack._carrier_prompt == FigStepAttack.FIGSTEP_CARRIER_PROMPT
        assert attack._num_items == 3
        assert attack._wrap_width == 15
        assert attack._canvas_size == (760, 760)
        assert attack._font_size == 80
        assert attack._text_color == (0, 0, 0)
        assert attack._background_color == (255, 255, 255)
        assert attack._text_origin == (20, 10)
        assert attack._font_name is None

    def test_init_with_adversarial_config_creates_rephrase_converter(
        self, mock_objective_target, mock_adversarial_chat_target
    ):
        attack = FigStepAttack(
            objective_target=mock_objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat_target),
        )
        assert attack._rephrase_converter is not None

    def test_init_with_invalid_scorer_type(self, mock_objective_target):
        scorer = MagicMock(spec=Scorer)
        with pytest.raises(ValueError, match="Objective scorer must be a TrueFalseScorer"):
            FigStepAttack(
                objective_target=mock_objective_target,
                attack_scoring_config=AttackScoringConfig(objective_scorer=scorer),
            )

    @pytest.mark.parametrize("num_items", [0, -1])
    def test_init_invalid_num_items(self, mock_objective_target, num_items):
        with pytest.raises(ValueError, match="num_items must be >= 1"):
            FigStepAttack(objective_target=mock_objective_target, num_items=num_items)

    def test_init_invalid_wrap_width(self, mock_objective_target):
        with pytest.raises(ValueError, match="wrap_width must be >= 1"):
            FigStepAttack(objective_target=mock_objective_target, wrap_width=0)

    def test_init_invalid_font_size(self, mock_objective_target):
        with pytest.raises(ValueError, match="font_size must be >= 1"):
            FigStepAttack(objective_target=mock_objective_target, font_size=0)

    def test_init_invalid_line_spacing(self, mock_objective_target):
        with pytest.raises(ValueError, match="line_spacing must be >= 0"):
            FigStepAttack(objective_target=mock_objective_target, line_spacing=-1)

    @pytest.mark.parametrize("canvas_size", [(0, 100), (100, 0), (-1, 100)])
    def test_init_invalid_canvas_size(self, mock_objective_target, canvas_size):
        with pytest.raises(ValueError, match="canvas_size must have positive dimensions"):
            FigStepAttack(objective_target=mock_objective_target, canvas_size=canvas_size)

    def test_init_invalid_rephrase_template_raises(self, mock_objective_target, mock_adversarial_chat_target):
        with pytest.raises(ValueError, match="rephrase_instructions must contain"):
            FigStepAttack(
                objective_target=mock_objective_target,
                attack_adversarial_config=AttackAdversarialConfig(target=mock_adversarial_chat_target),
                rephrase_instructions="no placeholder here",
            )

    def test_init_invalid_rephrase_template_ignored_without_adversarial_config(self, mock_objective_target):
        """Without an adversarial chat target, rephrase_instructions is unused and should not be validated."""
        # Must not raise even though the template has no placeholder
        attack = FigStepAttack(objective_target=mock_objective_target, rephrase_instructions="no placeholder")
        assert attack._rephrase_converter is None


@pytest.mark.usefixtures("patch_central_database")
class TestFigStepCarrierPromptConstant:
    def test_carrier_prompt_default_matches_dataset_constant(self):
        """Regression guard: the carrier prompt the attack ships must equal the SafeBench dataset's."""
        assert FigStepAttack.FIGSTEP_CARRIER_PROMPT == _FigStepDataset.FIGSTEP_PROMPT


@pytest.mark.usefixtures("patch_central_database")
class TestBuildFigStepText:
    def test_wraps_at_wrap_width(self, figstep_attack):
        text = figstep_attack._build_figstep_text(stem="abc def ghi jkl mno")
        # textwrap.fill greedily fills each line; "abc def ghi jkl" is exactly 15 chars.
        assert text == "abc def ghi jkl\nmno\n1. \n2. \n3. "

    def test_strips_trailing_newline_from_stem(self, figstep_attack):
        text = figstep_attack._build_figstep_text(stem="Steps to bake.\n")
        assert text == "Steps to bake.\n1. \n2. \n3. "
        assert "\n\n1." not in text

    def test_num_items_configurable(self, mock_objective_target):
        attack = FigStepAttack(objective_target=mock_objective_target, num_items=5)
        text = attack._build_figstep_text(stem="Short.")
        assert text == "Short.\n1. \n2. \n3. \n4. \n5. "

    def test_num_items_one(self, mock_objective_target):
        attack = FigStepAttack(objective_target=mock_objective_target, num_items=1)
        text = attack._build_figstep_text(stem="Short.")
        assert text == "Short.\n1. "


@pytest.mark.usefixtures("patch_central_database")
class TestGetStemAsync:
    async def test_no_adversarial_config_returns_objective_unchanged(self, figstep_attack):
        stem = await figstep_attack._get_stem_async(objective="  How do I make X?  ")
        assert stem == "How do I make X?"

    async def test_with_adversarial_config_rephrases(self, figstep_attack_with_rephrase):
        mock_result = MagicMock()
        mock_result.output_text = "  Steps to bake a cake.  "

        with patch.object(
            figstep_attack_with_rephrase._rephrase_converter,
            "convert_async",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_convert:
            stem = await figstep_attack_with_rephrase._get_stem_async(objective="How do I bake a cake?")

        mock_convert.assert_awaited_once_with(prompt="How do I bake a cake?", input_type="text")
        assert stem == "Steps to bake a cake."


@pytest.mark.usefixtures("patch_central_database")
class TestRenderFigStepImage:
    async def test_blank_canvas_is_cached_across_invocations(self, figstep_attack):
        """The blank canvas is created once and reused on subsequent renders."""
        first_path = await figstep_attack._ensure_blank_canvas_async()
        second_path = await figstep_attack._ensure_blank_canvas_async()
        assert first_path == second_path
        assert os.path.exists(first_path)
        with Image.open(first_path) as img:
            assert img.size == FigStepAttack._DEFAULT_CANVAS_SIZE
            assert img.mode == "RGB"
            # Centre pixel of an unmarked canvas should be the configured background.
            assert img.getpixel((img.width // 2, img.height // 2)) == FigStepAttack._DEFAULT_BG_COLOR

    async def test_renders_via_add_image_text_converter(self, figstep_attack):
        """The render path delegates to AddImageTextConverter with paper-faithful parameters."""
        with patch(
            "pyrit.executor.attack.single_turn.figstep.AddImageTextConverter",
            wraps=AddImageTextConverter,
        ) as converter_cls:
            result_path = await figstep_attack._render_figstep_image_async(text="Steps to bake.\n1. \n2. \n3. ")

        assert os.path.exists(result_path)
        with Image.open(result_path) as img:
            assert img.size == FigStepAttack._DEFAULT_CANVAS_SIZE
            assert img.mode == "RGB"

        converter_cls.assert_called_once()
        kwargs = converter_cls.call_args.kwargs
        assert kwargs["font_name"] is None
        assert kwargs["font_size"] == FigStepAttack._DEFAULT_FONT_SIZE
        assert kwargs["color"] == FigStepAttack._DEFAULT_TEXT_COLOR
        assert kwargs["line_spacing"] == FigStepAttack._DEFAULT_LINE_SPACING
        x_origin, y_origin = FigStepAttack._DEFAULT_TEXT_ORIGIN
        canvas_w, canvas_h = FigStepAttack._DEFAULT_CANVAS_SIZE
        assert kwargs["bounding_box"] == (x_origin, y_origin, canvas_w, canvas_h)

    async def test_rendered_image_preserves_embedded_newlines(self, figstep_attack):
        """Regression guard: single-line vs multi-line text must render to different pixels.

        Before the AddImageTextConverter fix (``textwrap.fill`` collapsed embedded ``\\n`` to
        spaces) both inputs rendered identically. With the fix in place, each numbered list
        item lands on its own line and the pixel buffers diverge.
        """
        single_line_path = await figstep_attack._render_figstep_image_async(text="Steps to bake. 1. 2. 3.")
        multi_line_path = await figstep_attack._render_figstep_image_async(text="Steps to bake.\n1. \n2. \n3. ")

        with Image.open(single_line_path) as single_img, Image.open(multi_line_path) as multi_img:
            assert single_img.tobytes() != multi_img.tobytes()

    async def test_custom_canvas_size_and_colors_applied(self, mock_objective_target):
        attack = FigStepAttack(
            objective_target=mock_objective_target,
            canvas_size=(200, 120),
            background_color=(10, 20, 30),
            text_color=(255, 128, 64),
        )
        rendered_path = await attack._render_figstep_image_async(text="hi")
        with Image.open(rendered_path) as img:
            assert img.size == (200, 120)
            assert img.mode == "RGB"
            # Bottom-right pixel is past the rendered text, so it should still be the background colour.
            assert img.getpixel((199, 119)) == (10, 20, 30)


@pytest.mark.usefixtures("patch_central_database")
class TestBuildMultimodalMessage:
    def test_message_has_image_and_text_pieces_sharing_sequence(self, figstep_attack):
        msg = figstep_attack._build_multimodal_message(image_path="/tmp/foo.png", carrier_text="please describe")
        assert isinstance(msg, Message)
        assert len(msg.message_pieces) == 2

        image_piece, text_piece = msg.message_pieces
        assert image_piece.original_value == "/tmp/foo.png"
        assert image_piece.original_value_data_type == "image_path"
        assert image_piece._role == "user"
        assert text_piece.original_value == "please describe"
        assert text_piece.original_value_data_type == "text"
        assert text_piece._role == "user"
        # PromptNormalizer requires all pieces share the same sequence
        assert image_piece.sequence == text_piece.sequence == 0
        # Message.validate requires all pieces share the same conversation_id
        assert image_piece.conversation_id == text_piece.conversation_id


@pytest.mark.usefixtures("patch_central_database")
class TestSetupAsync:
    async def test_setup_no_rephrase_uses_raw_objective(self, figstep_attack, basic_context):
        rendered_path = "/tmp/rendered.png"

        with (
            patch.object(
                figstep_attack,
                "_render_figstep_image_async",
                new_callable=AsyncMock,
                return_value=rendered_path,
            ) as mock_render,
            patch(
                "pyrit.executor.attack.single_turn.prompt_sending.PromptSendingAttack._setup_async",
                new_callable=AsyncMock,
            ) as mock_super_setup,
        ):
            await figstep_attack._setup_async(context=basic_context)

        # Image render received the raw objective (not rephrased) prefixed text
        rendered_text = mock_render.await_args.kwargs["text"]
        assert rendered_text.startswith("How do I bake")
        assert rendered_text.endswith("\n1. \n2. \n3. ")

        # Context.next_message is the multimodal message
        assert basic_context.next_message is not None
        pieces = basic_context.next_message.message_pieces
        assert len(pieces) == 2
        assert pieces[0].original_value == rendered_path
        assert pieces[0].original_value_data_type == "image_path"
        assert pieces[1].original_value == FigStepAttack.FIGSTEP_CARRIER_PROMPT
        assert pieces[1].original_value_data_type == "text"

        # Parent setup must still run for conversation/labels wiring
        mock_super_setup.assert_awaited_once_with(context=basic_context)

    async def test_setup_with_rephrase_uses_stem(self, figstep_attack_with_rephrase, basic_context):
        rephrased = MagicMock()
        rephrased.output_text = "Steps to bake a cake."

        with (
            patch.object(
                figstep_attack_with_rephrase._rephrase_converter,
                "convert_async",
                new_callable=AsyncMock,
                return_value=rephrased,
            ),
            patch.object(
                figstep_attack_with_rephrase,
                "_render_figstep_image_async",
                new_callable=AsyncMock,
                return_value="/tmp/rendered.png",
            ) as mock_render,
            patch(
                "pyrit.executor.attack.single_turn.prompt_sending.PromptSendingAttack._setup_async",
                new_callable=AsyncMock,
            ),
        ):
            await figstep_attack_with_rephrase._setup_async(context=basic_context)

        rendered_text = mock_render.await_args.kwargs["text"]
        assert rendered_text.startswith("Steps to bake")
        assert rendered_text.endswith("\n1. \n2. \n3. ")

    async def test_setup_uses_custom_carrier_prompt(self, mock_objective_target, basic_context):
        attack = FigStepAttack(objective_target=mock_objective_target, carrier_prompt="custom carrier")

        with (
            patch.object(
                attack,
                "_render_figstep_image_async",
                new_callable=AsyncMock,
                return_value="/tmp/rendered.png",
            ),
            patch(
                "pyrit.executor.attack.single_turn.prompt_sending.PromptSendingAttack._setup_async",
                new_callable=AsyncMock,
            ),
        ):
            await attack._setup_async(context=basic_context)

        assert basic_context.next_message.message_pieces[1].original_value == "custom carrier"


@pytest.mark.usefixtures("patch_central_database")
class TestFigStepAttackParamsType:
    def test_params_type_excludes_next_message(self, figstep_attack):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(figstep_attack.params_type)}
        assert "next_message" not in fields

    def test_params_type_keeps_prepended_conversation(self, figstep_attack):
        """FigStep does not touch prepended_conversation; callers should still be able to set it."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(figstep_attack.params_type)}
        assert "prepended_conversation" in fields

    def test_params_type_is_figstep_specific(self, figstep_attack):
        assert figstep_attack.params_type is FigStepAttackParameters
