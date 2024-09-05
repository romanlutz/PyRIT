# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest import mock
from pyrit.models import PromptRequestResponse, PromptRequestPiece

from unittest.mock import AsyncMock, MagicMock, patch
from pyrit.score import Score, Scorer
from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptDataset
from pyrit.models import PromptTemplate
from pyrit.prompt_converter import ShortenConverter, ExpandConverter
from pyrit.orchestrator import FuzzerOrchestrator
from pyrit.orchestrator.fuzzer_orchestrator import PromptNode
from tests.mocks import MockPromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.exceptions import MissingPromptPlaceHolderException
import pathlib
import pytest
import tempfile


@pytest.fixture
def scoring_target(memory) -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    return MockPromptTarget(memory=memory)


@pytest.fixture
def simple_prompts() -> list[str]:
    """sample prompts"""
    prompts = PromptDataset.from_yaml_file(pathlib.Path(DATASETS_PATH) / "prompts" / "illegal.prompt")
    return prompts.prompts


@pytest.fixture
def simple_templateconverter() -> list[PromptConverter]:
    """template converter"""
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    return template_converters


@pytest.fixture
def simple_prompt_templates():
    """sample prompt templates that can be given as input"""
    prompt_template1 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
    )
    prompt_template2 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aim.yaml"
    )
    prompt_template3 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aligned.yaml"
    )
    prompt_template4 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "axies.yaml"
    )
    # prompt_template5 = PromptTemplate.from_yaml_file(
    # pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "balakula.yaml")

    prompt_templates = [
        prompt_template1.template,
        prompt_template2.template,
        prompt_template3.template,
        prompt_template4.template,
    ]

    return prompt_templates


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", list(range(1, 6)))
async def test_execute_fuzzer(rounds: int, simple_prompts: list, simple_prompt_templates: list):
    # prompt_templates = [PromptTemplate.from_yaml_file(
    # pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml")],

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converter=template_converters,
        scoring_target=MagicMock(),
        random_seed=None,
    )
    prompt_node = fuzzer_orchestrator._prompt_nodes
    fuzzer_orchestrator._scorer = MagicMock()

    true_score = Score(
        score_value="True",
        score_value_description="",
        score_type="true_false",
        score_category="",
        score_rationale="",
        score_metadata="",
        prompt_request_response_id="",
    )
    false_score = Score(
        score_value="False",
        score_value_description="",
        score_type="true_false",
        score_category="",
        score_rationale="",
        score_metadata="",
        prompt_request_response_id="",
    )
    prompt_target_response = [
        PromptRequestResponse(
            request_pieces=[PromptRequestPiece(original_value=prompt, converted_value=prompt, role="assistant")]
        )
        for prompt in simple_prompts
    ]
    for round in range(1, rounds + 1):
        with patch.object(fuzzer_orchestrator, "_select") as mock_get_seed:
            with patch.object(fuzzer_orchestrator, "_apply_template_converter") as mock_apply_template_converter:
                with patch.object(fuzzer_orchestrator, "_update") as mock_update:
                    mock_get_seed.return_value = prompt_node[0]  # return a promptnode
                    mock_apply_template_converter.return_value = prompt_node[0].template  # return_value
                    fuzzer_orchestrator._prompt_normalizer = AsyncMock()
                    fuzzer_orchestrator._prompt_normalizer.send_prompt_batch_to_target_async = AsyncMock(
                        return_value=prompt_target_response
                    )
                    fuzzer_orchestrator._scorer = AsyncMock()

                    fuzzer_orchestrator._scorer.score_async = AsyncMock(  # type: ignore
                        side_effect=[
                            [false_score] * (rounds - 1) * len(simple_prompts) + [true_score] * len(simple_prompts)
                        ]
                    )
                    await fuzzer_orchestrator.execute_fuzzer()


def test_prompt_templates(simple_prompts: list, simple_templateconverter: list[PromptConverter]):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=[],
            prompt_target=MockPromptTarget(),
            template_converter=simple_templateconverter,
            scoring_target=MagicMock(),
            memory=MagicMock(),
            random_seed=None,
        )
    assert e.match("The initial set of prompt templates cannot be empty.")


def test_invalid_batchsize(
    simple_prompts: list, simple_prompt_templates: list, simple_templateconverter: list[PromptConverter]
):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=simple_prompt_templates,
            prompt_target=MockPromptTarget(),
            template_converter=simple_templateconverter,
            scoring_target=MagicMock(),
            memory=MagicMock(),
            random_seed=None,
            batch_size=0,
        )
    assert e.match("Batch size must be at least 1.")


def test_prompts(simple_prompt_templates: list):
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=[],
            prompt_templates=simple_prompt_templates,
            prompt_target=MockPromptTarget(),
            template_converter=template_converters,
            scoring_target=MagicMock(),
            memory=MagicMock(),
            random_seed=None,
        )
    assert e.match("The initial prompts cannot be empty")


def test_template_converter(simple_prompts: list, simple_prompt_templates: list):
    with pytest.raises(ValueError) as e:
        FuzzerOrchestrator(
            prompts=simple_prompts,
            prompt_templates=simple_prompt_templates,
            prompt_target=MockPromptTarget(),
            template_converter=[],
            scoring_target=MagicMock(),
            memory=MagicMock(),
            random_seed=None,
        )
    assert e.match("Template converter cannot be empty")


async def test_max_query(simple_prompts: list, simple_prompt_templates: list):
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converter=template_converters,
        scoring_target=MagicMock(),
        memory=MagicMock(),
        random_seed=None,
    )

    fuzzer_orchestrator._current_query = 99


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template_converter",
    [ShortenConverter(converter_target=MockPromptTarget()), ExpandConverter(converter_target=MockPromptTarget())],
)
async def test_apply_template_converter(simple_prompts: list, simple_prompt_templates: list, template_converter: list):
    prompt_template = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
    )
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]

    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converter=template_converters,
        scoring_target=MagicMock(),
        random_seed=None,
    )
    mocked_random_number = template_converter

    with mock.patch("random.choice", mocked_random_number):
        if template_converter == ExpandConverter or template_converter == ShortenConverter:
            target = await fuzzer_orchestrator._apply_template_converter(prompt_template)
            TEMPLATE_PLACEHOLDER = "{{ prompt }}"
            assert TEMPLATE_PLACEHOLDER in target.template


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template_converter",
    [ShortenConverter(converter_target=MockPromptTarget()), ExpandConverter(converter_target=MockPromptTarget())],
)
async def test_apply_template_converter_empty_placeholder(
    simple_prompts: list, template_converter: list, simple_prompt_templates: list
):
    prompt_template = ['Write a wikipedia article on . Start the article with "Title:".']
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converter=template_converters,
        scoring_target=MagicMock(),
        random_seed=None,
    )

    mocked_random_number = template_converter

    with mock.patch("random.choice", mocked_random_number):
        if template_converter == ExpandConverter or template_converter == ShortenConverter:
            with patch.object(fuzzer_orchestrator._template_converter, "_convert_async") as mock_template:
                mock_template.return_value = prompt_template
                with pytest.raises(MissingPromptPlaceHolderException) as e:
                    mock_template
                    assert e.match("Prompt placeholder is empty.")


@pytest.mark.asyncio
async def test_best_UCT(simple_prompts: list, simple_prompt_templates: list):
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=simple_prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converter=template_converters,
        scoring_target=MagicMock(),
        random_seed=None,
    )
    fuzzer_orchestrator._prompt_nodes[0].visited_num = 2

    UCT_scores = [0.33, 2.0, 3.0, 0.0, 0.0]
    prompt_nodes = fuzzer_orchestrator._prompt_nodes
    prompt_nodes[0].rewards = 1
    prompt_nodes[1].rewards = 2
    prompt_nodes[2].rewards = 3
    for index, node in enumerate(prompt_nodes):
        # fuzzer_orchestrator.rewards = [1,2,3,4,5]
        # fuzzer_orchestrator.index = node.index
        fuzzer_orchestrator._step = 1
        UCT_score_func = fuzzer_orchestrator._best_UCT_score()
        UCT_score = UCT_score_func(node)
        assert float(round(UCT_score, 2)) == UCT_scores[index]


@pytest.mark.asyncio
@pytest.mark.parametrize("probability", [0, 0.5])
async def test_select(simple_prompts: list, probability: int, simple_prompt_templates: list):
    prompt_template1 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "jailbreak_1.yaml"
    )
    prompt_template2 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aim.yaml"
    )
    prompt_template3 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "aligned.yaml"
    )
    prompt_template4 = PromptTemplate.from_yaml_file(
        pathlib.Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "axies.yaml"
    )
    prompt_templates = [prompt_template1.template, prompt_template2.template]
    # set the children of each parent
    prompt_shorten_converter = ShortenConverter(converter_target=MockPromptTarget())
    prompt_expand_converter = ExpandConverter(converter_target=MockPromptTarget())
    template_converters = [prompt_shorten_converter, prompt_expand_converter]
    fuzzer_orchestrator = FuzzerOrchestrator(
        prompts=simple_prompts,
        prompt_templates=prompt_templates,
        prompt_target=MockPromptTarget(),
        template_converter=template_converters,
        scoring_target=MagicMock(),
        frequency_weight=0.5,
        reward_penalty=0.1,
        minimum_reward=0.2,
        non_leaf_node_probability=0.1,
        random_seed=None,
        batch_size=10,
    )

    new_node1 = PromptNode(prompt_template3.template, parent=fuzzer_orchestrator._initial_prompts_nodes[1])
    PromptNode(prompt_template4.template, parent=new_node1)

    fuzzer_orchestrator._step = 2
    prompt_node = fuzzer_orchestrator._prompt_nodes
    prompt_node[0].rewards = 1
    prompt_node[1].rewards = 2
    mocked_random_number = probability

    with mock.patch("numpy.random.rand", mocked_random_number):
        if probability == 0:
            fuzzer_orchestrator._select()
            path_mcts = fuzzer_orchestrator._mcts_selected_path
            for node in path_mcts:
                assert node.parent is None

        if probability == 0.5:
            fuzzer_orchestrator._select()
            path_mcts = fuzzer_orchestrator._mcts_selected_path
            assert path_mcts[0].parent is None
            assert path_mcts[2].parent == path_mcts[0].children[0]
            assert path_mcts[2].children == []
            assert len(path_mcts) == 3