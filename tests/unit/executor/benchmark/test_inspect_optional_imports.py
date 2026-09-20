# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import sys


def test_core_and_adapter_definitions_import_without_inspect() -> None:
    code = """
import sys
sys.modules["inspect_ai"] = None
sys.modules["inspect_evals"] = None
from pyrit.models import Message
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.score import InspectEvalScorer, SubStringScorer
from pyrit.executor.benchmark import InspectBenchmark, InspectTaskBinding
assert "inspect_ai.log" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stderr
