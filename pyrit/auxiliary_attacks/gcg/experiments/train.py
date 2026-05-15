# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Backwards-compatibility shim for the legacy ``generate_suffix`` entry point.

New code should use :class:`pyrit.auxiliary_attacks.gcg.GCGGenerator` directly:

    generator = GCGGenerator(
        models=[GCGModelConfig(name="meta-llama/Llama-2-7b-chat-hf")],
    )
    result = await generator.execute_async(goals=[...], targets=[...])

This module remains so existing scripts that call
``GreedyCoordinateGradientAdversarialSuffixGenerator().generate_suffix(...)``
keep working for one release. Every call now emits a ``DeprecationWarning``.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any, Optional

from pyrit.auxiliary_attacks.gcg.config import (
    GCGAlgorithmConfig,
    GCGDataConfig,
    GCGModelConfig,
    GCGOutputConfig,
    GCGStrategyConfig,
)
from pyrit.auxiliary_attacks.gcg.data import load_goals_and_targets
from pyrit.auxiliary_attacks.gcg.generator import GCGGenerator

logger = logging.getLogger(__name__)


class GreedyCoordinateGradientAdversarialSuffixGenerator:
    """Deprecated. Use :class:`pyrit.auxiliary_attacks.gcg.GCGGenerator` instead."""

    _DEFAULT_CONTROL_INIT: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

    def generate_suffix(
        self,
        *,
        token: str = "",
        tokenizer_paths: Optional[list[str]] = None,
        model_name: str = "",
        model_paths: Optional[list[str]] = None,
        result_prefix: str = "",
        train_data: str = "",
        control_init: str = _DEFAULT_CONTROL_INIT,
        n_train_data: int = 50,
        n_steps: int = 500,
        test_steps: int = 50,
        batch_size: int = 512,
        transfer: bool = False,
        target_weight: float = 1.0,
        control_weight: float = 0.0,
        progressive_goals: bool = False,
        progressive_models: bool = False,
        anneal: bool = False,
        incr_control: bool = False,
        stop_on_success: bool = False,
        verbose: bool = True,
        allow_non_ascii: bool = False,
        num_train_models: int = 1,
        devices: Optional[list[str]] = None,
        model_kwargs: Optional[list[dict[str, Any]]] = None,
        tokenizer_kwargs: Optional[list[dict[str, Any]]] = None,
        n_test_data: int = 0,
        test_data: str = "",
        learning_rate: float = 0.01,
        topk: int = 256,
        temp: int = 1,
        filter_cand: bool = True,
        gbda_deterministic: bool = True,
        logfile: str = "",
        random_seed: int = 42,
    ) -> None:
        """
        Deprecated. Use :meth:`GCGGenerator.execute_async`.

        ``model_name``, ``num_train_models``, and ``gbda_deterministic`` are accepted
        for backwards compatibility but no longer affect behaviour: ``model_name``
        was a free-form identifier used only in log lines; training-vs-test split
        is now expressed via ``GCGGenerator(test_models=[...])``; and
        ``gbda_deterministic`` was already dead.
        """
        warnings.warn(
            "GreedyCoordinateGradientAdversarialSuffixGenerator.generate_suffix() is deprecated; "
            "use pyrit.auxiliary_attacks.gcg.GCGGenerator.execute_async() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        del gbda_deterministic, model_name  # accepted for backcompat, ignored
        generator, data_config = self._build_generator_and_data(
            token=token,
            tokenizer_paths=tokenizer_paths,
            model_paths=model_paths,
            result_prefix=result_prefix,
            train_data=train_data,
            control_init=control_init,
            n_train_data=n_train_data,
            n_steps=n_steps,
            test_steps=test_steps,
            batch_size=batch_size,
            transfer=transfer,
            target_weight=target_weight,
            control_weight=control_weight,
            progressive_goals=progressive_goals,
            progressive_models=progressive_models,
            anneal=anneal,
            incr_control=incr_control,
            stop_on_success=stop_on_success,
            verbose=verbose,
            allow_non_ascii=allow_non_ascii,
            num_train_models=num_train_models,
            devices=devices,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            n_test_data=n_test_data,
            test_data=test_data,
            learning_rate=learning_rate,
            topk=topk,
            temp=temp,
            filter_cand=filter_cand,
            logfile=logfile,
            random_seed=random_seed,
        )
        train_goals, train_targets, test_goals, test_targets = load_goals_and_targets(
            data=data_config, random_seed=random_seed
        )
        asyncio.run(
            generator.execute_async(
                goals=train_goals,
                targets=train_targets,
                test_goals=test_goals,
                test_targets=test_targets,
            )
        )

    @staticmethod
    def _build_generator_and_data(
        *,
        token: str,
        tokenizer_paths: Optional[list[str]],
        model_paths: Optional[list[str]],
        result_prefix: str,
        train_data: str,
        control_init: str,
        n_train_data: int,
        n_steps: int,
        test_steps: int,
        batch_size: int,
        transfer: bool,
        target_weight: float,
        control_weight: float,
        progressive_goals: bool,
        progressive_models: bool,
        anneal: bool,
        incr_control: bool,
        stop_on_success: bool,
        verbose: bool,
        allow_non_ascii: bool,
        num_train_models: int,
        devices: Optional[list[str]],
        model_kwargs: Optional[list[dict[str, Any]]],
        tokenizer_kwargs: Optional[list[dict[str, Any]]],
        n_test_data: int,
        test_data: str,
        learning_rate: float,
        topk: int,
        temp: int,
        filter_cand: bool,
        logfile: str,
        random_seed: int,
    ) -> tuple[GCGGenerator, GCGDataConfig]:
        """Translate the legacy generate_suffix kwargs into a GCGGenerator + data config."""
        if not model_paths:
            raise ValueError("generate_suffix(): model_paths must be a non-empty list of model identifiers.")
        if tokenizer_paths is None:
            tokenizer_paths = list(model_paths)
        if devices is None:
            devices = ["cuda:0"] * len(model_paths)
        if model_kwargs is None:
            model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False}] * len(model_paths)
        if tokenizer_kwargs is None:
            tokenizer_kwargs = [{"use_fast": False}] * len(model_paths)

        if not (len(tokenizer_paths) == len(model_paths) == len(devices) == len(model_kwargs) == len(tokenizer_kwargs)):
            raise ValueError(
                "generate_suffix(): tokenizer_paths, model_paths, devices, model_kwargs and tokenizer_kwargs "
                "must all have the same length."
            )

        all_models = [
            GCGModelConfig(
                name=model_paths[i],
                device=devices[i],
                model_kwargs=dict(model_kwargs[i]),
                tokenizer_kwargs=dict(tokenizer_kwargs[i]),
            )
            for i in range(len(model_paths))
        ]
        train_models = all_models[:num_train_models]
        test_models = all_models[num_train_models:]

        generator = GCGGenerator(
            models=train_models,
            test_models=test_models,
            algorithm=GCGAlgorithmConfig(
                n_steps=n_steps,
                test_steps=test_steps,
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                target_weight=target_weight,
                control_weight=control_weight,
                learning_rate=learning_rate,
                allow_non_ascii=allow_non_ascii,
                filter_cand=filter_cand,
                random_seed=random_seed,
                control_init=control_init,
            ),
            strategy=GCGStrategyConfig(
                transfer=transfer,
                progressive_goals=progressive_goals,
                progressive_models=progressive_models,
                anneal=anneal,
                incr_control=incr_control,
                stop_on_success=stop_on_success,
            ),
            output=GCGOutputConfig(
                result_prefix=result_prefix,
                logfile=logfile,
                verbose=verbose,
            ),
            hf_token=token or None,
        )
        data_config = GCGDataConfig(
            train_data=train_data,
            test_data=test_data,
            n_train_data=n_train_data,
            n_test_data=n_test_data,
        )
        return generator, data_config
