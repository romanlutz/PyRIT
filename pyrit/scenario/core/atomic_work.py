# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The small execution contract shared by target-owned and task-owned work."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pyrit.executor.attack import AttackExecutor, AttackExecutorResult
    from pyrit.models import AttackResult


@runtime_checkable
class AtomicWork(Protocol):
    """One named, attributable unit scheduled by a Scenario."""

    atomic_attack_name: str
    display_group: str

    @property
    def technique_eval_hash(self) -> str:
        """The configuration hash used to attribute this work."""
        ...

    @property
    def logical_group_id(self) -> str:
        """The stable work-group ID within a scenario run."""
        ...

    def set_scenario_result_id(self, scenario_result_id: str | None) -> None:
        """Bind the parent scenario result before execution."""
        ...

    async def run_async(
        self,
        *,
        executor: AttackExecutor | None = None,
        return_partial_on_failure: bool = True,
        **attack_params: Any,
    ) -> AttackExecutorResult[AttackResult]:
        """Execute and return persisted results or surface failures."""
        ...
