# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Cooperative deadlines for a single owned database operation."""

import threading
import time
from dataclasses import dataclass, field

from pyrit.exceptions.analytics_exception import AnalyticsTimeoutException


@dataclass
class QueryControl:
    """A monotonic execution deadline and a request-local cancellation signal."""

    deadline: float
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def remaining(self) -> float:
        """The seconds remaining before the execution deadline."""
        return max(0.0, self.deadline - time.monotonic())

    @property
    def expired(self) -> bool:
        """Whether cancellation or the execution deadline has been reached."""
        return self.cancel_event.is_set() or self.remaining <= 0

    def cancel(self) -> None:
        """Request cancellation of this operation only."""
        self.cancel_event.set()

    def check(self) -> None:
        """
        Stop an expired operation before it acquires or uses database resources.

        Raises:
            AnalyticsTimeoutException: If the operation was cancelled or expired.
        """
        if self.expired:
            raise AnalyticsTimeoutException
