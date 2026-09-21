# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""FastAPI dependency lifetime for the SDK analytics entry point."""

from functools import lru_cache

from pyrit.analytics import AttackResultAnalytics


@lru_cache(maxsize=1)
def get_analytics_service() -> AttackResultAnalytics:
    """
    Resolve the SDK dependency lazily, after backend startup configures memory.

    Routes consume the SDK directly; there is no second analytics implementation
    or forwarding service object. Tests can override this FastAPI dependency.

    Returns:
        AttackResultAnalytics: The process-local SDK entry point bound to CentralMemory.
    """
    return AttackResultAnalytics()


def shutdown_analytics_service() -> None:
    """
    Release initialized analytics workers without creating an unused dependency.

    Shutdown joins worker threads, so the backend lifespan invokes this helper
    off the event loop. Clearing the cache permits a later startup to use new memory.
    """
    if get_analytics_service.cache_info().currsize:
        get_analytics_service().shutdown()
        get_analytics_service.cache_clear()
