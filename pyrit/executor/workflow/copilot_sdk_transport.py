# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Compatible Copilot names for the harness-neutral whole-body observation transport."""

from pyrit.executor.workflow.native_model_transport import (
    NativeModelForwarder as CopilotSdkModelForwarder,
)
from pyrit.executor.workflow.native_model_transport import (
    NativeModelRequest as CopilotSdkModelRequest,
)
from pyrit.executor.workflow.native_model_transport import (
    NativeModelResponse as CopilotSdkModelResponse,
)
from pyrit.executor.workflow.native_model_transport import (
    NativeModelTransport as CopilotSdkModelTransport,
)

__all__ = [
    "CopilotSdkModelForwarder",
    "CopilotSdkModelRequest",
    "CopilotSdkModelResponse",
    "CopilotSdkModelTransport",
]
