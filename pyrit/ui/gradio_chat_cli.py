#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
CLI for launching the PyRIT Endpoint Chat Gradio app.

This CLI allows you to launch the chat interface with different prompt targets.
Target parameters (API keys, endpoints, model names) are read from environment
variables, which in Docker can be populated from Azure Key Vault.

Usage:
    # OpenAI Chat Target (parameters from environment)
    export ENDPOINT="https://api.openai.com/v1/chat/completions"
    export API_KEY="sk-..."
    export MODEL_NAME="gpt-4o"
    python -m pyrit.ui.gradio_chat_cli --target-class OpenAIChatTarget

    # Azure ML Target
    export ENDPOINT="https://your-endpoint.inference.ml.azure.com/score"
    export API_KEY="your-key"
    python -m pyrit.ui.gradio_chat_cli --target-class AzureMLChatTarget

    # Hugging Face Endpoint Target
    export HF_TOKEN="hf_..."
    export ENDPOINT="https://api-inference.huggingface.co/models/..."
    export MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
    python -m pyrit.ui.gradio_chat_cli --target-class HuggingFaceEndpointTarget

Docker Usage (with Key Vault integration):
    # The Docker container can be configured to pull secrets from Azure Key Vault
    # and populate them as environment variables automatically
    docker run -p 7860:7860 \\
        -e ENDPOINT \\
        -e API_KEY \\
        -e MODEL_NAME \\
        pyrit-chat --target-class OpenAIChatTarget

    # Or with Key Vault auto-injection (no -e flags needed)
    docker run -p 7860:7860 pyrit-chat --target-class OpenAIChatTarget
"""

import argparse
import importlib.util
import inspect
import sys
from importlib import import_module
from typing import Any, Optional, Type


def check_gradio_installed() -> bool:
    """Check if gradio is installed."""
    return importlib.util.find_spec("gradio") is not None


def load_target_class(*, class_name: str) -> Type[Any]:
    """
    Dynamically import a target class from pyrit.prompt_target by name.

    Args:
        class_name: Name of the target class (e.g., 'OpenAIChatTarget')

    Returns:
        The target class type

    Raises:
        RuntimeError: If the class cannot be imported or is not a valid class
    """
    try:
        mod = import_module("pyrit.prompt_target")
        cls = getattr(mod, class_name)
        if not inspect.isclass(cls):
            raise TypeError(f"The attribute {class_name} in module pyrit.prompt_target is not a class.")
    except AttributeError as ex:
        raise RuntimeError(
            f"Failed to import {class_name} from pyrit.prompt_target. "
            f"Available targets can be found at: https://github.com/Azure/PyRIT/tree/main/pyrit/prompt_target/"
        ) from ex
    except Exception as ex:
        raise RuntimeError(f"Failed to import {class_name} from pyrit.prompt_target: {ex}") from ex

    return cls


def create_target(*, target_class_name: str, **target_kwargs: Any) -> Any:
    """
    Create a prompt target instance by dynamically loading the class and instantiating it.

    Args:
        target_class_name: Name of the target class (e.g., 'OpenAIChatTarget', 'AzureMLChatTarget')
        **target_kwargs: Keyword arguments to pass to the target's __init__ method

    Returns:
        Configured PromptTarget instance

    Raises:
        RuntimeError: If the target class cannot be loaded
        TypeError: If required parameters are missing for the target
    """
    target_class = load_target_class(class_name=target_class_name)

    # Filter out None values to avoid passing them to the constructor
    filtered_kwargs = {k: v for k, v in target_kwargs.items() if v is not None}

    try:
        return target_class(**filtered_kwargs)
    except TypeError as ex:
        # Get the constructor signature to provide helpful error message
        sig = inspect.signature(target_class.__init__)
        required_params = [
            param.name
            for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and param.name not in ("self", "cls")
        ]
        raise TypeError(
            f"Failed to instantiate {target_class_name}. "
            f"Required parameters: {required_params}. "
            f"Error: {ex}"
        ) from ex


def parse_args():
    """
    Parse command-line arguments.
    
    This parser only handles app-specific arguments (target class, host, port, etc.).
    Target constructor parameters should be provided via environment variables,
    which in Docker can be populated from Azure Key Vault.
    """
    parser = argparse.ArgumentParser(
        description="Launch PyRIT Endpoint Chat Gradio app with dynamic target loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target-class OpenAIChatTarget

Available Target Classes:
  OpenAIChatTarget, OpenAISoraTarget, OpenAIDALLETarget, OpenAITTSTarget,
  AzureMLChatTarget, HuggingFaceChatTarget, HuggingFaceEndpointTarget,
  HTTPTarget, HTTPXAPITarget, PlaywrightTarget, TextTarget, and more.

  See: https://github.com/Azure/PyRIT/tree/main/pyrit/prompt_target/

Note:
  Target constructor parameters (endpoint, api_key, etc.) should be set as
  environment variables. Refer to the target class documentation for required parameters.
        """,
    )

    parser.add_argument(
        "--target-class",
        "-t",
        required=True,
        help="Target class name (e.g., OpenAIChatTarget)",
    )

    # Server configuration (not target parameters)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0, allows external access)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link (for demos)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Check dependencies
    if not check_gradio_installed():
        print("❌ Error: Gradio is not installed.")
        print("\nTo use the Endpoint Chat App, please install gradio:")
        print("  pip install gradio")
        print("\nOr install PyRIT with gradio support:")
        print("  pip install pyrit[gradio]")
        sys.exit(1)

    # Enable debug logging if requested
    if args.debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    # Initialize PyRIT memory (required for targets)
    from pyrit.common import SQLITE, initialize_pyrit
    from pyrit.memory import CentralMemory

    print(f"🔧 Initializing PyRIT memory...")
    initialize_pyrit(memory_db_type=SQLITE)
    
    # Verify memory is working by doing a test query
    memory = CentralMemory.get_memory_instance()
    print(f"✅ Memory initialized: {type(memory).__name__}")
    
    # Test that we can query the database (this will create tables if they don't exist)
    try:
        _ = memory.get_all_embeddings()
        print(f"✅ Memory database verified and ready")
    except Exception as e:
        print(f"❌ Memory database verification failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Create the target (parameters come from environment variables)
    try:
        print(f"🔧 Creating {args.target_class} target...")
        print(f"📝 Note: Target parameters should be set as environment variables")
        target = create_target(target_class_name=args.target_class)
        print(f"✅ Target created: {target.__class__.__name__}")
    except Exception as e:
        print(f"❌ Error creating target: {e}")
        print(f"\n💡 Tip: Ensure all required parameters are set as environment variables.")
        print(f"    Refer to the {args.target_class} documentation for required parameters.")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Create and launch the app
    try:
        from pyrit.ui.endpoint_chat_app import EndpointChatApp

        print(f"\n🚀 Launching PyRIT Endpoint Chat App...")
        print(f"📍 Server: http://{args.host}:{args.port}")
        if args.host == "0.0.0.0":
            print(f"📍 Access from host: http://localhost:{args.port}")
        print(f"🎯 Target: {target.__class__.__name__}")
        print("\n✨ Features:")
        print("  • Multi-modal input (text, image, video, audio)")
        print("  • Multi-modal output (text, image, video, audio)")
        print("  • Conversation management")
        print("\nPress Ctrl+C to stop the app.\n")

        app = EndpointChatApp(target=target)
        app.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
        )

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
