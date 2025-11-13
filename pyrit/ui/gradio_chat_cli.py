#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
CLI for launching the PyRIT Endpoint Chat Gradio app.

This CLI allows you to launch the chat interface with different prompt targets.
Target parameters (API keys, endpoints, model names) are read from environment
variables, which in Docker can be populated from Azure Key Vault.

The app includes a Configuration tab where you can dynamically select and configure
any PyRIT target without restarting the application.

Usage:
    # Launch with default OpenAI Chat Target (uses OPENAI_CHAT_ENDPOINT, OPENAI_CHAT_KEY, OPENAI_CHAT_MODEL)
    python -m pyrit.ui.gradio_chat_cli

    # Launch on a specific port
    python -m pyrit.ui.gradio_chat_cli --port 8080

    # Enable debug logging
    python -m pyrit.ui.gradio_chat_cli --debug

Docker Usage (with Key Vault integration):
    # The Docker container can be configured to pull secrets from Azure Key Vault
    # and populate them as environment variables automatically
    docker run -p 7860:7860 \\
        -e OPENAI_CHAT_ENDPOINT \\
        -e OPENAI_CHAT_KEY \\
        -e OPENAI_CHAT_MODEL \\
        pyrit-chat

    # Or with Key Vault auto-injection (no -e flags needed)
    docker run -p 7860:7860 pyrit-chat

Configuration Tab:
    Once the app is running, use the ⚙️ Configuration tab to:
    - Select from ALL available PyRIT targets (discovered dynamically)
    - Configure environment variables for endpoint, API key, and model
    - Apply changes without restarting the application
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


def print_target_config(target: Any, target_kwargs: dict) -> None:
    """
    Print the configuration of the created target using reflection.
    Automatically discovers all instance attributes (except sensitive ones).
    
    Args:
        target: The instantiated target instance
        target_kwargs: The kwargs that were passed to create the target
    """
    print(f"\n📋 Target Configuration:")
    print(f"{'='*70}")
    print(f"Target Class: {target.__class__.__name__}")
    print(f"Target Module: {target.__class__.__module__}")
    
    # Use reflection to discover all attributes
    print(f"\nDiscovered Parameters:")
    
    # Get all attributes that don't start with '__' (dunder methods)
    all_attrs = {name: getattr(target, name) for name in dir(target) if not name.startswith('__')}
    
    # Sort attributes: instance variables first (starting with _), then public
    instance_vars = {k: v for k, v in all_attrs.items() if k.startswith('_') and not callable(v)}
    
    # Filter sensitive attributes
    sensitive_keywords = ['key', 'token', 'secret', 'password', 'credential', 'auth']
    
    # Print all instance variables
    for attr_name in sorted(instance_vars.keys()):
        value = instance_vars[attr_name]
        
        # Check if this is a sensitive attribute
        is_sensitive = any(keyword in attr_name.lower() for keyword in sensitive_keywords)
        
        if is_sensitive:
            # Mask sensitive values
            if value and len(str(value)) > 8:
                value = f"{str(value)[:8]}...{str(value)[-4:]}"
            elif value:
                value = "***"
        
        # Format dict values nicely
        if isinstance(value, dict):
            if value:
                print(f"  {attr_name}:")
                for k, v in value.items():
                    # Mask sensitive dict values
                    if any(keyword in k.lower() for keyword in sensitive_keywords):
                        v = "***"
                    print(f"    {k}: {v}")
            continue
        
        # Format list values
        if isinstance(value, list):
            if value:
                print(f"  {attr_name}: {value}")
            continue
        
        # Skip None, empty strings, and callables
        if value is None or value == "" or callable(value):
            continue
        
        # Skip very long strings (likely internal data)
        if isinstance(value, str) and len(value) > 200:
            continue
        
        print(f"  {attr_name}: {value}")
    
    # Print any kwargs that were explicitly passed
    if target_kwargs:
        print(f"\nExplicitly Provided Parameters:")
        for key, value in sorted(target_kwargs.items()):
            print(f"  {key}: {value}")
    
    print(f"{'='*70}\n")


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
    
    The app always starts with OpenAIChatTarget by default.
    Use the Configuration tab in the UI to switch to other targets.
    """
    parser = argparse.ArgumentParser(
        description="Launch PyRIT Endpoint Chat Gradio app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with default settings
  %(prog)s

  # Launch on a specific port
  %(prog)s --port 8080

  # Enable debug logging
  %(prog)s --debug

Configuration Tab:
  The app includes a ⚙️ Configuration tab where you can dynamically switch between
  ANY available PyRIT target and configure environment variables without restarting.
  All targets from pyrit.prompt_target are discovered automatically.

Default Target:
  The app starts with OpenAIChatTarget using these environment variables:
  - OPENAI_CHAT_ENDPOINT (required)
  - OPENAI_CHAT_KEY (optional, for API authentication)
  - OPENAI_CHAT_MODEL (optional, model name)
  
  You can switch to any other target using the Configuration tab.
        """,
    )

    # Server configuration
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
    from pyrit.setup import SQLITE, initialize_pyrit
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

    # Create the default target (OpenAIChatTarget)
    try:
        print(f"🔧 Creating OpenAIChatTarget (default)...")
        print(f"📝 Parameters will be loaded from environment variables:")
        print(f"    - OPENAI_CHAT_ENDPOINT (required)")
        print(f"    - OPENAI_CHAT_KEY (optional)")
        print(f"    - OPENAI_CHAT_MODEL (optional)")
        
        target = create_target(target_class_name="OpenAIChatTarget")
        print(f"✅ Target created: {target.__class__.__name__}")
        
        # Print detailed configuration
        print_target_config(target, {})
        
    except Exception as e:
        print(f"❌ Error creating target: {e}")
        print(f"\n💡 Tip: Ensure the required environment variables are set:")
        print(f"    - OPENAI_CHAT_ENDPOINT is required")
        print(f"    - OPENAI_CHAT_KEY (if authentication is needed)")
        print(f"    - OPENAI_CHAT_MODEL (optional model name)")
        print(f"\n💡 You can also switch to a different target using the ⚙️ Configuration tab after launch.")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)    # Create and launch the app
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
        print("  • Dynamic target switching via ⚙️ Configuration tab")
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
