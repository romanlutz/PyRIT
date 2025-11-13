# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helper functions for the PyRIT Gradio chat application.
"""

import inspect
import logging
import os

logger = logging.getLogger(__name__)


def get_available_targets():
    """
    Discover all concrete target classes from pyrit.prompt_target module.
    
    Returns:
        List of class names (sorted)
    """
    targets = []
    
    try:
        # Import the prompt_target module
        import pyrit.prompt_target as pt_module
        
        # Get all exported classes from __all__ if available
        if hasattr(pt_module, '__all__'):
            for name in pt_module.__all__:
                try:
                    obj = getattr(pt_module, name)
                    # Check if it's a class and has 'Target' in the name
                    if inspect.isclass(obj) and 'Target' in name:
                        # Check if it's not abstract
                        if not inspect.isabstract(obj):
                            targets.append(name)
                except Exception as e:
                    logger.debug(f"Skipping {name}: {e}")
        else:
            # Fallback: inspect all members
            for name, obj in inspect.getmembers(pt_module, inspect.isclass):
                if 'Target' in name and not inspect.isabstract(obj):
                    targets.append(name)
    
    except Exception as e:
        logger.error(f"Failed to discover targets: {e}")
        # Fallback to a minimal set
        targets = ['OpenAIChatTarget', 'AzureMLChatTarget', 'HuggingFaceChatTarget']
    
    # Sort by name
    targets.sort()
    
    return targets


def get_available_env_vars():
    """
    Get all available environment variables.
    
    Returns:
        List of environment variable names (sorted)
    """
    return sorted(os.environ.keys())


def get_env_var_suggestions(target_class_name: str):
    """
    Get suggested environment variable names for a target class.
    
    Returns:
        dict with keys: endpoint_var, api_key_var, model_var
    """
    # Default values
    defaults = {
        'endpoint_var': 'OPENAI_CHAT_ENDPOINT',
        'api_key_var': 'OPENAI_CHAT_KEY',
        'model_var': 'OPENAI_CHAT_MODEL',
    }
    
    # Map target classes to their environment variable prefixes
    env_var_map = {
        'OpenAIChatTarget': ('OPENAI_CHAT_ENDPOINT', 'OPENAI_CHAT_KEY', 'OPENAI_CHAT_MODEL'),
        'OpenAICompletionTarget': ('OPENAI_COMPLETION_ENDPOINT', 'OPENAI_COMPLETION_KEY', 'OPENAI_COMPLETION_MODEL'),
        'OpenAIDALLETarget': ('OPENAI_DALLE_ENDPOINT', 'OPENAI_DALLE_API_KEY', 'OPENAI_DALLE_MODEL'),
        'OpenAISoraTarget': ('OPENAI_SORA_ENDPOINT', 'OPENAI_SORA_KEY', 'OPENAI_SORA_MODEL'),
        'OpenAITTSTarget': ('OPENAI_TTS_ENDPOINT', 'OPENAI_TTS_KEY', 'OPENAI_TTS_MODEL'),
        'RealtimeTarget': ('OPENAI_REALTIME_ENDPOINT', 'OPENAI_REALTIME_KEY', 'OPENAI_REALTIME_MODEL'),
        'OpenAIResponseTarget': ('OPENAI_RESPONSE_ENDPOINT', 'OPENAI_RESPONSE_KEY', 'OPENAI_RESPONSE_MODEL'),
    }
    
    if target_class_name in env_var_map:
        endpoint, key, model = env_var_map[target_class_name]
        return {
            'endpoint_var': endpoint,
            'api_key_var': key,
            'model_var': model,
        }
    
    return defaults
