# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Helper functions for discovering and configuring prompt converters.
"""

import inspect
import logging
import pkgutil
import importlib
from typing import Dict, List, Any, Optional, Type
from pyrit.prompt_converter import PromptConverter

logger = logging.getLogger(__name__)


def get_available_converters() -> List[str]:
    """
    Dynamically discover all available converter classes from pyrit.prompt_converter.
    
    Filters out converters that:
    - Start with "Fuzzer" (specialized fuzzing converters)
    - Have required parameters that can't be configured in the UI (SeedPrompt, TextJailBreak, etc.)
    - Have >4 UI-representable parameters (too complex for simple UI)
    
    Returns:
        List of converter class names found in the pyrit.prompt_converter module
    """
    import pyrit.prompt_converter as converter_module
    
    # Converters to exclude due to required complex parameters or too many parameters
    excluded_converters = {
        'SearchReplaceConverter',  # requires list[str] parameter
        'TextJailbreakConverter',  # requires TextJailBreak template
        'ImageCompressionConverter',  # 9 parameters - too complex
        'AzureSpeechTextToAudioConverter',  # 7 parameters - too complex
        'AddImageTextConverter',  # 5 parameters - too complex
        'AddTextImageConverter',  # 5 parameters - too complex
        'AzureSpeechAudioToTextConverter',  # 5 parameters - too complex
        'TransparencyAttackConverter',  # 5 parameters - too complex
    }
    
    converters = []
    
    # Get the package path
    package_path = converter_module.__path__
    
    # Iterate through all modules in the package
    for importer, modname, ispkg in pkgutil.iter_modules(package_path):
        if ispkg:
            continue  # Skip subdirectories
            
        try:
            # Import the module
            full_module_name = f"pyrit.prompt_converter.{modname}"
            module = importlib.import_module(full_module_name)
            
            # Find all classes in the module that are subclasses of PromptConverter
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (obj != PromptConverter and 
                    issubclass(obj, PromptConverter) and 
                    obj.__module__ == full_module_name):
                    # Skip Fuzzer* converters
                    if name.startswith('Fuzzer'):
                        logger.debug(f"Skipping {name} (Fuzzer converter)")
                        continue
                    # Skip converters with required complex parameters
                    if name in excluded_converters:
                        logger.debug(f"Skipping {name} (required complex parameters)")
                        continue
                    converters.append(name)
                    
        except Exception as e:
            logger.debug(f"Could not import {modname}: {e}")
            continue
    
    return sorted(converters)


def get_converter_parameters(converter_name: str) -> Dict[str, Any]:
    """
    Get the initialization parameters for a given converter.
    
    Args:
        converter_name: Name of the converter class
        
    Returns:
        Dictionary with parameter information including name, type, default, and required flag
    """
    try:
        # Dynamically import the converter
        module_name = _converter_name_to_module(converter_name)
        module = __import__(f"pyrit.prompt_converter.{module_name}", fromlist=[converter_name])
        converter_class = getattr(module, converter_name)
        
        # Get the __init__ signature
        sig = inspect.signature(converter_class.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'args', 'kwargs', 'converter_target'):
                continue  # Skip these - converter_target is auto-provided
                
            # Extract parameter info
            param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
            # Simplify type names
            param_type = param_type.replace('typing.', '').replace('<class ', '').replace('>', '').replace("'", '').replace('"', '')
            
            param_info = {
                'name': param_name,
                'required': param.default == inspect.Parameter.empty,
                'default': None if param.default == inspect.Parameter.empty else param.default,
                'type': param_type,
            }
            
            params[param_name] = param_info
            
        return params
        
    except Exception as e:
        logger.warning(f"Could not get parameters for {converter_name}: {e}")
        return {}


def is_simple_parameter_type(param_type: str) -> bool:
    """
    Check if a parameter type can be represented with a simple text input.
    
    Args:
        param_type: The parameter type string
        
    Returns:
        True if it's a simple type (str, int, float, bool), False otherwise
    """
    simple_types = ['str', 'int', 'float', 'bool']
    # Check if the base type is simple (handle Optional[str], etc.)
    for simple_type in simple_types:
        if simple_type in param_type.lower():
            return True
    return False


def get_parameter_ui_type(param_info: Dict[str, Any]) -> str:
    """
    Determine the UI input type for a parameter.
    
    Args:
        param_info: Parameter information dict with 'name', 'type', 'required', 'default'
        
    Returns:
        One of: 'textbox', 'number', 'checkbox', 'dropdown', 'complex', 'skip'
    """
    param_type = param_info['type']
    
    # Skip complex types
    if any(x in param_type.lower() for x in ['seedprompt', 'jailbreak', 'callable', 'list[', 'dict[', 'tuple[']):
        return 'skip'
    
    # Check for Literal (dropdown)
    if 'Literal[' in param_type:
        return 'dropdown'
    
    # Check for bool
    if 'bool' in param_type.lower():
        return 'checkbox'
    
    # Check for int/float
    if 'int' in param_type.lower() or 'float' in param_type.lower():
        # Skip if it looks like a tuple (e.g., color: tuple[int, int, int])
        if '(' in str(param_info.get('default', '')) and ',' in str(param_info.get('default', '')):
            return 'skip'
        return 'number'
    
    # Check for str or Path
    if 'str' in param_type.lower() or 'path' in param_type.lower():
        return 'textbox'
    
    # Default: textbox for simple optional types
    if 'Optional' in param_type:
        return 'textbox'
    
    return 'skip'


def extract_literal_choices(param_type: str) -> List[str]:
    """
    Extract choices from a Literal type annotation.
    
    Args:
        param_type: Type string like "Literal['val1', 'val2', 'val3']" or "Literal[val1, val2, val3]"
        
    Returns:
        List of choice values
    """
    import re
    # Match content inside Literal[]
    match = re.search(r'Literal\[(.*?)\]', param_type)
    if match:
        content = match.group(1)
        # First try to extract quoted strings (for "Literal['a', 'b']" format)
        quoted_choices = re.findall(r"'([^']*)'|\"([^\"]*)\"", content)
        if quoted_choices:
            # Flatten tuples and filter empty strings
            return [c[0] or c[1] for c in quoted_choices if c[0] or c[1]]
        else:
            # For unquoted format like "Literal[val1, val2]", split by comma
            choices = [c.strip() for c in content.split(',')]
            return [c for c in choices if c]  # Filter empty strings
    return []


def _converter_name_to_module(converter_name: str) -> str:
    """
    Convert converter class name to module name.
    E.g., Base64Converter -> base64_converter
    """
    # Insert underscore before capital letters and convert to lowercase
    import re
    module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', converter_name).lower()
    return module_name


def create_converter_instance(
    converter_name: str, 
    params: Dict[str, Any],
    default_target: Optional[Any] = None
) -> Optional[PromptConverter]:
    """
    Create an instance of a converter with the given parameters.
    
    Args:
        converter_name: Name of the converter class
        params: Dictionary of parameter names to values
        default_target: Optional default PromptChatTarget to use for LLM-based converters
        
    Returns:
        Instantiated converter or None if creation fails
    """
    try:
        # Dynamically import and instantiate
        module_name = _converter_name_to_module(converter_name)
        module = __import__(f"pyrit.prompt_converter.{module_name}", fromlist=[converter_name])
        converter_class = getattr(module, converter_name)
        
        # Check if this converter needs a converter_target and provide default if available
        # This must be done BEFORE filtering out None/empty values
        sig = inspect.signature(converter_class.__init__)
        if 'converter_target' in sig.parameters and 'converter_target' not in params:
            if default_target is not None:
                params['converter_target'] = default_target
                logger.debug(f"Auto-providing converter_target for {converter_name}")
        
        # Filter out None/empty values
        filtered_params = {k: v for k, v in params.items() if v not in (None, '', [])}
        
        # Instantiate with keyword arguments
        instance = converter_class(**filtered_params)
        logger.info(f"✅ Created converter: {converter_name}")
        return instance
        
    except Exception as e:
        logger.error(f"❌ Failed to create converter {converter_name}: {e}")
        return None



