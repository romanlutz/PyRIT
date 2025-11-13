# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configuration tab UI for the PyRIT Gradio chat application.
"""

import logging
import os
from uuid import uuid4

import gradio as gr

from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.ui.tabs.helpers import get_available_targets, get_available_env_vars, get_env_var_suggestions

logger = logging.getLogger(__name__)


def build_configuration_tab(app_instance, tabs, chatbot, target_name: str):
    """
    Build the configuration tab UI with target selection and event handlers.
    
    Args:
        app_instance: The EndpointChatApp instance for accessing methods and state
        tabs: The Tabs component for navigation
        chatbot: The Chatbot component to clear when applying configuration
        target_name: Current target name for display
    """
    if not app_instance.enable_config_tab:
        return
        
    with gr.Tab("⚙️ Configuration", id="config_tab"):
        gr.Markdown("## Target Configuration")
        gr.Markdown("Configure the prompt target for the chat interface. All available targets from `pyrit.prompt_target` are listed.")
        
        # Get available targets dynamically
        available_targets = get_available_targets()
        
        # Current target display
        with gr.Row():
            current_target_display = gr.Markdown(f"**Current Target:** `{target_name}`")
        
        # Target selection
        target_dropdown = gr.Dropdown(
            choices=available_targets,
            value=target_name if target_name in available_targets else (available_targets[0] if available_targets else None),
            label="Select Target Type",
            info=f"Choose from {len(available_targets)} available targets"
        )
        
        # Environment variable inputs
        with gr.Group():
            gr.Markdown("### Environment Variables")
            gr.Markdown("💡 **Select environment variables from your system.** API key values are never displayed for security.")
            
            # Get available env vars
            available_env_vars = get_available_env_vars()
            
            endpoint_var_input = gr.Dropdown(
                choices=available_env_vars,
                value="OPENAI_CHAT_ENDPOINT" if "OPENAI_CHAT_ENDPOINT" in available_env_vars else None,
                label="Endpoint Variable Name",
                info="Select the environment variable containing the API endpoint",
                allow_custom_value=True
            )
            
            api_key_var_input = gr.Dropdown(
                choices=available_env_vars,
                value="OPENAI_CHAT_KEY" if "OPENAI_CHAT_KEY" in available_env_vars else None,
                label="API Key Variable Name",
                info="Select the environment variable containing the API key",
                allow_custom_value=True
            )
            
            model_var_input = gr.Dropdown(
                choices=available_env_vars,
                value="OPENAI_CHAT_MODEL" if "OPENAI_CHAT_MODEL" in available_env_vars else None,
                label="Model Variable Name",
                info="Select the environment variable containing the model name",
                allow_custom_value=True
            )
        
        # Current values display (without API key)
        with gr.Group():
            gr.Markdown("### Current Configuration")
            endpoint_display = gr.Textbox(
                label="Endpoint",
                value=os.environ.get("OPENAI_CHAT_ENDPOINT", "Not set"),
                interactive=False
            )
            model_display = gr.Textbox(
                label="Model",
                value=os.environ.get("OPENAI_CHAT_MODEL", "Not set"),
                interactive=False
            )
            api_key_status = gr.Textbox(
                label="API Key Status",
                value="✅ Set" if os.environ.get("OPENAI_CHAT_KEY") else "❌ Not set",
                interactive=False
            )
        
        # Update env var suggestions when target changes
        def update_env_var_suggestions(target_class_name):
            """Update environment variable suggestions based on selected target"""
            suggestions = get_env_var_suggestions(target_class_name)
            
            # Get current values
            endpoint = os.environ.get(suggestions['endpoint_var'], "Not set")
            model = os.environ.get(suggestions['model_var'], "Not set")
            api_key_set = "✅ Set" if os.environ.get(suggestions['api_key_var']) else "❌ Not set"
            
            return (
                suggestions['endpoint_var'],
                suggestions['api_key_var'],
                suggestions['model_var'],
                endpoint,
                model,
                api_key_set
            )
        
        target_dropdown.change(
            fn=update_env_var_suggestions,
            inputs=[target_dropdown],
            outputs=[endpoint_var_input, api_key_var_input, model_var_input, endpoint_display, model_display, api_key_status]
        )
        
        # Apply configuration button
        with gr.Row():
            apply_config_btn = gr.Button("🔄 Apply Configuration", variant="primary")
            config_status = gr.Markdown("")
        
        def apply_configuration(target_class_name, endpoint_var, api_key_var, model_var):
            """Apply the new configuration and recreate the target"""
            new_target, error = app_instance._create_target_from_config(
                target_class_name, endpoint_var, api_key_var, model_var
            )
            
            if error:
                return (
                    f"**Status:** {error}",
                    f"**Current Target:** `{app_instance.target.__class__.__name__}`",
                    []  # Clear chat on error
                )
            
            # Update the target
            app_instance.target = new_target
            app_instance.prompt_normalizer = PromptNormalizer()  # Reset normalizer
            
            # Start a new conversation
            old_id = app_instance.conversation_id
            new_id = str(uuid4())
            app_instance.conversation_id = new_id
            app_instance._conversation_message_count = 0
            
            # Get endpoint and model for display
            endpoint = os.environ.get(endpoint_var, "Not set")
            model = os.environ.get(model_var, "Not set")
            
            logger.info(f"✅ Applied new configuration: {target_class_name}, endpoint={endpoint}, model={model}")
            logger.info(f"🆕 Started new conversation: {new_id} (previous: {old_id})")
            
            status_msg = f"""**Status:** ✅ Configuration applied successfully!
            
**Endpoint:** `{endpoint}`
**Model:** `{model}`
**New conversation started.**"""
            
            return (
                status_msg,
                f"**Current Target:** `{target_class_name}`",
                []  # Clear chat history
            )
        
        apply_config_btn.click(
            fn=apply_configuration,
            inputs=[target_dropdown, endpoint_var_input, api_key_var_input, model_var_input],
            outputs=[config_status, current_target_display, chatbot]
        )
        
        gr.Markdown("""
        ### Instructions:
        1. **Select a target type** from the dropdown (all PyRIT targets are available)
        2. **Select environment variables** from the dropdowns (or type custom names)
        3. **Check current configuration** to see what values are loaded from your environment
        4. **Click "Apply Configuration"** to recreate the target with the selected settings
        
        💡 **Tips:**
        - All environment variables from your system are available in the dropdowns
        - You can also type a custom variable name if needed
        - Environment variables must be set before launching the app (e.g., in `.env` file)
        - Applying a new configuration will start a fresh conversation
        - API keys are never displayed for security reasons
        - Different targets may require different environment variables
        - The env var suggestions auto-update based on the selected target
        """)
