# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Co-PyRIT: Gradio chat application for interacting with PyRIT endpoints.

This app supports multi-modal inputs (text, image, video, audio) and outputs,
allowing users to chat with any PyRIT-compatible endpoint.

Usage:
    python -m pyrit.ui.endpoint_chat_app
"""

import asyncio
import importlib
import inspect
import logging
import os
from pathlib import Path
from typing import Union
from uuid import uuid4

import gradio as gr

from pyrit.common.path import DB_DATA_PATH
from pyrit.memory import CentralMemory
from pyrit.models import SeedGroup, SeedPrompt
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit
from pyrit.ui.tabs import build_chat_tab, build_conversations_tab, build_configuration_tab
from pyrit.ui.tabs.helpers import get_available_targets, get_available_env_vars, get_env_var_suggestions

logger = logging.getLogger(__name__)


class EndpointChatApp:
    """Gradio application for chatting with PyRIT endpoints."""

    def __init__(self, *, target: Union[PromptTarget, PromptChatTarget], enable_config_tab: bool = True):
        """
        Initialize the chat app with a prompt target.

        Args:
            target: The prompt target to use for sending messages.
                    Can be any PromptTarget or PromptChatTarget instance
                    (e.g., OpenAIChatTarget, AzureMLChatTarget, etc.)
            enable_config_tab: Whether to enable the configuration tab for dynamic target switching
        
        Note:
            Memory must be initialized before creating the app (via initialize_pyrit).
            The CLI handles this automatically.
        """
        self.target = target
        self.conversation_id = str(uuid4())
        self.enable_config_tab = enable_config_tab
        
        # Get the already-initialized memory instance
        self.memory = CentralMemory.get_memory_instance()
        
        # Initialize prompt normalizer for proper conversation handling
        self.prompt_normalizer = PromptNormalizer()
        
        # Track messages per conversation to prevent mixing
        self._conversation_message_count = 0
        self._last_cleared_conversation_id = None

    def _create_target_from_config(self, target_class_name: str, endpoint_var: str, api_key_var: str, model_var: str):
        """
        Create a new target instance from configuration.
        
        Args:
            target_class_name: Name of the target class (e.g., 'OpenAIChatTarget')
            endpoint_var: Environment variable name for endpoint
            api_key_var: Environment variable name for API key
            model_var: Environment variable name for model name
            
        Returns:
            Tuple of (target_instance, error_message or None)
        """
        try:
            # Get environment variable values
            endpoint = os.environ.get(endpoint_var)
            api_key = os.environ.get(api_key_var)
            model_name = os.environ.get(model_var)
            
            # Validate required values
            if not endpoint:
                return None, f"❌ Environment variable '{endpoint_var}' is not set"
            
            # Import the target class
            module = importlib.import_module('pyrit.prompt_target')
            target_class = getattr(module, target_class_name)
            
            # Create kwargs dict with non-None values
            kwargs = {}
            if endpoint:
                kwargs['endpoint'] = endpoint
            if api_key:
                kwargs['api_key'] = api_key
            if model_name:
                kwargs['model_name'] = model_name
            
            # Create the target
            target = target_class(**kwargs)
            
            logger.info(f"✅ Created {target_class_name} with endpoint={endpoint}, model={model_name}")
            
            return target, None
            
        except Exception as e:
            error_msg = f"❌ Failed to create target: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def _rebuild_history_from_database(self) -> list[dict]:
        """
        Rebuild the complete conversation history from the database for the current conversation_id.
        Handles text, images, video, and audio content types.
        
        Returns:
            List of message dictionaries in Gradio's "messages" format with multi-modal content
            Format: [{"role": "user"|"assistant", "content": str | {"path": str} | {"url": str}}]
        """
        conversation = self.memory.get_conversation(conversation_id=self.conversation_id)
        gradio_history = []
        
        for response in conversation:
            for piece in response.message_pieces:
                if piece.role in ["user", "assistant"]:
                    # Determine content type and format accordingly
                    data_type = piece.converted_value_data_type or piece.original_value_data_type
                    value = piece.converted_value or piece.original_value
                    
                    if data_type == "image_path":
                        # Image - use file path for Gradio to render
                        if value and os.path.exists(value):
                            gradio_history.append({
                                "role": piece.role,
                                "content": {"path": value}
                            })
                        else:
                            # If image doesn't exist, show path as text
                            gradio_history.append({
                                "role": piece.role,
                                "content": f"🖼️ Image: {value}"
                            })
                    elif data_type in ["video_path", "audio_path"]:
                        # Video/Audio - use file path
                        if value and os.path.exists(value):
                            gradio_history.append({
                                "role": piece.role,
                                "content": {"path": value}
                            })
                        else:
                            media_type = "🎥 Video" if data_type == "video_path" else "🎵 Audio"
                            gradio_history.append({
                                "role": piece.role,
                                "content": f"{media_type}: {value}"
                            })
                    elif data_type == "text":
                        # Text content
                        gradio_history.append({
                            "role": piece.role,
                            "content": value
                        })
                    else:
                        # Fallback to text representation
                        gradio_history.append({
                            "role": piece.role,
                            "content": str(value)
                        })
        
        logger.info(f"📚 Rebuilt history: {len(gradio_history)} messages from database for conversation {self.conversation_id}")
        return gradio_history


    async def _chat_async(self, message: dict, history: list[dict]) -> tuple[str, list[dict]]:
        """
        Process a chat message with multi-modal support using PromptNormalizer.
        
        IMPORTANT: We completely ignore Gradio's history parameter and rebuild the entire
        conversation from PyRIT's database to ensure conversations stay isolated.

        Args:
            message: Dictionary containing 'text' and optional 'files' list
            history: List of previous messages from Gradio (IGNORED - we use database instead)

        Returns:
            Tuple of (response text, complete conversation history from database)
        """
        # Log the current state
        logger.info(f"💬 Sending message in conversation {self.conversation_id} (ignoring Gradio history with {len(history)} messages)")
        
        # Build seed prompts for the message
        seed_prompts = []

        # Add text if present
        text_content = message.get("text", "")
        if text_content and text_content.strip():
            seed_prompts.append(
                SeedPrompt(
                    value=text_content,
                    data_type="text",
                )
            )

        # Add any files (images, videos, audio)
        files = message.get("files", [])
        for file_path in files:
            if not file_path:
                continue

            # Determine file type from extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
                data_type = "image_path"
            elif file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                data_type = "video_path"
            elif file_ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
                data_type = "audio_path"
            else:
                data_type = "text"  # Default to text for unknown types

            seed_prompts.append(
                SeedPrompt(
                    value=file_path,
                    data_type=data_type,
                )
            )

        if not seed_prompts:
            # Return error and current history
            return "❌ Please provide a message or file.", self._rebuild_history_from_database()

        try:
            # Create seed prompt group
            seed_group = SeedGroup(prompts=seed_prompts)
            
            # Send using PromptNormalizer - this handles conversation history automatically
            # by using the conversation_id to track and include previous messages
            response = await self.prompt_normalizer.send_prompt_async(
                seed_group=seed_group,
                target=self.target,
                conversation_id=self.conversation_id,
            )

            # Collect all response pieces into a single message
            response_parts = []
            for piece in response.message_pieces:
                if piece.converted_value_data_type == "text":
                    response_parts.append(piece.converted_value)
                elif piece.converted_value_data_type in ["image_path", "video_path", "audio_path"]:
                    # Include file references in the response
                    response_parts.append(f"[{piece.converted_value_data_type}: {piece.converted_value}]")

            response_text = "\n".join(response_parts) if response_parts else "✅ Response received"
            
            # Rebuild the complete conversation history from database
            # This ensures we show only messages from the current conversation_id
            updated_history = self._rebuild_history_from_database()
            
            return response_text, updated_history

        except Exception as e:
            # Return error and current history
            return f"❌ Error: {str(e)}", self._rebuild_history_from_database()

    def chat(self, message: dict, history: list[dict]) -> str:
        """
        Synchronous wrapper for chat processing.

        Args:
            message: Dictionary containing 'text' and optional 'files' list
            history: List of previous messages (ignored)

        Returns:
            Response text only (ChatInterface manages display)
        """
        response_text, _ = asyncio.run(self._chat_async(message, history))
        return response_text

    def _get_all_conversations_table(self):
        """
        Get all conversations from memory and format them as a table.
        
        Returns:
            List of lists for Gradio DataFrame
        """
        # Get all message pieces from memory
        all_pieces = self.memory.get_message_pieces()
        
        # Group by conversation_id and collect metadata
        conversations_dict = {}
        for piece in all_pieces:
            conv_id = piece.conversation_id
            if conv_id not in conversations_dict:
                conversations_dict[conv_id] = {
                    'conversation_id': conv_id,
                    'labels': piece.labels or {},
                    'metadata': piece.prompt_metadata or {},
                    'non_system_messages': set(),  # Track unique (sequence, role) pairs for non-system messages
                    'first_message': None,
                    'last_message': None,
                    'first_user_prompt': None,
                }
            
            # Count non-system messages by (sequence, role) pair - each unique pair is one message
            # This correctly handles multimodal messages that have multiple pieces with the same sequence
            if piece.role in ["user", "assistant"] and piece.sequence is not None:
                conversations_dict[conv_id]['non_system_messages'].add((piece.sequence, piece.role))
            
            # Track first user prompt (role == "user" and converted_value_data_type == "text")
            if (conversations_dict[conv_id]['first_user_prompt'] is None 
                and piece.role == "user" 
                and piece.converted_value_data_type == "text"):
                # Truncate long prompts
                prompt_text = piece.converted_value or ""
                if len(prompt_text) > 100:
                    prompt_text = prompt_text[:97] + "..."
                conversations_dict[conv_id]['first_user_prompt'] = prompt_text
            
            # Track first and last message timestamps
            if piece.timestamp:
                if conversations_dict[conv_id]['first_message'] is None:
                    conversations_dict[conv_id]['first_message'] = piece.timestamp
                else:
                    conversations_dict[conv_id]['first_message'] = min(
                        conversations_dict[conv_id]['first_message'], piece.timestamp
                    )
                
                if conversations_dict[conv_id]['last_message'] is None:
                    conversations_dict[conv_id]['last_message'] = piece.timestamp
                else:
                    conversations_dict[conv_id]['last_message'] = max(
                        conversations_dict[conv_id]['last_message'], piece.timestamp
                    )
        
        # Convert to table format
        table_data = []
        for conv_data in conversations_dict.values():
            table_data.append([
                conv_data['conversation_id'],
                len(conv_data['non_system_messages']),  # Count unique non-system messages
                conv_data['first_user_prompt'] or 'N/A',
                str(conv_data['labels']),
                str(conv_data['metadata']),
                str(conv_data['first_message']) if conv_data['first_message'] else 'N/A',
                str(conv_data['last_message']) if conv_data['last_message'] else 'N/A',
            ])
        
        # Sort by last message (most recent first)
        table_data.sort(key=lambda x: x[6], reverse=True)
        
        return table_data

    def build_interface(self) -> gr.Blocks:
        """
        Build and return the Gradio interface with multiple pages.

        Returns:
            Gradio Blocks interface
        """
        # Get target info for display
        target_name = self.target.__class__.__name__
        
        # Get path to roakey.png image
        roakey_path = Path(__file__).parent.parent.parent / "doc" / "roakey.png"
        
        with gr.Blocks(title="Co-PyRIT", theme=gr.themes.Default()) as demo:
            # Header with logo and title
            with gr.Row():
                if roakey_path.exists():
                    gr.Image(value=str(roakey_path), height=60, width=70, show_label=False, show_download_button=False, show_fullscreen_button=False, container=False, scale=0, min_width=70)
                gr.Markdown("# Co-PyRIT")
            
            # Create chatbot component first (outside tabs) so all tabs can reference it
            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                show_copy_button=True,
                value=[],
                visible=False,  # Will be made visible in Chat tab
                render=False  # Don't render it yet
            )
            
            # Navigation tabs - Configuration tab is selected by default
            with gr.Tabs(selected="config_tab") as tabs:
                # ===== CONFIGURATION TAB (FIRST) =====
                build_configuration_tab(self, tabs, chatbot, target_name)
        
                # ===== CONVERSATIONS TAB (SECOND) =====
                conversations_tab_component, conversations_table = build_conversations_tab(self, tabs, chatbot)

                # ===== CHAT TAB (THIRD) =====
                build_chat_tab(self, tabs, target_name, chatbot)
            
            # Auto-refresh conversations table when tab is selected
            def refresh_on_tab_select(evt: gr.SelectData):
                """Refresh conversations table when Conversations tab is selected"""
                # Check if the selected tab is the conversations tab
                # evt.value contains the tab ID
                if evt.value == "conversations_tab" or "Conversations" in str(evt.value):
                    logger.info(f"🔄 Auto-refreshing conversations table (tab selected: {evt.value})")
                    return self._get_all_conversations_table()
                return gr.update()  # Return empty update for other tabs
            
            tabs.select(
                fn=refresh_on_tab_select,
                inputs=None,
                outputs=[conversations_table]
            )

        return demo

    def launch(self, **kwargs):
        """
        Launch the Gradio app.

        Args:
            **kwargs: Additional arguments to pass to gr.launch()
        """
        demo = self.build_interface()
        
        # Add DB_DATA_PATH to allowed_paths so Gradio can serve media files
        # Using .resolve() to get the absolute path that Gradio recognizes
        allowed_paths = kwargs.get("allowed_paths", [])
        if not isinstance(allowed_paths, list):
            allowed_paths = []
        
        db_data_str = str(DB_DATA_PATH.resolve())
        if db_data_str not in allowed_paths:
            allowed_paths.append(db_data_str)
        
        kwargs["allowed_paths"] = allowed_paths
        
        logger.info(f"🔓 Gradio allowed_paths: {db_data_str}")
        
        demo.launch(**kwargs)
