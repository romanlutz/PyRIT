# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Gradio chat application for interacting with PyRIT endpoints.

This app supports multi-modal inputs (text, image, video, audio) and outputs,
allowing users to chat with any PyRIT-compatible endpoint.

Usage:
    python -m pyrit.ui.endpoint_chat_app
"""

import asyncio
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

logger = logging.getLogger(__name__)


class EndpointChatApp:
    """Gradio application for chatting with PyRIT endpoints."""

    def __init__(self, *, target: Union[PromptTarget, PromptChatTarget]):
        """
        Initialize the chat app with a prompt target.

        Args:
            target: The prompt target to use for sending messages.
                    Can be any PromptTarget or PromptChatTarget instance
                    (e.g., OpenAIChatTarget, AzureMLChatTarget, etc.)
        
        Note:
            Memory must be initialized before creating the app (via initialize_pyrit).
            The CLI handles this automatically.
        """
        self.target = target
        self.conversation_id = str(uuid4())
        
        # Get the already-initialized memory instance
        self.memory = CentralMemory.get_memory_instance()
        
        # Initialize prompt normalizer for proper conversation handling
        self.prompt_normalizer = PromptNormalizer()
        
        # Track messages per conversation to prevent mixing
        self._conversation_message_count = 0
        self._last_cleared_conversation_id = None

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

    def build_interface(self) -> gr.Blocks:
        """
        Build and return the Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        # Get target info for display
        target_name = self.target.__class__.__name__
        
        with gr.Blocks(title="PyRIT Endpoint Chat", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# PyRIT Endpoint Chat 🤖")
            
            with gr.Row():
                gr.Markdown(
                    f"**Target:** `{target_name}` | **Multi-modal**: Text, Images, Videos, Audio"
                )
                new_chat_btn = gr.Button("🆕 New Chat", size="sm", scale=0)

            # Manual chatbot with full control over history
            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                show_copy_button=True,
                value=[],
            )
            
            # Input textbox
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1)
            
            # File upload
            files = gr.Files(
                label="Attach files (images, videos, audio)",
                file_count="multiple",
                type="filepath",
            )
            
            # Handle send message - show user message immediately, then get response
            async def send_message_async(user_message, uploaded_files, current_history):
                """Send a message and update UI immediately"""
                if not user_message or not user_message.strip():
                    yield current_history, "", None
                    return
                
                # Build the user's message for display
                user_content = user_message
                
                # Add user message to history immediately for instant feedback
                updated_history = current_history + [{"role": "user", "content": user_content}]
                
                # Yield the updated history with user message (clears input too)
                yield updated_history, "", None
                
                # Now build message dict for chat function
                message_dict = {"text": user_message}
                if uploaded_files:
                    message_dict["files"] = [f.name if hasattr(f, 'name') else f for f in uploaded_files]
                
                # Call chat function (returns response and rebuilt history from database)
                response_text, rebuilt_history = await self._chat_async(message_dict, [])
                
                # Return the final rebuilt history (which now includes both user message and response)
                yield rebuilt_history, "", None
            
            def send_message(user_message, uploaded_files, current_history):
                """Synchronous wrapper with generator support"""
                # Run the async generator and yield results
                async_gen = send_message_async(user_message, uploaded_files, current_history)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    while True:
                        try:
                            result = loop.run_until_complete(async_gen.__anext__())
                            yield result
                        except StopAsyncIteration:
                            break
                finally:
                    loop.close()
            
            # Connect send button and Enter key
            send_btn.click(
                fn=send_message,
                inputs=[msg, files, chatbot],
                outputs=[chatbot, msg, files],
            )
            msg.submit(
                fn=send_message,
                inputs=[msg, files, chatbot],
                outputs=[chatbot, msg, files],
            )
            
            # Connect new chat button
            def handle_new_chat():
                """Handle new chat button click - reset conversation and clear UI"""
                old_id = self.conversation_id
                new_id = str(uuid4())
                self.conversation_id = new_id
                self._conversation_message_count = 0
                logger.info(f"🆕 Started new conversation: {new_id} (cleared: {old_id})")
                return []  # Return empty history
            
            new_chat_btn.click(
                fn=handle_new_chat,
                inputs=None,
                outputs=chatbot,
            )

            with gr.Accordion("ℹ️ About", open=False):
                gr.Markdown(
                    f"""
                    ## PyRIT Endpoint Chat
                    
                    Chat naturally with AI endpoints through PyRIT. Send text messages and attach images, videos, or audio files inline.
                    
                    **Current Target:** `{target_name}`
                    
                    ### Features:
                    - 💬 **Natural Chat Flow**: Conversation history in a single pane
                    - 📎 **Multi-modal Input**: Attach images, videos, or audio to your messages
                    - 🔄 **Conversation History**: All turns visible in chat interface
                    - 🎯 **Multiple Targets**: OpenAI, Azure OpenAI, Azure ML, HuggingFace, and more
                    
                    ### Supported Targets:
                    - `OpenAIChatTarget`: OpenAI API, Azure OpenAI, Groq, OpenRouter, etc.
                    - `OpenAISoraTarget`: Video generation endpoints
                    - And any PyRIT PromptTarget
                    
                    ### Usage Tips:
                    - Type your message and press Enter or click Send
                    - Click the 📎 icon to attach images, videos, or audio
                    - Use 🔄 Retry to resend the last message
                    - Use ↩️ Undo to remove the last exchange
                    - Use 🗑️ Clear Chat to start a new conversation
                    """
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
