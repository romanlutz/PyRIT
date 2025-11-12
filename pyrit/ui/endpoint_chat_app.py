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
        
        with gr.Blocks(title="PyRIT Endpoint Chat", theme=gr.themes.Default()) as demo:
            gr.Markdown("# PyRIT Endpoint Chat 🤖")
            
            # Navigation tabs - store reference to control selected tab
            with gr.Tabs(selected="chat_tab") as tabs:
                # ===== CHAT PAGE =====
                with gr.Tab("💬 Chat", id="chat_tab") as chat_tab:
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
                    
                    # Multimodal input - combines text and file uploads
                    chat_input = gr.MultimodalTextbox(
                        interactive=True,
                        file_count="multiple",
                        placeholder="Enter message or upload files...",
                        show_label=False,
                    )
                    
                    # Handle send message - show user message immediately, then get response
                    async def send_message_async(message, current_history):
                        """Send a message and update UI immediately"""
                        # Check if there's any content
                        text_content = message.get("text", "")
                        files = message.get("files", [])
                        
                        if not text_content and not files:
                            yield current_history, gr.MultimodalTextbox(value=None, interactive=True)
                            return
                        
                        # Add user message(s) to history immediately for instant feedback
                        updated_history = current_history.copy()
                        
                        # Add files first
                        for file_path in files:
                            updated_history.append({"role": "user", "content": {"path": file_path}})
                        
                        # Add text if present
                        if text_content:
                            updated_history.append({"role": "user", "content": text_content})
                        
                        # Yield the updated history with user message (clears input too)
                        yield updated_history, gr.MultimodalTextbox(value=None, interactive=False)
                        
                        # Call chat function (returns response and rebuilt history from database)
                        response_text, rebuilt_history = await self._chat_async(message, [])
                        
                        # Return the final rebuilt history and re-enable input
                        yield rebuilt_history, gr.MultimodalTextbox(value=None, interactive=True)
                    
                    def send_message(message, current_history):
                        """Synchronous wrapper with generator support"""
                        # Run the async generator and yield results
                        async_gen = send_message_async(message, current_history)
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
                    
                    # Connect input submit
                    chat_input.submit(
                        fn=send_message,
                        inputs=[chat_input, chatbot],
                        outputs=[chatbot, chat_input],
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
                            - 📎 **Multi-modal Input**: Type text and attach images, videos, or audio in one input box
                            - 🔄 **Conversation History**: All turns visible in chat interface
                            - 🎯 **Multiple Targets**: OpenAI, Azure OpenAI, Azure ML, HuggingFace, and more
                            
                            ### Supported Targets:
                            - `OpenAIChatTarget`: OpenAI API, Azure OpenAI, Groq, OpenRouter, etc.
                            - `OpenAISoraTarget`: Video generation endpoints
                            - And any PyRIT PromptTarget
                            
                            ### Usage Tips:
                            - Type your message and press Enter to send
                            - Click the 📎 icon to attach images, videos, or audio files
                            - Use 🆕 New Chat to start a fresh conversation
                            - Visit the � Conversations tab to browse and load previous chats
                            """
                        )
        
                # ===== CONVERSATIONS TABLE PAGE =====
                with gr.Tab("📋 Conversations", id="conversations_tab"):
                    gr.Markdown("## All Conversations in Memory")
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    # Conversations table
                    conversations_table = gr.Dataframe(
                        headers=["Conversation ID", "Message Count", "First User Prompt", "Labels", "Metadata", "First Message", "Last Message"],
                        datatype=["str", "number", "str", "str", "str", "str", "str"],
                        interactive=False,
                        wrap=True,
                        value=self._get_all_conversations_table(),
                    )
                    
                    # Handle table row selection - load conversation and switch to chat
                    def handle_table_click(evt: gr.SelectData):
                        """Handle clicking on a table row - loads conversation and switches to Chat tab"""
                        if evt.index is not None and len(evt.index) >= 1:
                            row_index = evt.index[0]
                            
                            # Get current table data
                            current_data = self._get_all_conversations_table()
                            if row_index < len(current_data):
                                conv_id = current_data[row_index][0]  # First column is conversation ID
                                
                                # Set the app's conversation ID to the selected one
                                self.conversation_id = conv_id.strip()
                                
                                # Rebuild history from database
                                history = self._rebuild_history_from_database()
                                
                                logger.info(f"📖 Loaded conversation {conv_id} from table click - switching to Chat tab")
                                
                                # Return: history and switch to Chat tab
                                return history, gr.Tabs(selected="chat_tab")
                        
                        return [], gr.Tabs(selected="conversations_tab")
                    
                    conversations_table.select(
                        fn=handle_table_click,
                        inputs=None,
                        outputs=[chatbot, tabs],  # Update chatbot and switch tabs
                    )
                    
                    # Handle refresh button
                    def handle_refresh():
                        """Refresh the conversations table"""
                        return self._get_all_conversations_table()
                    
                    refresh_btn.click(
                        fn=handle_refresh,
                        inputs=None,
                        outputs=conversations_table,
                    )
                    
                    gr.Markdown("""
                    ### Instructions:
                    - **Click on any row** in the table to load that conversation and switch to the Chat tab
                    - Use the **🔄 Refresh** button to update the table with latest conversations
                    
                    💡 **Tip**: Clicking a row automatically loads it and switches to the chat view!
                    """)

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
