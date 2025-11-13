# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Chat tab UI for the PyRIT Gradio chat application.
"""

import asyncio
import logging
from uuid import uuid4

import gradio as gr

logger = logging.getLogger(__name__)


def build_chat_tab(app_instance, tabs, target_name: str, chatbot):
    """
    Build the chat tab UI with all event handlers.
    
    Args:
        app_instance: The EndpointChatApp instance for accessing methods and state
        tabs: The Tabs component for navigation
        target_name: Current target name for display
        chatbot: The Chatbot component (pre-created, will be made visible)
    """
    with gr.Tab("💬 Chat", id="chat_tab") as chat_tab:
        with gr.Row():
            new_chat_btn = gr.Button("➕ New Chat", size="sm")

        # Render and make chatbot visible in this tab
        chatbot.render()
        chatbot.visible = True
        
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
            
            # Add loading indicator
            thinking_history = updated_history.copy()
            thinking_history.append({"role": "assistant", "content": "..."})
            yield thinking_history, gr.MultimodalTextbox(value=None, interactive=False)
            
            # Call chat function (returns response and rebuilt history from database)
            response_text, rebuilt_history = await app_instance._chat_async(message, [])
            
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
        
        # Handle retry - duplicate conversation excluding last turn and resend
        async def handle_retry_async(current_history):
            """Handle retry by duplicating conversation without last turn and resending"""
            # Get the current conversation history BEFORE duplication to find the last user message(s)
            original_history = app_instance._rebuild_history_from_database()
            
            # Find all consecutive user messages (skipping past any assistant messages at the end)
            # This handles multimodal messages that appear as multiple user entries
            last_user_messages = []
            found_assistant = False
            
            for msg in reversed(original_history):
                if msg["role"] == "assistant":
                    if found_assistant and last_user_messages:
                        # We've already collected user messages and hit another assistant - stop here
                        break
                    # Skip assistant messages at the end
                    found_assistant = True
                elif msg["role"] == "user":
                    if found_assistant:
                        # We've found user messages after skipping assistant messages
                        last_user_messages.insert(0, msg)  # Insert at beginning to maintain order
                    else:
                        # Still at the end, no assistant message yet - shouldn't happen but handle it
                        last_user_messages.insert(0, msg)
                else:
                    # Hit a non-user, non-assistant message (like system) - stop
                    if last_user_messages:
                        break
            
            if not last_user_messages:
                # No user message found, just return current history
                logger.warning("🔄 Retry failed - no user message found in conversation")
                yield current_history, gr.MultimodalTextbox(value=None, interactive=True)
                return
            
            # Duplicate conversation excluding the last turn (removes last user message(s) and assistant response)
            new_conv_id = app_instance.memory.duplicate_conversation_excluding_last_turn(
                conversation_id=app_instance.conversation_id
            )
            
            # Update to the new conversation ID
            old_id = app_instance.conversation_id
            app_instance.conversation_id = new_conv_id
            
            logger.info(f"🔄 Retrying - duplicated conversation {old_id} -> {new_conv_id} (excluding last turn)")
            
            # Get the history from the new conversation (without the last turn)
            updated_history = app_instance._rebuild_history_from_database()
            
            # Build message dict from the last user message(s) we saved
            # Collect text and files from all the user messages
            text_parts = []
            files = []
            
            for msg in last_user_messages:
                if isinstance(msg["content"], str):
                    text_parts.append(msg["content"])
                elif isinstance(msg["content"], dict) and "path" in msg["content"]:
                    # It's a file
                    files.append(msg["content"]["path"])
                else:
                    # Fallback - convert to string
                    text_parts.append(str(msg["content"]))
            
            message_dict = {
                "text": " ".join(text_parts) if text_parts else "",
                "files": files
            }
            
            # Show the user message(s) immediately by adding them to the history
            display_history = updated_history.copy()
            for msg in last_user_messages:
                display_history.append(msg)
            
            # Yield to show user message(s) immediately
            yield display_history, gr.MultimodalTextbox(value=None, interactive=False)
            
            # Add loading indicator
            thinking_history = display_history.copy()
            thinking_history.append({"role": "assistant", "content": "..."})
            yield thinking_history, gr.MultimodalTextbox(value=None, interactive=False)
            
            # Resend the message
            response_text, rebuilt_history = await app_instance._chat_async(message_dict, [])
            
            # Return final history with new response
            yield rebuilt_history, gr.MultimodalTextbox(value=None, interactive=True)
        
        def handle_retry(current_history):
            """Synchronous wrapper for retry"""
            async_gen = handle_retry_async(current_history)
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
        
        # Connect retry event
        chatbot.retry(
            fn=handle_retry,
            inputs=[chatbot],
            outputs=[chatbot, chat_input],
        )
        
        # Connect new chat button
        def handle_new_chat():
            """Handle new chat button click - reset conversation and clear UI"""
            old_id = app_instance.conversation_id
            new_id = str(uuid4())
            app_instance.conversation_id = new_id
            app_instance._conversation_message_count = 0
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
                ## Co-PyRIT
                
                Chat naturally with AI endpoints through PyRIT. Send text messages and attach images, videos, or audio files inline.
                
                **Current Target:** `{target_name}`
                
                ### Features:
                - 💬 **Natural Chat Flow**: Conversation history in a single pane
                - 📎 **Multi-modal Input**: Type text and attach images, videos, or audio in one input box
                - 🔄 **Conversation History**: All turns visible in chat interface
                - 🎯 **Dynamic Target Selection**: Switch between any PyRIT target via the ⚙️ Configuration tab
                - ⚙️ **Runtime Configuration**: Change targets without restarting the app
                
                ### Supported Targets:
                All targets from `pyrit.prompt_target` are supported, including:
                - `OpenAIChatTarget`, `OpenAIDALLETarget`, `OpenAISoraTarget`, `OpenAITTSTarget`
                - `AzureMLChatTarget`, `HuggingFaceChatTarget`, `HuggingFaceEndpointTarget`
                - `HTTPTarget`, `HTTPXAPITarget`, `PlaywrightTarget`, `TextTarget`
                - And many more!
                
                ### Usage Tips:
                - Type your message and press Enter to send
                - Click the 📎 icon to attach images, videos, or audio files
                - Use ➕ New Chat to start a fresh conversation
                - Visit the 📋 Conversations tab to browse and load previous chats
                - Visit the ⚙️ Configuration tab to switch targets dynamically
                """
            )
