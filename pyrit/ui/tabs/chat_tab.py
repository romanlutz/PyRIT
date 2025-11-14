# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Chat tab UI for the PyRIT Gradio chat application.
"""

import asyncio
import inspect
import logging
from uuid import uuid4

import gradio as gr

from pyrit.ui.tabs.converters_helper import (
    get_available_converters,
    get_converter_parameters,
    create_converter_instance,
    get_parameter_ui_type,
    extract_literal_choices,
)

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

        # Main layout: Converter config on left (30%), Chat on right (70%)
        with gr.Row():
            # Left: Converter configuration panel
            with gr.Column(scale=3):
                gr.Markdown("### 🔄 Converter Settings")
                
                # Add converter button (visible by default)
                add_converter_btn = gr.Button("➕ Add Converter", size="sm", visible=True)
                
                # Converter 1
                with gr.Group(visible=False) as converter1_group:
                    with gr.Row():
                        with gr.Column(scale=10):
                            gr.Markdown("**Converter 1**")
                        with gr.Column(scale=1, min_width=30):
                            remove1_btn = gr.Button("❌", size="sm", visible=True)
                    converter1_dropdown = gr.Dropdown(
                        choices=["None"] + get_available_converters(),
                        value="None",
                        label="",
                        show_label=False,
                        container=False
                    )
                    # Fixed parameter input fields (show/hide based on converter)
                    conv1_param1 = gr.Textbox(label="param1", visible=False)
                    conv1_param1_bool = gr.Checkbox(label="param1", visible=False)
                    conv1_param1_dropdown = gr.Dropdown(label="param1", visible=False, choices=[], allow_custom_value=False)
                    conv1_param2 = gr.Textbox(label="param2", visible=False)
                    conv1_param2_bool = gr.Checkbox(label="param2", visible=False)
                    conv1_param2_dropdown = gr.Dropdown(label="param2", visible=False, choices=[], allow_custom_value=False)
                    conv1_param3 = gr.Textbox(label="param3", visible=False)
                    conv1_param3_bool = gr.Checkbox(label="param3", visible=False)
                    conv1_param3_dropdown = gr.Dropdown(label="param3", visible=False, choices=[], allow_custom_value=False)
                    conv1_param4 = gr.Textbox(label="param4", visible=False)
                    conv1_param4_bool = gr.Checkbox(label="param4", visible=False)
                    conv1_param4_dropdown = gr.Dropdown(label="param4", visible=False, choices=[], allow_custom_value=False)
                    # Apply button to confirm converter configuration
                    apply1_btn = gr.Button("✓ Confirm", size="sm", visible=False, variant="primary")
                    # Info about parameters
                    converter1_info = gr.Markdown("", visible=False)
                
                # Converter 2
                with gr.Group(visible=False) as converter2_group:
                    with gr.Row():
                        with gr.Column(scale=10):
                            gr.Markdown("**Converter 2**")
                        with gr.Column(scale=1, min_width=30):
                            remove2_btn = gr.Button("❌", size="sm", visible=True)
                    converter2_dropdown = gr.Dropdown(
                        choices=["None"] + get_available_converters(),
                        value="None",
                        label="",
                        show_label=False,
                        container=False
                    )
                    # Fixed parameter input fields
                    conv2_param1 = gr.Textbox(label="param1", visible=False)
                    conv2_param1_bool = gr.Checkbox(label="param1", visible=False)
                    conv2_param1_dropdown = gr.Dropdown(label="param1", visible=False, choices=[], allow_custom_value=False)
                    conv2_param2 = gr.Textbox(label="param2", visible=False)
                    conv2_param2_bool = gr.Checkbox(label="param2", visible=False)
                    conv2_param2_dropdown = gr.Dropdown(label="param2", visible=False, choices=[], allow_custom_value=False)
                    conv2_param3 = gr.Textbox(label="param3", visible=False)
                    conv2_param3_bool = gr.Checkbox(label="param3", visible=False)
                    conv2_param3_dropdown = gr.Dropdown(label="param3", visible=False, choices=[], allow_custom_value=False)
                    conv2_param4 = gr.Textbox(label="param4", visible=False)
                    conv2_param4_bool = gr.Checkbox(label="param4", visible=False)
                    conv2_param4_dropdown = gr.Dropdown(label="param4", visible=False, choices=[], allow_custom_value=False)
                    # Apply button to confirm converter configuration
                    apply2_btn = gr.Button("✓ Confirm", size="sm", visible=False, variant="primary")
                    # Info about parameters
                    converter2_info = gr.Markdown("", visible=False)
                
                # Converter 3
                with gr.Group(visible=False) as converter3_group:
                    with gr.Row():
                        with gr.Column(scale=10):
                            gr.Markdown("**Converter 3**")
                        with gr.Column(scale=1, min_width=30):
                            remove3_btn = gr.Button("❌", size="sm", visible=True)
                    converter3_dropdown = gr.Dropdown(
                        choices=["None"] + get_available_converters(),
                        value="None",
                        label="",
                        show_label=False,
                        container=False
                    )
                    # Fixed parameter input fields
                    conv3_param1 = gr.Textbox(label="param1", visible=False)
                    conv3_param1_bool = gr.Checkbox(label="param1", visible=False)
                    conv3_param1_dropdown = gr.Dropdown(label="param1", visible=False, choices=[], allow_custom_value=False)
                    conv3_param2 = gr.Textbox(label="param2", visible=False)
                    conv3_param2_bool = gr.Checkbox(label="param2", visible=False)
                    conv3_param2_dropdown = gr.Dropdown(label="param2", visible=False, choices=[], allow_custom_value=False)
                    conv3_param3 = gr.Textbox(label="param3", visible=False)
                    conv3_param3_bool = gr.Checkbox(label="param3", visible=False)
                    conv3_param3_dropdown = gr.Dropdown(label="param3", visible=False, choices=[], allow_custom_value=False)
                    conv3_param4 = gr.Textbox(label="param4", visible=False)
                    conv3_param4_bool = gr.Checkbox(label="param4", visible=False)
                    conv3_param4_dropdown = gr.Dropdown(label="param4", visible=False, choices=[], allow_custom_value=False)
                    # Apply button to confirm converter configuration
                    apply3_btn = gr.Button("✓ Confirm", size="sm", visible=False, variant="primary")
                    # Info about parameters
                    converter3_info = gr.Markdown("", visible=False)                # Status display
                converter_status = gr.Markdown("**Status:** No converter active")
            
            # Right: Chat interface
            with gr.Column(scale=7):
                # Render and make chatbot visible in this tab
                chatbot.render()
                chatbot.visible = True
                
                # Original input - combines text and file uploads
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    placeholder="Enter message or upload files...",
                    show_label=False,
                )
                
                # Converted text area (initially hidden)
                with gr.Group(visible=False) as converted_group:
                    gr.Markdown("**� Converted Text:**")
                    converted_textbox = gr.Textbox(
                        label="",
                        interactive=False,
                        lines=4,
                        max_lines=10,
                        show_label=False,
                    )
                    with gr.Row():
                        send_converted_btn = gr.Button("📤 Send Converted", variant="primary", size="sm")
                        delete_conversion_btn = gr.Button("❌ Clear Conversion", size="sm")
                
                # Convert button (visible when converter is active)
                convert_btn = gr.Button("🔄 Convert", visible=False, size="sm")
        
        
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
            """Handle retry by duplicating conversation without last turn and resending from memory"""
            
            # Get the last user message(s) from memory (not from gradio history)
            conversation = app_instance.memory.get_conversation(conversation_id=app_instance.conversation_id)
            
            # Find the last user message(s) in the conversation
            last_request_pieces = []
            
            # Iterate from the end to find the last request
            for response in reversed(conversation):
                # Check if this is a request (has user role pieces)
                request_pieces = [p for p in response.message_pieces if p.role == "user"]
                if request_pieces:
                    last_request_pieces = request_pieces
                    break
            
            if not last_request_pieces:
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
            
            # Build message dict from the last request pieces stored in memory
            # Use ORIGINAL values (not converted) because we'll re-apply converters
            text_parts = []
            files = []
            
            for piece in last_request_pieces:
                # Use original_value to get the user's actual input before conversion
                value = piece.original_value
                data_type = piece.original_value_data_type
                
                if data_type == "text":
                    text_parts.append(value)
                elif data_type in ["image_path", "video_path", "audio_path"]:
                    files.append(value)
            
            message_dict = {
                "text": " ".join(text_parts) if text_parts else "",
                "files": files
            }
            
            # Show the user message(s) immediately by reconstructing display from memory
            display_history = updated_history.copy()
            for piece in last_request_pieces:
                value = piece.original_value
                data_type = piece.original_value_data_type
                
                if data_type == "text":
                    display_history.append({"role": "user", "content": value})
                elif data_type in ["image_path", "video_path", "audio_path"]:
                    display_history.append({"role": "user", "content": {"path": value}})
            
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

        # ===== CONVERTER EVENT HANDLERS =====
        
        def update_converter_chain(conv1, conv2, conv3, group1_visible, group2_visible, group3_visible):
            """Update the converter chain based on all three dropdowns and current visibility"""
            # Don't create converters here - just track which ones are selected
            # Actual converter creation with parameters happens on Convert button click
            names = []
            
            # Track selected converters (only if their group is visible)
            if conv1 and conv1 != "None" and group1_visible:
                names.append(conv1)
            
            if conv2 and conv2 != "None" and group2_visible:
                names.append(conv2)
            
            if conv3 and conv3 != "None" and group3_visible:
                names.append(conv3)
            
            # Generate parameter info and field updates for each converter
            def get_param_updates(converter_name):
                """Returns (info_update, param1_text, param1_bool, param1_dropdown, param2_text, param2_bool, param2_dropdown, ...)"""
                if not converter_name or converter_name == "None":
                    return (
                        gr.update(value="", visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                    )
                
                params = get_converter_parameters(converter_name)
                # Filter to UI-representable params
                ui_params = [(name, info) for name, info in params.items() 
                            if get_parameter_ui_type(info) != 'skip']
                
                # Prepare parameter field updates (up to 4, each with textbox, checkbox, and dropdown)
                param_updates = []
                for i in range(4):
                    if i < len(ui_params):
                        param_name, param_info = ui_params[i]
                        default_val = param_info.get('default')
                        
                        # Determine parameter type
                        param_type = param_info['type']
                        ui_type = get_parameter_ui_type(param_info)
                        
                        label = f"{param_name}"
                        if param_info['required']:
                            label += " *"
                        label += f" ({param_info['type']})"
                        
                        if ui_type == 'checkbox':
                            # For bool parameters, show checkbox
                            if default_val is None or default_val == inspect.Parameter.empty:
                                bool_val = True  # Default to True for required bools
                            else:
                                bool_val = bool(default_val)
                            
                            param_updates.append(gr.update(visible=False))  # Hide textbox
                            param_updates.append(gr.update(
                                label=label,
                                value=bool_val,
                                visible=True,
                                interactive=True
                            ))  # Show checkbox
                            param_updates.append(gr.update(visible=False))  # Hide dropdown
                        elif ui_type == 'dropdown':
                            # For Literal parameters, show dropdown
                            choices = extract_literal_choices(param_type)
                            logger.info(f"Dropdown for {param_name}: type={param_type}, choices={choices}")
                            if not choices:
                                logger.warning(f"No choices extracted for Literal parameter {param_name} with type {param_type}")
                            if default_val is None or default_val == inspect.Parameter.empty:
                                dropdown_val = choices[0] if choices else None
                            else:
                                dropdown_val = str(default_val)
                            
                            dropdown_update = gr.update(
                                label=label,
                                choices=choices,
                                value=dropdown_val,
                                visible=True,
                                interactive=True
                            )
                            logger.info(f"Dropdown update dict for {param_name}: {dropdown_update}")
                            
                            param_updates.append(gr.update(visible=False))  # Hide textbox
                            param_updates.append(gr.update(visible=False))  # Hide checkbox
                            param_updates.append(dropdown_update)  # Show dropdown
                        else:
                            # For other parameters (textbox, number), show textbox
                            if default_val is None or default_val == inspect.Parameter.empty:
                                display_val = ''
                            else:
                                display_val = str(default_val)
                            
                            param_updates.append(gr.update(
                                label=label,
                                value=display_val,
                                visible=True,
                                interactive=True
                            ))  # Show textbox
                            param_updates.append(gr.update(visible=False))  # Hide checkbox
                            param_updates.append(gr.update(visible=False))  # Hide dropdown
                    else:
                        # Hide all three components for unused slots
                        param_updates.append(gr.update(visible=False))  # Textbox
                        param_updates.append(gr.update(visible=False))  # Checkbox
                        param_updates.append(gr.update(visible=False))  # Dropdown
                
                # Build info message
                required_params = [p for p in params.values() if p['required']]
                if not required_params:
                    info = gr.update(value="✅ No required parameters", visible=True)
                else:
                    param_list = [f"• `{p['name']}` ({p['type']})" for p in required_params]
                    info_text = "⚠️ **Required parameters:**\n" + "\n".join(param_list)
                    info_text += "\n\n*Note: Some converters may fail without proper configuration*"
                    info = gr.update(value=info_text, visible=True)
                
                return (info, param_updates[0], param_updates[1], param_updates[2], 
                        param_updates[3], param_updates[4], param_updates[5],
                        param_updates[6], param_updates[7], param_updates[8],
                        param_updates[9], param_updates[10], param_updates[11])
            
            info1, p1_1, p1_1b, p1_1d, p1_2, p1_2b, p1_2d, p1_3, p1_3b, p1_3d, p1_4, p1_4b, p1_4d = get_param_updates(conv1)
            info2, p2_1, p2_1b, p2_1d, p2_2, p2_2b, p2_2d, p2_3, p2_3b, p2_3d, p2_4, p2_4b, p2_4d = get_param_updates(conv2)
            info3, p3_1, p3_1b, p3_1d, p3_2, p3_2b, p3_2d, p3_3, p3_3b, p3_3d, p3_4, p3_4b, p3_4d = get_param_updates(conv3)
            
            # Update UI visibility and status
            count = len(names)
            
            # Show "Add Converter" button if:
            # - We haven't reached the max (3)
            # - The next slot isn't already visible
            show_add_btn = False
            if not group1_visible:
                show_add_btn = True
            elif not group2_visible:
                show_add_btn = True
            elif not group3_visible:
                show_add_btn = True
            
            # Show remove buttons only if converter is selected
            show_remove1 = conv1 and conv1 != "None"
            show_remove2 = conv2 and conv2 != "None"
            show_remove3 = conv3 and conv3 != "None"
            
            # Show Apply buttons when converter is selected
            show_apply1 = conv1 and conv1 != "None"
            show_apply2 = conv2 and conv2 != "None"
            show_apply3 = conv3 and conv3 != "None"
            
            # Status message
            if count == 0:
                status = "**Status:** No converter selected"
            else:
                chain = " → ".join(names)
                status = f"**Status:** {count} converter(s) selected: {chain}\n\n💡 Click 'Apply Converter' to confirm settings"
            
            # Show convert button only if at least one converter has been applied
            # (check app_instance.prompt_converters which is populated by Apply buttons)
            show_convert = len(app_instance.prompt_converters) > 0
            
            return (
                status,
                gr.update(visible=show_add_btn),
                gr.update(visible=show_remove1),
                gr.update(visible=show_remove2),
                gr.update(visible=show_remove3),
                gr.update(visible=show_convert),
                gr.update(visible=False),  # Hide converted group when chain changes
                "",  # Clear converted text
                info1,  # Converter 1 parameter info
                info2,  # Converter 2 parameter info
                info3,  # Converter 3 parameter info
                p1_1, p1_1b, p1_1d, p1_2, p1_2b, p1_2d, p1_3, p1_3b, p1_3d, p1_4, p1_4b, p1_4d,  # Conv1 params (text+bool+dropdown triples)
                p2_1, p2_1b, p2_1d, p2_2, p2_2b, p2_2d, p2_3, p2_3b, p2_3d, p2_4, p2_4b, p2_4d,  # Conv2 params
                p3_1, p3_1b, p3_1d, p3_2, p3_2b, p3_2d, p3_3, p3_3b, p3_3d, p3_4, p3_4b, p3_4d,  # Conv3 params
                gr.update(visible=show_apply1),  # Apply button 1
                gr.update(visible=show_apply2),  # Apply button 2
                gr.update(visible=show_apply3),  # Apply button 3
            )
        
        # Track visibility state of all groups
        group1_visible_state = gr.State(False)
        group2_visible_state = gr.State(False)
        group3_visible_state = gr.State(False)
        
        # Connect all converter dropdowns to update function
        def handle_converter1_change(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis):
            return update_converter_chain(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis)
        
        def handle_converter2_change(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis):
            return update_converter_chain(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis)
        
        def handle_converter3_change(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis):
            return update_converter_chain(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis)
        
        converter1_dropdown.change(
            fn=handle_converter1_change,
            inputs=[converter1_dropdown, converter2_dropdown, converter3_dropdown, 
                   group1_visible_state, group2_visible_state, group3_visible_state],
            outputs=[converter_status, add_converter_btn, 
                    remove1_btn, remove2_btn, remove3_btn, convert_btn, converted_group, converted_textbox,
                    converter1_info, converter2_info, converter3_info,
                    conv1_param1, conv1_param1_bool, conv1_param1_dropdown,
                    conv1_param2, conv1_param2_bool, conv1_param2_dropdown,
                    conv1_param3, conv1_param3_bool, conv1_param3_dropdown,
                    conv1_param4, conv1_param4_bool, conv1_param4_dropdown,
                    conv2_param1, conv2_param1_bool, conv2_param1_dropdown,
                    conv2_param2, conv2_param2_bool, conv2_param2_dropdown,
                    conv2_param3, conv2_param3_bool, conv2_param3_dropdown,
                    conv2_param4, conv2_param4_bool, conv2_param4_dropdown,
                    conv3_param1, conv3_param1_bool, conv3_param1_dropdown,
                    conv3_param2, conv3_param2_bool, conv3_param2_dropdown,
                    conv3_param3, conv3_param3_bool, conv3_param3_dropdown,
                    conv3_param4, conv3_param4_bool, conv3_param4_dropdown,
                    apply1_btn, apply2_btn, apply3_btn],
        )
        
        converter2_dropdown.change(
            fn=handle_converter2_change,
            inputs=[converter1_dropdown, converter2_dropdown, converter3_dropdown, 
                   group1_visible_state, group2_visible_state, group3_visible_state],
            outputs=[converter_status, add_converter_btn,
                    remove1_btn, remove2_btn, remove3_btn, convert_btn, converted_group, converted_textbox,
                    converter1_info, converter2_info, converter3_info,
                    conv1_param1, conv1_param1_bool, conv1_param1_dropdown,
                    conv1_param2, conv1_param2_bool, conv1_param2_dropdown,
                    conv1_param3, conv1_param3_bool, conv1_param3_dropdown,
                    conv1_param4, conv1_param4_bool, conv1_param4_dropdown,
                    conv2_param1, conv2_param1_bool, conv2_param1_dropdown,
                    conv2_param2, conv2_param2_bool, conv2_param2_dropdown,
                    conv2_param3, conv2_param3_bool, conv2_param3_dropdown,
                    conv2_param4, conv2_param4_bool, conv2_param4_dropdown,
                    conv3_param1, conv3_param1_bool, conv3_param1_dropdown,
                    conv3_param2, conv3_param2_bool, conv3_param2_dropdown,
                    conv3_param3, conv3_param3_bool, conv3_param3_dropdown,
                    conv3_param4, conv3_param4_bool, conv3_param4_dropdown,
                    apply1_btn, apply2_btn, apply3_btn],
        )
        
        converter3_dropdown.change(
            fn=handle_converter3_change,
            inputs=[converter1_dropdown, converter2_dropdown, converter3_dropdown, 
                   group1_visible_state, group2_visible_state, group3_visible_state],
            outputs=[converter_status, add_converter_btn,
                    remove1_btn, remove2_btn, remove3_btn, convert_btn, converted_group, converted_textbox,
                    converter1_info, converter2_info, converter3_info,
                    conv1_param1, conv1_param1_bool, conv1_param1_dropdown,
                    conv1_param2, conv1_param2_bool, conv1_param2_dropdown,
                    conv1_param3, conv1_param3_bool, conv1_param3_dropdown,
                    conv1_param4, conv1_param4_bool, conv1_param4_dropdown,
                    conv2_param1, conv2_param1_bool, conv2_param1_dropdown,
                    conv2_param2, conv2_param2_bool, conv2_param2_dropdown,
                    conv2_param3, conv2_param3_bool, conv2_param3_dropdown,
                    conv2_param4, conv2_param4_bool, conv2_param4_dropdown,
                    conv3_param1, conv3_param1_bool, conv3_param1_dropdown,
                    conv3_param2, conv3_param2_bool, conv3_param2_dropdown,
                    conv3_param3, conv3_param3_bool, conv3_param3_dropdown,
                    conv3_param4, conv3_param4_bool, conv3_param4_dropdown,
                    apply1_btn, apply2_btn, apply3_btn],
        )
        
        # Apply button handlers - create converter instances with parameters
        # We'll use a simple list to track which slot each converter is in
        converter_slot1 = [None]  # Mutable container to store converter
        converter_slot2 = [None]
        converter_slot3 = [None]
        
        def apply_converter(converter_name, p1, p1b, p1d, p2, p2b, p2d, p3, p3b, p3d, p4, p4b, p4d, slot):
            """Create a converter instance with the given parameters and store in slot"""
            if not converter_name or converter_name == "None":
                slot[0] = None
                return "No converter selected"
            
            # Get the adversarial chat target for converters that need it
            # This is separate from the target being tested/attacked
            default_target = app_instance.adversarial_chat
            
            params_info = get_converter_parameters(converter_name)
            ui_params = [(name, info) for name, info in params_info.items() 
                        if get_parameter_ui_type(info) != 'skip']
            
            # Build params dict from textbox, checkbox, and dropdown values
            params = {}
            param_values = [(p1, p1b, p1d), (p2, p2b, p2d), (p3, p3b, p3d), (p4, p4b, p4d)]
            for i, (param_name, param_info) in enumerate(ui_params):
                if i < len(param_values):
                    text_val, bool_val, dropdown_val = param_values[i]
                    ui_type = get_parameter_ui_type(param_info)
                    param_type = param_info['type'].lower()
                    
                    # Determine which value to use based on parameter UI type
                    if ui_type == 'checkbox':
                        # Use checkbox value for bool parameters
                        params[param_name] = bool_val
                    elif ui_type == 'dropdown':
                        # Use dropdown value for Literal parameters
                        if dropdown_val:
                            params[param_name] = dropdown_val
                    elif text_val:
                        # Use textbox value for other types
                        try:
                            if 'int' in param_type:
                                params[param_name] = int(text_val)
                            elif 'float' in param_type:
                                params[param_name] = float(text_val)
                            else:
                                params[param_name] = text_val
                        except ValueError as e:
                            slot[0] = None
                            return f"❌ Error: Could not convert {param_name}={text_val} to {param_type}: {e}"
            
            # Try to create the converter
            inst = create_converter_instance(converter_name, params, default_target=default_target)
            if inst:
                slot[0] = inst
                # Update app_instance.prompt_converters with the full chain
                app_instance.prompt_converters = [c for c in [converter_slot1[0], converter_slot2[0], converter_slot3[0]] if c is not None]
                return f"✅ {converter_name} applied successfully!"
            else:
                slot[0] = None
                return f"❌ Failed to create {converter_name}. Check parameters and logs."
        
        def handle_apply1(conv1, p1, p1b, p1d, p2, p2b, p2d, p3, p3b, p3d, p4, p4b, p4d):
            result = apply_converter(conv1, p1, p1b, p1d, p2, p2b, p2d, p3, p3b, p3d, p4, p4b, p4d, converter_slot1)
            # Show convert button if we now have converters
            show_convert = len(app_instance.prompt_converters) > 0
            return gr.update(value=result, visible=True), gr.update(visible=show_convert)
        
        def handle_apply2(conv2, p1, p1b, p1d, p2, p2b, p2d, p3, p3b, p3d, p4, p4b, p4d):
            result = apply_converter(conv2, p1, p1b, p1d, p2, p2b, p2d, p3, p3b, p3d, p4, p4b, p4d, converter_slot2)
            show_convert = len(app_instance.prompt_converters) > 0
            return gr.update(value=result, visible=True), gr.update(visible=show_convert)
        
        def handle_apply3(conv3, p1, p1b, p1d, p2, p2b, p2d, p3, p3b, p3d, p4, p4b, p4d):
            result = apply_converter(conv3, p1, p1b, p1d, p2, p2b, p2d, p3, p3b, p3d, p4, p4b, p4d, converter_slot3)
            show_convert = len(app_instance.prompt_converters) > 0
            return gr.update(value=result, visible=True), gr.update(visible=show_convert)
        
        apply1_btn.click(
            fn=handle_apply1,
            inputs=[converter1_dropdown, 
                    conv1_param1, conv1_param1_bool, conv1_param1_dropdown,
                    conv1_param2, conv1_param2_bool, conv1_param2_dropdown,
                    conv1_param3, conv1_param3_bool, conv1_param3_dropdown,
                    conv1_param4, conv1_param4_bool, conv1_param4_dropdown],
            outputs=[converter1_info, convert_btn],
        )
        
        apply2_btn.click(
            fn=handle_apply2,
            inputs=[converter2_dropdown, 
                    conv2_param1, conv2_param1_bool, conv2_param1_dropdown,
                    conv2_param2, conv2_param2_bool, conv2_param2_dropdown,
                    conv2_param3, conv2_param3_bool, conv2_param3_dropdown,
                    conv2_param4, conv2_param4_bool, conv2_param4_dropdown],
            outputs=[converter2_info, convert_btn],
        )
        
        apply3_btn.click(
            fn=handle_apply3,
            inputs=[converter3_dropdown, 
                    conv3_param1, conv3_param1_bool, conv3_param1_dropdown,
                    conv3_param2, conv3_param2_bool, conv3_param2_dropdown,
                    conv3_param3, conv3_param3_bool, conv3_param3_dropdown,
                    conv3_param4, conv3_param4_bool, conv3_param4_dropdown],
            outputs=[converter3_info, convert_btn],
        )
        
        # Add converter button - reveals next slot
        def handle_add_converter(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis):
            """Show the next converter slot"""
            # If group 1 not visible, show it
            if not g1_vis:
                new_g1_vis = True
                new_g2_vis = g2_vis
                new_g3_vis = g3_vis
            # If group 1 visible but not group 2, show group 2
            elif not g2_vis:
                new_g1_vis = g1_vis
                new_g2_vis = True
                new_g3_vis = g3_vis
            # If group 2 visible but not group 3, show group 3
            elif not g3_vis:
                new_g1_vis = g1_vis
                new_g2_vis = g2_vis
                new_g3_vis = True
            else:
                # All visible already
                new_g1_vis = g1_vis
                new_g2_vis = g2_vis
                new_g3_vis = g3_vis
            
            # Update converter chain with new visibility
            result = update_converter_chain(conv1, conv2, conv3, new_g1_vis, new_g2_vis, new_g3_vis)
            
            return (
                gr.update(visible=new_g1_vis),
                gr.update(visible=new_g2_vis),
                gr.update(visible=new_g3_vis),
                new_g1_vis,
                new_g2_vis,
                new_g3_vis,
                *result
            )
        
        add_converter_btn.click(
            fn=handle_add_converter,
            inputs=[converter1_dropdown, converter2_dropdown, converter3_dropdown, 
                   group1_visible_state, group2_visible_state, group3_visible_state],
            outputs=[converter1_group, converter2_group, converter3_group, 
                    group1_visible_state, group2_visible_state, group3_visible_state,
                    converter_status, add_converter_btn,
                    remove1_btn, remove2_btn, remove3_btn, convert_btn, converted_group, converted_textbox,
                    converter1_info, converter2_info, converter3_info],
        )
        
        # Remove buttons - hide the converter group and shift remaining converters up
        def handle_remove1(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis):
            """Remove converter 1, shift others up"""
            # Shift converters: conv2 -> conv1, conv3 -> conv2, clear conv3
            new_conv1 = conv2 if (conv2 and conv2 != "None") else "None"
            new_conv2 = conv3 if (conv3 and conv3 != "None") else "None"
            new_conv3 = "None"
            
            # Calculate new visibility based on whether we have converters
            new_g1_vis = new_conv1 != "None"
            new_g2_vis = new_conv2 != "None"
            new_g3_vis = False
            
            # Update converter chain
            result = update_converter_chain(new_conv1, new_conv2, new_conv3, new_g1_vis, new_g2_vis, new_g3_vis)
            
            return (
                new_conv1, new_conv2, new_conv3,
                new_g1_vis, new_g2_vis, new_g3_vis,
                gr.update(visible=new_g1_vis),
                gr.update(visible=new_g2_vis),
                gr.update(visible=new_g3_vis),
                *result
            )
        
        def handle_remove2(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis):
            """Remove converter 2, shift conv3 up"""
            # Shift: conv3 -> conv2, clear conv3
            new_conv2 = conv3 if (conv3 and conv3 != "None") else "None"
            new_conv3 = "None"
            
            # Group 1 stays as is, group 2 visible if it has a converter, group 3 hidden
            new_g2_vis = new_conv2 != "None"
            new_g3_vis = False
            
            result = update_converter_chain(conv1, new_conv2, new_conv3, g1_vis, new_g2_vis, new_g3_vis)
            
            return (
                conv1, new_conv2, new_conv3,
                g1_vis, new_g2_vis, new_g3_vis,
                gr.update(visible=g1_vis),
                gr.update(visible=new_g2_vis),
                gr.update(visible=new_g3_vis),
                *result
            )
        
        def handle_remove3(conv1, conv2, conv3, g1_vis, g2_vis, g3_vis):
            """Remove converter 3"""
            new_conv3 = "None"
            new_g3_vis = False
            
            result = update_converter_chain(conv1, conv2, new_conv3, g1_vis, g2_vis, new_g3_vis)
            
            return (
                conv1, conv2, new_conv3,
                g1_vis, g2_vis, new_g3_vis,
                gr.update(visible=g1_vis),
                gr.update(visible=g2_vis),
                gr.update(visible=new_g3_vis),
                *result
            )
        
        # Wire up all remove buttons with the same inputs and outputs
        remove_outputs = [
            converter1_dropdown, converter2_dropdown, converter3_dropdown,
            group1_visible_state, group2_visible_state, group3_visible_state,
            converter1_group, converter2_group, converter3_group,
            converter_status, add_converter_btn,
            remove1_btn, remove2_btn, remove3_btn, convert_btn, converted_group, converted_textbox,
            converter1_info, converter2_info, converter3_info
        ]
        
        remove_inputs = [
            converter1_dropdown, converter2_dropdown, converter3_dropdown,
            group1_visible_state, group2_visible_state, group3_visible_state
        ]
        
        remove1_btn.click(fn=handle_remove1, inputs=remove_inputs, outputs=remove_outputs)
        remove2_btn.click(fn=handle_remove2, inputs=remove_inputs, outputs=remove_outputs)
        remove3_btn.click(fn=handle_remove3, inputs=remove_inputs, outputs=remove_outputs)
        
        async def handle_convert(message):
            """Convert the text by applying all converters in the chain"""
            if not app_instance.prompt_converters:
                return gr.update(visible=False), gr.update(visible=False), ""
            
            text_content = message.get("text", "") if isinstance(message, dict) else ""
            if not text_content:
                return gr.update(visible=False), gr.update(visible=False), ""
            
            try:
                # Apply converters in sequence
                result_text = text_content
                for converter in app_instance.prompt_converters:
                    result = await converter.convert_async(prompt=result_text, input_type="text")
                    result_text = result.output_text
                
                # Show converted group, hide convert button
                return gr.update(visible=True), gr.update(visible=False), result_text
            except Exception as e:
                logger.error(f"Conversion error: {e}")
                return gr.update(visible=False), gr.update(visible=False), f"Error: {str(e)}"
        
        def convert_sync(message):
            """Synchronous wrapper for convert"""
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(handle_convert(message))
                return result
            finally:
                loop.close()
        
        convert_btn.click(
            fn=convert_sync,
            inputs=[chat_input],
            outputs=[converted_group, convert_btn, converted_textbox],
        )
        
        def handle_delete_conversion():
            """Clear the converted text and hide the group, show convert button again"""
            show_convert = len(app_instance.prompt_converters) > 0
            return gr.update(visible=False), "", gr.update(visible=show_convert)
        
        delete_conversion_btn.click(
            fn=handle_delete_conversion,
            inputs=None,
            outputs=[converted_group, converted_textbox, convert_btn],
        )
        
        # Handle sending converted text
        async def send_converted_async(original_message, converted_text, current_history):
            """Send the converted text instead of original"""
            # Create a modified message with the converted text
            modified_message = {
                "text": converted_text,
                "files": original_message.get("files", []) if isinstance(original_message, dict) else []
            }
            
            # Add user message to history immediately
            updated_history = current_history.copy()
            
            # Add files first
            for file_path in modified_message.get("files", []):
                updated_history.append({"role": "user", "content": {"path": file_path}})
            
            # Add converted text with label showing both original and converted
            original_text = original_message.get("text", "") if isinstance(original_message, dict) else ""
            display_text = f"**Original:** {original_text}\n\n**Converted:** {converted_text}"
            updated_history.append({"role": "user", "content": display_text})
            
            # Determine if convert button should be visible
            show_convert = len(app_instance.prompt_converters) > 0
            
            # Yield updated history, hide converted group, show convert button again
            yield updated_history, gr.MultimodalTextbox(value=None, interactive=False), gr.update(visible=False), "", gr.update(visible=show_convert)
            
            # Add loading indicator
            thinking_history = updated_history.copy()
            thinking_history.append({"role": "assistant", "content": "..."})
            yield thinking_history, gr.MultimodalTextbox(value=None, interactive=False), gr.update(visible=False), "", gr.update(visible=show_convert)
            
            # Send - but we need to send with the ORIGINAL text so it gets stored correctly in memory
            # The converter will be applied by _chat_async
            response_text, rebuilt_history = await app_instance._chat_async(original_message, [])
            
            # Return final history with convert button visible again
            yield rebuilt_history, gr.MultimodalTextbox(value=None, interactive=True), gr.update(visible=False), "", gr.update(visible=show_convert)
        
        def send_converted(original_message, converted_text, current_history):
            """Synchronous wrapper for sending converted text"""
            async_gen = send_converted_async(original_message, converted_text, current_history)
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
        
        send_converted_btn.click(
            fn=send_converted,
            inputs=[chat_input, converted_textbox, chatbot],
            outputs=[chatbot, chat_input, converted_group, converted_textbox, convert_btn],
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
