# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Conversations tab UI for the PyRIT Gradio chat application.
"""

import logging

import gradio as gr

logger = logging.getLogger(__name__)


def build_conversations_tab(app_instance, tabs, chatbot):
    """
    Build the conversations tab UI with table and event handlers.
    
    Args:
        app_instance: The EndpointChatApp instance for accessing methods and state
        tabs: The Tabs component for navigation
        chatbot: The Chatbot component to update when loading conversations
        
    Returns:
        conversations_tab: The Tab component (needed for tab selection events)
        conversations_table: The table component (needed for auto-refresh)
    """
    with gr.Tab("📋 Conversations", id="conversations_tab") as conversations_tab_component:
        gr.Markdown("## All Conversations in Memory")
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
        
        # Conversations table
        conversations_table = gr.Dataframe(
            headers=["Conversation ID", "Message Count", "First User Prompt", "Labels", "Metadata", "First Message", "Last Message"],
            datatype=["str", "number", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            value=app_instance._get_all_conversations_table(),
        )
        
        # Handle table row selection - load conversation and switch to chat
        def handle_table_click(table_data, evt: gr.SelectData):
            """Handle clicking on a table row - loads conversation and switches to Chat tab"""
            if evt.index is not None and len(evt.index) >= 1:
                row_index = evt.index[0]
                
                # Use the current table data passed as input (not re-fetching)
                # table_data comes as a pandas DataFrame or list of lists
                if table_data is not None and len(table_data) > row_index:
                    # Handle both DataFrame and list formats
                    if hasattr(table_data, 'iloc'):
                        # It's a DataFrame
                        conv_id = str(table_data.iloc[row_index, 0])
                    else:
                        # It's a list of lists
                        conv_id = table_data[row_index][0]
                    
                    # Set the app's conversation ID to the selected one
                    app_instance.conversation_id = conv_id.strip()
                    
                    # Rebuild history from database
                    history = app_instance._rebuild_history_from_database()
                    
                    logger.info(f"📖 Loaded conversation {conv_id} from table click - switching to Chat tab")
                    
                    # Return: history and switch to Chat tab
                    return history, gr.Tabs(selected="chat_tab")
            
            return [], gr.Tabs(selected="conversations_tab")
        
        conversations_table.select(
            fn=handle_table_click,
            inputs=[conversations_table],  # Pass current table data as input
            outputs=[chatbot, tabs],  # Update chatbot and switch tabs
        )
        
        # Handle refresh button
        def handle_refresh():
            """Refresh the conversations table"""
            return app_instance._get_all_conversations_table()
        
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
    
    return conversations_tab_component, conversations_table
