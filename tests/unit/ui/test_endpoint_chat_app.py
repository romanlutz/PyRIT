# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for the Endpoint Chat App.

Note: These are basic unit tests. Full integration tests require actual endpoints.
"""

import importlib.util
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


def is_gradio_installed():
    """Check if gradio is installed."""
    return importlib.util.find_spec("gradio") is not None


@pytest.mark.skipif(not is_gradio_installed(), reason="Gradio is not installed")
class TestEndpointChatApp:
    """Tests for EndpointChatApp."""

    def test_app_initialization(self, sqlite_instance):
        """Test that the app can be initialized."""
        from pyrit.ui.endpoint_chat_app import EndpointChatApp

        mock_target = MagicMock()
        app = EndpointChatApp(target=mock_target)
        
        assert app.target is not None
        assert app.conversation_id is not None
        assert app.memory is not None

    def test_build_interface(self, sqlite_instance):
        """Test that the interface can be built."""
        from pyrit.ui.endpoint_chat_app import EndpointChatApp

        mock_target = MagicMock()
        mock_target.__class__.__name__ = "MockTarget"
        app = EndpointChatApp(target=mock_target)
        interface = app.build_interface()

        assert interface is not None
        # Check that it's a Gradio Blocks object
        assert hasattr(interface, "launch")


    def test_rebuild_history_from_database_empty(self, sqlite_instance):
        """Test rebuilding history when database is empty."""
        from pyrit.ui.endpoint_chat_app import EndpointChatApp

        mock_target = MagicMock()
        app = EndpointChatApp(target=mock_target)
        
        history = app._rebuild_history_from_database()
        
        assert history == []

    def test_chat_with_text_message(self, sqlite_instance):
        """Test chat method with text message."""
        from pyrit.ui.endpoint_chat_app import EndpointChatApp
        from pyrit.models import PromptRequestPiece, PromptRequestResponse
        from unittest.mock import patch
        from uuid import uuid4

        mock_target = MagicMock()
        mock_target.get_identifier = MagicMock(return_value={"id": str(uuid4())})
        
        mock_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="assistant",
                    original_value="Hello!",
                    converted_value="Hello!",
                    conversation_id="test",
                    prompt_target_identifier={"id": str(uuid4())},
                    converter_identifiers=[],
                )
            ]
        )
        mock_target.send_prompt_async = AsyncMock(return_value=mock_response)

        app = EndpointChatApp(target=mock_target)

        message = {"text": "Hi there"}
        response = app.chat(message, [])

        assert "Hello!" in response
@pytest.mark.skipif(not is_gradio_installed(), reason="Gradio is not installed")
class TestCLI:
    """Tests for the CLI."""

    def test_check_gradio_installed(self):
        """Test gradio detection."""
        from pyrit.ui.gradio_chat_cli import check_gradio_installed

        # If we're running these tests, gradio must be installed
        assert check_gradio_installed() is True

    def test_load_target_class_success(self):
        """Test loading a valid target class."""
        from pyrit.ui.gradio_chat_cli import load_target_class

        target_class = load_target_class(class_name="OpenAIChatTarget")
        
        assert target_class is not None
        assert target_class.__name__ == "OpenAIChatTarget"

    def test_load_target_class_invalid(self):
        """Test loading an invalid target class."""
        from pyrit.ui.gradio_chat_cli import load_target_class

        with pytest.raises(RuntimeError, match="Failed to import InvalidTarget"):
            load_target_class(class_name="InvalidTarget")

    def test_load_target_class_not_a_class(self):
        """Test loading something that's not a class."""
        from pyrit.ui.gradio_chat_cli import load_target_class

        with pytest.raises(RuntimeError):
            load_target_class(class_name="__name__")  # This is a module attribute, not a class

    def test_create_target_with_valid_class(self, sqlite_instance):
        """Test creating a target with valid parameters."""
        from pyrit.ui.gradio_chat_cli import create_target
        from unittest.mock import patch
        import os

        # Set up environment variables for OpenAIChatTarget (correct names)
        with patch.dict(os.environ, {
            'OPENAI_CHAT_ENDPOINT': 'https://api.openai.com/v1/chat/completions',
            'OPENAI_CHAT_KEY': 'test-key',
            'OPENAI_CHAT_DEPLOYMENT': 'gpt-4o'
        }):
            target = create_target(target_class_name="OpenAIChatTarget")
            
            assert target is not None
            assert target.__class__.__name__ == "OpenAIChatTarget"

    def test_create_target_missing_required_params(self, sqlite_instance):
        """Test creating a target with missing required parameters."""
        from pyrit.ui.gradio_chat_cli import create_target
        from unittest.mock import patch
        import os

        # Clear environment variables that might be set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Environment variable OPENAI_CHAT_ENDPOINT is required"):
                create_target(target_class_name="OpenAIChatTarget")

    def test_parse_args_default(self):
        """Test parsing arguments with defaults."""
        from pyrit.ui.gradio_chat_cli import parse_args
        from unittest.mock import patch
        import sys

        test_args = ['gradio_chat_cli.py', '--target-class', 'OpenAIChatTarget']
        
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
            
            assert args.target_class == 'OpenAIChatTarget'
            assert args.host == '0.0.0.0'
            assert args.port == 7860
            assert args.share is False
            assert args.debug is False

    def test_parse_args_custom_port_and_host(self):
        """Test parsing arguments with custom port and host."""
        from pyrit.ui.gradio_chat_cli import parse_args
        from unittest.mock import patch
        import sys

        test_args = [
            'gradio_chat_cli.py',
            '--target-class', 'OpenAIChatTarget',
            '--host', '127.0.0.1',
            '--port', '8080',
            '--debug'
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
            
            assert args.target_class == 'OpenAIChatTarget'
            assert args.host == '127.0.0.1'
            assert args.port == 8080
            assert args.debug is True

    def test_parse_args_share_flag(self):
        """Test parsing arguments with share flag."""
        from pyrit.ui.gradio_chat_cli import parse_args
        from unittest.mock import patch
        import sys

        test_args = [
            'gradio_chat_cli.py',
            '--target-class', 'OpenAIChatTarget',
            '--share'
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = parse_args()
            
            assert args.share is True
