# Endpoint Chat App - Multi-Modal UI

The PyRIT Endpoint Chat App provides a Gradio-based web interface for interacting with AI endpoints.
Simply set your environment variables and run the CLI command - no coding required!

## Features

- Multi-modal input (text, images, videos, audio) through web interface
- Conversation history management with database persistence
- Support for OpenAI and Azure OpenAI endpoints
- Easy configuration through environment variables

## Quick Start

### 1. Install Gradio

If not already installed:

```bash
pip install pyrit[gradio]
```

### 2. Set Environment Variables for your target in .env

See [the setup guide](../../setup/populating_secrets.md).

### 3. Launch the App

```bash
python -m pyrit.ui.gradio_chat_cli --target-class OpenAIChatTarget
```

### 4. Use the Web Interface

The app will open in your browser where you can:
- Type messages and get responses
- Upload images, videos, or audio files
- View conversation history
- Start new conversations with the "🆕 New Chat" button

## That's It!

The CLI handles all the setup for you. No need to write any code - just set your environment 
variables and run the command above. The web interface provides everything you need for 
interactive testing and experimentation.

## Using the Web Interface

Once the app launches, you'll see a web interface with:

- **Chat area**: View your conversation history
- **Message input**: Type your messages here
- **File upload**: Click to attach images, videos, or audio files
- **New Chat button**: Start a fresh conversation
- **Send button**: Submit your message

All conversation history is automatically saved to PyRIT's database, so you can
close and restart the app without losing your conversations.
