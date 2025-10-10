# PyRIT Endpoint Chat App

> A Gradio-based chat interface for PyRIT with multi-modal support and video generation capabilities

## 🎯 Quick Start

```bash
# 1. Install
pip install pyrit[gradio]

# 2. Create .env file in workspace root (see ../../.env_example for examples)

# 3. Run (PyRIT automatically loads .env)
python -m pyrit.ui.gradio_chat_cli --target-class OpenAIChatTarget

# 4. Open http://localhost:7860
```

> 💡 **Note**: PyRIT automatically loads `.env` and `.env.local` files from the workspace root when `initialize_pyrit()` is called.

---

## ✨ Features

- **Multi-Modal I/O**: Text, images, videos, audio input and output
- **Video Generation**: Perfect for testing Sora and similar models
- **Multiple Endpoints**: OpenAI, Azure OpenAI, Azure ML, Groq, Ollama, and more
- **Conversation History**: Persisted across sessions
- **Docker Support**: Easy containerized deployment
- **Python API**: Use programmatically in scripts/notebooks

---

## 🚀 Usage

### CLI (Recommended)

```bash
# Basic usage (.env automatically loaded from workspace root)
python -m pyrit.ui.gradio_chat_cli --target-class OpenAIChatTarget

# Custom port/host
python -m pyrit.ui.gradio_chat_cli --target-class OpenAIChatTarget --host 0.0.0.0 --port 8080

# Debug mode
python -m pyrit.ui.gradio_chat_cli --target-class OpenAIChatTarget --debug
```

**Available Target Classes:**
- `OpenAIChatTarget` - OpenAI, Azure OpenAI, Groq, Ollama, etc.
- `OpenAISoraTarget` - Video generation
- `OpenAIDALLETarget` - Image generation  
- [See all targets →](https://github.com/Azure/PyRIT/tree/main/pyrit/prompt_target/)

### Docker

```bash
# Docker Compose (easiest)
docker-compose -f pyrit/ui/docker-compose.yml up

# Docker CLI
docker build -t pyrit-chat -f pyrit/ui/Dockerfile .
docker run -p 7860:7860 --env-file .env pyrit-chat --target-class OpenAIChatTarget

# With persistence
docker run -p 7860:7860 --env-file .env -v $(pwd)/dbdata:/workspace/dbdata pyrit-chat --target-class OpenAIChatTarget
```

> ⚠️ **Docker Environment File Format**: When using `--env-file` with Docker, the file must follow Docker's env file syntax:
> - **No quotes** around values: `KEY=value` not `KEY="value"`
> - **No spaces** around `=`: `KEY=value` not `KEY = value`
> - **No variable substitution**: Use actual values, not `${OTHER_VAR}`
> - JSON values should be unquoted: `HEADERS={"key": "value"}` not `HEADERS='{"key": "value"}'`

## 🔧 Configuration Examples

> 💡 See [`../../.env_example`](../../.env_example) for complete examples with all PyRIT targets
