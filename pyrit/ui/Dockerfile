# Dockerfile for PyRIT Endpoint Chat App
# 
# Build: docker build -t pyrit-chat -f pyrit/ui/Dockerfile .
# Run: docker run -p 7860:7860 --env-file .env pyrit-chat --target OpenAIChatTarget

FROM python:3.11-slim

LABEL maintainer="PyRIT Team"
LABEL description="PyRIT Endpoint Chat - Gradio app for multi-modal AI interactions"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy PyRIT source
COPY . /app/

# Install PyRIT with gradio support
RUN pip install --no-cache-dir -e ".[gradio]"

# Expose Gradio default port
EXPOSE 7860

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Entrypoint is the CLI
ENTRYPOINT ["python", "-m", "pyrit.ui.gradio_chat_cli"]

# Default to showing help if no args provided
CMD ["--help"]
