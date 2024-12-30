# AH OpenRouter Plugin

A MindRoot plugin that provides OpenRouter API integration for LLM access.

## Features
- Streaming chat completion service
- Configurable model selection
- Debug output options
- Environment-based configuration

## Configuration
Required environment variables:
- OPENROUTER_KEY: Your OpenRouter API key

Optional environment variables:
- AH_OVERRIDE_LLM_MODEL: Override default model
- AH_DEBUG: Enable debug output (set to "True")

## Installation
```bash
pip install -e .
```

## Usage
This plugin provides a service that can be used by other plugins or components:

```python
from ah_openrouter.mod import stream_chat

# Use the streaming service
response_stream = await stream_chat(
    model="meta-llama/llama-3.1-405b-instruct",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```
