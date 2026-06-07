from lib.providers.services import service
import os
import openai
import termcolor
from os import getenv
import base64
from io import BytesIO
import json
import time


client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_KEY")
)


def get_thinking_budget(context):
    """Get thinking budget from environment variable or context, mapped to
    OpenRouter reasoning parameters.

    Returns a dict suitable for passing as extra_body['reasoning'], or None
    if thinking is disabled.
    """
    thinking_level = os.environ.get('MR_THINKING_LEVEL', 'off').lower()
    if context is not None:
        thinking_level = context.agent.get('thinking_level', thinking_level)

    print("found thinking level")
    # Map MindRoot thinking levels to OpenRouter reasoning effort levels
    effort_map = {
        'off': None,
        'minimal': 'minimal',
        'low': 'low',
        'medium': 'medium',
        'high': 'high',
        'very_high': 'xhigh',
        'maximum': 'xhigh',
    }

    if thinking_level in effort_map:
        effort = effort_map[thinking_level]
        if effort is None:
            return None
        return {'effort': effort}

    # Try parsing as integer (token budget)
    try:
        budget = int(thinking_level)
        if budget <= 0:
            return None
        return {'max_tokens': max(1024, budget)}
    except ValueError:
        return {'effort': 'medium'}


@service()
async def stream_chat(model="meta-llama/llama-3.1-405b-instruct", messages=[], context=None, num_ctx=2048, temperature=0.0, max_tokens=20024, num_gpu_layers=12):
    """OpenRouter streaming chat service with reasoning token support.

    Args:
        model (str): Model identifier (defaults to llama-3.1-405b-instruct if not overridden)
        messages (list): List of conversation messages
        context: Session context
        num_ctx (int): Context window size (unused)
        temperature (float): Sampling temperature
        max_tokens (int): Maximum tokens to generate
        num_gpu_layers (int): GPU layers (unused)

    Returns:
        AsyncGenerator: Streams response tokens
    """
    try:
        # Build reasoning config from context thinking level
        reasoning_config = get_thinking_budget(context)
        thinking_enabled = reasoning_config is not None

        #if reasoning_config is None:
        #    reasoning_config = {'max_tokens': max_tokens // 2}
        #    thinking_enabled = True

        kwargs = {
            'model': model,
            'stream': True,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'extra_body': {
                "reasoning": {
                    "exclude": True
               }
            }
        }

        if thinking_enabled:
            # if extra_body already has a reasoning config (e.g. from context), merge it with the default
            reasoning_config = {**{'exclude': True}, **reasoning_config} if kwargs.get('extra_body', {}).get('reasoning') else reasoning_config
            # Some reasoning models require temperature=1
            # but OpenRouter handles this per-provider, so we leave it

        print(kwargs['extra_body'])
        stream = await client.chat.completions.create(**kwargs)
        print(stream)

        async def content_stream(original_stream):
            in_reasoning = False
            reasoning_started = False
            need_strip_bracket = False
            reasoning_pending = ''  # Buffer for backslash at chunk boundaries
            pending_content = ''
            last_content_flush = time.monotonic()
            # Coalesce tiny provider token chunks before handing them to MindRoot's
            # JSON command parser. This avoids O(n^2)-ish reparsing/partial-SSE churn.
            flush_interval = float(os.getenv("AH_STREAM_FLUSH_INTERVAL", "0.025"))
            flush_chars = int(os.getenv("AH_STREAM_FLUSH_CHARS", "512"))
            slow_after_chars = int(os.getenv("AH_STREAM_FLUSH_SLOW_AFTER_CHARS", "2000"))
            slow_interval = float(os.getenv("AH_STREAM_FLUSH_SLOW_INTERVAL", "0.5"))
            slow_flush_chars = int(os.getenv("AH_STREAM_FLUSH_SLOW_CHARS", "4096"))
            very_slow_after_chars = int(os.getenv("AH_STREAM_FLUSH_VERY_SLOW_AFTER_CHARS", "8000"))
            very_slow_interval = float(os.getenv("AH_STREAM_FLUSH_VERY_SLOW_INTERVAL", "1.0"))
            very_slow_flush_chars = int(os.getenv("AH_STREAM_FLUSH_VERY_SLOW_CHARS", "8192"))
            streamed_content_chars = 0
            def current_flush_interval():
                return very_slow_interval if streamed_content_chars >= very_slow_after_chars else slow_interval if streamed_content_chars >= slow_after_chars else flush_interval
            def current_flush_chars():
                return very_slow_flush_chars if streamed_content_chars >= very_slow_after_chars else slow_flush_chars if streamed_content_chars >= slow_after_chars else flush_chars

            async for chunk in original_stream:
                if os.getenv("AH_DEBUG", "False") == "True":
                    print(chunk)
                    try:
                        if len(chunk.choices) > 0:
                            print(termcolor.colored(chunk.choices[0].delta.content, "green"), end="")
                    except Exception as e1:
                        print('1 openrouter debug error:', e1)
                        pass

                if len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta

                # Check for reasoning_details (newer unified format)
                reasoning_details = getattr(delta, 'reasoning_details', None)
                if False and reasoning_details:
                    for detail in reasoning_details:
                        detail_type = detail.get('type', '') if isinstance(detail, dict) else getattr(detail, 'type', '')
                        text = None
                        if detail_type == 'reasoning.text':
                            text = detail.get('text', '') if isinstance(detail, dict) else getattr(detail, 'text', '')
                        elif detail_type == 'reasoning.summary':
                            text = detail.get('summary', '') if isinstance(detail, dict) else getattr(detail, 'summary', '')
                        # Skip encrypted reasoning

                        if text:
                            if not reasoning_started:
                                reasoning_started = True
                                in_reasoning = True
                                yield '[{"reasoning": "'
                            json_str = json.dumps(text)
                            without_quotes = json_str[1:-1]
                            # Buffer backslash at end of chunk to prevent
                            # invalid escape sequences at chunk boundaries
                            combined = reasoning_pending + without_quotes
                            reasoning_pending = ''
                            if combined.endswith('\\'):
                                reasoning_pending = '\\'
                                combined = combined[:-1]
                            if combined:
                                yield combined
                    continue

                # Check for reasoning / reasoning_content (legacy format, e.g. DeepSeek R1)
                reasoning_content = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                if False and reasoning_content:
                    if not reasoning_started:
                        reasoning_started = True
                        in_reasoning = True
                        # Start the array and reasoning object
                        yield '[{"reasoning": "'
                    json_str = json.dumps(reasoning_content)
                    without_quotes = json_str[1:-1]
                    combined = reasoning_pending + without_quotes
                    reasoning_pending = ''
                    if combined.endswith('\\'):
                        reasoning_pending = '\\'
                        combined = combined[:-1]
                    if combined:
                        yield combined
                    continue

                # Regular content
                content = getattr(delta, 'content', None)
                if content is not None and content != "":
                    # If we were in reasoning, close the reasoning block first
                    if in_reasoning:
                        in_reasoning = False
                        # Flush any pending reasoning buffer before closing
                        if reasoning_pending:
                            yield reasoning_pending
                            reasoning_pending = ''
                        # Close reasoning value and object, add comma to continue the array
                        yield '"}, '
                        need_strip_bracket = True
                    # Strip the leading [ from LLM's command array so it merges
                    # into the reasoning array
                    if need_strip_bracket:
                        content = content.lstrip()
                        if not content:
                            # Pure whitespace chunk, keep waiting for the bracket
                            continue
                        elif content.startswith('['):
                            content = content[1:]
                            need_strip_bracket = False
                        else:
                            need_strip_bracket = False
                    if flush_interval <= 0 or flush_chars <= 0:
                        yield content
                    else:
                        pending_content += content
                        now = time.monotonic()
                        interval = current_flush_interval()
                        chars = current_flush_chars()
                        stripped = pending_content.rstrip()
                        if (
                            len(pending_content) >= chars
                            or (now - last_content_flush) >= interval
                            or stripped.endswith(']')
                        ):
                            yield pending_content
                            streamed_content_chars += len(pending_content)
                            pending_content = ''
                            last_content_flush = now

            if pending_content:
                streamed_content_chars += len(pending_content)
                yield pending_content

            # If stream ends while still in reasoning block, close it
            if False and in_reasoning:
                # Flush any pending reasoning buffer before closing
                if reasoning_pending:
                    yield reasoning_pending
                    reasoning_pending = ''
                yield '"}]\n'

        return content_stream(stream)

    except Exception as e:
        print('openrouter error:', e)
        raise  # Re-raise the exception for proper error handling


@service()
async def get_image_dimensions(context=None):
    return (1568, 1568, 1192464)


@service()
async def format_image_message(pil_image, context=None):
    """Format image for OpenRouter using OpenAI's image format"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }


@service()
async def format_audio_message(
    audio_bytes: bytes,
    *,
    mime_type: str = "audio/wav",
    context=None,
):
    """Format audio for Mistral Voxtral via OpenRouter.

    Voxtral expects chat message content chunks of type `input_audio` where
    `input_audio` is a base64-encoded audio payload.
    """
    if audio_bytes is None:
        raise ValueError("audio_bytes is required")

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "type": "input_audio",
        "input_audio": audio_base64,
        "_mime_type": mime_type,
    }


@service()
async def get_service_models(context=None):
    """Get available models for the service"""
    try:
        print("....!")
        all_models = await client.models.list()
        ids = []
        for model in all_models.data:
            ids.append(model.id)

        return {"stream_chat": ids}
    except Exception as e:
        print('Error getting models (OpenRouter):', e)
        return {"stream_chat": []}
