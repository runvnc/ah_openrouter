from lib.providers.services import service
import os
import openai
import termcolor
from os import getenv

client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_KEY")
)

@service()
async def stream_chat(model="meta-llama/llama-3.1-405b-instruct", messages=[], context=None, num_ctx=2048, temperature=0.0, max_tokens=3024, num_gpu_layers=12):
    """OpenRouter streaming chat service.
    
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
        # Allow environment override of model
        if os.environ.get("AH_OVERRIDE_LLM_MODEL", None) is not None:
            model = os.environ.get("AH_OVERRIDE_LLM_MODEL")
            print("Overriding model env specified ", model)

        stream = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        async def content_stream(original_stream):
            async for chunk in original_stream:
                if os.getenv("AH_DEBUG", "False") == "True":
                    print(termcolor.colored(chunk.choices[0].delta.content, "green"), end="")

                yield chunk.choices[0].delta.content or ""

        return content_stream(stream)

    except Exception as e:
        print('openrouter error:', e)
        raise  # Re-raise the exception for proper error handling
