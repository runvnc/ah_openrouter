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
async def stream_chat(model="meta-llama/llama-3.1-405b-instruct", messages=[], context=None, num_ctx=2048, temperature=0.0, max_tokens=18024, num_gpu_layers=12):
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
        #if model is None and os.environ.get("AH_OVERRIDE_LLM_MODEL", None) is not None:
            #model = os.environ.get("AH_OVERRIDE_LLM_MODEL")
            #print("Overriding model env specified ", model)

        stream = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=messages,
            temperature=temperature,
            #response_format={"type": "json_object"},
            max_tokens=max_tokens
        )
        print(stream)
        async def content_stream(original_stream):
            async for chunk in original_stream:
                if True or os.getenv("AH_DEBUG", "False") == "True":
                    print(chunk)
                    try:
                        print(termcolor.colored(chunk.choices[0].delta.content, "green"), end="")
                    except Exception as e1:
                        pass

                yield chunk.choices[0].delta.content or ""

        return content_stream(stream)

    except Exception as e:
        print('openrouter error:', e)
        raise  # Re-raise the exception for proper error handling

@service()
async def get_image_dimensions(context=None):
    return (1568, 1568, 1192464)

@service()
async def format_image_message(pil_image, context=None):
    """Format image for DeepSeek using OpenAI's image format"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')
    
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
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

        return { "stream_chat": ids }
    except Exception as e:
        print('Error getting models (OpenRouter):', e)
        return { "stream_chat": [] }

