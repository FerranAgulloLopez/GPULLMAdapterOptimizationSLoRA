import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    user: str
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_slora(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    '''
    api_url = request_func_input.api_url
    assert api_url.endswith("/generate_stream")
    request_start_time = time.time()
    headers = {"User-Agent": "Benchmark Client"}
    payload = {
        "lora_dir": request_func_input.model,
        "inputs": request_func_input.prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": request_func_input.output_len,
        }
    }

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    ttft = None
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        if ttft is None:
                            ttft = time.time() - request_start_time
                        chunks.append(chunk)
                result = b"".join(chunks).decode("utf-8")
                print(result)
                result = json.loads(result)
                print(result)
                print(result['generated_text'][0].encode())
                if response.status == 200:
                    output.generated_text = result['generated_text'][0].encode()
                    output.success = True
                    output.ttft = ttft
                    output.latency = time.perf_counter() - request_start_time
                else:
                    output.error = response.reason or ""
                    output.success = False
                break
    except Exception as e:
        if isinstance(e, asyncio.CancelledError):
            raise e
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))



    if pbar:
        pbar.update(1)
    return output'''

    try:
        api_url = request_func_input.api_url
        assert api_url.endswith("/generate_stream")

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            assert not request_func_input.use_beam_search
            headers = {"User-Agent": "Benchmark Client"}
            payload = {
                "lora_dir": request_func_input.model,
                "inputs": request_func_input.prompt,
                "parameters": {
                    "do_sample": False,
                    "ignore_eos": True,
                    "max_new_tokens": request_func_input.output_len,
                }
            }

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(url=api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")
                            data = json.loads(chunk)
                            if data["finished"]:
                                latency = time.perf_counter() - st
                            else:
                                if data["token"]["text"]:
                                    timestamp = time.perf_counter()
                                    # First token
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    # NOTE: Some completion API might have a last
                                    # usage summary response without a token so we
                                    # do not want to include as inter-token-latency
                                    elif data["token"]["text"] is not None:
                                        output.itl.append(timestamp -  most_recent_timestamp)

                                    most_recent_timestamp = timestamp
                                    generated_text += data["token"]["text"]

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    raise e
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output
    except asyncio.CancelledError:
        pass


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    try:
        api_url = request_func_input.api_url
        assert api_url.endswith(
            "v1/completions"
        ), "OpenAI Completions API URL must end with 'v1/completions'."

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            assert not request_func_input.use_beam_search
            payload = {
                "user": request_func_input.user,
                "model": request_func_input.model,
                "prompt": request_func_input.prompt,
                "temperature": 0.0,
                "best_of": request_func_input.best_of,
                "max_tokens": request_func_input.output_len,
                "stream": True,
            }
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            }

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(url=api_url, json=payload,
                                        headers=headers) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                                  "data: ")
                            if chunk == "[DONE]":
                                latency = time.perf_counter() - st
                            else:
                                data = json.loads(chunk)

                                if data["choices"][0]["text"]:
                                    timestamp = time.perf_counter()
                                    # First token
                                    if ttft == 0.0:
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    # NOTE: Some completion API might have a last
                                    # usage summary response without a token so we
                                    # do not want to include as inter-token-latency
                                    elif data.get("usage", None) is None:
                                        output.itl.append(timestamp -
                                                          most_recent_timestamp)

                                    most_recent_timestamp = timestamp
                                    generated_text += data["choices"][0]["text"]

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    raise e
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output
    except asyncio.CancelledError:
        pass


# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "openai": async_request_openai_completions,
    "slora": async_request_slora,
}
