"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    python -m vllm.entrypoints.openai.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import shlex
import argparse
import asyncio
import json
import os
import random
import signal
import time
import warnings
import requests
import subprocess
from subprocess import Popen
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Tuple, Dict

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    mean_ttfts_ms_by_lora: Dict[str, float]
    mean_ttfts_ms_by_user: Dict[str, float]
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if 'mean' in dataset_path:  # TODO refactor all of this
            prompt_len = 250
            output_len = 231
        elif 'p25' in dataset_path:
            prompt_len = 23
            output_len = 27
        elif 'p75' in dataset_path:
            prompt_len = 423
            output_len = 358
        else:
            raise Exception
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    input_requests_loras: List[str],
    input_requests_users: List[str],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int, str, str], None]:
    input_requests = iter(input_requests)
    for index, (prompt, prompt_len, output_len) in enumerate(input_requests):
        yield prompt, prompt_len, output_len, input_requests_loras[index], input_requests_users[index]

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    input_requests_loras: List[str],
    input_requests_users: List[str],
    outputs: List[RequestFuncOutput],
    dur_s: float,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens = []
    total_input = 0
    completed = 0
    itls = []
    tpots = []
    ttfts = []
    ttfts_by_lora: Dict[str, List[float]] = {}
    ttfts_by_user: Dict[str, List[float]] = {}
    for i in range(len(outputs)):
        if outputs[i] is not None and outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note: this may inflate the output token count slightly
            output_len = len(outputs[i].itl) + 1  # TODO refactor
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            request_lora: str = input_requests_loras[i]
            if request_lora in ttfts_by_lora:
                ttfts_by_lora[request_lora] += [outputs[i].ttft]
            else:
                ttfts_by_lora[request_lora] = [outputs[i].ttft]
            request_user: str = input_requests_users[i]
            if request_user in ttfts_by_user:
                ttfts_by_user[request_user] += [outputs[i].ttft]
            else:
                ttfts_by_user[request_user] = [outputs[i].ttft]
            completed += 1
        else:
            actual_output_lens.append(0)
            ttfts.append(dur_s)
            request_lora: str = input_requests_loras[i]
            if request_lora in ttfts_by_lora:
                ttfts_by_lora[request_lora] += [dur_s]
            else:
                ttfts_by_lora[request_lora] = [dur_s]
            request_user: str = input_requests_users[i]
            if request_user in ttfts_by_user:
                ttfts_by_user[request_user] += [dur_s]
            else:
                ttfts_by_user[request_user] = [dur_s]

    mean_ttfts_by_lora: Dict[str, float] = {}
    for user, values in ttfts_by_lora.items():
        mean_ttfts_by_lora[user] = np.mean(values or 0) * 1000
    mean_ttfts_by_user: Dict[str, float] = {}
    for user, values in ttfts_by_user.items():
        mean_ttfts_by_user[user] = np.mean(values or 0) * 1000

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        mean_ttfts_ms_by_lora=mean_ttfts_by_lora,
        mean_ttfts_ms_by_user=mean_ttfts_by_user,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


class Server:

    def __init__(self, server_args: str, output_path: str, available_loras: List[str]):
        super(Server, self).__init__()
        self.server_args = server_args
        self.output_path = output_path
        self.server_out = None
        self.server_err = None
        self.available_loras = available_loras

    def run(self) -> Popen:
        try:
            self.server_out = open(os.path.join(self.output_path, 'server_out.log'), 'w')
            self.server_err = open(os.path.join(self.output_path, 'server_err.log'), 'w')
            assert '--lora-dirs' not in self.server_args
            lora_arg: str = ''
            for lora in self.available_loras:
                lora_arg += f' --lora-dirs {lora}'
            command = f'python -m slora.server.api_server {lora_arg} {self.server_args}'

            open_subprocess = subprocess.Popen(
                shlex.split(command),
                shell=False,
                cwd='/',
                stdout=self.server_out,
                stderr=self.server_err,
            )
            return open_subprocess
        except Exception as e:
            print(e)
            if self.server_out:
                self.server_out.close()
            if self.server_err:
                self.server_err.close()
            raise e

    def terminate(self, open_subprocess: Popen) -> None:
        open_subprocess.kill()
        open_subprocess.terminate()
        open_subprocess.wait()
        if self.server_out:
            self.server_out.close()
        if self.server_err:
            self.server_err.close()

        # SLoRA bug solver, pid using GPU is different from the fork done here
        try:
            response = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            if response.returncode == 0:
                gpu_process_pid = int(response.stdout)
                subprocess.run(['kill', '-9', str(gpu_process_pid)])
            else:
                print(f'Nvidia-sim command return with code {response.returncode}')
        except Exception as e:
            print(e)


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    input_requests: List[Tuple[str, int, int]],
    input_requests_loras: List[str],
    input_requests_users: List[str],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    infinite_behaviour: bool,
    total_time: Optional[int] = None,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len = input_requests[0]
    test_input = RequestFuncInput(
        model=input_requests_loras[0],
        user=input_requests_users[0],
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")
    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    time_to_send = len(input_requests) / request_rate
    tasks = []
    async for request in get_request(input_requests, input_requests_loras, input_requests_users, request_rate):
        prompt, prompt_len, output_len, lora, user = request
        request_func_input = RequestFuncInput(
            model=lora,
            user=user,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))

    if infinite_behaviour:
        if total_time is not None:
            time_to_wait = total_time
            print(f"Using specified total_time: {time_to_wait}s")
        else:
            time_to_wait = time_to_send
            print(f"Using calculated time_to_send: {time_to_wait}s")

        await asyncio.sleep(time_to_wait - (time.perf_counter() - benchmark_start_time))
        intermediate_benchmark_duration = time.perf_counter() - benchmark_start_time

        outputs = []
        for index in range(len(tasks)):
            if tasks[index].done():
                outputs.append(tasks[index].result())
            else:
                outputs.append(None)

        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            input_requests_loras=input_requests_loras,
            input_requests_users=input_requests_users,
            outputs=outputs,
            dur_s=intermediate_benchmark_duration,
        )
        result_intermediate = {
            "duration": intermediate_benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "mean_ttfts_ms_by_lora": metrics.mean_ttfts_ms_by_lora,
            "mean_ttfts_ms_by_user": metrics.mean_ttfts_ms_by_user,
            "median_ttft_ms": metrics.median_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
        }

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks)
        result = {'infinite_behaviour': ''}
        print("{s:{c}^{n}}".format(s=' Serving Benchmark Intermediate Result ', n=50, c='='))
        print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                        intermediate_benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:",
                                     metrics.total_output))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                        metrics.request_throughput))
        print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                        metrics.input_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                        metrics.output_throughput))
        print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
        print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                        metrics.median_ttft_ms))
        print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
        print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                                   n=50,
                                   c='-'))
        print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
        print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                        metrics.median_tpot_ms))
        print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
        print("{s:{c}^{n}}".format(s='Inter-token Latency', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
        print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
        print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
        print("=" * 50)
    else:
        result_intermediate = {'not_infinite_behaviour': ''}
        outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

        if not disable_tqdm:
            pbar.close()

        benchmark_duration = time.perf_counter() - benchmark_start_time

        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            input_requests_loras=input_requests_loras,
            input_requests_users=input_requests_users,
            outputs=outputs,
            dur_s=benchmark_duration,
        )

        print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
        print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                        benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:",
                                     metrics.total_output))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                        metrics.request_throughput))
        print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                        metrics.input_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                        metrics.output_throughput))
        print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
        print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                        metrics.median_ttft_ms))
        print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
        print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                                   n=50,
                                   c='-'))
        print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
        print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                        metrics.median_tpot_ms))
        print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
        print("{s:{c}^{n}}".format(s='Inter-token Latency', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
        print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
        print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
        print("=" * 50)

        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "mean_ttfts_ms_by_lora": metrics.mean_ttfts_ms_by_lora,
            "mean_ttfts_ms_by_user": metrics.mean_ttfts_ms_by_user,
            "median_ttft_ms": metrics.median_ttft_ms,
            "p99_ttft_ms": metrics.p99_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "p99_tpot_ms": metrics.p99_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "p99_itl_ms": metrics.p99_itl_ms,
        }
    return result_intermediate, result


def main(args: argparse.Namespace):
    def __assign_users_loras(
            input_requests: List[Tuple[str, int, int]],
            user_lora_request_relation: str,
            available_loras: List[str],
    ) -> Tuple[List[str], List[str]]:
        input_requests_users: List[str] = []
        input_requests_loras: List[str] = []

        # ADD DEBUGGING
        print(f"DEBUG: Available LoRAs count: {len(available_loras)}")
        print(f"DEBUG: First few LoRAs: {available_loras[:5]}")
        
        # Check if directories exist
        for i, lora in enumerate(available_loras[:5]):
            exists = os.path.exists(lora)
            print(f"DEBUG: LoRA {i} exists: {exists} - {lora}")
            if exists:
                files = os.listdir(lora)
                print(f"  Files: {files}")

        if user_lora_request_relation is None or user_lora_request_relation in ['default', 'balance']:
            index = 0
            while len(input_requests_users) < len(input_requests):
                input_requests_users.append(str(index))
                input_requests_loras.append(available_loras[index])
                index += 1
                if index >= len(available_loras):
                    index = 0
            aux_shuffled_list = list(zip(input_requests_users, input_requests_loras))
            random.shuffle(aux_shuffled_list)
            input_requests_users, input_requests_loras = zip(*aux_shuffled_list)
        elif user_lora_request_relation == 'imbalance':
            index = 0
            while len(input_requests_users) < len(input_requests):
                input_requests_users.append(str(index))
                input_requests_loras.append(available_loras[index])
                if index % 2 == 0 and len(input_requests_users) < len(input_requests):
                    input_requests_users.append(str(index))
                    input_requests_loras.append(available_loras[index])
                index += 1
                if index >= len(available_loras):
                    index = 0
            aux_shuffled_list = list(zip(input_requests_users, input_requests_loras))
            random.shuffle(aux_shuffled_list)
            input_requests_users, input_requests_loras = zip(*aux_shuffled_list)
        else:
            raise ValueError(f"User assignation {user_lora_request_relation} not implemented")

        values, counts = np.unique(input_requests_users, return_counts=True)
        print(f"Requests users. Values: {values}. Counts: {counts}")
        values, counts = np.unique(input_requests_loras, return_counts=True)
        print(f"Requests loras. Values: {values}. Counts: {counts}")
        return input_requests_users, input_requests_loras
    
    # Modifying the number of prompts based on the request rate to match the total time
    if args.infinite_behaviour and args.total_time is not None:
        if args.request_rate_by_lora is not None:
            calculated_prompts = int(args.lora_number * args.request_rate_by_lora * args.total_time)
            print(f"Calculating num_prompts for {args.total_time}s duration: "
                  f"{args.lora_number} loras * {args.request_rate_by_lora} rate * {args.total_time}s = {calculated_prompts}")
            # Use buffer factor to ensure we don't run out
            buffer_factor = 1.2
            args.num_prompts = int(calculated_prompts * buffer_factor)
            print(f"Setting num_prompts to {args.num_prompts} (with {buffer_factor}x buffer)")
    

    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        health_api_url = f"{args.base_url}/health/"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        health_api_url = f"http://{args.host}:{args.port}/health/"

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    server = None
    open_server_process = None
    try:
        request_rate = args.request_rate
        available_loras: List[str] = []
        '''for lora_index in range(args.lora_number):
            available_loras.append(os.path.join(args.lora_dir, f'dummy-lora_{lora_index}'))'''
        for lora_index in range(args.lora_number):
            available_loras.append(f'{args.lora_dir}-{lora_index}')
        input_requests_users, input_requests_loras = __assign_users_loras(
            input_requests,
            args.user_lora_request_relation,
            available_loras
        )
        if args.request_rate_by_lora is not None:
            request_rate = args.request_rate_by_lora * len(available_loras)

        if args.launch_server:
            server = Server(args.server_args, args.result_dir, available_loras)
            open_server_process = server.run()

            init_time = time.time()
            server_started = False
            while not server_started and time.time() - init_time < args.max_server_waiting_time:
                try:
                    if requests.get(health_api_url).status_code == 200:
                        server_started = True
                    else:
                        time.sleep(5)
                except Exception as e:
                    time.sleep(5)
            server_init_time = time.time() - init_time
            if not server_started:
                raise Exception("Server did not start on time")
            print(f"Server started in {server_init_time:.2f} seconds")

        benchmark_result = asyncio.run(
            benchmark(
                backend=backend,
                api_url=api_url,
                model_id=model_id,
                input_requests=input_requests,
                input_requests_loras=input_requests_loras,
                input_requests_users=input_requests_users,
                best_of=args.best_of,
                use_beam_search=args.use_beam_search,
                request_rate=request_rate,
                disable_tqdm=args.disable_tqdm,
                infinite_behaviour=args.infinite_behaviour,
                total_time=args.total_time,
            ))

        # Save config and results to json
        if args.save_result:
            result_json = {}

            # Setup
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_json["date"] = current_dt
            result_json["backend"] = backend
            result_json["model_id"] = model_id
            result_json["best_of"] = args.best_of
            result_json["use_beam_search"] = args.use_beam_search
            result_json["num_prompts"] = args.num_prompts

            # Metadata
            if args.metadata:
                for item in args.metadata:
                    if "=" in item:
                        kvstring = item.split("=")
                        result_json[kvstring[0].strip()] = kvstring[1].strip()
                    else:
                        raise ValueError(
                            "Invalid metadata format. Please use KEY=VALUE format."
                        )

            # Traffic
            result_json["request_rate"] = (
                args.request_rate if request_rate < float("inf") else "inf")

            # Merge with benchmark result
            result_json_intermediate = {**result_json, **benchmark_result[0]}
            result_json_final = {**result_json, **benchmark_result[1]}

            # Save to file
            base_model_id = model_id.split("/")[-1]
            file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  #noqa
            if args.result_dir:
                file_name = os.path.join(args.result_dir, file_name)
            with open(file_name, "w") as outfile:
                json.dump(result_json_final, outfile)
            base_model_id = model_id.split("/")[-1]
            file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}_intermediate.json"  # noqa
            if args.result_dir:
                file_name = os.path.join(args.result_dir, file_name)
            with open(file_name, "w") as outfile:
                json.dump(result_json_intermediate, outfile)
    finally:
        try:
            if args.launch_server and server and open_server_process:
                server.terminate(open_server_process)
                print('Server terminated')
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--request-rate-by-lora",
        type=float,
        default=None,
        help="Number of requests per second by LoRA. If this is inf, "
             "then all the requests are sent at time 0. "
             "Otherwise, we use Poisson process to synthesize "
             "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch server in addition to benchmark",
    )
    parser.add_argument(
        "--server-args",
        type=str,
        default="",
        help="Args to send to the server when launching. Only useful when passing --launch-server as well",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        required=True,
        help="LoRA directory to use",
    )
    parser.add_argument(
        "--lora-number",
        type=int,
        default=1,
        help="Number of LoRA replicas",
    )
    parser.add_argument(
        "--user-lora-request-relation",
        type=str,
        default=None,
        help="Relation of lora<->request<->user",
    )
    parser.add_argument('--infinite-behaviour',
                        action='store_true',
                        help='Finish benchmark once all requests have been sent',
                        default=False
                        )

    parser.add_argument(
        "--max-server-waiting-time",
        type=int,
        default=300,
        help="Maximum time in seconds to wait for the server to be ready",
    )

    parser.add_argument(
    "--total-time",
    type=int,
    default=None,
    help="Total time in seconds for the experiment when using infinite behaviour. "
         "If set with --infinite-behaviour, num-prompts will be calculated automatically.",
)

    args = parser.parse_args()
    main(args)
