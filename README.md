# A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving
_This repository was created through a fork of the [S-LoRA repository](https://github.com/S-LoRA/S-LoRA) as a result of the workshop manuscript [A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving](https://arxiv.org/abs/2508.08343)_

### Introduction
This repository includes the code modifications made to the original S-LoRA repository to support the experiments presented in the mentioned manuscript for Figure 5. It also contains all the results and code used to generate the figure.

The sections below detail:
- The specific code changes
- The required steps for setting up the environment
- The required steps to run the experiments
- The results of the manuscript

For the rest of figures and tables of the manuscript that were done over vLLM, please refer to the [counterpart repository](https://github.com/FerranAgulloLopez/GPULLMAdapterOptimization).

### Code changes
This repository includes modifications both to the benchmarking and server components of the S-LoRA serving system. Our main changes in these two components are the following:
- Benchmark component: 
  - An updated benchmark script, _benchmarks/scaling_adapters/benchmark_serving.py_, equal to the vLLM benchmark script to be able to compare between the two frameworks that supports the following:
    - Adapter management: It is able to collect the adapters deployed in the server and send requests to them.
    - Adapter creation: Two scripts, _benchmarks/scaling_adapters/create_dummy_loras.py_ and _benchmarks/scaling_adapters/replicate_dummy_loras.py_, as auxiliary elements to create dummy adapters to later use in the experiments.
    - Integrated server launching: It can launch the server automatically using the `--launch-server` flag. Server arguments can be passed with `--server-args`. This simplifies the process of running a benchmark from a single entry point.
- Server component
  - HPC Slurm deployment: We included the required steps and data to deploy and run the server and benchmark with Slurm in HPC environments (singularity images) in the directory _benchmarks/scaling_adapters/deployment_.
  - Batch running: It includes the script _benchmarks/scaling_adapters/deployment/launcher.py_ that acts as a launcher to run multiple experiments with different configurations transparent from Slurm.
  - Bug solving: This launcher includes the solution for a bug that appears in Slurm deployments with Triton (__benchmarks/scaling_adapters/bug_solver_).
  - More logging: The server code is updated to provide more logging. 
  - Easier usage: The server code is updated to solve clashes between experiments when they were running at the same time because of port oversubscribing.

### How to set up
We show how to reproduce the experiments that appear in the paper, where we use Singularity and Slurm, which must be installed prior to execution. Nevertheless, as the original S-LoRA code, everything can also be run with docker or plainly with Python. Follow the required steps:
- Create the base Docker image as the foundation for the Singularity image: `docker build -f benchmarks/scaling_adapters/deployment/Dockerfile -t slora .`
- Create the Singularity image: `sudo singularity build slora.sif benchmarks/scaling_adapters/deployment/Singularityfile`
- If having problems with memory errors, you can include swap space like following:
```
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### How to run
Once the Singularity image has been built, you can easily reproduce the experiments from the paper using the commands provided in the .txt configuration files. For every run set of experiments we include a config*.txt file with the command that was used ot run it (it is created automatically). These commands invoke the launcher that does all effort in running the desired experiments in the Slurm cluster transparent to the user.

For instance, the directory _benchmarks/scaling_adapters/definitive_results/llama-2-7b/mean_dataset/rank_32_ contains the file _config-90903.txt_ with the ready-to-run command that was used to run the corresponding experiments. Simply execute it from the root directory of the repository, and these experiments will be submitted again to the Slurm queue for execution.

In this example we have the following:
```
PYTHONPATH=. python benchmarks/scaling_adapters/deployment/launcher.py \
--user bsc98 \
--queue acc_debug \
--max-duration 00:30:00 \
--results-path benchmarks/scaling_adapters/definitive_results/gpu_capacity/experiments_joan/llama-2-7b/mean_dataset/rank_32 \
--default-server-args "{'--disable_log_stats': '', '--model_dir': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--tokenizer_mode': 'auto', '--max_total_token_num': '82528', '--swap': ''}" \
--default-benchmark-args "{'--backend': 'slora', '--disable-tqdm': '', '--dataset-path': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data/dummy_dataset_mean.json', '--endpoint': '/generate_stream', '--model': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b', '--save-result': '', '--lora-dir': '/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b/lora/yard1_sql-lora-test_dummy_rank_32_simplified', '--infinite-behaviour': '', '--total-time': '1200'}" \
--test-server-args "{}" \
--test-benchmark-args "{'--lora-number': ['32', '64', '96', '128', '160', '192', '256', '320', '384', '448', '512', '640', '768'], '--request-rate-by-lora': ['0.4', '0.2', '0.1', '0.05', '0.025', '0.0125']}" \
```

As shown, the command invokes the launcher in deployment. While the full list of arguments can be found in the corresponding launcher script, here is a summary of the key components defined in this example:

- `PYTHONPATH`: Set to the root of the repository to define the Python working directory.
- `--user`: Specifies the Slurm user account.
- `--queue`: Specifies the Slurm queue or partition.
- `--max-duration`: Sets the maximum allowed runtime for the Slurm job.
- `--results-path`: Path where all experiment results will be saved.
- `--default-server-args`: Arguments passed to the vLLM server for all experiments.
- `--default-benchmark-args`: Common benchmark arguments used across all experiments.
- `--test-server-args`: Server arguments that will vary across different experiments.
- `--test-benchmark-args`: Benchmark arguments that will vary across different experiments.

The launcher will automatically initiate an experiment for every combination of server and benchmark arguments defined via the `--test-server-args` and `--test-benchmark-args` flags. Each of these experiments will also include the default arguments specified by the `--default-server-args` and `--default-benchmark-args` flags. Every experiment is executed through the appropriate benchmark script, which also handles launching of the vLLM server.

Take into account that these commands use input arguments for defining the locations of the datasets and models between others. They will need to be change for them to work properly.

### Manuscript results
The results used to create Figure 5 of the manuscript appear in the directory _benchmarks/scaling_adapters/definitive_results_. In that folder appears the outcome of the run experiments with Slurm, along a chart.py python script that reads these experiment outcomes and produces the corresponding figure.

### How to cite
If using these code modifications please cite this paper:
```
Agulló, F., Oliveras, J., Wang, C., Gutiérrez-Torre, A., Tardieu, O., Youssef, A., Torres, J., & Berral, J. Ll. (2025). A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving. In NeurIPS 2025 Machine Learning Systems (MLSys) Workshop.
```