#!/bin/bash
#SBATCH --job-name=__192_0.4
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=benchmarks/scaling_adapters/definitive_results/gpu_capacity/experiments_joan/llama-2-7b/mean_dataset/rank_32/__192_0.4/log_%j.out
#SBATCH --error=benchmarks/scaling_adapters/definitive_results/gpu_capacity/experiments_joan/llama-2-7b/mean_dataset/rank_32/__192_0.4/log_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --gres gpu:1
#SBATCH --time=00:30:00
module load singularity
singularity exec --nv  --env PYTHONPATH=. --env TOKENIZERS_PARALLELISM=false --bind /gpfs/home/bsc/bsc098408/MN4/bsc98408/S-LoRA/slora/server:/workspace/slora/server --bind /gpfs/home/bsc/bsc098408/MN4/bsc98408/S-LoRA/benchmarks/scaling_adapters/bug_solver/triton_build.py:/usr/local/lib/python3.9/dist-packages/triton/common/build.py /gpfs/scratch/bsc98/bsc098069/llm_benchmarking/images/slora.sif python benchmarks/scaling_adapters/benchmark_serving.py --port='7680' --result-dir='benchmarks/scaling_adapters/definitive_results/gpu_capacity/experiments_joan/llama-2-7b/mean_dataset/rank_32/__192_0.4' --backend='slora' --disable-tqdm --dataset-path='/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/data/dummy_dataset_mean.json' --endpoint='/generate_stream' --model='/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b' --save-result --lora-dir='/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b/lora/yard1_sql-lora-test_dummy_rank_32_simplified' --infinite-behaviour --total-time='1200' --lora-number='192' --request-rate-by-lora='0.4' --launch-server --server-args='--port=7680 --nccl_port=7681 --disable_log_stats --model_dir=/gpfs/scratch/bsc98/bsc098069/llm_benchmarking/models/llama/llama-2-7b --tokenizer_mode=auto --max_total_token_num=82528 --swap'