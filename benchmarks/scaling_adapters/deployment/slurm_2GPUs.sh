#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH -D ./
#SBATCH --ntasks=1
#SBATCH --output=${EXP_RESULTS_PATH}/log_%j.out
#SBATCH --error=${EXP_RESULTS_PATH}/log_%j.err
#SBATCH --cpus-per-task=40
#SBATCH --gres gpu:2
#SBATCH --time=${EXP_MAX_DURATION_SECONDS}
module load singularity
singularity exec --nv ${EXP_ENV_VARS} --bind ${EXP_HOME_CODE_DIR}/slora/server:${EXP_CONTAINER_CODE_DIR}/slora/server --bind ${EXP_HOME_CODE_DIR}/benchmarks/scaling_adapters/bug_solver/triton_build.py:/usr/local/lib/python3.9/dist-packages/triton/common/build.py ${EXP_CONTAINER_IMAGE} python ${EXP_BENCHMARK_EXECUTABLE} ${EXP_ARGS}