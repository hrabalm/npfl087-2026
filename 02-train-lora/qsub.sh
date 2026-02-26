#!/bin/bash
#PBS -N train
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=16gb:ngpus=1:gpu_cap=compute_70:gpu_mem=20gb
#PBS -l walltime=01:00:00
#PBS -m bae
# mail on begin, abort, end

# print info about the GPUs used
nvidia-smi

module add cuda/12.6.1-gcc-10.2.1-hplxoqp

source /storage/brno12-cerit/home/hrabalm/venvs.uv/3.12train_cuda_202602/bin/activate


FILE="$PBS_O_WORKDIR/train.py"
cd "$PBS_O_WORKDIR"  # note that in some cases this can affect performance, depending on how the cwd is used
python "$FILE"

clean_scratch
