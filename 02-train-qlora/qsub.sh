#!/bin/bash
#PBS -N train
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=16gb:ngpus=1:gpu_cap=compute_70:gpu_mem=20gb
#PBS -l walltime=01:00:00
#PBS -m bae
# mail on begin, abort, end

module add cuda/12.6.1-gcc-10.2.1-hplxoqp

export PATH=/storage/brno12-cerit/home/hrabalm/.local/bin:$PATH
source /storage/brno12-cerit/home/hrabalm/venvs.uv/npfl101/bin/activate


FILE="$SCRIPT_DIR/train.py"
cd "$SCRIPT_DIR"  # note that in some cases this can affect performance, depending on how the cwd is used
"$PYTHON" "$FILE"

clean_scratch
