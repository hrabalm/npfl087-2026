# npfl087-2026

## Useful links

- [https://docs.metacentrum.cz](https://docs.metacentrum.cz)
    - Metacentrum documentation - I highly recommend going through it
- [https://huggingface.co/docs/trl/index](https://huggingface.co/docs/trl/index)
    - Documentation for TRL (Transformer Reinforcement Learning) - the training library we are using
- [https://docs.metacentrum.cz/en/docs/tutorials/vscode-devel](https://docs.metacentrum.cz/en/docs/tutorials/vscode-devel)
    - Guide on how to run VSCode server on compute nodes on Metacentrum
- [https://huggingface.co/docs/transformers/perf_train_gpu_one](https://huggingface.co/docs/transformers/perf_train_gpu_one)
    - Some details about your options and optimizations for single GPU training
- [https://wandb.ai/](https://wandb.ai/)
    - Commercial service for experiment tracking

## Installation

First, we will install `uv` and create two virtual environments: one for vLLM and one for Transformers/TRL (fine-tuning).

Connect to the zenith frontend.

```bash
ssh username@zenith.metacentrum.cz
```

Clone this repo.

```bash
mkdir workspace
cd workspace
git clone https://github.com/hrabalm/npfl087-2026
```

Install `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run a bash subshell to load the updated `.bashrc`.

```bash
bash
```

You should now be able to call the `uv` binary:

```bash
uv --version
```

If the command fails, you might have to set up the PATH environment variables manually, for example:

```bash
export PATH=$PATH:/storage/brno12-cerit/home/hrabalm/.local/bin/
```

Or use it with its absolute path, for example:

```
/storage/brno12-cerit/home/hrabalm/.local/bin/uv --version
```

Note that you should probably use the absolute path to the binary in scripts anyway, because different geolocations use different storages.

Create a virtual environment.

```bash
mkdir ~/venvs.uv
uv venv --seed --python 3.12 ~/venvs.uv/3.12train_cuda_202602
```

Note down the absolute path to the venv:

```bash
pwd
```

For example, mine is `/storage/brno12-cerit/home/hrabalm/venvs.uv/3.12train_cuda_202602`. Activate the environment.

```bash
source /storage/brno12-cerit/home/hrabalm/venvs.uv/3.12train_cuda_202602/bin/activate
```

Install libraries.

```bash
source /storage/brno12-cerit/home/hrabalm/venvs.uv/3.12train_cuda_202602/bin/activate
uv pip install --torch-backend=cu118 transformers trl peft bitsandbytes pandas accelerate cytoolz sacrebleu wandb click typer dspy
```

Now create a second venv for vLLM and install it in a similar fashion.

```bash
uv venv --seed --python 3.12 ~/venvs.uv/3.12vllm_202602
source /storage/brno12-cerit/home/hrabalm/venvs.uv/3.12vllm_202602/bin/activate
uv pip install --torch-backend=cu118 vllm
```


Let's run the examples. We have two options: interactive job and batch job.
Beware that your working directory will be changed inside the job! In scripts,
you can return to the directory with the script using "cd $SCRIPT_DIR"

### Batch job

```bash
cd /storage/brno12-cerit/home/hrabalm/workspace/npfl087-2026/01-vllm-inference
```

View and modify the `predict.py` and `qsub.sh` files, then run the example with:

```bash
qsub qsub.sh
```

### Interactive job

```bash
qsub -I -l select=1:ncpus=2:mem=16gb:ngpus=1:gpu_mem=20gb -l walltime=2:00:00
```

```bash
cd /storage/brno12-cerit/home/hrabalm/workspace/npfl087-2026/01-vllm-inference
```

First, fix the `predict.py` file.

Read the contents of the `qsub.sh` script file and execute the updated commands manually.

## Tips

You can check the state of your jobs with `qstat`.

```bash
qstat -u USERNAME
```

Look at the job details.

```bash
qstat -f JOB_ID
```

Check job progress/outputs while it is running.

```bash
qstat -u USERNAME # find job id
qstat -f JOB_ID | grep host  # find host
cd /var/spool/pbs/spool
tail -f JOB_ID*  # e.g. tail -f 123456*
```

## Models you might check out

For many models, you need to register huggingface account and ask for permission
or agree with the conditions to access the model. Beware that some licenses
limit its use.

- Gemma 3 series by Google
    - Gemma license, not as permissive
    - [https://huggingface.co/google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
    - [https://huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
    - [https://huggingface.co/google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)
    - [https://huggingface.co/google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)
- EuroLLM
    - Apache 2 license
    - EuroLLM-9B-Instruct is quite good at translation, worse at general tasks
        - also very sensitive to prompt used
    - [https://huggingface.co/utter-project](https://huggingface.co/utter-project)
- Qwen
    - [https://huggingface.co/Qwen](https://huggingface.co/Qwen)
- and many more...