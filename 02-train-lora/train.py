import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATASET_NAME_OR_PATH = "../data/cs-en.jsonl"
OUTPUT_DIRECTORY = "../lora_output"

TEMPLATE = (
    "Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}"
)


def formatting_prompts_func(example):
    src_lang = example["src_lang"]
    tgt_lang = example["tgt_lang"]
    prompt = TEMPLATE.format(
        src=example["src"],
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": example["tgt"],
        },
    ]
    return {
        "messages": messages,
    }


def my_load_dataset():
    ds = load_dataset("json", data_files=DATASET_NAME_OR_PATH, split="train")
    ds = ds.map(formatting_prompts_func, num_proc=16)
    ds = ds.train_test_split(test_size=0.1)

    return ds["train"], ds["test"]


dataset, eval_dataset = my_load_dataset()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    r=16,
    lora_dropout=0.0,
    lora_alpha=32,
    bias="none",
    target_modules="all-linear",  # or list of modules
    task_type="CAUSAL_LM",
)


sft_config = SFTConfig(
    # train on assistant tokens only THIS NEEDS a tokenizer.chat_template with generation and endgeneration markers, see above. If you use model without them and can't fix it, just set this to False
    # assistant_only_loss=True,
    # Main hyperparameters
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,  # can be used to simulate larger batch sizes
    max_length=1024,  # you should properly set this depending on your data, task, model and VRAM
    learning_rate=4e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_steps=50,
    eval_strategy="steps",
    logging_steps=1,
    output_dir=OUTPUT_DIRECTORY,  # where to save the model checkpoints
    seed=42,
    save_strategy="steps",
    save_steps=50,
    # Technical parameters to save time or VRAM
    bf16=True,
    optim="adamw_8bit",  # saves VRAM by using 8-bit optimizers
    gradient_checkpointing=True,
    # some other parameters you can play around, these two are for regularization
    # max_grad_norm=0.3,
    # weight_decay=0.1,
    # report_to="wandb",  # uncomment if you want to use wandb
)


trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    args=sft_config,
    processing_class=tokenizer,
    peft_config=peft_config,
)

try:
    # Try to resume training from the last checkpoint if it exists. This is useful in case your training gets interrupted or takes longer than 24 hours (maximum runtime for gpu queue on Metacentrum)
    trainer_stats = trainer.train(resume_from_checkpoint=True)
except ValueError:
    trainer_stats = trainer.train()
