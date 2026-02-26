import vllm
import vllm.lora
import json

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_MODEL = "../lora_output/checkpoint-225/"
llm = vllm.LLM(
    BASE_MODEL,
    enforce_eager=True,
    seed=42,
    enable_lora=True,
    max_lora_rank=32,
)
lora_request = vllm.lora.request.LoRARequest("adapter", 1, ADAPTER_MODEL)


def format_prompt(src_lang, tgt_lang, src):
    return (
        "Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}"
    ).format(src_lang=src_lang, tgt_lang=tgt_lang, src=src)


def print_messages(messages):
    print("Messages:")
    print(json.dumps(messages, indent=2))
    print("=" * 20)


sampling_params = vllm.SamplingParams(
    temperature=0,
    max_tokens=512,
    stop=["\n"],
)

messages = [
    [
        {
            "role": "user",
            "content": "Translate the following Czech source text to English:\nCzech: Praha je hlavní město České republiky.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following Czech source text to English:\nCzech: Máma mele maso.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following Czech source text to English:\nCzech: Maso mele máma.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following Czech source text to English:\nCzech: Mele máma maso.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Translate the following Czech source text to English:\nCzech: Mele maso máma.",
        },
    ],
    [
        {
            "role": "user",
            "content": "What is the largest city in the Czech Republic and what is its population?",
        },
    ],
]
print_messages(messages)

outputs = llm.chat(
    messages,
    sampling_params=sampling_params,
    lora_request=lora_request,
)

print("print(outputs)", outputs, "=" * 20, "\n\n")

for output in outputs:
    print(output.outputs[0].text)
    print("=" * 20)
