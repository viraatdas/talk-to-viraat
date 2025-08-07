import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# === CONFIG ===
DATASET_PATH = "viraat_finetune.jsonl"  # your processed dataset
MODEL_NAME = "openai/gpt-oss-20b"
OUTPUT_DIR = "viraat-oss-ft"

# === Load Dataset ===
dataset = load_dataset("json", data_files=DATASET_PATH)["train"]
def fix_channel_fields(dataset):
    for i in range(len(dataset)):
        dataset[i]["messages"] = [
            {k: v for k, v in msg.items() if not (msg["role"] in {"developer", "user"} and k == "channel")}
            for msg in dataset[i]["messages"]
        ]
    return dataset

dataset = fix_channel_fields(dataset)


# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === Load Model with Quantization ===
quant_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config,
    use_cache=False,
    device_map="auto",
)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# === Apply LoRA ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# === Training Arguments ===
training_args = SFTConfig(
    learning_rate=2e-4,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir=OUTPUT_DIR,
    report_to="none",  # or "trackio", wandb, etc.
    push_to_hub=False,
)

# === Trainer ===
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()

# === Save Final Model ===
trainer.save_model(OUTPUT_DIR)
print(f"✅ Finished fine-tuning. Model saved to {OUTPUT_DIR}")

