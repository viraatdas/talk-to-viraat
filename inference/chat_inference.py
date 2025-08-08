import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === CONFIG ===
BASE_MODEL = "openai/gpt-oss-20b"
LORA_DIR = "viraat-oss-ft-1k"

# === Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=True
)
model = PeftModel.from_pretrained(model, LORA_DIR)
model = model.merge_and_unload()
model.eval()

# === Starter message with harmony role
system_msg = {
    "role": "developer",
    "content": "You are ViraatBot. Speak like Viraat: dry, witty, text-like bursts. Use newlines to separate thoughts."
}
chat_history = [system_msg]

print("💬 Chat with ViraatBot. Type 'exit' to quit.\n")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("👋 Exiting.")
        break

    chat_history.append({"role": "user", "content": user_input})

    input_ids = tokenizer.apply_chat_template(
        chat_history,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True
    ).to(model.device)

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Remove the prompt portion
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Fix hallucinated channel names or roles
    cleaned = decoded_output.replace("final", "").replace("assistant", "").strip()

    print("ViraatBot:", cleaned)
    chat_history.append({"role": "assistant", "channel": "final", "content": cleaned})

