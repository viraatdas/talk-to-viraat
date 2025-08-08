import csv
import json
import re
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Config ===
CSV_PATH = "messages.csv"
OUTPUT_PATH = "filtered_finetune.jsonl"
MODEL_NAME = "openai/gpt-oss-20b"

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)
model.eval()

# === Load messages from CSV ===
messages = []
with open(CSV_PATH, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["text"].strip():
            messages.append({
                "timestamp": datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S"),
                "is_from_me": row["is_from_me"] == "1",
                "sender": row["sender"],
                "text": row["text"].strip()
            })

# === LLM Filter ===
def extract_json_from_response(response: str) -> dict | None:
    try:
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if not match:
            return None

        # Fix unquoted booleans
        raw = match.group()
        raw = raw.replace("{humanlike: yes", '{"humanlike": "yes"')
        raw = raw.replace("{humanlike: no", '{"humanlike": "no"')
        raw = re.sub(r'"explanation":\s*([^"].*?)([}\n])', r'"explanation": "\1"\2', raw)

        return json.loads(raw)
    except Exception as e:
        print(f"⚠️ JSON parsing error: {e}")
        return None



def llm_filter(user, assistant):
    prompt = (
        "Below is a short exchange between two people: one is the user, the other is the assistant. "
        "Both messages are informal, text-style, and may include emojis, slang, or typos. "
        "Determine if this feels like a natural, realistic exchange you'd find in an actual human text conversation. "
        "Do NOT penalize for grammar, punctuation, or style — only reject if the exchange is confusing, incoherent, or unnatural.\n\n"
        "Respond ONLY with a JSON object using *double quotes*, no explanation outside the object:\n"
        '{"humanlike": "yes" or "no", "explanation": "brief reason here"}\n\n'
        f"User: {user.strip()}\n"
        f"Assistant: {assistant.strip()}"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids.input_ids, max_new_tokens=96, do_sample=False)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    parsed = extract_json_from_response(decoded)
    if not parsed:
        print("⚠️ LLM response parsing failed:")
        print(decoded)
        return False

    return parsed["humanlike"].lower() == "yes"

# === Group into user/assistant pairs ===
def build_pairs(messages):
    dataset = []
    i = 1
    while i < len(messages):
        if not messages[i - 1]["is_from_me"] and messages[i]["is_from_me"]:
            user_msg = messages[i - 1]["text"]
            assistant_msg = messages[i]["text"]

            if llm_filter(user_msg, assistant_msg):
                dataset.append({
                    "messages": [
                        {
                            "role": "developer",
                            "content": "You are Viraat. Speak like Viraat: dry, witty, text-like bursts. Respond casually, smartly, often using newlines to separate thoughts."
                        },
                        {
                            "role": "user",
                            "content": user_msg
                        },
                        {
                            "role": "assistant",
                            "channel": "final",
                            "content": assistant_msg
                        }
                    ]
                })
        i += 1
    return dataset

# === Process and save ===
print("🔍 Filtering message pairs...")
filtered_dataset = build_pairs(messages)

print(f"💾 Saving {len(filtered_dataset)} examples to {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
    for item in filtered_dataset:
        out_file.write(json.dumps(item) + "\n")

