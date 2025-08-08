import csv
import json
import re
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Config ===
CSV_PATH = "messages.csv"
OUTPUT_PATH = "filtered_finetune.jsonl"
MODEL_NAME = "openai/gpt-oss-20b"

# === Load tokenizer and model ===
(tokenizer := AutoTokenizer.from_pretrained(MODEL_NAME))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Left padding is more stable for decoder-only models when batching
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# === Stats ===
STATS: Dict[str, int] = {
    "pairs_total": 0,
    "pairs_kept": 0,
    "pairs_rejected": 0,
    "parse_failures": 0,
    "retry_success": 0,
}

# === Load messages from CSV ===

def parse_timestamp(value: str) -> datetime:
    value = (value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(value)
    except Exception:
        # Fall back to epoch to maintain deterministic ordering
        return datetime.fromtimestamp(0)


messages: List[Dict[str, Any]] = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        is_from_me_raw = (row.get("is_from_me") or "").strip()
        messages.append(
            {
                "timestamp": parse_timestamp(row.get("timestamp", "")),
                "is_from_me": is_from_me_raw in {"1", "true", "True", "yes", "Y"},
                "sender": (row.get("sender") or "").strip(),
                "text": text,
            }
        )

# Ensure chronological order even if CSV isn't sorted
messages.sort(key=lambda m: m["timestamp"])  

# === LLM Filter ===

def _find_balanced_json_objects(text: str) -> List[str]:
    objects: List[str] = []
    stack: List[int] = []
    in_string = False
    escape_next = False
    start_idx: Optional[int] = None

    for idx, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append(idx)
            if start_idx is None:
                start_idx = idx
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx : idx + 1]
                    objects.append(candidate)
                    start_idx = None
    return objects


def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    try:
        candidates = _find_balanced_json_objects(response_text)
        if not candidates:
            return None
        raw = candidates[-1].strip()

        # Normalize bad quotes and formatting issues
        raw = raw.replace("“", '"').replace("”", '"')
        raw = raw.replace("‘", "'").replace("’", "'")
        raw = re.sub(r"(\{|,)\s*(\w+)\s*:", r'\1 "\2":', raw)  # fix unquoted keys
        raw = re.sub(r":\s*(yes|no)([,\s}])", r': "\1"\2', raw, flags=re.IGNORECASE)  # quote yes/no

        parsed = json.loads(raw)
        humanlike_val = str(parsed.get("humanlike", "")).strip().lower()
        if humanlike_val not in {"yes", "no"}:
            return None
        explanation_val = parsed.get("explanation", "")
        if not isinstance(explanation_val, str):
            explanation_val = str(explanation_val)
        return {"humanlike": humanlike_val, "explanation": explanation_val}
    except Exception as e:
        return None


def build_prompt(user: str, assistant: str, *, retry: bool = False) -> str:
    constraints = (
        "You are a strict classifier. Output exactly ONE JSON object and nothing else.\n"
        "Rules:\n"
        "- Use double quotes for all keys and string values.\n"
        "- Keys: humanlike, explanation.\n"
        "- humanlike must be \"yes\" or \"no\".\n"
        "- explanation: max 12 words.\n"
        "- No code fences, no commentary, no extra text before/after the JSON.\n"
        "- No trailing commas.\n\n"
        "Schema:\n{\"humanlike\": \"yes\" or \"no\", \"explanation\": \"brief reason\"}\n\n"
    )

    if retry:
        return (
            constraints
            + f"User: {user.strip()}\n"
            + f"Assistant: {assistant.strip()}\n\n"
            + "ONLY return the JSON object."
        )

    few_shots = (
        "Examples:\n"
        "User: Ohh I am fine with it, I just wanted to know. Thanks\n"
        "Assistant: No wait! ...?\n"
        "Output:\n{\"humanlike\": \"no\", \"explanation\": \"assistant reply is abrupt and unclear\"}\n\n"
        "User: Wait what?\n"
        "Assistant: Got an iPhone 6S. I kinda get it why you like it now 😲\n"
        "Output:\n{\"humanlike\": \"no\", \"explanation\": \"assistant reply ignores user context\"}\n\n"
    )

    task = (
        f"User: {user.strip()}\n"
        f"Assistant: {assistant.strip()}\n\n"
        "Output only the JSON object:"
    )

    return constraints + few_shots + task


def _generate_json_decision(prompt: str) -> Optional[Dict[str, Any]]:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=96,
            do_sample=False,
        )
    completion_tokens = output[0, input_len:]
    completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
    return extract_json_from_response(completion_text)


def llm_filter(user: str, assistant: str) -> bool:
    # First attempt with few-shot prompt
    parsed = _generate_json_decision(build_prompt(user, assistant, retry=False))
    if parsed is None:
        STATS["parse_failures"] += 1
        # Retry with stricter prompt
        parsed = _generate_json_decision(build_prompt(user, assistant, retry=True))
        if parsed is not None:
            STATS["retry_success"] += 1
    if parsed is None:
        # Persist failure context for debugging
        try:
            with open("llm_filter_failures.log", "a", encoding="utf-8") as lf:
                lf.write(
                    json.dumps(
                        {
                            "user": user,
                            "assistant": assistant,
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        return False

    return parsed["humanlike"].lower() == "yes"


def _merge_consecutive_turns(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for msg in raw_messages:
        role = "assistant" if msg["is_from_me"] else "user"
        if merged and merged[-1]["role"] == role:
            merged[-1]["text"] = (merged[-1]["text"] + "\n" + msg["text"]).strip()
            merged[-1]["timestamp_end"] = msg["timestamp"]
        else:
            merged.append(
                {
                    "role": role,
                    "text": msg["text"],
                    "timestamp_start": msg["timestamp"],
                    "timestamp_end": msg["timestamp"],
                }
            )
    return merged


# === Group into user/assistant pairs ===

def build_pairs(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turns = _merge_consecutive_turns(raw_messages)
    dataset: List[Dict[str, Any]] = []
    for i in tqdm(range(1, len(turns)), desc="Filtering", unit="pair"):
        prev_turn = turns[i - 1]
        curr_turn = turns[i]
        if prev_turn["role"] == "user" and curr_turn["role"] == "assistant":
            STATS["pairs_total"] += 1
            user_msg = prev_turn["text"].strip()
            assistant_msg = curr_turn["text"].strip()

            if llm_filter(user_msg, assistant_msg):
                STATS["pairs_kept"] += 1
                dataset.append(
                    {
                        "messages": [
                            {
                                "role": "developer",
                                "content": "You are Viraat. Speak like Viraat: dry, witty, text-like bursts. Respond casually, smartly, often using newlines to separate thoughts.",
                            },
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "channel": "final", "content": assistant_msg},
                        ]
                    }
                )
            else:
                STATS["pairs_rejected"] += 1
    return dataset


# === Process and save ===
print("🔍 Filtering message pairs...")
filtered_dataset = build_pairs(messages)

print(f"💾 Saving {len(filtered_dataset)} examples to {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
    for item in filtered_dataset:
        out_file.write(json.dumps(item) + "\n")

print(
    "Summary -> "
    f"total_pairs={STATS['pairs_total']}, kept={STATS['pairs_kept']}, rejected={STATS['pairs_rejected']}, "
    f"parse_failures={STATS['parse_failures']}, retry_success={STATS['retry_success']}"
)

