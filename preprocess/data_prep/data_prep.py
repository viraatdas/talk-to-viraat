import csv
import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Config ===
CSV_PATH = "messages.csv"
OUTPUT_PATH = "filtered_finetune.jsonl"
MODEL_NAME = "openai/gpt-oss-20b"
BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "8"))  # heuristic for H100 + 20B
MAX_NEW_TOKENS = 96

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

        # Normalize common mistakes
        raw = re.sub(r"\{\s*humanlike\s*:\s*", '{"humanlike": ', raw)
        raw = re.sub(r"\,\s*explanation\s*:\s*", ', "explanation": ', raw)
        raw = re.sub(
            r'(\"humanlike\"\s*:\s*)(yes|no)([\s,}])',
            r'\1"\2"\3',
            raw,
            flags=re.IGNORECASE,
        )

        parsed = json.loads(raw)
        humanlike_val = str(parsed.get("humanlike", "")).strip().lower()
        if humanlike_val not in {"yes", "no"}:
            return None
        explanation_val = parsed.get("explanation", "")
        if not isinstance(explanation_val, str):
            explanation_val = str(explanation_val)
        return {"humanlike": humanlike_val, "explanation": explanation_val}
    except Exception:
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


def _generate_batch(prompts: List[str]) -> List[Optional[Dict[str, Any]]]:
    if not prompts:
        return []
    try_batch = len(prompts)
    # We may need to split into chunks if prompts > BATCH_SIZE
    results: List[Optional[Dict[str, Any]]] = []
    for start in range(0, len(prompts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(prompts))
        sub_prompts = prompts[start:end]
        # Tokenize
        inputs = tokenizer(sub_prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
        input_lens = inputs.attention_mask.sum(dim=1)
        # Generate
        with torch.inference_mode():
            try:
                output = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            except torch.cuda.OutOfMemoryError:
                # Fallback: run each in single-item batches to make progress
                torch.cuda.empty_cache()
                for i in range(len(sub_prompts)):
                    mini_inputs = {k: v[i:i+1] for k, v in inputs.items()}
                    mini_len = int(input_lens[i].item())
                    out = model.generate(
                        mini_inputs["input_ids"],
                        attention_mask=mini_inputs["attention_mask"],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                    tokens = out[0, mini_len:]
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    results.append(extract_json_from_response(text))
                continue
        # Decode per sample
        for i in range(end - start):
            in_len = int(input_lens[i].item())
            tokens = output[i, in_len:]
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            results.append(extract_json_from_response(text))
    return results


def llm_filter_batch(users: List[str], assistants: List[str]) -> List[bool]:
    # First pass
    prompts = [build_prompt(u, a, retry=False) for u, a in zip(users, assistants)]
    parsed = _generate_batch(prompts)
    decisions: List[Optional[bool]] = [None if p is None else p["humanlike"].lower() == "yes" for p in parsed]
    # Retry only failed parses
    failed_indices = [i for i, d in enumerate(decisions) if d is None]
    if failed_indices:
        STATS["parse_failures"] += len(failed_indices)
        retry_prompts = [build_prompt(users[i], assistants[i], retry=True) for i in failed_indices]
        retry_parsed = _generate_batch(retry_prompts)
        for idx, rp in zip(failed_indices, retry_parsed):
            if rp is not None:
                STATS["retry_success"] += 1
                decisions[idx] = rp["humanlike"].lower() == "yes"
            else:
                decisions[idx] = False
                # Log failure context
                try:
                    with open("llm_filter_failures.log", "a", encoding="utf-8") as lf:
                        lf.write(json.dumps({"user": users[idx], "assistant": assistants[idx]}) + "\n")
                except Exception:
                    pass
    # Convert Optional[bool] to bool (safe now)
    return [bool(d) for d in decisions]


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
    # Build candidate pairs list
    pairs: List[Tuple[str, str]] = []
    for i in range(1, len(turns)):
        prev_turn = turns[i - 1]
        curr_turn = turns[i]
        if prev_turn["role"] == "user" and curr_turn["role"] == "assistant":
            pairs.append((prev_turn["text"].strip(), curr_turn["text"].strip()))

    STATS["pairs_total"] = len(pairs)
    dataset: List[Dict[str, Any]] = []

    # Batched filtering
    with tqdm(total=len(pairs), desc="Filtering", unit="pair") as pbar:
        for start in range(0, len(pairs), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(pairs))
            batch_users = [pairs[i][0] for i in range(start, end)]
            batch_assistants = [pairs[i][1] for i in range(start, end)]
            keep_mask = llm_filter_batch(batch_users, batch_assistants)
            for u, a, keep in zip(batch_users, batch_assistants, keep_mask):
                if keep:
                    STATS["pairs_kept"] += 1
                    dataset.append(
                        {
                            "messages": [
                                {
                                    "role": "developer",
                                    "content": "You are Viraat. Speak like Viraat: dry, witty, text-like bursts. Respond casually, smartly, often using newlines to separate thoughts.",
                                },
                                {"role": "user", "content": u},
                                {"role": "assistant", "channel": "final", "content": a},
                            ]
                        }
                    )
                else:
                    STATS["pairs_rejected"] += 1
            pbar.update(end - start)
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

