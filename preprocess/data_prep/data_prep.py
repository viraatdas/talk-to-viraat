import csv
import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime

# === Config ===
CSV_PATH = "messages.csv"
OUTPUT_PATH = "filtered_finetune.jsonl"
URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)

# === Simple URL exclusion only ===

# === Stats ===
STATS: Dict[str, int] = {
    "pairs_total": 0,
    "pairs_kept": 0,
    "pairs_rejected": 0,
    "pairs_url_excluded": 0,
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

# (All LLM-based filtering removed)


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
    with tqdm(total=len(pairs), desc="Filtering", unit="pair") as pbar:
        for u, a in pairs:
            # Exclude if either side contains a URL
            if URL_RE.search(u) or URL_RE.search(a):
                STATS["pairs_rejected"] += 1
                STATS["pairs_url_excluded"] += 1
                pbar.update(1)
                continue

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
            pbar.update(1)
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
    f"url_excluded={STATS['pairs_url_excluded']}"
)

