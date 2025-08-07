import os
import json
from bs4 import BeautifulSoup

# === CONFIG ===
BASE_DIR = "/Users/viraat/Documents/talk-to-viraat/iMessage-Export/messages"  # <-- change to your directory
OUTPUT_FILE = "viraat_finetune.jsonl"

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    messages = []
    for div in soup.find_all("div", class_="h-entry"):
        text = div.find("span", class_="e-content p-name")
        if text and text.text.strip():
            messages.append(text.text.strip())
    return messages

def normalize_newlines(text):
    return text

def convert_to_samples(messages):
    samples = []
    for raw in messages:
        normalized = normalize_newlines(raw)
        if not normalized.strip():
            continue

        sample = {
            "messages": [
                {
                    "role": "developer",
                    
                    "content": "You are Viraat. Speak like Viraat: dry, witty, text-like bursts. Respond casually, smartly, often using newlines to separate thoughts."
                },
                {
                    "role": "user",
                    "content": "Respond to this naturally."
                },
                {
                    "role": "assistant",
                    "channel": "final",
                    "content": normalized
                }
            ]
        }
        samples.append(sample)
    return samples

def parse_all_html(base_dir):
    all_samples = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".html"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    html = f.read()
                messages = extract_text_from_html(html)
                samples = convert_to_samples(messages)
                all_samples.extend(samples)
            except Exception as e:
                print(f"❌ Error parsing {path}: {e}")
    return all_samples

def save_jsonl(samples, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    dataset = parse_all_html(BASE_DIR)
    save_jsonl(dataset, OUTPUT_FILE)
    print(f"✅ Saved {len(dataset)} samples to {OUTPUT_FILE}")

