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
