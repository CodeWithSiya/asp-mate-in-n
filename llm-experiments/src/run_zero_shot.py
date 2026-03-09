"""Run a zero-shot DeepSeek-generation locally via Hugging Face transformers.

Environment variables:
  MATE1_MODEL     – repo id (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
  MATE1_LOCAL_DIR – optional path to locally downloaded weights
"""
from pathlib import Path
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = os.environ.get("MATE1_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
LOCAL_DIR = os.environ.get("MATE1_LOCAL_DIR")
MODEL_SOURCE = LOCAL_DIR if LOCAL_DIR else MODEL_ID

PROMPT_PATH = Path("prompts/zero_shot_template.txt")
OUTPUT_DIR = Path("results/" + MODEL_ID.replace("/", "_"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

prompt = PROMPT_PATH.read_text().strip()
print("Loaded prompt template with", len(prompt.split()), "words")
print(f"Using model: {MODEL_ID}")
if LOCAL_DIR:
    print(f"Loading weights from local dir: {LOCAL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_SOURCE,
    device_map="auto",
    trust_remote_code=True,
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

response = pipe(prompt, max_new_tokens=600, temperature=0.2, do_sample=True)[0][
    "generated_text"
]
output_file = OUTPUT_DIR / "mate_in_one_generated.lp"
output_file.write_text(response)
print("Saved generation to", output_file)
