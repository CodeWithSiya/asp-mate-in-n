# Mate-in-1 ASP Experiments

This workspace captures zero-shot experiments where open-weight LLMs generate Answer Set Programs (ASPs) that identify mate-in-one moves in chess positions. The structure mirrors the SchASPLM pipeline but is trimmed down to the minimal pieces needed for a single-model, zero-shot study.

## Local DeepSeek Setup (Hugging Face)

1. **Install dependencies**
   ```bash
   pip install transformers accelerate bitsandbytes huggingface-hub
   ```
2. **Authenticate with Hugging Face**
   ```bash
   hf auth login
   ```
3. **Download the checkpoint once (example: DeepSeek-R1 Distill Qwen 7B)**

   ```bash
   cd /Users/nhliziyosibiya/Library/CloudStorage/OneDrive-UniversityofCapeTown/CSC4042Z\ ASP
   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
   --repo-type model \
   --include "*" \
   --local-dir models/deepseek-r1-qwen7b
   ```

   Replace the repo id if you prefer another DeepSeek variant. The codebase will look for the folder referenced above.

4. **Set environment variables before running the scripts**
   ```bash
   export MATE1_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
   export MATE1_LOCAL_DIR=models/deepseek-r1-qwen7b
   ```

## Layout

- `prompts/` – reusable prompt templates and instructions given to the LLM.
- `data/boards/` – FEN or JSON descriptions of board states plus metadata (whose turn, expected mates).
- `results/` – raw LLM generations and solver logs, organised by date/model.
- `scripts/` – helper utilities to run prompts, call the LLM, and evaluate with clingo.

## Minimal Zero-Shot Workflow

1. **Select a board instance** from `data/boards/*.fen`.
2. **Fill the zero-shot prompt** in `prompts/zero_shot_template.txt`, inserting the board description and task instructions.
3. **Query the chosen open-source LLM** (e.g., via Hugging Face `transformers` pipeline) and save the raw ASP output under `results/<date>/<model>/mate_in_one.lp`.
4. **Run clingo** using `scripts/eval_with_clingo.sh` to check syntax/semantics against the reference facts in the manual encoding.
5. **Log findings** inside `results/<date>/<model>/notes.md`, noting whether the generated program produced the expected mate-move answer sets.
