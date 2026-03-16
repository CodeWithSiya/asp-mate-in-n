# Mate-in-1 ASP Experiments

This workspace captures zero-shot experiments where Anthropic Claude generates Answer Set Programs (ASPs) that identify mate-in-one moves in chess positions. The structure mirrors the SchASPLM pipeline but is trimmed down to the minimal pieces needed for a single-model, zero-shot study.

## Anthropic Setup

1. **Install the SDK**
   ```bash
   pip install anthropic
   ```
2. **Export your API key**
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. *(Optional)* Set a default Claude model
   ```bash
   export CLAUDE_MODEL=claude-sonnet-4-20250514
   ```
   If `CLAUDE_MODEL` is unset the scripts fall back to `claude-sonnet-4-20250514`.

## Layout

- `prompts/zero_shot_template.txt` – baseline zero-shot instruction block.
- `prompts/few_shot/` – three curated few-shot prompts (`version1/2/3.txt`) that mirror the structures you specified.
- `src/run_prompt.py` – Python entry point that calls Claude and understands prompt presets.
- `src/run_few_shot.py` / `src/run_zero_shot.py` – convenience wrappers around the shared logic.
- `results/` – completions organised by model id and prompt label.
- `src/eval_with_clingo.sh` – helper to ground/check the produced ASP.

## Running Claude

1. **Zero-shot via presets**
   ```bash
   python src/run_prompt.py --prompt-type zero-shot
   ```
   This loads `prompts/zero_shot_template.txt` by default. Pass `--prompt-file` to override.
2. **Run all few-shot prompts with one command**
   ```bash
   python src/run_prompt.py --prompt-type few-shot
   ```
   The script cycles through the three templates in `prompts/few_shot/` and writes outputs to `results/<model>/few_shot_<version>/generated.lp`.
3. **Helper scripts (optional)**
   ```bash
   python src/run_zero_shot.py
   python src/run_few_shot.py
   ```
4. **Inspect results**
   ```
   results/<model>/<prompt-label>/generated.lp
   ```
5. **Evaluate with clingo (optional)**
   ```bash
   ./src/eval_with_clingo.sh results/<model>/few_shot_version1/generated.lp data/boards/sample_position.lp
   ```

All scripts share the same flags for `--max-new-tokens`, `--temperature`, and `--model-id`, so you can tweak sampling or override the `CLAUDE_MODEL` environment variable without touching multiple entry points.
