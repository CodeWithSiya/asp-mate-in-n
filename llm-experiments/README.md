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
- `src/fen_to_board_lp.py` – converts a FEN string/file into ASP facts.
- `docs/board_configurations.md` – end-to-end checklist for curating chess positions.
- `src/run_clingo.py` – Python wrapper to invoke Clingo with board/fact files.

## Board configurations

Follow the workflow in `docs/board_configurations.md` whenever you add a new puzzle. In short:

1. Save the raw FEN under `data/boards/<name>.fen` (first line = FEN, subsequent `#` lines = metadata).
2. Translate it into ASP facts:
   ```bash
   python src/fen_to_board_lp.py \
     --fen-file data/boards/sample_mate_in_one.fen \
     --output data/boards/sample_mate_in_one.lp
   ```
3. Feed that `.lp` file to either an LLM-generated solver or a handwritten baseline via `src/run_clingo.py`.

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

## Installing Clingo

Pick whichever package manager matches your environment:

- **macOS (Homebrew)**
  ```bash
  brew install clingo
  ```
- **Linux (conda-forge)**
  ```bash
  conda install -c conda-forge clingo
  ```
- **Python-only workflow**
  ```bash
  pip install clingo
  ```

After installation, verify everything is on the path:

```bash
clingo --version
```

You can optionally export `CLINGO_BIN=/full/path/to/clingo` if the binary lives outside your default `PATH`.

## Running Clingo from Python

Use the wrapper when you want a reproducible command that mixes generated programs, board facts, and optional gold constraints:

```bash
python src/run_clingo.py \
  results/<model>/<prompt-label>/generated.lp \
  --board data/boards/sample_mate_in_one.lp \
  --facts gold/mate_in_one.lp \
  --stats
```

The script assembles a `clingo` command with sane defaults (`--models 0`) and streams the solver output. Pass additional flags directly to Clingo by appending `-- <extra args>`, for example `-- --time-limit=60`.
