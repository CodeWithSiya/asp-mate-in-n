# Mate-in-1 ASP Experiments

This workspace houses a trimmed-down reproduction of the SchASPLM workflow for
studying whether Anthropic Claude can synthesize Answer Set Programs (ASPs) that
detect mate-in-one moves from Forsyth–Edwards Notation (FEN) boards. The scripts
convert FENs into ASP facts, run several prompting strategies, solve the
resulting programs with Clingo, and record semantic validation results.

## Anthropic Setup

1. **Install the Python deps** (needed by every runner)
   ```bash
   python -m pip install anthropic clingo
   ```
2. **Export your Claude key**
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. **(Optional) Set a default model**
   ```bash
   export CLAUDE_MODEL=claude-sonnet-4-6
   ```
   Each CLI also accepts `--model-id`; if omitted it falls back to
   `$CLAUDE_MODEL` or `claude-sonnet-4-6`.

## Layout

- `data/boards/*.fen` – curated mate-in-one boards. The first non-comment line
  must be the raw FEN; subsequent lines starting with `#` capture metadata such
  as the expected mate.
- `outputs/<board-id>/<strategy>/` – artefacts for every run (generated ASP,
  CNL/CoT traces when applicable, solver logs, metadata). Created by the
  strategy runners and by `src/tools/run_experiment.py`.
- `results/semantic/` – JSON verdicts plus `summary.md`, updated automatically
  whenever a strategy finishes semantic validation.
- `prompts/` – editable prompt templates for zero-shot, few-shot, chain of
  thought, and the shared `syntax_guardrail.txt` snippet.
- `src/tools/run_experiment.py` – orchestrator that runs zero-shot, few-shot,
  chain-of-thought, and pipeline strategies on the same board set.
- `src/strategies/*.py` – strategy helpers consumed by
  `src/tools/run_experiment.py` plus the configurable pipeline script (which
  remains invokable on its own).
- `src/tools/run_clingo.py` – CLI wrapper that converts a FEN to facts on the
  fly before invoking the `clingo` binary.
- `src/utils/fen_to_board_lp.py` – utility to inspect the facts produced for a
  board (also used internally by the runners).

## Preparing board configurations

1. Drop a new FEN file under `data/boards/`. Example:
   ```
   6k1/5ppp/8/8/8/8/5PPP/6KQ w - - 0 1
   # Expected mates: Qh7#
   ```
2. Run `python src/utils/fen_to_board_lp.py --fen-file data/boards/<name>.fen \
   --output data/boards/board_conversion.lp` whenever you want to inspect the
   derived ASP facts manually. The helper overwrites `board_conversion.lp` every
   time, so there is never a pile-up of `.lp` fact files.
3. Every CLI accepts either individual `.fen` files or directories; internally
   `collect_board_specs` walks the directories, deduplicates files, parses the
   comments for `Expected` metadata, and exposes the facts plus a controlled
   natural-language (CNL) description to downstream strategies.

## Running Anthropic strategies

### Run every strategy in one pass

```bash
python src/tools/run_experiment.py data/boards/sample_mate_in_one.fen \
  data/boards/mate_kf6_qd7.fen \
  --output-dir outputs \
  --syntax-guardrail \
  --strategies zero_shot few_shot chain_of_thought pipeline
```

Key flags:

- `--strategies …` to pick a subset (default runs all four).
- `--output-dir` changes where board folders are created (default: `outputs`).
- `--few-shot-prompt`, `--zero-shot-prompt`, `--cot-prompt`, `--asp-prompt` let
  you swap prompt files without editing the code.
- `--max-new-tokens`, `--temperature`, and `--model-id` are forwarded to every
  LLM call. If `--model-id` is omitted the resolved model is printed.
- `--syntax-guardrail` appends `prompts/syntax_guardrail.txt` to every ASP
  generation step (all strategy runners support the same flag).
- `--clingo-retries N` controls how many times the pipeline will regenerate an
  ASP when the solver fails (defaults to two retries, i.e., up to three
  attempts per board).

The orchestrator automatically runs Clingo and semantic validation for every
strategy, so each board winds up with `asp.lp`, `clingo_models.txt`, and
`metadata.json` under `outputs/<board>/<strategy>/`.

### Run a single strategy

Use `--strategies` with `run_experiment.py` to focus on one variant at a time.

- **Zero-shot** (FEN → ASP):
  ```bash
  python src/tools/run_experiment.py data/boards/mate_kf6_qd7.fen \
    --strategies zero_shot --syntax-guardrail
  ```
  Zero-shot deliberately operates on raw FEN strings to keep a baseline that
  skips all intermediate representations.
- **Few-shot** (facts + optional CNL summary → ASP):
  ```bash
  python src/tools/run_experiment.py data/boards/*.fen \
    --strategies few_shot --few-shot-prompt prompts/few_shot/prompt.txt
  ```
  The prompt template still embeds both the ASP facts and a short CNL summary.
- **Chain of thought** (ASP facts → CoT reasoning → ASP):
  ```bash
  python src/tools/run_experiment.py data/boards/mate_kf6_qd7.fen \
    --strategies chain_of_thought --cot-prompt prompts/chain-of-thought/chess-cot.txt \
    --asp-prompt prompts/chain-of-thought/chess-asp.txt
  ```
  The standalone CoT runner now reasons directly over the fact list; the CNL
  stage lives exclusively inside the pipeline variant.
- **Pipeline** (configurable multi-stage flow):
  ```bash
  python src/tools/run_experiment.py data/boards/mate_kf6_qd7.fen \
    --strategies pipeline --clingo-retries 2 --syntax-guardrail
  ```
  Supported variants remain `zero_shot`, `cnl_only`, `cot_only`, `cnl_cot`
  (orchestrator default). The pipeline still performs the full CNL → CoT → ASP
  path (plus retries) and always runs Clingo plus semantic validation.

### Outputs and semantic scoring

- `outputs/<board>/<strategy>/asp.lp` – the latest ASP program.
- `clingo_models.txt` – solver transcript with models/statistics from the final
  attempt. Intermediate attempts (pipeline) are retained as
  `clingo_models_attempt*.txt`.
- `cnl.txt` / `cot.txt` – emitted when a strategy reasons through those stages.
- `metadata.json` – run metadata, token usage, solver diagnostics, and semantic
  scores.
- `results/semantic/<strategy>/<board>.json` – structured verdict logged by
  `semantic_validate`. The rendered Markdown table lives in
  `results/semantic/summary.md`.

## Installing Clingo

Install the command-line solver with whichever package manager you prefer:

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
  python -m pip install clingo
  ```

After installation, confirm the executable is discoverable (or set
`CLINGO_BIN`):

```bash
clingo --version
```

## Running Clingo from Python

Use the wrapper to combine a generated ASP with a board on the fly:

```bash
python src/tools/run_clingo.py \
  outputs/mate_kf6_qd7/few_shot/asp.lp \
  --fen-board data/boards/mate_kf6_qd7.fen \
  --output outputs/mate_kf6_qd7/few_shot/clingo_manual.txt
```

The script converts the FEN into `data/boards/board_conversion.lp` (overwriting
it each time), feeds the resulting facts plus the ASP file into the same
`run_clingo_program` helper used by every strategy, and writes a
`clingo_manual_run.txt` (or the provided `--output` path) containing stdout,
statistics, and solver verdicts.

Because the tool calls the Python API directly, it picks up the same logging
format and guardrails as the batched runners—no extra solver flags needed.
