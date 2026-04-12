# Mate-in-1 ASP Experiments

This workspace houses a trimmed-down reproduction of the SchASPLM workflow for
studying whether Anthropic Claude can synthesize Answer Set Programs (ASPs) that
detect mate-in-one moves from curated base fragments derived from our manual
encodings. Each base fragment already contains the board state, the reusable
move-generation predicates, and the side-to-move indicator; the strategies
prompt Claude to *complete* the reasoning layer rather than regenerate chess
fundamentals. The
generated programs are solved with Clingo and scored via automated semantic
validation.

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

- `data/base/*_base.lp` – curated base fragments for each manual puzzle. Every
  file contains the board state facts (`placement/5`), the rules that derive
  legal moves (`legal_move/5`), plus `to_move/1`. These fragments are the
  context we hand to Claude so it can focus solely on the mate-in-one reasoning
  layer (they no longer enumerate `king_can_go/2` or other escape lists).
- `outputs/<board-id>/<strategy>/` – artefacts for every run (generated ASP,
  CNL/CoT traces when applicable, solver logs, metadata). Created by the
  strategy runners and by `src/tools/run_experiment.py`.
- `results/semantic/` – JSON verdicts plus `summary.md`, updated automatically
  whenever a strategy finishes semantic validation.
- `prompts/` – editable prompt templates for zero-shot, few-shot, and
  chain-of-thought strategies.
- `src/tools/run_experiment.py` – orchestrator that runs zero-shot, few-shot,
  chain-of-thought, and pipeline strategies on the same board set.
- `src/strategies/*.py` – strategy helpers consumed by
  `src/tools/run_experiment.py` plus the configurable pipeline script (which
  remains invokable on its own).
- `src/tools/run_clingo.py` – CLI wrapper to run Clingo on a self-contained ASP
  artifact (already includes its base fragment).

## Preparing base fragments

Manual encodings live in the sibling repository `../encoding/src/`. For every
puzzle you plan to evaluate, extract the static board facts into
`data/base/<board>_base.lp`, and make sure the shared helper predicates still
match the pieces present in that position. Those `.lp` fragments are treated as
immutable context and are embedded verbatim into every LLM prompt.

When adding a new puzzle:

1. Copy `data/base/mate_in_one_easy_base.lp` (or any existing base file) as a
   template.
2. Update the `placement/5` facts using the manual encoding.
3. Update/extend the `legal_move/5` helper rules if the new puzzle introduces
   pieces or movement patterns not already covered.
4. Commit the finished fragment so strategy runners can mount it when invoking
   Claude.

## Running Anthropic strategies

### Run every strategy in one pass

```bash
python src/tools/run_experiment.py data/base/mate_in_one_easy_base.lp \
  data/base/mate_in_one_medium_base.lp \
  --output-dir outputs \
  --strategies zero_shot few_shot chain_of_thought pipeline
```

Key flags:

- `--strategies …` to pick a subset (default runs all four).
- `--output-dir` changes where board folders are created (default: `outputs`).
- `--few-shot-prompt`, `--zero-shot-prompt`, `--cot-prompt`, `--asp-prompt` let
  you swap prompt files without editing the code.
- `--max-new-tokens`, `--temperature`, and `--model-id` are forwarded to every
  LLM call. If `--model-id` is omitted the resolved model is printed.
- `--clingo-retries N` controls how many times the pipeline will regenerate an
  ASP when the solver fails (defaults to two retries, i.e., up to three
  attempts per board).

The orchestrator automatically runs Clingo and semantic validation for every
strategy, so each board winds up with `asp.lp`, `clingo_models.txt`, and
`metadata.json` under `outputs/<board>/<strategy>/`.

### Run a single strategy

Use `--strategies` with `run_experiment.py` to focus on one variant at a time.

- **Zero-shot** (base fragment → ASP in one hop):
  ```bash
  python src/tools/run_experiment.py data/base/mate_in_one_easy_base.lp \
    --strategies zero_shot
  ```
  Claude receives the base fragment verbatim and must return the full program
  (fragment copied first, reasoning appended afterwards).
- **Few-shot** (few exemplars + base fragment + CNL summary):
  ```bash
  python src/tools/run_experiment.py data/base/*.lp \
    --strategies few_shot --few-shot-prompt prompts/few_shot/prompt.txt
  ```
  The prompt template appends both the CNL summary and the literal base
  fragment before asking Claude to extend it.
- **Chain of thought** (base fragment → CoT reasoning → ASP):
  ```bash
  python src/tools/run_experiment.py data/base/mate_in_one_medium_base.lp \
    --strategies chain_of_thought --cot-prompt prompts/chain-of-thought/chess-cot.txt \
    --asp-prompt prompts/chain-of-thought/chess-asp.txt
  ```
  Stage 1 produces a natural-language reasoning trace directly from the base
  fragment; Stage 2 copies the fragment and emits the completed ASP while
  consulting the CoT.
- **Pipeline** (configurable multi-stage flow):
  ```bash
  python src/tools/run_experiment.py data/base/mate_in_one_easy_base.lp \
    --strategies pipeline --clingo-retries 2
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

Use the helper when you want to sanity-check a generated ASP locally (each ASP
already embeds its base fragment):

```bash
python src/tools/run_clingo.py \
  outputs/mate_in_one_easy/zero_shot/asp.lp \
  --output outputs/mate_in_one_easy/zero_shot/clingo_manual.txt
```

The script simply feeds the file into the shared `run_clingo_program` helper and
writes a transcript containing models, statistics, and solver logs.
