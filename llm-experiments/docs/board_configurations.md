# Board Configuration Workflow

This project expects every chess test case to be described twice:

1. As a human friendly Forsyth–Edwards Notation (FEN) string (stored under `data/boards/*.fen`).
2. As ASP-ready facts (`piece/4`, `to_move/1`, optional `king_in_check/1`) that can be passed directly to Clingo.

Use the checklist below whenever you add new positions.

## 1. Choose high-quality positions

- **Start with vetted sources.** Pull puzzles from lichess.org, Chess.com, or endgame tablebases so you already know the ground truth move(s).  
- **Prefer variety.** Capture different piece configurations (open king, pawn storms, opposite-colored bishops, etc.) so the ASP solver does not overfit to one motif.  
- **Balance colors.** Ensure both White-to-move and Black-to-move positions are represented.  
- **Record metadata.** Keep short comments about the expected mating move(s) or puzzle origin directly below the FEN string.

> Tip: When you import from PGN, jump to the target ply in a GUI (e.g., lichess analysis), copy the displayed FEN, and paste it into a new `.fen` file.

## 2. Store the raw FEN

Create a new file in `data/boards/` that ends with `.fen`. Place the FEN string on the first line and optionally add `#` comments with context:

```
4r1k1/pp3ppp/2p2n2/3p4/3P4/2PB1N2/PP3PPP/4R1K1 w - - 0 1
# Expected mates: Re8#
# Source: Lichess Puzzle 12345
```

## 3. Convert FEN to ASP facts

Use `python src/fen_to_board_lp.py` to translate the FEN into an `.lp` fact file:

```bash
python src/fen_to_board_lp.py \
  --fen-file data/boards/sample_mate_in_one.fen \
  --output data/boards/sample_mate_in_one.lp
```

Key behaviors:

- Produces `piece(Color, Type, File, Rank).` facts (files are `a`–`h`, ranks `1`–`8`).
- Adds `to_move(Color).` based on the FEN side-to-move field.
- Use `--king-in-check white|black` if you also want `king_in_check/1`.
- Accepts `--fen` directly so you can script batch conversions without writing intermediate files.

## 4. Test the configuration

Once the `.lp` exists, run it through Clingo with either a hand-written reference solution or an LLM-generated program:

```bash
python src/run_clingo.py \
  results/<model>/<prompt-label>/generated.lp \
  --board data/boards/sample_mate_in_one.lp \
  --stats
```

Check that the resulting stable models contain the expected `mate_move/5` atoms. If not, either the puzzle or the generated ASP needs attention.

## 5. Track provenance

Maintain a simple table (spreadsheet or markdown) that links the `.fen`, `.lp`, expected mates, and citation/source. This makes it trivial to rerun experiments or regenerate facts later.
