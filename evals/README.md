# Diagram evals (DeepEval, no LLM needed)

Six evals for the chat → board diagram feature in `smh-mohsen/chat`
(`diagram.html`, `parseDiagramSteps` + `buildProcessElements`). They measure
the **real code output** — no agent code is touched; `diagram_gen.py` only
reads `diagram.html` and runs its functions in Node.

## The evals

| # | Eval | Question | Metric |
|---|------|----------|--------|
| 1 | Arrow binding | نود را تکان بدهم، فلش هم چسبیده به آن می‌آید؟ | `ArrowBindingMetric` — every arrow between step i and i+1 must carry `startBinding` → rect[i] and `endBinding` → rect[i+1]. Score = fraction of arrows fully bound. |
| 2 | Node overlap | نودها روی هم قرار می‌گیرند؟ | `NodeOverlapMetric` — every pair of rectangles is checked for a positive-area AABB intersection. Score is 1.0 only with zero overlaps. |
| 3 | Faithfulness | نمودار دقیقاً همان چیزی را می‌کشد که خواسته شده؟ | `FaithfulnessMetric` — the drawn steps (read back from the elements) are compared with the expected steps: F1 over matched steps, so an invented step hurts precision and a missing step hurts recall. |
| 4 | Arrow geometry | سر و ته فلش واقعاً به لبه‌ی مربع‌ها چسبیده؟ | `ArrowGeometryMetric` — every arrow must start at the bottom edge of the upper rectangle (+6px), end at the top edge of the lower one (−6px), run vertically through the center, and carry a downward arrowhead. Score = fraction of arrows passing all checks. |
| 5 | Layout order | قدم‌ها به ترتیب، وسط‌چین و با فاصله‌ی یکنواخت‌اند؟ | `LayoutOrderMetric` — rectangles in step order (y strictly increasing), same x and width, equal gaps, title centered above the first rectangle. Score = fraction of checks passed. |
| 6 | Text fit | متن از باکس بیرون نزده؟ | `TextFitMetric` — every step's bound text must lie fully inside its rectangle; the title must fit the 360px column. Score = fraction of texts that fit. |

Cases: evals 1, 2, 4, 5, 6 run on each of `2-steps` (شروع/پایان),
`3-steps-demo`, `7-steps`, `long-desc` (forces text wrap + taller box).
Eval 3 runs on 4 model replies for «نمودار شروع و پایان بکش»:
`correct-2-steps`, `invented-step`, `missing-step`, `wrong-title`.
Total: 24 tests.

## Run (on your own machine)

```bash
cd evals
python3 -m venv .venv
.venv/bin/pip install -r requirements-evals.txt   # takes a few minutes
.venv/bin/pytest test_diagram.py -v
```

No API key is needed — the metrics are deterministic custom `BaseMetric`s,
not LLM judges. Text width in Node is stubbed (`measureText ~= 9px/char`);
the real browser measures with the system font.

## Results on the current code (2026-11-17)

- **Eval 1 FAILS (0/10 arrows bound)** — arrows are created with
  `startBinding: null, endBinding: null`, so dragging a node leaves the
  arrow behind. Fixing it means giving each arrow bindings to its two
  neighbour rectangles (and the rectangles' `boundElements` entries).
- **Eval 2 PASSES (4/4 cases)** — no overlapping nodes in any case.
- **Eval 3 discriminates (1/4 cases)** — the correct reply passes; the
  invented-step reply scores F1 0.80, the wrong-title reply 0.50, and the
  one-step reply draws nothing (score 0).
- **Eval 4 PASSES (4/4 cases)** — every arrow touches both rectangles.
- **Eval 5 PASSES (4/4 cases)** — steps ordered, aligned, evenly spaced.
- **Eval 6 PASSES (4/4 cases)** — every text fits inside its box.
