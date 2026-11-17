# Diagram evals (DeepEval, no LLM needed)

Four evals for the chat → board diagram feature in `smh-mohsen/chat`
(`diagram.html`, `buildProcessElements` + scene helpers). They measure the
**real code output** — no agent code is touched; `diagram_gen.py` only
reads `diagram.html` and runs its functions in Node.

## The evals

| # | Eval | Question | Metric |
|---|------|----------|--------|
| 1 | Arrow binding | نود را تکان بدهم، فلش هم چسبیده به آن می‌آید؟ | `ArrowBindingMetric` — every arrow between step i and i+1 must carry `startBinding` → rect[i] and `endBinding` → rect[i+1]. Score = fraction of arrows fully bound. |
| 2 | Node overlap | نودها روی هم قرار می‌گیرند؟ | `NodeOverlapMetric` — every pair of rectangles is checked for a positive-area AABB intersection. Score is 1.0 only with zero overlaps. |
| 3 | Text visibility | متنِ کشیده‌شده فوراً دیده می‌شود، بدون رفرش؟ | `TextVisibilityMetric` — every text must have `lineHeight == 1.25`, `autoResize == true`, finite width/height > 0, and no stale `baseline` field. Score = fraction of valid texts. |
| 4 | Scene isolation | هر گفتگو بومِ خودش را دارد؟ | `SceneIsolationMetric` — the real scene helpers are exercised in Node: legacy scene migrates to the active chat, a new chat starts empty, writing one chat's scene leaves the other's untouched. Score = fraction of checks true. |

Cases: evals 1–3 run on each of `2-steps` (شروع/پایان), `3-steps-demo`,
`7-steps`, `long-desc` (forces text wrap + taller box). Eval 4 runs on
`with-legacy-scene` and `fresh-start`.

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

## Results on the current code (2026-11-17, PR #4 branch)

- **Eval 1 FAILS (0/10 arrows bound)** — arrows are created with
  `startBinding: null, endBinding: null`, so dragging a node leaves the
  arrow behind. Fixing it means giving each arrow bindings to its two
  neighbour rectangles (and the rectangles' `boundElements` entries).
- **Eval 2 PASSES (4/4 cases)** — no overlapping nodes in any case.
- **Eval 3 PASSES (4/4 cases)** — all texts have lineHeight 1.25,
  autoResize true, finite dimensions, no baseline.
- **Eval 4 PASSES (2/2 scenarios)** — legacy scene migrates, new chats
  start empty, per-chat scenes stay separate.
