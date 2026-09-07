# judge_2026-08-31

`judge_item_bank.csv` is the item bank behind *Three Ways Classical Test Theory Misleads for LLM
Judges* (Zhu, 2026): 210 items, of which 180 carry judge verdicts, each scored on K = 10 rubric
elements by gold annotation (`gold_e1..e10`) and by one LLM judge (`judge_e1..e10`), with the
question id, the answer text and the element texts. `sweep_grid.json` carries the simulation values
as published in the paper (the two-way KR-20 grid and the 60-replicate run at the measured 4.72 %
error rate). Both are copied unchanged from https://github.com/louisyzhu/llm-judge-reliability at
commit `126d1bce` and are released by their author under CC BY 4.0. `MANIFEST.json` records their
SHA-256; `benchprobe.io.load_snapshot("judge_2026-08-31")` verifies it on every load.
