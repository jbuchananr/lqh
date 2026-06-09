# Base vs Instruct — fine-tuning benchmark

Answers: **which LFM2.5 variant is the better base for fine-tuning, and does
the ranking hold across tasks?** For each `(task, model)` it runs the full
pipeline and reports three judge scores — baseline, best-SFT, best-DPO — on a
held-out eval set.

Models (from `BASE_VS_INSTRUCT.md`):

| key              | HuggingFace id                   |
|------------------|----------------------------------|
| `350M-Instruct`  | `LiquidAI/LFM2.5-350M`           |
| `350M-Base`      | `LiquidAI/LFM2.5-350M-Base`      |
| `1.2B-Instruct`  | `LiquidAI/LFM2.5-1.2B-Instruct`  |
| `1.2B-Base`      | `LiquidAI/LFM2.5-1.2B-Base`      |

Tasks: `translation` (EN→DE, format discipline), `extraction` (invite text →
JSON), `classification` (3-class sentiment), `messy_extraction` (noisy support
thread → latest-value JSON), and `style_rewrite` (support reply rewritten to a
tone/style brief). Each is a self-contained datagen pipeline under
`pipelines/` plus a judge rubric in `tasks.py`.

## How it works

- **Compute is local GPU.** Training (`lqh.train.sweep`) and inference
  (`lqh.infer`) run as local subprocesses on this machine's CUDA GPU.
- **Eval is local, not API.** Every reported number comes from one primitive:
  local inference of the model weights → judge scoring via `run_scoring`. The
  judge (`judge:small|medium|large`) is the only API call; the model never
  runs through the LFM router. This is what lets us score the `-Base` variants
  and trained checkpoints, none of which the API serves.
- **The system prompt is baked into the ChatML** at datagen time, so training,
  eval, and scoring all share it.
- **Winner selection** uses the sweep's validated in-training proxy
  (`eval_loss` for SFT, `eval_ce_chosen_mean` for DPO) — no judge needed. The
  script then re-evaluates the winner's `model/` dir with the local-eval
  primitive for the reported score.
- **Self-scoring contract.** A standalone script has no TUI watcher, so
  `run.py` exports `LQH_API_TOKEN` + `LQH_BASE_URL`; the training subprocess
  then self-scores inline (`lqh.train.cloud_score.is_cloud_mode`). This is
  **mandatory for on-policy DPO**, which builds preference pairs from
  judge-scored rollouts every iteration.

## Run it

Authenticate first (`lqh` → `/login`, or set `LQH_API_TOKEN`). Then:

```bash
# Smoke (cheap, validates the whole pipeline end-to-end)
uv run python -m tests.benchmarks.base_vs_instruct.run \
    --tasks translation --models 350M-Instruct,350M-Base \
    --train-size 100 --eval-size 20 --grid-size tiny

# Full run (the spec's 20k/400, all tasks, all models)
uv run python -m tests.benchmarks.base_vs_instruct.run \
    --train-size 20000 --eval-size 400 --grid-size small
```

Outputs land in the workdir (default `~/.lqh-bvi/<run-name>/`):
`datasets/`, `scorers/`, `runs/<task>__<model>__{baseline,sft,sft_eval,dpo,dpo_eval}/`,
and `report/{results.json,report.md}`. The report is rewritten after every
`(task, model)` so a long run is inspectable mid-flight.

### Key flags

- `--train-size` / `--eval-size` — dataset sizes (default 200 / 40 smoke).
- `--grid-size {tiny,small}` — sweep grid (3 vs 6 configs).
- `--skip-dpo` — SFT only.
- `--dpo-train-size` — prompt count for the DPO stage (default 1000, capped at
  `--train-size`). DPO regenerates rollouts on **all** its prompts every
  iteration, so it must stay bounded and decoupled from the (possibly 20k) SFT
  train set — SFT uses the full set; DPO uses a slice. Set to `0` for an
  SFT-only comparison. Positive values must be ≥ 400 or DPO auto-skips.
- `--judge-size {small,medium,large}` — scoring judge (use `large` for the
  final reported run).
- `--no-resume` — recompute everything (default resumes: completed datasets,
  sweeps, and scored evals are reused).
- `--workdir`, `--run-name`, `--sweep-timeout`, `--eval-timeout`.

## Cost & time

Full scale is large: 20k datagen × 5 tasks (≈100k generation calls), then
4 models × 5 tasks × (SFT sweep + DPO sweep + 3 evals). Validate with the smoke
command, then scale up one task/model at a moderate `--train-size` (e.g. 1000)
to gauge wall-time before committing to 20k.

## Known caveats

- **Base-model chat template — verified OK.** SFT and infer call
  `tokenizer.apply_chat_template`. The `-Base` repos ship a
  `chat_template.jinja` (separate-file format), which transformers 5.x loads
  automatically — `LiquidAI/LFM2.5-350M-Base` applies the same `<|im_start|>`
  ChatML template as the instruct variant. No fallback needed on this stack.
  If you pin an older transformers (<4.43) that ignores `chat_template.jinja`,
  inject a template before training, or upgrade.
- **Sweep grid breadth.** The shared `sft_grid_small` (lr 2e-5–1e-4) may be too
  narrow for base variants, which often want different LRs — the whole premise
  of the comparison. The sweep supports `grid_override`; widen per-variant if
  the base models look under-trained.
- **Small-dataset cadence (handled).** Two thresholds in the training stack
  assume larger datasets, and the orchestrator now scales around both:
  - SFT evals/saves on a *step* schedule (default 50). `run.py` sizes
    `eval_steps`/`save_steps` to the dataset so the sweep's `eval_loss` proxy
    actually gets logged (otherwise every config is marked "failed").
  - The DPO sweep proxy (`eval_ce_chosen_mean`) is read from **iter_000**'s
    held-out split of the per-iteration preference pairs, and `split_train_eval`
    has a hard floor of 10 eval examples. On-policy preference yield is only
    ~0.13–0.37 pairs per train prompt per iter — and the **stronger the SFT
    model, the fewer pairs** it produces (a near-perfect greedy rollout rarely
    disagrees with the gold; the 1.2B-Instruct SFT yielded only ~13 pairs from
    100 prompts). So a small train set cannot produce 10 held-out pairs at any
    split ratio. `run.py` therefore **auto-skips DPO when `--train-size < 400`**
    (`_DPO_MIN_TRAIN_SIZE`), logging a warning and noting it in the report, and
    sizes `eval_split_ratio` off a conservative 0.13 yield when it does run.
    **For a DPO smoke use `--train-size 400`+**; the fastest SFT-only smoke is
    `--train-size 100 --dpo-train-size 0` (or `--skip-dpo`, or just let it
    auto-skip). DPO's real signal needs `--train-size` in the thousands, where
    pairs are plentiful.
