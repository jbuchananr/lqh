"""Base-vs-Instruct fine-tuning benchmark.

A standalone, deterministic orchestration of the full pipeline —
datagen -> baseline local eval -> SFT sweep -> eval -> DPO sweep -> eval ->
result table — run across four LFM2.5 variants and multiple tasks, to answer
"which variant is the better base for fine-tuning, and is the trend
consistent across tasks?".

See ``README.md`` for usage. Entry point: ``run.py`` (``python -m
tests.benchmarks.base_vs_instruct.run``).
"""
