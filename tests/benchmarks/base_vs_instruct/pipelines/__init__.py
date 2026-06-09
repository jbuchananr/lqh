"""Data-generation pipelines for the base-vs-instruct benchmark.

Each module is **self-contained** on purpose: the engine loads them by file
path via ``importlib`` (``lqh.engine.load_pipeline``), executing them as a
standalone synthetic module with no package context. So a pipeline file may
import only from ``lqh.pipeline`` / stdlib / installed deps, and must define
exactly one concrete ``Pipeline`` subclass.

Each module also exposes a module-level ``SYSTEM_PROMPT`` constant — the
instruction baked as the first message of every generated conversation.
``tasks.py`` imports these normally (as package modules) to keep the system
prompt the single source of truth, even though the engine loads the same
file by path.
"""
