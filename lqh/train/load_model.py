"""Unified model loading for HF Hub IDs, merged dirs, and PEFT adapter dirs.

A "model path" in lqh configs (``base_model`` in sft / dpo / infer) can now
be any of three things, and downstream consumers should not care which:

- ``"hub"``     — a HF Hub id like ``"LiquidAI/LFM2-1.2B"`` (no local dir).
- ``"merged"``  — a local dir containing ``config.json`` + weights.
- ``"adapter"`` — a local dir containing ``adapter_config.json`` and
                  adapter weights (typically ``adapter_model.safetensors``).

This module classifies a path and dispatches to the right ``from_pretrained``
chain. Adapter dirs are produced by :mod:`lqh.train.sft` when
``lora.merge=False`` — the artifact is ~tens of MB instead of multi-GB
merged model, which sidesteps publish-time tar OOMs on resource-bounded
sandboxes.

Backwards compatibility: anything that worked before (hub id, merged dir)
keeps working with no config change. Detection is automatic.

Reference-model gotcha (for DPO): when starting from an adapter dir, both
the policy and the reference model must be loaded through
:func:`load_for_training` so they share the same effective starting point
(base + pre-existing adapter merged in). Loading the reference from a
bare base id while the policy starts from an adapter dir produces wrong
KL — the older DPO code path silently did this.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

ModelKind = Literal["hub", "merged", "adapter"]

__all__ = [
    "ModelKind",
    "detect_kind",
    "resolve_base_model",
    "load_for_inference",
    "load_for_training",
]


def detect_kind(path_or_id: str) -> ModelKind:
    """Classify a model reference as hub / merged / adapter.

    Anything that isn't an existing directory is treated as a hub id
    (the caller will get a clean HF download error if the id is bogus).
    """
    p = Path(path_or_id)
    if not p.exists() or not p.is_dir():
        return "hub"
    if (p / "adapter_config.json").is_file():
        return "adapter"
    if (p / "config.json").is_file():
        return "merged"
    # A dir without either — most likely an empty / corrupted save.
    # Fall back to "merged" so the AutoModel path raises a clear
    # FileNotFoundError on its own terms instead of us masking it.
    return "merged"


def resolve_base_model(adapter_dir: str, override: str | None = None) -> str:
    """Find the base model for an adapter dir.

    ``override`` wins when set (lets callers pin a base even when the
    adapter was trained against a hub id that's since moved). Otherwise
    reads ``adapter_config.json["base_model_name_or_path"]``.

    Raises ``ValueError`` with a clear message if neither is available.
    """
    if override:
        return override
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.is_file():
        raise ValueError(
            f"{adapter_dir} is not an adapter dir (no adapter_config.json); "
            f"cannot resolve base model. Pass base_override= explicitly."
        )
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{cfg_path}: invalid JSON: {exc}") from exc
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"{cfg_path} has no 'base_model_name_or_path'; "
            f"pass base_override= explicitly."
        )
    return str(base)


def _load_tokenizer(primary: str, fallback: str) -> "PreTrainedTokenizerBase":
    """Try the primary location first; fall back to the secondary on
    failure. Adapter dirs from SFT include the tokenizer, but some PEFT
    adapter dirs in the wild don't — the base ships its own."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(primary)
    except (OSError, ValueError) as exc:
        if primary == fallback:
            raise
        logger.debug("tokenizer load from %s failed (%s); falling back to %s",
                     primary, exc, fallback)
        return AutoTokenizer.from_pretrained(fallback)


def load_for_inference(
    path_or_id: str,
    *,
    dtype: "torch.dtype | None" = None,
    device_map: "str | dict | None" = "auto",
    base_override: str | None = None,
) -> "tuple[PreTrainedModel, PreTrainedTokenizerBase]":
    """Return a ready-to-infer model + tokenizer.

    For hub / merged: a single ``AutoModelForCausalLM.from_pretrained`` call.
    For adapter: load the base, wrap with ``PeftModel``, then
    ``merge_and_unload`` — the merge is transient (no disk write) since
    the model is already in memory.
    """
    import torch  # local: keep import-time light
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.bfloat16

    kind = detect_kind(path_or_id)
    if kind in ("hub", "merged"):
        model = AutoModelForCausalLM.from_pretrained(
            path_or_id, dtype=dtype, device_map=device_map,
        )
        tokenizer = _load_tokenizer(path_or_id, path_or_id)
        return model, tokenizer

    # adapter
    from peft import PeftModel

    base = resolve_base_model(path_or_id, base_override)
    logger.info("load_for_inference: adapter %s on base %s (transient merge)",
                path_or_id, base)
    base_model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, device_map=device_map,
    )
    wrapped = PeftModel.from_pretrained(base_model, path_or_id)
    merged = wrapped.merge_and_unload()
    tokenizer = _load_tokenizer(path_or_id, base)
    return merged, tokenizer


def load_for_training(
    path_or_id: str,
    *,
    dtype: "torch.dtype | None" = None,
    device_map: "str | dict | None" = "auto",
    base_override: str | None = None,
    merge_before_attach: bool = True,
) -> "tuple[PreTrainedModel, PreTrainedTokenizerBase, str]":
    """Like ``load_for_inference`` but returns the resolved base id too.

    Returns ``(model, tokenizer, effective_base_id)``.

    The third return is what callers should use to load a reference copy
    (DPO needs this — see module docstring). For hub/merged paths,
    ``effective_base_id == path_or_id``. For adapter paths, it's the
    underlying base. When ``merge_before_attach=True`` (the default and
    only sensible choice for current callers), the returned model is the
    fully-merged result and the caller can attach a fresh ``LoraConfig``
    on top without PEFT-on-PEFT awkwardness.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.bfloat16

    kind = detect_kind(path_or_id)
    if kind in ("hub", "merged"):
        model = AutoModelForCausalLM.from_pretrained(
            path_or_id, dtype=dtype, device_map=device_map,
        )
        tokenizer = _load_tokenizer(path_or_id, path_or_id)
        return model, tokenizer, path_or_id

    # adapter
    from peft import PeftModel

    base = resolve_base_model(path_or_id, base_override)
    base_model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, device_map=device_map,
    )
    wrapped = PeftModel.from_pretrained(base_model, path_or_id)
    if merge_before_attach:
        model = wrapped.merge_and_unload()
    else:
        # Caller wants the live PeftModel — they accept responsibility
        # for any PEFT-on-PEFT gymnastics that follow.
        model = wrapped
    tokenizer = _load_tokenizer(path_or_id, base)
    return model, tokenizer, base
