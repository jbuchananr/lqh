"""Append one-time secrets to the project's .env, out-of-band from the agent
conversation.

Used by the agent loop's secret-delivery path (see ``SecretDelivery`` in
``lqh.tools.handlers``) so a plaintext key reaches the user's ``.env`` without
ever entering ``session.messages`` (and thus the local JSONL log or the backend
payload capture). All operations are append-only: existing entries are never
rewritten or deleted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _has_env_var(env_text: str, env_var: str) -> bool:
    """True if a non-comment line already assigns ``env_var``."""
    target = env_var.strip()
    for raw in env_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.split("=", 1)[0].strip() == target:
            return True
    return False


def _git_tracked(project_dir: Path, rel: str) -> bool:
    """Best-effort check whether ``rel`` is tracked by git. Never raises."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            cwd=project_dir,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        # Not a git repo, git missing, timeout — treat as "unknown / not tracked".
        return False


def ensure_gitignored(project_dir: Path, entry: str = ".env") -> None:
    """Ensure ``entry`` appears in the project's .gitignore (creating it if
    absent). Append-only; existing content is preserved."""
    gitignore = project_dir / ".gitignore"
    try:
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    except Exception:
        existing = ""

    lines = {ln.strip() for ln in existing.splitlines()}
    if entry in lines or f"/{entry}" in lines:
        return

    block = "" if (existing == "" or existing.endswith("\n")) else "\n"
    block += f"{entry}\n"
    try:
        with gitignore.open("a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        # Best effort — a failed .gitignore update must not break key delivery.
        pass


def append_env_secret(
    project_dir: Path,
    env_var: str,
    value: str,
    comment: str | None = None,
) -> str:
    """Append ``env_var=value`` to ``project_dir/.env`` (append-only — never
    overwrites or deletes existing entries) and ensure ``.env`` is gitignored.

    Returns a short human-readable note (for the redacted conversation message)
    describing what happened, including a duplicate warning if ``env_var``
    already had a value, and a warning if ``.env`` is tracked by git.
    """
    env_path = project_dir / ".env"
    try:
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    except Exception:
        existing = ""

    is_dup = _has_env_var(existing, env_var)

    block = "" if (existing == "" or existing.endswith("\n")) else "\n"
    if comment:
        block += f"{comment}\n"
    block += f"{env_var}={value}\n"

    try:
        with env_path.open("a", encoding="utf-8") as f:
            f.write(block)
    except Exception as e:  # noqa: BLE001
        return f" ⚠️ Could not write to .env: {e}. The key was NOT saved."

    ensure_gitignored(project_dir, ".env")

    note = f" Appended to .env as {env_var}."
    if is_dup:
        note += (
            f" ⚠️ {env_var} already existed in .env — a second entry was appended; "
            f"remove the stale one to avoid ambiguity."
        )
    if _git_tracked(project_dir, ".env"):
        note += (
            " ⚠️ .env is tracked by git — untrack it (git rm --cached .env) so the "
            "key is not committed."
        )
    return note
