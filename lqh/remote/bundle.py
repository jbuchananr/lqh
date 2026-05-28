"""Build the submit bundle that ``CloudBackend`` ships to ``api.lqh.ai``.

A bundle is a gzipped tarball containing:

  config.json           # the run config, written verbatim
  <relative paths>      # every file the manifest points at, preserving
                        # its position under the project root

The tar's directory layout mirrors the project — so when the cloud
runner extracts it into ``/workspace/runs/<job_id>/inputs/`` and the
trainer's config refers to ``datasets/train.parquet``, the file is at
exactly that relative path. No path rewriting required.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

from lqh.sync import resolve_manifest

__all__ = ["build_bundle"]


def build_bundle(
    config: dict[str, Any],
    project_dir: Path,
) -> bytes:
    """Build the in-memory tarball.

    Kept in memory because the bundle is uploaded via httpx multipart
    in one go; spilling to disk would just add latency without
    bounding memory (we'd still buffer it for the request). If we
    grow large datasets that don't fit in RAM, that's a sign to use
    lqh.sources / HF datasets instead of embedding them in a submit.
    """
    paths = resolve_manifest(config, project_dir)
    seen: set[str] = set()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # 1. config.json at the tar root.
        cfg_bytes = (json.dumps(config, indent=2) + "\n").encode("utf-8")
        info = tarfile.TarInfo(name="config.json")
        info.size = len(cfg_bytes)
        tar.addfile(info, io.BytesIO(cfg_bytes))

        # 2. manifest files, anchored under the project dir.
        project_resolved = project_dir.resolve()
        for p in paths:
            p_resolved = p.resolve()
            try:
                rel = p_resolved.relative_to(project_resolved)
            except ValueError:
                # Manifest pointed outside the project — drop it under
                # an `extern/` prefix so the trainer can still find it
                # by reading the rewritten config. (Not used today;
                # safety net.)
                rel = Path("extern") / p_resolved.name
            arc = str(rel)
            if arc in seen:
                continue
            seen.add(arc)

            if p_resolved.is_dir():
                # Recurse for directory entries — typical for dataset
                # parquet shards.
                for f in sorted(p_resolved.rglob("*")):
                    if f.is_file():
                        try:
                            sub = f.relative_to(project_resolved)
                        except ValueError:
                            sub = Path("extern") / f.name
                        if str(sub) in seen:
                            continue
                        seen.add(str(sub))
                        tar.add(f, arcname=str(sub))
            else:
                tar.add(p_resolved, arcname=arc)

    return buf.getvalue()
