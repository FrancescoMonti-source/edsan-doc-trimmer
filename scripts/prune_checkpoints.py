#!/usr/bin/env python3
"""Prune intermediate PyTorch training checkpoints while preserving best_model and history."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def prune_checkpoints(artifacts_dir: str = "artifacts/active_learning", dry_run: bool = False):
    base_dir = Path(artifacts_dir)
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return

    checkpoint_dirs = sorted(base_dir.rglob("checkpoint-*"))
    if not checkpoint_dirs:
        print("No intermediate checkpoints found to prune.")
        return

    print("=" * 80)
    print(f"PRUNING INTERMEDIATE CHECKPOINTS IN: {base_dir} (Dry Run: {dry_run})")
    print("=" * 80)

    total_bytes = 0
    for ckpt in checkpoint_dirs:
        if ckpt.is_dir():
            size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
            total_bytes += size
            print(f"  {'[WOULD DELETE]' if dry_run else '[DELETED]'} {ckpt} ({size / (1024*1024):.1f} MB)")
            if not dry_run:
                # Save trainer_state.json to sibling best_model if present
                ts = ckpt / "trainer_state.json"
                best_model = ckpt.parent / "best_model"
                if ts.exists() and best_model.exists() and not (best_model / "trainer_state.json").exists():
                    shutil.copyfile(ts, best_model / "trainer_state.json")
                shutil.rmtree(ckpt)

    print("-" * 80)
    action = "Would reclaim" if dry_run else "Successfully reclaimed"
    print(f"{action}: {total_bytes / (1024*1024*1024):.2f} GiB ({total_bytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune intermediate PyTorch checkpoints")
    parser.add_argument("--artifacts_dir", default="artifacts/active_learning")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be pruned without deleting")
    args = parser.parse_args()

    prune_checkpoints(artifacts_dir=args.artifacts_dir, dry_run=args.dry_run)

