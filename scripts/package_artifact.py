#!/usr/bin/env python3
"""Packages a compliant edsan-doc-trimmer release archive with artifact.json (v1.1.0)."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from redsan_doc_trimmer.model_resolver import get_user_cache_dirs


def package_artifact(
    onnx_dir: str = "artifacts/active_learning/onnx_export",
    worker_script: str = "scripts/trim_batch_service.py",
    output_zip: str = "artifacts/edsan-doc-trimmer-v1.1.0.zip",
    version: str = "1.1.0",
    contract: str = "rectype-aware-v1",
    install_to_cache: bool = True,
):
    model_path = Path(onnx_dir).resolve()
    worker_path = Path(worker_script).resolve()
    out_zip_path = Path(output_zip).resolve()
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Create artifact.json
    manifest = {
        "artifact_name": "edsan-doc-trimmer",
        "artifact_version": version,
        "worker_contract": contract,
        "model_type": "DrBERT-sequence-classification",
        "exported_at": "2026-09-21",
    }
    manifest_path = model_path / "artifact.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Created {manifest_path}")

    # 2. Copy worker script to onnx_dir so direct checkouts satisfy redsan validation
    shutil.copy2(worker_path, model_path / "trim_batch_service.py")
    print(f"Copied worker script to {model_path / 'trim_batch_service.py'}")

    # 3. Verify all required files exist
    required_files = [
        "model.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "artifact.json",
        "trim_batch_service.py",
    ]
    for rf in required_files:
        p = model_path / rf
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact file: {p}")

    # 4. Create zip archive
    print(f"Creating release archive: {out_zip_path}...")
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rf in required_files:
            zf.write(model_path / rf, arcname=rf)
        # Add safetensors if present
        if (model_path / "model.safetensors").exists():
            zf.write(model_path / "model.safetensors", arcname="model.safetensors")

    print(f"Archive packaged successfully: {out_zip_path} ({out_zip_path.stat().st_size:,} bytes)")

    # 5. Optional: Install directly into user cache directory
    if install_to_cache:
        cache_dirs = get_user_cache_dirs()
        if cache_dirs:
            target_cache = cache_dirs[0]
            target_cache.mkdir(parents=True, exist_ok=True)
            print(f"Installing validated artifact to user cache: {target_cache}...")
            for rf in required_files:
                shutil.copy2(model_path / rf, target_cache / rf)
            if (model_path / "model.safetensors").exists():
                shutil.copy2(model_path / "model.safetensors", target_cache / "model.safetensors")
            print(f"Artifact successfully installed to {target_cache}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package compliant trimmer release archive")
    parser.add_argument("--onnx_dir", type=str, default="artifacts/active_learning/onnx_export")
    parser.add_argument("--worker", type=str, default="scripts/trim_batch_service.py")
    parser.add_argument("--output", type=str, default="artifacts/edsan-doc-trimmer-v1.1.0.zip")
    parser.add_argument("--version", type=str, default="1.1.0")
    parser.add_argument("--no_cache_install", action="store_true")
    args = parser.parse_args()

    package_artifact(
        onnx_dir=args.onnx_dir,
        worker_script=args.worker,
        output_zip=args.output,
        version=args.version,
        install_to_cache=not args.no_cache_install,
    )
