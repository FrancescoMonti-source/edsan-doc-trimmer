from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from package_artifact import package_artifact, read_worker_contract

REQUIRED_MODEL_FILES = (
    "model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
)


def test_package_contains_tested_worker_bytes_and_declared_contract(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    tested_worker = repo_root / "scripts" / "trim_batch_service.py"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for filename in REQUIRED_MODEL_FILES:
        (model_dir / filename).write_bytes(b"test artifact")

    archive = tmp_path / "edsan-doc-trimmer.zip"
    package_artifact(
        onnx_dir=str(model_dir),
        worker_script=str(tested_worker),
        output_zip=str(archive),
        install_to_cache=False,
    )

    expected_worker = tested_worker.read_bytes()
    assert (model_dir / "trim_batch_service.py").read_bytes() == expected_worker

    manifest = json.loads((model_dir / "artifact.json").read_text(encoding="utf-8"))
    assert manifest["worker_contract"] == "rectype-aware-v1"
    assert manifest["worker_contract"] == read_worker_contract(tested_worker)

    with zipfile.ZipFile(archive) as packaged:
        assert packaged.read("trim_batch_service.py") == expected_worker
        assert (
            json.loads(packaged.read("artifact.json"))["worker_contract"]
            == "rectype-aware-v1"
        )


def test_packaging_rejects_worker_without_declared_contract(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("print('worker')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="WORKER_CONTRACT"):
        read_worker_contract(worker)
