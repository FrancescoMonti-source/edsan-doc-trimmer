from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import trim_batch_service as worker


@pytest.fixture
def onnx_runtime(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"test model")
    calls = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _model_dir):
            return cls()

        def __call__(self, targets, _contexts, **_kwargs):
            count = len(targets)
            return {
                "input_ids": np.ones((count, 2), dtype=np.int64),
                "attention_mask": np.ones((count, 2), dtype=np.int64),
            }

    class FakeSession:
        def __init__(self, _model_path, _options, providers):
            self.providers = list(providers)
            calls["providers"] = self.providers

        def get_providers(self):
            return [
                provider[0] if isinstance(provider, tuple) else provider
                for provider in self.providers
            ]

        def disable_fallback(self):
            calls["fallback_disabled"] = True

        def run(self, _outputs, inputs):
            count = inputs["input_ids"].shape[0]
            return [np.tile(np.array([[10.0, 0.0]]), (count, 1))]

    monkeypatch.setattr(worker, "resolve_model_dir", lambda _candidate=None: model_dir)
    monkeypatch.setattr(worker, "USE_RUST_TOKENIZER", False)
    monkeypatch.setattr(worker, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(worker.ort, "InferenceSession", FakeSession)
    return calls


def test_auto_uses_cuda_onnx_provider_for_detected_nvidia_gpu(
    monkeypatch, onnx_runtime
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "auto")
    monkeypatch.setattr(
        worker,
        "_detect_accelerators",
        lambda: {"nvidia"},
        raising=False,
    )
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    assert onnx_runtime["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert result["execution_provider"] == "CUDAExecutionProvider"


def test_auto_defaults_to_cpu_without_detected_accelerators(monkeypatch, onnx_runtime):
    monkeypatch.delenv("EDSAN_TRIMMER_DEVICE", raising=False)
    monkeypatch.setattr(worker, "_detect_accelerators", lambda: set())
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: [
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    assert onnx_runtime["providers"] == ["CPUExecutionProvider"]
    assert result["execution_provider"] == "CPUExecutionProvider"


def test_auto_uses_openvino_for_detected_intel_gpu(monkeypatch, onnx_runtime):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "auto")
    monkeypatch.setattr(worker, "_detect_accelerators", lambda: {"intel"})
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    assert onnx_runtime["providers"] == [
        ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
        "CPUExecutionProvider",
    ]
    assert result["execution_provider"] == "OpenVINOExecutionProvider"


@pytest.mark.parametrize(
    ("mode", "provider"),
    [
        ("cpu", "CPUExecutionProvider"),
        ("cuda", "CUDAExecutionProvider"),
        ("openvino", "OpenVINOExecutionProvider"),
        ("dml", "DmlExecutionProvider"),
        ("migraphx", "MIGraphXExecutionProvider"),
    ],
)
def test_device_override_uses_the_requested_onnx_provider(
    monkeypatch, onnx_runtime, mode, provider
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", mode)
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: [provider, "CPUExecutionProvider"],
    )

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    expected_providers = [provider]
    if provider != "CPUExecutionProvider":
        expected_providers.append("CPUExecutionProvider")
    if provider == "OpenVINOExecutionProvider":
        expected_providers[0] = (
            provider,
            {"device_type": "GPU"},
        )
    assert onnx_runtime["providers"] == expected_providers
    assert result["execution_provider"] == provider


def test_missing_cuda_provider_falls_back_with_marked_install_suggestion(
    monkeypatch, capsys, onnx_runtime
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "auto")
    monkeypatch.setattr(worker, "_detect_accelerators", lambda: {"nvidia"})
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    assert onnx_runtime["providers"] == ["CPUExecutionProvider"]
    assert result["execution_provider"] == "CPUExecutionProvider"
    captured = capsys.readouterr()
    assert "EDSAN_TRIMMER_NOTICE:WARNING:" in captured.err
    assert "onnxruntime-gpu" in captured.err


def test_auto_prefers_cuda_when_both_gpu_providers_are_available(
    monkeypatch, onnx_runtime
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "auto")
    monkeypatch.setattr(worker, "_detect_accelerators", lambda: {"nvidia", "intel"})
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: [
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    assert onnx_runtime["providers"] == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert result["execution_provider"] == "CUDAExecutionProvider"


def test_unknown_device_override_is_rejected(monkeypatch, onnx_runtime):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "gpu")

    with pytest.raises(ValueError, match="EDSAN_TRIMMER_DEVICE"):
        worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])


def test_missing_openvino_provider_suggests_install_and_uses_cpu(
    monkeypatch, capsys, onnx_runtime
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "auto")
    monkeypatch.setattr(worker, "_detect_accelerators", lambda: {"intel"})
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    captured = capsys.readouterr()
    assert onnx_runtime["providers"] == ["CPUExecutionProvider"]
    assert result["execution_provider"] == "CPUExecutionProvider"
    assert "EDSAN_TRIMMER_NOTICE:WARNING:" in captured.err
    assert "onnxruntime-openvino" in captured.err
    assert "OpenVINO runtime" in captured.err


def test_provider_initialization_failure_retries_with_cpu(
    monkeypatch, capsys, onnx_runtime
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "cuda")
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    calls = []

    class CpuSession:
        def __init__(self, _model_path, _options, providers):
            self.providers = list(providers)

        def get_providers(self):
            return [
                provider[0] if isinstance(provider, tuple) else provider
                for provider in self.providers
            ]

        def disable_fallback(self):
            pass

        def run(self, _outputs, inputs):
            count = inputs["input_ids"].shape[0]
            return [np.tile(np.array([[10.0, 0.0]]), (count, 1))]

    def create_session(_model_path, _options, providers):
        calls.append(list(providers))
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("CUDA runtime library unavailable")
        return CpuSession(_model_path, _options, providers)

    monkeypatch.setattr(worker.ort, "InferenceSession", create_session)

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    captured = capsys.readouterr()
    assert calls == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert result["execution_provider"] == "CPUExecutionProvider"
    assert "CUDA runtime library unavailable" in captured.err
    assert "onnxruntime-gpu" in captured.err


def test_openvino_session_initializes_with_gpu_device_options(
    monkeypatch, onnx_runtime
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "openvino")
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )

    worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])

    assert onnx_runtime["providers"][0] == (
        "OpenVINOExecutionProvider",
        {"device_type": "GPU"},
    )


def test_onnx_runtime_constructor_fallback_is_reported_and_recorded(
    monkeypatch, capsys, onnx_runtime
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "cuda")
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    class CpuFallbackSession:
        def __init__(self, _model_path, _options, providers):
            onnx_runtime["providers"] = list(providers)

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def disable_fallback(self):
            pass

        def run(self, _outputs, inputs):
            count = inputs["input_ids"].shape[0]
            return [np.tile(np.array([[10.0, 0.0]]), (count, 1))]

    monkeypatch.setattr(worker.ort, "InferenceSession", CpuFallbackSession)

    result = worker.trim_batch([{"id": "doc-1", "text": "Clinical narrative"}])[0]

    assert result["execution_provider"] == "CPUExecutionProvider"
    assert "CUDAExecutionProvider" not in result["execution_provider"]
    assert "EDSAN_TRIMMER_NOTICE:WARNING:" in capsys.readouterr().err


def test_runtime_provider_failure_restarts_the_batch_on_cpu(
    monkeypatch, capsys, onnx_runtime
):
    monkeypatch.setenv("EDSAN_TRIMMER_DEVICE", "cuda")
    monkeypatch.setattr(
        worker.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    session_calls = []

    class Session:
        def __init__(self, _model_path, _options, providers):
            self.providers = list(providers)
            session_calls.append(self.providers)
            self.run_count = 0

        def get_providers(self):
            return [
                provider[0] if isinstance(provider, tuple) else provider
                for provider in self.providers
            ]

        def disable_fallback(self):
            pass

        def run(self, _outputs, inputs):
            self.run_count += 1
            if self.providers[0] == "CUDAExecutionProvider":
                if self.run_count == 1:
                    count = inputs["input_ids"].shape[0]
                    return [np.tile(np.array([[0.0, 10.0]]), (count, 1))]
                raise worker.ort.capi.onnxruntime_pybind11_state.EPFail(
                    "CUDA execution provider failed"
                )
            count = inputs["input_ids"].shape[0]
            return [np.tile(np.array([[10.0, 0.0]]), (count, 1))]

    monkeypatch.setattr(worker.ort, "InferenceSession", Session)
    result = worker.trim_batch(
        [{"id": "doc-1", "text": "first\nsecond\nthird"}],
        batch_size=1,
    )[0]

    assert session_calls == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert result["execution_provider"] == "CPUExecutionProvider"
    assert result["trimmed_text"] == "first\nsecond\nthird"
    assert "CUDA execution provider failed" in capsys.readouterr().err
