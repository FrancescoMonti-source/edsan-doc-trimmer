"""ONNX export and verification for in-process inference in redsan."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export_to_onnx(
    model_path: str,
    output_dir: str = "./artifacts/drbert-trimmer-onnx",
    onnx_filename: str = "model.onnx",
    opset_version: int = 17,
):
    """Exports a fine-tuned DrBERT model to ONNX format with dynamic batch and sequence axes."""
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    output_onnx_path = str(out_dir_path / onnx_filename)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Save tokenizer assets next to model.onnx for standalone consumption by redsan
    tokenizer.save_pretrained(str(out_dir_path))
    model.config.save_pretrained(str(out_dir_path))

    # Dummy inputs for tracing
    dummy_text = "Dr. Martin - Service de Médecine Interne"
    dummy_context = "CHRU de Rouen \n Bâtiment B"
    inputs = tokenizer(
        dummy_text,
        dummy_context,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    input_names = ["input_ids", "attention_mask"]
    dummy_args = (inputs["input_ids"], inputs["attention_mask"])

    # If model uses token_type_ids
    if "token_type_ids" in inputs:
        input_names.append("token_type_ids")
        dummy_args = (*dummy_args, inputs["token_type_ids"])

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }
    if "token_type_ids" in inputs:
        dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}

    print(f"Exporting model to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_args,
        output_onnx_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"ONNX export completed: {output_onnx_path}")

    # Verification with ONNX Runtime
    print("Verifying numerical parity with ONNX Runtime...")
    with torch.no_grad():
        pt_outputs = model(*dummy_args).logits.cpu().numpy()

    session = ort.InferenceSession(output_onnx_path)
    ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items() if k in input_names}
    ort_outputs = session.run(["logits"], ort_inputs)[0]

    np.testing.assert_allclose(pt_outputs, ort_outputs, rtol=1e-03, atol=1e-04)
    print("Verification successful: PyTorch and ONNX Runtime predictions match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export fine-tuned trimmer model to ONNX for redsan"
    )
    parser.add_argument(
        "model_path", type=str, help="Path to checkpoint directory or HF hub model name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./artifacts/drbert-trimmer-onnx"
    )
    parser.add_argument("--filename", type=str, default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model_path,
        output_dir=args.output_dir,
        onnx_filename=args.filename,
        opset_version=args.opset,
    )
