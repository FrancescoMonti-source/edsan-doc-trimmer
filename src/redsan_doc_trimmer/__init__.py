"""redsan_doc_trimmer: DrBERT training and ONNX export for hospital document trimming."""

from redsan_doc_trimmer.model_resolver import (
    format_model_not_found_message,
    get_user_cache_dirs,
    resolve_model_dir,
)

__version__ = "0.1.0"
__all__ = [
    "format_model_not_found_message",
    "get_user_cache_dirs",
    "resolve_model_dir",
]
