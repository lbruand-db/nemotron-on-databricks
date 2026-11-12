"""Custom LLM Serving placeholder for Nano VL (SPEC §5.4).

Serving runs the model's `metadata["entrypoint"]` (a vLLM OpenAI server on port 8080, started in the model's
artifacts folder) instead of `predict()`. The endpoint takes OpenAI chat requests with `image_url` parts.
"""

from __future__ import annotations

try:
    from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel
except ImportError:  # pragma: no cover - mlflow is only needed at registration time
    ChatModel = object  # type: ignore[misc,assignment]
    ChatCompletionResponse = None  # type: ignore[assignment]

ARTIFACT_NAME = "nano_vl_ft"  # folder name of the merged weights inside the model's artifacts


class EntrypointModel(ChatModel):
    """Placeholder; Custom LLM Serving starts the metadata entrypoint."""

    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})
