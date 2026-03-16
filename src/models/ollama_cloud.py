import os
import time
from typing import Dict, List, Tuple

from ollama import Client
from ollama._types import ResponseError


class OllamaCloudModel:
    """
    Wrapper around Ollama cloud chat API for multiple-choice questions.

    It expects the underlying model to answer with a single option letter
    (A/B/C/D, etc.) at the start of its response, consistent with the
    existing OpenAI model wrapper.
    """

    def __init__(self, model_name: str):
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError("OLLAMA_API_KEY is not set.")
        self.model_name = model_name
        self.client = Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def predict(self, prompt: str, letters: List[str]) -> Tuple[str, Dict[str, float]]:
        messages = [{"role": "user", "content": prompt}]
        max_retries = 5
        backoff_server = [15, 30, 60, 120, 180]   # 500, 502, 503
        backoff_rate_limit = [30, 60, 120, 180, 240]  # 429 too many requests
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                )
                break
            except ResponseError as e:
                last_error = e
                retryable = e.status_code in (429, 500, 502, 503) and attempt < max_retries - 1
                if retryable:
                    wait = backoff_rate_limit[attempt] if e.status_code == 429 else backoff_server[attempt]
                    time.sleep(wait)
                    continue
                raise

        content = (resp["message"]["content"] or "").strip().upper()

        # First character as prediction; default to first letter if parsing fails
        pred = content[0] if content and content[0] in letters else letters[0]

        # Ollama cloud API does not expose token-level logprobs; return a
        # placeholder distribution so downstream code can still compute
        # anchored bias and accuracy-based metrics.
        logprobs: Dict[str, float] = {L: -100.0 for L in letters}
        logprobs[pred] = 0.0
        return pred, logprobs

