import math
from typing import Dict, List, Optional

import openai


class OpenAIModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = openai.OpenAI()

    def predict(self, prompt: str, letters: List[str]) -> tuple:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            logprobs=True,
            top_logprobs=min(len(letters), 20),  # API allows 0–20
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip().upper()
        pred = content[0] if content and content[0] in letters else letters[0]
        logprobs = {}
        if choice.logprobs and choice.logprobs.content:
            for t in choice.logprobs.content:
                if t.top_logprobs:
                    for lp in t.top_logprobs:
                        if lp.token and lp.token.strip().upper() in letters:
                            logprobs[lp.token.strip().upper()] = lp.logprob
        for L in letters:
            if L not in logprobs:
                logprobs[L] = -100.0
        return pred, logprobs
