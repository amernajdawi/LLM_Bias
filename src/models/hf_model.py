import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _attn_to_options(attn_tuple: tuple, opt_idx: Dict[str, int], letters: List[str]) -> List[Dict[str, float]]:
    out = []
    for attn in attn_tuple:
        a = attn[0]
        d = {}
        for L in letters:
            i = opt_idx.get(L)
            d[L] = float(a[:, -1, i].mean().item()) if i is not None and i < a.size(-1) else 0.0
        out.append(d)
    return out


def _probs_per_layer(hid_tuple: tuple, model: torch.nn.Module, ids: Dict[str, int], letters: List[str], device: str) -> List[Dict[str, float]]:
    lm = getattr(model, "lm_head", getattr(model, "embed_tokens", None))
    if lm is None:
        lm = getattr(model.model, "lm_head", None)
    if lm is None:
        return []
    out = []
    for h in hid_tuple:
        x = h[0:1, -1:, :].to(device)
        logits = lm(x)[0, 0, :].float().cpu()
        sm = torch.softmax(logits, dim=-1)
        out.append({c: float(sm[ids[c]].item()) for c in letters if c in ids})
    return out


class HFModel:
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        tokenizer_kw = {"token": hf_token} if hf_token else {}
        model_kw = {"token": hf_token} if hf_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kw)

        load_8bit = os.environ.get("LOAD_8BIT", "").strip().lower() in ("1", "true", "yes")
        if load_8bit:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    load_in_8bit=True,
                    device_map="auto",
                    attn_implementation="eager",
                    **model_kw,
                )
                self.device = next(self.model.parameters()).device
            except Exception:
                load_8bit = False
        if not load_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                **model_kw,
            ).to(self.device)
        self.model.eval()
        self._ids = {c: self._token_id(c) for c in ["A", "B", "C", "D"]}

    def _token_id(self, c: str) -> int:
        for t in [c, " " + c, c + ")", " " + c + ")"]:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if ids:
                return ids[0]
        return self.tokenizer.unk_token_id

    def _option_indices(self, text: str) -> Dict[str, int]:
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        offsets = enc["offset_mapping"]
        out = {}
        for L in ["A", "B", "C", "D"]:
            p = text.find(L + ")")
            if p >= 0:
                for i, (a, b) in enumerate(offsets):
                    if a <= p < b or (i + 1 < len(offsets) and a <= p < offsets[i + 1][0]):
                        out[L] = i
                        break
                if L not in out:
                    for i, (a, b) in enumerate(offsets):
                        if b > p:
                            out[L] = i
                            break
        return out

    def predict(self, prompt: str, letters: List[str]) -> Tuple[str, Dict[str, float], List, List, Dict[str, int]]:
        to_chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inp = to_chat if isinstance(to_chat, str) else f"<|user|>\n{prompt}\n<|assistant|>\n"
        enc = self.tokenizer(inp, return_tensors="pt", add_special_tokens=True).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, output_attentions=True, output_hidden_states=True)
        logits = out.logits[0, -1, :].float().cpu()
        sm = torch.softmax(logits, dim=-1)
        probs = {c: float(sm[self._ids[c]].item()) for c in letters if c in self._ids}
        for c in letters:
            if c not in probs:
                probs[c] = 0.0
        pred = max(letters, key=lambda x: probs.get(x, 0.0))
        opt_idx = self._option_indices(inp)
        attn_tup = tuple(a.detach().cpu() for a in out.attentions) if getattr(out, "attentions", None) else None
        hid_tup = tuple(h.detach().cpu() for h in out.hidden_states) if out.hidden_states else None
        attn_to_opts = _attn_to_options(attn_tup, opt_idx, letters) if attn_tup and opt_idx else []
        probs_ly = _probs_per_layer(hid_tup, self.model, self._ids, letters, self.device) if hid_tup else []
        return pred, probs, attn_to_opts, probs_ly, opt_idx
