"""Optional real Transformers/PEFT backend. Imported only for GPU commands.

This module performs *genuine gradient updates*. There is intentionally no
simulated-model fallback for training: incompatible checkpoints fail loudly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable


def require_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install GPU dependencies: pip install -r experiments/real_lora/requirements-real.txt") from exc


def format_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    # Explicit fallback is reproducible for instruction models without a
    # tokenizer chat template, but a native chat template is preferred.
    return "".join(f"[{m['role'].upper()}]\n{m['content']}\n" for m in messages) + "[ASSISTANT]\n"


def encoded_examples(tokenizer, rows: list[dict], max_length: int):
    encoded = []
    eos = tokenizer.eos_token or ""
    for row in rows:
        prompt = format_prompt(tokenizer, row["messages"])
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suffix = tokenizer.encode(row["target"] + eos, add_special_tokens=False)
        if len(suffix) + 1 >= max_length:
            raise ValueError("target length exceeds max_length")
        prefix = prefix[-(max_length - len(suffix)):]
        ids = prefix + suffix
        labels = [-100] * len(prefix) + suffix
        encoded.append((ids, labels))
    return encoded


def load_base(model_id: str, revision: str, quantization: str, model_class: str):
    require_deps()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kw = {"revision": revision, "trust_remote_code": False,
          "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if quantization == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit LoRA requires CUDA in this runner")
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = "auto"
    elif quantization == "none":
        kw["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
    else:
        raise ValueError("quantization must be 4bit or none")
    if model_class == "causal":
        cls = AutoModelForCausalLM
    elif model_class == "image-text":
        from transformers import AutoModelForImageTextToText
        cls = AutoModelForImageTextToText
    else:
        raise ValueError("model_class must be causal or image-text")
    model = cls.from_pretrained(model_id, **kw)
    config_hash = hashlib.sha256(model.config.to_json_string().encode()).hexdigest()
    return model, tokenizer, config_hash


def train_adapter(model_id: str, revision: str, model_class: str, quantization: str,
                  rows: list[dict], output: Path, *, steps: int, lr: float,
                  rank: int, target_modules: list[str], seed: int,
                  max_length: int, grad_accum: int) -> dict:
    require_deps()
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("This command trains a real 4B model and requires a CUDA GPU")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Adapter directory must be empty: {output}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed)
    begin = time.monotonic()
    model, tokenizer, config_hash = load_base(model_id, revision, quantization, model_class)
    observed_modules = {n.split(".")[-1] for n, _ in model.named_modules()}
    missing = set(target_modules) - observed_modules
    if missing:
        raise ValueError(f"LoRA target modules missing from this checkpoint: {sorted(missing)}")
    if quantization == "4bit":
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    config = LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=0.05,
                        target_modules=target_modules, bias="none",
                        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.train()
    data = encoded_examples(tokenizer, rows, max_length)
    if not data:
        raise ValueError("empty training dataset")
    supervised_target_tokens = sum(sum(x != -100 for x in y) for _, y in data)
    total_tokens = sum(len(ids) for ids, _ in data)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    processed = 0
    sum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for iteration in range(steps * grad_accum):
        sample_id = rng.randrange(len(data))
        ids, labels = data[sample_id]
        device = next(model.parameters()).device
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = torch.tensor([labels], dtype=torch.long, device=device)
        loss = model(input_ids=x, labels=y).loss / grad_accum
        loss.backward()
        sum_loss += float(loss.detach().cpu()) * grad_accum
        processed += 1
        if (iteration+1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output / "tokenizer_reference")
    torch.cuda.synchronize()
    return {"model_config_sha256": config_hash,
            "training_steps": steps, "gradient_microsteps": processed,
            "supervised_examples": len(rows),
            "total_training_input_tokens_before_resampling": total_tokens,
            "supervised_target_tokens_before_resampling": supervised_target_tokens,
            "mean_microstep_loss": round(sum_loss / processed, 6),
            "elapsed_seconds": round(time.monotonic() - begin, 3),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())}


def load_pair(model_id: str, revision: str, model_class: str,
              quantization: str, adapter: Path):
    require_deps()
    from peft import PeftModel
    base, tok, config_hash = load_base(model_id, revision, quantization, model_class)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model.eval()
    return model, tok, config_hash


def make_responder(model, tokenizer, *, max_new_tokens: int) -> Callable[[list[dict]], str]:
    import torch

    @torch.no_grad()
    def respond(messages: list[dict]) -> str:
        text = format_prompt(tokenizer, messages)
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        result = model.generate(**encoded, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(result[0][encoded["input_ids"].shape[-1]:],
                                skip_special_tokens=True)
    return respond


def adapter_hash(path: Path) -> str:
    """Hash saved LoRA files, not the large frozen base checkpoint."""
    hasher = hashlib.sha256()
    for file in sorted(path.glob("adapter*")):
        if file.is_file():
            hasher.update(file.name.encode())
            with file.open("rb") as fp:
                for block in iter(lambda: fp.read(1024 * 1024), b""):
                    hasher.update(block)
    return hasher.hexdigest()
