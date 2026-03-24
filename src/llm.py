"""Load HF causal LMs and run inference for parcel explanations."""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.gpu import discover_gpus

log = logging.getLogger(__name__)


def select_llm_profile(cfg: dict) -> tuple[str, dict]:
    """Return (profile_name, profile_dict) for LLM loading."""
    llm_cfg = cfg.get("llm") or {}
    profiles = llm_cfg.get("profiles") or {}
    if not profiles:
        raise ValueError("config llm.profiles is missing or empty")

    override = llm_cfg.get("profile_override")
    if override:
        name = str(override)
        if name not in profiles:
            raise ValueError(f"llm.profile_override {name!r} not in llm.profiles")
        log.info("Using LLM profile override %r", name)
        return name, profiles[name]

    threshold = float(llm_cfg.get("vram_threshold_gb", 8))
    gpus = discover_gpus()
    best_vram = max((g["vram_gb"] for g in gpus), default=0.0)

    profile_name = "high" if best_vram >= threshold else "low"
    if profile_name not in profiles:
        profile_name = "low" if "low" in profiles else next(iter(profiles))
    log.info(
        "Auto-selected LLM profile %r (best VRAM: %.1f GB, threshold %.1f GB)",
        profile_name,
        best_vram,
        threshold,
    )
    return profile_name, profiles[profile_name]


def load_model_and_tokenizer(profile: dict) -> tuple[Any, Any]:
    """Load Hugging Face causal LM with dtype / optional 4-bit quantisation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = profile["model_id"]
    kwargs: dict[str, Any] = {"trust_remote_code": True}

    use_cuda = torch.cuda.is_available()
    quant = profile.get("quantization")

    if quant == "4bit" and use_cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = "auto"
    elif quant == "4bit" and not use_cuda:
        log.warning("4-bit quantisation requires CUDA — loading %s in float16 on CPU", model_id)
        kwargs["torch_dtype"] = torch.float16
    elif profile.get("dtype") == "float16":
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto" if use_cuda else None

    if not use_cuda and kwargs.get("device_map") == "auto":
        kwargs.pop("device_map", None)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not use_cuda and kwargs.get("device_map") is None:
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def generate_json_response(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    user_text: str,
    profile: dict,
    llm_cfg: dict,
    temperature: float | None = None,
) -> str:
    """Run chat generation; return decoded new tokens (strip prompt)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    temp = float(temperature if temperature is not None else llm_cfg.get("temperature", 0.1))
    top_p = float(llm_cfg.get("top_p", 0.9))
    max_new = int(profile.get("max_new_tokens", 512))

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        log.warning("apply_chat_template failed (%s), falling back to plain concat", e)
        prompt = f"{system_prompt}\n\n{user_text}\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temp > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temp
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    gen = out[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text.strip()
