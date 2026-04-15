"""Load HF causal LMs and run inference for parcel explanations."""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.gpu import discover_gpus

log = logging.getLogger(__name__)


def select_llm_profile(cfg: dict) -> tuple[str, dict, int | None]:
    """Return (profile_name, profile_dict, best_gpu_index) for LLM loading.

    best_gpu_index is the CUDA index of the GPU with the most total VRAM, or
    None when CUDA is not available. It is used by load_model_and_tokenizer to
    target a single device instead of letting accelerate distribute the model
    across all visible GPUs (which can OOM smaller secondary GPUs).
    """
    llm_cfg = cfg.get("llm") or {}
    profiles = llm_cfg.get("profiles") or {}
    if not profiles:
        raise ValueError("config llm.profiles is missing or empty")

    gpus = discover_gpus()
    best_gpu: dict | None = max(gpus, key=lambda g: g["vram_gb"]) if gpus else None
    best_gpu_index: int | None = best_gpu["index"] if best_gpu else None
    best_vram = best_gpu["vram_gb"] if best_gpu else 0.0

    override = llm_cfg.get("profile_override")
    if override:
        name = str(override)
        if name not in profiles:
            raise ValueError(f"llm.profile_override {name!r} not in llm.profiles")
        log.info("Using LLM profile override %r (target GPU: %s)", name, best_gpu_index)
        return name, profiles[name], best_gpu_index

    threshold = float(llm_cfg.get("vram_threshold_gb", 8))
    profile_name = "high" if best_vram >= threshold else "low"
    if profile_name not in profiles:
        profile_name = "low" if "low" in profiles else next(iter(profiles))
    log.info(
        "Auto-selected LLM profile %r (best VRAM: %.1f GB on GPU %s, threshold %.1f GB)",
        profile_name,
        best_vram,
        best_gpu_index,
        threshold,
    )
    return profile_name, profiles[profile_name], best_gpu_index


def load_model_and_tokenizer(
    profile: dict,
    target_gpu: int | None = None,
) -> tuple[Any, Any]:
    """Load Hugging Face causal LM with dtype / optional 4-bit quantisation.

    target_gpu is the CUDA device index to load the model onto. When provided,
    all model layers are placed on that single device instead of letting
    accelerate distribute across all visible GPUs (which can OOM smaller GPUs).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = profile["model_id"]
    # Do not use trust_remote_code: Phi-3.5-mini and Qwen2.5 are both natively
    # supported by transformers >= 4.40. Using trust_remote_code causes HF to
    # cache the model's custom Python code (modeling_phi3.py, etc.) in
    # modules/transformers_modules/, which can become incompatible with newer
    # transformers versions and is difficult to invalidate.
    kwargs: dict[str, Any] = {}

    use_cuda = torch.cuda.is_available()
    quant = profile.get("quantization")

    # Target a single GPU rather than "auto" to avoid spreading model layers
    # onto secondary GPUs that may have limited or already-occupied VRAM.
    device_map: str | dict = f"cuda:{target_gpu}" if (use_cuda and target_gpu is not None) else "auto"

    if quant == "4bit" and use_cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = device_map
    elif quant == "4bit" and not use_cuda:
        log.warning("4-bit quantisation requires CUDA — loading %s in float16 on CPU", model_id)
        kwargs["dtype"] = torch.float16
    elif profile.get("dtype") == "float16":
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = device_map if use_cuda else None

    if not use_cuda and kwargs.get("device_map") is not None:
        kwargs.pop("device_map", None)

    log.info("Loading %s onto %s", model_id, kwargs.get("device_map", "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
