from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class Message:
    def __init__(self, content: str):
        self.content = content


class Choice:
    def __init__(self, message: Message):
        self.message = message


class ChatCompletionResult:
    def __init__(self, content: str):
        self.choices = [Choice(Message(content))]


_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None


def get_local_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n[Local LLM] Loading {model_name} onto GPU...")
        _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"[Local LLM] {model_name} loaded successfully.")
    return _LOCAL_MODEL, _LOCAL_TOKENIZER


def local_chat_completion(
    messages: list[dict],
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    model_name: str | None = None,
) -> ChatCompletionResult:
    import torch

    target_model = model_name or os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    model, tokenizer = get_local_model(target_model)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.05),
            temperature=max(temperature, 0.1) if temperature > 0.05 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0][inputs.input_ids.shape[1] :]
    resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return ChatCompletionResult(resp_text)


def safe_chat_completion(client, **kwargs):
    """Wrapper around client.chat.completions.create with automatic retry
    and backoff when hitting API rate limits.
    If LLM_PROVIDER is 'local' or model starts with 'Qwen/', runs locally via GPU.
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    model_name = kwargs.get("model", "")
    if provider == "local" or model_name.startswith("Qwen/") or model_name.startswith("local:"):
        if model_name.startswith("local:"):
            clean_model = model_name.replace("local:", "")
        elif "/" in model_name:
            clean_model = model_name
        else:
            clean_model = os.environ.get("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
        return local_chat_completion(
            messages=kwargs.get("messages", []),
            temperature=kwargs.get("temperature", 0.0),
            model_name=clean_model,
        )

    from openai import RateLimitError

    max_retries = 8
    default_wait = 20.0

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            msg = str(e)
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
            delay = float(match.group(1)) + 2.0 if match else default_wait
            print(f"\n[Rate limit 429] Waiting {delay:.1f}s before retry (attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)
            default_wait = max(default_wait * 1.5, delay)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                delay = float(match.group(1)) + 2.0 if match else default_wait
                print(f"\n[Quota exceeded] Waiting {delay:.1f}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                default_wait = max(default_wait * 1.5, delay)
            else:
                raise e

    return client.chat.completions.create(**kwargs)
