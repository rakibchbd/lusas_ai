from __future__ import annotations

import argparse
from pathlib import Path
import re


LUSAS_IDENTITY = (
    "I am LUSAS AI, developed by Lusa Chowdhury (Rakib). "
    "My current model is based on an open-weight coding model and fine-tuned locally."
)


def identity_response(prompt: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", prompt.lower())
    words = set(normalized.split())
    asks_about_creator = (
        bool(words & {"created", "creator", "made", "developer", "developed"})
        and bool(words & {"you", "your"})
    )
    asks_about_openai_identity = "openai" in words and bool(
        words & {"you", "your", "created", "made", "developer", "developed"}
    )
    if asks_about_creator or asks_about_openai_identity:
        return LUSAS_IDENTITY
    return None

from lusas_ai.identity import CREATOR_QUESTION, IDENTITY_RESPONSE

# Backward-compatible name for existing local identity tests.
identity_response = IDENTITY_RESPONSE


MODEL_SYSTEM_PROMPT = (
    "You are LUSAS AI, also known as Lusa. If asked about your creator, "
    "developer, maker, author, or designer, provide the complete official "
    f"creator biography:\n{IDENTITY_RESPONSE}\n"
    "Never claim OpenAI or another company created you."
)


def load_model(model_path: Path):
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model runtime dependencies are missing. Install them with the same "
            "Python interpreter: python3 -m pip install -r training/requirements.txt"
        ) from exc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
    ).to(device)
    return torch, tokenizer, model, device


<<<<<<< Updated upstream
def generate_loaded(
    torch,
    tokenizer,
    model,
    device: str,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repeat_penalty: float = 1.0,
    num_ctx: int | None = None,
    seed: int | None = None,
) -> str:
    if CREATOR_QUESTION.search(prompt):
        return IDENTITY_RESPONSE
=======
def generate_loaded(torch, tokenizer, model, device: str, prompt: str, max_new_tokens: int) -> str:
    branded_response = identity_response(prompt)
    if branded_response is not None:
        return branded_response
>>>>>>> Stashed changes
    formatted = (
        f"### System:\n{MODEL_SYSTEM_PROMPT}\n\n"
        "### Instruction:\n"
        f"{prompt}\n\n"
        "### Response:\n"
    )
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=num_ctx is not None,
        max_length=num_ctx,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if seed is not None:
        torch.manual_seed(seed)
    generation_options = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "repetition_penalty": repeat_penalty,
    }
    if temperature > 0:
        generation_options.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    with torch.no_grad():
        output = model.generate(**inputs, **generation_options)
    generated_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def generate(model_path: Path, prompt: str, max_new_tokens: int = 128) -> str:
    torch, tokenizer, model, device = load_model(model_path)
    return generate_loaded(torch, tokenizer, model, device, prompt, max_new_tokens)


def interactive(model_path: Path, max_new_tokens: int) -> None:
    torch, tokenizer, model, device = load_model(model_path)
    print("LUSAS model chat. Type /exit to quit.")
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"/exit", "/quit"}:
            return
        if not prompt:
            continue
        print("\nLUSAS> " + generate_loaded(
            torch, tokenizer, model, device, prompt, max_new_tokens
        ))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a trained LUSAS model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the model loaded and accept multiple prompts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.interactive and args.prompt:
        parser.error("--interactive cannot be combined with --prompt")
    if not args.interactive and not args.prompt:
        parser.error("provide --prompt or use --interactive")
    if args.interactive:
        interactive(args.model, args.max_new_tokens)
    else:
        print(generate(args.model, args.prompt, args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
