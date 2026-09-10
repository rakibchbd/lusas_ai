from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_cases(path: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            if not isinstance(case, dict):
                raise ValueError("Each evaluation line must be an object.")
            if not isinstance(case.get("instruction"), str):
                raise ValueError("Each evaluation case needs an instruction.")
            if not isinstance(case.get("expected"), str):
                raise ValueError("Each evaluation case needs an expected string.")
            cases.append(case)
    if not cases:
        raise ValueError("The evaluation dataset is empty.")
    return cases


def evaluate(model_path: Path, data_path: Path, max_new_tokens: int = 128) -> dict:
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model runtime dependencies are missing. Install training/requirements.txt."
        ) from exc

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
    ).to(device)
    results = []
    for case in load_cases(data_path):
        formatted = (
            "### Instruction:\n"
            f"{case['instruction']}\n\n"
            "### Response:\n"
        )
        inputs = tokenizer(formatted, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        expected = case["expected"]
        passed = expected in generated
        results.append(
            {
                "instruction": case["instruction"],
                "expected": expected,
                "passed": passed,
                "output": generated,
            }
        )

    passed_count = sum(1 for result in results if result["passed"])
    return {
        "passed": passed_count == len(results),
        "score": passed_count / len(results),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a LUSAS model candidate.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    report = evaluate(args.model, args.data, args.max_new_tokens)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
