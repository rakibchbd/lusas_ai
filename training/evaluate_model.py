from __future__ import annotations

import argparse
import gc
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
        from training.run_model import generate_loaded, load_model
    except ImportError as exc:
        raise RuntimeError(
            "Model runtime dependencies are missing. Install them with the same "
            "Python interpreter: python3 -m pip install -r training/requirements.txt"
        ) from exc

    torch, tokenizer, model, device = load_model(model_path)
    try:
        results = []
        for case in load_cases(data_path):
            generated = generate_loaded(
                torch,
                tokenizer,
                model,
                device,
                case["instruction"],
                max_new_tokens,
            )
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
    finally:
        del model
        del tokenizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


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
