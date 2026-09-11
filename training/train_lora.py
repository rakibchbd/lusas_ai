from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lusas_ai.model_registry import get_model_spec
from lusas_ai.data_pipeline import approved_training_records
from training.hf_auth import auth_kwargs


def load_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} is not an object.")
        records.extend(approved_training_records([record]))
    if not records:
        raise ValueError("The training dataset has no approved, clean records.")
    return [
        {
            "text": (
                "### Instruction:\n"
                f"{record['instruction']}\n\n"
                "### Response:\n"
                f"{record['output']}"
            )
        }
        for record in records
    ]


def train(
    data_path: Path,
    output_path: Path,
    model_id: str,
    foundation_model: str | None,
    foundation_path: Path | None,
    epochs: float,
    max_length: int,
) -> None:
    spec = get_model_spec(model_id)
    foundation_source = str(foundation_path.expanduser().resolve()) if foundation_path else foundation_model
    if not foundation_source:
        raise ValueError(
            f"{spec.display_name} requires an explicit foundation model or local foundation path."
        )
    if foundation_path and not foundation_path.is_dir():
        raise ValueError(f"Foundation path does not exist: {foundation_path}")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install the documented training extra."
        ) from exc

    records = load_records(data_path)
    dataset = Dataset.from_list(records)
    hub_auth = auth_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(foundation_source, **hub_auth)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        foundation_source,
        dtype=dtype,
        **hub_auth,
    )
    model.config.use_cache = False
    if mps_available or not torch.cuda.is_available():
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()
    output_path.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="epoch",
            report_to=[],
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
            optim="adafactor" if (mps_available or not torch.cuda.is_available()) else "adamw_torch",
            gradient_checkpointing=mps_available or not torch.cuda.is_available(),
        ),
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    try:
        trainer.train()
        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        (output_path / "model.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "display_name": spec.display_name,
                    "foundation_model": foundation_model,
                    "foundation_path": str(foundation_path.expanduser().resolve()) if foundation_path else None,
                    "training_records": len(records),
                    "deployment_status": "candidate",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        del trainer
        del model
        gc.collect()
        if mps_available:
            torch.mps.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an official LUSAS AI model adapter.")
    parser.add_argument("--model-id", choices=("sara-1.0", "lira-1.0"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--foundation-model", help="Explicit foundation model identifier")
    parser.add_argument("--foundation-path", type=Path, help="Explicit local foundation model path")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    train(
        data_path=args.data,
        output_path=args.output,
        model_id=args.model_id,
        foundation_model=args.foundation_model,
        foundation_path=args.foundation_path,
        epochs=args.epochs,
        max_length=args.max_length,
    )
    print(f"{get_model_spec(args.model_id).display_name} candidate written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
