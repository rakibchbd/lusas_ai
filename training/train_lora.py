from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} is not an object.")
        instruction = record.get("instruction")
        output = record.get("output")
        if not isinstance(instruction, str) or not isinstance(output, str):
            raise ValueError(
                f"Line {line_number} needs string instruction and output fields."
            )
        records.append(
            {
                "text": (
                    "### Instruction:\n"
                    f"{instruction}\n\n"
                    "### Response:\n"
                    f"{output}"
                )
            }
        )
    if not records:
        raise ValueError("The training dataset is empty.")
    return records


def train(
    data_path: Path,
    output_path: Path,
    base_model: str,
    epochs: float,
    max_length: int,
) -> None:
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
            "Training dependencies are missing. Install training/requirements.txt."
        ) from exc

    records = load_records(data_path)
    dataset = Dataset.from_list(records)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    if torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
    )
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
        ),
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a LUSAS LoRA adapter from local training data."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    train(
        data_path=args.data,
        output_path=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        max_length=args.max_length,
    )
    print(f"Candidate model written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
