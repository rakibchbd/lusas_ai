from __future__ import annotations

import json
from pathlib import Path


class LearningStore:
    """Stores user-approved examples for the next local training cycle."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def add(self, instruction: str, output: str) -> None:
        instruction = instruction.strip()
        output = output.strip()
        if not instruction or not output:
            raise ValueError("Both instruction and output are required.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"instruction": instruction, "output": output},
                    ensure_ascii=True,
                )
                + "\n"
            )

    def examples(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        examples: list[dict[str, str]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("instruction"), str)
                or not isinstance(record.get("output"), str)
            ):
                raise ValueError(f"Invalid learned example on line {line_number}.")
            examples.append(
                {
                    "instruction": record["instruction"],
                    "output": record["output"],
                }
            )
        return examples
