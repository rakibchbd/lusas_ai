from __future__ import annotations

from pathlib import Path


EXCLUDED_PARTS = {".git", ".lusas", ".venv", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {
    ".c", ".cpp", ".css", ".go", ".html", ".java", ".js", ".json", ".md",
    ".py", ".rs", ".sh", ".sql", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}


class WorkspaceBoundaryError(ValueError):
    """Raised when an operation escapes the configured workspace."""


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        candidate_path = Path(relative_path)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise WorkspaceBoundaryError("Workspace paths must be relative.")

        resolved = (self.root / candidate_path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceBoundaryError("Path escapes the workspace.")
        return resolved

    def list_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if path.is_file() and not EXCLUDED_PARTS.intersection(relative.parts):
                files.append(path)
        return sorted(files)

    def snapshot(self, max_chars: int = 60_000) -> str:
        chunks: list[str] = []
        used = 0
        for path in self.list_files():
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            remaining = max_chars - used
            if remaining <= 0:
                break
            clipped = content[:remaining]
            relative = path.relative_to(self.root)
            chunks.append(f"--- {relative} ---\n{clipped}")
            used += len(clipped)
        return "\n\n".join(chunks) or "(workspace is empty)"

    def read(self, relative_path: str) -> str:
        return self.resolve(relative_path).read_text(encoding="utf-8")

    def write(self, relative_path: str, content: str) -> Path:
        destination = self.resolve(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination
