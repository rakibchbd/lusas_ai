from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser


DEFAULT_SOURCES = (
    "https://docs.python.org/3/",
    "https://developer.mozilla.org/en-US/docs/Web/",
    "https://pytorch.org/docs/stable/",
)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer", "header"}:
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "header"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


@dataclass(frozen=True)
class WebCollector:
    pending_path: Path
    user_agent: str = "LUSAS-AI-LearningBot/1.0"
    max_bytes: int = 512_000
    delay_seconds: float = 5.0

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).hostname
        return host in {"docs.python.org", "developer.mozilla.org", "pytorch.org"}

    def _robots_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        robots = RobotFileParser()
        robots.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        try:
            robots.read()
        except OSError:
            return False
        return robots.can_fetch(self.user_agent, url)

    def collect(self, sources: tuple[str, ...] = DEFAULT_SOURCES) -> int:
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        collected = 0
        for url in sources:
            if not self._allowed(url) or not self._robots_allowed(url):
                continue
            request = Request(url, headers={"User-Agent": self.user_agent})
            with urlopen(request, timeout=20) as response:
                raw = response.read(self.max_bytes + 1)
            if len(raw) > self.max_bytes:
                continue
            parser = _TextParser()
            parser.feed(raw.decode("utf-8", errors="replace"))
            text = " ".join(parser.parts)
            if len(text) < 200:
                continue
            with self.pending_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "instruction": f"Summarize the technical information from {url}.",
                            "output": text[:20_000],
                            "source": url,
                            "status": "pending_review",
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
            collected += 1
            time.sleep(self.delay_seconds)
        return collected
