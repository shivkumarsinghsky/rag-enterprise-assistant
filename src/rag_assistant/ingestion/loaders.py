"""Convert source formats into normalised Markdown-like text so chunking can rely on headings and paragraphs."""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".html", ".htm"}


class _HtmlToText(HTMLParser):
    BLOCK = {"p", "div", "li", "tr", "br", "section", "article", "table"}
    SKIP = {"script", "style", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")
        elif tag == "li":
            self.parts.append("\n- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _HtmlToText()
    parser.feed(html)
    return "".join(parser.parts)


def normalise(text: str) -> str:
    """Unicode-normalise, drop control characters, collapse whitespace while keeping paragraph breaks."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def load_text(content: str, content_type: str) -> str:
    if content_type in ("text/html", ".html", ".htm"):
        return normalise(html_to_text(content))
    return normalise(content)


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"unsupported file type {suffix}; supported: {sorted(SUPPORTED)}")
    return load_text(path.read_text(encoding="utf-8"), suffix)


def title_from(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else fallback
