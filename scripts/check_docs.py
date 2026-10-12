from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
IGNORED_SCHEMES = {"http", "https", "mailto", "tel", "data"}


def local_link_errors(root: Path) -> list[str]:
    errors: list[str] = []
    documents = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    for document in documents:
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            raw_target = match.group(1).strip().strip("<>")
            target = urlsplit(raw_target)
            if target.scheme.lower() in IGNORED_SCHEMES or raw_target.startswith("#"):
                continue
            relative_path = unquote(target.path)
            if not relative_path:
                continue
            resolved = (document.parent / relative_path).resolve()
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{document.relative_to(root)}:{line}: missing {relative_path}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    errors = local_link_errors(root)
    if errors:
        print("Broken local documentation links:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Local documentation links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
