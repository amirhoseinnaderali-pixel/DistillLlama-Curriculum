from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()

    findings = []
    root = Path(args.root)

    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            continue

        if any(pattern.search(text) for pattern in PATTERNS):
            findings.append(str(path))

    if findings:
        print("\n".join(sorted(set(findings))))
        raise SystemExit(1)

    print("No obvious credential patterns found.")


if __name__ == "__main__":
    main()
