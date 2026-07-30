"""Embed the project print stylesheet in an nbconvert-generated HTML notebook."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Inject CSS before the closing head tag of one HTML document."""

    parser = argparse.ArgumentParser()
    parser.add_argument("html_path", type=Path)
    parser.add_argument("css_path", type=Path)
    args = parser.parse_args()
    html = args.html_path.read_text(encoding="utf-8")
    css = args.css_path.read_text(encoding="utf-8")
    style = f'\n<style id="group148-print-style">\n{css}\n</style>\n'
    if "</head>" not in html:
        raise ValueError("Cannot inject print CSS: closing head tag not found.")
    args.html_path.write_text(html.replace("</head>", f"{style}</head>", 1), encoding="utf-8")


if __name__ == "__main__":
    main()
