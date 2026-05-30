# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Render a project markdown doc to self-contained HTML via pandoc.

- MathJax for LaTeX math ($…$ inline, $$…$$ display).
- Embedded CSS for clean typography, no external assets.
- Edit constants below to render a different file. Per project
  convention: no CLI args.

Run:  uv run scripts/render_doc.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "docs" / "dggs-dnc-investigation.md"
OUT = REPO / "docs" / "dggs-dnc-investigation.html"
TITLE = "DGGS DNC investigation"


CSS = """
:root {
  --fg: #1f2328;
  --muted: #57606a;
  --accent: #0969da;
  --bg: #ffffff;
  --code-bg: #f6f8fa;
  --rule: #d0d7de;
  --table-stripe: #f8fafc;
}
html { font-size: 16px; }
body {
  margin: 0 auto;
  max-width: 56rem;
  padding: 2.5rem 1.25rem 4rem;
  color: var(--fg);
  background: var(--bg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.55;
}
h1, h2, h3, h4 { line-height: 1.25; margin-top: 2.2em; margin-bottom: 0.6em; }
h1 { font-size: 2rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.3em; margin-top: 0; }
h2 { font-size: 1.45rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.25em; }
h3 { font-size: 1.18rem; }
p, ul, ol, table, pre, blockquote { margin: 0.9em 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre, kbd, samp {
  font-family: "SF Mono", ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
  font-size: 0.92em;
}
code {
  background: var(--code-bg);
  padding: 0.12em 0.35em;
  border-radius: 4px;
}
pre {
  background: var(--code-bg);
  padding: 0.9em 1em;
  border-radius: 6px;
  overflow-x: auto;
  line-height: 1.45;
}
pre code { background: transparent; padding: 0; border-radius: 0; font-size: 0.88em; }
blockquote {
  border-left: 3px solid var(--rule);
  padding: 0.2em 1em;
  color: var(--muted);
  margin-left: 0;
}
ul, ol { padding-left: 1.6em; }
li { margin: 0.25em 0; }
table {
  border-collapse: collapse;
  display: block;
  overflow-x: auto;
  max-width: 100%;
}
table th, table td {
  border: 1px solid var(--rule);
  padding: 0.45em 0.8em;
  text-align: left;
}
table th { background: var(--code-bg); font-weight: 600; }
table tbody tr:nth-child(even) { background: var(--table-stripe); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2em 0; }
mjx-container { font-size: 1.02em !important; }
mjx-container[display="true"] { margin: 1.1em 0 !important; }
/* Suppress pandoc's title-block header (we already have an H1 in the doc). */
#title-block-header { display: none; }
/* Make right-aligned numeric table columns actually right-align. */
table td:has(mjx-container), table td.right, table th.right { text-align: right; }
"""


def render(src: Path, out: Path, title: str) -> None:
    if not shutil.which("pandoc"):
        sys.exit("error: pandoc not found on PATH")
    if not src.exists():
        sys.exit(f"error: source not found: {src}")

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(f"<style>{CSS}</style>\n")
        header = Path(f.name)

    try:
        cmd = [
            "pandoc",
            "--from=gfm+tex_math_dollars",
            "--to=html5",
            "--standalone",
            "--mathjax",
            f"--metadata=title:{title}",
            f"--include-in-header={header}",
            str(src),
            "-o", str(out),
        ]
        subprocess.run(cmd, check=True)
        print(f"wrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    finally:
        header.unlink(missing_ok=True)


if __name__ == "__main__":
    render(SRC, OUT, TITLE)
