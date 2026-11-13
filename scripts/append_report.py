"""Append the generated "Final Quantitative Results" section to PROJECT_PROGRESS.md (once).

Idempotent: if the heading is already present the script does nothing, so repeated runs cannot
pile up duplicate sections. Reads results/summary.jsonl (read-only).
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

HEADING = "## Final Quantitative Results"


def main() -> int:
    path = Path("PROJECT_PROGRESS.md")
    if path.exists() and HEADING in path.read_text(encoding="utf-8"):
        print(f"{path} already contains '{HEADING}'; nothing to do.")
        return 0

    from scripts.generate_final_report import generate_report
    f = io.StringIO()
    with redirect_stdout(f):
        generate_report()

    with open(path, "a", encoding="utf-8") as out:
        out.write(f"\n{HEADING}\n")
        out.write(f.getvalue())
    print(f"Appended '{HEADING}' to {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
