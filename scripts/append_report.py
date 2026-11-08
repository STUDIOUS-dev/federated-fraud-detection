import io
from contextlib import redirect_stdout
from scripts.generate_final_report import generate_report

f = io.StringIO()
with redirect_stdout(f):
    generate_report()
report_text = f.getvalue()

with open("PROJECT_PROGRESS.md", "a") as out:
    out.write("\n## Final Quantitative Results\n")
    out.write(report_text)
