"""Make the dashboard's text bigger: scales every font-size (CSS, inline styles and JS)
in static/index.html and widens the side panels to fit. Keeps a backup.

Run from the project folder:

    python scripts/bigger_fonts.py        # 1.3x
    python scripts/bigger_fonts.py 1.5    # 1.5x
    python scripts/bigger_fonts.py undo   # restore the backup
"""
import re, shutil, sys
from pathlib import Path

page = Path("static/index.html")
backup = page.with_name("index.orig.html")
arg = sys.argv[1] if len(sys.argv) > 1 else "1.3"

if arg == "undo":
    shutil.copy(backup, page); print("restored", page); sys.exit()
if not backup.exists():
    shutil.copy(page, backup)          # always scale from the original, never twice
k = float(arg)
src = backup.read_text(encoding="utf-8")

def px(m):
    v = float(m.group(2)) * k
    return f"{m.group(1)}{round(v, 1):g}px"

out = re.sub(r"(font-size:\s*)(\d+(?:\.\d+)?)px", px, src)                 # CSS + inline styles
out = re.sub(r"(fontSize\s*=\s*['\"])(\d+(?:\.\d+)?)px", px, out)          # JS el.style.fontSize='10px'
out = re.sub(r"(grid-template-columns:\s*)(\d+)px 1fr (\d+)px",           # side panels
             lambda m: f"{m.group(1)}{round(int(m.group(2)) * k)}px 1fr {round(int(m.group(3)) * k)}px", out)
out = re.sub(r"(\.subheader\{[^}]*?height:)(\d+)px",                        # bar under the header
             lambda m: f"{m.group(1)}{round(int(m.group(2)) * k)}px", out)
page.write_text(out, encoding="utf-8")
print(f"fonts x{k} written to {page} (backup: {backup})")
