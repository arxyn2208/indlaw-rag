"""Turn pdftotext output of Supreme Court Reports PDFs into indlaw .txt files.

Usage: python scripts/prepare_sc.py IN_DIR OUT_DIR
The title is the party block above the first line that is just "v.".
Extraction is plain pdftotext: margin letters (A, B, C) and headnote text remain in the body.
"""
import os
import re
import sys

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
for name in sorted(os.listdir(src)):
    if not name.endswith(".txt"):
        continue
    with open(os.path.join(src, name), encoding="utf-8", errors="replace") as handle:
        lines = [ln.strip() for ln in handle]
    title = name[:-4]
    for i, ln in enumerate(lines):
        if ln in ("v.", "vs.", "versus"):
            before = [re.sub(r"^[A-H]\s+", "", x) for x in lines[max(0, i - 4):i] if x and not re.match(r"^\d+\s|REPORTS|SUPREME", x)]
            after = [x for x in lines[i + 1:i + 4] if x][:1]
            title = " ".join(before[-1:] + ["v."] + after)
            title = re.sub(r"\s+", " ", title)
            title = re.sub(r"\s[A-H] v\. ", " v. ", title)
            break
    body = "\n".join(lines)
    with open(os.path.join(out, name), "w", encoding="utf-8") as handle:
        handle.write(f"CASE TITLE: {title}\n\n{body}\n")
    print(name, "->", title)
