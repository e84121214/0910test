from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.oxml.ns import qn


path = Path(r"C:\Users\CCAM144F\Desktop\專題\0910test\書審資料_母版初稿.docx")
issues = []

with ZipFile(path) as zf:
    bad = zf.testzip()
    if bad:
        issues.append(f"corrupt zip member: {bad}")
    required = {"word/document.xml", "[Content_Types].xml", "word/styles.xml"}
    missing = required.difference(zf.namelist())
    if missing:
        issues.append(f"missing OOXML parts: {sorted(missing)}")

doc = Document(path)
text = "\n".join(p.text for p in doc.paragraphs)

for token in ("turn0", "codex-file-citation", "TODO", "lorem ipsum"):
    if token.lower() in text.lower():
        issues.append(f"unexpected token: {token}")

headings = []
for p in doc.paragraphs:
    if p.style and p.style.name.startswith("Heading"):
        headings.append((p.style.name, p.text))
        if not p.text.strip():
            issues.append("empty heading")

for ti, table in enumerate(doc.tables, 1):
    if not table.rows or not table.columns:
        issues.append(f"table {ti} has no rows or columns")
        continue
    expected = len(table.columns)
    for ri, row in enumerate(table.rows, 1):
        if len(row.cells) != expected:
            issues.append(f"table {ti} row {ri} column mismatch")
        for ci, cell in enumerate(row.cells, 1):
            if not cell.text.strip():
                issues.append(f"table {ti} row {ri} cell {ci} is empty")

section = doc.sections[0]
usable_width = section.page_width - section.left_margin - section.right_margin

print({
    "file_bytes": path.stat().st_size,
    "paragraphs": len(doc.paragraphs),
    "tables": len(doc.tables),
    "headings": len(headings),
    "sections": len(doc.sections),
    "usable_width_twips": usable_width,
    "issues": issues,
})
