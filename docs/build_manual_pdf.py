from __future__ import annotations

"""Build docs/Manual.pdf from docs/Manual.md.

This is a minimal renderer: it keeps monospaced blocks readable and supports
Chinese by using a CID font.

Run:
  python docs/build_manual_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    # crude wrapping that works well enough for mixed Chinese/English
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.rstrip("\n")
        if not s:
            lines.append("")
            continue
        while len(s) > max_chars:
            lines.append(s[:max_chars])
            s = s[max_chars:]
        lines.append(s)
    return lines


def build_pdf(md_path: Path, pdf_path: Path) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    w, h = A4
    left = 18 * mm
    right = 18 * mm
    top = 18 * mm
    bottom = 18 * mm

    font_main = "STSong-Light"
    font_code = "Courier"
    size_main = 10
    size_code = 9
    line_h = 4.2 * mm

    max_chars = 90  # empirical

    text = md_path.read_text(encoding="utf-8")
    in_code = False
    y = h - top

    def new_page() -> None:
        nonlocal y
        c.showPage()
        y = h - top

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        # Basic markdown cosmetics
        if not in_code:
            if line.startswith("#"):
                # headings: bigger, bold-ish by size
                lvl = len(line) - len(line.lstrip("#"))
                content = line.lstrip("#").strip()
                c.setFont(font_main, 16 - 2 * min(lvl - 1, 4))
                for wl in _wrap_text(content, max_chars=max_chars):
                    if y < bottom:
                        new_page()
                    c.drawString(left, y, wl)
                    y -= line_h
                y -= 1.5 * mm
                c.setFont(font_main, size_main)
                continue

        # Normal / code text
        if in_code:
            c.setFont(font_code, size_code)
            wrapped = _wrap_text(line, max_chars=max_chars)
        else:
            c.setFont(font_main, size_main)
            wrapped = _wrap_text(line, max_chars=max_chars)

        for wl in wrapped:
            if y < bottom:
                new_page()
                c.setFont(font_code if in_code else font_main, size_code if in_code else size_main)
            c.drawString(left, y, wl)
            y -= line_h

        # extra gap after blank lines
        if line.strip() == "":
            y -= 1.5 * mm

    c.save()


def main() -> None:
    here = Path(__file__).resolve().parent
    md = here / "Manual.md"
    pdf = here / "Manual.pdf"
    build_pdf(md, pdf)
    print(f"[OK] wrote: {pdf}")


if __name__ == "__main__":
    main()
