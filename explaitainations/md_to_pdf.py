"""Convert all markdown files in this directory to styled PDFs."""
import re
import sys
from pathlib import Path
import markdown
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Unicode font paths (Arial Unicode covers em-dash, Greek letters, etc.)
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONT_REGULAR    = str(FONT_DIR / "Arial.ttf")
FONT_BOLD       = str(FONT_DIR / "Arial Bold.ttf")
FONT_ITALIC     = str(FONT_DIR / "Arial Italic.ttf")
FONT_BOLD_ITALIC= str(FONT_DIR / "Arial Bold Italic.ttf")
FONT_MONO       = str(FONT_DIR / "Courier New.ttf")
FONT_MONO_BOLD  = str(FONT_DIR / "Courier New Bold.ttf")

MD_DIR = Path(__file__).parent
FIG_DIR = (MD_DIR.parent / "output" / "figures").resolve()

PRIMARY   = (27, 79, 138)     # #1B4F8A
ACCENT    = (224, 123, 57)    # #E07B39
LIGHT_BG  = (247, 249, 252)
GRAY      = (120, 120, 120)
BLACK     = (26, 26, 26)
TABLE_HDR = (27, 79, 138)
TABLE_ALT = (240, 244, 250)
CODE_BG   = (244, 244, 244)

MARGIN    = 18
PAGE_W    = 210
CONTENT_W = PAGE_W - 2 * MARGIN


class MdPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register Unicode font family
        self.add_font("Arial", style="",  fname=FONT_REGULAR)
        self.add_font("Arial", style="B", fname=FONT_BOLD)
        self.add_font("Arial", style="I", fname=FONT_ITALIC)
        self.add_font("Arial", style="BI",fname=FONT_BOLD_ITALIC)
        self.add_font("Mono",  style="",  fname=FONT_MONO)
        self.add_font("Mono",  style="B", fname=FONT_MONO_BOLD)
        self._doc_title = ""

    def header(self):
        pass  # no running header

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 6, f"Streetsmart \u2014 {self._doc_title}  |  Page {self.page_no()}", align="C")

    def set_doc_title(self, title: str):
        self._doc_title = title


def resolve_image(src: str, md_path: Path) -> Path | None:
    """Try to find the image relative to the md file or in the figures dir."""
    # Try relative to md file
    candidate = (md_path.parent / src).resolve()
    if candidate.exists():
        return candidate
    # Try just the filename in figures dir
    fname = Path(src).name
    candidate2 = FIG_DIR / fname
    if candidate2.exists():
        return candidate2
    return None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())


def parse_table(lines: list[str]) -> list[list[str]] | None:
    """Parse a GFM table block into rows."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            return None
        if re.match(r"^\|[-| :]+\|$", line):
            continue  # separator row
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows if rows else None


def clean_inline(text: str) -> str:
    """Strip inline markdown to plain text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text


def render_inline(pdf: FPDF, text: str, base_font_size: float = 11):
    """Render text with bold/italic/code inline styling."""
    parts = re.split(r"(\*\*[^*]+?\*\*|\*[^*]+?\*|`[^`]+`)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font("Arial", "B", base_font_size)
            pdf.set_text_color(*BLACK)
            pdf.write(6, part[2:-2])
        elif part.startswith("*") and part.endswith("*"):
            pdf.set_font("Arial", "I", base_font_size)
            pdf.set_text_color(*BLACK)
            pdf.write(6, part[1:-1])
        elif part.startswith("`") and part.endswith("`"):
            pdf.set_font("Mono", "", base_font_size - 1)
            pdf.set_text_color(180, 40, 40)
            pdf.write(6, part[1:-1])
            pdf.set_text_color(*BLACK)
        else:
            # Handle markdown links → plain text
            part = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", part)
            pdf.set_font("Arial", "", base_font_size)
            pdf.set_text_color(*BLACK)
            pdf.write(6, part)


def render_paragraph(pdf: FPDF, text: str):
    pdf.set_x(MARGIN)
    render_inline(pdf, text, 11)
    pdf.ln(8)


def render_code_block(pdf: FPDF, lines: list[str]):
    block = "\n".join(lines)
    pdf.set_fill_color(*CODE_BG)
    pdf.set_draw_color(210, 210, 210)
    pdf.set_font("Mono", "", 9)
    pdf.set_text_color(50, 50, 50)
    line_h = 5
    block_h = len(lines) * line_h + 8
    x = MARGIN
    y = pdf.get_y()
    if y + block_h > pdf.h - 20:
        pdf.add_page()
        y = pdf.get_y()
    pdf.rect(x, y, CONTENT_W, block_h, style="FD")
    pdf.set_y(y + 4)
    for line in lines:
        pdf.set_x(MARGIN + 4)
        pdf.cell(CONTENT_W - 8, line_h, line[:110], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(pdf.get_y() + 4)
    pdf.set_text_color(*BLACK)


def render_table(pdf: FPDF, rows: list[list[str]]):
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    col_w = CONTENT_W / n_cols

    pdf.set_font("Arial", "B", 9.5)
    pdf.set_fill_color(*TABLE_HDR)
    pdf.set_text_color(255, 255, 255)
    pdf.set_draw_color(255, 255, 255)
    for cell in rows[0]:
        pdf.set_x(pdf.get_x()) if pdf.get_x() != MARGIN else None
        pdf.cell(col_w, 8, clean_inline(cell)[:50], border=0, fill=True,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln(8)

    pdf.set_font("Arial", "", 9.5)
    pdf.set_text_color(*BLACK)
    for i, row in enumerate(rows[1:]):
        fill = i % 2 == 0
        pdf.set_fill_color(*TABLE_ALT) if fill else pdf.set_fill_color(255, 255, 255)
        for cell in row:
            pdf.cell(col_w, 7, clean_inline(cell)[:60], border=0, fill=True,
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(7)
    pdf.ln(4)


def render_image(pdf: FPDF, src: str, md_path: Path):
    img_path = resolve_image(src, md_path)
    if not img_path:
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(*GRAY)
        pdf.set_x(MARGIN)
        pdf.cell(0, 6, f"[Image: {Path(src).name}]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*BLACK)
        return

    try:
        # Max width = content width; scale proportionally
        img_w = CONTENT_W
        remaining = pdf.h - pdf.get_y() - 25
        if remaining < 40:
            pdf.add_page()
        pdf.image(str(img_path), x=MARGIN, w=img_w)
        pdf.ln(4)
    except Exception as e:
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(*GRAY)
        pdf.set_x(MARGIN)
        pdf.cell(0, 6, f"[Could not embed image: {Path(src).name}]",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*BLACK)


def render_heading(pdf: FPDF, level: int, text: str):
    sizes   = {1: 22, 2: 17, 3: 14, 4: 12}
    top_gap = {1: 10, 2: 8,  3: 6,  4: 4}
    sz = sizes.get(level, 11)

    if level <= 2:
        remaining = pdf.h - pdf.get_y() - 25
        if remaining < 30:
            pdf.add_page()

    pdf.ln(top_gap.get(level, 4))
    pdf.set_x(MARGIN)

    if level == 1:
        pdf.set_font("Arial", "B", sz)
        pdf.set_text_color(*PRIMARY)
        pdf.multi_cell(CONTENT_W, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # underline rule
        pdf.set_draw_color(*PRIMARY)
        pdf.set_line_width(0.6)
        y = pdf.get_y()
        pdf.line(MARGIN, y, MARGIN + CONTENT_W, y)
        pdf.set_line_width(0.2)
        pdf.ln(5)
    elif level == 2:
        pdf.set_font("Arial", "B", sz)
        pdf.set_text_color(*PRIMARY)
        pdf.multi_cell(CONTENT_W, 9, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
    else:
        pdf.set_font("Arial", "B", sz)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(CONTENT_W, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    pdf.set_text_color(*BLACK)


def render_hr(pdf: FPDF):
    pdf.ln(4)
    pdf.set_draw_color(*GRAY)
    pdf.set_line_width(0.4)
    y = pdf.get_y()
    pdf.line(MARGIN, y, MARGIN + CONTENT_W, y)
    pdf.set_line_width(0.2)
    pdf.ln(6)


def convert_md_to_pdf(md_path: Path):
    text = md_path.read_text(encoding="utf-8")
    title = md_path.stem.replace("_", " ").title()

    pdf = MdPDF(orientation="P", unit="mm", format="A4")
    pdf.set_doc_title(title)
    pdf.set_margins(MARGIN, 15, MARGIN)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    lines = text.splitlines()
    i = 0
    bullet_buffer: list[tuple[int, str]] = []  # (indent_level, text)

    def flush_bullets():
        nonlocal bullet_buffer
        if not bullet_buffer:
            return
        for (indent, btxt) in bullet_buffer:
            x_off = MARGIN + indent * 5
            avail = CONTENT_W - indent * 5
            pdf.set_x(x_off)
            pdf.set_font("Arial", "", 10.5)
            pdf.set_text_color(*BLACK)
            bullet = "\u2022" if indent == 0 else "\u2013"
            # measure first line
            pdf.set_x(x_off)
            pdf.cell(5, 6, bullet)
            # render rest inline
            render_inline(pdf, btxt, 10.5)
            pdf.ln(6)
        bullet_buffer = []

    while i < len(lines):
        line = lines[i]

        # --- Heading ---
        hm = re.match(r"^(#{1,4})\s+(.*)", line)
        if hm:
            flush_bullets()
            render_heading(pdf, len(hm.group(1)), hm.group(2).strip())
            i += 1
            continue

        # --- HR ---
        if re.match(r"^---+$", line.strip()):
            flush_bullets()
            render_hr(pdf)
            i += 1
            continue

        # --- Fenced code block ---
        if line.strip().startswith("```"):
            flush_bullets()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # closing ```
            render_code_block(pdf, code_lines)
            continue

        # --- Image ---
        img_m = re.match(r"^!\[([^\]]*)\]\(([^\)]+)\)", line.strip())
        if img_m:
            flush_bullets()
            render_image(pdf, img_m.group(2), md_path)
            i += 1
            continue

        # --- Table ---
        if line.strip().startswith("|"):
            flush_bullets()
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            rows = parse_table(tbl_lines)
            if rows:
                render_table(pdf, rows)
            continue

        # --- Bullet (- or * or +) ---
        bm = re.match(r"^(\s*)([-*+])\s+(.*)", line)
        if bm:
            indent = len(bm.group(1)) // 2
            bullet_buffer.append((indent, bm.group(3)))
            i += 1
            continue

        # --- Numbered list ---
        nm = re.match(r"^(\s*)\d+\.\s+(.*)", line)
        if nm:
            indent = len(nm.group(1)) // 2
            bullet_buffer.append((indent, nm.group(2)))
            i += 1
            continue

        # --- Blockquote ---
        if line.strip().startswith(">"):
            flush_bullets()
            content = re.sub(r"^>\s?", "", line.strip())
            pdf.set_x(MARGIN + 6)
            pdf.set_fill_color(*LIGHT_BG)
            pdf.set_draw_color(*PRIMARY)
            pdf.set_line_width(1)
            y = pdf.get_y()
            pdf.line(MARGIN + 2, y, MARGIN + 2, y + 8)
            pdf.set_line_width(0.2)
            pdf.set_font("Arial", "I", 10.5)
            pdf.set_text_color(80, 80, 80)
            pdf.set_x(MARGIN + 6)
            pdf.multi_cell(CONTENT_W - 6, 6, clean_inline(content),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*BLACK)
            i += 1
            continue

        # --- Blank line ---
        if not line.strip():
            flush_bullets()
            if i > 0 and lines[i - 1].strip():
                pdf.ln(3)
            i += 1
            continue

        # --- Plain paragraph ---
        flush_bullets()
        pdf.set_x(MARGIN)
        render_inline(pdf, line, 11)
        pdf.ln(7)
        i += 1

    flush_bullets()

    out_path = md_path.with_suffix(".pdf")
    pdf.output(str(out_path))
    print(f"  ✓  {out_path.name}")


if __name__ == "__main__":
    md_files = sorted(MD_DIR.glob("*.md"))
    md_files = [f for f in md_files if f.name != "md_to_pdf.py"]
    print(f"Converting {len(md_files)} files...\n")
    errors = []
    for md_path in md_files:
        try:
            convert_md_to_pdf(md_path)
        except Exception as e:
            print(f"  ✗  {md_path.name}: {e}")
            errors.append(md_path.name)
    print(f"\nDone. {len(md_files) - len(errors)}/{len(md_files)} succeeded.")
