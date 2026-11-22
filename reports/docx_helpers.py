"""Utilidades de python-docx para el reporte final (estilo de los reportes de las Fases 1 y 2).

Times New Roman 12, texto justificado, tablas con encabezado gris claro (#E2E8F0) y bordes #CBD5E1,
pies de figura "Figura N." + "Nota:" y títulos de tabla "Tabla N." (APA). Los marcadores [PENDIENTE: ...]
y las guías para los integrantes se resaltan en amarillo y se registran para el resumen final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

FONT = "Times New Roman"
MONO = "Consolas"
HEADER_FILL = "E2E8F0"
BORDER = "CBD5E1"
NAVY = RGBColor(0x1F, 0x29, 0x33)


@dataclass
class Pend:
    """Marcador [PENDIENTE: ...] resaltado en amarillo."""
    text: str


@dataclass
class Guide:
    """Texto guía para que un integrante escriba su parte (resaltado en amarillo)."""
    text: str


@dataclass
class Link:
    text: str
    url: str


@dataclass
class Ctx:
    doc: Any
    nums: Dict[str, int] = field(default_factory=dict)        # números de la pasada anterior
    assigned: Dict[str, int] = field(default_factory=dict)    # números asignados en esta pasada
    counters: Dict[str, int] = field(default_factory=lambda: {"fig": 0, "tab": 0})
    pending: List[str] = field(default_factory=list)
    guides: List[str] = field(default_factory=list)

    def ref(self, key: str) -> str:
        return str(self.nums.get(key, self.assigned.get(key, "?")))

    def next(self, kind: str, key: str) -> int:
        self.counters[kind] += 1
        self.assigned[key] = self.counters[kind]
        return self.counters[kind]


# --------------------------------------------------------------------------------------------- documento
def new_document() -> Any:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = normal.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    for name, size, color in (("Heading 1", 15, NAVY), ("Heading 2", 13, NAVY), ("Heading 3", 12, NAVY)):
        st = doc.styles[name]
        st.font.name = FONT
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.italic = False
        st.font.color.rgb = color
        rpr = st.element.get_or_add_rPr()
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.append(fonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            fonts.set(qn(attr), FONT)
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if fonts.get(qn(attr)) is not None:
                del fonts.attrib[qn(attr)]
        st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True
    # Idioma español (México) para la revisión ortográfica.
    lang = OxmlElement("w:lang")
    lang.set(qn("w:val"), "es-MX")
    normal.element.get_or_add_rPr().append(lang)
    return doc


def _field(run, instr: str, placeholder: str = "") -> None:
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    text = OxmlElement("w:instrText")
    text.set(qn("xml:space"), "preserve")
    text.text = instr
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    ph = OxmlElement("w:t")
    ph.set(qn("xml:space"), "preserve")
    ph.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, text, sep, ph, end):
        run._r.append(el)


def page_number_footer(doc, header_text: str) -> None:
    sec = doc.sections[0]
    sec.different_first_page_header_footer = True
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run(header_text)
    r.font.size, r.font.name, r.font.color.rgb = Pt(9), FONT, RGBColor(0x52, 0x60, 0x6D)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for part in ("Página ", None, " de ", "NUMPAGES"):
        run = fp.add_run()
        run.font.size, run.font.name = Pt(10), FONT
        if part is None:
            _field(run, "PAGE", "1")
        elif part == "NUMPAGES":
            _field(run, "NUMPAGES", "1")
        else:
            run.text = part


def toc(doc) -> None:
    p = doc.add_paragraph()
    _field(p.add_run(), 'TOC \\o "1-3" \\h \\z \\u',
           "Índice: en Word, clic derecho > Actualizar campo (o F9) para generarlo.")


def set_update_fields_on_open(doc) -> None:
    settings = doc.settings.element
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)


# ----------------------------------------------------------------------------------------------- texto
def _add_parts(ctx: Ctx, p, parts: Sequence[Any], size: Optional[float] = None, color=None) -> None:
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            _add_parts(ctx, p, part, size=size, color=color)
            continue
        if isinstance(part, Pend):
            r = p.add_run(f"[PENDIENTE: {part.text}]")
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            r.bold = True
            ctx.pending.append(part.text)
        elif isinstance(part, Guide):
            r = p.add_run(part.text)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            r.italic = True
            ctx.guides.append(part.text[:90])
        elif isinstance(part, Link):
            add_hyperlink(p, part.url, part.text, size=size)
            continue
        elif isinstance(part, tuple):
            kind, text = part[0], part[1]
            r = p.add_run(text)
            if "b" in kind:
                r.bold = True
            if "i" in kind:
                r.italic = True
            if "c" in kind:
                r.font.name = MONO
                r.font.size = Pt((size or 12) - 1.5)
        else:
            r = p.add_run(str(part))
        if size and not (isinstance(part, tuple) and "c" in part[0]):
            r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color


def para(ctx: Ctx, *parts, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, space_after=None, keep=False, style=None):
    p = ctx.doc.add_paragraph(style=style)
    p.alignment = align
    _add_parts(ctx, p, parts, size=size)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def bullet(ctx: Ctx, *parts, level: int = 0):
    p = ctx.doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3)
    _add_parts(ctx, p, parts)
    return p


def numbered(ctx: Ctx, *parts):
    p = ctx.doc.add_paragraph(style="List Number")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3)
    _add_parts(ctx, p, parts)
    return p


def _after_section_break(ctx: Ctx) -> bool:
    """True si lo último del documento es un salto de sección (que ya empieza página nueva).

    Un salto de página manual justo después dejaría una hoja en blanco (p. ej. al volver de una página horizontal).
    """
    body = ctx.doc.element.body
    last_p = [el for el in body if el.tag == qn("w:p")]
    if not last_p:
        return False
    ppr = last_p[-1].find(qn("w:pPr"))
    return ppr is not None and ppr.find(qn("w:sectPr")) is not None


def heading(ctx: Ctx, text: str, level: int = 1, page_break: bool = False):
    if page_break and not _after_section_break(ctx):
        ctx.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    return ctx.doc.add_heading(text, level=level)


def code(ctx: Ctx, text: str, size: float = 8.5):
    p = ctx.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Cm(0.4)
    _shade_paragraph(p, "F4F6F8")
    lines = text.rstrip("\n").split("\n")
    for i, line in enumerate(lines):
        r = p.add_run(line)
        r.font.name = MONO
        r.font.size = Pt(size)
        r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), MONO)
        if i < len(lines) - 1:
            r.add_break()
    return p


def _shade_paragraph(p, fill: str) -> None:
    ppr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.append(shd)


def add_hyperlink(p, url: str, text: str, size: Optional[float] = None):
    part = p.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                          is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), FONT)
    fonts.set(qn("w:hAnsi"), FONT)
    rpr.append(fonts)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1F5FA8")
    rpr.append(color)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(u)
    if size:
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), str(int(size * 2)))
        rpr.append(sz)
    run.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    run.append(t)
    link.append(run)
    p._p.append(link)
    return link


# ---------------------------------------------------------------------------------------------- tablas
def _cell_borders(cell) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for side in ("top", "bottom", "left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), BORDER)
        borders.append(el)
    tcpr.append(borders)


def _cell_fill(cell, fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def table_title(ctx: Ctx, key: str, title: str):
    n = ctx.next("tab", key)
    p = ctx.doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"Tabla {n}. ")
    r.bold = True
    r.font.size = Pt(10.5)
    r2 = p.add_run(title)
    r2.italic = True
    r2.font.size = Pt(10.5)


def table(ctx: Ctx, key: str, title: str, header: Sequence[str], rows: Sequence[Sequence[Any]],
          widths: Optional[Sequence[float]] = None, size: float = 9, note: Optional[Sequence[Any]] = None,
          align: Optional[Sequence[str]] = None, fills: Optional[Dict[tuple, str]] = None):
    """Tabla numerada. Celdas: str, Pend, Guide, Link o lista de partes (como `para`)."""
    table_title(ctx, key, title)
    t = ctx.doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    total = 6.5
    if widths is None:
        widths = [total / len(header)] * len(header)
    else:
        s = sum(widths)
        widths = [w * total / s for w in widths]
    for ri, values in enumerate([header] + [list(r) for r in rows]):
        row = t.rows[ri]
        trpr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trpr.append(cant)
        if ri == 0:
            th = OxmlElement("w:tblHeader")
            trpr.append(th)
        for ci, value in enumerate(values):
            cell = row.cells[ci]
            cell.width = Inches(widths[ci])
            _cell_borders(cell)
            if ri == 0:
                _cell_fill(cell, HEADER_FILL)
            elif fills and (ri - 1, ci) in fills:
                _cell_fill(cell, fills[(ri - 1, ci)])
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            a = (align[ci] if align else "l")
            p.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER,
                           "r": WD_ALIGN_PARAGRAPH.RIGHT}[a]
            parts = value if isinstance(value, list) else [value]
            if ri == 0:
                parts = [("b", str(v)) for v in parts]
            _add_parts(ctx, p, parts, size=size)
    if note:
        p = ctx.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(10)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run("Nota: ")
        r.italic = True
        r.font.size = Pt(9.5)
        _add_parts(ctx, p, note, size=9.5)
    else:
        ctx.doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def orientation(ctx: Ctx, landscape: bool) -> None:
    """Nueva sección (en página nueva) en horizontal o vertical; encabezado y pie se heredan."""
    from docx.enum.section import WD_ORIENT, WD_SECTION

    sec = ctx.doc.add_section(WD_SECTION.NEW_PAGE)
    sec.different_first_page_header_footer = False
    sec.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    w, h = (Inches(11), Inches(8.5)) if landscape else (Inches(8.5), Inches(11))
    sec.page_width, sec.page_height = w, h
    m = Inches(0.75) if landscape else Inches(1)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, m)


def figure(ctx: Ctx, key: str, path: Optional[Path], title: str, note: Sequence[Any], width: float = 6.3,
           missing: Optional[str] = None):
    n = ctx.next("fig", key)
    p = ctx.doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    if path is not None and Path(path).exists():
        p.add_run().add_picture(str(path), width=Inches(width))
    else:
        _add_parts(ctx, p, [Pend(missing or f"figura {title}")])
    cap = ctx.doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cap.paragraph_format.space_after = Pt(10)
    r = cap.add_run(f"Figura {n}. ")
    r.bold = True
    r.font.size = Pt(10)
    r = cap.add_run(title)
    r.italic = True
    r.font.size = Pt(10)
    r.add_break()
    r = cap.add_run("Nota: ")
    r.italic = True
    r.font.size = Pt(9.5)
    _add_parts(ctx, cap, note, size=9.5)
