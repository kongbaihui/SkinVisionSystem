"""Portable, text-only PDF export. Untrusted report markup cannot load resources."""
import io
import os
import re
import threading
from pathlib import Path
from html import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _plain(line):
    line = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', line)
    return escape(line.replace('**', '').replace('`', ''))


_font_lock = threading.Lock()


def _font_name():
    # Embed a local TrueType CJK subset so downloaded reports render without
    # relying on the reader's system fonts. Linux uses fonts-wqy-zenhei.
    font = 'SkinVisionCJK'
    with _font_lock:
        if font in pdfmetrics.getRegisteredFontNames():
            return font
        candidates = [os.getenv('SKINVISION_PDF_FONT', ''),
                      str(Path(os.getenv('WINDIR', 'C:/Windows')) / 'Fonts/simsun.ttc'),
                      '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                      '/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc']
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                pdfmetrics.registerFont(TTFont(font, candidate, subfontIndex=0))
                return font
        raise RuntimeError('请配置 SKINVISION_PDF_FONT 为含中文字形的 TrueType 字体；Linux 可安装 fonts-wqy-zenhei')


def pdf_report(markdown):
    font = _font_name()
    body = ParagraphStyle('body', fontName=font, fontSize=10, leading=16,
                          wordWrap='CJK', spaceAfter=7, textColor=colors.HexColor('#1e293b'))
    headings = {i: ParagraphStyle(f'h{i}', parent=body, fontSize=21-i*3, leading=25-i*2,
                                  spaceBefore=12, spaceAfter=10, keepWithNext=True,
                                  textColor=colors.HexColor('#0e6476')) for i in range(1, 4)}
    code_style = ParagraphStyle('provenance', parent=body, fontSize=8, leading=11, spaceAfter=1)
    output, story = io.BytesIO(), []
    lines = markdown.splitlines()
    index = 0
    in_code = False
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            story.append(Paragraph(escape(line), code_style))
            continue
        if not line or line == '---':
            story.append(Spacer(1, 6))
            continue
        if line.startswith('|') and line.endswith('|'):
            rows = []
            while True:
                cells = [cell.strip() for cell in line.strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                    rows.append([Paragraph(_plain(cell), body) for cell in cells])
                if index >= len(lines) or not lines[index].strip().startswith('|'):
                    break
                line = lines[index].strip()
                index += 1
            if rows:
                columns = max(map(len, rows))
                rows = [row + ['']*(columns-len(row)) for row in rows]
                table = Table(rows, colWidths=[483/columns]*columns, repeatRows=1, hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8f2f4')),
                    ('GRID', (0,0), (-1,-1), .4, colors.HexColor('#d4e1e5')),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 8),
                    ('RIGHTPADDING', (0,0), (-1,-1), 8), ('TOPPADDING', (0,0), (-1,-1), 6),
                ]))
                story.extend([table, Spacer(1,10)])
            continue
        heading = re.match(r'^(#{1,3})\s+(.*)', line)
        if heading:
            story.append(Paragraph(_plain(heading[2]), headings[len(heading[1])]))
        else:
            story.append(Paragraph(_plain(line.lstrip('> ')), body))
    def footer(canvas, doc):
        canvas.setFont(font, 9)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(56, 28, 'SkinVision · 辅助分析记录')
        canvas.drawRightString(A4[0]-56, 28, str(doc.page))
    SimpleDocTemplate(output, pagesize=A4, leftMargin=56, rightMargin=56, topMargin=44,
                      bottomMargin=48, title='SkinVision 辅助分析报告').build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def export_pdf(markdown):
    if os.getenv('SKINVISION_PDF_ENGINE', 'reportlab') != 'weasyprint':
        return pdf_report(markdown)
    # Keep the previous optional renderer for existing installations. Escape HTML
    # and deny all external/file resource loads, including Markdown image URLs.
    import markdown as md
    import weasyprint
    def deny_resources(url, *args, **kwargs):
        raise ValueError('External PDF resources are disabled')
    html = md.markdown(escape(markdown), extensions=['tables'])
    return weasyprint.HTML(string=html, url_fetcher=deny_resources).write_pdf()
