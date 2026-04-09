#!/usr/bin/env python3
"""
Render RCP paper from Markdown to PDF via HTML + WeasyPrint.
Proven process. Do not replace with reportlab, pandoc, pdfkit, or anything else.
"""

import sys
import os
import re
import base64
from pathlib import Path

import markdown
from weasyprint import HTML


def read_and_embed_images(md_text, md_dir):
    """Replace image paths with base64 data URIs so HTML is self-contained."""
    def replace_img(match):
        alt = match.group(1)
        path = match.group(2)
        full_path = os.path.join(md_dir, path)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = Path(path).suffix.lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".svg": "image/svg+xml"}.get(ext, "image/png")
            return f"![{alt}](data:{mime};base64,{data})"
        else:
            print(f"  WARNING: image not found: {full_path}")
            return match.group(0)
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, md_text)


def md_to_html(md_text):
    extensions = ["tables", "footnotes", "toc", "attr_list", "md_in_html"]
    extension_configs = {"footnotes": {"BACKLINK_TEXT": "&#8617;"}}
    return markdown.markdown(md_text, extensions=extensions,
                             extension_configs=extension_configs,
                             output_format="html5")


CSS = """
@page {
    size: letter;
    margin: 1in;
    @bottom-center {
        content: counter(page);
        font-family: "Times New Roman", Times, serif;
        font-size: 10pt;
        color: #666;
    }
}
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
}
h1 { font-size: 16pt; text-align: center; margin-top: 0; margin-bottom: 0.3em; page-break-after: avoid; }
h2 { font-size: 13pt; margin-top: 1.5em; margin-bottom: 0.5em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 1.2em; margin-bottom: 0.4em; page-break-after: avoid; }
h1 + p { text-align: center; font-size: 11pt; margin-bottom: 1em; }
p { margin-top: 0.4em; margin-bottom: 0.4em; orphans: 2; widows: 2; }
table { border-collapse: collapse; margin: 1em auto; font-size: 9.5pt; width: auto; }
th, td { border: 1px solid #999; padding: 4px 8px; text-align: left; }
th { background-color: #f0f0f0; font-weight: bold; }
code { font-family: "Courier New", monospace; font-size: 9pt; background-color: #f5f5f5; padding: 1px 3px; }
pre { background-color: #f5f5f5; padding: 8px; border: 1px solid #ddd; font-size: 8.5pt; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
img { max-width: 100%; display: block; margin: 1em auto; }
a { color: #1a5276; text-decoration: none; }
.footnote { font-size: 9pt; margin-top: 2em; border-top: 1px solid #ccc; padding-top: 0.5em; }
.footnote ol { padding-left: 1.5em; }
.footnote li { margin-bottom: 0.3em; }
sup { font-size: 8pt; line-height: 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: 1em; color: #555; font-style: italic; }
ul, ol { margin-top: 0.3em; margin-bottom: 0.3em; padding-left: 2em; }
li { margin-bottom: 0.2em; }
h1, h2, h3 { orphans: 3; widows: 3; }
"""


md_path = Path(sys.argv[1])
md_dir = str(md_path.parent)
pdf_path = md_path.with_suffix(".pdf")

print(f"Reading {md_path}...")
md_text = md_path.read_text(encoding="utf-8")

print("Embedding images...")
md_text = read_and_embed_images(md_text, md_dir)

# Convert the single author footnote to inline text (WeasyPrint can't make
# footnote internal links clickable in PDF)
fn_match = re.search(r'^\[\^1\]:\s*(.+)$', md_text, re.MULTILINE)
if fn_match:
    fn_text = fn_match.group(1).strip()
    md_text = md_text.replace('[^1]', '')
    md_text = md_text[:fn_match.start()] + md_text[fn_match.end():]
    md_text = md_text.replace('Declan Michaels\n', f'Declan Michaels\n\n*{fn_text}*\n', 1)

print("Converting to HTML...")
html_body = md_to_html(md_text)

# Fix footnote anchors: colons in IDs break PDF internal links
html_body = html_body.replace('id="fn:', 'id="fn-')
html_body = html_body.replace('href="#fn:', 'href="#fn-')
html_body = html_body.replace('id="fnref:', 'id="fnref-')
html_body = html_body.replace('href="#fnref:', 'href="#fnref-')

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body>
</html>"""

print(f"Rendering PDF to {pdf_path}...")
HTML(string=html_doc, base_url=md_dir).write_pdf(str(pdf_path))
print(f"Done: {pdf_path}")
