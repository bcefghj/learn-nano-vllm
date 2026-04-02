#!/usr/bin/env python3
"""
nano-vllm 面试学习指南 - PDF/HTML 生成脚本

使用方法:
    pip install markdown jinja2 weasyprint
    python scripts/generate_pdf_html.py

会在 output/ 目录下生成:
    - index.html (单页HTML，包含所有课程)
    - 各课程单独HTML
    - nano-vllm-interview-guide.pdf (完整PDF)
"""

import os
import sys
import glob
import markdown
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"
IMAGES_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --bg: #ffffff;
            --text: #1e293b;
            --code-bg: #f1f5f9;
            --border: #e2e8f0;
            --accent: #3b82f6;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                         "Microsoft YaHei", sans-serif;
            line-height: 1.8;
            color: var(--text);
            background: var(--bg);
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}
        h1 {{
            font-size: 2rem;
            color: var(--primary);
            border-bottom: 3px solid var(--accent);
            padding-bottom: 0.5rem;
            margin: 2rem 0 1rem;
        }}
        h2 {{
            font-size: 1.5rem;
            color: var(--primary);
            margin: 1.5rem 0 0.8rem;
            border-left: 4px solid var(--accent);
            padding-left: 0.8rem;
        }}
        h3 {{
            font-size: 1.2rem;
            margin: 1.2rem 0 0.6rem;
        }}
        p {{ margin: 0.6rem 0; }}
        code {{
            background: var(--code-bg);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
            font-family: "Fira Code", "Source Code Pro", monospace;
        }}
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            margin: 1rem 0;
            line-height: 1.5;
        }}
        pre code {{
            background: transparent;
            padding: 0;
            color: inherit;
        }}
        blockquote {{
            border-left: 4px solid var(--accent);
            padding: 0.5rem 1rem;
            margin: 1rem 0;
            background: #eff6ff;
            border-radius: 0 8px 8px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            border: 1px solid var(--border);
            padding: 0.6rem 1rem;
            text-align: left;
        }}
        th {{
            background: #f0f9ff;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ background: #f8fafc; }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        ul, ol {{ padding-left: 1.5rem; margin: 0.5rem 0; }}
        li {{ margin: 0.3rem 0; }}
        a {{ color: var(--accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .toc {{
            background: #f0f9ff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        .toc h2 {{ border: none; padding: 0; margin-top: 0; }}
        .toc ul {{ list-style: none; padding: 0; }}
        .toc li {{ margin: 0.4rem 0; }}
        .toc a {{ font-weight: 500; }}
        .page-break {{ page-break-before: always; }}
        hr {{ border: none; border-top: 2px solid var(--border); margin: 2rem 0; }}
        .nav {{
            display: flex;
            justify-content: space-between;
            padding: 1rem 0;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }}
        .nav a {{
            padding: 0.5rem 1rem;
            background: var(--accent);
            color: white;
            border-radius: 6px;
        }}
        @media print {{
            body {{ max-width: 100%; padding: 1rem; }}
            pre {{ white-space: pre-wrap; }}
            .nav {{ display: none; }}
        }}
    </style>
</head>
<body>
{content}
</body>
</html>
"""

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>nano-vllm 面试导向学习指南</title>
    <style>
        :root {{ --primary: #2563eb; --accent: #3b82f6; --bg: #f8fafc; }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
            background: var(--bg);
            color: #1e293b;
        }}
        .hero {{
            background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
        }}
        .hero h1 {{ font-size: 2.5rem; margin-bottom: 1rem; }}
        .hero p {{ font-size: 1.2rem; opacity: 0.9; }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 2rem; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        .card h3 {{ color: var(--primary); margin-bottom: 0.5rem; }}
        .card a {{ text-decoration: none; color: inherit; display: block; }}
        .section-title {{
            font-size: 1.5rem;
            color: var(--primary);
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--accent);
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }}
        .badge {{
            display: inline-block;
            background: #eff6ff;
            color: var(--accent);
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 0.3rem;
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>nano-vllm 面试导向学习指南</h1>
        <p>20节课程 | 60+面试题 | STAR面试法 | 哆啦A梦风格漫画</p>
    </div>
    <div class="container">
        <h2 class="section-title">课程目录</h2>
        <div class="grid">
{course_cards}
        </div>
        <h2 class="section-title">面试准备</h2>
        <div class="grid">
{interview_cards}
        </div>
    </div>
</body>
</html>
"""


def md_to_html(md_content: str) -> str:
    extensions = ["tables", "fenced_code", "codehilite", "toc", "nl2br"]
    return markdown.markdown(md_content, extensions=extensions)


def generate_single_html(md_file: Path, output_dir: Path):
    content = md_file.read_text(encoding="utf-8")
    title = content.split("\n")[0].lstrip("# ").strip()
    html_body = md_to_html(content)
    html = HTML_TEMPLATE.format(title=title, content=html_body)
    out_name = md_file.stem + ".html"
    (output_dir / out_name).write_text(html, encoding="utf-8")
    return out_name, title


def generate_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print("No markdown files found in docs/")
        sys.exit(1)

    print(f"Found {len(md_files)} markdown files")

    course_cards = []
    interview_cards = []

    for md_file in md_files:
        out_name, title = generate_single_html(md_file, OUTPUT_DIR)
        print(f"  Generated: {out_name}")

        card = f'            <div class="card"><a href="{out_name}"><h3>{title}</h3></a></div>'
        num = md_file.stem.split("-")[0]
        if num.isdigit() and int(num) <= 20:
            course_cards.append(card)
        else:
            interview_cards.append(card)

    index_html = INDEX_TEMPLATE.format(
        course_cards="\n".join(course_cards),
        interview_cards="\n".join(interview_cards),
    )
    (OUTPUT_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print("  Generated: index.html")

    all_content = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        all_content.append(md_to_html(content))
        all_content.append('<hr class="page-break">')

    full_html = HTML_TEMPLATE.format(
        title="nano-vllm 面试导向学习指南 - 完整版",
        content="\n".join(all_content),
    )
    (OUTPUT_DIR / "full-guide.html").write_text(full_html, encoding="utf-8")
    print("  Generated: full-guide.html")

    try:
        from weasyprint import HTML as WPHTML
        WPHTML(string=full_html).write_pdf(str(OUTPUT_DIR / "nano-vllm-interview-guide.pdf"))
        print("  Generated: nano-vllm-interview-guide.pdf")
    except (ImportError, OSError) as e:
        print(f"  [SKIP] PDF generation skipped: {type(e).__name__}")
        print("  To generate PDF, install weasyprint and its system dependencies:")
        print("    brew install pango (macOS) / apt install libpango1.0-dev (Ubuntu)")
        print("    pip install weasyprint")
        print("  Or open full-guide.html in a browser and use Print -> Save as PDF")

    print(f"\nDone! Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all()
