"""Sphinx configuration of the PocketVina documentation: MyST Markdown pages, the furo theme, the command reference
from the click commands (sphinx-click) and the Python API from the docstrings (autodoc).

    pip install -r docs/requirements.txt
    sphinx-build -W -b html docs docs/_build/html

The package is imported from this checkout, with its compiled and heavy dependencies (RDKit, OpenBabel, NumPy,
pandas, pyarrow) mocked, so the build needs neither them nor a GPU. ``docs/dev/`` (contracts between developers) is
not part of the documentation.
"""
import os
import re
import sys
from datetime import date

DOCS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DOCS)
sys.path.insert(0, ROOT)  # the pocketvina package of this checkout, also when another one is installed


def _version() -> str:
    # read, not imported: importing pocketvina needs RDKit, pandas, ... (mocked for autodoc only)
    with open(os.path.join(ROOT, "pocketvina", "__init__.py"), encoding="utf-8") as f:
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', f.read(), re.M)
    if not match:
        raise RuntimeError("no __version__ in pocketvina/__init__.py")
    return match.group(1)


project = "PocketVina"
author = "Ahmet Sarigun, Bora Uyar, Vedran Franke, Altuna Akalin"
copyright = f"2025-{date.today().year}, {author}"
release = _version()
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_click",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "dev", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_heading_anchors = 3
# option names stay "--option" (not an en dash) and quotes straight in the command reference
smartquotes = False

# imported at module level by pocketvina; DuckDB, PoseBusters, molviewspec and py3Dmol are imported lazily already
autodoc_mock_imports = ["numpy", "pandas", "pyarrow", "rdkit", "openbabel", "tqdm"]
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_default_options = {}
autoclass_content = "class"

html_theme = "furo"
html_title = f"PocketVina {release}"
html_short_title = "PocketVina"
html_theme_options = {
    "source_repository": "https://github.com/BIMSBbioinfo/PocketVina",
    "source_branch": "main",
    "source_directory": "docs/",
}
# the example 3D pages of visualisation.md (written by show() for the quick start run) and of atlas.md (by
# docs/atlas_outputs.py, from the public Atlas), and the style of output blocks (pocketvina.css)
html_static_path = ["_static"]
html_css_files = ["pocketvina.css"]
pygments_style = "github-light"  # code blocks in GitHub's colours, as on the PocketVina Atlas website
pygments_dark_style = "github-dark"

# --- docstrings and click help as reStructuredText -------------------------------------------------------------------

# docstrings that list fields or arguments as "name: text" lines with indented continuation lines (read as plain
# text by help()); rendered as a definition list, one entry per field
FIELD_DOCSTRINGS = {
    "pocketvina.main.PocketVinaConfig",
    *(f"pocketvina.main.PocketVinaGPU.{m}" for m in ("best_scores", "select", "show")),
    *(f"pocketvina.io.results.Results.{m}" for m in ("best_scores", "select", "show")),
    *(f"pocketvina.selection.Selection.{m}" for m in ("show", "bust", "export", "export_protein")),
    "pocketvina.atlas.Atlas",
    *(f"pocketvina.atlas.Atlas.{m}" for m in ("protein", "search", "drug", "best_scores", "select", "show",
                                              "download")),
}
_FIELD_ITEM = re.compile(r"^([A-Za-z_]\w*(?:, [A-Za-z_]\w*)*): (\S.*)$")
# a "*" that starts a file pattern in click help ("*.pdb"), not emphasis
_GLOB_STAR = re.compile(r"(?<![\\\w*`])\*(?=[\w.])")
# a name in single backticks in click help (as Markdown writes code: "`prank`"), not reST's default role (italics)
_CODE_TICKS = re.compile(r"(?<![`\w])`([^`\s][^`]*)`(?![`\w])")


def _field_definitions(app, what, name, obj, options, lines):
    # "PV_" (the prefix of PocketVina's file names) in plain text, not a reST reference named "PV"
    lines[:] = [line.replace('"PV_"', '"PV\\_"') for line in lines]
    if name not in FIELD_DOCSTRINGS:
        return
    out, i = [], 0
    while i < len(lines):
        match = _FIELD_ITEM.match(lines[i])
        if not match:
            out.append(lines[i])
            i += 1
            continue
        if out and out[-1].strip():
            out.append("")
        out.append(", ".join(f"``{field}``" for field in match.group(1).split(", ")))
        out.append("   " + match.group(2))
        i += 1
        while i < len(lines) and lines[i][:1].isspace() and lines[i].strip():
            out.append("   " + lines[i].strip())
            i += 1
        out.append("")
    lines[:] = out


def _module_examples(app, what, name, obj, options, lines):
    """Indented examples in module docstrings (written for help(), without a "::") as literal blocks, not block
    quotes that run the lines together."""
    if what != "module":
        return
    out = []
    for line in lines:
        if line[:1].isspace() and line.strip() and out and not out[-1].strip():
            text = [previous for previous in out if previous.strip()]
            if text and not text[-1][:1].isspace() and not text[-1].rstrip().endswith("::"):
                out.extend(["::", ""])
        out.append(line)
    lines[:] = out


def _click_text(line: str) -> str:
    return _CODE_TICKS.sub(r"``\1``", _GLOB_STAR.sub(r"\\*", line))


def _click_globs(app, ctx, lines):
    lines[:] = [_click_text(line) for line in lines]


def _click_epilog(app, ctx, lines):
    """Paragraphs that click prints as they are (marked with ``\\b``; sphinx-click makes them "| " line blocks) as
    literal blocks: they hold examples and aligned text."""
    out, block = [], []

    def flush():
        if block:
            out.extend(["::", ""] + [f"    {line}" if line else "" for line in block] + [""])
            block.clear()

    for line in lines:
        if line.startswith("| ") or line == "|":
            block.append(line[2:])
        else:
            flush()
            out.append(_click_text(line))
    flush()
    lines[:] = out


def setup(app):
    app.connect("autodoc-process-docstring", _field_definitions)
    app.connect("autodoc-process-docstring", _module_examples)
    for event in ("description", "options", "arguments"):
        app.connect(f"sphinx-click-process-{event}", _click_globs)
    app.connect("sphinx-click-process-epilog", _click_epilog)
