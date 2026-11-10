"""Make the outputs of docs/atlas.md from the public PocketVina Atlas (run it again after a new release):

    python docs/atlas_outputs.py

It runs the page's code blocks in order, as a notebook would: the python blocks in one namespace, the bash blocks
(pocketvina commands) one command at a time, each language in its own empty working folder. After a block followed by
"<!-- output -->" it writes what the block prints to stdout (and the value of a python block's last line) into an
output block right after that line, replacing the one there; progress bars and log lines (stderr) are not shown.
After a block followed by "<!-- page: <file> -->" it copies the HTML page the block wrote to docs/<file>. Nothing is
written unless every block ran. Paths of the working folders become /path/to. Tables are shown as pandas does in a
console of WIDTH characters, with long text cut short. For the offline examples it downloads two proteins twice
(several hundred MB) into the temp folder ($TMPDIR), and leaves them there.
"""
import ast
import contextlib
import glob
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

WIDTH = 100  # width of a table (pandas)
LONGEST = 160  # longest output line; longer lines end in "..."
DOCS = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(DOCS, "atlas.md")
BLOCK = re.compile(r"```(python|bash)\n(.*?)```\n\n(<!-- output -->\n(?:\n```(?:text\n|\{code-block\} text\n:class: "
                   r"pv-output\n\n).*?```\n)?|<!-- page: (\S+) -->\n)", re.DOTALL)
OUTPUT = "<!-- output -->\n\n```{{code-block}} text\n:class: pv-output\n\n{}\n```\n"


def main():
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="pocketvina_cache_")
    os.environ.pop("POCKETVINA_ATLAS", None)
    folders = {"python": tempfile.mkdtemp(prefix="atlas_docs_"), "bash": tempfile.mkdtemp(prefix="atlas_docs_")}
    for option, value in (("width", WIDTH), ("max_columns", 9), ("max_colwidth", 28), ("expand_frame_repr", False)):
        pd.set_option(f"display.{option}", value)
    namespace: dict = {}
    text = open(PAGE, encoding="utf-8").read()
    pieces, pos, outputs, pages = [], 0, 0, []
    for m in BLOCK.finditer(text):
        language, code, marker, page = m.groups()
        folder = folders[language]
        before = set(glob.glob(os.path.join(folder, "*.html")))
        shown = run_python(code, namespace, folder) if language == "python" else run_bash(code, folder)
        for path in folders.values():
            shown = shown.replace(path, "/path/to")
        pieces.append(text[pos:m.start(3)])
        if page:
            written = sorted(set(glob.glob(os.path.join(folder, "*.html"))) - before)
            if len(written) != 1:
                raise SystemExit(f"the block before {marker.strip()} wrote {len(written)} pages, not 1")
            pages.append((written[0], os.path.join(DOCS, page)))
            pieces.append(marker)
        else:
            if not shown.strip():
                raise SystemExit(f"a block gave no output:\n{code}")
            lines = [line if len(line) <= LONGEST else line[:LONGEST - 3] + "..." for line in
                     shown.rstrip("\n").split("\n")]
            pieces.append(OUTPUT.format("\n".join(lines)))
            outputs += 1
        pos = m.end()
    pieces.append(text[pos:])
    for source, target in pages:
        shutil.copyfile(source, target)
    open(PAGE, "w", encoding="utf-8").write("".join(pieces))
    print(f"{PAGE}: {outputs} outputs written")


def run_python(code: str, namespace: dict, folder: str) -> str:
    """What a notebook cell shows: printed text, then the repr of the value of its last line (if not None)."""
    tree = ast.parse(code)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    out = io.StringIO()
    here = os.getcwd()
    os.chdir(folder)
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(tree, "<atlas.md>", "exec"), namespace)
            value = eval(compile(ast.Expression(last.value), "<atlas.md>", "eval"), namespace) if last else None
    finally:
        os.chdir(here)
    return out.getvalue() + (repr(value) + "\n" if value is not None else "")


def run_bash(code: str, folder: str) -> str:
    """What the pocketvina commands of a block print (stdout); each runs with this Python."""
    shown = ""
    for line in code.strip().splitlines():
        words = shlex.split(line)
        if words[:1] != ["pocketvina"]:
            raise SystemExit(f"only pocketvina commands can run here: {line}")
        done = subprocess.run([sys.executable, "-m", "pocketvina", *words[1:]], cwd=folder, capture_output=True,
                              text=True)
        if done.returncode:
            raise SystemExit(f"{line} failed:\n{done.stderr[-2000:]}")
        shown += done.stdout
    return shown


if __name__ == "__main__":
    main()
