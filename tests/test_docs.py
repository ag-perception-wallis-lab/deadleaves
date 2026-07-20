import re
import os
import pytest
from pathlib import Path
from deadleaves import ImageRenderer
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DOCS_DIR = (HERE / "../docs").resolve()


def pytest_generate_tests(metafunc):
    if "md_file" in metafunc.fixturenames:
        md_files = sorted(DOCS_DIR.rglob("*.md"))
        metafunc.parametrize(
            "md_file", md_files, ids=lambda p: str(p.relative_to(DOCS_DIR))
        )


def test_code_cell_execution(md_file: Path):
    text = md_file.read_text(encoding="utf-8")

    pattern = re.compile(r"```{code-cell}\n(?::tags:.*\n*)*((?:.+?\s+?)+?)(?=```)")
    code_cells = pattern.findall(text)

    if not code_cells:
        pytest.skip("No code cells found.")

    g = {}
    old_cwd = Path.cwd()
    os.chdir(md_file.parent)

    old_dl_show = ImageRenderer.show
    old_plt_show = plt.show

    def _no_show(*args, **kwargs):
        return None

    ImageRenderer.show = _no_show
    plt.show = _no_show

    try:
        for code_cell in code_cells:
            exec(code_cell, g, g)
    finally:
        os.chdir(old_cwd)
        ImageRenderer.show = old_dl_show
        plt.show = old_plt_show
