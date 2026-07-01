from pathlib import Path
import os
import re
from deadleaves import ImageRenderer

HERE = Path(__file__).resolve().parent
GALLERY_DIR = (HERE / "../examples/gallery").resolve()


def pytest_generate_tests(metafunc):
    if "py_file" in metafunc.fixturenames:
        py_files = sorted(GALLERY_DIR.rglob("*.py"))
        metafunc.parametrize(
            "py_file", py_files, ids=lambda p: str(p.relative_to(GALLERY_DIR))
        )


def test_example_execution(py_file: Path):
    text = py_file.read_text(encoding="utf-8")

    # remove HTML calls if applicable
    text = re.sub(r"HTML\(\s*ani\.to_jshtml\(\)\s*\)$", "", text, flags=re.MULTILINE)

    g = {}
    old_cwd = Path.cwd()
    os.chdir(py_file.parent)

    old_show = ImageRenderer.show

    def _no_show(*args, **kwargs):
        return None

    ImageRenderer.show = _no_show

    try:
        exec(text, g, g)
    finally:
        os.chdir(old_cwd)
        ImageRenderer.show = old_show
