import os
import subprocess
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
BUILD = DOCS / "_build"
HTML = DOCS / "_build" / "html"

BUILD.mkdir(exist_ok=True)
HTML.mkdir(exist_ok=True)

# Build the Jupyter Book
print("📚 Building HTML files...")
subprocess.run(["sphinx-build", "-M", "html", str(DOCS), str(BUILD)])

# Start local server
print("🚀 Starting local server at http://localhost:8000")

os.chdir(HTML)
# webbrowser.open("http://localhost:8000")

subprocess.run([os.sys.executable, "-m", "http.server"])
