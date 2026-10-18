import sys
from pathlib import Path

from streamlit.web.cli import main as streamlit_main


def main() -> None:
    app_path = Path(__file__).resolve().parents[2] / "frontend.py"
    if not app_path.is_file():
        raise FileNotFoundError(f"Streamlit app not found: {app_path}")

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "127.0.0.1",
        *sys.argv[1:],
    ]
    streamlit_main()
