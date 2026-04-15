import os
import sys
import subprocess
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

STORY_ID = os.environ.get("ARBIND_DID_QUARTO_ID")
TEAM_TOKEN = os.environ.get("TEAM_TOKEN_PROD")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QUARTO_DIR = REPO_ROOT / "quarto"
BOOK_DIR = QUARTO_DIR / "_book"

if __name__=="__main__":
    missing = [name for name, val in [("ARBIND_DID_QUARTO_ID", STORY_ID), ("TEAM_TOKEN_PROD", TEAM_TOKEN)] if val is None]
    if missing:
        print(f"Error: missing required environment variable(s): {', '.join(missing)}")
        sys.exit(1)

    # Runs the bash command: knatch STORY_ID _book TEAM_TOKEN --host datamarkedsplassen.intern.nav.no
    try:
        subprocess.run(
            ["knatch", STORY_ID, str(BOOK_DIR), TEAM_TOKEN, "--host", "datamarkedsplassen.intern.nav.no"],
            check=True
        )
        print("Publication successful!")
    except subprocess.CalledProcessError as e:
        print(f"Publication failed: {e}")
        sys.exit(1)