import sys
from pathlib import Path

# Ensure project src is importable without installation
root = Path(__file__).resolve().parents[1]
src = root / "src"
if src not in map(Path, map(Path.resolve, map(Path, sys.path))):
    sys.path.insert(0, str(src))
