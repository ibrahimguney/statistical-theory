"""Runs all Python code of the book in order (does not modify the book files).

Usage (from this folder or from the book root):
    python3 code/run_all.py            # all
    python3 code/run_all.py 05 18      # only Chapters 5 and 18
The outputs (PDF) of code that produces figures are written to the ../figures/ folder.
"""
import os
import pathlib
import subprocess
import sys
import time

root = pathlib.Path(__file__).resolve().parent
figures = root.parent / "figures"; figures.mkdir(exist_ok=True)
selection = set(sys.argv[1:])
files = sorted(p for p in root.glob("ch*/*.py") if not selection or p.parent.name[-2:] in selection)
failed = []
for p in files:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(p)], cwd=figures, capture_output=True, text=True,
                       env={**os.environ, "MPLBACKEND": "Agg"})
    status = "ok" if r.returncode == 0 else "ERROR"
    print(f"[{status}] {p.relative_to(root)}  ({time.time()-t0:.1f} s)")
    if r.returncode:
        failed.append(p); print(r.stderr)
print(f"\n{len(files)-len(failed)}/{len(files)} files ran successfully.")
sys.exit(1 if failed else 0)
