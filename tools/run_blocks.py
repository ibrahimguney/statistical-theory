r"""Runs the Python files referenced by \pycode{code/...py} in the chapter .tex files and inserts
or refreshes the [style=output] block right below each one. Figure-producing code is run inside
figures/ (savefig("x.pdf") -> figures/x.pdf).

Usage (from the book root):
    python3 tools/run_blocks.py ch05_inequalities_wlln.tex [more .tex ...]
    python3 tools/run_blocks.py ch*.tex          # whole book
"""
import re, subprocess, sys, os, pathlib

PY = re.compile(r"\\pycode\{([^}]+)\}")
OUT = re.compile(r"\A\s*\\begin\{lstlisting\}\[style=output\]\n.*?\\end\{lstlisting\}", re.S)

def run(dosya, cwd):
    r = subprocess.run([sys.executable, str(dosya.resolve())], cwd=cwd, capture_output=True,
                       text=True, timeout=1800, env={**os.environ, "MPLBACKEND": "Agg"})
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout.rstrip()

def process(path):
    path = pathlib.Path(path); kok = path.parent
    figdir = kok / "figures"; figdir.mkdir(exist_ok=True)
    src = path.read_text(encoding="utf-8")
    out, pos = [], 0
    for m in PY.finditer(src):
        out.append(src[pos:m.end()])
        om = OUT.match(src[m.end():])
        try:
            res = run(kok / m.group(1), figdir)
        except Exception as e:
            print(f"ERROR {m.group(1)}:\n{e}"); sys.exit(1)
        if res:
            out.append("\n\\begin{lstlisting}[style=output]\n" + res + "\n\\end{lstlisting}")
        pos = m.end() + (om.end() if om else 0)
        print(f"{m.group(1)} ok")
    out.append(src[pos:])
    path.write_text("".join(out), encoding="utf-8")

for p in sys.argv[1:]:
    process(p)
