"""Builds a Jupyter notebook for each chapter from the code/ files: notebooks/ch_XX.ipynb

Usage (from the book root):
    python3 tools/make_notebooks.py           # regenerates the notebooks (without outputs)
    python3 tools/make_notebooks.py --run     # generates them, runs them and stores the outputs
    python3 tools/make_notebooks.py --check   # exit code 1 if the code/text cells of the notebooks
                                              # differ from the sources (for CI)

Notebook contents: chapter title and Colab link, a setup cell, then, in the order of the book,
for each subsection a text cell (subsection title and introductory paragraph from the book) and
a code cell (the contents of the code/... file from line 3 on). \\ref's in the text are replaced
by the numbers in main.aux, so compile the book first. Do not edit the notebooks by hand.
"""
import pathlib
import re
import sys

import nbformat

DEPO = "ibrahimguney/statistical-theory"
DAL = "main"
KOK = pathlib.Path(__file__).resolve().parent.parent
HEDEF = KOK / "notebooks"

MAKROLAR = r"""$$
\newcommand{\R}{\mathbb{R}}\newcommand{\N}{\mathbb{N}}\newcommand{\E}{\mathbb{E}}
\newcommand{\P}{\mathbb{P}}\newcommand{\F}{\mathcal{F}}\newcommand{\I}{\mathbb{I}}
\newcommand{\Var}{\operatorname{Var}}\newcommand{\Cov}{\operatorname{Cov}}
\newcommand{\Corr}{\operatorname{Corr}}\newcommand{\MSE}{\operatorname{MSE}}
\newcommand{\Bias}{\operatorname{Bias}}\newcommand{\Normal}{\mathcal{N}}
\newcommand{\iid}{\overset{\text{i.i.d.}}{\sim}}\newcommand{\pto}{\xrightarrow{P}}
\newcommand{\dto}{\xrightarrow{d}}\newcommand{\X}{\mathbf{X}}\newcommand{\x}{\mathbf{x}}
$$"""

KURULUM = """# Setup: numpy, scipy and matplotlib are preinstalled on Colab. Locally:
#   pip install -r code/requirements.txt
# Figure files (PDF) saved by the code are written to the _figures/ folder; the figures are
# also shown inside the notebook.
import os
os.makedirs("_figures", exist_ok=True)
os.chdir("_figures")
%matplotlib inline"""


def etiketler():
    aux = KOK / "main.aux"
    d = {}
    if aux.exists():
        for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux.read_text(encoding="utf-8")):
            d[m.group(1)] = m.group(2)
    return d


def parantez_sonu(s, i):
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return j
    return len(s) - 1


def latex_md(t, et):
    """Converts the introductory paragraph to approximate Markdown (maths stays as $...$)."""
    t = re.sub(r"\\index\{(?:[^{}]|\{[^{}]*\})*\}", "", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    t = re.sub(r"~?\\cite(\[[^\]]*\])?\{[^}]*\}", "", t)
    t = re.sub(r"\\(?:eq)?ref\{([^}]+)\}", lambda m: et.get(m.group(1), "?"), t)
    t = re.sub(r"\\hyperref\[[^\]]*\]\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", t)
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", t)
    t = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", t)
    t = t.replace("\\allowbreak", "").replace("\\_", "_").replace("\\%", "%")
    t = t.replace("``", "“").replace("''", "”").replace("---", "—").replace("--", "–")
    t = re.sub(r"(?<!\\)~", " ", t)
    t = t.replace("\\[", "$$").replace("\\]", "$$")
    t = re.sub(r"\\begin\{(itemize|enumerate)\}|\\end\{(itemize|enumerate)\}", "", t)
    t = re.sub(r"\\item\s*", "\n- ", t)
    t = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", t, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def defter(tex, et):
    s = tex.read_text(encoding="utf-8")
    no = re.match(r"ch(\d+)", tex.name).group(1)
    m = re.search(r"\\chapter(\[[^\]]*\])?\{", s)
    baslik = s[m.end():parantez_sonu(s, m.end() - 1)]
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    yol = f"notebooks/ch_{no}.ipynb"
    colab = f"https://colab.research.google.com/github/{DEPO}/blob/{DAL}/{yol}"
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"# Chapter {int(no)}: {latex_md(baslik, et)}\n\n"
        f"The computational applications of this chapter of *Theory of Statistics*. The code is "
        f"generated automatically from the book's `code/ch{no}/` files; see the book for the "
        f"explanations and proofs.\n\n"
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})\n\n"
        + MAKROLAR))
    nb.cells.append(nbformat.v4.new_code_cell(KURULUM))
    for pm in re.finditer(r"\\pycode\{([^}]+)\}", s):
        once = s[:pm.start()]
        alt = list(re.finditer(r"\\subsection\*?\{", once))
        if alt:
            a = alt[-1]
            kapa = parantez_sonu(once, a.end() - 1)
            altbaslik = latex_md(once[a.end():kapa], et)
            giris = latex_md(once[kapa + 1:], et)
        else:
            altbaslik, giris = pathlib.Path(pm.group(1)).stem, ""
        # for a second code block in the same subsection, the intro is the text after the previous block
        onceki_kod = once.rfind("\\end{lstlisting}")
        if alt and onceki_kod > alt[-1].start():
            giris = latex_md(once[onceki_kod + len("\\end{lstlisting}"):], et)
            altbaslik += " (continued)"
        md = f"## {altbaslik}\n\n{giris}".strip() + f"\n\n*Source file:* `{pm.group(1)}`"
        nb.cells.append(nbformat.v4.new_markdown_cell(md))
        kod = (KOK / pm.group(1)).read_text(encoding="utf-8").splitlines()[2:]
        nb.cells.append(nbformat.v4.new_code_cell("\n".join(kod).strip("\n")))
    for i, c in enumerate(nb.cells):
        c["id"] = f"h{i:02d}"
    return no, nb


def ozet(nb):
    return [(c.cell_type, c.source) for c in nb.cells]


def main():
    calistir = "--run" in sys.argv
    kontrol = "--check" in sys.argv
    HEDEF.mkdir(exist_ok=True)
    et = etiketler()
    farkli = []
    for tex in sorted(KOK.glob("ch[0-9][0-9]_*.tex")):
        if "\\pycode{" not in tex.read_text(encoding="utf-8"):
            continue
        no, nb = defter(tex, et)
        dosya = HEDEF / f"ch_{no}.ipynb"
        if kontrol:
            if not dosya.exists() or ozet(nbformat.read(dosya, as_version=4)) != ozet(nb):
                farkli.append(dosya.name)
            continue
        if calistir:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=1800, kernel_name="python3",
                           resources={"metadata": {"path": str(HEDEF)}}).execute()
        nbformat.write(nb, dosya)
        print(f"{dosya.relative_to(KOK)} written" + (" (executed)" if calistir else ""))
    if kontrol:
        if farkli:
            print("Out-of-date notebooks:", ", ".join(farkli))
            print("To fix: python3 tools/make_notebooks.py --run")
            sys.exit(1)
        print("All notebooks are consistent with the sources.")


if __name__ == "__main__":
    main()
