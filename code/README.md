# Theory of Statistics — Python code

Every code block in the book is a separate file in this folder. File names consist of the chapter
number, the position within the chapter and the title of the corresponding subsection:

    ch05/c05_2_sample_size_rules.py
         │  │ └ short subsection title
         │  └ position in the chapter
         └ chapter number

In the book each file is shown, with its file name as title, by the `\pycode{...}` command; the
first two lines of the file (the header comment) are hidden in the book. When you change a file,
the book automatically shows the current code.

## Setup and running
    pip install -r code/requirements.txt
    python3 code/run_all.py              # all code (~1.5 min)
    python3 code/run_all.py 18           # Chapter 18 only
    python3 code/ch04/<file>.py          # a single file

## Random numbers
All simulations use numpy's recommended interface: `rng = np.random.default_rng(42)` and methods
such as `rng.normal(...)`, `rng.integers(...)`. Since the seed is fixed, the outputs are
reproducible, and they are identical to those of the Turkish edition. (The legacy
`np.random.seed` + `np.random.*` interface is not used.)

## Figures
Code that produces figures writes them with `savefig("name.pdf")` to the working directory;
`run_all.py` and `tools/run_blocks.py` run this code in the `figures/` directory, and the book
reads the figures from there.

## Refreshing the output boxes in the book
After changing a file, refresh the "Output" boxes of the corresponding chapter (from the book root):
    python3 tools/run_blocks.py ch05_inequalities_lln.tex
If the text quotes numbers from simulation results (e.g. "coverage 0.91"), check them by hand.
