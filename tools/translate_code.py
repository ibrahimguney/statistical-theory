"""Helper used while translating: copies a Turkish code file, sets the English header line and
applies exact string replacements (each must occur). Not needed for normal use of the book."""
import pathlib, sys

def tr(src, dst, chapter, title, repl):
    s = pathlib.Path(src).read_text(encoding="utf-8").splitlines()
    s[0] = f"# Theory of Statistics — Chapter {chapter}: {title}"
    s = "\n".join(s) + "\n"
    for a, b in repl:
        assert a in s, (src, a)
        s = s.replace(a, b)
    bad = [l for l in s.splitlines()[1:] if any(c in l for c in "ığşçöüİĞŞÇÖÜ")]
    assert not bad, (dst, bad)
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(dst).write_text(s, encoding="utf-8")
