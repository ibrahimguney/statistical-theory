"""Checks main.log: exits with code 1 on undefined references/citations or overfull boxes above the limit.

Usage: python3 tools/check_log.py [main.log] [--limit 4]
"""
import re
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
log = args[0] if args else "main.log"
limit = 4.0
if "--limit" in sys.argv:
    limit = float(sys.argv[sys.argv.index("--limit") + 1])
s = open(log, encoding="latin-1").read()
problems = []
for m in re.finditer(r"LaTeX Warning: (Reference|Citation) `([^']+)' .*undefined", s):
    problems.append(f"undefined {m.group(1).lower()}: {m.group(2)}")
if "There were undefined references" in s and not problems:
    problems.append("undefined referans var")
for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)([^\n]*)", s):
    if float(m.group(1)) > limit:
        problems.append(f"overfull {m.group(1)}pt{m.group(2)}")
if "Rerun to get" in s:
    print("warning: LaTeX asks for a rerun")
for x in problems:
    print("!!", x)
print(f"{len(problems)} problems")
sys.exit(1 if problems else 0)
