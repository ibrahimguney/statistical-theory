"""Keeps the Turkish (istatistik-teorisi) and English (statistical-theory) editions in step.
Türkçe (istatistik-teorisi) ve İngilizce (statistical-theory) baskıları eşzamanlı tutar.

The two editions share chapter numbers, code numbering and every \\label, so a place in one
edition can be found in the other by its label. The same file lives in both repositories
(tools/sync_editions.py and araclar/baski_esle.py); run it from either one.

Usage / Kullanım (from the root of either repository):
    python3 tools/sync_editions.py check     [--other PATH]
        Structural comparison: labels of every chapter and appendix (set and order), citation keys,
        figures, code files (structure fingerprint: identifiers, strings and comments ignored)
        and the numbers in the Output boxes. Exit code 1 if anything differs.
    python3 tools/sync_editions.py changes   [--other PATH] [--from tr|en] [--since REV]
        Lists what has changed in the source edition (default: the one you run it from) since the
        last sync point, including uncommitted changes, and where each change belongs in the other
        edition: by label, or by section number, with line numbers in the target file.
    python3 tools/sync_editions.py mark      [--other PATH]
        Records the current commits of both editions as the new sync point
        (statistical-theory/tools/sync_point.json). Commit that file in the English repository.

The other repository is looked for with --other, then the STAT_OTHER environment variable,
then sibling folders (../statistical-theory, ../istatistik-teorisi, ../en, ../src).

Typical workflow / Tipik akış:
    1. edit the Turkish edition and commit;
    2. run  "changes"  -> the list of places to update in the English edition;
    3. update them, run  "check"  until it reports no difference;
    4. run  "mark", commit tools/sync_point.json in the English repository.
"""
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tokenize
import keyword

HERE = pathlib.Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ editions and file mapping
def edition_of(root):
    if any(root.glob("bolum_01_*.tex")):
        return "tr"
    if any(root.glob("ch01_*.tex")):
        return "en"
    return None


LAYOUT = {
    "tr": {"chapter": "bolum_{n:02d}_*.tex", "chapter_rx": r"^bolum_(\d\d)_.*\.tex$",
           "code_rx": r"^kodlar/bolum_(\d\d)/b\d\d_(\d+)_.*\.py$", "code_glob": "kodlar/bolum_{n:02d}/b{n:02d}_{k}_*.py",
           "solutions_rx": r"^ek_c_cozumler_(\d)\.tex$", "solutions": "ek_c_cozumler_{k}.tex",
           "fixed": {"preface": "onsoz.tex", "appendix_a": "ek_a_dagilimlar.tex", "bibliography": "kaynakca.tex",
                     "main": "main.tex", "solutions_intro": "ek_c_cozumler.tex"},
           "output_style": "cikti", "code_cmd": "pykod", "exercises": "Alıştırmalar"},
    "en": {"chapter": "ch{n:02d}_*.tex", "chapter_rx": r"^ch(\d\d)_.*\.tex$",
           "code_rx": r"^code/ch(\d\d)/c\d\d_(\d+)_.*\.py$", "code_glob": "code/ch{n:02d}/c{n:02d}_{k}_*.py",
           "solutions_rx": r"^app_b_solutions_(\d)\.tex$", "solutions": "app_b_solutions_{k}.tex",
           "fixed": {"preface": "preface.tex", "appendix_a": "app_a_distributions.tex", "bibliography": "bibliography.tex",
                     "main": "main.tex", "solutions_intro": "app_b_solutions.tex"},
           "output_style": "output", "code_cmd": "pycode", "exercises": "Exercises"},
}
NO_COUNTERPART = {"tr": {"ek_b_terimler.tex"}, "en": set()}


def unit_of(ed, rel):
    """File path (relative) -> edition-independent unit key, or None."""
    L = LAYOUT[ed]
    m = re.match(L["chapter_rx"], rel)
    if m:
        return ("chapter", int(m.group(1)))
    m = re.match(L["code_rx"], rel)
    if m:
        return ("code", int(m.group(1)), int(m.group(2)))
    m = re.match(L["solutions_rx"], rel)
    if m:
        return ("solutions", int(m.group(1)))
    for k, v in L["fixed"].items():
        if rel == v:
            return (k,)
    if rel.startswith("figures/"):
        return ("figure", rel)
    return None


def path_of(ed, root, unit):
    L = LAYOUT[ed]
    kind = unit[0]
    hits = []
    if kind == "chapter":
        hits = sorted(root.glob(L["chapter"].format(n=unit[1])))
    elif kind == "code":
        hits = sorted(root.glob(L["code_glob"].format(n=unit[1], k=unit[2])))
    elif kind == "solutions":
        p = root / L["solutions"].format(k=unit[1]); hits = [p] if p.exists() else []
    elif kind == "figure":
        p = root / unit[1]; hits = [p] if p.exists() else []
    elif kind in L["fixed"]:
        p = root / L["fixed"][kind]; hits = [p] if p.exists() else []
    return hits[0] if hits else None


def describe(unit, msg):
    k = unit[0]
    if k == "chapter":
        return msg["chapter"].format(unit[1])
    if k == "code":
        return msg["code"].format(unit[1], unit[2])
    if k == "solutions":
        return msg["solutions"].format(unit[1])
    if k == "figure":
        return unit[1]
    return k


# ------------------------------------------------------------------ messages
MSG = {
    "en": {"chapter": "Chapter {}", "code": "code {}.{}", "solutions": "solutions part {}",
           "labels_missing": "labels only in {}: {}", "labels_order": "label order differs (first difference: {} vs {})",
           "cites": "citation keys differ: only in {}: {}", "figs": "figures differ: only in {}: {}",
           "code_missing": "code file only in {}", "code_struct": "code structure differs (first difference near line {} / {}: {} vs {})",
           "out_count": "number of Output boxes differs: {} vs {}", "out_nums": "Output box {} (after {}): numbers differ: {} vs {}",
           "ok": "The two editions agree: {} units compared.", "diffs": "{} difference(s) found.",
           "no_sync": "No sync point recorded yet; use --since REV, or run 'mark' when the editions agree.",
           "since": "Changes in the {} edition since {} (including uncommitted changes):",
           "nochange": "No changes since the sync point.", "marked": "Sync point recorded: tr {} | en {} -> {}",
           "where": "  -> {}:{}", "new_label": "  new label {} (no counterpart yet); insert after {}",
           "removed": "  removed in source: {}", "section": "section {} \u201c{}\u201d",
           "codeblock": "code block {} and its Output box", "outbox": "Output box after {}",
           "deleted_file": "file deleted in source", "added_file": "new file, no counterpart",
           "no_target": "no counterpart file in the target edition", "intro": "chapter introduction (before the first section)", "deleted_tag": " [deleted]", "env": "{} {}{} (no. {} of its kind in section {}){}",
           "using": "Comparing {} with {}", "other_missing": "Other repository not found; use --other PATH.",
           "dirty": "warning: {} has uncommitted changes; the sync point records only commits."},
    "tr": {"chapter": "Bölüm {}", "code": "kod {}.{}", "solutions": "çözümler, kısım {}",
           "labels_missing": "yalnızca {} baskıda olan etiketler: {}", "labels_order": "etiket sırası farklı (ilk fark: {} / {})",
           "cites": "atıf anahtarları farklı: yalnızca {} baskıda: {}", "figs": "şekiller farklı: yalnızca {} baskıda: {}",
           "code_missing": "kod dosyası yalnızca {} baskıda", "code_struct": "kod yapısı farklı (ilk fark yaklaşık {} / {}. satırda: {} / {})",
           "out_count": "Çıktı kutusu sayısı farklı: {} / {}", "out_nums": "{}. Çıktı kutusu ({} sonrası): sayılar farklı: {} / {}",
           "ok": "İki baskı uyumlu: {} birim karşılaştırıldı.", "diffs": "{} fark bulundu.",
           "no_sync": "Henüz eşleme noktası kaydedilmemiş; --since REV kullanın ya da baskılar uyumluyken 'mark' çalıştırın.",
           "since": "{} baskıda {} sonrasındaki değişiklikler (işlenmemiş değişiklikler dahil):",
           "nochange": "Eşleme noktasından bu yana değişiklik yok.", "marked": "Eşleme noktası kaydedildi: tr {} | en {} -> {}",
           "where": "  -> {}:{}", "new_label": "  yeni etiket {} (karşılığı yok); {} etiketinden sonra eklenmeli",
           "removed": "  kaynakta silinmiş: {}", "section": "{}. kısım \u201c{}\u201d",
           "codeblock": "{} kod bloğu ve Çıktı kutusu", "outbox": "{} sonrasındaki Çıktı kutusu",
           "deleted_file": "kaynakta silinmiş dosya", "added_file": "yeni dosya, karşılığı yok",
           "no_target": "hedef baskıda karşılık gelen dosya yok", "intro": "bölüm girişi (ilk kısımdan önce)", "deleted_tag": " [silindi]", "env": "{} {}{} ({}. kısımdaki {}. ortam){}",
           "using": "{} ile {} karşılaştırılıyor", "other_missing": "Öbür depo bulunamadı; --other YOL kullanın.",
           "dirty": "uyarı: {} deposunda işlenmemiş değişiklikler var; eşleme noktası yalnızca commit'leri kaydeder."},
}
ED_NAME = {"en": {"tr": "Turkish", "en": "English"}, "tr": {"tr": "Türkçe", "en": "İngilizce"}}


# ------------------------------------------------------------------ TeX analysis
LABEL_RX = re.compile(r"\\label\{([^}]+)\}")
INDEX_RX = re.compile(r"\\index\{(?:[^{}]|\{[^{}]*\})*\}")


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def labels(s):
    return [m.group(1) for m in LABEL_RX.finditer(strip_comments(s))]


def citations(s):
    keys = []
    for m in re.finditer(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", strip_comments(s)):
        keys += [k.strip() for k in m.group(1).split(",")]
    return sorted(set(keys))


def figures(s):
    return sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", strip_comments(s))))


def output_boxes(s, style):
    return re.findall(r"\\begin\{lstlisting\}\[style=" + style + r"\]\n(.*?)\\end\{lstlisting\}", s, flags=re.S)


NUM_RX = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def numbers(text):
    return NUM_RX.findall(text)


# ------------------------------------------------------------------ code fingerprint
def fingerprint(src):
    """Token structure with identifiers, strings, f-strings and comments abstracted away.
    Numbers, operators, keywords and attribute names (np.mean, ...) are kept.
    Returns a list of (token, line)."""
    out = []
    depth = 0
    prev = None
    lines = src.splitlines(True)[2:]          # the two header lines differ by language
    try:
        toks = list(tokenize.generate_tokens(io.StringIO("".join(lines)).readline))
    except (tokenize.TokenError, SyntaxError):
        return [("<untokenizable>", 0)]
    for t in toks:
        tn = tokenize.tok_name[t.type]
        if tn == "FSTRING_START":
            if depth == 0:
                out.append(("S", t.start[0] + 2))
            depth += 1
            continue
        if tn == "FSTRING_END":
            depth -= 1
            continue
        if depth or tn in ("COMMENT", "NL", "ENDMARKER", "FSTRING_MIDDLE"):
            continue
        if tn == "STRING":
            tok = "S"
        elif tn == "NAME":
            tok = t.string if (keyword.iskeyword(t.string) or prev == ".") else "N"
        elif tn in ("NEWLINE", "INDENT", "DEDENT"):
            tok = tn
        else:
            tok = t.string
        out.append((tok, t.start[0] + 2))
        prev = t.string
    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x[0] != y[0]:
            return x[1], y[1], x[0], y[0]
    if len(a) != len(b):
        i = min(len(a), len(b))
        return ((a[i][1] if i < len(a) else "end"), (b[i][1] if i < len(b) else "end"),
                (a[i][0] if i < len(a) else "-"), (b[i][0] if i < len(b) else "-"))
    return None


# ------------------------------------------------------------------ check
def all_units(ed, root):
    units = set()
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.parts:
            u = unit_of(ed, p.relative_to(root).as_posix())
            if u and u[0] in ("chapter", "code", "solutions", "appendix_a", "preface"):
                units.add(u)
    return units


def check(roots, lang):
    msg = MSG[lang]
    names = ED_NAME[lang]
    problems = []
    units = all_units("tr", roots["tr"]) | all_units("en", roots["en"])
    for u in sorted(units, key=lambda u: (u[0], u[1:])):
        p = {ed: path_of(ed, roots[ed], u) for ed in ("tr", "en")}
        where = describe(u, msg)
        missing = [ed for ed in p if p[ed] is None]
        if missing:
            other = "en" if missing[0] == "tr" else "tr"
            problems.append(f"{where}: " + msg["code_missing"].format(names[other]))
            continue
        s = {ed: p[ed].read_text(encoding="utf-8") for ed in p}
        if u[0] == "code":
            d = first_diff(fingerprint(s["tr"]), fingerprint(s["en"]))
            if d:
                problems.append(f"{where}: " + msg["code_struct"].format(*d)
                                + f"  [{p['tr'].relative_to(roots['tr'])} | {p['en'].relative_to(roots['en'])}]")
            continue
        lab = {ed: labels(s[ed]) for ed in p}
        for a, b in (("tr", "en"), ("en", "tr")):
            only = [x for x in lab[a] if x not in set(lab[b])]
            if only:
                problems.append(f"{where}: " + msg["labels_missing"].format(names[a], ", ".join(only)))
        common = set(lab["tr"]) & set(lab["en"])
        lt = [x for x in lab["tr"] if x in common]
        le = [x for x in lab["en"] if x in common]
        if lt != le:
            i = next(i for i, (x, y) in enumerate(zip(lt, le)) if x != y)
            problems.append(f"{where}: " + msg["labels_order"].format(lt[i], le[i]))
        for fn, key in ((citations, "cites"), (figures, "figs")):
            v = {ed: fn(s[ed]) for ed in p}
            for a, b in (("tr", "en"), ("en", "tr")):
                only = [x for x in v[a] if x not in v[b]]
                if only:
                    problems.append(f"{where}: " + msg[key].format(names[a], ", ".join(only)))
        if u[0] == "chapter":
            ob = {ed: output_boxes(s[ed], LAYOUT[ed]["output_style"]) for ed in p}
            if len(ob["tr"]) != len(ob["en"]):
                problems.append(f"{where}: " + msg["out_count"].format(len(ob["tr"]), len(ob["en"])))
            codes = re.findall(r"\\" + LAYOUT["en"]["code_cmd"] + r"\{([^}]+)\}", s["en"])
            for i, (x, y) in enumerate(zip(ob["tr"], ob["en"])):
                nx, ny = numbers(x), numbers(y)
                # word order may differ between languages: compare the sequence of decimal
                # numbers and the multiset of all numbers
                dec = lambda v: [t for t in v if re.search(r"[.eE]", t)]
                if dec(nx) != dec(ny) or sorted(nx) != sorted(ny):
                    if dec(nx) != dec(ny):
                        nx, ny = dec(nx), dec(ny)
                    j = next((k for k, (a, b) in enumerate(zip(nx, ny)) if a != b), min(len(nx), len(ny)))
                    after = pathlib.Path(codes[i]).name if i < len(codes) else "?"
                    problems.append(f"{where}: " + msg["out_nums"].format(
                        i + 1, after, " ".join(nx[j:j + 3]) or "-", " ".join(ny[j:j + 3]) or "-"))
    for x in problems:
        print("!!", x)
    print(msg["diffs"].format(len(problems)) if problems else msg["ok"].format(len(units)))
    return 1 if problems else 0


# ------------------------------------------------------------------ git helpers
def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def head(root):
    return git(root, "rev-parse", "HEAD").strip()


SYNC_FILE = "tools/sync_point.json"   # in the English repository


def read_sync(roots):
    p = roots["en"] / SYNC_FILE
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ anchors for changed lines
def anchors(s, ed):
    """For a TeX source returns a function line -> description of the innermost labelled
    environment (or section) containing it."""
    lines = s.splitlines()
    envs = []            # (start, end, name, label or None)
    stack = []
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
            if m.group(2) == "document":
                continue
            if m.group(1) == "begin":
                stack.append((m.group(2), i))
            elif stack:
                name, start = stack.pop()
                block = "\n".join(lines[start - 1:min(i, start + 3)])
                lab = LABEL_RX.search(block)
                envs.append((start, i, name, lab.group(1) if lab else None))
    # a proof belongs to the labelled environment that ends just before it
    envs.sort()
    for k, (a, b, name, lab) in enumerate(envs):
        if name == "proof" and lab is None:
            prev = [e for e in envs if e[1] <= a and e[3] and e[2] != "proof" and a - e[1] <= 3]
            if prev:
                envs[k] = (a, b, name, max(prev, key=lambda e: e[1])[3])
    secs = [(i, t) for i, line in enumerate(lines, 1) for m in [re.match(r"\\section\*?\{(.*?)\}", line)] if m for t in [m.group(1)]]
    code_cmd = LAYOUT[ed]["code_cmd"]
    style = LAYOUT[ed]["output_style"]

    def at(n):
        # code block or output box?
        line = lines[n - 1] if 0 < n <= len(lines) else ""
        m = re.search(r"\\" + code_cmd + r"\{([^}]+)\}", line)
        if m:
            return ("code", m.group(1))
        for (a, b, name, lab) in envs:
            if name == "lstlisting" and a <= n <= b and f"style={style}" in lines[a - 1]:
                before = [re.search(r"\\" + code_cmd + r"\{([^}]+)\}", lines[k]) for k in range(a - 2, -1, -1)]
                before = next((x.group(1) for x in before if x), "?")
                return ("output", before)
        inner = [e for e in envs if e[0] <= n <= e[1] and e[3]]
        if inner:
            e = min(inner, key=lambda e: e[1] - e[0])
            return ("label", e[3])
        plain = [e for e in envs if e[0] <= n <= e[1] and e[2] in NAMED_ENVS]
        if plain:
            e = min(plain, key=lambda e: e[1] - e[0])
            k = sum(1 for i, _ in secs if i <= e[0])
            lo = secs[k - 1][0] if k else 0
            nth = sum(1 for f in envs if f[2] == e[2] and lo <= f[0] <= e[0])
            title = re.match(r".*?\\begin\{" + re.escape(e[2]) + r"\}(\[[^\]]*\])?", lines[e[0] - 1])
            return ("env", e[2], k, nth, (title.group(1) or "") if title else "", False)
        # chapter label line itself
        lab = LABEL_RX.search(line)
        if lab and not line.lstrip().startswith("\\chapter"):
            return ("label", lab.group(1))
        k = sum(1 for i, _ in secs if i <= n)
        title = secs[k - 1][1] if k else ""
        return ("section", k, title)
    return at


NAMED_ENVS = ("theorem", "lemma", "proposition", "corollary", "definition", "example", "remark",
              "exercise", "solution", "cozum", "proof", "itemize", "enumerate", "figure", "center", "tabular",
              "longtable")


def env_in(s, name, k, nth):
    """Line of the nth `name` environment in section k of s."""
    lines = s.splitlines()
    secs = [i for i, line in enumerate(lines, 1) if re.match(r"\\section\*?\{", line)]
    lo = secs[k - 1] if k else 0
    hi = secs[k] if k < len(secs) else len(lines) + 1
    hits = [i for i, line in enumerate(lines, 1) if lo <= i < hi and "\\begin{" + name + "}" in line]
    return hits[nth - 1] if 0 < nth <= len(hits) else None


def section_in(s, k):
    secs = [(i, m.group(1)) for i, line in enumerate(s.splitlines(), 1) for m in [re.match(r"\\section\*?\{(.*?)\}", line)] if m]
    if k == 0:
        return 1, "(intro)"
    return secs[k - 1] if k <= len(secs) else (None, "?")


def line_of(s, pattern):
    for i, line in enumerate(s.splitlines(), 1):
        if pattern in line:
            return i
    return None


def changed_lines(root, rev, rel):
    """New-side line numbers touched by the diff rev..working tree (deletions mapped to the
    following line)."""
    out = git(root, "diff", "-U0", rev, "--", rel)
    lines = {}
    for m in re.finditer(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", out, flags=re.M):
        old, start, count = int(m.group(1) or 1), int(m.group(2)), int(m.group(3) or 1)
        old_start = int(m.group(0).split()[1].split(",")[0][1:])
        if count == 0:                            # pure deletion after line `start`
            lines[max(start, 1)] = (True, old_start)
        for n in range(start, start + count):     # replacements always count, even if blank
            prev = lines.get(n, (False, None))
            lines[n] = (prev[0] or old > 0, old_start if old > 0 else prev[1])
    return sorted((n, d, o) for n, (d, o) in lines.items())   # (line, is_deletion, old line)


def changes(roots, src, rev, lang):
    msg = MSG[lang]
    names = ED_NAME[lang]
    dst = "en" if src == "tr" else "tr"
    R, T = roots[src], roots[dst]
    print(msg["since"].format(names[src], rev[:10]))
    status = git(R, "diff", "--name-status", rev, "--")
    untracked = git(R, "ls-files", "--others", "--exclude-standard")
    entries = [l.split("\t") for l in status.splitlines() if l.strip()]
    entries += [["A", f] for f in untracked.splitlines() if f.strip()]
    if not entries:
        print(msg["nochange"])
        return 0
    shown = 0
    for e in entries:
        code, rel = e[0][0], e[-1]
        u = unit_of(src, rel)
        if u is None:
            continue
        if u[0] in ("figure", "main"):
            continue      # figures are regenerated by run_blocks; main.tex is edition specific
        shown += 1
        tgt = path_of(dst, T, u)
        tdesc = tgt.relative_to(T).as_posix() if tgt else msg["no_target"]
        print(f"\n* {describe(u, msg)}  [{rel}] -> {tdesc}")
        if code == "D":
            print("  " + msg["deleted_file"]); continue
        if tgt is None:
            print("  " + msg["added_file"]); continue
        if u[0] == "code":
            continue
        s_src = (R / rel).read_text(encoding="utf-8")
        s_tgt = tgt.read_text(encoding="utf-8")
        at = anchors(s_src, src)
        tgt_labels = set(labels(s_tgt))
        src_labels = labels(s_src)
        seen = []
        lines = changed_lines(R, rev, rel) if code != "A" else []
        src_lines = s_src.splitlines()
        old_src = git(R, "show", f"{rev}:{rel}") if code != "A" else ""
        at_old = anchors(old_src, src)
        for n, deletion, old_line in lines:
            if not deletion and 0 < n <= len(src_lines) and not src_lines[n - 1].strip():
                continue                      # blank-line-only change
            a = at(n)
            if deletion and old_line:
                b = at_old(old_line)
                if b[0] in ("label", "env"):
                    # an unlabelled environment whose title no longer occurs was deleted
                    a = b if b[0] == "label" else b[:5] + (bool(b[4]) and b[4] not in s_src,)
            if a not in seen:
                seen.append(a)
        for a in seen:
            if a[0] == "label":
                lab = a[1]
                if lab in tgt_labels:
                    print(f"  {lab}" + msg["where"].format(tgt.name, line_of(s_tgt, "\\label{" + lab + "}")))
                else:
                    i = src_labels.index(lab) if lab in src_labels else -1
                    prev = next((x for x in reversed(src_labels[:i]) if x in tgt_labels), "?")
                    print(msg["new_label"].format(lab, prev))
            elif a[0] in ("code", "output"):
                k = re.search(r"_(\d+)_", pathlib.Path(a[1]).name)
                u2 = ("code", u[1], int(k.group(1))) if (k and u[0] == "chapter") else None
                p2 = path_of(dst, T, u2) if u2 else None
                ln = line_of(s_tgt, p2.relative_to(T).as_posix()) if p2 else None
                text = msg["codeblock" if a[0] == "code" else "outbox"].format(pathlib.Path(a[1]).name)
                print("  " + text + (msg["where"].format(tgt.name, ln) if ln else ""))
            elif a[0] == "env":
                ln = env_in(s_tgt, a[1], a[2], a[3])
                gone = msg["deleted_tag"] if a[5] else ""
                if lang == "tr":
                    text = msg["env"].format(a[1], a[4], gone, a[2], a[3], "")
                else:
                    text = msg["env"].format(a[1], a[4], gone, a[3], a[2], "")
                print("  " + text + (msg["where"].format(tgt.name, ln) if ln else ""))
            elif a[1] == 0:
                print("  " + msg["intro"] + msg["where"].format(tgt.name, line_of(s_tgt, "\\chapter")))
            else:
                ln, title = section_in(s_tgt, a[1])
                print("  " + msg["section"].format(a[1], a[2]) + msg["where"].format(tgt.name, ln) + f"  \u201c{title}\u201d")
        removed = [x for x in labels(git(R, "show", f"{rev}:{rel}") if code != "A" else "") if x not in src_labels]
        for x in removed:
            print(msg["removed"].format(x) + (msg["where"].format(tgt.name, line_of(s_tgt, "\\label{" + x + "}"))
                                              if x in tgt_labels else ""))
    if not shown:
        print(msg["nochange"])
    return 0


# ------------------------------------------------------------------ main
def find_other(own_ed, args):
    want = "en" if own_ed == "tr" else "tr"
    cands = []
    if "--other" in args:
        cands.append(pathlib.Path(args[args.index("--other") + 1]))
    if os.environ.get("STAT_OTHER"):
        cands.append(pathlib.Path(os.environ["STAT_OTHER"]))
    for n in ("statistical-theory", "istatistik-teorisi", "en", "src"):
        cands.append(HERE.parent / n)
    for c in cands:
        c = c.resolve()
        if c.is_dir() and edition_of(c) == want:
            return c
    return None


def main():
    args = sys.argv[1:]
    cmd = next((a for a in args if a in ("check", "changes", "mark")), "check")
    own = edition_of(HERE)
    lang = own or "en"
    other = find_other(own, args)
    if other is None:
        print(MSG[lang]["other_missing"]); return 2
    roots = {own: HERE, ("en" if own == "tr" else "tr"): other}
    print(MSG[lang]["using"].format(HERE, other))
    if cmd == "check":
        return check(roots, lang)
    if cmd == "mark":
        for ed in ("tr", "en"):
            if git(roots[ed], "status", "--porcelain", "--untracked-files=no").strip():
                print(MSG[lang]["dirty"].format(roots[ed].name))
        data = {"tr": head(roots["tr"]), "en": head(roots["en"])}
        p = roots["en"] / SYNC_FILE
        p.write_text(json.dumps(data, indent=2) + "\n")
        print(MSG[lang]["marked"].format(data["tr"][:10], data["en"][:10], p))
        return 0
    src = args[args.index("--from") + 1] if "--from" in args else own
    if "--since" in args:
        rev = args[args.index("--since") + 1]
    else:
        sp = read_sync(roots)
        if not sp:
            print(MSG[lang]["no_sync"]); return 2
        rev = sp[src]
    return changes(roots, src, rev, lang)


if __name__ == "__main__":
    sys.exit(main())
