"""Generates the cover artwork (TikZ): cover_art.tex

Front cover: a Galton board. Balls bounce left or right off the pins; the binomial distribution
builds up in the bins below and approaches the normal curve (central limit theorem). The ball paths
and the bin counts come from a genuine simulation with a fixed seed.
Back cover: confidence intervals built from repeated samples for the same parameter.

Usage (from the book root): python3 tools/make_cover.py
The artwork is identical in both editions (the Turkish edition uses araclar/kapak_uret.py).
"""
import math
import pathlib

import numpy as np

KOK = pathlib.Path(__file__).resolve().parent.parent
rng = np.random.default_rng(1894)        # fixed seed: the artwork is the same in every build

W, H = 21.0, 29.7                        # A4 (cm)
CX = W / 2
R = 14                                   # rows of pins
DX = 0.82                                # horizontal pin spacing
DY = 0.52                                # vertical row spacing
TOP = 19.5                               # height of the first pin
BIN_TOP = TOP - R * DY - 0.55            # top edge of the bins
FLOOR = 4.2                              # floor of the bins
NBALL = 1500
BR = 0.062                               # ball radius
SUB = 5                                  # ball columns side by side per bin


def f(x):
    return f"{x:.3f}"


def renk(t):
    """0 (centre) -> warm gold, 1 (tail) -> cyan."""
    sicak = np.array([247, 184, 64])
    soguk = np.array([92, 200, 214])
    c = (1 - t) * sicak + t * soguk
    return "{rgb,255:red,%d;green,%d;blue,%d}" % tuple(int(v) for v in c)


satirlar = []
yaz = satirlar.append

# ------------------------------------------------------------------ front cover artwork
yaz(r"\newcommand{\coverfrontart}{%")
# background: dark navy, slightly lighter in the middle
yaz(r"\fill[covernavy] (0,0) rectangle (%s,%s);" % (f(W), f(H)))
yaz(r"\shade[inner color=coverlight,outer color=covernavy,opacity=0.85] (%s,%s) ellipse (9.5 and 11);" % (f(CX), f(12.5)))
# pins
for r in range(R):
    y = TOP - r * DY
    for k in range(r + 1):
        x = CX + (k - r / 2) * DX
        yaz(r"\fill[white,opacity=0.7] (%s,%s) circle (0.052);" % (f(x), f(y)))
# funnel
yaz(r"\draw[white,opacity=0.35,line width=0.6pt] (%s,%s) -- (%s,%s) (%s,%s) -- (%s,%s);" % (
    f(CX - 1.3), f(TOP + 1.25), f(CX - 0.25), f(TOP + 0.45), f(CX + 1.3), f(TOP + 1.25), f(CX + 0.25), f(TOP + 0.45)))
# paths of the first few balls
for b in range(14):
    adim = rng.integers(0, 2, R)
    x, y = CX, TOP + 0.45
    pts = [(x, y)]
    k = 0
    for r in range(R):
        yp = TOP - r * DY
        xp = CX + (k - r / 2) * DX
        pts.append((xp + (0.5 if adim[r] else -0.5) * 0.36, yp + 0.12))
        k += adim[r]
    xb = CX + (k - R / 2) * DX
    pts.append((xb, BIN_TOP - 0.2))
    t = abs(k - R / 2) / (R / 2)
    yol = " -- ".join(f"({f(a)},{f(c)})" for a, c in pts)
    yaz(r"\draw[color=%s,opacity=0.6,line width=0.6pt,rounded corners=3pt] %s;" % (renk(t), yol))
# bin walls
for k in range(R + 2):
    x = CX + (k - 0.5 - R / 2) * DX
    yaz(r"\draw[white,opacity=0.25,line width=0.4pt] (%s,%s) -- (%s,%s);" % (f(x), f(FLOOR), f(x), f(BIN_TOP)))
yaz(r"\draw[white,opacity=0.5,line width=0.6pt] (%s,%s) -- (%s,%s);" % (
    f(CX - (R / 2 + 0.5) * DX), f(FLOOR), f(CX + (R / 2 + 0.5) * DX), f(FLOOR)))
# balls: a genuine simulation
kutu = rng.binomial(R, 0.5, NBALL)
sayim = np.bincount(kutu, minlength=R + 1)
aralik = (DX - 0.08) / SUB
for k in range(R + 1):
    t = abs(k - R / 2) / (R / 2)
    x0 = CX + (k - R / 2) * DX - (SUB - 1) / 2 * aralik
    for j in range(sayim[k]):
        sat, sut = divmod(j, SUB)
        x = x0 + sut * aralik
        y = FLOOR + BR + 0.01 + sat * 2 * BR * 0.98
        yaz(r"\fill[color=%s] (%s,%s) circle (%s);" % (renk(t), f(x), f(y), f(BR)))
# normal curve N(R/2, R/4) on the same scale (expected balls per bin -> height)
yuk = 2 * BR * 0.98 / SUB                # average height contributed by one ball
pts = []
for i in range(241):
    xs = -0.9 + i * (R + 1.8) / 240
    d = NBALL * math.exp(-(xs - R / 2) ** 2 / (2 * R / 4)) / math.sqrt(2 * math.pi * R / 4)
    pts.append((CX + (xs - R / 2) * DX, FLOOR + 0.01 + d * yuk))
yaz(r"\draw[white,line width=1.1pt,opacity=0.9] " + " -- ".join(f"({f(a)},{f(c)})" for a, c in pts) + ";")
yaz("}")

# ------------------------------------------------------------------ back cover artwork: confidence intervals
yaz(r"\newcommand{\coverbackart}{%")
yaz(r"\fill[covernavy] (0,0) rectangle (%s,%s);" % (f(W), f(H)))
yaz(r"\shade[inner color=coverlight,outer color=covernavy,opacity=0.6] (%s,%s) ellipse (10 and 8);" % (f(CX), f(6.5)))
n, M = 10, 40
t975 = 2.262157
for tohum in range(1000):                # a representative sample for 95% coverage: 2 of the 40 intervals miss
    g = np.random.default_rng(tohum)
    Xs = g.normal(0, 1, (M, n))
    m_, s_ = Xs.mean(1), Xs.std(1, ddof=1)
    if np.sum(np.abs(m_) > t975 * s_ / math.sqrt(n)) == 2:
        break
X0, SC = CX, 2.6                         # horizontal position of mu = 0 and scale (cm per unit)
yaz(r"\draw[white,opacity=0.5,dashed,line width=0.5pt] (%s,%s) -- (%s,%s);" % (f(X0), f(1.7), f(X0), f(11.0)))
for i in range(M):
    m, s = m_[i], s_[i]
    lo, hi = m - t975 * s / math.sqrt(n), m + t975 * s / math.sqrt(n)
    y = 2.0 + i * 0.22
    iskalar = lo > 0 or hi < 0
    renk_ = "coverred" if iskalar else "covergold"
    yaz(r"\draw[color=%s,opacity=%s,line width=1.5pt] (%s,%s) -- (%s,%s);" % (
        renk_, "1" if iskalar else "0.9", f(X0 + SC * lo), f(y), f(X0 + SC * hi), f(y)))
    yaz(r"\fill[white,opacity=0.85] (%s,%s) circle (0.04);" % (f(X0 + SC * m), f(y)))
yaz("}")

# ------------------------------------------------------------------ title-page emblem (light background)
yaz(r"\newcommand{\coveremblem}{%")
for k in range(R + 1):
    t = abs(k - R / 2) / (R / 2)
    c = int(round(sayim[k] / SUB / 3))
    x = (k - R / 2) * 0.32
    for j in range(c):
        yaz(r"\fill[color=%s] (%s,%s) circle (0.055);" % (renk(t), f(x), f(0.07 + j * 0.13)))
pts = []
for i in range(121):
    xs = -0.9 + i * (R + 1.8) / 120
    d = NBALL / SUB / 3 * math.exp(-(xs - R / 2) ** 2 / (2 * R / 4)) / math.sqrt(2 * math.pi * R / 4)
    pts.append(((xs - R / 2) * 0.32, 0.02 + d * 0.13))
yaz(r"\draw[covernavy,line width=0.8pt] " + " -- ".join(f"({f(a)},{f(c)})" for a, c in pts) + ";")
yaz("}")

baslik = "% Generated by tools/make_cover.py (araclar/kapak_uret.py in the Turkish edition); do not edit by hand.\n"
metin = baslik + "\n".join(satirlar) + "\n"
hedef = KOK / ("kapak_sanat.tex" if (KOK / "araclar").is_dir() else "cover_art.tex")
hedef.write_text(metin, encoding="utf-8")
print(f"{hedef.name}: {len(satirlar)} lines; bin counts: {sayim.tolist()}")
