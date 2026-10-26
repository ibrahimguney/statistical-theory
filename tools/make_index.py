"""Subject index tools.

  python3 tools/make_index.py add      # inserts \\index commands into the ch*.tex files
  python3 tools/make_index.py compile  # main.idx -> main.ind (alphabetical order, letter headings)

Build order:  pdflatex main  ->  make_index.py compile  ->  pdflatex main (x2)

To add a term, append a triple (displayed form, regular expression to search for in the source,
chapter numbers or None) to TERMS and rerun "add"; the script removes the old \\index commands
and inserts them all afresh. A term is indexed only at its first occurrence in each \\section of
each chapter; the page where it appears in a definition title is set in bold.
Mathematics, code, headings, label/citation arguments, summary boxes and the Exercises section
are not scanned.
"""
import glob
import pathlib
import re
import sys

# (displayed form in the index, regular expression, only in these chapters [None: all])
TERIMLER = [
    # Chapter 1
    (r"$\sigma$-algebra", r"\$\\sigma\$-algebra", None),
    ("probability space", r"probability spaces?", None),
    ("Kolmogorov axioms", r"Kolmogorov('s)? (probability )?axioms?", None),
    ("conditional probability", r"conditional probabilit(y|ies)", None),
    ("independence (events)", r"independen(ce|t)", {1}),
    ("Bayes' formula", r"Bayes' formula", None),
    ("law of total probability", r"law of total probability", None),
    ("Borel--Cantelli lemmas", r"Borel--Cantelli", None),
    ("inclusion--exclusion principle", r"inclusion--exclusion", None),
    ("Boole's inequality", r"Boole's inequality", None),
    # Chapter 2
    ("random variable", r"random variables?", {2}),
    ("distribution function (CDF)", r"distribution functions?|CDF", {2}),
    ("quantile function", r"quantile functions?|generali[sz]ed inverse", None),
    ("inverse transform method", r"inverse transform (method|sampling)", None),
    ("probability integral transform", r"probability integral transform", None),
    ("empirical distribution function", r"empirical (distribution function|CDF)", None),
    ("Glivenko--Cantelli theorem", r"Glivenko--Cantelli", None),
    ("DKW inequality", r"DKW|Dvoretzky--Kiefer--Wolfowitz", None),
    (r"Borel $\sigma$-algebra", r"Borel \$\\sigma\$-algebra", None),
    # Chapter 3
    ("expectation", r"expectations?|expected values?", {3}),
    ("variance", r"variances?\b", {3}),
    ("covariance", r"covariances?(?! matri)", {3, 4}),
    ("correlation", r"correlations?", {3, 4}),
    ("moment generating function (MGF)", r"moment generating functions?|MGF", {3, 4, 5, 6}),
    ("Cauchy--Schwarz inequality", r"Cauchy--Schwarz", None),
    ("Jensen's inequality", r"Jensen", None),
    ("tower property", r"tower (property|rule)", None),
    ("law of total variance", r"law of total variance|total variance", None),
    ("conditional expectation", r"conditional expectations?", None),
    ("skewness", r"skewness", None),
    ("kurtosis", r"kurtosis", None),
    ("law of the unconscious statistician (LOTUS)", r"LOTUS|unconscious statistician", None),
    # Chapter 4
    ("joint distribution", r"joint (distributions?|densit(y|ies)|CDF)", {4}),
    ("marginal distribution", r"marginal", {4}),
    ("convolution", r"convolution", None),
    ("Jacobian method", r"Jacobian", None),
    ("Box--Muller transform", r"Box--Muller", None),
    ("covariance matrix", r"covariance matri(x|ces)", None),
    ("multivariate normal distribution", r"multivariate normal", None),
    ("order statistics", r"order statistics?", None),
    ("copula", r"copulas?", None),
    ("Sklar's theorem", r"Sklar", None),
    # Chapter 5
    ("Markov's inequality", r"Markov's inequality", None),
    ("Chebyshev's inequality", r"Chebyshev", None),
    ("Chernoff bound", r"Chernoff", None),
    ("Hoeffding's inequality", r"Hoeffding('s)? (inequality|lemma|bound)", None),
    ("convergence in probability", r"convergence in probability|converges? in probability", None),
    ("almost sure convergence", r"almost sure convergence|converges? almost surely", None),
    ("law of large numbers!weak (WLLN)", r"weak law of large numbers|WLLN", None),
    ("law of large numbers!strong (SLLN)", r"strong law of large numbers|SLLN", None),
    ("Khinchin's theorem", r"Khinchin", None),
    ("continuous mapping theorem", r"continuous mapping theorem", None),
    ("Monte Carlo method", r"Monte Carlo", {5}),
    ("hierarchy of convergence", r"(modes|hierarchy) of convergence", None),
    # Chapter 6
    ("convergence in distribution", r"convergence in distribution|converges? in distribution", None),
    ("characteristic function", r"characteristic functions?", None),
    ("portmanteau lemma", r"portmanteau", None),
    ("inversion formula", r"inversion formula", None),
    ("Lévy continuity theorem", r"Lévy('s)? continuity", None),
    ("central limit theorem (CLT)", r"central limit theorem|CLT", None),
    ("Lindeberg condition", r"Lindeberg condition", None),
    ("Lindeberg--Feller theorem", r"Lindeberg--Feller", None),
    ("Berry--Esseen theorem", r"Berry--Esseen", None),
    ("Cramér--Wold device", r"Cramér--Wold", None),
    ("Slutsky's theorem", r"Slutsky", None),
    ("Helly selection theorem", r"Helly", None),
    ("tightness", r"tight(ness)?\b", {6}),
    # Chapter 7
    ("estimator", r"estimators?", {7}),
    ("unbiased estimator", r"unbiased estimators?", None),
    ("mean squared error (MSE)", r"mean squared error|MSE", {7}),
    ("bias--variance decomposition", r"bias--variance", None),
    ("consistency", r"consisten(cy|t estimator)", {7, 12}),
    ("loss function", r"loss functions?", None),
    ("risk function", r"risk functions?", None),
    ("admissibility", r"admissib(le|ility)|inadmissib(le|ility)", None),
    ("method of moments", r"method of moments", None),
    ("James--Stein estimator", r"James--Stein", None),
    # Chapter 8
    ("score function", r"score functions?", None),
    ("Fisher information", r"Fisher information(?! matri)", None),
    ("Fisher information matrix", r"Fisher information matri(x|ces)", None),
    ("information identity", r"information identity", None),
    ("observed information", r"observed (Fisher )?information", None),
    ("Cramér--Rao lower bound (CRLB)", r"Cramér--Rao|CRLB", None),
    ("efficient estimator", r"efficient estimators?", None),
    ("regularity conditions", r"regularity conditions?", None),
    ("nuisance parameter", r"nuisance parameters?", None),
    # Chapter 9
    ("sufficient statistic", r"sufficient statistics?", None),
    ("minimal sufficient statistic", r"minimal sufficient", None),
    ("factorisation theorem", r"factori[sz]ation", None),
    ("exponential family", r"exponential famil(y|ies)", None),
    ("Rao--Blackwell theorem", r"Rao--Blackwell", None),
    ("natural parameter", r"natural parameters?", None),
    # Chapter 10
    ("complete statistic", r"complete (and sufficient |sufficient )?statistics?|completeness", None),
    ("ancillary statistic", r"ancillary", None),
    ("UMVUE", r"UMVUE", None),
    ("Lehmann--Scheffé theorem", r"Lehmann--Scheffé", None),
    ("Basu's theorem", r"Basu", None),
    # Chapter 11
    ("likelihood function", r"likelihood functions?", None),
    ("log-likelihood", r"log-likelihoods?", None),
    ("maximum likelihood estimator (MLE)", r"maximum likelihood|MLE", {11, 12}),
    ("profile likelihood", r"profile likelihood|induced likelihood", None),
    ("Zehna's invariance principle", r"Zehna|invariance principle", None),
    ("Neyman--Scott problem", r"Neyman--Scott", None),
    ("Newton--Raphson method", r"Newton--Raphson", None),
    # Chapter 12
    ("Kullback--Leibler divergence", r"Kullback--Leibler", None),
    ("information inequality", r"information inequality", None),
    ("asymptotic normality", r"asymptotic(ally)? normal(ity)?", None),
    ("asymptotic relative efficiency", r"asymptotic (relative )?efficiency", None),
    ("Wald confidence interval", r"Wald (confidence )?intervals?", None),
    ("delta method", r"delta method", None),
    ("variance-stabilising transformation", r"variance-stabilis(ing|ation)", None),
    ("Hodges' superefficiency", r"Hodges'|superefficien", None),
    (r"Fisher's $z$ transformation", r"Fisher('s)? \$z\$", None),
    # Chapter 13
    ("hypothesis test", r"hypothesis test(s|ing)?", {13}),
    ("null hypothesis", r"null hypothes[ei]s", {13}),
    ("type I and type II errors", r"type I\b|type II\b", None),
    ("power function", r"power functions?", None),
    ("most powerful (MP) test", r"most powerful tests?|MP tests?", None),
    ("Neyman--Pearson lemma", r"Neyman--Pearson", None),
    ("randomised test", r"randomised tests?", {13, 14}),
    (r"$p$-value", r"\$p\$-values?", None),
    ("likelihood ratio", r"likelihood ratios?", {13, 14}),
    # Chapter 14
    ("uniformly most powerful (UMP) test", r"uniformly most powerful|UMP\b", None),
    ("monotone likelihood ratio (MLR)", r"monotone likelihood|MLR", None),
    ("Karlin--Rubin theorem", r"Karlin--Rubin", None),
    ("unbiased test", r"unbiased tests?", None),
    ("UMPU test", r"UMPU", None),
    ("Neyman--Pearson lemma!generalised", r"generali[sz]ed Neyman--Pearson", None),
    # Chapter 15
    ("likelihood ratio test (LRT)", r"LRT|likelihood ratio tests?", None),
    ("Wilks' theorem", r"Wilks", None),
    ("Wald test", r"Wald tests?", None),
    ("score (Rao) test", r"score tests?|Rao test", None),
    ("goodness of fit", r"goodness[- ]of[- ]fit", None),
    (r"Pearson chi-square ($X^2$) statistic", r"Pearson('s)? \$X\^2\$|Pearson('s)? chi-square", None),
    ("multinomial distribution", r"multinomial", None),
    # Chapter 16
    ("confidence interval", r"confidence intervals?", {16}),
    ("coverage probability", r"coverage", {16}),
    ("pivot", r"pivot(al|s)?\b", None),
    ("test--interval duality", r"duality", None),
    (r"$t$ distribution (Student)", r"Student|\$t\$ distribution", None),
    ("chi-square distribution", r"chi-square distributions?", None),
    (r"$F$ distribution", r"\$F\$ distribution", None),
    ("Wilson interval", r"Wilson", None),
    ("Clopper--Pearson interval", r"Clopper--Pearson", None),
    ("Agresti--Coull interval", r"Agresti--Coull", None),
    ("Welch approximation", r"Welch", None),
    ("normal sample theorem", r"normal sample theorem", None),
    # Chapter 17
    ("linear model", r"linear models?", {17}),
    ("least squares (LS)", r"least squares|\bLS\b", {17}),
    ("hat matrix", r"hat matri(x|ces)", None),
    ("leverage", r"leverages?", None),
    ("Gauss--Markov theorem", r"Gauss--Markov", None),
    ("BLUE", r"BLUE", None),
    (r"coefficient of determination ($R^2$)", r"coefficient of determination", None),
    (r"$F$-test", r"\$F\$-tests?", None),
    ("heteroscedasticity", r"heteroscedastic(ity)?", None),
    ("sandwich estimator", r"sandwich", None),
    ("ridge regression", r"ridge", None),
    ("idempotent matrix", r"idempotent", None),
    ("prediction interval", r"prediction intervals?", None),
    # Chapter 18
    ("bootstrap", r"bootstrap", {18}),
    ("plug-in principle", r"plug-in", None),
    ("bootstrap!parametric", r"parametric bootstrap", None),
    ("bootstrap!BCa interval", r"BCa", None),
    ("bootstrap!percentile interval", r"percentile", None),
    ("jackknife", r"jackknife", None),
    ("permutation test", r"permutation tests?", None),
    ("exchangeability", r"exchangeab(le|ility)", None),
    # Chapter 19
    ("prior distribution", r"priors?\b", {19}),
    ("posterior distribution", r"posteriors?\b", {19}),
    ("conjugate prior", r"conjuga(te|cy)", None),
    ("credible interval", r"credible (intervals?|sets?)", None),
    ("HPD region", r"HPD", None),
    ("posterior predictive distribution", r"posterior predictive", None),
    ("Bayes estimator", r"Bayes estimators?", None),
    ("Jeffreys prior", r"Jeffreys", None),
    ("Bernstein--von Mises theorem", r"Bernstein--von Mises", None),
    ("Lindley's paradox", r"Lindley", None),
    ("Bayes factor", r"Bayes factors?", None),
    ("Metropolis algorithm", r"Metropolis", None),
    ("MCMC", r"MCMC", None),
    ("shrinkage", r"shrinkage", None),
    # Chapter 20
    ("decision rule", r"decision rules?", None),
    ("minimax rule", r"minimax", None),
    ("Bayes risk", r"Bayes risks?", None),
    ("least favourable prior", r"least favourable", None),
    ("equaliser rule", r"equali[sz]er", None),
    ("Blyth's method", r"Blyth", None),
    ("limiting Bayes method", r"limiting Bayes", None),
    ("risk set", r"risk sets?", None),
    ("randomised decision rule", r"randomised (decision )?rules?", {20}),
    # Chapter 21
    ("nonparametric methods", r"nonparametric", None),
    ("distribution-free statistic", r"distribution-free", None),
    ("sign test", r"sign tests?", None),
    ("Wilcoxon signed-rank test", r"Wilcoxon signed-rank|signed-rank", None),
    ("Wilcoxon--Mann--Whitney test", r"Mann--Whitney", None),
    ("Kolmogorov--Smirnov test", r"Kolmogorov--Smirnov|KS\b", None),
    ("Lilliefors problem", r"Lilliefors", None),
    ("kernel density estimation", r"kernel (density )?estimat(e|or|ion)s?", None),
    ("bandwidth", r"bandwidths?", None),
    ("Silverman's rule", r"Silverman", None),
    ("Pitman efficiency", r"Pitman", None),
    ("Hodges--Lehmann estimator", r"Hodges--Lehmann", None),
    ("MISE", r"A?MISE", None),
    # Chapter 22
    ("multiple testing", r"multiple testing", None),
    ("family-wise error rate (FWER)", r"family-wise error|FWER", None),
    ("false discovery rate (FDR)", r"false discovery|FDR", None),
    ("Bonferroni correction", r"Bonferroni", None),
    ("Šidák correction", r"Šidák", None),
    ("Holm procedure", r"Holm", None),
    ("Benjamini--Hochberg procedure", r"Benjamini--Hochberg|BH\b", None),
    ("Benjamini--Yekutieli procedure", r"Benjamini--Yekutieli", None),
    (r"adjusted $p$-value", r"adjusted \$p\$", None),
    # Chapter 23
    ("EM algorithm", r"EM algorithm|EM\b", {23}),
    ("missing data", r"missing data", None),
    ("mixture distribution", r"mixtures?", {23}),
    ("generalised linear model (GLM)", r"generali[sz]ed linear|GLMs?", None),
    ("link function", r"link functions?|canonical link", None),
    ("logistic regression", r"logistic regression", None),
    ("Poisson regression", r"Poisson regression", None),
    ("Fisher scoring", r"Fisher scoring", None),
    ("IRLS", r"IRLS|iteratively reweighted", None),
    ("deviance", r"deviances?", {23}),
    ("complete separation", r"complete separation", None),
    ("overdispersion", r"overdispersion", None),
    ("exponential dispersion family", r"exponential dispersion famil(y|ies)", None),
]

TANIM_ORT = ("definition",)
TEOREM_ORT = ("theorem", "lemma", "proposition", "corollary", "definition")

ALFABE = "abcdefghijklmnopqrstuvwxyz"
SIRA = {h: i for i, h in enumerate(ALFABE)}


def kucuk(s):
    return s.lower()


def anahtar(goster):
    """Sort key from the displayed form (letters and digits only)."""
    s = re.sub(r"\\([A-Za-z]+)", r"\1", goster)
    s = s.replace("Š", "S").replace("é", "e").replace("è", "e").replace("--", " ")
    s = re.sub(r"[^\w ]", " ", s)
    return re.sub(r"\s+", " ", kucuk(s)).strip()


def sira_anahtari(s):
    out = []
    for ch in anahtar(s):
        if ch == " ":
            out.append(-1)
        elif ch in SIRA:
            out.append(SIRA[ch] + 20)
        elif ch.isdigit():
            out.append(int(ch))
        else:
            out.append(100)
    return out


def ilk_harf_buyuk(rx):
    """Allow the first letter of each alternative to be capitalised (sentence start)."""
    def tek(alt):
        if alt and alt[0].isalpha() and alt[0].islower():
            b = alt[0].upper()
            return f"[{alt[0]}{b}]" + alt[1:]
        return alt
    parcalar, derin, bas = [], 0, 0
    for i, ch in enumerate(rx):
        if ch == "(":
            derin += 1
        elif ch == ")":
            derin -= 1
        elif ch == "|" and derin == 0 and (i == 0 or rx[i - 1] != "\\"):
            parcalar.append(rx[bas:i]); bas = i + 1
    parcalar.append(rx[bas:])
    return "|".join(tek(p) for p in parcalar)


# ---------------------------------------------------------------- maskeleme
def eslesen_parantez(s, i, ac="{", kapa="}"):
    """s[i]==ac; returns the index of the matching closing bracket."""
    d = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2; continue
        if c == ac:
            d += 1
        elif c == kapa:
            d -= 1
            if d == 0:
                return j
        j += 1
    return len(s) - 1


ATLA_KOMUT = ("section", "subsection", "subsubsection", "chapter", "paragraph", "caption", "label",
              "ref", "eqref", "cite", "pycode", "includegraphics", "hyperref", "texttt", "url",
              "chaptermark", "index", "input", "subfile")
MATEMATIK_ORT = ("equation", "equation*", "align", "align*", "gather", "gather*", "multline",
                 "multline*", "lstlisting", "summary", "tabular", "longtable")


def maske(s):
    m = bytearray(len(s))  # 1 = not scanned

    def isaretle(a, b):
        for k in range(a, min(b, len(s))):
            m[k] = 1

    # everything after the Exercises section
    k = s.find("\\section{Exercises}")
    if k >= 0:
        isaretle(k, len(s))
    # environments
    for ort in MATEMATIK_ORT:
        for mm in re.finditer(r"\\begin\{" + re.escape(ort) + r"\}", s):
            son = s.find("\\end{" + ort + "}", mm.end())
            isaretle(mm.start(), (son if son >= 0 else len(s)) + len(ort) + 6)
    i = 0
    while i < len(s):
        c = s[i]
        if c == "%" and (i == 0 or s[i - 1] != "\\"):
            j = s.find("\n", i); j = len(s) if j < 0 else j
            isaretle(i, j); i = j; continue
        if s.startswith("\\[", i):
            j = s.find("\\]", i); isaretle(i, j + 2); i = j + 2; continue
        if c == "$" and (i == 0 or s[i - 1] != "\\"):
            j = i + 1
            while j < len(s) and not (s[j] == "$" and s[j - 1] != "\\"):
                j += 1
            isaretle(i, j + 1); i = j + 1; continue
        if c == "\\":
            mm = re.match(r"\\([A-Za-z]+)\*?", s[i:])
            if mm:
                ad = mm.group(1); j = i + mm.end()
                if ad in ATLA_KOMUT:
                    bas = i
                    while j < len(s) and s[j] in "[{":
                        j = eslesen_parantez(s, j, s[j], "]" if s[j] == "[" else "}") + 1
                    isaretle(bas, j); i = j; continue
                if ad == "begin":  # \begin{env}[optional]
                    j = eslesen_parantez(s, j) + 1
                    if j < len(s) and s[j] == "[":
                        j = eslesen_parantez(s, j, "[", "]") + 1
                    isaretle(i, j); i = j; continue
                if ad in ("end", "item"):
                    if ad == "end":
                        j = eslesen_parantez(s, j) + 1
                    elif j < len(s) and s[j] == "[":
                        j = eslesen_parantez(s, j, "[", "]") + 1
                    isaretle(i, j); i = j; continue
                isaretle(i, j); i = j; continue
            isaretle(i, i + 2); i += 2; continue
        i += 1
    return m


def index_komutu(goster, kalin=False):
    ana = goster.split("!")
    parca = "!".join(f"{anahtar(p)}@{p}" for p in ana)
    return "\\index{" + parca + ("|textbf" if kalin else "") + "}"


INDEX_RX = re.compile(r"\\index\{(?:[^{}]|\{[^{}]*\})*\}")


def ekle():
    derlenmis = [(g, re.compile(r"(?<![\w\\])(?:" + ilk_harf_buyuk(rx) + ")"), b) for g, rx, b in TERIMLER]
    toplam = 0
    for f in sorted(glob.glob("ch[0-9][0-9]_*.tex")):
        no = int(re.match(r"ch(\d+)", f).group(1))
        p = pathlib.Path(f)
        s = INDEX_RX.sub("", p.read_text(encoding="utf-8"))
        msk = maske(s)
        bolumler = [mm.start() for mm in re.finditer(r"\\section\{", s)] + [len(s)]
        bolumler = [0] + bolumler
        ekler = []  # (position, text)
        # 1) definition/theorem titles
        for mm in re.finditer(r"\\begin\{(" + "|".join(TEOREM_ORT) + r")\}\[((?:[^\[\]]|\[[^\]]*\])*)\](\\label\{[^}]*\})?", s):
            if msk_alistirma(s, mm.start()):
                continue
            baslik = mm.group(2)
            for g, rx, b in derlenmis:
                if b is not None and no not in b:
                    continue
                if rx.search(baslik):
                    ekler.append((mm.end(), index_komutu(g, kalin=mm.group(1) in TANIM_ORT), g, mm.end()))
        baslikta = {(g, bolum_no(bolumler, k)) for k, _, g, _ in ekler}
        # 2) first occurrence in each \section
        for g, rx, b in derlenmis:
            if b is not None and no not in b:
                continue
            for a, z in zip(bolumler[:-1], bolumler[1:]):
                if (g, bolum_no(bolumler, a)) in baslikta:
                    continue
                for mm in rx.finditer(s, a, z):
                    if msk[mm.start()]:
                        continue
                    e = mm.end()
                    w = re.match(r"[\w]*", s[e:])
                    e += w.end()
                    if any(msk[k] for k in range(mm.start(), e)):
                        continue
                    ekler.append((e, index_komutu(g), g, e))
                    break
        ekler.sort(key=lambda t: t[0], reverse=True)
        for konum, metin, _, _ in ekler:
            s = s[:konum] + metin + s[konum:]
        toplam += len(ekler)
        p.write_text(s, encoding="utf-8")
    print(f"{toplam} \\index commands inserted ({len(TERIMLER)} terms)")


def msk_alistirma(s, k):
    a = s.find("\\section{Exercises}")
    return a >= 0 and k > a


def bolum_no(sinirlar, k):
    for i in range(len(sinirlar) - 1):
        if sinirlar[i] <= k < sinirlar[i + 1]:
            return i
    return len(sinirlar)


# ---------------------------------------------------------------- derleme
def derle(idx="main.idx", ind="main.ind"):
    girdiler = {}  # (anahtar tuple) -> {"goster": [..], "sayfa": {n: kalin}}
    for satir in pathlib.Path(idx).read_text(encoding="utf-8").splitlines():
        mm = re.match(r"\\indexentry\{(.*)\}\{(\d+)\}$", satir)
        if not mm:
            continue
        govde, sayfa = mm.group(1), int(mm.group(2))
        bicim = ""
        if "|" in govde:
            govde, bicim = govde.split("|", 1)
        seviyeler = []
        for parca in govde.split("!"):
            a, _, g = parca.partition("@")
            seviyeler.append((a, g or a))
        anah = tuple(a for a, _ in seviyeler)
        d = girdiler.setdefault(anah, {"goster": [g for _, g in seviyeler], "sayfa": {}})
        d["sayfa"][sayfa] = d["sayfa"].get(sayfa, False) or ("textbf" in bicim)

    def sayfalar(d):
        out = []
        for n in sorted(d):
            out.append(f"\\textbf{{\\hyperpage{{{n}}}}}" if d[n] else f"\\hyperpage{{{n}}}")
        return ", ".join(out)

    # create an empty top-level entry if only subentries exist
    for anah in list(girdiler):
        if len(anah) == 2 and anah[:1] not in girdiler:
            girdiler[anah[:1]] = {"goster": girdiler[anah]["goster"][:1], "sayfa": {}}

    sirali = sorted(girdiler, key=lambda a: [sira_anahtari(x) for x in a])
    satirlar = ["\\begin{theindex}",
                "\\phantomsection\\addcontentsline{toc}{chapter}{\\indexname}",
                "\\raggedright", ""]
    onceki_harf = None
    for anah in sirali:
        d = girdiler[anah]
        harf = anahtar(anah[0])[:1]
        if len(anah) == 1 and harf != onceki_harf:
            if onceki_harf is not None:
                satirlar.append("\n  \\indexspace")
            buyuk = harf.upper()
            satirlar.append(f"  {{\\bfseries\\large {buyuk}}}\\nopagebreak")
            onceki_harf = harf
        komut = "\\item" if len(anah) == 1 else "\\subitem"
        sp = sayfalar(d["sayfa"])
        satirlar.append(f"  {komut} {d['goster'][-1]}" + (f", {sp}" if sp else ""))
    satirlar.append("\n\\end{theindex}")
    pathlib.Path(ind).write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print(f"{len(girdiler)} index entries -> {ind}")


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else "compile"
    {"add": ekle, "compile": derle}[komut]()
