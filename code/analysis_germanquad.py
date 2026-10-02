"""Paired significance tests and figure for BM25 vs. multilingual-e5-large (GermanQuAD pilot).

Reads the per-query first-gold ranks from output/analysis/per_query_germanquad.csv (produced by the
ranking in repro_germanquad.py; the CSV is left untouched) and writes significance_germanquad.json,
SUMMARY.md and the figure to code/output/analysis/.
"""
from __future__ import annotations
import json, sys
from importlib.metadata import version as pv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "pylib"))
from local_dataset_io import load_jsonl  # noqa: E402

D, OUT = CODE / "output" / "germanquad", CODE / "output" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
SEED, B_PERM, B_BOOT, KS = 42, 100_000, 10_000, [1, 5, 10]
qids = [str(q["query_id"]) for q in load_jsonl(D / "queries.normalized.jsonl")]
df = pd.read_csv(OUT / "per_query_germanquad.csv", dtype=str, keep_default_na=False)
assert df["query_id"].tolist() == qids, "CSV query order differs from queries.normalized.jsonl"


def arr(col):
    return np.array([int(r) if str(r) != "none" else 0 for r in df[col]], dtype=int)


rk = {"bm25": arr("rank_bm25"), "e5": arr("rank_e5")}
n = len(df)
succ = lambda r, k: ((r > 0) & (r <= k)).astype(float)  # noqa: E731
rr = lambda r, k=10: np.where((r > 0) & (r <= k), 1.0 / np.maximum(r, 1), 0.0)  # noqa: E731

# consistency with REPRO.md
repro = {"bm25": json.load(open(CODE / "output/repro/bm25_germanquad_repro.json"))["metrics"],
         "e5": json.load(open(CODE / "output/repro/dense_e5_germanquad_repro.json"))["metrics"]}
checks, ok = [], True
for m in ("bm25", "e5"):
    for row in repro[m]:
        k = row["k"]
        for name, val in (("success_at_k", succ(rk[m], k).mean()), ("mrr_at_k", rr(rk[m], k).mean())):
            diff = abs(val - row[name]); ok &= diff < 1e-12
            checks.append((m, k, name, row[name], val, diff))

rng = np.random.default_rng(SEED)
clusters = np.array([g.split(";")[0] for g in df["gold_doc_id"]])
_, cinv = np.unique(clusters, return_inverse=True)
ncl = cinv.max() + 1
csize = np.bincount(cinv)


def boot_ci(d: np.ndarray, rng) -> tuple[list, list]:
    idx = rng.integers(0, n, size=(B_BOOT, n))
    q = np.percentile(d[idx].mean(1), [2.5, 97.5])
    csum = np.bincount(cinv, weights=d, minlength=ncl)
    w = rng.multinomial(ncl, np.full(ncl, 1 / ncl), size=B_BOOT)  # resample clusters with replacement
    # ratio estimator: query-weighted mean of d over the resampled clusters
    c = np.percentile((w @ csum) / (w @ csize), [2.5, 97.5])
    return [float(x) for x in q], [float(x) for x in c]


def sign_flip(d: np.ndarray, rng) -> float:
    obs, cnt, chunk = abs(d.mean()), 0, 10_000
    for _ in range(B_PERM // chunk):
        s = rng.choice([-1.0, 1.0], size=(chunk, n))
        cnt += int((np.abs((s * d).mean(1)) >= obs - 1e-15).sum())
    return (cnt + 1) / (B_PERM + 1)


def cluster_sign_flip(d: np.ndarray, rng) -> float:
    """Two-sided sign-flip test with one random sign per gold-doc cluster (cluster sums of d)."""
    dc = np.bincount(cinv, weights=d, minlength=ncl) / n
    obs, cnt, chunk = abs(dc.sum()), 0, 10_000
    for _ in range(B_PERM // chunk):
        s = rng.choice([-1.0, 1.0], size=(chunk, ncl))
        cnt += int((np.abs(s @ dc) >= obs - 1e-15).sum())
    return (cnt + 1) / (B_PERM + 1)


def holm(ps: list[float]) -> list[float]:
    out, run = [0.0] * len(ps), 0.0
    for rnk, i in enumerate(np.argsort(ps, kind="stable")):
        run = max(run, min(1.0, (len(ps) - rnk) * ps[i])); out[i] = run
    return out


def fp(p: float, mc: bool) -> str:
    """Format p; Monte-Carlo p at the floor 1/(B+1) is shown as an upper bound."""
    return f"<= 1/(B+1) = {1 / (B_PERM + 1):.3g}" if mc and p <= 1 / (B_PERM + 1) * (1 + 1e-9) else f"{p:.3g}"


tests = []
for k in KS:
    a, b_ = succ(rk["bm25"], k), succ(rk["e5"], k)
    b = int(((a == 1) & (b_ == 0)).sum()); c = int(((a == 0) & (b_ == 1)).sum())  # b: BM25 only, c: E5 only
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    ci_q, ci_c = boot_ci(b_ - a, rng)
    tests.append({"name": f"Success@{k}", "test": "exact McNemar (two-sided binomial)", "statistic": min(b, c),
                  "b_bm25_only": b, "c_e5_only": c, "mean_bm25": a.mean(), "mean_e5": b_.mean(),
                  "diff_e5_minus_bm25": float((b_ - a).mean()), "p": float(p),
                  "ci95_query_bootstrap": ci_q, "ci95_cluster_bootstrap": ci_c})
a, b_ = rr(rk["bm25"]), rr(rk["e5"])
d = b_ - a
ci_q, ci_c = boot_ci(d, rng)
tests.append({"name": "MRR@10", "test": "paired sign-flip randomization (two-sided)", "statistic": float(d.mean()),
              "b_bm25_only": None, "c_e5_only": None, "mean_bm25": a.mean(), "mean_e5": b_.mean(),
              "diff_e5_minus_bm25": float(d.mean()), "p": sign_flip(d, rng),
              "ci95_query_bootstrap": ci_q, "ci95_cluster_bootstrap": ci_c})
# Holm over the family of 4 (query level, unchanged)
for t, h in zip(tests, holm([t["p"] for t in tests])):
    t["p_holm"] = h
# cluster-level sign-flip (own rng, does not affect the query-level streams)
rng_c = np.random.default_rng(SEED)
dvecs = [succ(rk["e5"], k) - succ(rk["bm25"], k) for k in KS] + [rr(rk["e5"]) - rr(rk["bm25"])]
for t, dv in zip(tests, dvecs):
    t["p_cluster_signflip"] = cluster_sign_flip(dv, rng_c)
    t["p_query_mc_floor"] = bool(t["test"].startswith("paired sign-flip") and t["p"] <= 1 / (B_PERM + 1) * (1 + 1e-9))
    t["p_cluster_mc_floor"] = bool(t["p_cluster_signflip"] <= 1 / (B_PERM + 1) * (1 + 1e-9))
for t, h in zip(tests, holm([t["p_cluster_signflip"] for t in tests])):
    t["p_holm_cluster_signflip"] = h
csz = np.sort(csize)
edges = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]
hist = [{"size": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">={lo}"),
         "n_clusters": int(((csize >= lo) & (csize <= hi)).sum())} for lo, hi in edges]
cluster_sizes = {"n_clusters": int(ncl), "min": int(csz[0]), "median": float(np.median(csz)), "max": int(csz[-1]),
                 "histogram": hist}
for t in tests:
    for key in ("mean_bm25", "mean_e5"):
        t[key] = float(t[key])

meta = {"status": "Explorativer Pilot; Auswerteprotokoll noch nicht vom Menschen freigegeben (Bead dls-4ko.25); multiple-comparison family (Holm over Success@1/5/10 + MRR@10) and cluster inference are proposals",
        "dataset": "germanquad (local normalized)", "n_queries": n, "n_gold_clusters": int(ncl),
        "cluster_definition": "first (sorted) relevant doc_id per query",
        "seed": SEED, "B_permutation": B_PERM, "B_bootstrap": B_BOOT, "ci_method": "percentile 95%",
        "sign_flip_p": "(1+#{|mean|>=|obs|})/(B+1)",
        "cluster_sign_flip_p": "one random sign per gold-doc cluster on cluster-summed differences; (1+hits)/(B+1); B=100000, seed 42; Holm computed separately over the 4 cluster p-values",
        "query_level_note": "query-level p-values are optimistic (ignore clustering of queries by gold doc)",
        "mc_floor_note": "Monte-Carlo p-values equal to 1/(B+1) are upper bounds (<= 1/(B+1)); McNemar p-values are exact",
        "cluster_size_distribution": cluster_sizes,
        "mode": "from_csv",
        "packages": {p: pv(p) for p in ("numpy", "scipy", "pandas", "matplotlib", "rank-bm25", "sentence-transformers")},
        "consistency_with_repro_all_exact": bool(ok)}
(OUT / "significance_germanquad.json").write_text(json.dumps({"meta": meta, "tests": tests}, indent=2))

# Figure: dot plot with 95% cluster-bootstrap CIs per retriever (own rng; does not touch the test streams)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10})
rng_f = np.random.default_rng(SEED + 1)
fig, ax = plt.subplots(figsize=(7, 4.6))
for off, m, col, mk, lab in ((-0.05, "bm25", "#7f7f7f", "s", "BM25"), (0.05, "e5", "#1f77b4", "o", "multilingual-e5-large")):
    v, lo, hi = [], [], []
    for k in KS:
        d = succ(rk[m], k)
        csum = np.bincount(cinv, weights=d, minlength=ncl)
        w = rng_f.multinomial(ncl, np.full(ncl, 1 / ncl), size=B_BOOT)
        ci = np.percentile((w @ csum) / (w @ csize), [2.5, 97.5])
        v.append(d.mean()); lo.append(d.mean() - ci[0]); hi.append(ci[1] - d.mean())
    xs = np.arange(len(KS)) + off
    ax.errorbar(xs, v, yerr=[lo, hi], fmt=mk, color=col, capsize=4, markersize=7, label=lab)
    for xv, val, h in zip(xs, v, hi):
        ax.annotate(f"{val:.3f}", (xv, val), xytext=(-11 if m == "bm25" else 11, 0), textcoords="offset points",
                    ha="right" if m == "bm25" else "left", va="center", fontsize=9, color=col)
ax.set_ylim(0, 1.05); ax.set_xlim(-0.5, len(KS) - 0.5); ax.set_xticks(range(len(KS)), [f"k = {k}" for k in KS])
ax.set_ylabel("Success@k (Anteil der Queries)"); ax.set_xlabel("Cutoff k")
ax.set_title(f"GermanQuAD, explorativer Pilot (n = {n} Queries)")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower right", frameon=False)
fig.text(0.5, 0.005, "Explorativ; Balken: 95%-Cluster-Bootstrap-Konfidenzintervall (Cluster = Gold-Dokument, B = 10000)", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "svg"):
    fig.savefig(OUT / f"fig_success_at_k_germanquad.{ext}", dpi=200)


f = lambda v: f"{v:.4f}"  # noqa: E731
L = ["# Analyse GermanQuAD-Pilot: BM25 vs. multilingual-e5-large (dls-4ko.16)", "",
     "**Explorativer Pilot; Auswerteprotokoll noch nicht vom Menschen freigegeben (Bead dls-4ko.25).**", "",
     "Hinweis: Die Mehrfachvergleichsfamilie (Holm über Success@1/5/10 + MRR@10) und die Cluster-Inferenz "
     "(Cluster = relevantes Dokument) sind Vorschläge, keine festgelegte Analyse.", "",
     f"n = {n} Queries, {ncl} Cluster; Seed {SEED}; Permutation B={B_PERM}; Bootstrap B={B_BOOT}; Differenz = E5 minus BM25.", "",
     "## Tests", "", "| Metrik | BM25 | E5 | Diff | b (nur BM25) | c (nur E5) | p query (optimistisch, ignoriert Cluster) | p_Holm query | p cluster-sign-flip | p_Holm cluster | 95%-CI query | 95%-CI cluster |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in tests:
    mcq = t["p_query_mc_floor"]
    L.append(f"| {t['name']} | {f(t['mean_bm25'])} | {f(t['mean_e5'])} | {t['diff_e5_minus_bm25']:+.4f} | {t['b_bm25_only']} | {t['c_e5_only']} | "
             f"{fp(t['p'], mcq)} | {fp(t['p_holm'], mcq)} | {fp(t['p_cluster_signflip'], True)} | {fp(t['p_holm_cluster_signflip'], True)} | "
             f"[{t['ci95_query_bootstrap'][0]:+.4f}, {t['ci95_query_bootstrap'][1]:+.4f}] | "
             f"[{t['ci95_cluster_bootstrap'][0]:+.4f}, {t['ci95_cluster_bootstrap'][1]:+.4f}] |")
L += ["", "Query-Ebene (McNemar exakt; MRR@10 Sign-Flip pro Query) ist optimistisch, da sie die Cluster (Queries mit gleichem Gold-Dokument) ignoriert. "
      "Cluster-Sign-Flip: ein zufälliges Vorzeichen je Cluster auf der je Cluster aggregierten Differenz, zweiseitig, p=(Treffer+1)/(B+1), "
      f"B={B_PERM}, Seed {SEED}; Holm separat über die 4 Cluster-p-Werte. "
      f"Monte-Carlo-p-Werte an der Untergrenze sind als Schranke \"<= 1/(B+1)\" (= {1 / (B_PERM + 1):.3g}) ausgewiesen; McNemar-p sind exakt. "
      "Holm-Lesart: Die vier Cluster-p-Werte liegen bei oder knapp über 1/(B+1); Holm multipliziert den kleinsten mit 4 (Familiengröße), daher 4/(B+1) = 4e-05 als Untergrenze für alle vier p_Holm cluster.", "",
      "## Clustergrößenverteilung (Queries je Gold-Dokument)", "",
      f"Anzahl Cluster: {ncl}; min {cluster_sizes['min']}, Median {cluster_sizes['median']:g}, max {cluster_sizes['max']}.", "",
      "| Clustergröße | Anzahl Cluster |", "|---|---|"] + [f"| {h['size']} | {h['n_clusters']} |" for h in hist]
L += ["", "Tests: Success@k exakter McNemar (zweiseitig); MRR@10 gepaarter Sign-Flip-Randomisierungstest (zweiseitig).", "",
      "## Konsistenzprüfung gegen REPRO.md (Werte aus per_query_germanquad.csv)", "",
      "| Retriever | k | Metrik | REPRO | aus CSV | abs. Diff |", "|---|---|---|---|---|---|"]
for m, k, name, ref, val, dd in checks:
    L.append(f"| {m} | {k} | {name} | {ref:.6f} | {val:.6f} | {dd:.1e} |")
L += ["", f"Ergebnis: {'alle Werte exakt (Diff < 1e-12)' if ok else 'ABWEICHUNGEN VORHANDEN'}.", "",
      "Grafik: fig_success_at_k_germanquad.png/.svg (Punktdiagramm, y-Achse ab 0, 95%-Cluster-Bootstrap-CI, als explorativ gekennzeichnet).", "",
      "Limitationen: Ein Datensatz, ein Lauf; Cluster = erstes relevantes Dokument; E5 auf CPU (numerische Reihenfolge bei Gleichstand nicht garantiert)."]
(OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
print("consistency ok:", ok)
