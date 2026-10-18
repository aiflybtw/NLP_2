import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
import statsmodels.formula.api as smf

X_GRID = [75, 70, 65, 60, 50, 40, 30]
Y_VAL = 80
HUMAN_IDF, HUMAN_PRESENT, HUMAN_FUTURE = 0.83, 0.36, 0.09
YEARS_GAP = 23
TABLE2 = {
    "IDF": {"age_yr": -0.002, "male": -0.057, "college": 0.042, "const": 0.975},
    "Interval IDF_T": {"age_yr": -0.002, "male": -0.070, "college": 0.050, "const": None},
    "PresentBias": {"age_yr": -0.005, "male": 0.092, "college": 0.181, "const": 0.441},
}
TERMS = ["older", "male", "college"]


def paper_star(seq):
    if all(s == "B" for s in seq):
        return 75.0
    if all(s == "A" for s in seq):
        return 30.0
    ch = [i for i in range(6) if seq[i] != seq[i + 1]]
    if len(ch) == 1 and seq[ch[0]] == "A":
        return float(X_GRID[ch[0]])
    return None


def classify(seq):
    if any(s not in ("A", "B") for s in seq):
        return "invalid"
    if all(s == "B" for s in seq):
        return "all_B"
    if all(s == "A" for s in seq):
        return "all_A"
    ch = [i for i in range(6) if seq[i] != seq[i + 1]]
    if len(ch) == 1 and seq[ch[0]] == "A":
        return "forward"
    if len(ch) == 1:
        return "reverse"
    return "multiple"


def frame_interval(seq):
    if all(s == "A" for s in seq):
        return 0.0, X_GRID[-1] / Y_VAL
    if all(s == "B" for s in seq):
        return X_GRID[0] / Y_VAL, np.inf
    ch = [i for i in range(6) if seq[i] != seq[i + 1]]
    if len(ch) == 1 and seq[ch[0]] == "A":
        i = ch[0]
        return X_GRID[i + 1] / Y_VAL, X_GRID[i] / Y_VAL
    return None


def parse_choices(text):
    try:
        t = re.sub(r"```(?:json)?", "", str(text))
        m = re.search(r"\{.*\}", t, re.DOTALL)
        return json.loads(m.group(0)[: m.group(0).rfind("}") + 1])["choices"]
    except Exception:
        return None


def interval_fit(sub, model_fe):
    lo, hi, meta = [], [], []
    for _, r in sub.iterrows():
        ch = parse_choices(r["raw_response"])
        if not ch:
            continue
        for k in ("today_vs_1m", "6m_vs_7m"):
            iv = frame_interval(ch[k])
            if iv is None:
                continue
            lo.append(iv[0])
            hi.append(iv[1])
            meta.append((r["older"], r["male"], r["college"],
                         1.0 if k == "6m_vs_7m" else 0.0, r["mn"]))
    lo, hi = np.array(lo, float), np.array(hi, float)
    names = ["const", "older", "male", "college", "frame67"]
    cols = [np.ones(len(lo)), [m[0] for m in meta], [m[1] for m in meta],
            [m[2] for m in meta], [m[3] for m in meta]]
    if model_fe:
        for m in sorted(sub["mn"].unique())[1:]:
            names.append("model=" + m)
            cols.append([1.0 if x[4] == m else 0.0 for x in meta])
    X = np.column_stack(cols)

    def nll(p):
        mu = X @ p[:-1]
        s = np.exp(p[-1])
        a, b = (lo - mu) / s, (hi - mu) / s
        ll = np.zeros(len(mu))
        fin = np.isfinite(lo) & np.isfinite(hi)
        ll[fin] = np.log(np.clip(stats.norm.cdf(b[fin]) - stats.norm.cdf(a[fin]), 1e-300, None))
        lc = ~np.isfinite(lo)
        ll[lc] = np.log(np.clip(stats.norm.cdf(b[lc]), 1e-300, None))
        rc = ~np.isfinite(hi)
        ll[rc] = np.log(np.clip(1 - stats.norm.cdf(a[rc]), 1e-300, None))
        return -ll.sum()

    mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2,
                   np.where(np.isfinite(hi), hi, lo))
    start = np.r_[np.linalg.lstsq(X, mid, rcond=None)[0], np.log(mid.std())]
    res = optimize.minimize(nll, start, method="BFGS")
    se = np.sqrt(np.diag(res.hess_inv))
    out = {}
    for i, nm in enumerate(names):
        z = res.x[i] / se[i]
        out[nm] = (res.x[i], se[i], 2 * (1 - stats.norm.cdf(abs(z))))
    return out, len(lo)


def main():
    out = Path("analysis_disc")
    out.mkdir(exist_ok=True)
    rec = []
    for p in sys.argv[1:]:
        df = pd.read_csv(p)
        p1s, p3s, x1s, x3s = [], [], [], []
        for raw in df["raw_response"]:
            ch = parse_choices(raw)
            if ch is None:
                p1s.append("parse"), p3s.append("parse"), x1s.append(None), x3s.append(None)
                continue
            p1s.append(classify(ch["today_vs_1m"]))
            p3s.append(classify(ch["6m_vs_7m"]))
            x1s.append(paper_star(ch["today_vs_1m"]))
            x3s.append(paper_star(ch["6m_vs_7m"]))
        df["pat1"], df["pat3"] = p1s, p3s
        df["X1p"] = pd.to_numeric(pd.Series(x1s), errors="coerce")
        df["X3p"] = pd.to_numeric(pd.Series(x3s), errors="coerce")
        rec.append(df)
    df = pd.concat(rec, ignore_index=True)
    ok = ("forward", "all_B", "all_A")
    df["valid"] = df["pat1"].isin(ok) & df["pat3"].isin(ok) & df["X1p"].notna() & df["X3p"].notna()
    df["beta"] = df["X1p"] / df["X3p"]
    df["IDF1"] = df["X1p"] / Y_VAL
    df["IDF3"] = df["X3p"] / Y_VAL
    df["IDF"] = (df["IDF1"] + df["IDF3"]) / 2
    df["dtype"] = "excluded"
    df.loc[df["valid"] & (df["X1p"] == df["X3p"]), "dtype"] = "exponential"
    df.loc[df["valid"] & (df["X1p"] < df["X3p"]), "dtype"] = "hyperbolic"
    df.loc[df["valid"] & (df["X1p"] > df["X3p"]), "dtype"] = "future_bias"
    df["present"] = (df["dtype"] == "hyperbolic").astype(int)
    df["male"] = (df["gender"] == "male").astype(int)
    df["college"] = (df["education"] == "college").astype(int)
    df["older"] = (df["age_group"] == "older").astype(int)
    df["mn"] = df["model_name"].str.split("/").str[0]
    df.to_csv(out / "dataset.csv", index=False)

    models = sorted(df["model_name"].unique())
    shares, tests = [], []
    for m in models:
        v = df[(df["model_name"] == m) & df["valid"]]
        n_all, n = int((df["model_name"] == m).sum()), int(((df["model_name"] == m) & df["valid"]).sum())
        shares.append({
            "model": m, "N": n_all, "N_valid": n, "excl": 1 - n / n_all,
            "sh_exp": (v["dtype"] == "exponential").mean(),
            "sh_hyp": (v["dtype"] == "hyperbolic").mean(),
            "sh_fut": (v["dtype"] == "future_bias").mean(),
            "beta_mean": v["beta"].mean(), "beta_sd": v["beta"].std(ddof=1),
            "IDF1_mean": v["IDF1"].mean(), "IDF1_sd": v["IDF1"].std(ddof=1),
            "IDF3_mean": v["IDF3"].mean(), "IDF3_sd": v["IDF3"].std(ddof=1),
            "IDF_mean": v["IDF"].mean(), "IDF_sd": v["IDF"].std(ddof=1),
        })
        t_b, p_b = stats.ttest_1samp(v["beta"], 1.0)
        t_1, p_1 = stats.ttest_1samp(v["IDF1"], HUMAN_IDF)
        t_3, p_3 = stats.ttest_1samp(v["IDF3"], HUMAN_IDF)
        t_a, p_a = stats.ttest_1samp(v["IDF"], HUMAN_IDF)
        ph = (v["dtype"] == "hyperbolic").mean()
        z = (ph - HUMAN_PRESENT) / (HUMAN_PRESENT * (1 - HUMAN_PRESENT) / n) ** 0.5
        pf = (v["dtype"] == "future_bias").mean()
        zf = (pf - HUMAN_FUTURE) / (HUMAN_FUTURE * (1 - HUMAN_FUTURE) / n) ** 0.5
        tests.append({"model": m, "t_beta_vs1": t_b, "p_beta_vs1": p_b,
                      "t_idf1_vs_human": t_1, "p_idf1_vs_human": p_1,
                      "t_idf3_vs_human": t_3, "p_idf3_vs_human": p_3,
                      "t_idf_avg_vs_human": t_a, "p_idf_avg_vs_human": p_a,
                      "z_hyp_vs_human": z, "p_hyp_vs_human": 2 * (1 - stats.norm.cdf(abs(z))),
                      "z_fut_vs_human": zf, "p_fut_vs_human": 2 * (1 - stats.norm.cdf(abs(zf)))})
    pd.DataFrame(shares).to_csv(out / "shares.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)

    mw = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = df[(df["model_name"] == models[i]) & df["valid"]]["beta"]
            b = df[(df["model_name"] == models[j]) & df["valid"]]["beta"]
            u, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            ks, p_ks = stats.ks_2samp(a, b)
            mw.append({"pair": f"{models[i]} vs {models[j]}", "MW_p": p_mw, "KS_D": ks, "KS_p": p_ks})
    pd.DataFrame(mw).to_csv(out / "model_tests.csv", index=False)

    reg = df[df["valid"] & (df["profile_id"] != "P0")].copy()
    allc = []

    def dump(scope, dep, fit):
        for term in fit.params.index:
            if term.startswith("C("):
                continue
            allc.append({"scope": scope, "dep": dep, "term": term, "coef": fit.params[term],
                         "se": fit.bse[term], "p": fit.pvalues[term], "N": int(fit.nobs)})

    for scope, sub in [("pooled", reg)] + [(m.split("/")[0], reg[reg["model_name"] == m]) for m in models]:
        fe = " + C(model_name)" if scope == "pooled" else ""
        for dep, dv in [("IDF1", "IDF1"), ("IDF3", "IDF3"), ("IDF_avg", "IDF"), ("beta", "beta")]:
            dump(scope, dep, smf.ols(f"{dv} ~ older + male + college{fe}", data=sub).fit(cov_type="HC3"))
        dump(scope, "present", smf.logit(f"present ~ older + male + college{fe}",
                                         data=sub).fit(disp=False, cov_type="HC0"))
    pd.DataFrame(allc).to_csv(out / "all_coefficients.csv", index=False)

    ivc = []
    for scope, sub in [("pooled", reg)] + [(m.split("/")[0], reg[reg["model_name"] == m]) for m in models]:
        coefs, n = interval_fit(sub, model_fe=(scope == "pooled"))
        for term, (b, se, p) in coefs.items():
            if term.startswith("model="):
                continue
            ivc.append({"scope": scope, "term": term, "coef": b, "se": se, "p": p, "N": n})
    pd.DataFrame(ivc).to_csv(out / "interval.csv", index=False)

    t2 = []
    for scope, sub in [("pooled", reg)] + [(m.split("/")[0], reg[reg["model_name"] == m]) for m in models]:
        fe = " + C(model_name)" if scope == "pooled" else ""
        o_idf = smf.ols(f"IDF ~ older + male + college{fe}", data=sub).fit(cov_type="HC3")
        o_pr = smf.ols(f"present ~ older + male + college{fe}", data=sub).fit(cov_type="HC3")
        coefs, n = interval_fit(sub, model_fe=(scope == "pooled"))
        for term in TERMS + ["const"]:
            src = "Intercept" if term == "const" else term
            t2.append({"scope": scope, "dep": "IDF", "term": term,
                       "coef": o_idf.params[src], "se": o_idf.bse[src], "p": o_idf.pvalues[src], "N": len(sub)})
            t2.append({"scope": scope, "dep": "PresentBias", "term": term,
                       "coef": o_pr.params[src], "se": o_pr.bse[src], "p": o_pr.pvalues[src], "N": len(sub)})
        for term in TERMS + ["const"]:
            b, se, p = coefs[term]
            t2.append({"scope": scope, "dep": "Interval IDF_T", "term": term,
                       "coef": b, "se": se, "p": p, "N": n})
    for dep, d in TABLE2.items():
        for term, coef in d.items():
            t2.append({"scope": "Humans", "dep": dep, "term": term, "coef": coef, "se": "", "p": "", "N": ""})
    pd.DataFrame(t2).to_csv(out / "table2.csv", index=False)

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        v = df[(df["model_name"] == m) & df["valid"]]["beta"]
        ax.hist(v, bins=25, alpha=0.7)
        ax.axvline(1.0, linestyle="--")
        ax.axvline(v.mean(), color="red")
        ax.set_title(f"{m} (N={len(v)})")
    fig.tight_layout()
    fig.savefig(out / "beta_hist.png", dpi=150)

    print(pd.DataFrame(shares).round(3).to_string(index=False))
    print()
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print()
    print(pd.DataFrame(t2).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
