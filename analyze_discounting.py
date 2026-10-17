"""Сравнение ответов LLM с экспоненциальным и гиперболическим дисконтированием.

Классификация каждого валидного ответа (по X1*, X3*):
  exponential : X1* == X3*  (динамическая согласованность, beta = 1)
  hyperbolic  : X1* < X3*   (present bias, beta = X1*/X3* < 1)
  future_bias : X1* > X3*   (beta > 1, не предсказывается ни одной теорией)

Контроли: female, college, older + FE моделей (только P1–P8; P0 без демографии).

Использование: python3 analyze_discounting.py full_*.csv
Выход: analysis_disc/ (shares.csv, beta.csv, tests.csv, logit_present.txt,
        ols_beta.txt, ols_idf1.txt, by_model.csv, disc_hist.png)
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

X_GRID = [75, 70, 65, 60, 50, 40, 30]
Y_VAL = 80
# Бенчмарки из опубликованной статьи, Table 1 Panel C, primary sample N=541
HUMAN_IDF1, HUMAN_PRESENT, HUMAN_FUTURE = 0.83, 0.36, 0.09
# Градиенты из Table 2 статьи: OLS IDF (col 2), OLS PresentBias (col 6)
PAPER_GRAD = {
    "age_yr": {"idf": -0.002, "idf_p": 0.01, "hyp": -0.003, "hyp_p": 0.05},
    "male": {"idf": -0.057, "idf_p": 0.01, "hyp": 0.076, "hyp_p": 0.2},
    "college": {"idf": 0.042, "idf_p": 0.05, "hyp": 0.181, "hyp_p": 0.01},
}


def paper_star(seq):
    """X* по конвенции статьи: последний X, при котором выбрано A (нижняя граница).
    all_B -> 75.0 (правоцензурировано), all_A -> 30.0 (левоцензурировано).
    Reverse/multiple -> None (исключаются, как multiple у людей)."""
    if all(s == "B" for s in seq):
        return 75.0
    if all(s == "A" for s in seq):
        return 30.0
    ch = [i for i in range(6) if seq[i] != seq[i + 1]]
    if len(ch) == 1 and seq[ch[0]] == "A":
        return float(X_GRID[ch[0]])
    return None


def classify(seq):
    if all(s == "B" for s in seq):
        return "all_B", 77.5
    if all(s == "A" for s in seq):
        return "all_A", 25.0
    ch = [i for i in range(6) if seq[i] != seq[i + 1]]
    if len(ch) == 1:
        i = ch[0]
        xs = (X_GRID[i] + X_GRID[i + 1]) / 2
        return ("forward", xs) if seq[i] == "A" else ("reverse", xs)
    for i in range(6):
        if seq[i] == "A" and seq[i + 1] == "B":
            return "multiple", (X_GRID[i] + X_GRID[i + 1]) / 2
    for i in range(6):
        if seq[i] == "B" and seq[i + 1] == "A":
            return "multiple_rev", (X_GRID[i] + X_GRID[i + 1]) / 2
    return "multiple", None


def load(path):
    df = pd.read_csv(path)
    x1s, x3s = [], []
    for raw in df["raw_response"]:
        try:
            m = re.search(r"\{.*\}", str(raw), re.DOTALL)
            ch = json.loads(m.group(0)[: m.group(0).rfind("}") + 1])["choices"]
            _, s1 = classify(ch["today_vs_1m"])
            _, s3 = classify(ch["6m_vs_7m"])
        except Exception:
            s1, s3 = None, None
        x1s.append(s1)
        x3s.append(s3)
    df["X1"] = x1s
    df["X3"] = x3s
    ok = df["X1"].notna() & df["X3"].notna()
    # валидны только прямые паттерны; reverse/multiple исключены (как multiple у людей)
    df["valid"] = False
    df["dtype"] = "excluded"
    fwd = lambda c: c in ("forward", "all_B", "all_A")  # noqa
    df["beta"] = df["X1"] / df["X3"]
    return df, ok, fwd


def main():
    out = Path("analysis_disc")
    out.mkdir(exist_ok=True)
    frames = [pd.read_csv(p) for p in sys.argv[1:]]
    # пересчёт классификации единым кодом
    rec = []
    for p in sys.argv[1:]:
        df = pd.read_csv(p)
        x1s, x3s, p1s, p3s = [], [], [], []
        for raw in df["raw_response"]:
            try:
                t = re.sub(r"```(?:json)?", "", str(raw))
                m = re.search(r"\{.*\}", t, re.DOTALL)
                ch = json.loads(m.group(0)[: m.group(0).rfind("}") + 1])["choices"]
                a, s1 = classify(ch["today_vs_1m"])
                b, s3 = classify(ch["6m_vs_7m"])
            except Exception:
                a, s1, b, s3 = "parse", None, "parse", None
            p1s.append(a)
            p3s.append(b)
            x1s.append(s1)
            x3s.append(s3)
        df["pat1"], df["pat3"], df["X1"], df["X3"] = p1s, p3s, x1s, x3s
        rec.append(df)
    df = pd.concat(rec, ignore_index=True)
    ok_pat = ("forward", "all_B", "all_A")
    df["valid"] = df["pat1"].isin(ok_pat) & df["pat3"].isin(ok_pat) & df["X1"].notna() & df["X3"].notna()
    # X* по конвенции статьи (нижняя граница интервала): пересчёт из raw
    x1p, x3p = [], []
    for raw in df["raw_response"]:
        try:
            t = re.sub(r"```(?:json)?", "", str(raw))
            m = re.search(r"\{.*\}", t, re.DOTALL)
            ch = json.loads(m.group(0)[: m.group(0).rfind("}") + 1])["choices"]
            x1p.append(paper_star(ch["today_vs_1m"]))
            x3p.append(paper_star(ch["6m_vs_7m"]))
        except Exception:
            x1p.append(None)
            x3p.append(None)
    df["X1p"] = pd.to_numeric(pd.Series(x1p), errors="coerce")
    df["X3p"] = pd.to_numeric(pd.Series(x3p), errors="coerce")
    df["valid"] = df["valid"] & df["X1p"].notna() & df["X3p"].notna()
    df["beta"] = df["X1p"] / df["X3p"]
    df["IDF1"] = df["X1p"] / Y_VAL  # IDF(0,1): today vs 1 month
    df["IDF3"] = df["X3p"] / Y_VAL  # IDF(6,7): 6 vs 7 months
    df["IDF"] = (df["IDF1"] + df["IDF3"]) / 2  # основная мера статьи: среднее двух рам
    df.loc[df["valid"] & (df["X1p"] == df["X3p"]), "dtype"] = "exponential"
    df.loc[df["valid"] & (df["X1p"] < df["X3p"]), "dtype"] = "hyperbolic"
    df.loc[df["valid"] & (df["X1p"] > df["X3p"]), "dtype"] = "future_bias"
    df["hyperbolic"] = (df["dtype"] == "hyperbolic").astype(int)
    df.to_csv(out / "dataset.csv", index=False)

    models = sorted(df["model_name"].unique())
    shares, tests = [], []
    for m in models:
        v = df[(df["model_name"] == m) & df["valid"]]
        n_all = int((df["model_name"] == m).sum())
        n = len(v)
        shares.append({
            "model": m, "N": n_all, "N_valid": n, "excl": 1 - n / n_all,
            "sh_exp": (v["dtype"] == "exponential").mean(),
            "sh_hyp": (v["dtype"] == "hyperbolic").mean(),
            "sh_fut": (v["dtype"] == "future_bias").mean(),
            "beta_mean": v["beta"].mean(), "beta_med": v["beta"].median(), "beta_sd": v["beta"].std(ddof=1),
            "IDF1_mean": v["IDF1"].mean(), "IDF1_sd": v["IDF1"].std(ddof=1),
            "IDF3_mean": v["IDF3"].mean(), "IDF3_sd": v["IDF3"].std(ddof=1),
            "IDF_mean": v["IDF"].mean(), "IDF_sd": v["IDF"].std(ddof=1),
        })
        # t-тест beta против 1 (норма экспоненты); t-тест IDF1 против 0.86 (люди)
        t_b, p_b = stats.ttest_1samp(v["beta"], 1.0)
        t_i, p_i = stats.ttest_1samp(v["IDF1"], HUMAN_IDF1)
        t_i3, p_i3 = stats.ttest_1samp(v["IDF3"], HUMAN_IDF1)
        t_ia, p_ia = stats.ttest_1samp(v["IDF"], HUMAN_IDF1)
        ph = (v["dtype"] == "hyperbolic").mean()
        se = (HUMAN_PRESENT * (1 - HUMAN_PRESENT) / n) ** 0.5
        z = (ph - HUMAN_PRESENT) / se
        pf = (v["dtype"] == "future_bias").mean()
        sef = (HUMAN_FUTURE * (1 - HUMAN_FUTURE) / n) ** 0.5
        zf = (pf - HUMAN_FUTURE) / sef
        tests.append({"model": m, "t_beta_vs1": t_b, "p_beta_vs1": p_b,
                      "t_idf1_vs_human": t_i, "p_idf1_vs_human": p_i,
                      "t_idf3_vs_human": t_i3, "p_idf3_vs_human": p_i3,
                      "t_idf_avg_vs_human": t_ia, "p_idf_avg_vs_human": p_ia,
                      "z_hyp_vs_human": z, "p_hyp_vs_human": 2 * (1 - stats.norm.cdf(abs(z))),
                      "z_fut_vs_human": zf, "p_fut_vs_human": 2 * (1 - stats.norm.cdf(abs(zf)))})
    pd.DataFrame(shares).to_csv(out / "shares.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)

    # MW/KS по beta между моделями
    mw = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = df[(df["model_name"] == models[i]) & df["valid"]]["beta"]
            b = df[(df["model_name"] == models[j]) & df["valid"]]["beta"]
            u, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            ks, p_ks = stats.ks_2samp(a, b)
            mw.append({"pair": f"{models[i]} vs {models[j]}", "MW_p": p_mw, "KS_D": ks, "KS_p": p_ks})
    pd.DataFrame(mw).to_csv(out / "model_tests.csv", index=False)

    # Регрессии с контролями (P1–P8)
    reg = df[df["valid"] & (df["profile_id"] != "P0")].copy()
    reg["female"] = (reg["gender"] == "female").astype(int)
    reg["college"] = (reg["education"] == "college").astype(int)
    reg["older"] = (reg["age_group"] == "older").astype(int)
    smf.logit("hyperbolic ~ female + college + older + C(model_name)", data=reg).fit(
        disp=False, cov_type="HC0").summary().as_text()
    with open(out / "logit_present.txt", "w") as f:
        f.write(str(smf.logit("hyperbolic ~ female + college + older + C(model_name)",
                              data=reg).fit(disp=False, cov_type="HC0").summary()))
    with open(out / "ols_beta.txt", "w") as f:
        f.write(str(smf.ols("beta ~ female + college + older + C(model_name)",
                            data=reg).fit(cov_type="HC3").summary()))
    with open(out / "ols_idf1.txt", "w") as f:
        f.write(str(smf.ols("IDF1 ~ female + college + older + C(model_name)",
                            data=reg).fit(cov_type="HC3").summary()))
    with open(out / "ols_idf3.txt", "w") as f:
        f.write(str(smf.ols("IDF3 ~ female + college + older + C(model_name)",
                            data=reg).fit(cov_type="HC3").summary()))
    with open(out / "ols_idf_avg.txt", "w") as f:
        f.write(str(smf.ols("IDF ~ female + college + older + C(model_name)",
                            data=reg).fit(cov_type="HC3").summary()))
    bym = []
    for m in models:
        r = reg[reg["model_name"] == m]
        fb = smf.ols("beta ~ female + college + older", data=r).fit(cov_type="HC3")
        fi = smf.ols("IDF1 ~ female + college + older", data=r).fit(cov_type="HC3")
        fi3 = smf.ols("IDF3 ~ female + college + older", data=r).fit(cov_type="HC3")
        fia = smf.ols("IDF ~ female + college + older", data=r).fit(cov_type="HC3")
        lg = smf.logit("hyperbolic ~ female + college + older", data=r).fit(disp=False, cov_type="HC0")
        bym.append({"model": m, "N": len(r),
                    "idf_female_b": fi.params["female"], "idf_female_p": fi.pvalues["female"],
                    "idf_college_b": fi.params["college"], "idf_college_p": fi.pvalues["college"],
                    "idf_older_b": fi.params["older"], "idf_older_p": fi.pvalues["older"],
                    "idf3_college_b": fi3.params["college"], "idf3_college_p": fi3.pvalues["college"],
                    "idfavg_college_b": fia.params["college"], "idfavg_college_p": fia.pvalues["college"],
                    "beta_college_b": fb.params["college"], "beta_college_p": fb.pvalues["college"],
                    "beta_female_b": fb.params["female"], "beta_female_p": fb.pvalues["female"],
                    "beta_older_b": fb.params["older"], "beta_older_p": fb.pvalues["older"],
                    "logit_female_b": lg.params["female"], "logit_female_p": lg.pvalues["female"],
                    "logit_college_b": lg.params["college"], "logit_college_p": lg.pvalues["college"],
                    "logit_older_b": lg.params["older"], "logit_older_p": lg.pvalues["older"]})
    pd.DataFrame(bym).to_csv(out / "by_model.csv", index=False)

    # График beta
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        v = df[(df["model_name"] == m) & df["valid"]]["beta"]
        ax.hist(v, bins=25, alpha=0.7)
        ax.axvline(1.0, linestyle="--", label="экспонента β=1")
        ax.axvline(v.mean(), color="red", linestyle="-", label=f"среднее {v.mean():.3f}")
        ax.set_title(f"{m} (N={len(v)})")
        ax.legend(fontsize=8)
    fig.suptitle("β̂ = X1*/X3* по моделям")
    fig.tight_layout()
    fig.savefig(out / "beta_hist.png", dpi=150)

    print("== SHARES (exp / hyp / fut) ==")
    print(pd.DataFrame(shares).round(3).to_string(index=False))
    print("\n== TESTS ==")
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print("\n== MW/KS beta ==")
    print(pd.DataFrame(mw).round(4).to_string(index=False))
    print("\n== BY MODEL (controls) ==")
    print(pd.DataFrame(bym).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
