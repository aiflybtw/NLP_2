"""Выгрузка всех коэффициентов: pooled OLS (IDF1, beta) + pooled Logit (hyperbolic)
с FE моделей, и по-модельные OLS/Logit. Контроли: female, college, older.
Выход: analysis_disc/all_coefficients.csv"""

import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("analysis_disc/dataset.csv")
reg = df[df["valid"] & (df["profile_id"] != "P0")].copy()
reg["female"] = (reg["gender"] == "female").astype(int)
reg["college"] = (reg["education"] == "college").astype(int)
reg["older"] = (reg["age_group"] == "older").astype(int)

rows = []


def dump(name, fit):
    for term in fit.params.index:
        rows.append({"regression": name, "N": int(fit.nobs), "term": term,
                     "coef": fit.params[term], "se": fit.bse[term], "p": fit.pvalues[term]})


dump("pooled_ols_idf1", smf.ols("IDF1 ~ female + college + older + C(model_name)", data=reg).fit(cov_type="HC3"))
dump("pooled_ols_idf3", smf.ols("IDF3 ~ female + college + older + C(model_name)", data=reg).fit(cov_type="HC3"))
dump("pooled_ols_idf_avg", smf.ols("IDF ~ female + college + older + C(model_name)", data=reg).fit(cov_type="HC3"))
dump("pooled_ols_beta", smf.ols("beta ~ female + college + older + C(model_name)", data=reg).fit(cov_type="HC3"))
dump("pooled_logit_hyp", smf.logit("hyperbolic ~ female + college + older + C(model_name)", data=reg).fit(disp=False, cov_type="HC0"))
for m in sorted(reg["model_name"].unique()):
    r = reg[reg["model_name"] == m]
    dump(f"by_{m}_ols_idf1", smf.ols("IDF1 ~ female + college + older", data=r).fit(cov_type="HC3"))
    dump(f"by_{m}_ols_idf3", smf.ols("IDF3 ~ female + college + older", data=r).fit(cov_type="HC3"))
    dump(f"by_{m}_ols_idf_avg", smf.ols("IDF ~ female + college + older", data=r).fit(cov_type="HC3"))
    dump(f"by_{m}_ols_beta", smf.ols("beta ~ female + college + older", data=r).fit(cov_type="HC3"))
    dump(f"by_{m}_logit_hyp", smf.logit("hyperbolic ~ female + college + older", data=r).fit(disp=False, cov_type="HC0"))

out = pd.DataFrame(rows)
out.to_csv("analysis_disc/all_coefficients.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
