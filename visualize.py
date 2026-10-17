"""Визуализации: люди (из статьи Meier & Sprenger 2010) vs 3 LLM-модели.

Бенчмарки людей (Table 1, N=541):
  IDF01 = 0.83; present-biased 36%; future-biased 9%; consistent ~55%.
Градиенты (Table 2): male -0.057 IDF; college +0.042 IDF; age -0.002 IDF/год.

Выход: analysis_disc/figures/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("analysis_disc/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("analysis_disc/dataset.csv")
v = df[df["valid"]].copy()

SHORT = {
    "deepseek/deepseek-v4.1-flash": "DeepSeek",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "qwen/qwen3.8-flash": "Qwen 3.8",
}
ORDER = ["deepseek/deepseek-v4.1-flash", "openai/gpt-6-luna", "qwen/qwen3.8-flash"]
labels = ["Humans\n(paper)"] + [SHORT[m] for m in ORDER]

HUMAN = {"exp": 0.55, "hyp": 0.36, "fut": 0.09, "idf": 0.83}
NORM_IDF = 0.996
C = {"exp": "#4C72B0", "hyp": "#DD8452", "fut": "#937860", "human": "#C44E52", "norm": "#55A868"}

# ---- Fig 1: доли типов дисконтирования -------------------------------------------------
exp = [HUMAN["exp"]] + [(v[v.model_name == m].dtype == "exponential").mean() for m in ORDER]
hyp = [HUMAN["hyp"]] + [(v[v.model_name == m].dtype == "hyperbolic").mean() for m in ORDER]
fut = [HUMAN["fut"]] + [(v[v.model_name == m].dtype == "future_bias").mean() for m in ORDER]
x = np.arange(len(labels))
w = 0.26
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w, exp, w, label="Exponential (β=1)", color=C["exp"])
ax.bar(x, hyp, w, label="Hyperbolic (β<1)", color=C["hyp"])
ax.bar(x + w, fut, w, label="Future-biased (β>1)", color=C["fut"])
for xi, (a, b, c) in enumerate(zip(exp, hyp, fut)):
    for off, val in ((-w, a), (0, b), (w, c)):
        ax.text(xi + off, val + 0.01, f"{val*100:.0f}%", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Доля участников")
ax.set_ylim(0, 1.05)
ax.set_title("Тип дисконтирования: люди (статья) vs LLM")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.axvline(0.5, color="grey", ls=":", lw=1)
fig.tight_layout()
fig.savefig(FIG / "fig1_shares.png", dpi=160)
plt.close(fig)

# ---- Fig 2: распределение IDF1 vs человек и норма -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})
ax = axes[0]
bins = np.linspace(0.3, 1.0, 22)
for m in ORDER:
    ax.hist(v[v.model_name == m]["IDF1"], bins=bins, alpha=0.5, label=SHORT[m], density=True)
ax.axvline(HUMAN["idf"], color=C["human"], lw=2, ls="--", label="Humans (0.83)")
ax.axvline(NORM_IDF, color=C["norm"], lw=2, ls=":", label="≈ norms (0.996)")
ax.set_xlabel("IDF₁ = X₁*/80")
ax.set_ylabel("Плотность")
ax.set_title("Распределение IDF₁ по моделям")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
means = [v[v.model_name == m]["IDF1"].mean() for m in ORDER]
ses = [v[v.model_name == m]["IDF1"].std(ddof=1) / np.sqrt((v.model_name == m).sum()) for m in ORDER]
ax.bar([SHORT[m] for m in ORDER], means, yerr=ses, capsize=5, color=[C["exp"], C["hyp"], C["fut"]])
ax.axhline(HUMAN["idf"], color=C["human"], ls="--", lw=2)
ax.text(2.35, HUMAN["idf"] + 0.01, "Humans", color=C["human"], ha="right", fontsize=9)
ax.axhline(NORM_IDF, color=C["norm"], ls=":", lw=2)
ax.set_ylim(0.6, 1.02)
ax.set_ylabel("Средний IDF₁ ± SE")
ax.set_title("Уровень терпеливости")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig2_idf.png", dpi=160)
plt.close(fig)

# ---- Fig 3: β̂ распределение vs экспонента --------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
data = [v[v.model_name == m]["beta"].dropna().values for m in ORDER]
parts = ax.violinplot(data, showmeans=True, showextrema=False)
for pc in parts["bodies"]:
    pc.set_alpha(0.5)
ax.axhline(1.0, color=C["norm"], lw=2, ls="--")
ax.text(3.45, 1.005, "Exponential β=1", color=C["norm"], ha="right", fontsize=9)
ax.set_xticks(np.arange(1, 4))
ax.set_xticklabels([SHORT[m] for m in ORDER])
ax.set_ylabel("β̂ = X₁*/X₃*")
ax.set_title("Present bias: β̂ (β<1 — гиперболическое, β=1 — экспоненциальное)")
ax.set_ylim(0.5, 1.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig3_beta.png", dpi=160)
plt.close(fig)

# ---- Fig 4: средний IDF1 по профилям (модель × профиль) -------------------------------
piv = v[v.profile_id != "P0"].pivot_table(index="profile_id", columns="model_name", values="IDF1", aggfunc="mean")
piv = piv[ORDER]
fig, ax = plt.subplots(figsize=(9, 4.6))
X = np.arange(len(piv.index))
w = 0.26
for i, m in enumerate(ORDER):
    ax.bar(X + (i - 1) * w, piv[m].values, w, label=SHORT[m])
ax.axhline(HUMAN["idf"], color=C["human"], ls="--", lw=2)
ax.text(len(piv.index) - 0.5, HUMAN["idf"] + 0.01, "Humans 0.83", color=C["human"], ha="right", fontsize=9)
ax.set_xticks(X)
ax.set_xticklabels(piv.index)
ax.set_xlabel("Профиль (пол × колледж × возраст)")
ax.set_ylabel("Средний IDF₁")
ax.set_ylim(0, 1.05)
ax.set_title("IDF₁ по демографическим профилям P1–P8")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig4_profiles.png", dpi=160)
plt.close(fig)

# ---- Fig 5: градиенты по полу / образованию / возрасту --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, (col, name) in zip(axes, [("gender", "Пол"), ("education", "Образование"), ("age_group", "Возраст")]):
    g = v[v.profile_id != "P0"].groupby(["model_name", col])["IDF1"].mean().unstack()
    g = g.loc[ORDER]
    g.index = [SHORT[m] for m in ORDER]
    g.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(f"IDF₁ по: {name}")
    ax.set_ylabel("Средний IDF₁")
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, fontsize=8, title=None)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Демографические градиенты LLM (у людей из Table 2 они слабые)")
fig.tight_layout()
fig.savefig(FIG / "fig5_gradients.png", dpi=160)
plt.close(fig)

# ---- Fig 6: расстояние до человека -----------------------------------------------------
rows = []
for m in ORDER:
    mm = v[v.model_name == m]
    rows.append({
        "model": SHORT[m],
        "d_idf": abs(mm["IDF1"].mean() - HUMAN["idf"]),
        "d_hyp": abs((mm.dtype == "hyperbolic").mean() - HUMAN["hyp"]),
        "d_fut": abs((mm.dtype == "future_bias").mean() - HUMAN["fut"]),
    })
d = pd.DataFrame(rows).set_index("model")
fig, ax = plt.subplots(figsize=(8, 4.6))
d.plot(kind="bar", ax=ax, rot=0)
ax.set_ylabel("|LLM − люди|")
ax.set_title("Расхождение с людьми (меньше — ближе)")
ax.legend(["Средний IDF₁", "Доля hyperbolic", "Доля future-biased"], frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig6_distance.png", dpi=160)
plt.close(fig)

print("Сохранено в", FIG)
for f in sorted(FIG.glob("*.png")):
    print(" -", f.name)
print("\nСводка для подписей:")
print(d.round(3).to_string())
