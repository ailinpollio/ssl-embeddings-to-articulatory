import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sensor_to_articulator = {
    "Ch12_Z": "UL_Z", "Ch12_X": "UL_X",
    "Ch13_Z": "LL_Z", "Ch13_X": "LL_X",
    "Ch7_Z":  "TT_Z", "Ch7_X":  "TT_X",
    "Ch9_Z":  "TB_Z", "Ch9_X":  "TB_X",
    "Ch8_Z":  "TD_Z", "Ch8_X":  "TD_X",
}

task_to_language = {"A": "L1", "B": "Fake L2", "C": "L2"}

ordered_articulators = ["UL_Z", "UL_X", "LL_Z", "LL_X", "TT_Z", "TT_X", "TB_Z", "TB_X", "TD_Z", "TD_X"]
ordered_langs = ["L1", "Fake L2", "L2"]

cmap = "cividis"
vmin, vmax = 0, 1
threshold = (vmin + vmax) / 2

def make_pivot(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Articulator"] = df["Sensor"].map(sensor_to_articulator)
    df["Language"] = df["LanguageType"].map(task_to_language)

    pivot = (
        df.pivot_table(
            values="Test_Pearson_Correlation",
            index="Language",
            columns="Articulator",
            aggfunc="mean"
        )
        .reindex(index=ordered_langs, columns=ordered_articulators)
    )
    return pivot

def apply_dynamic_text_contrast(ax, threshold: float):
    for text in ax.texts:
        try:
            val = float(text.get_text())
            text.set_color("white" if val < threshold else "black")
        except ValueError:
            pass

fin_path = ""
rus_path = ""  

pivot_fin = make_pivot(fin_path)
pivot_rus = make_pivot(rus_path)

fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), gridspec_kw={"wspace": 0.15})

# just one colorbar
cbar_ax = fig.add_axes([0.93, 0.18, 0.015, 0.64])

# Finnish
ax0 = axes[0]
sns.heatmap(
    pivot_fin,
    ax=ax0,
    annot=True, fmt=".2f",
    cmap=cmap, vmin=vmin, vmax=vmax,
    linewidths=0.8, linecolor="white",
    square=False,                    
    cbar=True, cbar_ax=cbar_ax,
    cbar_kws={"label": "Pearson Correlation", "ticks": np.linspace(vmin, vmax, 6)}
)
ax0.set_title("Finnish")
ax0.set_xlabel("")
ax0.set_ylabel("")
ax0.tick_params(axis='x', rotation=45)
ax0.tick_params(axis='y', rotation=0)
apply_dynamic_text_contrast(ax0, threshold)

# Russian
ax1 = axes[1]
sns.heatmap(
    pivot_rus,
    ax=ax1,
    annot=True, fmt=".2f",
    cmap=cmap, vmin=vmin, vmax=vmax,
    linewidths=0.8, linecolor="white",
    square=False,
    cbar=False
)
ax1.set_title("Russian")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.set_yticklabels([])
ax1.tick_params(axis='y', left=False)
ax1.tick_params(axis='x', rotation=45)

apply_dynamic_text_contrast(ax1, threshold)

plt.tight_layout(rect=[0, 0, 0.92, 1])
out_path = ""
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
