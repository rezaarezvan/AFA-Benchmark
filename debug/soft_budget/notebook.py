# %%
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

df = pd.read_csv("mock_evaluation_data.csv")
# %%
df
# %%
grouped = df.groupby(["Method", "Cost parameter", "Dataset"])
summary_df = pd.DataFrame(
    {
        "Avg. features chosen": grouped["Features chosen"].mean(),
        "Std. features chosen": grouped["Features chosen"].std(),
        "Avg. accuracy (external)": grouped["Accuracy (external)"].mean(),
        "Std. accuracy (external)": grouped["Accuracy (external)"].std(),
    }
).reset_index()
summary_df
# %%
fg = sns.relplot(
    data=summary_df,
    x="Avg. features chosen",
    y="Avg. accuracy (external)",
    hue="Method",
    col="Dataset",
    kind="line",
    marker="o",
    facet_kws={"sharey": False, "sharex": False},
).set(ylim=(0, 1))
fg.figure.savefig("accuracy_vs_features_chosen.png", dpi=300)
