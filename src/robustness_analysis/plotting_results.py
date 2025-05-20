import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_base_dir


def plot_model_performance(file_path_in, metric, file_path_out):

    df = pd.read_excel(file_path_in)
    df = df[["Drop", "Model", metric]].copy()
    df = df[df["Drop"] != 2]

    df["Model"] = df["Model"].str.split("masking+").str[-1].str.replace("+", " ")
    df["Mean"] = df[metric].str.split("±").str[0].astype(float)

    #drop_values = [0, 2, 4, 8, 16, 24]
    drop_values = sorted(df["Drop"].drop_duplicates().tolist())
    model_values = sorted(df["Model"].drop_duplicates().tolist())

    colors = sns.color_palette("Blues", n_colors=len(drop_values))

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=df,
        x="Model",
        y="Mean",
        hue="Drop",
        hue_order=drop_values,
        order =model_values,
        palette=colors
    )

    # Labels and formatting
    plt.xlabel("Masking strategy", fontsize=14)
    plt.ylabel(f"{metric} (Mean)", fontsize=14)
    #plt.title("PolitiFact dataset", fontsize=16)

    #plt.legend(title="Drop percentage")
    handles, labels = ax.get_legend_handles_labels()
    labels = [f"{int(label)}%" for label in labels]
    ax.legend(handles, labels, title="Drop percentage on PolitiFact dataset: ", title_fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(drop_values), fontsize=14)

    plt.xticks(fontsize=14, rotation=20)

    plt.savefig(file_path_out, format="pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    file_path_in = os.path.join(get_base_dir(), "robustness_analysis", "results_60_15_25_2layers", "results_all_merged.xlsx")
    metric = "F1_macro"
    file_path_out = os.path.join(get_base_dir(), "plots", f"robustness_analysis_{metric}.pdf")
    plot_model_performance(file_path_in, metric, file_path_out)