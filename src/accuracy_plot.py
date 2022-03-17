import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

ERROR_FILE_ROOT = Path(__file__).resolve().parent.parent
FIG_FOLDER_ROOT = Path(__file__).resolve().parent.parent
PLOT_OUTPUT_PATH = os.path.join(FIG_FOLDER_ROOT, "accuracy_vs_ec_plot/")


CSVS = [
    # ERROR_FILE_ROOT / "Heteroscedastic_error.csv",
    # ERROR_FILE_ROOT / "Outliers_error.csv",
    ERROR_FILE_ROOT / "output" / "Norm_error.csv",
    ERROR_FILE_ROOT / "output" / "Uniform+_error.csv",
    ERROR_FILE_ROOT / "output" / "Uniform-_error.csv",
]

# create a single dataframe
df = pd.concat([pd.read_csv(csv).assign(Dataset=csv.stem) for csv in CSVS]).reset_index(drop=True)


def ec_vs_accuracy():
    for method in df.Method.unique():
        data = df[df.Method.eq(method)]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x="EC", y="R2", hue="Dataset", style="Regressor")
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.xlabel("Error Consistency (EC)")
        plt.ylabel("R-Squered (R^2)")
        plt.title(f"Error Consistency ({method}) vsR-Squered (R^2)")
        plt.legend(loc="upper left")
        plt.savefig(PLOT_OUTPUT_PATH + f"{method}_R2.png")
        plt.clf()


if __name__ == "__main__":
    ec_vs_accuracy()
