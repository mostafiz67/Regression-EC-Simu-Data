import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

OUTPUT_FILE_ROOT = Path(__file__).resolve().parent.parent
FIG_FOLDER_ROOT = Path(__file__).resolve().parent.parent
PLOT_OUTPUT_PATH = os.path.join(FIG_FOLDER_ROOT, "accuracy_vs_ec_plot/")


CSVS = [
    OUTPUT_FILE_ROOT / "output" / "Heteroscedastic_error.csv",
    OUTPUT_FILE_ROOT / "output" / "Outliers_error.csv",
    OUTPUT_FILE_ROOT / "output" / "Norm_error.csv",
    OUTPUT_FILE_ROOT / "output" / "Uniform+_error.csv",
    OUTPUT_FILE_ROOT / "output" / "Uniform-_error.csv",
    OUTPUT_FILE_ROOT / "output" / "Uniform+-_error.csv",
]

# create a single dataframe
df = pd.concat([pd.read_csv(csv).assign(Dataset=csv.stem) for csv in CSVS]).reset_index(drop=True)
df['Dataset'].replace(regex=True,inplace=True,to_replace='_error',value=r'')

def ec_vs_accuracy():
    for method in df.Method.unique():
        data = df[df.Method.eq(method)]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x="EC", y="MAE", hue="Dataset", style="Regressor")
        plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.xlabel("Error Consistency (EC)")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title(f"Error Consistency ({method}) vs Mean Absolute Error (MAE)")
        plt.legend(loc="upper left")
        plt.savefig(PLOT_OUTPUT_PATH + f"{method}_MAE.png")
        plt.clf()

def ec_vs_accuracy_dataset():
    for dataset in df.Dataset.unique():
        # print(df, dataset)
        data = df[df["Dataset"] == dataset]
        # print(data)
        data = data.drop(data[(data.Method == "intersection_union_distance")].index) # droping distance for better view
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=data, x="EC", y="MAE", hue="Method", style="Regressor", s=100)
        plt.legend(bbox_to_anchor=(1, 0.5), borderaxespad=0, loc="center left")
        plt.xlabel("Error Consistency (EC)")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title(f"Error Consistency vs MAE for the {dataset} dataset ")
        plt.savefig(PLOT_OUTPUT_PATH + f"MAE_{dataset}_only_without_distance.png", bbox_inches='tight')
        plt.clf()



if __name__ == "__main__":
    # ec_vs_accuracy()
    ec_vs_accuracy_dataset()
