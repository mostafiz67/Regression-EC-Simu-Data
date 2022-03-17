from pathlib import Path
import pandas as pd
from pandas.core.frame import DataFrame

ROOT = Path(__file__).resolve().parent
CSVS = [
    ROOT / "Heteroscedastic_error.csv",
    ROOT / "Outliers_error.csv",
    ROOT / "Norm_error.csv",
    ROOT / "Uniform+_error.csv",
    ROOT / "Uniform-_error.csv",
]


def correlation_analysis() -> DataFrame:
    corrs = []
    for csv in CSVS:
        error_kind = csv.stem[: csv.stem.find("_")]
        df = pd.read_csv(csv).drop(columns=["Unnamed: 0", "k", "n_rep"])
        corr = df.groupby(["Regressor", "Method"]).corr().round(2).EC
        corr.name = error_kind
        corrs.append(corr)

    corr_df = pd.concat(corrs, axis=1)
    return corr_df


def describe_analysis() -> DataFrame:
    descr = {}
    for csv in CSVS:
        error_kind = csv.stem[: csv.stem.find("_")]
        df = pd.read_csv(csv).drop(columns=["Unnamed: 0", "k", "n_rep"])
        des = df.groupby(["Regressor", "Method"]).describe().round(2).EC
        descr[error_kind] = des.T
        # descr.append(des)

    descr_df = pd.concat(descr, axis="columns")
    return descr_df.stack(1).reorder_levels([0, 1])


if __name__ == "__main__":
    corr_df = correlation_analysis()
    corr_outfile = ROOT / "EC_correlations.csv"
    corr_df.to_csv(corr_outfile)

    # descr_df = describe_analysis()
    # descr_outfile = ROOT / "EC_descriptions.csv"
    # descr_df.to_csv(descr_outfile)
