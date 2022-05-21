"""
Author: Md Mostafizur Rahman
File: Calculating Regression Error Consistency using different methods and random dataset

Some code is inherited from https://stackoverflow.com/questions/71430032/how-to-compare-two-numpy-arrays-with-multiple-condition
                            https://stackoverflow.com/questions/71499798/pythonic-way-to-create-pandas-dataframe-based-on-if-else-condition-for-nd-array
"""

from itertools import combinations
from pathlib import Path
from typing import Any, List, Tuple
from warnings import filterwarnings

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler
from tqdm import tqdm
from typing_extensions import Literal

np.random.seed(42)

ECMethod = Literal["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
                    "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]


def generate_normal_data(
    train_size: int,
    test_size: int,
    sigma: float,
    n_outliers: int,
    heteroscedastic_value: int,
    bounds: Tuple[int, int] = (-100, 100)
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    X_train = np.random.uniform(bounds[0], bounds[1], size=train_size)
    y_train = X_train + np.random.normal(0, sigma, size=train_size)
    X_test = np.random.uniform(bounds[0], bounds[1], size=test_size)
    y_test = X_test + np.random.normal(0, sigma, size=test_size)
    return (
        X_train.reshape(-1, 1),
        X_test.reshape(-1, 1),
        y_train.reshape(-1, 1),
        y_test.reshape(-1, 1),
    )


def generate_uniform_add_data(
    train_size: int,
    test_size: int,
    sigma: float,
    n_outliers: int,
    heteroscedastic_value: int,
    bounds: Tuple[int, int] = (-100, 100)
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    X_train = np.random.uniform(bounds[0], bounds[1], size=train_size)
    y_train = X_train + np.random.uniform(0, sigma, size=train_size)
    X_test = np.random.uniform(bounds[0], bounds[1], size=test_size)
    y_test = X_test + np.random.uniform(0, sigma, size=test_size)
    return (
        X_train.reshape(-1, 1),
        X_test.reshape(-1, 1),
        y_train.reshape(-1, 1),
        y_test.reshape(-1, 1),
    )


def generate_uniform_sub_data(
    train_size: int,
    test_size: int,
    sigma: float,
    n_outliers: int,
    heteroscedastic_value: int,
    bounds: Tuple[int, int] = (-100, 100)
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    X_train = np.random.uniform(bounds[0], bounds[1], size=train_size)
    y_train = X_train - np.random.uniform(0, sigma, size=train_size)
    X_test = np.random.uniform(bounds[0], bounds[1], size=test_size)
    y_test = X_test - np.random.uniform(0, sigma, size=test_size)
    return (
        X_train.reshape(-1, 1),
        X_test.reshape(-1, 1),
        y_train.reshape(-1, 1),
        y_test.reshape(-1, 1),
    )
def generate_uniform_add_sub_data(
    train_size: int,
    test_size: int,
    sigma: float,
    n_outliers: int,
    heteroscedastic_value: int,
    bounds: Tuple[int, int] = (-100, 100)
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    X_train_add = np.random.uniform(bounds[0], bounds[1], size=train_size//2)
    y_train_add = X_train_add + np.random.uniform(0, sigma, size=train_size//2)
    X_train_sub = np.random.uniform(bounds[0], bounds[1], size=train_size//2)
    y_train_sub = X_train_sub - np.random.uniform(0, sigma, size=train_size//2)
    X_train = X_train_add + X_train_sub
    y_train = y_train_add + y_train_sub

    X_test_add = np.random.uniform(bounds[0], bounds[1], size=test_size//2)
    y_test_add = X_test_add + np.random.uniform(0, sigma, size=test_size//2)
    X_test_sub = np.random.uniform(bounds[0], bounds[1], size=test_size//2)
    y_test_sub = X_test_add - np.random.uniform(0, sigma, size=test_size//2)
    X_test = X_test_add + X_test_sub
    y_test = y_test_add + y_test_sub
    return (
        X_train.reshape(-1, 1),
        X_test.reshape(-1, 1),
        y_train.reshape(-1, 1),
        y_test.reshape(-1, 1),
    )

def generate_heteroscedastic(
    train_size: int,
    test_size: int,
    n_outliers: int,
    heteroscedastic_value: float,
    sigma: float,
    sigma_h: float = 5.0,
    bounds: Tuple[int, int] = (-100, 100)
) -> Tuple[ndarray, ndarray, ndarray]:
    X_train = np.linspace(bounds[0], bounds[1], num=train_size)
    error = (np.abs(X_train) ** heteroscedastic_value) * np.random.normal(0, sigma_h, size=train_size)
    y_train = error + X_train

    X_test = np.random.uniform(bounds[0], bounds[1], size=test_size)
    y_test = X_test + np.random.normal(0, sigma_h, size=test_size)

    return (
        (X_train).reshape(-1, 1),
        (X_test).reshape(-1, 1),
        (y_train).reshape(-1, 1),
        (y_test).reshape(-1, 1),
    )


def generate_outliers(
    train_size: int,
    test_size: int,
    sigma: float,
    n_outliers: int,
    heteroscedastic_value: int,
    outlier_size: Tuple[float, float] = (1e4, 1e5),
    bounds: Tuple[int, int] = (-100, 100)
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    X_train = np.random.uniform(bounds[0], bounds[1], train_size)
    e_train = np.random.normal(0, sigma, size=train_size)
    outliers = np.random.uniform(outlier_size[0], outlier_size[1], n_outliers)
    y_train = X_train + e_train
    y_train[-n_outliers:] = outliers

    X_test = np.random.uniform(bounds[0], bounds[1], test_size)
    e_test = np.random.normal(0, sigma, size=test_size)
    y_test = X_test + e_test
    return (
        (X_train).reshape(-1, 1),
        (X_test).reshape(-1, 1),
        (y_train).reshape(-1, 1),
        (y_test).reshape(-1, 1),
    )

def regression_ec(residuals: List[ndarray], method: ECMethod) -> List[ndarray]:
    filterwarnings("ignore", "invalid value encountered in true_divide", category=RuntimeWarning)
    consistencies = []
    for pair in combinations(residuals, 2):
        r1, r2 = pair
        r = np.vstack(pair)
        sign = np.sign(np.array(r1) * np.array(r2))
        if method == "ratio-signed":
            consistency = np.multiply(sign, np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0))
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio":
            consistency = np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0)
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio-diff-signed":
            consistency = np.multiply(sign, (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2)))
            consistency[np.isnan(consistency)] = 0
        elif method == "ratio-diff":
            consistency = (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2))
            consistency[np.isnan(consistency)] = 0
        elif method =="intersection_union_sample":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choice_numerator = [np.minimum(r1, r2), np.zeros(len(r1))]
            choice_denominator = [np.maximum(r1, r2), -np.add(r1,r2)]
            numerator = np.select(conditions, choice_numerator)
            denominator = np.select(conditions, choice_denominator, np.add(np.abs(r1), np.abs(r2)))
            consistency = np.divide(numerator, denominator)
            consistency[np.isnan(consistency)] = 1
        elif method =="intersection_union_all":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choice_numerator = [np.minimum(r1, r2), np.zeros(len(r1))]
            choice_denominator = [np.maximum(r1, r2), -np.add(r1,r2)]
            numerator = np.select(conditions, choice_numerator)
            denominator = np.select(conditions, choice_denominator, np.add(np.abs(r1), np.abs(r2)))
            consistency = np.divide(np.sum(numerator), np.sum(denominator)) # all sum and then divide
            consistency = np.nan_to_num(consistency, copy=True, nan=1.0)
        elif method =="intersection_union_distance":
            conditions = [(r1>=0)&(r2>=0), (r1<=0)&(r2<=0)]
            choiceValue = [np.abs(np.subtract(np.abs(r1), np.abs(r2))), np.add(np.abs(r1), np.abs(r2))]
            consistency = np.select(conditions, choiceValue, np.add(np.abs(r1), np.abs(r2)))
        else:
            raise ValueError("Invalid method")
        consistencies.append(consistency)
    filterwarnings("default", "invalid value encountered in true_divide", category=RuntimeWarning)
    return consistencies


def calculate_EC(
    dataset: Tuple[ndarray, ndarray, ndarray, ndarray],
    regressor: Any,
    reg_name: Any,
    method: ECMethod,
    k: int,
    repetitions: int,
) -> DataFrame:
    X_train, X_test, y_train, y_test = dataset
    rep_residuals, rep_mse, rep_r2, rep_mae = [], [], [], []
    for _ in range(repetitions):
        fold_residuals, fold_mse, fold_r2, fold_mae = [], [], [], []
        kf = KFold(n_splits=k, shuffle=True)
        for train_index, _ in kf.split(X_train):
            preds = regressor.fit(X_train[train_index], y_train[train_index]).predict(X_test)
            resid = preds - y_test
            fold_residuals.append(resid)
            # metrics
            fold_mse.append(mean_squared_error(y_test, preds))
            fold_r2.append(r2_score(y_test, preds))
            fold_mae.append(mean_absolute_error(y_test, preds))
        rep_residuals.extend(fold_residuals)
        rep_mse.extend(fold_mse)
        rep_r2.extend(fold_r2)
        rep_mae.extend(fold_mae)
    rep_residuals = np.array(rep_residuals).reshape(repetitions*k, -1)
    consistencies: ndarray = np.array(regression_ec(rep_residuals, method))
    # np.array(consistencies).shape = (n_consistencies, n_samples)
    return DataFrame(
        {
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "Regressor": reg_name,
            "Method": method,
            "k": k,
            "n_rep": repetitions,
            "EC": consistencies.mean(),
            "EC_vec_sd":  consistencies.std(ddof=1) if method == "intersection_union_all" else consistencies.mean(axis=0).std(ddof=1),
            "EC_scalar_sd": "NA" if method == "intersection_union_all" else consistencies.mean(axis=1).std(ddof=1),
            "MAE": np.mean(rep_mae),
            "MAE_sd": np.std(rep_mae, ddof=1),
            "MSqE": np.mean(rep_mse),
            "MSqE_sd": np.std(rep_mse, ddof=1),
            "R2": np.mean(rep_r2),
            "R2_sd": np.std(rep_r2, ddof=1),
        },
        index=[0],
    )


def analyse_sigma_ECs(
    generate_dataset: Any,
    regressor: Any,
    reg_name: Any,
    method: ECMethod,
    sigmas: Tuple[float, float],
    n_train: Tuple[int, int],
    n_test: Tuple[int, int],
    bounds: Tuple[int, int],
    n_outliers: Tuple[int, int],
    outliers_size: Tuple[int, int],
    heteroscedastic_value: Tuple[float, float],
    k: int,
    n_reps: int,
    param_iters: int = 100,
) -> DataFrame:
    param_grid = {
        "sigma": uniform(sigmas[0], sigmas[1]),
        "train_size": randint(n_train[0], n_train[1]),
        "test_size": randint(n_test[0], n_test[1]),
        "n_outliers":  randint(n_outliers[0], n_outliers[1]),
        "heteroscedastic_value": uniform(heteroscedastic_value[0], heteroscedastic_value[1])
    }
    sampler = list(ParameterSampler(param_grid, n_iter=param_iters))
    rows = []
    for params in tqdm(sampler, total=len(sampler), desc=f"Computing ECs for method '{method}'"):
        params["bounds"] = bounds
        dataset = generate_dataset(**params)
        row = calculate_EC(dataset, regressor=regressor, reg_name=reg_name, method=method, k=k, repetitions=n_reps)
        # now add the sigma info
        row.insert(0, "sigma", params["sigma"])
        rows.append(row)
    df: DataFrame = pd.concat(rows, axis=0, ignore_index=True)
    df.sort_values(by=["sigma"], ascending=True, inplace=True)
    return df


if __name__ == "__main__":
    DATASETS = {
        # "Heteroscedastic": generate_heteroscedastic,
        # "Outliers": generate_outliers,
        # "Norm": generate_normal_data,
        # "Uniform+": generate_uniform_add_data,
        # "Uniform-": generate_uniform_sub_data,
        "Uniform+-": generate_uniform_add_sub_data
    }
    REGRESSORS = {"LinReg": LinearRegression(), 
                    "Knn-1": KNeighborsRegressor(n_neighbors=1),
                    "Knn-5": KNeighborsRegressor(n_neighbors=5)
    }
    EC_METHODS: List[ECMethod] = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
                                "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]
    BOUNDS = (-100, 100)
    SIGMA_RANGE = (0.1, float(max(BOUNDS)))
    N_OUTLIERS = (1, 5)
    OUTLIERS_SIZE = (1e4, 1e5)
    HETEROSCEDASTIC_VALUE = (0, 1)
    K = 5
    N_REPS = 50
    ROOT = Path(__file__).resolve().parent

    
    for dataset_name, generate_dataset in DATASETS.items():
        dfs = []
        for reg_name, regressor in REGRESSORS.items():
            for method in EC_METHODS:
                df = analyse_sigma_ECs(
                    generate_dataset=generate_dataset,
                    regressor=regressor,
                    reg_name=reg_name,
                    method=method,
                    sigmas=SIGMA_RANGE,
                    n_train=(100, 1000),
                    n_test=(100, 200),
                    bounds=BOUNDS,
                    n_outliers = N_OUTLIERS,
                    outliers_size = OUTLIERS_SIZE,
                    heteroscedastic_value = HETEROSCEDASTIC_VALUE,
                    k=K,
                    n_reps=N_REPS,
                    param_iters=100,
                )
                print(df)
                dfs.append(df)
            # df = pd.concat(dfs_method, axis=0, ignore_index=True)
            # print(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        filename = dataset_name + "_error.csv"
        outfile = ROOT / filename
        df.to_csv(outfile)
        print(f"Saved results for {dataset_name} error to {outfile}")
