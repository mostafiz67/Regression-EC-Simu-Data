"""
Author: Md Mostafizur Rahman
File: Checking heteroscedasticity of normal distribution dataset and linear regression.
        Also ploating the residuals
"""

from pathlib import Path
from typing import Any, Tuple
import os
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, ParameterSampler
from tqdm import tqdm
from typing_extensions import Literal
import matplotlib.pyplot as plt

np.random.seed(42)

Condition = Literal["normal-data", "uniform-add-data", "uniform-sub-data"]


def generate_normal_data(
    train_size: int, test_size: int, sigma: float, bounds: Tuple[int, int] = (-100, 100)
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
    train_size: int, test_size: int, sigma: float, bounds: Tuple[int, int] = (-100, 100)
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
    train_size: int, test_size: int, sigma: float, bounds: Tuple[int, int] = (-100, 100)
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
def generate_heteroscedastic(
    train_size: int, test_size: int, sigma: float, bounds: Tuple[int, int] = (-100, 100)
) -> Tuple[ndarray, ndarray, ndarray]:
    X_train = np.linspace(bounds[0], bounds[1], num=train_size)
    error = X_train * np.random.normal(0, sigma, size=train_size)
    y_train = error + X_train

    X_test = np.random.uniform(bounds[0], bounds[1], size=test_size)
    y_test = X_test + np.random.normal(0, sigma, size=test_size)

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
    bounds: Tuple[int, int] = (-100, 100),
    n_outliers: int = 1,
    outlier_size: Tuple[float, float] = (1e4, 1e5),
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    X_train = np.random.uniform(bounds[0], bounds[1], train_size)
    e_train = np.random.normal(0, sigma, size=train_size)
    outliers = np.random.uniform(outlier_size[0], outlier_size[1], n_outliers)
    y_train = X_train + e_train
    y_train[-n_outliers:] = outliers  # change last three points to be outliers

    X_test = np.random.uniform(bounds[0], bounds[1], test_size)
    e_test = np.random.normal(0, sigma, size=test_size)
    y_test = X_test + e_test
    return (
        (X_train).reshape(-1, 1),
        (X_test).reshape(-1, 1),
        (y_train).reshape(-1, 1),
        (y_test).reshape(-1, 1),
    )



def analyse_heteroscedasticity(
    generate_dataset: Any,
    dataset_name: Any,
    regressor: Any,
    reg_name: Any,
    sigmas: Tuple[float, float],
    n_train: Tuple[int, int],
    n_test: Tuple[int, int],
    bounds: Tuple[int, int],
    k: int,
    n_reps: int,
    param_iters: int = 10,
) -> Any:
    param_grid = {
        "sigma": uniform(sigmas[0], sigmas[1]),
        "train_size": randint(n_train[0], n_train[1]),
        "test_size": randint(n_test[0], n_test[1]),
    }
    sampler = list(ParameterSampler(param_grid, n_iter=param_iters))
    for params in tqdm(sampler, total=len(sampler)):
        params["bounds"] = bounds
        dataset = generate_dataset(**params)
    
        X_train, X_test, y_train, y_test = dataset
        rep_residuals, rep_y_test = [], []
        for i in range(n_reps):
            fold_residuals, fold_y_test = [], []
            kf = KFold(n_splits=k, shuffle=True)
            for train_index, _ in kf.split(X_train):
                preds = regressor.fit(X_train[train_index], y_train[train_index]).predict(X_test)
                resid = preds - y_test
                fold_residuals.extend(resid)
                fold_y_test.extend(y_test)
            rep_residuals.extend(fold_residuals)
            rep_y_test.extend(fold_y_test)
            # print(np.shape(rep_y_test), np.shape(rep_residuals))
            
        plt.plot(rep_y_test,rep_residuals, 'o', color='darkblue')
        plt.title("Residual Plot")
        plt.xlabel("Independent Variable")
        plt.ylabel("Residual")
        
        fig_name = "figure_'{0}'_params_'{1}'.png".format(dataset_name, params)
        outfile = os.path.join(ROOT, 'heteroscedasticity_test_figure/') + fig_name
        plt.savefig(outfile, format='png')
        plt.close()


if __name__ == "__main__":
    DATASETS = {
        "Heteroscedastic": generate_heteroscedastic,
        "Outliers": generate_outliers,
        "Norm": generate_normal_data,
        "Uniform+": generate_uniform_add_data,
        "Uniform-": generate_uniform_sub_data,
    }
    REGRESSORS = {"LinReg": LinearRegression()
    }
    BOUNDS = (-100, 100)
    SIGMA_RANGE = (0.1, float(max(BOUNDS)))
    K = 5
    N_REPS = 10
    ROOT = Path(__file__).resolve().parent

    dfs = []
    for dataset_name, generate_dataset in DATASETS.items():
        for reg_name, regressor in REGRESSORS.items():
            df = analyse_heteroscedasticity(
                generate_dataset=generate_dataset,
                dataset_name=dataset_name,
                regressor=regressor,
                reg_name=reg_name,
                sigmas=SIGMA_RANGE,
                n_train=(100, 1000),
                n_test=(100, 200),
                bounds=BOUNDS,
                k=K,
                n_reps=N_REPS,
                param_iters=10,
            )
