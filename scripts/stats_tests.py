from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np


def run_anova(df, value_col='HANDLE_TIME', group_col='AGENT_ID'):
    groups = [df[df[group_col] == agent][value_col].dropna() for agent in df[group_col].unique()]
    return f_oneway(*groups)


def run_tukey_test(df, value_col='HANDLE_TIME', group_col='AGENT_ID'):
    tukey_result = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col], alpha=0.05)
    return tukey_result


def calculate_z_scores(series):
    return (series - series.mean()) / series.std()


def detect_outliers_by_zscore(series, threshold=2):
    z_scores = calculate_z_scores(series)
    return z_scores[abs(z_scores) > threshold]