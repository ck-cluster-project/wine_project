import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def high_citric_col(train, validate, test):
    '''
    Adds a binary column 'high_citric' to the given train, validate, and test DataFrames based on the condition:
    - For red wine samples in the 'train' DataFrame, 'high_citric' is set to 1 if the 'citric_acid' value is greater than the mean 'citric_acid' of red wine samples; otherwise, it is set to 0.
    - The same condition is applied to the 'validate' and 'test' DataFrames as well.

    Parameters:
    train (DataFrame): The training dataset containing red and white wine samples.
    validate (DataFrame): The validation dataset containing red and white wine samples.
    test (DataFrame): The testing dataset containing red and white wine samples.

    Returns:
    None. Modifies the 'train', 'validate', and 'test' DataFrames in place by adding the 'high_citric' column.
    '''
    red_wine_mean_citric_acid = train[train.red_wine == 1].citric_acid.mean()

    # Add to train
    train['high_citric'] = (train.red_wine == 1) & (train.citric_acid > red_wine_mean_citric_acid)
    train['high_citric'] = train['high_citric'].astype(int)

    # Add to validate
    validate['high_citric'] = (validate.red_wine == 1) & (validate.citric_acid > red_wine_mean_citric_acid)
    validate['high_citric'] = validate['high_citric'].astype(int)

    # Add to test
    test['high_citric'] = (test.red_wine == 1) & (test.citric_acid > red_wine_mean_citric_acid)
    test['high_citric'] = test['high_citric'].astype(int)
