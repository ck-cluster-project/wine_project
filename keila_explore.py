import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, stats

def ind_test(samp1, samp2, alpha=0.05):
    '''
    Completes an sample t-test, based on the null hypothesis less than
    '''
    t, p = stats.ttest_ind(samp1, samp2, equal_var=False)

    if p/2 < alpha and t > 0 :
        print(f'''Reject the null hypothesis: Sufficient''')
    else:
        print(f''' Fail to reject the null: Insufficient evidence''')
    print(f" p-value: {p} , t: {t}")


def sugar_barplot(df):
    '''
    This function creates a custom bar chart for comparing homes with pools and homes without pools
    '''
    fig, ax =plt.subplots()
    
    # set color palette
    sns.set_palette("pastel")
    
    # create average line    
    plt.title("Low Quality Wine has More Sugar")
    sns.barplot(x="quality_type", y="residual_sugar", data=df, hue='wine_type')
    plt.xlabel("Quality")
    plt.ylabel("Residual Sugar Amount")
    tick_label = ["Low", "High"]
    ax.set_xticklabels(tick_label)
    property_value_average = df.residual_sugar.mean()
    plt.axhline(property_value_average, label="Residual Sugar Average", color='DarkSlateBlue')
    plt.legend(loc='upper right')
    plt.show()