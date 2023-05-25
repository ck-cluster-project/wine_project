import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, stats

def ind_var_test(samp1, samp2, alpha=0.05):
    '''
    Completes an sample t-test, based on the null hypothesis less than
    '''
    t, p = stats.ttest_ind(samp1, samp2)

    if p/2 < alpha and t > 0 :
        print(f'''Reject the null hypothesis: Sufficient''')
    else:
        print(f''' Fail to reject the null: Insufficient evidence''')
    print(f" p-value: {p} , t: {t}")

def alcohol_barplot(df):
    '''
    This function creates a custom bar chart for comparing low/high quality wine's alcohol content
    '''
    fig, ax =plt.subplots()
    
    # set color palette
    sns.set_palette("pastel")
    
    # create average line    
    plt.title("High Quality Wine has More Alcohol")
    sns.barplot(x="quality_type", y="alcohol", data=df)
    plt.xlabel("Quality")
    plt.ylabel("Alcohol Content")
    tick_label = ["Low", "High"]
    ax.set_xticklabels(tick_label)
    alcohol_average = df.alcohol.mean()
    plt.axhline(alcohol_average, label="Alcohol Content Average", color='DarkSlateBlue')
    plt.legend(loc='lower right')
    plt.show()

def sugar_barplot(df):
    '''
    This function creates a custom bar chart for comparing the residual sugar in low/high quality wine.
    '''
    fig, ax =plt.subplots()
    
    # set color palette
    sns.set_palette("pastel")
    
    # create average line    
    plt.title("Low Quality Wine has More Sugar")
    sns.barplot(x="quality_type", y="residual_sugar", data=df)
    plt.xlabel("Quality")
    plt.ylabel("Residual Sugar Amount")
    tick_label = ["Low", "High"]
    ax.set_xticklabels(tick_label)
    sugar_average = df.residual_sugar.mean()
    plt.axhline(sugar_average, label="Residual Sugar Average", color='DarkSlateBlue')
    plt.legend(loc='upper right')
    plt.show()

def chlorides_barplot(df):
    '''
    This function creates a custom bar chart for comparing low/high quality wine's chlorides
    '''
    fig, ax =plt.subplots()
    
    # set color palette
    sns.set_palette("pastel")
    
    # create average line    
    plt.title("Low Quality Wine has More Chlorides")
    sns.barplot(x="quality_type", y="chlorides", data=df)
    plt.xlabel("Quality")
    plt.ylabel("Chlorides")
    tick_label = ["Low", "High"]
    ax.set_xticklabels(tick_label)
    chlorides_average = df.chlorides.mean()
    plt.axhline(chlorides_average, label="Chlorides Average", color='DarkSlateBlue')
    plt.legend(loc='upper right')
    plt.show()

def tsd_barplot(df):
    '''
    This function creates a custom bar chart for comparing low/high quality wine's total sulfur dioxide content
    '''
    fig, ax =plt.subplots()
    
    # set color palette
    sns.set_palette("pastel")
    
    # create average line    
    plt.title("Higher Quality Wine has Less Total Sulfur Dioxide")
    sns.barplot(x="quality_type", y="total_sulfur_dioxide", data=df)
    plt.xlabel("Quality")
    plt.ylabel("Total Sulfur Dioxide")
    total_sulfur_dioxide = df.total_sulfur_dioxide.mean()
    plt.axhline(total_sulfur_dioxide, label="Total Sulfur Dioxide Average", color='DarkSlateBlue')
    plt.legend(loc='upper right')
    plt.show()

def volatile_barplot(df):
    '''
    This function creates a custom bar chart for comparing homes with pools and homes without pools
    '''
    fig, ax =plt.subplots()
    # creat average line
  
    
    plt.title("Low Quality Wine has More volatile_acidity")
    sns.barplot(x="quality_type", y="volatile_acidity", data=df)
    plt.xlabel("Quality")
    plt.ylabel("Amount of Volatile Acidity")
    tick_label = ["Low", "High"]
    ax.set_xticklabels(tick_label)
    v_avg = df.volatile_acidity.mean()
    plt.axhline(v_avg, label="Volatile Acidity Average", color='DarkSlateBlue')
    plt.legend(loc='upper right')
    plt.show()

def ph_barplot(df):
    '''
    This function creates a custom bar chart for comparing homes with pools and homes without pools
    '''
    fig, ax =plt.subplots()
    # creat average line
  
    
    plt.title("Low Quality Wine has More pH")
    sns.barplot(x="quality_type", y="ph", data=df)
    plt.xlabel("Quality")
    plt.ylabel("Amount of pH")
    tick_label = ["Low", "High"]
    ax.set_xticklabels(tick_label)
    ph_avg = df.ph.mean()
    plt.axhline(ph_avg, label="pH Average", color='DarkSlateBlue')
    plt.legend(loc='lower right')
    plt.show()