import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def read_wine():
    red = pd.read_csv('winequality-red.csv')
    white = pd.read_csv('winequality-white.csv')
    # create columns to seperate wine types --  encode
    red['red_wine'] = 1
    white['red_wine'] = 0

    red['wine_type'] = 'red'
    white['wine_type'] = 'white'
    # combine red & white wine dataset
    df = pd.concat([red, white])
    return df

def remove_outliers(df, exclude_column=[], sd=4):
    """
    Remove outliers from a pandas DataFrame using the Z-score method.
    
    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    pandas.DataFrame: The DataFrame with outliers removed.
    """
    num_outliers_total = 0
    for column in df.columns:
        if column == exclude_column:
            continue
        series = df[column]
        z_scores = np.abs(stats.zscore(series))
        num_outliers = len(z_scores[z_scores > sd])
        num_outliers_total += num_outliers
        df = df[(z_scores <= sd) | pd.isnull(df[column])]
        print(f"{num_outliers} outliers removed from {column}.")
    print(f"\nTotal of {num_outliers_total} outliers removed.")
    return df

def split_data(df, stratify_name=None):
    '''
    Takes in two arguments the dataframe name and the ("stratify_name" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[stratify_name])
    train, validate = train_test_split(train, #second split
                                    test_size=.25, 
                                    random_state=123,
                                    stratify=train[stratify_name])
    return train, validate, test

def clean_wine():
    # get datasets 
    df = read_wine()
    
    # remove outliers -- removed outliers outside of 4 standard deviation
    df = remove_outliers(df, 'wine_type')

    # categorize quality into high, med, low 
    df['quality_type'] = df['quality'].replace({3: 'low', 4: 'low', 5: 'medium', 6: 'medium', 7: 'high', 8: 'high', 9: 'high'})

    # fix names for columns
    new_col_name = []
    
    for col in df.columns:
        new_col_name.append(col.lower().replace(' ', '_'))

    df.columns = new_col_name

    # split data 
    train, validate, test = split_data(df, "quality")
    
    return train, validate, test
    return df

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    # distribution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df.head(3)}
          
=====================================================
          
          
Dataframe info: """)
    df.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df.describe().T}
          
=====================================================

    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df)): 
        print(f"""******** {col.upper()} - Value Counts:
{df[col].value_counts()}
    _______________________________________""")                   
        
    for col in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], ax=ax)
        ax.set_title(f'Histogram of {col}')
        plt.show()

def rename_col(df, list_of_columns=[]): 
    '''
    Take df with incorrect names and will return a renamed df using the 'list_of_columns' which will contain a list of appropriate names for the columns  
    '''
    df = df.rename(columns=dict(zip(df.columns, list_of_columns)))
    return df

def split_data_xy(train, validate, test, target):
    '''
    This function take in a dataframe performs a train, validate, test split
    Returns train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test
    and prints out the shape of train, validate, test
    '''
    #Split into X and y
    x_train = train.drop(columns=[target])
    y_train = train[target]

    x_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    x_test = test.drop(columns=[target])
    y_test = test[target]

    # Have function print datasets shape
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
   
    return train, validate, test, x_train, y_train, x_validate, y_validate, x_test, y_test

def mm_scale(x_train, x_validate, x_test):
    """
    Apply MinMax scaling to the input data.

    Args:
        x_train (pd.DataFrame): Training data features.
        x_validate (pd.DataFrame): Validation data features.
        x_test (pd.DataFrame): Test data features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Scaled versions of the input data
            (x_train_scaled, x_validate_scaled, x_test_scaled).
    """
    # remove string column wine_type
    keep_col = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol', 'red_wine']
    x_train, x_validate, x_test = x_train[keep_col], x_validate[keep_col], x_test[keep_col]
    
    
    scaler = MinMaxScaler()
    scaler.fit(x_train)


    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)

    col_name = list(x_train.columns)

    x_train_scaled, x_validate_scaled, x_test_scaled = pd.DataFrame(x_train_scaled), pd.DataFrame(x_validate_scaled), pd.DataFrame(x_test_scaled)
    x_train_scaled, x_validate_scaled, x_test_scaled  = rename_col(x_train_scaled, col_name), rename_col(x_validate_scaled, col_name), rename_col(x_test_scaled, col_name)
    
    return x_train_scaled, x_validate_scaled, x_test_scaled
