#Imports
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# FUNCTIONS:

def calculate_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    
    # Create a new DataFrame with the cluster labels
    df = pd.DataFrame(data)
    df['cluster'] = kmeans.labels_
    
    return df

def minmax_scale_data(X_train, X_validate, X_test):
    # Initialize MinMaxScaler object
    scaler = MinMaxScaler()

    # Fit scaler object to training data
    scaler.fit(X_train)

    # Transform training and validation data
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), columns=X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Return scaled data as DataFrames
    return X_train_scaled, X_validate_scaled, X_test_scaled

from sklearn.cluster import KMeans

def elbow_method(data, max_k):
    wcss = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Plot the WCSS values
    plt.plot(range(1, max_k+1), wcss)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()


def fit_KNN_random_features(X_scaled, y, n_random_features=0):
    """
    Fits a K-Nearest Neighbors model to the input pre-scaled data `X_scaled` and `y`,
    with an additional `n_random_features` randomly selected features from `X_scaled`.
    Returns the best model that achieves a score of 70% or higher on the `y` variable,
    along with the results of the top five best models.
    """
    results = []
    best_models = []
    for n_neighbors in range(1, 21):
        selected_cols = np.random.choice(X_scaled.columns, size=n_random_features, replace=False)
        X_with_random = X_scaled.copy()
        for col in selected_cols:
            X_with_random[col] = np.random.permutation(X_with_random[col].values)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_with_random, y)
        score = model.score(X_with_random, y)
        if 0.68 < score < 0.80:
            results.append((model, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    best_models = [result[0] for result in results[:5]]
    best_scores = [result[1] for result in results[:5]]
    top_models_df = pd.DataFrame({'Model': best_models, 'Score': best_scores})
    
    return best_models[0], top_models_df


def fit_DT_random_features(X_scaled, y, n_random_features=0):
    """
    Fits a decision tree model to the input pre-scaled data `X_scaled` and `y`,
    with an additional `n_random_features` randomly selected features from `X_scaled`.
    Returns the best model that achieves a score of 70% or higher on the `y` variable,
    along with the results of the top five best models.
    """
    results = []
    best_models = []
    for depth in range(1, 21):
        selected_cols = np.random.choice(X_scaled.columns, size=n_random_features, replace=False)
        X_with_random = X_scaled.copy()
        for col in selected_cols:
            X_with_random[col] = np.random.permutation(X_with_random[col].values)
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_with_random, y)
        score = model.score(X_with_random, y)
        if 0.68 < score < 0.80:
            results.append((model, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    best_models = [result[0] for result in results[:5]]
    best_scores = [result[1] for result in results[:5]]
    top_models_df = pd.DataFrame({'Model': best_models, 'Score': best_scores})
    
    return best_models[0], top_models_df


def run_decision_tree(X_train, X_test, y_train, y_test, max_depth):
    # Create a Decision Tree classifier
    dt = DecisionTreeClassifier(max_depth=max_depth)
    
    # Train the classifier on the training data
    dt.fit(X_train, y_train)
    
    # Predict the target variable for the test data
    y_pred = dt.predict(X_test)
    
    # Calculate and return the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def run_knn(X_train, X_test, y_train, y_test, n_neighbors):
    # Create a K-Nearest Neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the classifier on the training data
    knn.fit(X_train, y_train)
    
    # Predict the target variable for the test data
    y_pred = knn.predict(X_test)
    
    # Calculate and return the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

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
    return train, validate, test

def add_cluster_dummy(scaled_data, cluster_data):
    # Create dummy variables for the "cluster" column
    cluster_dummy = pd.get_dummies(cluster_data['cluster'], prefix='cluster', drop_first=True)
    
    # Add the cluster dummy variables to the scaled data
    scaled_data_with_cluster = pd.concat([scaled_data, cluster_dummy], axis=1)
    
    return scaled_data_with_cluster

def create_pie_chart(df, column_name,title):
    """ This function creates a pie chart for our categorical target variable"""
    values = df[column_name].value_counts()
    labels = values.index.tolist()
    sizes = values.tolist()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.show()