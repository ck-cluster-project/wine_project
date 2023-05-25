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
    """
    Perform K-means clustering on the given data.

    Parameters:
    - data (array-like): The input data to be clustered.
    - n_clusters (int): The number of clusters to create.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the original data with an additional 'cluster' column
                             indicating the cluster labels assigned by K-means.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    
    # Create a new DataFrame with the cluster labels
    df = pd.DataFrame(data)
    df['cluster'] = kmeans.labels_
    
    return df

def minmax_scale_data(X_train, X_validate, X_test):
    """
    Apply min-max scaling to the input data.

    Parameters:
        X_train (DataFrame): The training data.
        X_validate (DataFrame): The validation data.
        X_test (DataFrame): The test data.

    Returns:
        tuple: A tuple containing the scaled data as DataFrames.
            - X_train_scaled (DataFrame): Scaled training data.
            - X_validate_scaled (DataFrame): Scaled validation data.
            - X_test_scaled (DataFrame): Scaled test data.
    """
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
    """
    Applies the elbow method to determine the optimal number of clusters (k) for KMeans clustering.

    Args:
        data (array-like): The input data to be clustered.
        max_k (int): The maximum number of clusters to consider.

    Returns:
        None

    The function calculates the within-cluster sum of squares (WCSS) for different values of k and
    plots the WCSS values against the number of clusters. The 'elbow' point in the plot is often
    considered as the optimal value of k, indicating the number of clusters where adding more clusters
    does not significantly decrease the WCSS.

    Example usage:
        data = [[1, 2], [3, 4], [5, 6], ...]  # Input data
        max_k = 10  # Maximum number of clusters to consider
        elbow_method(data, max_k)
    """
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


def fit_KNN_random_features(X_scaled, y, x_validate_scaled, y_validate, n_random_features=0):
    """
    Fits a K-Nearest Neighbors model to the input pre-scaled data `X_scaled` and `y`,
    with an additional `n_random_features` randomly selected features from `X_scaled`.
    Returns the best model that achieves a score of 44% or higher on the `y` variable,
    along with the results of the top five best models.
    """
    results = []
    best_models = []
    train_scores = []
    validate_scores = []
    for n_neighbors in range(1, 21):
        selected_cols = np.random.choice(X_scaled.columns, size=n_random_features, replace=False)
        X_with_random = X_scaled.copy()
        for col in selected_cols:
            X_with_random[col] = np.random.permutation(X_with_random[col].values)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_with_random, y)
        train_score = model.score(X_with_random, y)
        validate_score = model.score(x_validate_scaled, y_validate)
        if 0.44 < train_score < 0.80 and 0.44 < validate_score < 0.80:
            results.append((model, train_score, validate_score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    best_models = [result[0] for result in results[:5]]
    best_train_scores = [result[1] for result in results[:5]]
    best_validate_scores = [result[2] for result in results[:5]]
    top_models_df = pd.DataFrame({'Model': best_models, 'Train Score': best_train_scores, 'Validate Score': best_validate_scores})
    
    return best_models[0], top_models_df, train_scores, validate_scores



def fit_DT_random_features(X_scaled, y, x_validate_scaled, y_validate, n_random_features=0):
    """
    Fits a decision tree model to the input pre-scaled data `X_scaled` and `y`,
    with an additional `n_random_features` randomly selected features from `X_scaled`.
    Returns the best model that achieves a score of 44% or higher on the `y` variable,
    along with the results of the top five best models and their train and validation scores.
    """
    results = []
    best_models = []
    train_scores = []
    validate_scores = []
    
    for depth in range(1, 21):
        selected_cols = np.random.choice(X_scaled.columns, size=n_random_features, replace=False)
        X_with_random = X_scaled.copy()
        for col in selected_cols:
            X_with_random[col] = np.random.permutation(X_with_random[col].values)
        model = DecisionTreeClassifier(random_state=123,max_depth=depth)
        model.fit(X_with_random, y)
        train_score = model.score(X_with_random, y)
        validate_score = model.score(x_validate_scaled, y_validate)
        if 0.44 < train_score < 0.80:
            results.append((model, train_score, validate_score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    best_models = results[:5]
    top_models_df = pd.DataFrame(best_models, columns=['Model', 'Train Score', 'Validate Score'])
    
    return best_models[0][0], top_models_df




def run_decision_tree(X_train, X_test, y_train, y_test, max_depth):
    """
    Trains a Decision Tree classifier on the provided training data and predicts the target variable
    for the test data.

    Parameters:
        X_train (array-like): Training features, shape (n_samples, n_features).
        X_test (array-like): Test features, shape (n_samples, n_features).
        y_train (array-like): Training target variable, shape (n_samples,).
        y_test (array-like): Test target variable, shape (n_samples,).
        max_depth (int): Maximum depth of the decision tree.

    Returns:
        float: Accuracy score of the decision tree classifier on the test data.
    """
    # Create a Decision Tree classifier
    dt = DecisionTreeClassifier(random_state=123,max_depth=max_depth)
    
    # Train the classifier on the training data
    dt.fit(X_train, y_train)
    
    # Predict the target variable for the test data
    y_pred = dt.predict(X_test)
    
    # Calculate and return the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def add_cluster_col(scaled_df, cluster_cols):
    """
    Adds a cluster column to the scaled dataframe.

    Parameters:
        scaled_df (pandas.DataFrame): The scaled dataframe to which the cluster column will be added.
        cluster_cols (pandas.DataFrame): The dataframe containing the cluster information.

    Returns:
        pandas.DataFrame: The new dataframe with the added cluster column.

    Example:
        scaled_df:
           feature_1  feature_2  feature_3
        0   0.1        0.5        0.7
        1   0.3        0.2        0.9

        cluster_cols:
           cluster
        0  A
        1  B

        add_cluster_col(scaled_df, cluster_cols) returns:
           feature_1  feature_2  feature_3  cluster_B
        0   0.1        0.5        0.7        0
        1   0.3        0.2        0.9        1
    """
    cluster_col = pd.DataFrame(cluster_cols.iloc[:, -1])
    cluster_col = pd.get_dummies(cluster_col['cluster'],prefix='cluster', drop_first=True)
    new_df = pd.concat([scaled_df, cluster_col], axis=1)
    return new_df

def run_knn(X_train, X_test, y_train, y_test, n_neighbors):
    """Runs a K-Nearest Neighbors classifier on the given data and returns the accuracy score.

    Args:
        X_train (array-like): Training data features.
        X_test (array-like): Test data features.
        y_train (array-like): Target variable for the training data.
        y_test (array-like): Target variable for the test data.
        n_neighbors (int): Number of neighbors to consider for classification.

    Returns:
        float: Accuracy score of the classifier on the test data.
    """
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


def create_pie_chart(df, column_name,title):
    """ This function creates a pie chart for our categorical target variable"""
    values = df[column_name].value_counts()
    labels = values.index.tolist()
    sizes = values.tolist()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.show()
    
