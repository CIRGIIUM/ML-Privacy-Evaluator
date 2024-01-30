import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def targetCol(file_path):
    uploaded_df = pd.read_csv(file_path)
    y_column = [col for col in uploaded_df.columns if re.search(r'DV', col, re.IGNORECASE)]
    print("CSV File: ", file_path)
    print("Target Column is ", y_column)
    print("\n")
    if len(y_column) != 1:
        raise ValueError("Unable to identify the target column.")
    y_column = y_column[0]

# Function to clear and delete everything in the UPLOAD_FOLDER
def clear_upload_folder():
    upload_folder_path = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(upload_folder_path):
        file_path = os.path.join(upload_folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
def preprocess_clustering_data(file_path):
    uploaded_df = pd.read_csv(file_path)

    y_column = [col for col in uploaded_df.columns if re.search(r'DV', col, re.IGNORECASE)]
    if len(y_column) != 1:
        raise ValueError("Unable to identify the target column.")
    y_column = y_column[0]

    X = uploaded_df.drop(columns=[y_column])

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    X_scaled_numerical = numeric_transformer.fit_transform(X[numeric_features])
    X_categorical_encoded = categorical_transformer.fit_transform(X[categorical_features])
    categorical_feature_names = categorical_transformer.get_feature_names_out(categorical_features)
    X_scaled_categorical = pd.DataFrame(X_categorical_encoded.toarray(), columns=categorical_feature_names)

    X_scaled = pd.concat([pd.DataFrame(X_scaled_numerical), X_scaled_categorical], axis=1).values

    return X_scaled, uploaded_df[y_column]

def preprocess_regression_classification(file_path):
    uploaded_df = pd.read_csv(file_path)

    y_column = [col for col in uploaded_df.columns if re.search(r'DV', col, re.IGNORECASE)]
    if len(y_column) != 1:
        raise ValueError("Unable to identify the dependent variable column.")
    y_column = y_column[0]

    X = uploaded_df.copy()  # Create a copy of the original DataFrame
    
    columns_to_remove = [col for col in X.columns if re.search(r'ID', col, re.IGNORECASE)]
    X = X.drop(columns=columns_to_remove)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    ordinal_encoder = OrdinalEncoder()
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    X[numeric_features] = numeric_transformer.fit_transform(X[numeric_features])
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Encode the y column using OrdinalEncoder
    y_encoded = ordinal_encoder.fit_transform(uploaded_df[y_column].values.reshape(-1, 1))

    return X, y_encoded

def preprocess_clustering(file_path):
    uploaded_df = pd.read_csv(file_path)

    y_column = [col for col in uploaded_df.columns if re.search(r'DV', col, re.IGNORECASE)]
    if len(y_column) != 1:
        raise ValueError("Unable to identify the dependent variable column.")
    y_column = y_column[0]

    X = uploaded_df.copy()  # Create a copy of the original DataFrame
    
    columns_to_remove = [col for col in X.columns if re.search(r'ID', col, re.IGNORECASE)]
    X = X.drop(columns=columns_to_remove)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    X[numeric_features] = numeric_transformer.fit_transform(X[numeric_features])
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    return X, uploaded_df[y_column]

def train_classification_model(X, y):

    y_encoded = y
    # # Check if y is not of type int64 and contains numerical values
    # if not y.select_dtypes(include='int64').equals(y):
    #     y_encoded = pd.get_dummies(y)
    # else:
    #     y_encoded = y  # If y is already int64 or does not contain numerical values, leave it as is

    X_train, X_test, Y_train, Y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    # Assuming Y_test and Y_pred are DataFrames after one-hot encoding
    accuracy = accuracy_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred, average='weighted', zero_division=1)
    precision = precision_score(Y_test, Y_pred, average='weighted', zero_division=1)
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    return accuracy, recall, precision, f1

def train_regression_model(X, y):

    y_encoded = y
    # # Check if y is not of type int64 and contains numerical values
    # if not y.select_dtypes(include='int64').equals(y):
    #     y_encoded = pd.get_dummies(y)
    # else:
    #     y_encoded = y.astype('int')  # If y is already int64 or does not contain numerical values, convert it to int

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.7, random_state=42)
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)

    r_squared = round(r2_score(y_test, y_pred), 4)
    adj_r_squared = round(1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X_train.shape[1] - 1), 4)
    mean_abs_error = round(mean_absolute_error(y_test, y_pred), 4)

    return r_squared, adj_r_squared, mean_abs_error




# def train_regression_model(X, y):
#     y=y.astype('int')
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
#     rf_regressor = RandomForestRegressor(random_state=42)
#     rf_regressor.fit(X_train, y_train)
#     y_pred = rf_regressor.predict(X_test)

#     r_squared = round(r2_score(y_test, y_pred), 4)
#     adj_r_squared = round(1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X_train.shape[1] - 1), 4)
#     mean_abs_error = round(mean_absolute_error(y_test, y_pred), 4)

#     return r_squared, adj_r_squared, mean_abs_error

# def train_classification_model(X, y):
#     y = process_generic_string(y)
#     y=y.astype('int')
#     X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     clf = RandomForestClassifier(random_state=42)
#     clf.fit(X_train, Y_train)
#     Y_pred = clf.predict(X_test)

#     accuracy = accuracy_score(Y_test, Y_pred)
#     recall = recall_score(Y_test, Y_pred, average='weighted', zero_division=1)
#     precision = precision_score(Y_test, Y_pred, average='weighted', zero_division=1)
#     f1 = f1_score(Y_test, Y_pred, average='weighted')

#     return accuracy, recall, precision, f1

def evaluate_clustering_model(X_scaled, true_labels):
    wcss = []
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    second_diff = [wcss[i] - 2 * wcss[i - 1] + wcss[i - 2] for i in range(2, len(wcss))]
    optimal_clusters = second_diff.index(min(second_diff)) + 1

    kmeans_optimal = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
    kmeans_optimal.fit(X_scaled)
    silhouette_score_optimal = silhouette_score(X_scaled, kmeans_optimal.labels_)
    ari_optimal = adjusted_rand_score(true_labels, kmeans_optimal.labels_)

    return optimal_clusters, silhouette_score_optimal, ari_optimal

def fetch_results_from_datasets(datasets):
    results = []
    for dataset in datasets:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset)

        X, y = preprocess_regression_classification(file_path)
        accuracy, recall, precision, f1 = train_classification_model(X, y)
        r_squared, adj_r_squared, mean_abs_error = train_regression_model(X, y)

        X, y = preprocess_clustering_data(file_path)
        optimal_clusters, silhouette_score_optimal, ari_optimal = evaluate_clustering_model(X, y)

        result = {
            'filename': dataset,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'mean_abs_error': mean_abs_error,
            'optimal_clusters': optimal_clusters,
            'silhouette_score': silhouette_score_optimal,
            'ari': ari_optimal
        }
        results.append(result)

    return results


def censor_data(column_data):
    data_len = len(column_data)
    first_3_letters = column_data[:2]
    censored_data = first_3_letters + '*' * (data_len - 3)
    return censored_data