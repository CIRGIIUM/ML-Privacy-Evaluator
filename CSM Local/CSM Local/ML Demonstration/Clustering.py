import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from diffprivlib.models import KMeans as DP_KMeans
import warnings

# Step 1: Read the csv file (Mall_Customers.csv in this case)
uploaded_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Step 2: Identify the target column (y) based on the presence of "DV" in the column name
y_column = [col for col in uploaded_df.columns if re.search(r'DV', col, re.IGNORECASE)]
if len(y_column) != 1:
    raise ValueError("Unable to identify the target column.")
y_column = y_column[0]

# Print the target column
print("Target Column:", y_column)

# Step 3: Remove columns containing the string "ID" (and its variations)
columns_to_remove = [col for col in uploaded_df.columns if re.search(r'ID', col, re.IGNORECASE)]
X = uploaded_df.drop(columns=columns_to_remove)

# Step 4: Extract numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline for numerical and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

# Apply preprocessing to numerical features
X_scaled_numerical = numeric_transformer.fit_transform(X[numerical_features])

# Apply preprocessing to categorical features
X_categorical_encoded = categorical_transformer.fit_transform(X[categorical_features])
categorical_feature_names = categorical_transformer.get_feature_names_out(categorical_features)
X_scaled_categorical = pd.DataFrame(X_categorical_encoded.toarray(), columns=categorical_feature_names)

# Concatenate numerical and categorical features
X_scaled = pd.concat([pd.DataFrame(X_scaled_numerical), X_scaled_categorical], axis=1).values

# Step 5: To figure out K for KMeans, use the Elbow Method on KMEANS++ Calculation
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Step 6: Finding the optimal number of clusters based on the Elbow Method
second_diff = [wcss[i] - 2 * wcss[i - 1] + wcss[i - 2] for i in range(2, len(wcss))]
optimal_clusters = second_diff.index(min(second_diff)) + 1
optimal_inertia = wcss[optimal_clusters - 1]  # Access the optimal inertia value from the list

print("Optimal number of clusters:", optimal_clusters)

# Step 7: Calculate Silhouette Score for the optimal number of clusters (regular K-means)
kmeans_optimal = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans_optimal.fit(X_scaled)
silhouette_score_optimal = silhouette_score(X_scaled, kmeans_optimal.labels_)
print("Silhouette Score (regular K-means):", round(silhouette_score_optimal, 4))

# Step 8: Calculate Adjusted Rand Index for the optimal number of clusters (regular K-means)
true_labels = uploaded_df[y_column].values
ari_optimal = adjusted_rand_score(true_labels, kmeans_optimal.labels_)
print("Adjusted Rand Index (regular K-means):", round(ari_optimal, 4))

# Step 9: Apply differential privacy to K-means clustering
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dp_kmeans_optimal = DP_KMeans(n_clusters=optimal_clusters, epsilon=1.0, random_state=42)
    dp_kmeans_optimal.fit(X_scaled)
dp_labels_optimal = dp_kmeans_optimal.predict(X_scaled)

# Step 10: Calculate Silhouette Score for the optimal number of clusters (differential private K-means)
silhouette_score_dp_optimal = silhouette_score(X_scaled, dp_labels_optimal)
print("Silhouette Score (DP K-means):", round(silhouette_score_dp_optimal, 4))

# Step 11: Calculate Adjusted Rand Index for the optimal number of clusters (differential private K-means)
ari_dp_optimal = adjusted_rand_score(true_labels, dp_labels_optimal)
print("Adjusted Rand Index (DP K-means):", round(ari_dp_optimal, 4))
