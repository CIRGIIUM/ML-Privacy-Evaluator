import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# Step 1: Read the csv file 
uploaded_df = pd.read_csv("pima_diabetesemail.csv")

# Step 2: Identify the dependent variable (y) column based on the presence of "DV" in the column name
y_column = [col for col in uploaded_df.columns if re.search(r'DV', col, re.IGNORECASE)]
if len(y_column) != 1:
    raise ValueError("Unable to identify the dependent variable column.")
y_column = y_column[0]

# Print the target column (dependent variable)
print("Target Column (DV):", y_column)

# Step 3: Define features (X) and target (y)
X = uploaded_df.drop(columns=[y_column])
y = uploaded_df[y_column]

# Step 4: Filter columns containing the string "ID" (and its variations) and remove them from X
columns_to_remove = [col for col in X.columns if re.search(r'ID', col, re.IGNORECASE)]
X = X.drop(columns=columns_to_remove)


# Preprocessing pipeline for numerical and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Print the numerical and categorical features
print("Numerical Features:", numeric_features.tolist())
print("Categorical Features:", categorical_features.tolist())

# Preprocess numerical features using StandardScaler
numeric_transformer = StandardScaler()
X[numeric_features] = numeric_transformer.fit_transform(X[numeric_features])

# Preprocess categorical features using OneHotEncoder
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)


# Step 5: Create RandomForestRegressor model object
rf_regressor = RandomForestRegressor(random_state=42)

# Step 6: Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Step 7: Train the RandomForestRegressor Model
rf_regressor.fit(X_train, y_train)

# Step 8: Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Step 9: Calculate and print evaluation metrics
r_squared = round(r2_score(y_test, y_pred), 4)
adj_r_squared = round(1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X_train.shape[1] - 1), 4)
mean_abs_error = round(mean_absolute_error(y_test, y_pred), 4)
print("R Squared =", r_squared)
print("Adjusted R Squared =", adj_r_squared)
print("Mean Absolute Error =", mean_abs_error)



