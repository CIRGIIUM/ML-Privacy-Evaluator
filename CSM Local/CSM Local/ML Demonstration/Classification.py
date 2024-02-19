import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Step 1: Upload the dataset into a pandas DataFrame
data = pd.read_csv("pima_diabetesemail.csv")
df = pd.DataFrame(data)

# Step 2: Identify the dependent variable (y) column based on the presence of "DV" in the column name
y_column = [col for col in df.columns if re.search(r'DV', col, re.IGNORECASE)]
if len(y_column) != 1:
    raise ValueError("Unable to identify the dependent variable column.")
y_column = y_column[0]

# Print the target column (dependent variable)
print("Target Column (DV):", y_column)

# Step 3: Split the data into features (X) and target (Y)
X = df.drop(columns=[y_column])
Y = df[y_column]

# Step 3.1: Filter columns containing the string "ID" (and its variations) and remove them from X
columns_to_remove = [col for col in X.columns if re.search(r'ID', col, re.IGNORECASE)]
X = X.drop(columns=columns_to_remove)

# Step 4: Automatically find categorical features and create dummy variables
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_features)

# Step 5: Automatically find numerical features and standard scale them
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

#Step 6: Print the categorical and numerical features
print("Categorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

# Step 7: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 8: Create the RandomForestClassifier model and fit it to the training data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, Y_train)

# Step 9: Make predictions on the test data
Y_pred = clf.predict(X_test)

# Step 10: Evaluate the model using various metrics
accuracy = accuracy_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred, average='weighted', zero_division=1)
precision = precision_score(Y_test, Y_pred, average='weighted', zero_division=1)
f1 = f1_score(Y_test, Y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
