# # Import Suite
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import LabelEncoder

# # Load the data for creating the model
# df = pd.read_csv('MMData/MarchMadnessData.csv')

# # Display the first few rows of the dataset to verify
# print("Dataset Preview:")
# print(df.head())

# # Step 1: Data Preprocessing
# # Drop columns not needed for the model (e.g., team names, season, or any irrelevant info)
# columns_to_drop = ['Mapped ESPN Team Name', 'Season', 'Seed', 'Post-Season Tournament', 'Post-Season Tournament Sorting Index', 'Vulnerable Top 2 Seed?']
# df = df.drop(columns=columns_to_drop, axis=1)

# # Encode the target variable (e.g., 'Tournament Winner?') if it's categorical
# target_columns = 'Tournament Winner?'  # Replace with the exact column name for the target
# label_encoder = LabelEncoder()
# df[target_columns] = label_encoder.fit_transform(df[target_columns])

# # Separate features (X) and the target variable (y)
# X = df.drop(columns=[target_columns])
# y = df[target_columns]

# # Check for missing values
# print("Missing Values in Dataset:")
# print(df.isnull().sum().sort_values(ascending=False).head(10))  # Display columns with most missing values

# # Handle missing values (if any)
# X = X.fillna(X.mean())  # Filling missing values with the column mean

# # Step 2: Split the Data into Training and Testing Sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 3: Train the Random Forest Classifier
# print("Training Random Forest Classifier...")
# random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest.fit(X_train, y_train)

# # Step 4: Evaluate the Model
# y_pred = random_forest.predict(X_test)

# # Print model performance metrics
# print("Model Evaluation:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Step 5: Save the Trained Model for Future Use
# model_file = 'march_madness_rf_model.pkl'
# joblib.dump(random_forest, model_file)
# print(f"Trained model saved to {model_file}")

# # Step 6: Load and Use the Model on 2025 Data
# data_2025_file = '2025_data.csv'  # Replace with the correct path to your 2025 CSV
# df_2025 = pd.read_csv(data_2025_file)

# # Preprocess the 2025 data (ensure it matches the training data structure)
# df_2025 = df_2025.drop(columns=columns_to_drop, axis=1)
# df_2025 = df_2025.fillna(df_2025.mean())  # Handle missing values

# # Load the saved model
# loaded_model = joblib.load(model_file)

# # Make predictions on the 2025 data
# predictions_2025 = loaded_model.predict(df_2025)

# # Add predictions to the 2025 dataset and save the output
# df_2025['Predicted Tournament Winner'] = label_encoder.inverse_transform(predictions_2025)
# df_2025.to_csv('2025_predictions.csv', index=False)
# print("Predictions for 2025 data saved to '2025_predictions.csv'")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the trained model

# Load the data for creating the model
data_file = 'MMData/MarchMadnessData.csv'  # Replace with the correct path to your CSV
df = pd.read_csv(data_file)

# Display the first few rows of the dataset to verify
print("Dataset Preview:")
print(df.head())

# Step 1: Data Preprocessing
# Drop columns not needed for the model (e.g., team names, season, or any irrelevant info)
columns_to_drop = ['Mapped ESPN Team Name', 'Season', 'Seed', 'Post-Season Tournament', 
                   'Post-Season Tournament Sorting Index']
df = df.drop(columns=columns_to_drop, axis=1)

# Encode "Yes"/"No" columns into numerical values (1 for "Yes", 0 for "No")
yes_no_columns = ['Vulnerable Top 2 Seed?', 'Tournament Winner?', 
                  'Tournament Championship?', 'Final Four?']
for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Separate features (X) and the target variable (y)
target_column = 'Tournament Winner?'  # Replace with the exact column name for the target
X = df.drop(columns=[target_column])
y = df[target_column]

# Check for missing values
print("Missing Values in Dataset:")
print(df.isnull().sum().sort_values(ascending=False).head(10))  # Display columns with most missing values

# Handle missing values (if any)
X = X.fillna(X.mean())  # Filling missing values with the column mean

# Step 2: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Random Forest Classifier
print("Training Random Forest Classifier...")
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = random_forest.predict(X_test)

# Print model performance metrics
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 5: Save the Trained Model for Future Use
model_file = 'march_madness_rf_model.pkl'
joblib.dump(random_forest, model_file)
print(f"Trained model saved to {model_file}")

# Step 6: Load and Use the Model on 2025 Data
data_2025_file = 'MMData/2025Teams.csv'  # Replace with the correct path to your 2025 CSV
df_2025 = pd.read_csv(data_2025_file)

# Preprocess the 2025 data (ensure it matches the training data structure)
df_2025 = df_2025.drop(columns=columns_to_drop, axis=1)


df_2025 = df_2025.fillna(df_2025.mean())  # Handle missing values
X_2025 = df_2025[X.columns]

# Load the saved model
loaded_model = joblib.load(model_file)

# Make predictions on the 2025 data
predictions_2025 = loaded_model.predict(X_2025)

# Add predictions to the 2025 dataset and save the output
df_2025['Predicted Tournament Winner'] = predictions_2025
df_2025.to_csv('2025_predictions.csv', index=False)
print("Predictions for 2025 data saved to '2025_predictions.csv'")