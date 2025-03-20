#Full credit to Jonathan Pilafas on Kaggle for creating the csv utilized in this code
#csv file can be found at https://www.kaggle.com/datasets/jonathanpilafas/2024-march-madness-statistical-analysis

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data for creating the model
df = pd.read_csv('MMData/Data.csv')

# Display the first few rows of the dataset to verify
print("Dataset Preview:")
print(df.head())

# Step 1: Data Preprocessing

# Dont need team names for training data, drop then
df = df.drop(columns='Mapped ESPN Team Name')

# Separate features (X) and the target variable (y)
X = df.drop(columns=['Tournament Winner?', 'Tournament Championship?', 'Final Four?'])
y = df[['Tournament Winner?', 'Tournament Championship?', 'Final Four?']]

# Step 2: Split the Data into Training and Testing Sets
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.30, random_state=42)
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.50, random_state=42)

#weigh the champion & runner up higher
y_train['Tournament Winner?'] *= 3
y_train['Tournament Championship?'] *= 2

# Step 3: Train the Random Forest Classifier
print("Training Random Forest Classifier...")
rfc = RandomForestClassifier(n_estimators=100, max_depth=3)
modelUsed = MultiOutputClassifier(rfc)
modelUsed = modelUsed.fit(X_train, y_train)

y_pred = modelUsed.predict(X_test)

# Evaluate accuracy for each target variable
winner_accuracy = accuracy_score(y_test['Tournament Winner?'], y_pred[:, 0])
runner_up_accuracy = accuracy_score(y_test['Tournament Championship?'], y_pred[:, 1])

print(f"Winner Accuracy: {winner_accuracy}")
print(f"Runner-Up Accuracy: {runner_up_accuracy}")

# Print model performance metrics
print("Classification Report:")
print(classification_report(y_test['Tournament Winner?'], y_pred[:, 0], zero_division=0))

# Step 6: Load and Use the Model on 2025 Data
df_2025 = pd.read_csv('MMData/2025Teams.csv')
X_2025 = df_2025[X.columns]

def whoWon(on, tw, rfModel):
    numeric_on = on[X.columns]
    numeric_tw = tw[X.columns]
    
    # Get predictions for both teams
    prob_on = rfModel.estimators_[0].predict_proba(numeric_on.values.reshape(1,-1))
    prob_tw = rfModel.estimators_[0].predict_proba(numeric_tw.values.reshape(1,-1))
    
    # Compare probabilities
    return 'team1' if prob_on[0][1] > prob_tw[0][1] else 'team2'

def gameTime(on, tw, Model, roundNum):
    # Don't modify the dataframes here
    winner = whoWon(on, tw, Model)
    
    if winner == 'team1':
        print(f"{on['Mapped ESPN Team Name'].iloc[0]} has won in round {roundNum}")
    else:
        print(f"{tw['Mapped ESPN Team Name'].iloc[0]} has won in round {roundNum}")
#Let the tourney start
# Champs
gameTime(df_2025.loc[df_2025['Mapped ESPN Team Name']=='Alabama'], df_2025.loc[df_2025['Mapped ESPN Team Name']=='Kansas'], modelUsed, 8)