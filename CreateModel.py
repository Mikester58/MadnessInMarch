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
df = df.drop(columns=['Mapped ESPN Team Name'])

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

def whoWon(one, two, rfModel):
    ballout = np.concatenate([one, two]).reshape(1,-1)
    prob = rfModel.predict_proba(ballout)

    return one if prob[0][1] > prob[0][0] else two
    
def gameTime(one, two, Model, roundNum):
    onet = one
    twot = two
    winner = whoWon(onet.drop(columns='Mapped ESPN Team Name'), two, Model)
    if(winner == onet):
        print(f"{one['Mapped ESPN Team Name']} has won in round {roundNum}")
    if(winner == twot):
        print(f"{two['Mapped ESPN Team Name']} has won in round {roundNum}")
#Let the tourney start
#South
gameTime(df_2025.loc[df_2025['Mapped ESPN Team Name']=='Kansas'].index, df_2025.loc[df_2025['Mapped ESPN Team Name']=='Alabama'].index, modelUsed, 8)