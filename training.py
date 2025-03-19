import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
# Load data
batting = pd.read_csv('baseball-reference data/all_batting_updated.csv')
pitching = pd.read_csv('baseball-reference data/all_pitching_updated.csv')
hof_batting = pd.read_csv('baseball-reference data/hof_batting_updated.csv')
hof_pitching = pd.read_csv('baseball-reference data/hof_pitching_updated.csv')

#drops some cols
batting = batting.drop(['IDfg','Team'],axis=1)
pitching = pitching.drop(['IDfg','Team'],axis=1)
hof_batting = hof_batting.drop('Rk',axis=1)
hof_pitching = hof_pitching.drop('Rk',axis=1)

# Label Hall of Fame players
hof_batting['HOF'] = 1
hof_pitching['HOF'] = 1

# Merge stats with Hall of Fame labels
batting = batting.merge(hof_batting[['playerID', 'HOF']], on='playerID', how='left')
pitching = pitching.merge(hof_pitching[['playerID', 'HOF']], on='playerID', how='left')

# Fill missing HOF labels with 0 (not in Hall of Fame)
batting['HOF'] = batting['HOF'].fillna(0)
pitching['HOF'] = pitching['HOF'].fillna(0)

# Select relevant features (excluding playerID and other non-numeric columns)
batting_features = batting.select_dtypes(include=['number']).drop(columns=['HOF']).fillna(0)
pitching_features = pitching.select_dtypes(include=['number']).drop(columns=['HOF']).fillna(0)

y_batting = batting['HOF']
y_pitching = pitching['HOF']

# Train-test split
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(batting_features, y_batting, test_size=0.2, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(pitching_features, y_pitching, test_size=0.2, random_state=42)

# Train XGBoost models
batting_model = xgb.XGBClassifier(eval_metric='logloss')
pitching_model = xgb.XGBClassifier(eval_metric='logloss')

batting_model.fit(X_train_b, y_train_b)
pitching_model.fit(X_train_p, y_train_p)

# Predictions
y_pred_b = batting_model.predict(X_test_b)
y_pred_p = pitching_model.predict(X_test_p)

# Saving models
with open("models/batting_model.pkl", "wb") as f:
    pickle.dump(batting_model, f)

with open("models/pitching_model.pkl", "wb") as f:
    pickle.dump(pitching_model, f)

# Evaluation
print("Batting Model Performance:")
print(accuracy_score(y_test_b, y_pred_b))
print(classification_report(y_test_b, y_pred_b))

print("\nPitching Model Performance:")
print(accuracy_score(y_test_p, y_pred_p))
print(classification_report(y_test_p, y_pred_p))
