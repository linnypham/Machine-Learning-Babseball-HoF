import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
#graph
def plot_confusion_and_roc(y_test, y_pred, y_proba, title_prefix):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{title_prefix} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curve")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f'graphs/{title_prefix}_rf.png')
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

#batting col=['AVG','OBP','SLG','OPS','WAR']
batting_features = batting_features[['AVG','OBP','SLG','OPS','RBI']]
#pitching col=['ERA','WHIP','K/9','FIP','WAR']
pitching_features = pitching_features[['ERA','WHIP','K/9','FIP','K/BB']]
#y = HOF
y_batting = batting['HOF']
y_pitching = pitching['HOF']

# Train-test split
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(batting_features, y_batting, test_size=0.2, random_state=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(pitching_features, y_pitching, test_size=0.2, random_state=42)

# Random Forest Models
batting_model = RandomForestClassifier(n_estimators=100, random_state=42)
pitching_model = RandomForestClassifier(n_estimators=100, random_state=42)

#SMOTE and fitting
sm = SMOTE(random_state=42)
X_resampled_b, y_resampled_b = sm.fit_resample(X_train_b, y_train_b)
X_resampled_p, y_resampled_p = sm.fit_resample(X_train_p, y_train_p)
batting_model.fit(X_resampled_b, y_resampled_b)
pitching_model.fit(X_resampled_p, y_resampled_p)


# Predictions
y_pred_b = batting_model.predict(X_test_b)
y_pred_p = pitching_model.predict(X_test_p)


# Get prediction probabilities
y_proba_b = batting_model.predict_proba(X_test_b)[:, 1]
y_proba_p = pitching_model.predict_proba(X_test_p)[:, 1]

# Plot graphs
plot_confusion_and_roc(y_test_b, y_pred_b, y_proba_b, "Batting Model")
plot_confusion_and_roc(y_test_p, y_pred_p, y_proba_p, "Pitching Model")
# Evaluation
print("Batting Model Performance:")
print(accuracy_score(y_test_b, y_pred_b))
print(classification_report(y_test_b, y_pred_b))

print("\nPitching Model Performance:")
print(accuracy_score(y_test_p, y_pred_p))
print(classification_report(y_test_p, y_pred_p))
