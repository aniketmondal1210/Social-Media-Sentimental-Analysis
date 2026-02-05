import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load processed data
print("Loading data...")
with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

print(f"Classes: {le.classes_}")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Check distribution
unique, counts = np.unique(y_train, return_counts=True)
print("Train distribution:", dict(zip(le.inverse_transform(unique), counts)))

# 5. Gradient Boosting
print("\nTraining Gradient Boosting...")
# Reduced param grid for speed, but still trying to get good results
param_grid_gb = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [3]
}
gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

best_gb = grid_search_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {acc_gb:.4f}")
print("Best GB Params:", grid_search_gb.best_params_)

with open('model_gb.pkl', 'wb') as f:
    pickle.dump(best_gb, f)
