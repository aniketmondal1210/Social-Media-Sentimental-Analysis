import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
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

# 3. SVM
print("\nTraining SVM...")
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
svm = SVC(random_state=42)
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

best_svm = grid_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {acc_svm:.4f}")
print("Best SVM Params:", grid_search_svm.best_params_)

with open('model_svm.pkl', 'wb') as f:
    pickle.dump(best_svm, f)

# 4. Naive Bayes
print("\nTraining Naive Bayes...")
param_grid_nb = {
    'alpha': [0.1, 0.5, 1.0, 5.0]
}
nb = MultinomialNB()
grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_nb.fit(X_train, y_train)

best_nb = grid_search_nb.best_estimator_
y_pred_nb = best_nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {acc_nb:.4f}")
print("Best NB Params:", grid_search_nb.best_params_)

with open('model_nb.pkl', 'wb') as f:
    pickle.dump(best_nb, f)
