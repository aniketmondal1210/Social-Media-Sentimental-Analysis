import pickle
from sklearn.metrics import accuracy_score

models = ['model_lr.pkl', 'model_rf.pkl', 'model_svm.pkl', 'model_nb.pkl', 'model_gb.pkl']
accuracies = {}

with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

for model_name in models:
    try:
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[model_name] = acc
        print(f"{model_name}: {acc:.4f}")
    except Exception as e:
        print(f"{model_name}: Error {e}")
