import pickle
from sklearn.metrics import accuracy_score

models = ['model_lr.pkl', 'model_rf.pkl', 'model_svm.pkl', 'model_nb.pkl', 'model_gb.pkl']

print(f"{'Model':<25} | {'Train Acc':<10} | {'Test Acc':<10}")
print("-" * 50)

with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

for model_name in models:
    try:
        with open(model_name, 'rb') as f:
            model = pickle.load(f)
        
        # Predict on Train
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # Predict on Test
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"{model_name:<25} | {train_acc:.4f}     | {test_acc:.4f}")
    except Exception as e:
        print(f"{model_name}: Error {e}")
