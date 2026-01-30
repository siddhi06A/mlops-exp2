import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")
X, y = load_iris(return_X_y=True)

preds = model.predict(X)
print("Accuracy:", accuracy_score(y, preds))
