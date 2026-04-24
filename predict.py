import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

# Cargar datos de prueba
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Cargar modelos
knn = joblib.load("models/modelo_knn.pkl")
lr = joblib.load("models/modelo_lr.pkl")
tree = joblib.load("models/modelo_tree.pkl")

# Predicciones
pred_knn = knn.predict(X_test)
pred_lr = lr.predict(X_test)
pred_tree = tree.predict(X_test)

# Evaluación
print("KNN MSE:", mean_squared_error(y_test, pred_knn))
print("LR MSE:", mean_squared_error(y_test, pred_lr))
print("TREE MSE:", mean_squared_error(y_test, pred_tree))