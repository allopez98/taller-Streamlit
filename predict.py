from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

required_files = [
    DATA_DIR / "X_test.csv",
    DATA_DIR / "y_test.csv",
    MODELS_DIR / "modelo_knn.pkl",
    MODELS_DIR / "modelo_lr.pkl",
    MODELS_DIR / "modelo_tree.pkl",
]

missing_files = [str(path.name) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "Faltan archivos para evaluar. Ejecuta primero el entrenamiento: "
        + ", ".join(missing_files)
    )

# Cargar datos de prueba
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

# Cargar modelos
knn = joblib.load(MODELS_DIR / "modelo_knn.pkl")
lr = joblib.load(MODELS_DIR / "modelo_lr.pkl")
tree = joblib.load(MODELS_DIR / "modelo_tree.pkl")

# Predicciones
pred_knn = knn.predict(X_test)
pred_lr = lr.predict(X_test)
pred_tree = tree.predict(X_test)

# Evaluación
print("KNN MSE:", mean_squared_error(y_test, pred_knn))
print("LR MSE:", mean_squared_error(y_test, pred_lr))
print("TREE MSE:", mean_squared_error(y_test, pred_tree))