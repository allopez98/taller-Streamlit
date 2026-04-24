from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TARGET_COLUMN = "fatiga"

# Crear carpetas necesarias si no existen.
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Cargar dataset
df = pd.read_csv(BASE_DIR / "dataset_ciclismo_fatiga.csv")
if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"La columna objetivo '{TARGET_COLUMN}' no existe en el dataset."
    )
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Separación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Guardar datos de entrenamiento/prueba
X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

# Entrenar modelos
pipeline_knn = Pipeline(
    [("scaler", StandardScaler()), ("model", KNeighborsRegressor())]
)
pipeline_lr = Pipeline(
    [("scaler", StandardScaler()), ("model", LinearRegression())]
)
modelo_tree = DecisionTreeRegressor(random_state=42)

pipeline_knn.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)
modelo_tree.fit(X_train, y_train)

# Guardar modelos
joblib.dump(pipeline_knn, MODELS_DIR / "modelo_knn.pkl")
joblib.dump(pipeline_lr, MODELS_DIR / "modelo_lr.pkl")
joblib.dump(modelo_tree, MODELS_DIR / "modelo_tree.pkl")

print("Datos y modelos generados correctamente")