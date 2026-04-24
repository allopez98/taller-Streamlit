import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# 📌 Cargar dataset (CORREGIDO)
df = pd.read_csv("data/dataset_ciclismo_fatiga.csv")

# ⚠️ Ajusta "target" si tu columna objetivo tiene otro nombre
X = df.drop("target", axis=1)
y = df["target"]

# Separación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Guardar datasets
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Datos separados y guardados")

# 🔹 Pipeline KNN
pipeline_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsRegressor())
])

pipeline_knn.fit(X_train, y_train)
joblib.dump(pipeline_knn, "models/modelo_knn.pkl")

# 🔹 Pipeline Regresión Lineal
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline_lr.fit(X_train, y_train)
joblib.dump(pipeline_lr, "models/modelo_lr.pkl")

# 🔹 Árbol (SIN estandarizar)
modelo_tree = DecisionTreeRegressor()
modelo_tree.fit(X_train, y_train)

joblib.dump(modelo_tree, "models/modelo_tree.pkl")

print("Modelos entrenados y guardados")