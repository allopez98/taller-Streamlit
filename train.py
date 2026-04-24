import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv("dataset_ciclismo_fatiga.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Separación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Guardar
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Datos separados y guardados")