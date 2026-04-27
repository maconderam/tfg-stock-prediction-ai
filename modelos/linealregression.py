import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("datos/procesados/BTCUSDT_1d_01-01-2016_18-01-2026.csv")

df = df.dropna()

X = df[["rsi_7","%K_7","%D_7"]] # features
y = df["nfr_1_atr_14"]                              # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("regresion", LinearRegression())
])
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2:", r2)
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr,
    annot=True,        # muestra valores dentro
    fmt=".2f",         # formato 2 decimales
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Matriz de correlaciones")
plt.show()