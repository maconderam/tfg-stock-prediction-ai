import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
from modelos.models import SklearnModel
from evaluacion.visualizador import Visualizer
from evaluacion.walkforward import WalkForwardEvaluator
from indicadores.indicators import RSI, Stochastic, StochasticRSI, MACD, PriceIntensity
from indicadores.targets import NormalizedFutureReturn  
 
# ------------------------------------------------------------------
# 1. Carga y limpieza
# ------------------------------------------------------------------
df = pd.read_csv("datos/procesados/BTCUSDT_1d_01-01-2016_18-01-2026.csv")
df = df.dropna()

features = ["rsi_7_3", "%K_14_3", "%D_14_3"]
target   = "nfr_1_atr_14"
 
# ------------------------------------------------------------------
# 2. Walk-forward
# ------------------------------------------------------------------

sklearn_pipe = Pipeline([
    ("regresion", LinearRegression())
])

model = SklearnModel(sklearn_pipe, name="LinearRegression")
wf = WalkForwardEvaluator(df, train_window=500, test_window=100)

# Para modelos
fold_results, predictions_df = wf.run_model(
    model,
    features=features,
    target=target,
    min_kept=30,
    mcpt = True,
    n_mcpt=20,
    mcpt_mode="retrain"
)

signal = predictions_df["y_pred"]
wf.summary()

viz = Visualizer(df, fold_results, signal)
fig1 = viz.plot()
fig1.show()

fig2 = viz.plot_walkforward_metrics()
fig2.show()

fig = viz.plot_mcpt_metrics()
fig.show()


# Para indicadores
"""
rsi = RSI(df, window=14)
rsi.compute()

wf = WalkForwardEvaluator(df, train_window=500, test_window=100)
wf.run_indicator(rsi.get_signal())
wf.summary()

viz = Visualizer(df, wf.fold_results, rsi.get_signal())
fig1 = viz.plot()
fig1.show()

fig2 = viz.plot_walkforward_metrics()
fig2.show()
"""