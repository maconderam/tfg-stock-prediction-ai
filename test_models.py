import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
from modelos.models import SklearnModel
from eda.eda import prepare_data
from evaluacion.visualizador import VisualizerWalkforward
from evaluacion.walkforward import WalkForwardEvaluator
from evaluacion.feature_analyzer import FeatureAnalyzer
from indicadores.indicators import RSI, Stochastic, StochasticRSI, MACD, PriceIntensity
from indicadores.targets import NormalizedFutureReturn  
 
# ------------------------------------------------------------------
# 1. Carga y limpieza
# ------------------------------------------------------------------
df = pd.read_csv("datos/procesados/BTCUSDT_1d_01-01-2016_18-01-2026.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.dropna()
ultima_fecha = df["timestamp"].max()
fecha_corte = ultima_fecha - pd.DateOffset(years=3)

df_train = df[df["timestamp"] < fecha_corte].copy()
df_test = df[df["timestamp"] >= fecha_corte].copy()

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

viz = VisualizerWalkforward(df, fold_results, signal)
fig1 = viz.plot()
fig1.show()

fig2 = viz.plot_walkforward_metrics()
fig2.show()

fig = viz.plot_mcpt_metrics()
fig.show()

fa = FeatureAnalyzer(df_train, df_test, columns=["rsi_14_3", "adx_14", "cmma_10_14_1.0"])

fa.plot_correlation_heatmap("train").show()
fa.plot_correlation_comparison().show()       # train y test lado a lado
fa.plot_correlation_difference().show()       # dónde la correlación es inestable

redundantes = fa.find_redundant_pairs(threshold=0.8)
print(redundantes)

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