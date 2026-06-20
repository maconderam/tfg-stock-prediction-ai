from indicadores.indicator_selector import IndicatorSelector
from eda.eda import prepare_data
import pandas as pd
import matplotlib.pyplot as plt

path = "datos/crudo/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv"

df = pd.read_csv(path, sep=",")
df = prepare_data(df)

df = df.dropna()
ultima_fecha = df["timestamp"].max()
fecha_corte = ultima_fecha - pd.DateOffset(years=3)

df_train = df[df["timestamp"] < fecha_corte].copy()
df_test = df[df["timestamp"] >= fecha_corte].copy()

sel_train = IndicatorSelector(df_train, n_mcpt=50, p_threshold=0.05)
df = sel_train.run()                        # grid search completo
print(sel_train.top_n(10))                  # top 10 significativos

fig1 = sel_train.plot_summary(top_n=20)     # dashboard comparativo
fig1.show()

fig2 = sel_train.plot_mc_distributions(top_n=6, only_significant=True)
fig2.show()

sel_test = IndicatorSelector(df_test, n_mcpt=50, p_threshold=0.05)
df = sel_test.run()                        # grid search completo
print(sel_test.top_n(10))                  # top 10 significativos

fig1 = sel_test.plot_summary(top_n=20)     # dashboard comparativo
fig1.show()

fig2 = sel_test.plot_mc_distributions(top_n=6, only_significant=True)
fig2.show()