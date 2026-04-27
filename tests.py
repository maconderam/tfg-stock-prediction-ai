from evaluacion.visualizador import Visualizer
from evaluacion.thresholds import ThresholdEvaluator
from evaluacion.mcpt import MonteCarloPT
import indicadores.indicators as ind
import indicadores.targets as targ
from eda.eda import prepare_data
from pathlib import Path
import pandas as pd

path = "datos/crudo/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv"

df = pd.read_csv(path, sep=",")
df = prepare_data(df)
te = ThresholdEvaluator(df)
mcpt = MonteCarloPT(df)

rsi_7 = ind.RSI(df, window=7)
sto_7 = ind.Stochastic(df, window=7)
sto_rsi_14_5 = ind.StochasticRSI(df, rsi_window=14)
macd_12_26 = ind.MACD(df)
pi_10 = ind.PriceIntensity(df)
nfr_1_atr_14 = targ.NormalizedFutureReturn(df)

indicators = [
    rsi_7,
    sto_7,
    sto_rsi_14_5,
    macd_12_26,
    pi_10,
    nfr_1_atr_14
]

for ind in indicators:
    df = pd.concat([df, ind.compute()], axis=1)

mcpt.mcpt_threshold(macd_12_26.get_result().iloc[:, 0], n_test=1000)
mcpt.mcpt_threshold(pi_10.get_result().iloc[:, 0], n_test=1000)

# df.to_csv("datos/procesados/BTCUSDT_1d_01-01-2016_18-01-2026.csv")

#visualizer.update_data(df)
#visualizer.plot_with_indicators(panels=["rsi_7"])