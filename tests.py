from evaluacion.thresholds import ThresholdEvaluator
from evaluacion.mcpt import MonteCarloPT
from indicadores.indicators import (
    RSI, Stochastic, StochasticRSI, MACD, PriceIntensity,
    ADX, Aroon, AroonOscillator, ATR, PriceChangeOscillator, CMMA, MADifference,
    PriceVarianceRatio, ChangeVarianceRatio
)
from indicadores.targets import NormalizedFutureReturn
from eda.eda import prepare_data
import pandas as pd
import numpy as np

path = "datos/crudo/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv"
df   = pd.read_csv(path, sep=",")
df   = prepare_data(df)

indicators = [
    RSI(df, window=7),
    RSI(df, window=14),
    Stochastic(df, window=14),
    StochasticRSI(df, rsi_window=14),
    MACD(df, short_length=12, long_length=26),
    PriceIntensity(df, smooth_window=20),
    ADX(df, window=14),
    Aroon(df, window=14),
    AroonOscillator(df, window=14),
    PriceChangeOscillator(df, short_length=10, mult=5),
    CMMA(df, window=10, atr_window=14),
    MADifference(df, short_length=10, long_length=50),
    PriceVarianceRatio(df, short_length=10, mult=4),
    ChangeVarianceRatio(df, short_length=10, mult=4),
    NormalizedFutureReturn(df, window=14)
]

for ind in indicators:
    ind.compute()
    print(f"{ind.name:<25} OK — columnas: {list(ind.result.columns)}")
    df = pd.concat([df, ind.result], axis=1)

df.to_csv("datos/procesados/BTCUSDT_1d_01-01-2016_18-01-2026.csv")
print("\nCSV exportado correctamente.")