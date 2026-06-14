from abc import ABC, abstractmethod
import talib
import pandas as pd
import numpy as np
from scipy.stats import entropy, norm

class Indicator(ABC):
    def __init__(self, data: pd.DataFrame, name: str, variables: dict):
        self.data = data.copy()
        self.name = name
        self.result = None
        self.variables = variables
        self.stats = None
        self.entropy = None

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        pass

    def info(self):
        print("-" * 20)
        print(f"Indicator: {self.name}")

        print("Variables:")
        for k, v in self.variables.items():
            print(f"  {k}: {v}")

        if self.result is None:
            print("Result: not computed yet")
            return

        # Stats
        if self.stats is not None:
            print("Statistics:")
            for col, stats in self.stats.items():
                print(f"  {col}:")
                for k, v in stats.items():
                    print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            self.calculate_stats()

        # Entropy
        if self.entropy is not None:
            print("Entropy:")
            for col, value in self.entropy.items():
                if np.isnan(value):
                    print(f"  {col}: nan")
                else:
                    print(f"  {col}: {value:.6f}")
        else:
            self.calculate_entropy()

    def get_data(self) -> pd.DataFrame:
        return self.data

    def get_result(self) -> pd.DataFrame:
        return self.result

    def get_signal(self, col: str = None) -> pd.Series:
        if self.result is None:
            raise RuntimeError("Call compute() first")
        if col is None:
            if len(self.result.columns) == 1:
                return self.result.iloc[:, 0]
            raise ValueError(f"Specify col. Available: {list(self.result.columns)}")
        if col not in self.result.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(self.result.columns)}")
        return self.result[col]

    def calculate_entropy(self) -> dict:
        if self.result is None:
            raise ValueError("You must run compute() first")

        entropies = {}

        for col in self.result.columns:

            x = self.result[col].values
            x = x[~np.isnan(x)]

            n = len(x)

            if n == 0:
                entropies[col] = np.nan
                continue

            if n > 10000:
                bins = 20
            elif n > 1000:
                bins = 10
            elif n > 100:
                bins = 5
            else:
                bins = 3

            hist, _ = np.histogram(x, bins=bins, density=True)
            hist = hist[hist > 0]

            h = entropy(hist)
            max_entropy = np.log(bins)

            entropies[col] = h / max_entropy

        self.entropy = entropies

        print("Entropy:")
        for col, value in entropies.items():
            if np.isnan(value):
                print(f"  {col}: nan")
            else:
                print(f"  {col}: {value:.6f}")

        return self.entropy

    def calculate_stats(self) -> dict:
        if self.result is None:
            raise ValueError("You must run compute() first")

        all_stats = {}

        for col in self.result.columns:

            x = self.result[col].values
            x = x[~np.isnan(x)]

            if len(x) == 0:
                continue

            stats = {
                "n": len(x),
                "mean": np.mean(x),
                "std": np.std(x),
                "min": np.min(x),
                "max": np.max(x),
                "range": np.max(x) - np.min(x),
                "iqr": np.percentile(x, 75) - np.percentile(x, 25),
            }

            all_stats[col] = stats

        self.stats = all_stats

        print("Statistics:")
        for col, stats in all_stats.items():
            print(f"  {col}:")
            for k, v in stats.items():
                print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

        return self.stats

class RSI(Indicator):
    def __init__(self, data: pd.DataFrame, window: int = 30, smooth_window: int = 3):
        super().__init__(
            data, 
            name="RSI", 
            variables={
                "window":window
            }
        )
        self.window = window
        self.smooth_window = smooth_window
    
    def compute(self) -> pd.DataFrame:
        close = self.data["close"]

        rsi = talib.EMA(
            talib.RSI(close, timeperiod=self.window),
            timeperiod=self.smooth_window
        )

        self.result = pd.DataFrame(
            {f"rsi_{self.window}_{self.smooth_window}": rsi},
            index=self.data.index
        )

        return self.result

class Stochastic(Indicator):
    def __init__(self, data: pd.DataFrame, window: int = 30, smooth_window: int = 3):
        super().__init__(
            data,
            name="Stochastic",
            variables={
                "window": window,
                "smooth_window": smooth_window
            }
        )

        self.window = window
        self.smooth_window = smooth_window

    def compute(self) -> pd.DataFrame:
        high = self.data["high"]
        low = self.data["low"]
        close = self.data["close"]

        k, d = talib.STOCH(
            high, low, close,
            fastk_period=self.window,
            slowk_period=self.smooth_window,
            slowd_period=self.smooth_window,
            slowk_matype=talib.MA_Type.EMA,
            slowd_matype=talib.MA_Type.EMA
        )

        self.result = pd.DataFrame(
            {
                f"%K_{self.window}_{self.smooth_window}": k,
                f"%D_{self.window}_{self.smooth_window}": d
            },
            index=self.data.index
        )

        return self.result

class StochasticRSI(Indicator):
    def __init__(self, data: pd.DataFrame, rsi_window: int = 30, stoch_window: int = 5, smooth_window: int = 3):
        super().__init__(
            data,
            name="StochasticRSI",
            variables={
                "rsi_window": rsi_window,
                "smooth_window": smooth_window
            }
        )

        self.rsi_window = rsi_window
        self.stoch_window = stoch_window
        self.smooth_window = smooth_window

    def compute(self) -> pd.DataFrame:
        close = self.data["close"]

        k, d = talib.STOCHRSI(
            close,
            timeperiod=self.rsi_window,
            fastk_period=self.stoch_window,
            fastd_period=self.smooth_window,
            fastd_matype=talib.MA_Type.EMA
        )

        self.result = pd.DataFrame(
            {
                f"rsi_%K_{self.rsi_window}_{self.stoch_window}_{self.smooth_window}": k,
                f"rsi_%D_{self.rsi_window}_{self.stoch_window}_{self.smooth_window}": d
            },
            index=self.data.index
        )

        return self.result

class MACD(Indicator):
    def __init__(self, data: pd.DataFrame, long_window: int = 12, short_window: int = 26, smooth_window: int = 9):
        super().__init__(
            data, 
            name="MACD", 
            variables={
                "long_window":long_window,
                "short_window":short_window,
                "smoothing_window":smooth_window
            }
        )
        self.long_window = long_window
        self.short_window = short_window
        self.smooth_window = smooth_window
    
    def compute(self) -> pd.DataFrame:
        close = self.data["close"]

        macd, signal, hist = talib.MACDEXT(
            close,
            fastperiod=self.long_window,
            fastmatype=talib.MA_Type.EMA,
            slowperiod=self.short_window,
            slowmatype=talib.MA_Type.EMA,
            signalperiod=self.smooth_window,
            signalmatype=talib.MA_Type.EMA
        )

        self.result = pd.DataFrame(
            {
                f"macd_{self.long_window}_{self.short_window}_{self.smooth_window}": macd,
                f"macd_signal_{self.long_window}_{self.short_window}_{self.smooth_window}": signal,
                f"macd_hist_{self.long_window}_{self.short_window}_{self.smooth_window}": hist
            },
            index=self.data.index
        )

        return self.result

class PriceIntensity(Indicator):
    def __init__(self, data: pd.DataFrame, smooth_window: int = 10):
        super().__init__(
            data, 
            name="PriceIntensity", 
            variables={
                "smoothing_window":smooth_window
            }
        )
        self.smooth_window = smooth_window
    
    def compute(self) -> pd.DataFrame:
        open_ = self.data["open"]
        close = self.data["close"]
        high = self.data["high"]
        low = self.data["low"]
        prior_close = close.shift(1)

        denom = pd.concat(
            [
                high - low,
                (high - prior_close).abs(),
                (prior_close - low).abs()
            ],
            axis=1
        ).max(axis=1)

        raw_pi = (close - open_) / (denom + 1e-12)
        raw_pi_smoothed = talib.EMA(raw_pi, timeperiod=self.smooth_window)
        pi = 100 * norm.cdf(0.8 * np.sqrt(self.smooth_window) * raw_pi_smoothed) - 50

        self.result = pd.DataFrame(
            {f"pi_{self.smooth_window}": pi},
            index=self.data.index
        )

        return self.result