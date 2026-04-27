from abc import ABC, abstractmethod
import talib
import pandas as pd
import numpy as np
from scipy.stats import entropy

class Target(ABC):
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
        print(f"Target: {self.name}")

        print("Variables:")
        for k, v in self.variables.items():
            print(f"  {k}: {v}")

        if self.result is None:
            print("Result: not computed yet")
            return

        # Stats
        if self.stats is None:
            self.calculate_stats()
        else:
            print("Statistics:")
            for col, stats in self.stats.items():
                print(f"  {col}:")
                for k, v in stats.items():
                    print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

        # Entropy
        if self.entropy is None:
            self.calculate_entropy()
        else:
            print("Entropy:")
            for col, value in self.entropy.items():
                if np.isnan(value):
                    print(f"  {col}: nan")
                else:
                    print(f"  {col}: {value:.6f}")

        # Entropy
        if self.entropy is None:
            self.calculate_entropy()

        else:
            print("Entropy:")
            for target, value in self.entropy.items():
                if np.isnan(value):
                    print(f"  {target}: nan")
                else:
                    print(f"  {target}: {value:.6f}")

    def get_data(self):
        return self.data

    def get_result(self):
        return self.result

    def calculate_entropy(self):
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

    def calculate_stats(self):
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

class NormalizedFutureReturn(Target):
    def __init__(self, data, horizon=1, method="atr", window=14):
        super().__init__(
            data, 
            name="NormalizedFutureReturn", 
            variables={
                "horizon":horizon,
                "method":method,
                "window":window
            }
        )
        self.horizon = horizon
        self.method = method
        self.window = window

    def compute(self) -> pd.DataFrame:
        close = self.data["close"]
        high = self.data["high"]
        low = self.data["low"]

        future_return = (close.shift(-self.horizon) - close)

        if self.method == "atr":
            atr = talib.ATR(
                high,
                low,
                close,
                timeperiod=self.window
            )
            denom = atr

        elif self.method == "std":
            denom = talib.STDDEV(close, timeperiod=self.window, nbdev=1)

        else:
            raise ValueError("method must be 'atr' or 'std'")

        self.result = pd.DataFrame({f"nfr_{self.horizon}_{self.method}_{self.window}": future_return / denom}, 
                                   index=self.data.index)

        return self.result