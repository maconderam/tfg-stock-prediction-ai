from abc import ABC, abstractmethod
import talib
import pandas as pd
import numpy as np
from scipy.stats import entropy, norm, f


class Indicator(ABC):
    """
    Clase base abstracta para la creación y análisis de indicadores financieros.

    Proporciona la estructura común para almacenar datos, calcular estadísticas
    descriptivas y evaluar la entropía de la señal generada.
    """

    def __init__(self, data: pd.DataFrame, name: str, variables: dict):
        """Inicializa el indicador básico con sus datos y parámetros.

        Args:
            data (pd.DataFrame): DataFrame con los datos de precios (OHLCV).
            name (str): Nombre identificativo del indicador.
            variables (dict): Parámetros de configuración utilizados en el cálculo.
        """
        self.data = data.copy()
        self.name = name
        self.result = None
        self.variables = variables
        self.stats = None
        self.entropy = None

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """Método abstracto encargado de realizar el cálculo matemático del indicador.

        Returns:
            pd.DataFrame: DataFrame con los resultados calculados.
        """
        pass

    def info(self):
        """Imprime por consola la información completa, variables, estadísticas y entropía."""
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
        """Devuelve el DataFrame de datos original (OHLCV)."""
        return self.data

    def get_result(self) -> pd.DataFrame:
        """Devuelve el DataFrame resultante del indicador calculado."""
        return self.result

    def get_signal(self, col: str = None) -> pd.Series:
        """Extrae una columna específica del resultado como una Serie de Pandas.

        Args:
            col (str, optional): Nombre de la columna a extraer. Defaults to None.

        Returns:
            pd.Series: Serie temporal con la señal seleccionada.

        Raises:
            RuntimeError: Si el indicador no se ha calculado con compute().
            ValueError: Si la columna no se especifica y hay múltiples opciones, o si no existe.
        """
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
        """
        Calcula la entropía de Shannon normalizada para cada columna de resultados.

        Divide el espectro de la señal en contenedores dinámicos (bins) según su tamaño
        y calcula el ratio respecto a la entropía máxima posible para medir la predictibilidad.

        Returns:
            dict: Entropías normalizadas por columna (rango 0 a 1).
        """
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

            # Determinación dinámica de bins según tamaño muestral
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

            # Normalización (0 = baja dispersión/predecible, 1 = ruido blanco/máxima incertidumbre)
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
        """
        Calcula estadísticas descriptivas básicas de la señal resultante.

        Returns:
            dict: Diccionario por columna con media, desviación estándar, mínimos, máximos e IQR.
        """
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


# ------------------------------------------------------------------
# Trend Indicators
# ------------------------------------------------------------------

class RSI(Indicator):
    """Relative Strength Index (RSI) suavizado mediante una media móvil exponencial."""

    def __init__(self, data: pd.DataFrame, window: int = 30, smooth_window: int = 3):
        super().__init__(data, name="RSI", variables={"window": window})
        self.window = window
        self.smooth_window = smooth_window
    
    def compute(self) -> pd.DataFrame:
        close = self.data["close"]
        # Aplica RSI clásico de TA-Lib y posteriormente suaviza el ruido con una EMA
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
    """Oscilador Estocástico Tradicional utilizando suavizados exponenciales."""

    def __init__(self, data: pd.DataFrame, window: int = 30, smooth_window: int = 3):
        super().__init__(
            data,
            name="Stochastic",
            variables={"window": window, "smooth_window": smooth_window}
        )
        self.window = window
        self.smooth_window = smooth_window

    def compute(self) -> pd.DataFrame:
        high = self.data["high"]
        low = self.data["low"]
        close = self.data["close"]

        # Calcula las líneas %K y %D utilizando medias móviles exponenciales en lugar de simples
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
    """Oscilador Estocástico aplicado directamente sobre la serie temporal del RSI."""

    def __init__(self, data: pd.DataFrame, rsi_window: int = 30, stoch_window: int = 5, smooth_window: int = 3):
        super().__init__(
            data,
            name="StochasticRSI",
            variables={"rsi_window": rsi_window, "smooth_window": smooth_window}
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


class MADifference(Indicator):
    """Diferencia de Medias Móviles acotada estocásticamente.

    Normaliza la distancia entre dos medias simples utilizando el ATR del activo y
    aplica la CDF Normal para acotar los resultados estrictamente en el rango [-50, 50].
    """

    def __init__(self, data: pd.DataFrame, short_length: int = 10, long_length: int = 100, lag: int = 0):
        super().__init__(
            data,
            name="MADifference",
            variables={
                "short_length": short_length,
                "long_length":  long_length,
                "lag":          lag,
            }
        )
        self.short_length = short_length
        self.long_length  = long_length
        self.lag          = lag

    def compute(self) -> pd.DataFrame:
        close = self.data["close"]

        short_ma = talib.SMA(close, timeperiod=self.short_length)
        long_ma  = talib.SMA(close.shift(self.lag), timeperiod=self.long_length)

        # Se calcula el rango de volatilidad para el denominador adaptativo
        atr_vals = ATR(self.data, window=self.long_length + self.lag).compute()[f"atr_{self.long_length + self.lag}"]

        # Varianza teórica de la diferencia basada en las longitudes de las ventanas
        diff  = 0.5 * (self.long_length - 1.0) + self.lag
        diff -= 0.5 * (self.short_length - 1.0)
        denom = np.sqrt(np.abs(diff)) * atr_vals

        # Puntuación Z escalada y normalizada por la función de distribución acumulada (CDF)
        raw    = (short_ma - long_ma) / (denom + 1e-60)
        output = 100.0 * norm.cdf(1.5 * raw) - 50.0

        self.result = pd.DataFrame(
            {f"madiff_{self.short_length}_{self.long_length}_{self.lag}": output},
            index=self.data.index
        )
        return self.result


class MACD(Indicator):
    """Moving Average Convergence Divergence (MACD) normalizado mediante volatilidad (ATR)."""

    def __init__(self, data: pd.DataFrame, short_length: int = 12, long_length: int = 26, smooth_window: int = 9):
        super().__init__(
            data,
            name="MACD",
            variables={
                "short_length":  short_length,
                "long_length":   long_length,
                "smooth_window": smooth_window,
            }
        )
        self.short_length  = short_length
        self.long_length   = long_length
        self.smooth_window = smooth_window

    def compute(self) -> pd.DataFrame:
        close = self.data["close"]

        macd, signal, hist = talib.MACDEXT(
            close,
            fastperiod=self.short_length,
            fastmatype=talib.MA_Type.EMA,
            slowperiod=self.long_length,
            slowmatype=talib.MA_Type.EMA,
            signalperiod=self.smooth_window,
            signalmatype=talib.MA_Type.EMA
        )

        atr_window = min(self.long_length + self.smooth_window, len(close))
        atr_vals   = ATR(self.data, window=atr_window).compute()[f"atr_{atr_window}"]

        diff  = 0.5 * (self.long_length - 1.0)
        diff -= 0.5 * (self.short_length - 1.0)
        denom = np.sqrt(np.abs(diff)) * atr_vals

        # Normalización estocástica: Transforma las métricas MACD absolutas a percentiles acotados [-50, 50]
        macd_norm   = 100.0 * norm.cdf(1.0 * macd   / (denom + 1e-15)) - 50.0
        signal_norm = 100.0 * norm.cdf(1.0 * signal / (denom + 1e-15)) - 50.0
        hist_norm   = 100.0 * norm.cdf(1.0 * hist   / (denom + 1e-15)) - 50.0

        self.result = pd.DataFrame(
            {
                f"macd_{self.short_length}_{self.long_length}_{self.smooth_window}":         macd_norm,
                f"macd_signal_{self.short_length}_{self.long_length}_{self.smooth_window}": signal_norm,
                f"macd_hist_{self.short_length}_{self.long_length}_{self.smooth_window}":   hist_norm,
            },
            index=self.data.index
        )
        return self.result


class PriceIntensity(Indicator):
    """Mide la fuerza del cierre intradiario relativo al rango verdadero (True Range)."""

    def __init__(self, data: pd.DataFrame, smooth_window: int = 20):
        super().__init__(data, name="PriceIntensity", variables={"smoothing_window": smooth_window})
        self.smooth_window = smooth_window
    
    def compute(self) -> pd.DataFrame:
        open_ = self.data["open"]
        close = self.data["close"]
        high = self.data["high"]
        low = self.data["low"]
        prior_close = close.shift(1)

        # Cálculo manual de la frontera del True Range (Máxima dispersión de precios)
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
        
        # Escalado por la raíz de la ventana para estabilizar la varianza temporal antes de aplicar CDF
        pi = 100 * norm.cdf(0.8 * np.sqrt(self.smooth_window) * raw_pi_smoothed) - 50

        self.result = pd.DataFrame(
            {f"pi_{self.smooth_window}": pi},
            index=self.data.index
        )
        return self.result


class ADX(Indicator):
    """Average Directional Movement Index (ADX) estándar para evaluar la fuerza de la tendencia."""

    def __init__(self, data: pd.DataFrame, window: int = 14):
        super().__init__(data, name="ADX", variables={"window": window})
        self.window = window

    def compute(self) -> pd.DataFrame:
        high  = self.data["high"]
        low   = self.data["low"]
        close = self.data["close"]

        adx = talib.ADX(high, low, close, timeperiod=self.window)

        self.result = pd.DataFrame(
            {f"adx_{self.window}": adx},
            index=self.data.index
        )
        return self.result


class Aroon(Indicator):
    """Indicador Aroon tradicional (Aroon Up y Aroon Down)."""

    def __init__(self, data: pd.DataFrame, window: int = 100):
        super().__init__(data, name="Aroon", variables={"window": window})
        self.window = window

    def compute(self) -> pd.DataFrame:
        high = self.data["high"]
        low  = self.data["low"]

        aroon_down, aroon_up = talib.AROON(high, low, timeperiod=self.window)

        self.result = pd.DataFrame(
            {
                f"aroon_up_{self.window}":   aroon_up,
                f"aroon_down_{self.window}": aroon_down,
            },
            index=self.data.index
        )
        return self.result


class AroonOscillator(Indicator):
    """Oscilador derivado de la diferencia neta entre Aroon Up y Aroon Down."""

    def __init__(self, data: pd.DataFrame, window: int = 100):
        super().__init__(data, name="AroonOscillator", variables={"window": window})
        self.window = window

    def compute(self) -> pd.DataFrame:
        high = self.data["high"]
        low  = self.data["low"]

        aroonosc = talib.AROONOSC(high, low, timeperiod=self.window)

        self.result = pd.DataFrame(
            {f"aroonosc_{self.window}": aroonosc},
            index=self.data.index
        )
        return self.result


# ------------------------------------------------------------------
# Volatility Indicators
# ------------------------------------------------------------------

class ATR(Indicator):
    """Average True Range (ATR) para cuantificar la volatilidad absoluta."""

    def __init__(self, data: pd.DataFrame, window: int = 14):
        super().__init__(data, name="ATR", variables={"window": window})
        self.window = window

    def compute(self) -> pd.DataFrame:
        high  = self.data["high"]
        low   = self.data["low"]
        close = self.data["close"]

        atr = talib.ATR(high, low, close, timeperiod=self.window)

        self.result = pd.DataFrame(
            {f"atr_{self.window}": atr},
            index=self.data.index
        )
        return self.result


class PriceChangeOscillator(Indicator):
    """Oscilador estadístico basado en la velocidad de los retornos logarítmicos absolutos."""

    def __init__(self, data: pd.DataFrame, short_length: int = 10, mult: int = 5):
        if mult < 2:
            mult = 2
        super().__init__(
            data,
            name="PriceChangeOscillator",
            variables={
                "short_length": short_length,
                "mult":         mult,
            }
        )
        self.short_length = short_length
        self.mult         = mult
        self.long_length  = short_length * mult

    def compute(self) -> pd.DataFrame:
        close = self.data["close"]

        log_returns = np.abs(np.log(close / close.shift(1)))
        atr_vals    = ATR(self.data, window=self.long_length).compute()[f"atr_{self.long_length}"]

        short_sum = talib.SMA(log_returns, timeperiod=self.short_length) 
        long_sum  = talib.SMA(log_returns, timeperiod=self.long_length)

        # Factor corrector analítico para estabilizar los grados de libertad del oscilador
        denom_base  = 0.36 + 1.0 / self.short_length
        v           = np.log(0.5 * self.mult) / 1.609
        denom_base += 0.7 * v

        denom = denom_base * atr_vals
        raw   = (short_sum - long_sum) / (denom + 1e-20)
        output = 100.0 * norm.cdf(4.0 * raw) - 50.0
        output[denom <= 1e-20] = 0.0

        self.result = pd.DataFrame(
            {f"pco_{self.short_length}_{self.mult}": output},
            index=self.data.index
        )
        return self.result 


class PriceVarianceRatio(Indicator):
    """Ratio de varianza logarítmica de precios mapeado a través de una distribución F de Snedecor."""

    def __init__(self, data: pd.DataFrame, short_length: int = 10, mult: int = 4):
        if mult < 2:
            mult = 2
        super().__init__(
            data,
            name="PriceVarianceRatio",
            variables={
                "short_length": short_length,
                "mult":         mult,
            }
        )
        self.short_length = short_length
        self.mult         = mult
        self.long_length  = short_length * mult

    def compute(self) -> pd.DataFrame:
        log_close = np.log(self.data["close"])

        var_short = talib.VAR(log_close, timeperiod=self.short_length, nbdev=1)
        var_long  = talib.VAR(log_close, timeperiod=self.long_length,  nbdev=1)

        # Utiliza la CDF de la distribución F para validar si la varianza de corto plazo
        # difiere significativamente de la varianza estructural a largo plazo.
        ratio  = var_short / (var_long + 1e-60)
        output = 100.0 * f.cdf(self.mult * ratio, 2, 2 * self.mult) - 50.0
        output[var_long <= 0] = 0.0

        self.result = pd.DataFrame(
            {f"pvr_{self.short_length}_{self.mult}": output},
            index=self.data.index
        )
        return self.result


class ChangeVarianceRatio(Indicator):
    """Ratio de varianza de retornos lógarítmicos evaluado bajo la distribución estadística F."""

    def __init__(self, data: pd.DataFrame, short_length: int = 10, mult: int = 4):
        if mult < 2:
            mult = 2
        super().__init__(
            data,
            name="ChangeVarianceRatio",
            variables={
                "short_length": short_length,
                "mult":         mult,
            }
        )
        self.short_length = short_length
        self.mult         = mult
        self.long_length  = short_length * mult

    def compute(self) -> pd.DataFrame:
        close       = self.data["close"]
        log_returns = np.log(close / close.shift(1))

        var_short = talib.VAR(log_returns, timeperiod=self.short_length, nbdev=1)
        var_long  = talib.VAR(log_returns, timeperiod=self.long_length,  nbdev=1)

        # A diferencia de PVR, evalúa directamente los cambios marginales o retornos del activo
        ratio  = var_short / (var_long + 1e-60)
        output = 100.0 * f.cdf(ratio, 4, 4 * self.mult) - 50.0
        output[var_long <= 0] = 0.0

        self.result = pd.DataFrame(
            {f"cvr_{self.short_length}_{self.mult}": output},
            index=self.data.index
        )
        return self.result


# ------------------------------------------------------------------
# Deviation from Expectation Indicators
# ------------------------------------------------------------------

class CMMA(Indicator):
    """Continuous Moving Average Deviation (CMMA).

    Mide la desviación del precio logarítmico respecto a su media móvil simple,
    estandarizándolo según la volatilidad histórica e incorporando un ajuste por raíz temporal.
    """

    def __init__(self, data: pd.DataFrame, window: int = 10, atr_window: int = 252, c: float = 1.0):
        super().__init__(
            data,
            name="CMMA",
            variables={
                "window":     window,
                "atr_window": atr_window,
                "c":          c,
            }
        )
        self.window     = window
        self.atr_window = atr_window
        self.c          = c

    def compute(self) -> pd.DataFrame:
        close     = self.data["close"]
        log_close = np.log(close)

        ma  = talib.SMA(log_close, timeperiod=self.window)
        atr = ATR(self.data, window=self.atr_window).compute()[f"atr_{self.atr_window}"]

        # Estandarización de distancia utilizando la raíz cuadrada de la ventana temporal
        raw  = (log_close - ma) / ((atr + 1e-12) * np.sqrt(self.window + 1))
        cmma = 100 * norm.cdf(self.c * raw) - 50

        self.result = pd.DataFrame(
            {f"cmma_{self.window}_{self.atr_window}_{self.c}": cmma},
            index=self.data.index
        )
        return self.result


# ------------------------------------------------------------------
# Volume Indicators
# ------------------------------------------------------------------

class IntradayIntensity(Indicator):
    """Intraday Intensity (II) ponderada por volumen y suavizada mediante medias móviles."""

    def __init__(self, data: pd.DataFrame, window: int = 14, smooth_window: int = 1):
        super().__init__(
            data,
            name="IntradayIntensity",
            variables={"window": window, "smooth_window": smooth_window}
        )
        self.window        = window
        self.smooth_window = smooth_window

    def compute(self) -> pd.DataFrame:
        high   = self.data["high"]
        low    = self.data["low"]
        close  = self.data["close"]
        volume = self.data["volume"]

        hl_range = high - low
        # Factor posicional de la barra de precios escalado por el flujo de volumen total
        raw      = (2 * close - high - low) / (hl_range + 1e-12) * 100.0 * volume
        raw[hl_range <= 0] = 0.0

        ii = talib.SMA(raw, timeperiod=self.window)

        # Si se solicita, des-escala la señal dividiendo por el volumen medio exponencial (EMA)
        if self.smooth_window > 1:
            vol_ema = talib.EMA(volume, timeperiod=self.smooth_window)
            ii      = ii / (vol_ema + 1e-12)

        self.result = pd.DataFrame(
            {f"ii_{self.window}_{self.smooth_window}": ii},
            index=self.data.index
        )
        return self.result


class ChaikinMoneyFlow(Indicator):
    """Chaikin Money Flow (CMF) para monitorizar la acumulación o distribución de capital."""

    def __init__(self, data: pd.DataFrame, window: int = 14):
        super().__init__(
            data,
            name="ChaikinMoneyFlow",
            variables={"window": window}
        )
        self.window = window

    def compute(self) -> pd.DataFrame:
        high   = self.data["high"]
        low    = self.data["low"]
        close  = self.data["close"]
        volume = self.data["volume"]

        hl_range = high - low
        raw      = (2 * close - high - low) / (hl_range + 1e-12) * 100.0 * volume
        raw[hl_range <= 0] = 0.0

        # Ratio entre la suma móvil de la intensidad intradiaria y la suma del volumen total diario
        ii_sma  = talib.SMA(raw,    timeperiod=self.window)
        vol_sma = talib.EMA(volume, timeperiod=self.window)
        cmf     = ii_sma / (vol_sma + 1e-12)

        self.result = pd.DataFrame(
            {f"cmf_{self.window}": cmf},
            index=self.data.index
        )
        return self.result