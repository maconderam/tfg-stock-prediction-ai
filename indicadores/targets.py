from abc import ABC, abstractmethod
import talib
import pandas as pd
import numpy as np
from scipy.stats import entropy


class Target(ABC):
    """
    Clase base abstracta para la generación de variables objetivo (labels).

    Diseñada para el desarrollo de modelos de Machine Learning aplicados a finanzas.
    Permite calcular de forma estandarizada la entropía de Shannon normalizada
    y estadísticas descriptivas sobre las etiquetas generadas.
    """

    def __init__(self, data: pd.DataFrame, name: str, variables: dict):
        """Inicializa la configuración base del Target.

        Args:
            data (pd.DataFrame): DataFrame histórico con los datos de entrada (OHLCV).
            name (str): Nombre identificativo del método de etiquetado.
            variables (dict): Hiperparámetros clave utilizados para el cálculo.
        """
        self.data = data.copy()
        self.name = name
        self.result = None
        self.variables = variables
        self.stats = None
        self.entropy = None

    @abstractmethod
    def compute(self) -> pd.DataFrame:
        """Método abstracto encargado de calcular la variable objetivo.

        Returns:
            pd.DataFrame: DataFrame indexado con las etiquetas de entrenamiento/test.
        """
        pass

    def info(self):
        """Muestra en consola un reporte con las estadísticas y la entropía de la etiqueta."""
        print("-" * 20)
        print(f"Target: {self.name}")

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
        """Devuelve el DataFrame de datos original."""
        return self.data

    def get_result(self) -> pd.DataFrame:
        """Devuelve el DataFrame con los resultados calculados del Target."""
        return self.result

    def get_signal(self, col: str = None) -> pd.Series:
        """
        Extrae una columna de la variable objetivo como una Serie de Pandas.

        Args:
            col (str, optional): Nombre de la columna objetivo. Defaults to None.

        Returns:
            pd.Series: Serie temporal con las etiquetas de la columna seleccionada.

        Raises:
            RuntimeError: Si la variable objetivo no ha sido calculada previamente.
            ValueError: Si la columna no se especifica ante múltiples resultados o si no existe.
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
        Calcula la entropía de Shannon normalizada para medir la incertidumbre del Target.

        Una entropía cercana a 1 implica una distribución uniforme de las etiquetas (máxima incertidumbre),
        mientras que valores cercanos a 0 indican desbalances drásticos o alta predictibilidad.

        Returns:
            dict: Diccionario por columna con sus respectivos ratios de entropía.
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

            # Determinación bayesiana del número de contenedores según tamaño de muestra
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
        """Calcula estadísticas descriptivas básicas sobre las etiquetas calculadas.

        Returns:
            dict: Diccionario por columna conteniendo métricas de tendencia central y dispersión.
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


class NormalizedFutureReturn(Target):
    """
    Retorno Futuro Normalizado por Volatilidad.

    Calcula el rendimiento implícito hacia adelante (look-ahead) en un horizonte dado
    y lo divide por una métrica de volatilidad actual (ATR o Desviación Estándar). Esto
    garantiza que la variable objetivo sea homogénea y comparable a lo largo del tiempo,
    eliminando el sesgo de regímenes alternantes de mercado (alta/baja volatilidad).
    """

    def __init__(self, data: pd.DataFrame, horizon: int = 1, method: str = "atr", window: int = 14):
        """Inicializa la clase con los parámetros de normalización y horizonte.

        Args:
            data (pd.DataFrame): DataFrame con precios de mercado.
            horizon (int, optional): Número de barras en el futuro para medir el retorno. Defaults to 1.
            method (str, optional): Método de normalización, acepta 'atr' o 'std'. Defaults to "atr".
            window (int, optional): Ventana de lookback para el cálculo de la volatilidad. Defaults to 14.
        """
        super().__init__(
            data, 
            name="NormalizedFutureReturn", 
            variables={
                "horizon": horizon,
                "method": method,
                "window": window
            }
        )
        self.horizon = horizon
        self.method = method
        self.window = window

    def compute(self) -> pd.DataFrame:
        close = self.data["close"]
        high = self.data["high"]
        low = self.data["low"]

        # IMPORTANTE: Desplazamiento hacia el futuro (Look-ahead). 
        # Este cálculo introduce intencionalmente "Data Leakage" porque representa lo que el modelo
        # debe aprender a predecir a partir de la información del tiempo actual.
        future_return = (close.shift(-self.horizon) - close)

        # Selección de la métrica de normalización para desestacionalizar los retornos
        if self.method == "atr":
            # Usa el Average True Range considerando gaps de apertura e intradía
            atr = talib.ATR(
                high,
                low,
                close,
                timeperiod=self.window
            )
            denom = atr

        elif self.method == "std":
            # Usa la Desviación Estándar muestral del precio de cierre
            denom = talib.STDDEV(close, timeperiod=self.window, nbdev=1)

        else:
            raise ValueError("method must be 'atr' or 'std'")

        # Construcción final del target transformado a un ratio adimensional
        self.result = pd.DataFrame(
            {f"nfr_{self.horizon}_{self.method}_{self.window}": future_return / denom}, 
            index=self.data.index
        )

        return self.result