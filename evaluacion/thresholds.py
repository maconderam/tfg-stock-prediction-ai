import numpy as np
import pandas as pd


class ThresholdEvaluator:
    """
    Evaluador y optimizador de umbrales basado en el Profit Factor.

    Permite analizar y encontrar los niveles críticos (umbrales superiores e
    inferiores) de una serie de señales con respecto a sus retornos, buscando
    maximizar la rentabilidad acumulada in-sample (train).
    """

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.fracs = np.array([0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50,
                               0.40, 0.30, 0.20, 0.10, 0.05, 0.01])
        self.optimized_threshold = {}

    @staticmethod
    def _pf(win, loss) -> float:
        """Calcula el Profit Factor de forma segura frente a divisiones por cero."""
        if loss == 0:
            return np.inf
        return win / loss
        
    def _print_thresholds(self, thr, nbin, gt_long, gt_short, lw_long, lw_short):
        """Formatea e imprime en consola una fila de la evaluación de umbrales."""
        def fmt(x):
            if np.isinf(x):
                return "inf"
            if np.isnan(x):
                return "nan"
            return f"{x:.3f}"

        print(
            f"{thr:10.4f} | "
            f"{self.fracs[nbin]:.2f} | "
            f"{fmt(gt_long):>10} | "
            f"{fmt(gt_short):>10} | "
            f"{fmt(lw_long):>10} | "
            f"{fmt(lw_short):>10}"
        )

    def _check_prepared(self):
        """Valida que los datos de trabajo hayan sido inicializados correctamente."""
        if not hasattr(self, "work_signal") or not hasattr(self, "work_return"):
            raise RuntimeError("Call prepare() before find_optimized_threshold()")

    def _align(self, signal):
        """Alinea una serie temporal de señal con los retornos de la clase."""
        mask = signal.notna() & self.returns.notna()
        return signal[mask].to_numpy(), self.returns[mask].to_numpy()

    def get_work_returns(self):
        """Devuelve el vector actual de retornos de trabajo ordenados."""
        try:
            return self.work_return
        except:
            return 
    
    def set_work_returns(self, work_return):
        """Asigna un nuevo vector de retornos de trabajo (útil para MCPT)."""
        self.work_return = work_return

    def prepare(self, signal, flip_sign=False):
        """
        Alinea, filtra y ordena los datos de entrada por el valor de la señal.

        Esta preparación optimiza la velocidad del algoritmo de búsqueda lineal,
        indexando de menor a mayor para permitir un cálculo acumulativo eficiente.

        Args:
            signal (pd.Series): Serie de señales temporales o predicciones numéricas.
            flip_sign (bool, optional): Si es True, invierte el signo de la señal. Defaults to False.

        Returns:
            self: Devuelve la propia instancia con los vectores internos configurados.

        Raises:
            ValueError: Si tras la alineación no quedan observaciones válidas disponibles.
        """
        returns = self.returns

        mask = signal.notna() & returns.notna()

        signal = signal[mask].to_numpy()
        returns = returns[mask].to_numpy()

        if flip_sign:
            signal = -signal

        idx = np.argsort(signal)

        self.work_signal = signal[idx]
        self.work_return = returns[idx]

        self.n = len(self.work_signal)

        if self.n == 0:
            raise ValueError("No valid observations")

        return self
    
    def evaluate_threshold(self, signal, threshold) -> dict:
        """
        Evalúa las métricas de rendimiento por encima y por debajo de un umbral fijo.

        Args:
            signal (pd.Series): Serie temporal con los valores del indicador a evaluar.
            threshold (float): Valor numérico del umbral de corte.

        Returns:
            dict: Diccionario con el Profit Factor para posiciones largas y cortas,
                tanto para los datos que superan el umbral como para los que no.
        """
        signal, returns = self._align(signal)

        mask_above = signal >= threshold
        mask_below = ~mask_above

        r_above = returns[mask_above]
        r_below = returns[mask_below]

        win_above = r_above[r_above > 0].sum()
        lose_above = -r_above[r_above < 0].sum()

        win_below = r_below[r_below > 0].sum()
        lose_below = -r_below[r_below < 0].sum()

        return {
            "pf_long_above": self._pf(win_above, lose_above),
            "pf_short_above": self._pf(lose_above, win_above),
            "pf_long_below": self._pf(win_below, lose_below),
            "pf_short_below": self._pf(lose_below, win_below)
        }

    def evaluate_all_thresholds(self, signal):
        """
        Imprime una tabla comparativa evaluando los percentiles de la señal provista.

        Args:
            signal (pd.Series): Serie temporal del indicador a testear.
        """
        signal, returns = self._align(signal)

        thresholds = np.quantile(signal, self.fracs)
        nbin = 0
        print(
            " threshold | frac | long_above | short_above | long_below | short_below"
        )
        print("-"*72)
        for thr in thresholds:
            mask_above = signal >= thr
            mask_below = ~mask_above

            r_above = returns[mask_above]
            r_below = returns[mask_below]

            win_above = r_above[r_above > 0].sum()
            lose_above = -r_above[r_above < 0].sum()

            win_below = r_below[r_below > 0].sum()
            lose_below = -r_below[r_below < 0].sum()

            self._print_thresholds(thr, nbin, 
                                  self._pf(win_above, lose_above), 
                                  self._pf(lose_above, win_above),
                                  self._pf(win_below, lose_below),
                                  self._pf(lose_below, win_below)
                                  )
            nbin += 1

    def find_optimized_threshold(self, min_kept=300) -> dict:
        """
        Encuentra linealmente los umbrales óptimos para operar en Long y en Short.

        Aprovecha que los datos internos están ordenados por señal para realizar un escaneo
        en tiempo $O(n)$, recalculando el Profit Factor al desplazar la frontera de decisión.

        Args:
            min_kept (int, optional): Número mínimo de muestras que deben quedar retenidas
                para que un umbral sea considerado válido. Defaults to 300.

        Returns:
            dict: Resultados de la optimización que contiene los mejores umbrales encontrados
                (`high_thresh` y `low_thresh`) junto a sus respectivos Profit Factors.
        """
        self._check_prepared()
        n = self.n
        work_signal = self.work_signal
        work_return = self.work_return

        if min_kept < 1:
            min_kept = 1

        mask = work_return > 0

        win_above = work_return[mask].sum()
        lose_above = -work_return[~mask].sum()

        win_below = 0.0
        lose_below = 0.0

        pf_all = self._pf(win_above, lose_above)

        best_high_pf = pf_all
        best_high_index = 0

        best_low_pf = -1.0
        best_low_index = n - 1

        # Escaneo lineal eficiente acumulando ganancias y pérdidas transicionadas
        for i in range(n - 1):
            r = work_return[i]

            if r > 0:
                win_above -= r
                lose_below += r
            else:
                lose_above += r
                win_below -= r

            # Evita segmentar muestras que comparten el mismo valor exacto de señal
            if work_signal[i + 1] == work_signal[i]:
                continue

            # Optimización de umbral superior (Long si la señal es alta)
            if n - i - 1 >= min_kept:
                pf_high = self._pf(win_above, lose_above)
                if pf_high > best_high_pf:
                    best_high_pf = pf_high
                    best_high_index = i + 1

            # Optimización de umbral inferior (Short si la señal es baja)
            if i + 1 >= min_kept:
                pf_low = self._pf(win_below, lose_below)
                if pf_low > best_low_pf:
                    best_low_pf = pf_low
                    best_low_index = i + 1

        self.optimized_threshold = {
            "pf_all": pf_all,
            "high_thresh": work_signal[best_high_index],
            "pf_high": best_high_pf,
            "low_thresh": work_signal[best_low_index],
            "pf_low": best_low_pf
        }

        return self.optimized_threshold