import copy
import numpy as np
import pandas as pd

from .thresholds import ThresholdEvaluator


class MonteCarloPT:
    """
    Ejecuta pruebas de permutación de Monte Carlo (MCPT) para validar estrategias.

    Evalúa si el rendimiento (Profit Factor) obtenido mediante la optimización
    de umbrales de señales o el entrenamiento de modelos es estadísticamente
    significativo o si puede ser replicado fácilmente por el azar.
    """

    def __init__(self, data: pd.DataFrame, seed: int = 42):
        self.data = data.copy()
        self.returns = np.log(self.data["close"].shift(-1) / self.data["close"])
        
        # Inicialización del evaluador de umbrales con los retornos calculados
        self.te = ThresholdEvaluator(self.returns)
        self.results_mc = None
        self.real_result = None
        self.work_return = None
        self.rng = np.random.default_rng(seed)

    def _score(self, result: dict) -> float:
        return max(result["pf_high"], result["pf_low"])

    def mcpt_threshold(self, signal: pd.Series, n_test: int = 100, min_kept: int = 300,
                        flip_sign: bool = False, verbose: bool = False) -> dict:
        """
        Realiza la prueba MCPT permutando los retornos asociados a una señal estática.

        Mantiene la señal original intacta pero desordena aleatoriamente los retornos de trabajo
        en cada iteración. Esto permite validar si la combinación de la señal con el mercado
        genera una ventaja real o si la optimización del umbral genera sobreajuste.

        Args:
            signal (pd.Series): Serie temporal con los valores del indicador o predicciones.
            n_test (int, optional): Número de permutaciones de Monte Carlo. Defaults to 100.
            min_kept (int, optional): Cantidad mínima de muestras requeridas tras el umbral. Defaults to 300.
            flip_sign (bool, optional): Invierte el signo de la señal si es necesario. Defaults to False.
            verbose (bool, optional): Si es True, imprime el resumen por consola. Defaults to False.

        Returns:
            dict: Resultados del test incluyendo el p-value y distribución de scores simulados.
        """
        self.signal_name = signal.name
        
        # Llamada al objeto ThresholdEvaluator: Carga de datos base y obtención de retornos de trabajo
        self.work_return = self.te.prepare(signal, flip_sign=flip_sign).get_work_returns()

        mc_scores   = []
        mc_high_thr = []
        mc_low_thr  = []

        # Llamada al objeto ThresholdEvaluator: Optimización de umbrales sobre la serie real
        real       = self.te.find_optimized_threshold(min_kept=min_kept)
        real_score = self._score(real)

        for _ in range(n_test):
            # Llamada al objeto ThresholdEvaluator: Inyección de retornos permutados aleatoriamente
            self.te.set_work_returns(self.rng.permutation(self.work_return))
            
            # Llamada al objeto ThresholdEvaluator: Re-optimización de umbrales bajo la serie aleatorizada
            res = self.te.find_optimized_threshold(min_kept=min_kept)

            mc_scores.append(self._score(res))
            mc_high_thr.append(res["high_thresh"])
            mc_low_thr.append(res["low_thresh"])

        mc_scores   = np.array(mc_scores)
        mc_high_thr = np.array(mc_high_thr)
        mc_low_thr  = np.array(mc_low_thr)
        p_value     = np.mean(mc_scores >= real_score)

        self.results_mc = {
            "real":                   real,
            "real_score":             real_score,
            "n_test":                 n_test,
            "mc_mean_score":          mc_scores.mean(),
            "mc_std_score":           mc_scores.std(),
            "mc_mean_high_threshold": mc_high_thr.mean(),
            "mc_mean_low_threshold":  mc_low_thr.mean(),
            "p_value":                p_value,
            "mc_distribution":        mc_scores,
        }

        if verbose:
            self.summary()

        return self.results_mc

    def mcpt_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                   n_test: int = 100, min_kept: int = 300, flip_sign: bool = False,
                   verbose: bool = False) -> dict:
        """
        Realiza la prueba MCPT destruyendo la relación X -> y mediante el reentrenamiento del modelo.

        En cada simulación, desordena aleatoriamente la variable objetivo (y_train),
        reajusta por completo una copia limpia del modelo y genera nuevas predicciones. Evalúa
        si la capacidad de aprendizaje del algoritmo encuentra patrones donde solo hay ruido.

        Args:
            model: Instancia u objeto del modelo predictivo (debe soportar fit y predict).
            X_train (pd.DataFrame): Matriz de características para el entrenamiento.
            y_train (pd.Series): Serie temporal con la variable objetivo real.
            n_test (int, optional): Número de iteraciones con target permutado. Defaults to 100.
            min_kept (int, optional): Muestras mínimas permitidas tras aplicar umbrales. Defaults to 300.
            flip_sign (bool, optional): Indica si se invierte el signo de la predicción. Defaults to False.
            verbose (bool, optional): Muestra el desglose de resultados si es True. Defaults to False.

        Returns:
            dict: Métricas de validación cruzada por Monte Carlo y p-value del modelo.
        """
        self.signal_name = getattr(model, "name", type(model).__name__)

        real_model = copy.deepcopy(model)
        
        # Llamada al objeto model: Entrenamiento con los datos reales intactos
        real_model.fit(X_train, y_train)
        
        # Llamada al objeto model: Generación de predicciones reales fuera de muestra (o train)
        y_pred_real = pd.Series(
            real_model.predict(X_train),
            index=X_train.index,
            name="y_pred_real"
        )

        # Llamada al objeto ThresholdEvaluator: Carga de las predicciones reales
        self.te.prepare(y_pred_real, flip_sign=flip_sign)
        
        # Llamada al objeto ThresholdEvaluator: Búsqueda del mejor umbral con el modelo real
        real       = self.te.find_optimized_threshold(min_kept=min_kept)
        real_score = self._score(real)

        mc_scores   = []
        mc_high_thr = []
        mc_low_thr  = []

        y_values = y_train.to_numpy()

        for _ in range(n_test):
            y_shuffled = pd.Series(
                self.rng.permutation(y_values),
                index=y_train.index,
                name="y_shuffled"
            )

            mc_model = copy.deepcopy(model)
            
            # Llamada al objeto model: Reentrenamiento exhaustivo utilizando la variable objetivo destruida
            mc_model.fit(X_train, y_shuffled)
            
            # Llamada al objeto model: Obtención de predicciones degradadas por la permutación
            y_pred_mc = pd.Series(
                mc_model.predict(X_train),
                index=X_train.index,
                name="y_pred_mc"
            )

            # Llamada al objeto ThresholdEvaluator: Preparación de la nueva señal sintética
            self.te.prepare(y_pred_mc, flip_sign=flip_sign)
            
            # Llamada al objeto ThresholdEvaluator: Evaluación del rendimiento máximo alcanzable por azar
            res = self.te.find_optimized_threshold(min_kept=min_kept)

            mc_scores.append(self._score(res))
            mc_high_thr.append(res["high_thresh"])
            mc_low_thr.append(res["low_thresh"])

        mc_scores   = np.array(mc_scores)
        mc_high_thr = np.array(mc_high_thr)
        mc_low_thr  = np.array(mc_low_thr)
        p_value     = np.mean(mc_scores >= real_score)

        self.results_mc = {
            "real":                   real,
            "real_score":             real_score,
            "n_test":                 n_test,
            "mc_mean_score":          mc_scores.mean(),
            "mc_std_score":           mc_scores.std(),
            "mc_mean_high_threshold": mc_high_thr.mean(),
            "mc_mean_low_threshold":  mc_low_thr.mean(),
            "p_value":                p_value,
            "mc_distribution":        mc_scores,
        }

        if verbose:
            self.summary()

        return self.results_mc

    def summary(self):
        """
        Imprime en consola los resultados estadísticos resumidos de la prueba MCPT.

        Muestra de forma clara los estadísticos básicos de la distribución simulada
        (media, desviación estándar) junto al score real y el p-value final.
        """
        if self.results_mc is None:
            print("Run mcpt_threshold or mcpt_model first")
            return

        r = self.results_mc

        print(f"\n--- MONTECARLO TEST N: {r['n_test']} ---")
        print(f"Signal/Model: {self.signal_name}")
        print(f"Real score: {r['real_score']:.4f}")
        print(f"MC mean:    {r['mc_mean_score']:.4f}")
        print(f"MC std:     {r['mc_std_score']:.4f}")
        print(f"Mean high threshold: {r['mc_mean_high_threshold']:.4f}")
        print(f"Mean low threshold:  {r['mc_mean_low_threshold']:.4f}")
        print(f"P-value:    {r['p_value']:.4f}")