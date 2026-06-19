import copy
import numpy as np
import pandas as pd

from modelos.models import Model
from .thresholds import ThresholdEvaluator
from .mcpt import MonteCarloPT


class WalkForwardEvaluator:
    """
    Evaluador de estrategias de trading utilizando validación cruzada Walk-Forward.

    Permite evaluar la robustez de modelos predictivos o indicadores técnicos a lo
    largo del tiempo mediante ventanas temporales deslizantes (rolling) o expansivas.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 train_window: int,
                 test_window: int,
                 step: int = None,
                 expanding: bool = False
                 ):
        self.data = data.copy()

        self.train_window = train_window
        self.test_window  = test_window
        self.step         = test_window if step is None else step
        self.expanding    = expanding

        self.n      = len(self.data)
        self.splits = self._build_splits()

        self.returns = np.log(self.data["close"].shift(-1) / self.data["close"])
        
        # Inicialización de ThresholdEvaluator para la gestión de umbrales óptimos
        self.te = ThresholdEvaluator(self.returns)

    def _build_splits(self):
        splits = []
        start = self.train_window

        while start + self.test_window <= self.n:
            train_start = 0 if self.expanding else start - self.train_window
            train_end   = start
            test_start  = start
            test_end    = start + self.test_window

            splits.append((train_start, train_end, test_start, test_end))
            start += self.step

        return splits

    def run_model(self,
                  model: Model,
                  features: list,
                  target: str,
                  mcpt: bool = False,
                  mcpt_mode: str = "signal",
                  n_mcpt: int = 200,
                  min_kept: int = 300,
                  seed: int = 42):
        """
        Ejecuta el análisis Walk-Forward entrenando y evaluando un modelo predictivo.

        Itera sobre cada fold temporal, entrena el modelo en la ventana de training,
        optimiza los umbrales de decisión y evalúa los resultados en la ventana de testing,
        con la opción de aplicar pruebas de permutación de Monte Carlo (MCPT) para validación.

        Args:
            model (Model): Instancia del modelo que implementa los métodos fit y predict.
            features (list): Lista con los nombres de las columnas de características (X).
            target (str): Nombre de la columna objetivo (y).
            mcpt (bool, optional): Indica si se activa el test de Monte Carlo por fold. Defaults to False.
            mcpt_mode (str, optional): "signal" para permutar predicciones de train o
                "retrain" para permutar y_train y reentrenar. Defaults to "signal".
            n_mcpt (int, optional): Número de permutaciones para el test MCPT. Defaults to 200.
            min_kept (int, optional): Cantidad mínima de muestras requeridas tras aplicar el umbral. Defaults to 300.
            seed (int, optional): Semilla para la generación de aleatoriedad. Defaults to 42.

        Returns:
            tuple: Contiene una lista de diccionarios con las métricas detalladas de cada fold
                y un pd.DataFrame con las predicciones fuera de muestra acumuladas.

        Raises:
            TypeError: Si el modelo no hereda o no es una instancia de la clase Model.
            ValueError: Si mcpt_mode no es una de las opciones válidas ('signal' o 'retrain').
        """
        if not isinstance(model, Model):
            raise TypeError("model must be an instance of Model")
        if mcpt_mode not in ("signal", "retrain"):
            raise ValueError("mcpt_mode must be 'signal' or 'retrain'")

        self.features = features
        self.target   = target

        X = self.data[features]
        y = self.data[target]

        fold_results    = []
        all_predictions = []

        for fold, (train_start, train_end, test_start, test_end) in enumerate(self.splits):

            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test  = X.iloc[test_start:test_end]
            y_test  = y.iloc[test_start:test_end]

            current_model = copy.deepcopy(model)
            # Llamada al objeto Model: Entrenamiento con los datos de la ventana actual
            current_model.fit(X_train, y_train)

            # Llamada al objeto Model: Generación de predicciones sobre el conjunto de entrenamiento
            y_pred_train = pd.Series(
                current_model.predict(X_train),
                index=X_train.index,
                name="y_pred_train"
            )
            # Llamada al objeto ThresholdEvaluator: Preparación de retornos asociados a las predicciones de train
            self.te.prepare(y_pred_train)
            # Llamada al objeto ThresholdEvaluator: Búsqueda de umbrales óptimos basados en Profit Factor
            opt = self.te.find_optimized_threshold(min_kept=min_kept)

            p_value_high = None
            p_value_low  = None
            if mcpt:
                # Inicialización del objeto MonteCarloPT para evaluar la significancia estadística del fold
                _mc = MonteCarloPT(
                    self.data.iloc[train_start:train_end].reset_index(drop=True),
                    seed=seed
                )

                if mcpt_mode == "signal":
                    pred_fold = y_pred_train.reset_index(drop=True).rename(y_pred_train.name)

                    # Llamada al objeto MonteCarloPT: Test de permutación rápido sobre las señales fijas
                    res_high = _mc.mcpt_threshold(pred_fold, n_test=n_mcpt, min_kept=min_kept, verbose=False)
                    res_low  = _mc.mcpt_threshold(pred_fold, n_test=n_mcpt, min_kept=min_kept, flip_sign=True, verbose=False)

                else:  # retrain
                    X_train_reset = X_train.reset_index(drop=True)
                    y_train_reset = y_train.reset_index(drop=True)

                    # Llamada al objeto MonteCarloPT: Test exhaustivo que reentrena el objeto Model por permutación
                    res_high = _mc.mcpt_model(current_model, X_train_reset, y_train_reset,
                                               n_test=n_mcpt, min_kept=min_kept, verbose=False)
                    res_low  = _mc.mcpt_model(current_model, X_train_reset, y_train_reset,
                                               n_test=n_mcpt, min_kept=min_kept, flip_sign=True, verbose=False)

                p_value_high = res_high["p_value"]
                p_value_low  = res_low["p_value"]

            # Llamada al objeto Model: Predicción fuera de muestra en la ventana de test
            y_pred_test = pd.Series(
                current_model.predict(X_test),
                index=X_test.index,
                name="y_pred_test"
            )
            # Llamada al objeto ThresholdEvaluator: Evaluación del rendimiento real usando los umbrales optimizados de train
            eval_high = self.te.evaluate_threshold(y_pred_test, opt["high_thresh"])
            eval_low  = self.te.evaluate_threshold(y_pred_test, opt["low_thresh"])

            row = {
                "fold":                fold,
                "train_start":         train_start,
                "train_end":           train_end,
                "test_start":          test_start,
                "test_end":            test_end,
                "train_size":          train_end - train_start,
                "test_size":           test_end - test_start,
                "model":               current_model,
                "pf_train_high":       opt["pf_high"],
                "pf_train_low":        opt["pf_low"],
                "high_thresh":         opt["high_thresh"],
                "low_thresh":          opt["low_thresh"],
                "pf_test_long_above":  eval_high["pf_long_above"],
                "pf_test_short_above": eval_high["pf_short_above"],
                "pf_test_long_below":  eval_low["pf_long_below"],
                "pf_test_short_below": eval_low["pf_short_below"],
            }

            if mcpt:
                row["p_value_high"] = p_value_high
                row["p_value_low"]  = p_value_low

            fold_results.append(row)

            for idx, y_true, y_pred in zip(X_test.index, y_test, y_pred_test):
                all_predictions.append({
                    "fold":   fold,
                    "index":  idx,
                    "y_true": y_true,
                    "y_pred": y_pred,
                })

        self.fold_results        = fold_results
        self.fold_results_source = "model"
        self.predictions_df      = pd.DataFrame(all_predictions).set_index("index")

        return fold_results, self.predictions_df

    def run_indicator(self, signal: pd.Series, mcpt: bool = False, n_mcpt: int = 200,
                       min_kept: int = 300, seed: int = 42):
        """
        Ejecuta el análisis Walk-Forward directamente sobre un indicador o señal precalculada.

        A diferencia de `run_model`, este método prescinde de la etapa de ajuste de un modelo
        y optimiza/evalúa los umbrales directamente sobre la serie temporal de la señal provista.

        Args:
            signal (pd.Series): Serie temporal del indicador o señal numérica a evaluar.
            mcpt (bool, optional): Indica si se activa la validación de Monte Carlo por fold. Defaults to False.
            n_mcpt (int, optional): Número de permutaciones para el test MCPT. Defaults to 200.
            min_kept (int, optional): Muestras mínimas retenidas tras la aplicación de umbrales. Defaults to 300.
            seed (int, optional): Semilla para garantizar reproducibilidad en las permutaciones. Defaults to 42.

        Returns:
            list: Lista de diccionarios que recopila las métricas obtenidas en cada fold temporal.
        """
        self._indicator_signal = signal.name or "signal"
        fold_results = []

        for fold, (train_start, train_end, test_start, test_end) in enumerate(self.splits):

            signal_train = signal.iloc[train_start:train_end]
            signal_test  = signal.iloc[test_start:test_end]

            # Llamada al objeto ThresholdEvaluator: Asignación de retornos para la señal de train actual
            self.te.prepare(signal_train)
            # Llamada al objeto ThresholdEvaluator: Optimización de límites superiores e inferiores
            opt = self.te.find_optimized_threshold(min_kept=min_kept)

            p_value_high = None
            p_value_low  = None
            if mcpt:
                # Inicialización del objeto MonteCarloPT para la prueba sobre indicadores
                _mc = MonteCarloPT(
                    self.data.iloc[train_start:train_end].reset_index(drop=True),
                    seed=seed
                )
                signal_fold = signal_train.reset_index(drop=True).rename(signal_train.name)

                # Llamada al objeto MonteCarloPT: Test de permutación directo sobre los valores de la señal
                res_high = _mc.mcpt_threshold(signal_fold, n_test=n_mcpt, verbose=False)
                res_low  = _mc.mcpt_threshold(signal_fold, n_test=n_mcpt, flip_sign=True, verbose=False)

                p_value_high = res_high["p_value"]
                p_value_low  = res_low["p_value"]

            # Llamada al objeto ThresholdEvaluator: Evaluación de la señal fuera de muestra usando los límites óptimos
            eval_high = self.te.evaluate_threshold(signal_test, opt["high_thresh"])
            eval_low  = self.te.evaluate_threshold(signal_test, opt["low_thresh"])

            row = {
                "fold":                fold,
                "train_start":         train_start,
                "train_end":           train_end,
                "test_start":          test_start,
                "test_end":            test_end,
                "train_size":          train_end - train_start,
                "test_size":           test_end - test_start,
                "pf_train_high":       opt["pf_high"],
                "pf_train_low":        opt["pf_low"],
                "high_thresh":         opt["high_thresh"],
                "low_thresh":          opt["low_thresh"],
                "pf_test_long_above":  eval_high["pf_long_above"],
                "pf_test_short_above": eval_high["pf_short_above"],
                "pf_test_long_below":  eval_low["pf_long_below"],
                "pf_test_short_below": eval_low["pf_short_below"],
            }

            if mcpt:
                row["p_value_high"] = p_value_high
                row["p_value_low"]  = p_value_low

            fold_results.append(row)

            print(f"Fold {fold:>2} | "
                  f"PF tr.hi: {opt['pf_high']:.3f} | "
                  f"PF tr.lo: {opt['pf_low']:.3f} | "
                  f"L.abv: {eval_high['pf_long_above']:.3f} | "
                  f"L.blw: {eval_low['pf_long_below']:.3f}"
                  + (f" | p_hi: {p_value_high:.3f} | p_lo: {p_value_low:.3f}" if mcpt else ""))

        self.fold_results        = fold_results
        self.fold_results_source = "indicator"
        return fold_results

    def summary(self):
        """
        Calcula y muestra en consola un reporte consolidado del rendimiento Walk-Forward.

        Agrega las métricas de todos los folds (medias de Profit Factor, tasas de degradación
        e indicadores de significancia estadística) proporcionando una visión global
        de la robustez de la estrategia analizada.

        Raises:
            RuntimeError: Si se intenta ejecutar sin haber corrido previamente `run_model` o `run_indicator`.
        """
        if not hasattr(self, "fold_results"):
            raise RuntimeError("Call run_model() or run_indicator() first")

        df_metrics = pd.DataFrame(self.fold_results)
        source     = getattr(self, "fold_results_source", "model")
        has_mcpt   = "p_value_high" in df_metrics.columns

        print("=" * 60)
        print("WALK-FORWARD EVALUATOR SUMMARY")
        print("=" * 60)

        print("\n--- Configuracion ---")
        print(f"  Mode:         {'expanding' if self.expanding else 'rolling'}")
        print(f"  Train window: {self.train_window}")
        print(f"  Test window:  {self.test_window}")
        print(f"  Step:         {self.step}")
        print(f"  Total folds:  {len(self.fold_results)}")
        print(f"  Total obs:    {self.n}")

        print("\n--- Datos ---")
        if source == "model":
            print(f"  Target:    {self.target}")
            print(f"  Features: {', '.join(self.features)}")
            print("\n--- Modelo ---")
            print(f"  {self.fold_results[0]['model'].name}")
        else:
            print(f"  Indicador: {self._indicator_signal}")
            if has_mcpt:
                print(f"  MCPT:      si")

        def safe_mean(col):
            return df_metrics[col].replace([np.inf], np.nan).mean()

        print("\n--- Metricas agregadas ---")
        print(f"  PF train high medio:         {safe_mean('pf_train_high'):.4f}")
        print(f"  PF train low medio:          {safe_mean('pf_train_low'):.4f}")
        print(f"  PF test long  above medio:   {safe_mean('pf_test_long_above'):.4f}")
        print(f"  PF test short above medio:   {safe_mean('pf_test_short_above'):.4f}")
        print(f"  PF test long  below medio:   {safe_mean('pf_test_long_below'):.4f}")
        print(f"  PF test short below medio:   {safe_mean('pf_test_short_below'):.4f}")
        print(f"  Degradacion high (train/test long above):  {safe_mean('pf_train_high') / safe_mean('pf_test_long_above'):.4f}")
        print(f"  Degradacion low  (train/test short below): {safe_mean('pf_train_low')  / safe_mean('pf_test_short_below'):.4f}")
        print(f"  Folds PF test long  above > 1: {(df_metrics['pf_test_long_above'] > 1).sum()} / {len(df_metrics)}")
        print(f"  Folds PF test short above > 1: {(df_metrics['pf_test_short_above'] > 1).sum()} / {len(df_metrics)}")
        print(f"  Folds PF test long  below > 1: {(df_metrics['pf_test_long_below'] > 1).sum()} / {len(df_metrics)}")
        print(f"  Folds PF test short below > 1: {(df_metrics['pf_test_short_below'] > 1).sum()} / {len(df_metrics)}")
        if has_mcpt:
            print(f"  P-value high medio:          {safe_mean('p_value_high'):.4f}")
            print(f"  P-value low  medio:          {safe_mean('p_value_low'):.4f}")
            print(f"  Folds p_value_high < 0.05:   {(df_metrics['p_value_high'] < 0.05).sum()} / {len(df_metrics)}")
            print(f"  Folds p_value_low  < 0.05:   {(df_metrics['p_value_low']  < 0.05).sum()} / {len(df_metrics)}")

        def fmt(v):
            return "inf" if np.isinf(v) else f"{v:.3f}"

        print("\n--- Detalle por fold ---")
        header = (f"{'Fold':>4} | {'Train':>12} | {'Test':>12} | "
                  f"{'PF tr.hi':>8} | {'PF tr.lo':>8} | "
                  f"{'L.abv':>6} | {'S.abv':>6} | {'L.blw':>6} | {'S.blw':>6} | "
                  f"{'Hi thr':>7} | {'Lo thr':>7}"
                  + (f" | {'p_hi':>5} | {'p_lo':>5}" if has_mcpt else ""))
        print(header)
        print("-" * len(header))
        for r in self.fold_results:
            line = (
                f"{r['fold']:>4} | "
                f"{r['train_start']:>5}-{r['train_end']:<5} | "
                f"{r['test_start']:>5}-{r['test_end']:<5} | "
                f"{fmt(r['pf_train_high']):>8} | "
                f"{fmt(r['pf_train_low']):>8} | "
                f"{fmt(r['pf_test_long_above']):>6} | "
                f"{fmt(r['pf_test_short_above']):>6} | "
                f"{fmt(r['pf_test_long_below']):>6} | "
                f"{fmt(r['pf_test_short_below']):>6} | "
                f"{r['high_thresh']:>7.4f} | "
                f"{r['low_thresh']:>7.4f}"
            )
            if has_mcpt:
                line += f" | {r['p_value_high']:>5.3f} | {r['p_value_low']:>5.3f}"
            print(line)
        print("=" * 60)