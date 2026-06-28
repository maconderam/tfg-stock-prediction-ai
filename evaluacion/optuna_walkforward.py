import copy
import numpy as np
import pandas as pd
import optuna
from optuna.logging import set_verbosity, WARNING

from modelos.models import Model
from .thresholds import ThresholdEvaluator
from .mcpt import MonteCarloPT


class OptunaWalkForward:
    """Walk-forward donde la selección de features e hiperparámetros se
    re-optimiza de forma independiente en cada fold, usando solo el train
    de ese fold (sin ver nunca el test del fold, ni el de ningún otro).

    A diferencia de ModelOptimizer (que busca una única combinación fija
    de features/hiperparámetros para todo el periodo histórico), aquí cada
    fold puede terminar usando un conjunto de features y unos hiperparámetros
    distintos, adaptados a su propio régimen temporal.

    Para evitar que Optuna vea el test real del fold durante la búsqueda,
    el train de cada fold se subdivide en:
      - inner_train: primer (1 - val_frac) % del train del fold.
      - inner_val:   último val_frac % del train del fold.

    Cada trial de Optuna se entrena con inner_train y se evalúa (mediante
    el profit factor del umbral óptimo) sobre inner_val. El trial ganador
    se reentrena con el train completo del fold y se evalúa, por fin, en
    el test real de ese fold.

    Args:
        data: DataFrame completo con features, target y close.
        model_builder: Función que recibe un dict de hiperparámetros y
            devuelve una instancia de Model lista para entrenar (sin fit).
        feature_pool: Lista de columnas candidatas a feature.
        target: Nombre de la columna objetivo.
        k_features: Número fijo de features, o tupla (low, high) para que
            Optuna decida también cuántas usar en cada trial.
        param_space: Igual formato que en ModelOptimizer:
            {nombre: (tipo, low, high[, step])} o {nombre: ("categorical", [...])}.
        train_window: Tamaño de la ventana de entrenamiento del walk-forward externo.
        test_window: Tamaño de la ventana de test del walk-forward externo.
        val_frac: Fracción del train de cada fold reservada como validación
            interna para las búsquedas de Optuna (por defecto 0.20).
        inner_metric: Métrica usada para evaluar cada trial sobre inner_val.
            Soportado: "pf_high" (profit factor del umbral óptimo, dirección long).
        n_trials_per_fold: Número de trials de Optuna a ejecutar en cada fold.
        step: Step del walk-forward externo (None = test_window).
        expanding: Modo expanding o rolling del walk-forward externo.
        min_kept: min_kept pasado a ThresholdEvaluator.find_optimized_threshold.
        mcpt: Si True, ejecuta un MCPT en modo "signal" (sin reentrenar) sobre
            el modelo final ya elegido por Optuna en cada fold, una sola vez.
            No se aplica dentro de los trials internos de Optuna por coste
            computacional (sería entrenar miles de veces extra).
        n_mcpt: Número de permutaciones del MCPT si mcpt=True.
        seed: Semilla para reproducibilidad.
        verbose: Si True, imprime el progreso de cada fold y cada trial.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_builder,
        feature_pool: list,
        target: str,
        k_features,
        param_space: dict,
        train_window: int,
        test_window: int,
        val_frac: float = 0.20,
        inner_metric: str = "pf_high",
        n_trials_per_fold: int = 30,
        step: int = None,
        expanding: bool = False,
        min_kept: int = 300,
        mcpt: bool = False,
        n_mcpt: int = 200,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.data         = data
        self.model_builder = model_builder
        self.feature_pool  = feature_pool
        self.target        = target
        self.k_features    = k_features
        self.param_space   = param_space

        self.train_window  = train_window
        self.test_window   = test_window
        self.step          = test_window if step is None else step
        self.expanding     = expanding

        self.val_frac          = val_frac
        self.inner_metric      = inner_metric
        self.n_trials_per_fold = n_trials_per_fold
        self.min_kept           = min_kept
        self.mcpt               = mcpt
        self.n_mcpt             = n_mcpt
        self.seed               = seed
        self.verbose             = verbose

        self.n      = len(data)
        self.splits = self._build_splits()

        self.fold_results = None

        if not self.verbose:
            set_verbosity(WARNING)

    # ------------------------------------------------------------------
    # Construcción de splits del walk-forward externo
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Sugerencias de Optuna (features + hiperparámetros)
    # ------------------------------------------------------------------

    def _suggest_features(self, trial: optuna.Trial) -> list:
        """Selecciona features distintas del pool, igual que en ModelOptimizer."""
        if isinstance(self.k_features, (tuple, list)):
            low, high = self.k_features[0], self.k_features[1]
            k = trial.suggest_int("k_features", low, high)
        else:
            k = self.k_features

        available = list(self.feature_pool)
        chosen = []

        for i in range(k):
            idx = trial.suggest_int(f"feature_idx_{i}", 0, len(available) - 1)
            chosen.append(available.pop(idx))

        return chosen

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """Construye el diccionario de hiperparámetros a partir de param_space."""
        params = {}

        for name, spec in self.param_space.items():
            kind = spec[0]

            if kind == "int":
                low, high = spec[1], spec[2]
                step = spec[3] if len(spec) > 3 else 1
                params[name] = trial.suggest_int(name, low, high, step=step)

            elif kind == "float":
                low, high = spec[1], spec[2]
                step = spec[3] if len(spec) > 3 else None
                params[name] = trial.suggest_float(name, low, high, step=step)

            elif kind == "loguniform":
                low, high = spec[1], spec[2]
                params[name] = trial.suggest_float(name, low, high, log=True)

            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])

            else:
                raise ValueError(f"Tipo de parámetro no soportado: {kind}")

        return params

    # ------------------------------------------------------------------
    # Optimización interna de un único fold
    # ------------------------------------------------------------------

    def _inner_objective(self, trial: optuna.Trial, X_inner_train, y_inner_train,
                          X_inner_val, te_inner) -> float:
        """Entrena con inner_train y evalúa con inner_val. Nunca toca el test real."""
        features = self._suggest_features(trial)
        params   = self._suggest_params(trial)

        model = self.model_builder(params)

        try:
            model.fit(X_inner_train[features], y_inner_train)
            y_pred_val = pd.Series(
                model.predict(X_inner_val[features]),
                index=X_inner_val.index,
                name="y_pred_inner_val"
            )

            te_inner.prepare(y_pred_val)
            opt = te_inner.find_optimized_threshold(min_kept=max(1, self.min_kept // 4))

        except Exception as e:
            trial.set_user_attr("error", str(e))
            return float("-inf")

        score = opt.get(self.inner_metric, float("-inf"))
        if np.isinf(score) or np.isnan(score):
            score = float("-inf") if score != float("inf") else 1e6

        trial.set_user_attr("features", features)
        trial.set_user_attr("params", params)

        return score

    def _optimize_fold(self, X_train, y_train) -> dict:
        """Lanza un mini-estudio de Optuna usando solo el train de un fold.

        Returns:
            dict con "features" y "params" de la mejor combinación encontrada.
        """
        n_train = len(X_train)
        n_val   = int(n_train * self.val_frac)

        X_inner_train = X_train.iloc[: n_train - n_val]
        y_inner_train = y_train.iloc[: n_train - n_val]
        X_inner_val   = X_train.iloc[n_train - n_val :]

        # ThresholdEvaluator interno: usa los returns del propio inner_val
        inner_data = self.data.loc[X_inner_val.index]
        te_inner = ThresholdEvaluator(
            np.log(inner_data["close"].shift(-1) / inner_data["close"])
        )

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(
            lambda trial: self._inner_objective(
                trial, X_inner_train, y_inner_train, X_inner_val, te_inner
            ),
            n_trials=self.n_trials_per_fold,
            show_progress_bar=False,
        )

        best = study.best_trial
        return {
            "features":    best.user_attrs.get("features"),
            "params":      best.user_attrs.get("params"),
            "inner_score": study.best_value,
        }

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self) -> list:
        """Ejecuta el walk-forward completo, re-optimizando en cada fold.

        Returns:
            Lista de dicts (uno por fold) con: features y parámetros
            elegidos para ese fold, y las métricas habituales de
            profit factor en train/test.
        """
        X = self.data[self.feature_pool]
        y = self.data[self.target]
        returns = np.log(self.data["close"].shift(-1) / self.data["close"])
        te = ThresholdEvaluator(returns)

        fold_results = []

        for fold, (train_start, train_end, test_start, test_end) in enumerate(self.splits):

            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test  = X.iloc[test_start:test_end]
            y_test  = y.iloc[test_start:test_end]

            if self.verbose:
                print(f"[Fold {fold}] optimizando con Optuna sobre train interno...")

            best = self._optimize_fold(X_train, y_train)
            features = best["features"]
            params   = best["params"]

            # Reentrena con el TRAIN COMPLETO del fold (inner_train + inner_val)
            # usando la mejor combinación encontrada, y evalúa en el test real.
            model = self.model_builder(params)
            model.fit(X_train[features], y_train)

            y_pred_train = pd.Series(
                model.predict(X_train[features]),
                index=X_train.index,
                name="y_pred_train"
            )
            te.prepare(y_pred_train)
            opt = te.find_optimized_threshold(min_kept=self.min_kept)

            # MCPT modo "signal" sobre el modelo final ya elegido (no reentrena)
            p_value_high = None
            p_value_low  = None
            if self.mcpt:
                _mc = MonteCarloPT(
                    self.data.iloc[train_start:train_end].reset_index(drop=True),
                    seed=self.seed
                )
                pred_fold = y_pred_train.reset_index(drop=True).rename(y_pred_train.name)

                res_high = _mc.mcpt_threshold(pred_fold, n_test=self.n_mcpt,
                                               min_kept=self.min_kept, verbose=False)
                res_low  = _mc.mcpt_threshold(pred_fold, n_test=self.n_mcpt,
                                               min_kept=self.min_kept, flip_sign=True, verbose=False)

                p_value_high = res_high["p_value"]
                p_value_low  = res_low["p_value"]

            y_pred_test = pd.Series(
                model.predict(X_test[features]),
                index=X_test.index,
                name="y_pred_test"
            )
            eval_high = te.evaluate_threshold(y_pred_test, opt["high_thresh"])
            eval_low  = te.evaluate_threshold(y_pred_test, opt["low_thresh"])

            row = {
                "fold":                fold,
                "train_start":         train_start,
                "train_end":           train_end,
                "test_start":          test_start,
                "test_end":            test_end,
                "features":            features,
                "params":              params,
                "inner_score":         best["inner_score"],
                "model":               model,
                "pf_train_high":       opt["pf_high"],
                "pf_train_low":        opt["pf_low"],
                "high_thresh":         opt["high_thresh"],
                "low_thresh":          opt["low_thresh"],
                "pf_test_long_above":  eval_high["pf_long_above"],
                "pf_test_short_above": eval_high["pf_short_above"],
                "pf_test_long_below":  eval_low["pf_long_below"],
                "pf_test_short_below": eval_low["pf_short_below"],
            }

            if self.mcpt:
                row["p_value_high"] = p_value_high
                row["p_value_low"]  = p_value_low

            fold_results.append(row)

            if self.verbose:
                p_str = f" p_hi={p_value_high:.3f}" if self.mcpt else ""
                print(
                    f"[Fold {fold}] features={len(features)} "
                    f"inner_score={best['inner_score']:.3f} "
                    f"PF test long above={eval_high['pf_long_above']:.3f}{p_str}"
                )

        self.fold_results = fold_results
        return fold_results

    def summary(self):
        """Imprime un resumen agregado y la evolución de features por fold."""
        if self.fold_results is None:
            raise RuntimeError("Llama a run() primero.")

        df = pd.DataFrame(self.fold_results)

        def safe_mean(col):
            return df[col].replace([np.inf, -np.inf], np.nan).mean()

        print("=" * 60)
        print("OPTUNA WALK-FORWARD SUMMARY")
        print("=" * 60)
        print(f"  Folds:                 {len(df)}")
        print(f"  Trials por fold:       {self.n_trials_per_fold}")
        print(f"  Val. interna (frac):   {self.val_frac}")
        print()
        print(f"  PF test long above medio:  {safe_mean('pf_test_long_above'):.4f}")
        print(f"  PF test short below medio: {safe_mean('pf_test_short_below'):.4f}")
        print(f"  Folds PF test long above >1: {(df['pf_test_long_above'] > 1).sum()} / {len(df)}")
        if "p_value_high" in df.columns:
            print(f"  P-value high medio:        {safe_mean('p_value_high'):.4f}")
            print(f"  P-value low medio:         {safe_mean('p_value_low'):.4f}")
            print(f"  Folds p_value_high < 0.05: {(df['p_value_high'] < 0.05).sum()} / {len(df)}")
        print()
        print("--- Features elegidas por fold ---")
        for _, row in df.iterrows():
            print(f"  Fold {row['fold']}: {row['features']}")
        print("=" * 60)