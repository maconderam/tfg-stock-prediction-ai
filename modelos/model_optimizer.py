import copy
import numpy as np
import pandas as pd
import optuna

from modelos.models import Model
from evaluacion.walkforward import WalkForwardEvaluator


class ModelOptimizer:
    """Optimiza conjuntamente hiperparámetros y selección de features de un modelo
    usando Optuna, evaluando cada combinación mediante walk-forward completo.

    En cada trial, Optuna elige:
      - Un subconjunto de tamaño fijo `k_features` dentro de `feature_pool`.
      - Un valor para cada hiperparámetro definido en `param_space`.

    El modelo resultante se evalúa con un `WalkForwardEvaluator` y se agrega
    (media o mediana) la métrica elegida sobre todos los folds de test.

    Args:
        data: DataFrame completo con features, target y close.
        model_builder: Función que recibe un dict de hiperparámetros y
            devuelve una instancia de Model lista para entrenar (sin fit).
            Ejemplo: lambda params: SklearnModel(RandomForestRegressor(**params))
        feature_pool: Lista de columnas candidatas a feature.
        target: Nombre de la columna objetivo.
        k_features: Número fijo de features a seleccionar en cada trial,
            o tupla (low, high) para que Optuna decida también cuántas
            features usar en cada trial dentro de ese rango.
        param_space: Diccionario {nombre: (tipo, low, high)} o
            {nombre: (tipo, low, high, step)} para cada hiperparámetro.
            Tipos soportados: "int", "float", "loguniform", "categorical".
            Para "categorical" el formato es (tipo, [opciones]).
        train_window: Tamaño de la ventana de entrenamiento del walk-forward.
            Puede ser un int fijo, o una tupla (low, high) o (low, high, step)
            para que Optuna explore distintos valores en cada trial.
        test_window: Tamaño de la ventana de test del walk-forward.
            Mismo formato que train_window: int fijo o tupla para explorar.
        metric: Columna de fold_results a optimizar. Ejemplos:
            "pf_test_long_above", "pf_test_short_below", "pf_train_high",
            "composite_score" (calculado automáticamente por WalkForwardEvaluator
            combinando PF y p-value normalizados), "p_value_high".
        metric_agg: "mean" o "median" para agregar la métrica entre folds.
        direction: "maximize" o "minimize" (p.ej. minimize para p_value).
        mcpt: Si True, activa el MCPT en cada walk-forward para poder usar
            métricas basadas en p_value o composite_score.
        n_mcpt: Número de permutaciones del MCPT si mcpt=True.
        step: Step del walk-forward (None = test_window).
        expanding: Modo expanding o rolling del walk-forward. Puede ser un
            bool fijo, o True/False/"both" para que Optuna explore ambos modos
            (internamente se sugiere como categorical [True, False]).
        seed: Semilla para reproducibilidad.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_builder,
        feature_pool: list,
        target: str,
        k_features: int,
        param_space: dict,
        train_window,
        test_window,
        metric: str = "pf_test_long_above",
        metric_agg: str = "mean",
        direction: str = "maximize",
        mcpt: bool = False,
        n_mcpt: int = 100,
        step: int = None,
        expanding = False,
        seed: int = 42,
    ):
        max_k = k_features[1] if isinstance(k_features, (tuple, list)) else k_features
        if max_k > len(feature_pool):
            raise ValueError("k_features no puede ser mayor que el tamaño de feature_pool")
        if metric_agg not in ("mean", "median"):
            raise ValueError("metric_agg debe ser 'mean' o 'median'")

        self.data         = data
        self.model_builder = model_builder
        self.feature_pool  = feature_pool
        self.target        = target
        self.k_features    = k_features
        self.param_space   = param_space
        self.train_window  = train_window
        self.test_window   = test_window
        self.metric        = metric
        self.metric_agg    = metric_agg
        self.direction     = direction
        self.mcpt          = mcpt
        self.n_mcpt        = n_mcpt
        self.step          = step
        self.expanding     = expanding
        self.seed          = seed

        self.study           = None
        self.best_features   = None
        self.best_params     = None
        self.best_wf_params  = None
        self.best_fold_results = None

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _suggest_features(self, trial: optuna.Trial) -> list:
        """Selecciona k_features distintas del pool usando índices sugeridos por Optuna.

        Si self.k_features es un int fijo, se usa tal cual. Si es una tupla
        (low, high), Optuna decide también cuántas features usar en este
        trial concreto, dentro de ese rango.

        Se sugiere un índice por cada hueco a rellenar y se evita repetir
        features ya elegidas dentro del mismo trial.
        """
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

    def _suggest_walkforward_params(self, trial: optuna.Trial) -> dict:
        """Resuelve train_window, test_window y expanding para este trial.

        Si el atributo correspondiente es un int/bool fijo, se usa tal cual.
        Si es una tupla (low, high) o (low, high, step), se sugiere a Optuna
        un valor dentro de ese rango. Si expanding es "both", se sugiere
        como categorical entre True y False.
        """
        def resolve_int(name, value):
            if isinstance(value, (tuple, list)):
                low, high = value[0], value[1]
                step = value[2] if len(value) > 2 else 1
                return trial.suggest_int(name, low, high, step=step)
            return value

        train_window = resolve_int("train_window", self.train_window)
        test_window  = resolve_int("test_window", self.test_window)

        if self.expanding == "both":
            expanding = trial.suggest_categorical("expanding", [True, False])
        else:
            expanding = self.expanding

        return {
            "train_window": train_window,
            "test_window":  test_window,
            "expanding":    expanding,
        }

    def _objective(self, trial: optuna.Trial) -> float:
        """Función objetivo: entrena un walk-forward completo y devuelve la métrica agregada."""
        features = self._suggest_features(trial)
        params   = self._suggest_params(trial)
        wf_params = self._suggest_walkforward_params(trial)

        model = self.model_builder(params)

        wf = WalkForwardEvaluator(
            self.data,
            train_window=wf_params["train_window"],
            test_window=wf_params["test_window"],
            step=self.step,
            expanding=wf_params["expanding"],
        )

        try:
            fold_results, _ = wf.run_model(
                model,
                features=features,
                target=self.target,
                mcpt=self.mcpt,
                n_mcpt=self.n_mcpt,
                seed=self.seed,
            )
        except Exception as e:
            # Trials que fallan (p.ej. combinación de hiperparámetros inválida)
            # se penalizan en lugar de detener el estudio completo.
            trial.set_user_attr("error", str(e))
            return float("-inf") if self.direction == "maximize" else float("inf")

        df_folds = pd.DataFrame(fold_results)

        if self.metric not in df_folds.columns:
            raise KeyError(
                f"Métrica '{self.metric}' no está en fold_results. "
                f"Columnas disponibles: {list(df_folds.columns)}"
            )

        values = df_folds[self.metric].replace([np.inf, -np.inf], np.nan).dropna()

        if values.empty:
            return float("-inf") if self.direction == "maximize" else float("inf")

        score = values.mean() if self.metric_agg == "mean" else values.median()

        # Guardamos info extra del trial para poder inspeccionarlo después
        trial.set_user_attr("features", features)
        trial.set_user_attr("fold_results", fold_results)

        return score

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def optimize(self, n_trials: int = 100, show_progress: bool = True) -> optuna.Study:
        """Lanza la búsqueda de Optuna.

        Args:
            n_trials: Número de combinaciones features+hiperparámetros a probar.
            show_progress: Si True, muestra la barra de progreso de Optuna.

        Returns:
            El objeto optuna.Study con el historial completo de trials.
        """
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(direction=self.direction, sampler=sampler)

        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
        )

        best_trial = self.study.best_trial
        self.best_features     = best_trial.user_attrs.get("features")
        self.best_params        = {
            k: v for k, v in best_trial.params.items()
            if not k.startswith("feature_idx_")
            and k not in ("train_window", "test_window", "expanding", "k_features")
        }
        self.best_wf_params     = {
            k: v for k, v in best_trial.params.items()
            if k in ("train_window", "test_window", "expanding")
        }
        self.best_fold_results  = best_trial.user_attrs.get("fold_results")

        return self.study

    def summary(self):
        """Imprime un resumen legible del mejor trial encontrado."""
        if self.study is None:
            raise RuntimeError("Llama a optimize() primero.")

        print("=" * 60)
        print("MODEL OPTIMIZER — MEJOR RESULTADO")
        print("=" * 60)
        print(f"  Métrica optimizada: {self.metric} ({self.metric_agg})")
        print(f"  Mejor valor:        {self.study.best_value:.4f}")
        print(f"  Trials ejecutados:  {len(self.study.trials)}")
        print("\n--- Mejores features ---")
        for f in self.best_features:
            print(f"  - {f}")
        print("\n--- Mejores hiperparámetros ---")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        if self.best_wf_params:
            print("\n--- Mejor configuración walk-forward ---")
            for k, v in self.best_wf_params.items():
                print(f"  {k}: {v}")
        print("=" * 60)

    def trials_dataframe(self) -> pd.DataFrame:
        """Devuelve el historial de trials como DataFrame, útil para análisis o gráficas."""
        if self.study is None:
            raise RuntimeError("Llama a optimize() primero.")
        return self.study.trials_dataframe()