import copy
import io
import sys
import numpy as np
import pandas as pd

from modelos.models import Model
from .thresholds import ThresholdEvaluator
from .mcpt import MonteCarloPT


class WalkForwardEvaluator:
    def __init__(self, 
                 data: pd.DataFrame, 
                 train_window: int, 
                 test_window: int, 
                 step: int=None, 
                 expanding: bool=False
                 ):
        self.data = data.copy()

        self.train_window = train_window
        self.test_window  = test_window
        self.step         = test_window if step is None else step
        self.expanding    = expanding

        self.n      = len(self.data)
        self.splits = self._build_splits()

        self.returns = np.log(self.data["close"].shift(-1) / self.data["close"])
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
                  n_mcpt: int = 200, 
                  min_kept: int = 300, 
                  seed: int = 42):
        if not isinstance(model, Model):
            raise TypeError("model must be an instance of Model")

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

            # Fresh copy of the model for each fold
            current_model = copy.deepcopy(model)
            current_model.fit(X_train, y_train)

            # Optimizar umbral con predicciones sobre train
            y_pred_train = pd.Series(
                current_model.predict(X_train),
                index=X_train.index,
                name="y_pred_train"
            )
            self.te.prepare(y_pred_train)
            opt = self.te.find_optimized_threshold(min_kept=min_kept)

            # Evaluar ambos umbrales sobre test
            y_pred_test = pd.Series(
                current_model.predict(X_test),
                index=X_test.index,
                name="y_pred_test"
            )
            eval_high = self.te.evaluate_threshold(y_pred_test, opt["high_thresh"])
            eval_low  = self.te.evaluate_threshold(y_pred_test, opt["low_thresh"])

            fold_results.append({
                "fold":                    fold,
                "train_start":             train_start,
                "train_end":               train_end,
                "test_start":              test_start,
                "test_end":                test_end,
                "train_size":              train_end - train_start,
                "test_size":               test_end - test_start,
                "model":                   current_model,
                "pf_train_high":           opt["pf_high"],
                "pf_train_low":            opt["pf_low"],
                "high_thresh":             opt["high_thresh"],
                "low_thresh":              opt["low_thresh"],
                "pf_test_long_above":      eval_high["pf_long_above"],
                "pf_test_short_above":     eval_high["pf_short_above"],
                "pf_test_long_below":      eval_low["pf_long_below"],
                "pf_test_short_below":     eval_low["pf_short_below"],
            })

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

    def run_indicator(self, signal: pd.Series, mcpt: bool = False, n_mcpt: int = 200, min_kept: int = 300, seed: int = 42):
        self._indicator_signal = signal.name or "signal"
        fold_results = []

        for fold, (train_start, train_end, test_start, test_end) in enumerate(self.splits):

            signal_train = signal.iloc[train_start:train_end]
            signal_test  = signal.iloc[test_start:test_end]

            # Optimizar umbral en train
            self.te.prepare(signal_train)
            opt = self.te.find_optimized_threshold(min_kept=min_kept)

            # MCPT opcional en train
            p_value_high = None
            p_value_low  = None
            if mcpt:
                _mc = MonteCarloPT(
                    self.data.iloc[train_start:train_end].reset_index(drop=True),
                    seed=seed
                )
                signal_fold = signal_train.reset_index(drop=True).rename(signal_train.name)

                res_high = _mc.mcpt_threshold(signal_fold, n_test=n_mcpt, verbose=False)
                res_low  = _mc.mcpt_threshold(signal_fold, n_test=n_mcpt, flip_sign=True, verbose=False)

                p_value_high = res_high["p_value"]
                p_value_low  = res_low["p_value"]
            
            # Evaluar umbrales en test
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
        if not hasattr(self, "fold_results"):
            raise RuntimeError("Call run_model() or run_indicator() first")

        df_metrics = pd.DataFrame(self.fold_results)
        source     = getattr(self, "fold_results_source", "model")
        has_mcpt   = "p_value_high" in df_metrics.columns

        print("=" * 60)
        print("WALK-FORWARD EVALUATOR SUMMARY")
        print("=" * 60)

        # Configuracion
        print("\n--- Configuracion ---")
        print(f"  Mode:         {'expanding' if self.expanding else 'rolling'}")
        print(f"  Train window: {self.train_window}")
        print(f"  Test window:  {self.test_window}")
        print(f"  Step:         {self.step}")
        print(f"  Total folds:  {len(self.fold_results)}")
        print(f"  Total obs:    {self.n}")

        # Datos
        print("\n--- Datos ---")
        if source == "model":
            print(f"  Target:   {self.target}")
            print(f"  Features: {', '.join(self.features)}")
            print("\n--- Modelo ---")
            print(f"  {self.fold_results[0]['model'].name}")
        else:
            print(f"  Indicador: {self._indicator_signal}")
            if has_mcpt:
                print(f"  MCPT:      si")

        # Metricas agregadas
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

        # Detalle por fold
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