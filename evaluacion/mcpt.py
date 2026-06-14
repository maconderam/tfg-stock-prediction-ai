import numpy as np
import pandas as pd

from .thresholds import ThresholdEvaluator

class MonteCarloPT:
    def __init__(self, data: pd.DataFrame, seed: int = 42):
        self.data = data.copy()
        self.returns = np.log(self.data["close"].shift(-1) / self.data["close"])
        self.te = ThresholdEvaluator(self.returns)
        self.results_mc = None
        self.real_result = None
        self.work_return = None
        self.rng = np.random.default_rng(seed)

    def _score(self, result: dict) -> float:
        return max(result["pf_high"], result["pf_low"])

    def mcpt_threshold(self, signal: pd.Series, n_test: int = 100, min_kept: int = 300, flip_sign: bool = False, verbose: bool = False) -> dict:
        self.signal_name = signal.name
        self.work_return = self.te.prepare(signal, flip_sign=flip_sign).get_work_returns()

        mc_scores = []
        mc_high_thr = []
        mc_low_thr = []

        real = self.te.find_optimized_threshold(min_kept=min_kept)
        real_score = self._score(real)

        for _ in range(n_test):
            self.te.set_work_returns(self.rng.permutation(self.work_return))

            res = self.te.find_optimized_threshold(min_kept=min_kept)

            mc_scores.append(self._score(res))
            mc_high_thr.append(res["high_thresh"])
            mc_low_thr.append(res["low_thresh"])

        mc_scores = np.array(mc_scores)
        mc_high_thr = np.array(mc_high_thr)
        mc_low_thr = np.array(mc_low_thr)
        p_value = np.mean(mc_scores >= real_score)

        self.results_mc = {
            "real": real,
            "real_score": real_score,
            "n_test": n_test,

            "mc_mean_score": mc_scores.mean(),
            "mc_std_score": mc_scores.std(),

            "mc_mean_high_threshold": mc_high_thr.mean(),
            "mc_mean_low_threshold": mc_low_thr.mean(),

            "p_value": p_value,
            "mc_distribution": mc_scores
        }

        if verbose:
            self.summary()

        return self.results_mc
    
    def mcpt_model(self, model, features: list, target: str, n_test: int = 100) -> dict:
        raise NotImplementedError("MCPT for models is not implemented yet")

    # ---------------------------------------------------------
    def summary(self):
        if self.results_mc is None:
            print("Run mcpt_threshold first")
            return

        r = self.results_mc

        print(f"\n--- MONTECARLO THRESHOLD TEST N: {r['n_test']} ---")
        print(f"Indicator: {self.signal_name}")
        print(f"Real score: {r['real_score']:.4f}")
        print(f"MC mean:    {r['mc_mean_score']:.4f}")
        print(f"MC std:     {r['mc_std_score']:.4f}")
        print(f"Mean high threshold: {r['mc_mean_high_threshold']:.4f}")
        print(f"Mean low threshold:  {r['mc_mean_low_threshold']:.4f}")
        print(f"P-value:    {r['p_value']:.4f}")