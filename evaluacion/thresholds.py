import numpy as np
import pandas as pd

class ThresholdEvaluator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.returns = np.log(
            self.data["close"].shift(-1) / self.data["close"]
        )
        self.fracs = np.array([0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50,
                               0.40, 0.30, 0.20, 0.10, 0.05, 0.01])
        self.optimized_threshold = {}

    @staticmethod
    def _pf(win, loss):
        if loss == 0:
            return np.inf
        return win / loss
        
    def _print_thresholds(self, thr, nbin, gt_long, gt_short, lw_long, lw_short):
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

    def get_work_returns(self):
        try:
            return self.work_return
        except:
            return 
    
    def set_work_returns(self, work_return):
        self.work_return = work_return

    def prepare(self, signal, flip_sign=False):
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

    def threshold_evaluation(self, signal):
        returns = self.returns

        mask = signal.notna() & returns.notna()
        signal = signal[mask].to_numpy()
        returns = returns[mask].to_numpy()

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

    def find_optimized_threshold(self, min_kept=300):
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

        for i in range(n - 1):
            r = work_return[i]

            if r > 0:
                win_above -= r
                lose_below += r
            else:
                lose_above += r
                win_below -= r

            if work_signal[i + 1] == work_signal[i]:
                continue

            if n - i - 1 >= min_kept:
                pf_high = self._pf(win_above, lose_above)
                if pf_high > best_high_pf:
                    best_high_pf = pf_high
                    best_high_index = i + 1

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