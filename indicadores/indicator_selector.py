import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

from .indicators import RSI, Stochastic, StochasticRSI, MACD, PriceIntensity
from evaluacion.mcpt import MonteCarloPT

# ---------------------------------------------------------------------------
# Default parameter grids for each indicator class
# ---------------------------------------------------------------------------
DEFAULT_GRIDS = {
    "RSI": {
        "window":        [10, 14, 20, 30],
        "smooth_window": [3, 5],
    },
    "Stochastic": {
        "window":        [10, 14, 20, 30],
        "smooth_window": [3, 5],
    },
    "StochasticRSI": {
        "rsi_window":    [14, 20, 30],
        "stoch_window":  [5, 10],
        "smooth_window": [3, 5],
    },
    "MACD": {
        "long_window":   [12, 20],
        "short_window":  [26, 40],
        "smooth_window": [9, 12],
    },
    "PriceIntensity": {
        "smooth_window": [5, 10, 20, 30],
    },
}

# Map name → class
_INDICATOR_CLASSES = {
    "RSI":            RSI,
    "Stochastic":     Stochastic,
    "StochasticRSI":  StochasticRSI,
    "MACD":           MACD,
    "PriceIntensity": PriceIntensity,
}


# ---------------------------------------------------------------------------
# IndicatorSelector
# ---------------------------------------------------------------------------
class IndicatorSelector:
    def __init__(
        self,
        data: pd.DataFrame,
        target=None,
        min_kepts: int = 300,
        n_mcpt: int = 200,
        p_threshold: float = 0.10,
        custom_grids: Optional[dict] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.data = data.copy()
        self.target = target
        self.min_kepts = min_kepts
        self.n_mcpt = n_mcpt
        self.p_threshold = p_threshold
        self.seed = seed
        self.verbose = verbose

        # Merge default grids with any user overrides
        self.grids = {k: dict(v) for k, v in DEFAULT_GRIDS.items()}
        if custom_grids:
            for name, params in custom_grids.items():
                if name in self.grids:
                    self.grids[name].update(params)
                else:
                    self.grids[name] = params

        self.results: list[dict] = []
        self.summary_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _param_combinations(grid: dict) -> list[dict]:
        """Return all combinations of a parameter grid as a list of dicts."""
        keys = list(grid.keys())
        values = list(grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _build_indicator(self, name: str, params: dict):
        """Instantiate an indicator by name with given params."""
        cls = _INDICATOR_CLASSES[name]
        return cls(self.data, **params)

    def _evaluate_signal(self, signal: pd.Series, name: str) -> dict:
        """Run MCPT on a single signal column and return metrics dict."""
        mcpt = MonteCarloPT(self.data, seed=self.seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress internal print output unless verbose
            if not self.verbose:
                import io, sys
                _stdout = sys.stdout
                sys.stdout = io.StringIO()

            res = mcpt.mcpt_threshold(signal, n_test=self.n_mcpt, min_kept=self.min_kepts)

            if not self.verbose:
                sys.stdout = _stdout

        return {
            "real_score":        res["real_score"],
            "pf_high":           res["real"]["pf_high"],
            "pf_low":            res["real"]["pf_low"],
            "high_thresh":       res["real"]["high_thresh"],
            "low_thresh":        res["real"]["low_thresh"],
            "p_value":           res["p_value"],
            "mc_mean_score":     res["mc_mean_score"],
            "mc_std_score":      res["mc_std_score"],
            "mc_distribution":   res["mc_distribution"],
            "signal_name":       name,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, indicators: Optional[list[str]] = None) -> pd.DataFrame:
        if indicators is None:
            indicators = list(self.grids.keys())

        self.results = []
        total = sum(
            len(self._param_combinations(self.grids[ind]))
            for ind in indicators
            if ind in self.grids
        )
        done = 0

        for ind_name in indicators:
            if ind_name not in _INDICATOR_CLASSES:
                warnings.warn(f"Unknown indicator '{ind_name}', skipping.")
                continue

            grid = self.grids.get(ind_name, {})
            combos = self._param_combinations(grid)

            for params in combos:
                done += 1
                try:
                    indicator = self._build_indicator(ind_name, params)
                    result_df = indicator.compute()
                except Exception as e:
                    warnings.warn(f"[{ind_name}] compute failed with {params}: {e}")
                    continue

                for col in result_df.columns:
                    signal = result_df[col].dropna()
                    if len(signal) < 100:
                        warnings.warn(
                            f"[{col}] too few observations ({len(signal)}), skipping."
                        )
                        continue

                    print(
                        f"[{done}/{total}] Evaluating {col} ...",
                        end="\r",
                        flush=True,
                    )

                    try:
                        metrics = self._evaluate_signal(
                            result_df[col].rename(col), col
                        )
                    except Exception as e:
                        warnings.warn(f"[{col}] MCPT failed: {e}")
                        continue

                    row = {
                        "indicator":    ind_name,
                        "signal":       col,
                        **{f"param_{k}": v for k, v in params.items()},
                        **{k: v for k, v in metrics.items()
                           if k != "mc_distribution"},
                        "_mc_dist":     metrics["mc_distribution"],
                    }
                    self.results.append(row)

        print()  # newline after \r progress

        if not self.results:
            print("No results collected.")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # --- Composite ranking score -------------------------------------------
        # Normalise profit factor (higher = better) and p-value (lower = better)
        pf_min, pf_max = df["real_score"].min(), df["real_score"].max()
        pf_range = pf_max - pf_min if pf_max != pf_min else 1.0
        df["pf_norm"] = (df["real_score"] - pf_min) / pf_range

        pv_min, pv_max = df["p_value"].min(), df["p_value"].max()
        pv_range = pv_max - pv_min if pv_max != pv_min else 1.0
        df["pv_norm"] = 1 - (df["p_value"] - pv_min) / pv_range  # inverted

        df["composite_score"] = 0.6 * df["pf_norm"] + 0.4 * df["pv_norm"]
        df["significant"] = df["p_value"] <= self.p_threshold

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        self.summary_df = df
        return df

    def get_summary(self, only_significant: bool = False) -> pd.DataFrame:
        if self.summary_df is None:
            raise RuntimeError("Call run() first.")

        df = self.summary_df
        if only_significant:
            df = df[df["significant"]]

        display_cols = [
            "rank", "indicator", "signal",
            "real_score", "pf_high", "pf_low",
            "high_thresh", "low_thresh",
            "p_value", "mc_mean_score", "mc_std_score",
            "composite_score", "significant",
        ]
        param_cols = [c for c in df.columns if c.startswith("param_")]
        return df[display_cols + param_cols].copy()

    def top_n(self, n: int = 10, only_significant: bool = True) -> pd.DataFrame:
        return self.get_summary(only_significant=only_significant).head(n)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        top_n: int = 20,
        only_significant: bool = False,
        figsize: tuple = (16, 10),
    ) -> plt.Figure:
        if self.summary_df is None:
            raise RuntimeError("Call run() first.")

        df = self.get_summary(only_significant=only_significant).head(top_n)

        if df.empty:
            print("No data to plot.")
            return None

        labels = df["signal"].tolist()
        x = np.arange(len(labels))

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        fig.suptitle(
            f"Indicator Grid Search — Top {top_n}"
            + (" (significant only)" if only_significant else ""),
            fontsize=14, fontweight="bold",
        )

        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # Colour map per indicator family
        families = df["indicator"].unique()
        cmap = plt.get_cmap("tab10")
        colour_map = {fam: cmap(i) for i, fam in enumerate(families)}
        bar_colours = [colour_map[fam] for fam in df["indicator"]]

        # --- Panel 1: Profit Factor ---
        ax1.bar(x, df["real_score"], color=bar_colours, edgecolor="white", linewidth=0.5)
        ax1.set_title("Profit Factor (real_score)", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("PF")
        ax1.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
                    label="PF = 1 (break-even)")
        ax1.legend(fontsize=8)

        # --- Panel 2: P-value ---
        ax2.bar(x, df["p_value"], color=bar_colours, edgecolor="white", linewidth=0.5)
        ax2.set_title("MCPT P-value", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("P-value")
        ax2.axhline(
            self.p_threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
            label=f"p = {self.p_threshold} threshold",
        )
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8)

        # --- Panel 3: Scatter PF vs P-value ---
        full_df = self.get_summary(only_significant=False)
        for fam in families:
            sub = full_df[full_df["indicator"] == fam]
            sig = sub[sub["significant"]]
            nsig = sub[~sub["significant"]]
            ax3.scatter(
                nsig["p_value"], nsig["real_score"],
                color=colour_map[fam], alpha=0.4, s=30, label=f"{fam} (n.s.)"
            )
            ax3.scatter(
                sig["p_value"], sig["real_score"],
                color=colour_map[fam], alpha=0.9, s=60, marker="*",
                label=f"{fam} (sig.)"
            )

        ax3.axvline(
            self.p_threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7
        )
        ax3.set_xlabel("P-value")
        ax3.set_ylabel("Profit Factor")
        ax3.set_title("PF vs P-value (all signals)", fontsize=11)
        ax3.legend(fontsize=7, ncol=2)

        # --- Panel 4: Composite score ---
        ax4.bar(
            x, df["composite_score"],
            color=bar_colours, edgecolor="white", linewidth=0.5
        )
        ax4.set_title("Composite Score (0.6·PF_norm + 0.4·(1−p)_norm)", fontsize=11)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax4.set_ylabel("Score")

        # Shared legend for colours
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=colour_map[fam], label=fam)
            for fam in families
        ]
        fig.legend(
            handles=handles, title="Indicator family",
            loc="lower center", ncol=len(families),
            bbox_to_anchor=(0.5, -0.02), fontsize=9,
        )

        return fig

    def plot_mc_distributions(
        self,
        top_n: int = 6,
        only_significant: bool = False,
        figsize: tuple = (14, 8),
    ) -> plt.Figure:
        if self.summary_df is None:
            raise RuntimeError("Call run() first.")

        df_top = self.summary_df.head(top_n)
        if only_significant:
            df_top = self.summary_df[self.summary_df["significant"]].head(top_n)

        n = len(df_top)
        if n == 0:
            print("No data to plot.")
            return None

        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        fig.suptitle("Monte Carlo Score Distributions", fontsize=13, fontweight="bold")

        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, (_, row) in enumerate(df_top.iterrows()):
            ax = axes_flat[i]
            dist = row["_mc_dist"]
            real = row["real_score"]
            pv = row["p_value"]

            ax.hist(dist, bins=30, color="steelblue", alpha=0.7, edgecolor="white",
                    linewidth=0.4, label="MC scores")
            ax.axvline(real, color="crimson", linewidth=1.8,
                       label=f"Real: {real:.3f}")
            ax.set_title(row["signal"], fontsize=9, fontweight="bold")
            ax.set_xlabel("Score", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            sig_tag = "✓ sig." if row["significant"] else "✗ n.s."
            ax.legend(
                title=f"p={pv:.3f}  {sig_tag}", fontsize=7, title_fontsize=7
            )

        # Hide unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        return fig