import itertools
import warnings
import io
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any

from .indicators import (
    RSI, Stochastic, StochasticRSI, MACD, PriceIntensity,
    ADX, Aroon, AroonOscillator, ATR, PriceChangeOscillator,
    PriceVarianceRatio, ChangeVarianceRatio, CMMA, MADifference,
    IntradayIntensity, ChaikinMoneyFlow,
)
from evaluacion.mcpt import MonteCarloPT

# ---------------------------------------------------------------------------
# Grillas de parámetros por defecto para cada clase de indicador
# ---------------------------------------------------------------------------
DEFAULT_GRIDS = {
    "RSI": {"window": [7, 10, 14, 20, 30, 50], "smooth_window": [2, 3, 5]},
    "Stochastic": {"window": [7, 10, 14, 20, 30, 50], "smooth_window": [2, 3, 5]},
    "StochasticRSI": {"rsi_window": [10, 14, 20, 30], "stoch_window": [3, 5, 10], "smooth_window": [2, 3]},
    "MACD": {"short_length": [8, 12, 16], "long_length": [20, 26, 35, 50], "smooth_window": [9]},
    "PriceIntensity": {"smooth_window": [5, 10, 14, 20, 30, 50]},
    "ADX": {"window": [7, 10, 14, 20, 30, 50]},
    "Aroon": {"window": [10, 14, 20, 30, 50, 100]},
    "AroonOscillator": {"window": [10, 14, 20, 30, 50, 100]},
    "PriceChangeOscillator": {"short_length": [5, 10, 20], "mult": [2, 3, 5]},
    "PriceVarianceRatio": {"short_length": [5, 10, 20], "mult": [2, 4, 6]},
    "ChangeVarianceRatio": {"short_length": [5, 10, 20], "mult": [2, 4, 6]},
    "CMMA": {"window": [5, 10, 20, 50], "atr_window": [14, 60, 252], "c": [1.0]},
    "MADifference": {"short_length": [5, 10, 20], "long_length": [50, 100, 150], "lag": [0]},
    "IntradayIntensity": {"window": [7, 14, 21, 30], "smooth_window": [1, 5, 10]},
    "ChaikinMoneyFlow": {"window": [7, 10, 14, 21, 30, 50]},
}

_INDICATOR_CLASSES = {
    "RSI": RSI, "Stochastic": Stochastic, "StochasticRSI": StochasticRSI,
    "MACD": MACD, "PriceIntensity": PriceIntensity, "ADX": ADX,
    "Aroon": Aroon, "AroonOscillator": AroonOscillator,
    "PriceChangeOscillator": PriceChangeOscillator, "PriceVarianceRatio": PriceVarianceRatio,
    "ChangeVarianceRatio": ChangeVarianceRatio, "CMMA": CMMA,
    "MADifference": MADifference, "IntradayIntensity": IntradayIntensity,
    "ChaikinMoneyFlow": ChaikinMoneyFlow,
}

class IndicatorSelector:
    """
    Motor de Grid Search y Selección de Características protegido por Monte Carlo.

    Evalúa múltiples combinaciones de parámetros sobre indicadores técnicos, mitigando el 
    sesgo de sobreajuste mediante pruebas de permutación de Monte Carlo (MCPT). Calcula
    un puntaje compuesto institucional que pondera la ganancia económica y la significancia estadística.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target=None,
        min_kepts: int = 300,
        n_mcpt: int = 200,
        p_threshold: float = 0.10,
        custom_grids: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Inicializa el Selector de Indicadores.

        Args:
            data (pd.DataFrame): Datos históricos del activo financiero (OHLCV).
            target (Any, optional): Variable objetivo. Defaults to None.
            min_kepts (int, optional): Mínimo de operaciones retenidas en MCPT. Defaults to 300.
            n_mcpt (int, optional): Número de permutaciones aleatorias de Monte Carlo. Defaults to 200.
            p_threshold (float, optional): Umbral para declarar significancia estadística. Defaults to 0.10.
            custom_grids (dict, optional): Parámetros de usuario para sobreescribir la grilla. Defaults to None.
            seed (int, optional): Semilla pseudoaleatoria para reproducibilidad. Defaults to 42.
            verbose (bool, optional): Muestra logs detallados si es True. Defaults to False.
        """
        self.data = data.copy()
        self.target = target
        self.min_kepts = min_kepts
        self.n_mcpt = n_mcpt
        self.p_threshold = p_threshold
        self.seed = seed
        self.verbose = verbose

        self.grids = {k: dict(v) for k, v in DEFAULT_GRIDS.items()}
        if custom_grids:
            for name, params in custom_grids.items():
                if name in self.grids:
                    self.grids[name].update(params)
                else:
                    self.grids[name] = params

        self.results: List[Dict[str, Any]] = []
        self.summary_df: Optional[pd.DataFrame] = None

    @staticmethod
    def _param_combinations(grid: dict) -> List[dict]:
        """Calcula el producto cartesiano de los parámetros de la grilla."""
        keys = list(grid.keys())
        values = list(grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _build_indicator(self, name: str, params: dict):
        """Instancia dinámicamente un objeto indicador a partir de su mapa de clase."""
        cls = _INDICATOR_CLASSES[name]
        return cls(self.data, **params)

    def _evaluate_signal(self, signal: pd.Series, name: str) -> dict:
        """Ejecuta una evaluación de permutación MCPT de forma aislada."""
        mcpt = MonteCarloPT(self.data, seed=self.seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self.verbose:
                _stdout = sys.stdout
                sys.stdout = io.StringIO()

            res = mcpt.mcpt_threshold(signal, n_test=self.n_mcpt, min_kept=self.min_kepts)

            if not self.verbose:
                sys.stdout = _stdout

        return {
            "real_score": res["real_score"],
            "pf_high": res["real"]["pf_high"],
            "pf_low": res["real"]["pf_low"],
            "high_thresh": res["real"]["high_thresh"],
            "low_thresh": res["real"]["low_thresh"],
            "p_value": res["p_value"],
            "mc_mean_score": res["mc_mean_score"],
            "mc_std_score": res["mc_std_score"],
            "mc_distribution": res["mc_distribution"],
            "signal_name": name,
        }

    def run(self, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Ejecuta el Grid Search y el filtrado estadístico masivo.

        Args:
            indicators (list, optional): Lista de indicadores a optimizar. Si es None,
                ejecuta todos los disponibles de la grilla. Defaults to None.

        Returns:
            pd.DataFrame: Resultados ordenados bajo la métrica del puntaje compuesto.
        """
        if indicators is None:
            indicators = list(self.grids.keys())

        self.results = []
        total = sum(len(self._param_combinations(self.grids[ind])) for ind in indicators if ind in self.grids)
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
                        warnings.warn(f"[{col}] too few observations ({len(signal)}), skipping.")
                        continue

                    print(f"[{done}/{total}] Evaluating {col} ...", end="\r", flush=True)

                    try:
                        metrics = self._evaluate_signal(result_df[col].rename(col), col)
                    except Exception as e:
                        warnings.warn(f"[{col}] MCPT failed: {e}")
                        continue

                    row = {
                        "indicator": ind_name,
                        "signal": col,
                        **{f"param_{k}": v for k, v in params.items()},
                        **{k: v for k, v in metrics.items() if k != "mc_distribution"},
                        "_mc_dist": metrics["mc_distribution"],
                    }
                    self.results.append(row)

        print()  # Liberar retorno de carro del print dinámico

        if not self.results:
            print("No results collected.")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # --- Mitigación de Infinitos en Profit Factor ---
        df["real_score"] = df["real_score"].replace([np.inf, -np.inf], np.nan)
        max_finite_score = df["real_score"].max()
        df["real_score"] = df["real_score"].fillna(max_finite_score if pd.notna(max_finite_score) else 5.0)

        # --- Escalamiento Robustecido (Min-Max) para Profit Factor ---
        pf_min, pf_max = df["real_score"].min(), df["real_score"].max()
        pf_range = pf_max - pf_min if pf_max != pf_min else 1.0
        df["pf_norm"] = (df["real_score"] - pf_min) / pf_range

        # --- Robustecimiento Estadístico para P-values (Transformación de Información) ---
        epsilon = 1e-5
        df["log_p"] = -np.log10(df["p_value"] + epsilon)
        lp_min, lp_max = df["log_p"].min(), df["log_p"].max()
        lp_range = lp_max - lp_min if lp_max != lp_min else 1.0
        df["pv_norm"] = (df["log_p"] - lp_min) / lp_range

        # --- Puntaje Compuesto: Balance Alpha vs Ruido Estadístico ---
        df["composite_score"] = 0.6 * df["pf_norm"] + 0.4 * df["pv_norm"]
        df["significant"] = df["p_value"] <= self.p_threshold

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        self.summary_df = df
        return df

    def get_summary(self, only_significant: bool = False) -> pd.DataFrame:
        """Devuelve una vista simplificada y ordenada de los resultados."""
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
        """Retorna las N mejores estrategias del ranking."""
        return self.get_summary(only_significant=only_significant).head(n)

    def plot_summary(self, top_n: int = 20, only_significant: bool = False) -> go.Figure:
        """Genera un dashboard interactivo en Plotly con 4 paneles de diagnóstico alpha."""
        if self.summary_df is None:
            raise RuntimeError("Call run() first.")

        df = self.get_summary(only_significant=only_significant).head(top_n)
        if df.empty:
            print("No data to plot.")
            return None

        param_cols = [c for c in df.columns if c.startswith("param_")]

        def hover_text(row):
            params_str = "<br>".join(f"{c.replace('param_', '')}: {row[c]}" for c in param_cols)
            return (
                f"<b>{row['signal']}</b><br>"
                f"Indicator: {row['indicator']}<br>"
                f"{params_str}<br>"
                f"PF: {row['real_score']:.3f}<br>"
                f"p-value: {row['p_value']:.3f}<br>"
                f"Composite: {row['composite_score']:.3f}"
            )

        df = df.copy()
        df["hover"] = df.apply(hover_text, axis=1)
        labels = df["signal"].tolist()

        families = df["indicator"].unique()
        palette = [
            "#5B8CFF", "#00C896", "#FF4C6A", "#FFD166", "#B07FFF",
            "#FF8C42", "#4ECDC4", "#F7B801", "#A8DADC", "#E76F51",
        ]
        colour_map = {fam: palette[i % len(palette)] for i, fam in enumerate(families)}
        bar_colours = df["indicator"].map(colour_map)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Profit Factor (real_score)",
                "MCPT P-value",
                "PF vs P-value (todas las señales)",
                "Composite Score",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        # Panel 1: Profit Factor
        fig.add_trace(go.Bar(
            x=labels, y=df["real_score"],
            marker_color=bar_colours,
            hovertext=df["hover"], hoverinfo="text",
            showlegend=False,
        ), row=1, col=1)
        fig.add_hline(y=1.0, line=dict(color="white", width=1, dash="dash"), row=1, col=1)

        # Panel 2: P-value
        fig.add_trace(go.Bar(
            x=labels, y=df["p_value"],
            marker_color=bar_colours,
            hovertext=df["hover"], hoverinfo="text",
            showlegend=False,
        ), row=1, col=2)
        fig.add_hline(y=self.p_threshold, line=dict(color="white", width=1, dash="dash"), row=1, col=2)

        # Panel 3: Scatter PF vs P-value
        full_df = self.get_summary(only_significant=False).copy()
        full_df["hover"] = full_df.apply(hover_text, axis=1)

        for fam in families:
            sub = full_df[full_df["indicator"] == fam]
            sig = sub[sub["significant"]]
            nsig = sub[~sub["significant"]]

            fig.add_trace(go.Scatter(
                x=nsig["p_value"], y=nsig["real_score"],
                mode="markers", name=f"{fam} (n.s.)",
                marker=dict(color=colour_map[fam], size=7, opacity=0.4),
                hovertext=nsig["hover"], hoverinfo="text",
                legendgroup=fam,
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=sig["p_value"], y=sig["real_score"],
                mode="markers", name=f"{fam} (sig.)",
                marker=dict(color=colour_map[fam], size=11, symbol="star", line=dict(width=1, color="white")),
                hovertext=sig["hover"], hoverinfo="text",
                legendgroup=fam,
            ), row=2, col=1)

        fig.add_vline(x=self.p_threshold, line=dict(color="white", width=1, dash="dash"), row=2, col=1)

        # Panel 4: Composite Score
        fig.add_trace(go.Bar(
            x=labels, y=df["composite_score"],
            marker_color=bar_colours,
            hovertext=df["hover"], hoverinfo="text",
            showlegend=False,
        ), row=2, col=2)

        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=2)

        fig.update_layout(
            title=f"Indicator Grid Search — Top {top_n}",
            template="plotly_dark",
            height=850, width=1300,
            showlegend=True,
            legend=dict(orientation="h", y=-0.25, font=dict(size=9)),
        )
        return fig

    def plot_mc_distributions(self, top_n: int = 6, only_significant: bool = False) -> go.Figure:
        """Grafica los histogramas empíricos generados por el proceso de barajado Monte Carlo."""
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

        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=df_top["signal"].tolist(),
            vertical_spacing=0.15, horizontal_spacing=0.08,
        )

        for i, (_, row) in enumerate(df_top.iterrows()):
            r = i // ncols + 1
            c = i % ncols + 1

            fig.add_trace(go.Histogram(
                x=row["_mc_dist"], nbinsx=30,
                marker_color="#5B8CFF", opacity=0.75,
                name="MC scores", showlegend=(i == 0),
            ), row=r, col=c)

            fig.add_vline(x=row["real_score"], line=dict(color="#FF4C6A", width=2.5), row=r, col=c)

        fig.update_layout(
            title="Distribuciones Empíricas de Monte Carlo (Ruido vs Realidad)",
            template="plotly_dark",
            height=320 * nrows, width=420 * ncols,
        )
        return fig