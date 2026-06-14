import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """
    Visualizador interactivo del walk-forward para indicadores.

    data        : DataFrame OHLCV original con índice temporal
    fold_results: lista de dicts devuelta por run_indicator()
    signal      : pd.Series con la señal completa (mismo índice que data)
    """
    def __init__(self, data: pd.DataFrame, fold_results: list, signal: pd.Series, time_col: str = "timestamp"):
        self.data         = data
        self.fold_results = fold_results
        self.signal       = signal
        self.signal_name  = signal.name or "signal"
        self.time_col     = time_col if time_col in data.columns else None

        self.returns = np.log(self.data["close"].shift(-1) / self.data["close"])

        self._signal_series, self._equity_curve = self._build_signals_and_equity()

    def _get_x(self, index):
        """Devuelve las fechas reales para un índice dado."""
        if self.time_col:
            return self.data.loc[index, self.time_col]
        return index

    def _build_signals_and_equity(self):
        parts = []

        for r in self.fold_results:
            test_start  = r["test_start"]
            test_end    = r["test_end"]
            high_thresh = r["high_thresh"]
            low_thresh  = r["low_thresh"]

            test_index = self.data.index[test_start:test_end]

            sig_fold = self.signal.reindex(test_index).shift(1)  # Shift para evitar look-ahead bias
            ret_fold = self.returns.reindex(test_index)

            valid    = sig_fold.notna() & ret_fold.notna()
            sig_fold = sig_fold[valid]
            ret_fold = ret_fold[valid]

            s = pd.Series(0, index=sig_fold.index, dtype=float)
            s[sig_fold >= high_thresh] =  1.0
            s[sig_fold <= low_thresh]  = -1.0

            parts.append(pd.DataFrame({
                "signal":           s,
                "strategy_returns": s * ret_fold,
            }))

        if not parts:
            empty = pd.Series(dtype=float)
            return empty, empty

        combined      = pd.concat(parts).sort_index()
        signal_series = combined["signal"]
        equity_curve  = combined["strategy_returns"].cumsum().rename("equity_curve")

        return signal_series, equity_curve

    # ------------------------------------------------------------------
    # Plot principal
    # ------------------------------------------------------------------

    def plot(self, title: str = None) -> go.Figure:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.35, 0.25, 0.15, 0.25],
            subplot_titles=(
                "Precio (close)",
                f"Indicador: {self.signal_name}",
                "Señal (1=long, -1=short, 0=neutral)",
                "Equity curve (log-returns acumulados)"
            )
        )

        idx = self._get_x(self.data.index)

        # --- 1. Close ---
        fig.add_trace(go.Scatter(
            x=idx, y=self.data["close"],
            mode="lines", name="Close",
            line=dict(color="#5B8CFF", width=1.2),
        ), row=1, col=1)

        long_idx  = self._signal_series[self._signal_series ==  1].index
        short_idx = self._signal_series[self._signal_series == -1].index

        if len(long_idx):
            fig.add_trace(go.Scatter(
                x=self._get_x(long_idx),
                y=self.data["close"].reindex(long_idx),
                mode="markers", name="Long",
                marker=dict(symbol="triangle-up", color="#00C896", size=8),
            ), row=1, col=1)

        if len(short_idx):
            fig.add_trace(go.Scatter(
                x=self._get_x(short_idx),
                y=self.data["close"].reindex(short_idx),
                mode="markers", name="Short",
                marker=dict(symbol="triangle-down", color="#FF4C6A", size=8),
            ), row=1, col=1)

        # --- 2. Indicador + thresholds ---
        signal_valid = self.signal.dropna()
        fig.add_trace(go.Scatter(
            x=self._get_x(signal_valid.index), y=signal_valid.values,
            mode="lines", name=self.signal_name,
            line=dict(color="#B07FFF", width=1.2),
            connectgaps=False,
        ), row=2, col=1)

        for i, r in enumerate(self.fold_results):
            x_range = self.data.index[r["test_start"]:r["test_end"]]
            if len(x_range) == 0:
                continue

            x0 = self._get_x(x_range[[0]]).iloc[0]
            x1 = self._get_x(x_range[[-1]]).iloc[-1]

            fig.add_shape(type="line",
                x0=x0, x1=x1,
                y0=r["high_thresh"], y1=r["high_thresh"],
                line=dict(color="#00C896", width=1, dash="dot"),
                row=2, col=1
            )
            fig.add_shape(type="line",
                x0=x0, x1=x1,
                y0=r["low_thresh"], y1=r["low_thresh"],
                line=dict(color="#FF4C6A", width=1, dash="dot"),
                row=2, col=1
            )
            fig.add_vline(
                x=x0,
                line=dict(color="rgba(200,200,200,0.3)", width=1, dash="dash"),
            )

        # Leyenda manual thresholds
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name="High threshold",
            line=dict(color="#00C896", dash="dot")
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name="Low threshold",
            line=dict(color="#FF4C6A", dash="dot")
        ), row=2, col=1)

        # --- 3. Señal ---
        colours = self._signal_series.map({
            1: "#00C896", -1: "#FF4C6A", 0: "rgba(150,150,150,0.3)"
        })

        fig.add_trace(go.Bar(
            x=self._get_x(self._signal_series.index),
            y=self._signal_series.values,
            name="Señal",
            marker_color=colours,
            showlegend=False,
        ), row=3, col=1)

        fig.add_hline(y=0, line=dict(color="white", width=0.5), row=3, col=1)

        # --- 4. Equity curve ---
        fig.add_trace(go.Scatter(
            x=self._get_x(self._equity_curve.index),
            y=self._equity_curve.values,
            mode="lines", name="Equity curve",
            line=dict(color="#FFD166", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,209,102,0.1)",
        ), row=4, col=1)

        fig.add_hline(y=0, line=dict(color="white", width=0.5), row=4, col=1)

        # --- Layout ---
        fig.update_layout(
            title=title or f"Walk-Forward — {self.signal_name}",
            template="plotly_dark",
            height=900,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.05),
        )

        fig.update_yaxes(title_text="Precio",         row=1, col=1)
        fig.update_yaxes(title_text="Valor",          row=2, col=1)
        fig.update_yaxes(title_text="Señal",          row=3, col=1,
                         tickvals=[-1, 0, 1], ticktext=["Short", "Neutral", "Long"])
        fig.update_yaxes(title_text="Log-ret. acum.", row=4, col=1)

        return fig

    # ------------------------------------------------------------------
    # Plot métricas walk-forward
    # ------------------------------------------------------------------

    def plot_walkforward_metrics(self) -> go.Figure:
        df_m  = pd.DataFrame(self.fold_results)
        folds = df_m["fold"]

        def fmt_inf(s):
            return s.replace([np.inf], np.nan)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Thresholds por fold", "Profit Factor por fold")
        )

        fig.add_trace(go.Scatter(
            x=folds, y=df_m["high_thresh"],
            mode="lines+markers", name="High threshold",
            line=dict(color="#00C896")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=folds, y=df_m["low_thresh"],
            mode="lines+markers", name="Low threshold",
            line=dict(color="#FF4C6A")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=folds, y=fmt_inf(df_m["pf_train_high"]),
            mode="lines+markers", name="PF train high",
            line=dict(color="#00C896", dash="dot")
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=folds, y=fmt_inf(df_m["pf_train_low"]),
            mode="lines+markers", name="PF train low",
            line=dict(color="#FF4C6A", dash="dot")
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=folds, y=fmt_inf(df_m["pf_test_long_above"]),
            mode="lines+markers", name="PF test long above",
            line=dict(color="#00C896")
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=folds, y=fmt_inf(df_m["pf_test_short_below"]),
            mode="lines+markers", name="PF test short below",
            line=dict(color="#FF4C6A")
        ), row=2, col=1)

        fig.add_hline(y=1.0, line=dict(color="white", width=0.8, dash="dash"), row=2, col=1)

        fig.update_layout(
            title=f"Métricas Walk-Forward — {self.signal_name}",
            template="plotly_dark",
            height=600,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.1),
        )

        fig.update_xaxes(title_text="Fold", row=2, col=1)
        fig.update_yaxes(title_text="Valor threshold", row=1, col=1)
        fig.update_yaxes(title_text="Profit Factor",   row=2, col=1)

        return fig