import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class VisualizerWalkForward:
    """
    Visualizador interactivo del walk-forward para indicadores y modelos.

    data        : DataFrame OHLCV original con índice temporal
    fold_results: lista de dicts devuelta por run_indicator(), run_model()
                  u OptunaWalkForward.run()
    signal      : pd.Series con la señal completa (mismo índice que data).
                  Si es None, se reconstruye automáticamente a partir de
                  los modelos y features guardados en cada fold_result
                  (requiere que cada fold tenga las claves "model" y
                  "features", como en OptunaWalkForward).
    """
    def __init__(self, data: pd.DataFrame, fold_results: list, signal: pd.Series = None,
                 time_col: str = "timestamp"):
        self.data         = data
        self.fold_results = fold_results
        self.time_col     = time_col if time_col in data.columns else None

        self.returns = np.log(self.data["close"].shift(-1) / self.data["close"])

        if signal is None:
            signal = self._reconstruct_signal_from_folds()
            self.signal_name = "model_prediction"
        else:
            self.signal_name = signal.name or "signal"

        self.signal = signal
        self._signal_series, self._equity_curve = self._build_signals_and_equity()

    def _reconstruct_signal_from_folds(self) -> pd.Series:
        """Reconstruye una señal continua concatenando predicciones por fold.

        Cada fold puede tener un modelo y un conjunto de features distinto
        (como ocurre en OptunaWalkForward). Predice sobre el test de cada
        fold usando su propio modelo, y concatena todo en orden temporal.
        """
        parts = []

        for r in self.fold_results:
            if "model" not in r or "features" not in r:
                raise ValueError(
                    "Para reconstruir la señal automáticamente, cada fold_result "
                    "necesita las claves 'model' y 'features' (p.ej. desde "
                    "OptunaWalkForward). Si no las tiene, pasa el parámetro "
                    "'signal' explícitamente."
                )

            test_start = r["test_start"]
            test_end   = r["test_end"]

            X_test = self.data[r["features"]].iloc[test_start:test_end]
            y_pred = pd.Series(
                r["model"].predict(X_test),
                index=X_test.index,
            )
            parts.append(y_pred)

        return pd.concat(parts).sort_index()

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

    @property
    def has_mcpt(self) -> bool:
        return len(self.fold_results) > 0 and "p_value_high" in self.fold_results[0]

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
    # Plot métricas walk-forward (thresholds + PF)
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

    # ------------------------------------------------------------------
    # Plot métricas MCPT (p-values + distribución agregada)
    # ------------------------------------------------------------------

    def plot_mcpt_metrics(self, p_threshold: float = 0.05) -> go.Figure:
        """
        Requiere que el walk-forward se haya ejecutado con mcpt=True.

        Muestra:
          - P-value high/low por fold con línea de significancia
          - Distribución agregada de p-values (histograma) con
            recuento de folds significativos vs no significativos
        """
        if not self.has_mcpt:
            raise RuntimeError(
                "fold_results no contiene p_values. "
                "Ejecuta run_indicator(mcpt=True) o run_model(mcpt=True) primero."
            )

        df_m  = pd.DataFrame(self.fold_results)
        folds = df_m["fold"]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"colspan": 2}, None],
                [{}, {}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
            subplot_titles=(
                "P-value por fold",
                "Distribución p_value_high",
                "Distribución p_value_low",
            )
        )

        # --- Panel 1: p-values por fold ---
        fig.add_trace(go.Scatter(
            x=folds, y=df_m["p_value_high"],
            mode="lines+markers", name="p_value high",
            line=dict(color="#00C896")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=folds, y=df_m["p_value_low"],
            mode="lines+markers", name="p_value low",
            line=dict(color="#FF4C6A")
        ), row=1, col=1)
        fig.add_hline(
            y=p_threshold,
            line=dict(color="white", width=1, dash="dash"),
            annotation_text=f"p = {p_threshold}",
            annotation_position="top right",
            row=1, col=1
        )
        fig.update_yaxes(title_text="P-value", range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="Fold", row=1, col=1)

        # --- Panel 2: histograma p_value_high ---
        fig.add_trace(go.Histogram(
            x=df_m["p_value_high"],
            nbinsx=20,
            marker_color="#00C896",
            opacity=0.75,
            name="p_value high",
            showlegend=False,
        ), row=2, col=1)
        fig.add_vline(
            x=p_threshold,
            line=dict(color="white", width=1, dash="dash"),
            row=2, col=1
        )
        fig.update_xaxes(title_text="p_value high", range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=2, col=1)

        # --- Panel 3: histograma p_value_low ---
        fig.add_trace(go.Histogram(
            x=df_m["p_value_low"],
            nbinsx=20,
            marker_color="#FF4C6A",
            opacity=0.75,
            name="p_value low",
            showlegend=False,
        ), row=2, col=2)
        fig.add_vline(
            x=p_threshold,
            line=dict(color="white", width=1, dash="dash"),
            row=2, col=2
        )
        fig.update_xaxes(title_text="p_value low", range=[0, 1], row=2, col=2)
        fig.update_yaxes(title_text="Frecuencia", row=2, col=2)

        # --- Estadísticas resumen como anotación ---
        n_folds       = len(df_m)
        sig_high      = (df_m["p_value_high"] < p_threshold).sum()
        sig_low       = (df_m["p_value_low"]  < p_threshold).sum()
        mean_p_high   = df_m["p_value_high"].mean()
        mean_p_low    = df_m["p_value_low"].mean()

        summary_text = (
            f"Folds significativos (p&lt;{p_threshold}):  "
            f"high {sig_high}/{n_folds} ({100*sig_high/n_folds:.0f}%)   |   "
            f"low {sig_low}/{n_folds} ({100*sig_low/n_folds:.0f}%)<br>"
            f"P-value medio:  high {mean_p_high:.3f}   |   low {mean_p_low:.3f}"
        )

        fig.update_layout(
            title=f"MCPT — {self.signal_name}",
            template="plotly_dark",
            height=650,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08),
            annotations=list(fig.layout.annotations) + [
                dict(
                    text=summary_text,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.12,
                    showarrow=False,
                    font=dict(size=12, color="#CCCCCC"),
                    align="center",
                )
            ]
        )

        return fig

    # ------------------------------------------------------------------
    # Visualizaciones específicas de OptunaWalkForward
    # (requieren que fold_results tenga la clave "features" por fold)
    # ------------------------------------------------------------------

    @property
    def has_features_per_fold(self) -> bool:
        return len(self.fold_results) > 0 and "features" in self.fold_results[0]

    def plot_feature_usage(self) -> go.Figure:
        """Muestra qué features se usaron en cada fold (mapa de calor binario).

        Útil para inspeccionar OptunaWalkForward: si una feature aparece en
        casi todos los folds, es una señal robusta y estable en el tiempo.
        Si solo aparece en unos pocos folds aislados, podría ser ruido que
        Optuna aprovechó puntualmente en ese régimen de mercado concreto.

        Returns:
            Figura de Plotly con un heatmap fold x feature.
        """
        if not self.has_features_per_fold:
            raise RuntimeError(
                "fold_results no contiene la clave 'features' por fold. "
                "Este gráfico requiere resultados de OptunaWalkForward."
            )

        df_m = pd.DataFrame(self.fold_results)
        folds = df_m["fold"].tolist()

        all_features = sorted(set(f for feats in df_m["features"] for f in feats))

        matrix = np.zeros((len(all_features), len(folds)))
        for j, feats in enumerate(df_m["features"]):
            for f in feats:
                i = all_features.index(f)
                matrix[i, j] = 1

        usage_pct = matrix.mean(axis=1) * 100
        order = np.argsort(-usage_pct)
        matrix_sorted = matrix[order]
        features_sorted = [all_features[i] for i in order]

        fig = go.Figure(data=go.Heatmap(
            z=matrix_sorted,
            x=folds,
            y=features_sorted,
            colorscale=[[0, "#1a1a1a"], [1, "#00C896"]],
            showscale=False,
            hovertemplate="Fold %{x}<br>%{y}<br>Usada: %{z}<extra></extra>",
        ))

        fig.update_layout(
            title="Uso de features por fold (OptunaWalkForward)",
            template="plotly_dark",
            height=max(400, 25 * len(features_sorted)),
            width=1000,
            xaxis_title="Fold",
            yaxis_title="Feature",
        )

        return fig

    def plot_feature_usage_bar(self) -> go.Figure:
        """Gráfico de barras con el % de folds en que se usó cada feature.

        Complementa a plot_feature_usage() con una vista más directa
        de qué features son consistentemente elegidas por Optuna.

        Returns:
            Figura de Plotly con un gráfico de barras horizontal.
        """
        if not self.has_features_per_fold:
            raise RuntimeError(
                "fold_results no contiene la clave 'features' por fold. "
                "Este gráfico requiere resultados de OptunaWalkForward."
            )

        df_m = pd.DataFrame(self.fold_results)
        n_folds = len(df_m)

        all_features = sorted(set(f for feats in df_m["features"] for f in feats))
        counts = {f: 0 for f in all_features}
        for feats in df_m["features"]:
            for f in feats:
                counts[f] += 1

        usage = pd.Series(counts).sort_values(ascending=True)
        pct = usage / n_folds * 100

        fig = go.Figure(go.Bar(
            x=pct.values,
            y=pct.index,
            orientation="h",
            marker_color="#5B8CFF",
            hovertemplate="%{y}: %{x:.1f}%% de los folds<extra></extra>",
        ))

        fig.update_layout(
            title=f"Frecuencia de uso por feature ({n_folds} folds)",
            template="plotly_dark",
            height=max(400, 25 * len(usage)),
            width=800,
            xaxis_title="% de folds en que se usó",
            xaxis_range=[0, 100],
        )

        return fig

    def plot_pf_evolution(self) -> go.Figure:
        """Evolución del Profit Factor y, si existe, el p-value, fold a fold.

        A diferencia de plot_walkforward_metrics() (pensado para un único
        indicador/modelo fijo), este método está pensado para resultados
        donde cada fold puede tener un modelo distinto (OptunaWalkForward),
        y opcionalmente muestra el inner_score de la búsqueda de Optuna
        para comparar con el resultado real en test.

        Returns:
            Figura de Plotly con 1 o 2 paneles según haya o no MCPT.
        """
        df_m  = pd.DataFrame(self.fold_results)
        folds = df_m["fold"]
        has_mcpt = "p_value_high" in df_m.columns
        has_inner = "inner_score" in df_m.columns

        def fmt_inf(s):
            return s.replace([np.inf, -np.inf], np.nan)

        rows = 2 if has_mcpt else 1
        titles = ["Profit Factor por fold"]
        if has_mcpt:
            titles.append("P-value por fold")

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=titles,
        )

        fig.add_trace(go.Scatter(
            x=folds, y=fmt_inf(df_m["pf_test_long_above"]),
            mode="lines+markers", name="PF test long above",
            line=dict(color="#00C896")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=folds, y=fmt_inf(df_m["pf_test_short_below"]),
            mode="lines+markers", name="PF test short below",
            line=dict(color="#FF4C6A")
        ), row=1, col=1)

        if has_inner:
            fig.add_trace(go.Scatter(
                x=folds, y=df_m["inner_score"],
                mode="lines+markers", name="Inner score (validación Optuna)",
                line=dict(color="#FFD166", dash="dot")
            ), row=1, col=1)

        fig.add_hline(y=1.0, line=dict(color="white", width=0.8, dash="dash"), row=1, col=1)
        fig.update_yaxes(title_text="Profit Factor", row=1, col=1)

        if has_mcpt:
            fig.add_trace(go.Scatter(
                x=folds, y=df_m["p_value_high"],
                mode="lines+markers", name="p_value high",
                line=dict(color="#00C896")
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=folds, y=df_m["p_value_low"],
                mode="lines+markers", name="p_value low",
                line=dict(color="#FF4C6A")
            ), row=2, col=1)
            fig.add_hline(y=0.05, line=dict(color="white", width=1, dash="dash"), row=2, col=1)
            fig.update_yaxes(title_text="P-value", range=[0, 1], row=2, col=1)
            fig.update_xaxes(title_text="Fold", row=2, col=1)
        else:
            fig.update_xaxes(title_text="Fold", row=1, col=1)

        fig.update_layout(
            title="Evolución por fold",
            template="plotly_dark",
            height=350 * rows,
            width=1000,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.15),
        )

        return fig