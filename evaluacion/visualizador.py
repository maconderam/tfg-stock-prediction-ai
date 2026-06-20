import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class VisualizerWalkforward:
    """
    Visualizador interactivo del análisis Walk-Forward para indicadores y modelos.

    Esta clase genera gráficos dinámicos utilizando Plotly para analizar la evolución
    del precio, los umbrales dinámicos optimizados por cada fold, las señales de
    operación generadas y la curva de capital (equity curve) resultante.
    """

    def __init__(self, data: pd.DataFrame, fold_results: list, signal: pd.Series, time_col: str = "timestamp"):
        self.data         = data
        self.fold_results = fold_results
        self.signal       = signal
        self.signal_name  = signal.name or "signal"
        self.time_col     = time_col if time_col in data.columns else None

        self.returns = np.log(self.data["close"].shift(-1) / self.data["close"])

        # Construcción interna de las series temporales de señales y retornos acumulados
        self._signal_series, self._equity_curve = self._build_signals_and_equity()

    def _get_x(self, index):
        """Devuelve las fechas reales o el componente temporal para un índice dado."""
        if self.time_col:
            return self.data.loc[index, self.time_col]
        return index

    def _build_signals_and_equity(self):
        """Une los fragmentos de test de cada fold para reconstruir la señal global sin solapamiento."""
        parts = []

        for r in self.fold_results:
            test_start  = r["test_start"]
            test_end    = r["test_end"]
            high_thresh = r["high_thresh"]
            low_thresh  = r["low_thresh"]

            test_index = self.data.index[test_start:test_end]

            # Desplazamiento temporal para mitigar por completo el sesgo de anticipación
            sig_fold = self.signal.reindex(test_index).shift(1)
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
        """Verifica si los resultados de los folds incluyen métricas del test de Monte Carlo."""
        return len(self.fold_results) > 0 and "p_value_high" in self.fold_results[0]

    def plot(self, title: str = None) -> go.Figure:
        """
        Genera un gráfico interactivo multi-panel con el rendimiento de la estrategia.

        Divide la visualización en cuatro sub-gráficos alineados por eje temporal:
        Precio de cierre con puntos de entrada, evolución del indicador con sus umbrales
        móviles por fold, representación en barras de la señal y la curva de capital.

        Args:
            title (str, optional): Título personalizado para el gráfico de Plotly.

        Returns:
            go.Figure: Objeto figura de Plotly listo para ser renderizado o mostrado.
        """
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

        # --- 1. Gráfico del precio de cierre ---
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

        # --- 2. Indicador junto con sus umbrales dinámicos ---
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

            # Inyección visual de las líneas horizontales de umbrales para cada ventana de test
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

        # Creación de leyendas manuales para los umbrales punteados
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

        # --- 3. Representación de la señal ejecutada ---
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

        # --- 4. Curva de rendimiento acumulado (Equity Curve) ---
        fig.add_trace(go.Scatter(
            x=self._get_x(self._equity_curve.index),
            y=self._equity_curve.values,
            mode="lines", name="Equity curve",
            line=dict(color="#FFD166", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,209,102,0.1)",
        ), row=4, col=1)

        fig.add_hline(y=0, line=dict(color="white", width=0.5), row=4, col=1)

        # --- Ajustes finales de estilo y maquetación ---
        fig.update_layout(
            title=title or f"Walk-Forward — {self.signal_name}",
            template="plotly_dark",
            height=900,
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.05),
        )

        fig.update_yaxes(title_text="Precio",        row=1, col=1)
        fig.update_yaxes(title_text="Valor",          row=2, col=1)
        fig.update_yaxes(title_text="Señal",          row=3, col=1,
                         tickvals=[-1, 0, 1], ticktext=["Short", "Neutral", "Long"])
        fig.update_yaxes(title_text="Log-ret. acum.", row=4, col=1)

        return fig

    def plot_walkforward_metrics(self) -> go.Figure:
        """
        Genera gráficos de control temporal para los umbrales y el Profit Factor por fold.

        Permite diagnosticar la estabilidad temporal de los parámetros optimizados y observar
        si el rendimiento se degrada bruscamente al pasar de entornos in-sample (train)
        a entornos fuera de muestra (test).

        Returns:
            go.Figure: Gráfico interactivo con las series de métricas agregadas por fold.
        """
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

    def plot_mcpt_metrics(self, p_threshold: float = 0.05) -> go.Figure:
        """
        Visualiza la distribución e histórico de p-values calculados mediante MCPT.

        Muestra de manera unificada el comportamiento de la significancia estadística a lo
        largo del tiempo y la concentración de frecuencias mediante histogramas independientes
        tanto para umbrales alcistas como bajistas.

        Args:
            p_threshold (float, optional): Nivel crítico alfa de significancia. Defaults to 0.05.

        Returns:
            go.Figure: Paneles interactivos con curvas e histogramas de distribución de p-values.

        Raises:
            RuntimeError: Si los datos provistos en `fold_results` carecen del cálculo de Monte Carlo.
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

        # --- Panel 1: Evolución secuencial de p-values ---
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

        # --- Panel 2: Histograma marginal para p_value_high ---
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

        # --- Panel 3: Histograma marginal para p_value_low ---
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

        # --- Cálculo y formateo de anotaciones del sumario general ---
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