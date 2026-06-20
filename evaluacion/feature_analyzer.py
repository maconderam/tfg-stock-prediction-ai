import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FeatureAnalyzer:
    """
    Analiza la correlación entre features (indicadores) en train y test.

    Permite detectar redundancia entre indicadores y comprobar si esa
    redundancia es estable fuera de muestra, o si solo aparece en train.

    Args:
        df_train: DataFrame con los datos de entrenamiento.
        df_test: DataFrame con los datos de test.
        columns: Lista de columnas (indicadores) a analizar. Si es None,
            se usan todas las columnas numéricas comunes a ambos DataFrames.
    """

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, columns: list = None):
        self.df_train = df_train
        self.df_test  = df_test

        if columns is None:
            numeric_train = set(df_train.select_dtypes(include=np.number).columns)
            numeric_test  = set(df_test.select_dtypes(include=np.number).columns)
            columns = sorted(numeric_train & numeric_test)

        self.columns = columns

    def correlation_matrix(self, dataset: str = "train") -> pd.DataFrame:
        """
        Calcula la matriz de correlación de Pearson para las columnas seleccionadas.

        Args:
            dataset: "train" o "test", qué conjunto de datos usar.

        Returns:
            DataFrame cuadrado con las correlaciones entre cada par de columnas.
        """
        df = self.df_train if dataset == "train" else self.df_test
        return df[self.columns].corr()

    def plot_correlation_heatmap(self, dataset: str = "train", title: str = None) -> go.Figure:
        """
        Genera un heatmap interactivo de correlaciones para un único dataset.

        Args:
            dataset: "train" o "test".
            title: Título de la figura. Si es None, se genera automáticamente.

        Returns:
            Figura de Plotly con el heatmap.
        """
        corr = self.correlation_matrix(dataset)

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=14, color="black"),
            colorbar=dict(title="Correlación"),
        ))

        fig.update_layout(
            title=title or f"Matriz de correlación — {dataset}",
            template="plotly_dark",
            height=700,
            width=750,
            xaxis=dict(tickangle=45),
        )

        return fig

    def plot_correlation_comparison(self) -> go.Figure:
        """
        Genera dos heatmaps lado a lado (train y test) para comparar visualmente.

        Útil para detectar si las correlaciones son estables entre ambos
        conjuntos o si cambian sustancialmente fuera de muestra.

        Returns:
            Figura de Plotly con los dos heatmaps lado a lado.
        """
        corr_train = self.correlation_matrix("train")
        corr_test  = self.correlation_matrix("test")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Train", "Test"),
            horizontal_spacing=0.12,
        )

        fig.add_trace(go.Heatmap(
            z=corr_train.values,
            x=corr_train.columns,
            y=corr_train.columns,
            colorscale="RdBu",
            zmid=0, zmin=-1, zmax=1,
            text=corr_train.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=14, color="black"),
            showscale=False,
        ), row=1, col=1)

        fig.add_trace(go.Heatmap(
            z=corr_test.values,
            x=corr_test.columns,
            y=corr_test.columns,
            colorscale="RdBu",
            zmid=0, zmin=-1, zmax=1,
            text=corr_test.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=14, color="black"),
            colorbar=dict(title="Correlación"),
        ), row=1, col=2)

        fig.update_layout(
            title="Comparación de correlaciones — Train vs Test",
            template="plotly_dark",
            height=600,
            width=1200,
        )
        fig.update_xaxes(tickangle=45)

        return fig

    def plot_correlation_difference(self) -> go.Figure:
        """
        Genera un heatmap con la diferencia absoluta de correlaciones (train - test).

        Valores altos indican que la relación entre ese par de indicadores
        no es estable fuera de muestra, y por tanto no debería usarse como
        criterio fiable para eliminar redundancia.

        Returns:
            Figura de Plotly con el heatmap de diferencias.
        """
        corr_train = self.correlation_matrix("train")
        corr_test  = self.correlation_matrix("test")
        diff = (corr_train - corr_test).abs()

        fig = go.Figure(data=go.Heatmap(
            z=diff.values,
            x=diff.columns,
            y=diff.columns,
            colorscale="Reds",
            zmin=0,
            zmax=1,
            text=diff.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=14, color="black"),
            colorbar=dict(title="|Δ correlación|"),
        ))

        fig.update_layout(
            title="Diferencia de correlación (|train - test|)",
            template="plotly_dark",
            height=700,
            width=750,
            xaxis=dict(tickangle=45),
        )

        return fig

    def find_redundant_pairs(self, threshold: float = 0.8, dataset: str = "train") -> pd.DataFrame:
        """
        Identifica pares de indicadores altamente correlacionados.

        Útil como primer filtro de selección de features: si dos
        indicadores están muy correlacionados, probablemente aportan
        información redundante y se puede prescindir de uno de ellos.

        Args:
            threshold: Umbral absoluto de correlación a partir del cual
                un par se considera redundante.
            dataset: "train" o "test", sobre qué conjunto calcular la correlación.

        Returns:
            DataFrame con columnas [feature_1, feature_2, correlation],
            ordenado por correlación absoluta descendente.
        """
        corr = self.correlation_matrix(dataset)

        pairs = []
        for i, col_i in enumerate(corr.columns):
            for col_j in corr.columns[i + 1:]:
                value = corr.loc[col_i, col_j]
                if abs(value) >= threshold:
                    pairs.append({
                        "feature_1":   col_i,
                        "feature_2":   col_j,
                        "correlation": value,
                    })

        result = pd.DataFrame(pairs)
        if not result.empty:
            result = result.sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)

        return result