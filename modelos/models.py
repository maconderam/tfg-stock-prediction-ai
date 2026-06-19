from abc import ABC, abstractmethod
import pickle
import numpy as np
import pandas as pd


class Model(ABC):
    """
    Clase base abstracta para la gestión del ciclo de vida de modelos predictivos.

    Proporciona una interfaz unificada para entrenar, predecir, almacenar métricas,
    resumir el estado del modelo y gestionar la persistencia en disco mediante serialización.
    """

    def __init__(self, name: str):
        """
        Inicializa los componentes base del modelo.

        Args:
            name (str): Nombre identificativo del modelo o algoritmo.
        """
        self.name = name
        self.metrics = {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Método abstracto encargado de ajustar o entrenar el modelo.

        Args:
            X (pd.DataFrame): Matriz de características (features).
            y (pd.Series): Vector o serie con la variable objetivo (target).
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Método abstracto encargado de generar predicciones a partir de nuevos datos.

        Args:
            X (pd.DataFrame): Matriz de características a evaluar.

        Returns:
            np.ndarray: Arreglo de NumPy con los valores o clases predichas.
        """
        pass

    def add_metric(self, key: str, value):
        """
        Registra o actualiza una métrica de rendimiento en el historial del modelo.

        Args:
            key (str): Nombre de la métrica (ej. 'accuracy', 'mse', 'sharpe_ratio').
            value (Any): Valor numérico o descriptor de la métrica obtenida.
        """
        self.metrics[key] = value

    def summary(self):
        """Imprime por consola un resumen detallado del modelo, su estado y métricas."""
        print("-" * 20)
        print(f"Model: {self.name}")
        print(f"Fitted: {self.is_fitted}")

        if not self.metrics:
            print("Metrics: none")
            return

        print("Metrics:")
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    def save(self, path: str):
        """
        Serializa el objeto completo y lo guarda en disco usando Pickle.

        Args:
            path (str): Ruta del archivo de destino (ej. 'models/random_forest.pkl').
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """
        Carga y deserializa un modelo previamente guardado en disco.

        Args:
            path (str): Ruta del archivo `.pkl` a leer.

        Returns:
            Model: Instancia de la clase recuperada con su estado e historial intactos.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model


class SklearnModel(Model):
    """
    Wrapper especializado para integrar cualquier estimador compatible con Scikit-Learn.

    Adapta la API estándar de sklearn a la arquitectura base del proyecto, heredando
    las funciones de reporte estadístico y persistencia de datos de la clase Model.
    """

    def __init__(self, model, name: str = None):
        """
        Inicializa el wrapper asociando el estimador de sklearn.

        Args:
            model (Estimator): Instancia de un modelo de Scikit-Learn (ej. RandomForestClassifier()).
            name (str, optional): Nombre personalizado. Si no se provee, se extrae el 
                nombre intrínseco de la clase del estimador. Defaults to None.
        """
        model_name = name or type(model).__name__
        super().__init__(name=model_name)
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena el estimador interno de Scikit-Learn.

        Args:
            X (pd.DataFrame): Matriz de características.
            y (pd.Series): Vector objetivo.

        Returns:
            SklearnModel: Retorna la propia instancia del objeto entrenado.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Genera predicciones con el modelo entrenado.

        Args:
            X (pd.DataFrame): Datos de entrada.

        Returns:
            np.ndarray: Predicciones generadas por el estimador de sklearn.

        Raises:
            RuntimeError: Si se intenta predecir antes de ejecutar el entrenamiento (`fit`).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        """
        Extrae y ordena la importancia de las variables de forma descendente.

        Asocia los pesos o importancias numéricas calculadas por el estimador
        con los nombres de las columnas del DataFrame original.

        Returns:
            pd.Series: Serie de Pandas indexada por el nombre de la característica.

        Raises:
            RuntimeError: Si el modelo no ha sido entrenado.
            AttributeError: Si el estimador interno no soporta o calcula la métrica 
                `feature_importances_` (ej. modelos lineales que usan coeficientes).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before feature_importance()")

        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError(
                f"{self.name} does not expose feature_importances_"
            )

        # Retorna la serie ordenada para facilitar la selección o descarte de indicadores
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.feature_names_in_
        ).sort_values(ascending=False)