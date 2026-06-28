"""
Wrappers sobre optuna.visualization para inspeccionar el estudio de un
ModelOptimizer ya ejecutado. Todas las figuras devueltas son de Plotly,
consistentes con el resto del proyecto.
"""

import optuna.visualization as ov


def plot_optimization_history(optimizer):
    """Evolución del mejor valor encontrado a lo largo de los trials.

    Útil para ver si el estudio convergió o si seguía mejorando al
    terminar n_trials (en cuyo caso valdría la pena ampliar la búsqueda).

    Args:
        optimizer: ModelOptimizer ya optimizado (optimize() llamado).

    Returns:
        Figura de Plotly.
    """
    return ov.plot_optimization_history(optimizer.study)


def plot_param_importances(optimizer):
    """Importancia relativa de cada hiperparámetro/feature_idx en el resultado.

    Permite ver si la elección de ciertas features (feature_idx_N) pesa más
    que los hiperparámetros del modelo en el resultado final, o viceversa.

    Args:
        optimizer: ModelOptimizer ya optimizado.

    Returns:
        Figura de Plotly.
    """
    return ov.plot_param_importances(optimizer.study)


def plot_parallel_coordinate(optimizer, params: list = None):
    """Coordenadas paralelas: cómo se relacionan los valores de cada
    parámetro con el score final, todos los trials a la vez.

    Args:
        optimizer: ModelOptimizer ya optimizado.
        params: lista de nombres de parámetros a incluir. Si es None,
            incluye todos (puede saturarse visualmente si k_features es alto).

    Returns:
        Figura de Plotly.
    """
    return ov.plot_parallel_coordinate(optimizer.study, params=params)


def plot_slice(optimizer, params: list = None):
    """Un subplot por parámetro: su valor en cada trial vs el score obtenido.

    Útil para ver rangos "ganadores" de cada hiperparámetro individualmente
    (p.ej. si alpha siempre funciona mejor en valores bajos).

    Args:
        optimizer: ModelOptimizer ya optimizado.
        params: lista de nombres de parámetros a incluir. Si es None, todos.

    Returns:
        Figura de Plotly.
    """
    return ov.plot_slice(optimizer.study, params=params)


def plot_contour(optimizer, params: list):
    """Mapa de contorno 2D entre dos parámetros y el score.

    Args:
        optimizer: ModelOptimizer ya optimizado.
        params: lista de exactamente 2 nombres de parámetros.

    Returns:
        Figura de Plotly.
    """
    if len(params) != 2:
        raise ValueError("plot_contour necesita exactamente 2 parámetros")
    return ov.plot_contour(optimizer.study, params=params)


def show_all(optimizer, params: list = None):
    """Muestra de golpe historial, importancias, coordenadas paralelas y slice.

    Args:
        optimizer: ModelOptimizer ya optimizado.
        params: lista de parámetros a usar en los gráficos que lo requieren
            (parallel_coordinate, slice). Si es None, se usan todos.
    """
    plot_optimization_history(optimizer).show()
    plot_param_importances(optimizer).show()
    plot_parallel_coordinate(optimizer, params=params).show()
    plot_slice(optimizer, params=params).show()