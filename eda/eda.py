import pandas as pd

# Para poder graficar los datos tenemos que transformar el indice de los datos a formato DatetimeIndex
def prepare_data(df):
    # Transformamos a timestamp y convertimos en índice
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    #df = df.set_index("timestamp")

    # Eliminamos las columnas: close_time, quote_av, tb_base_av, tb_quote_av y ignore
    cols_to_drop = [
        "close_time",
        "quote_av",
        "tb_base_av",
        "tb_quote_av",
        "ignore"
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df