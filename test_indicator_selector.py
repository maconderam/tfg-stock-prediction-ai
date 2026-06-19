from indicadores.indicator_selector import IndicatorSelector
from eda.eda import prepare_data
import pandas as pd
import matplotlib.pyplot as plt

path = "datos/crudo/1d/BTCUSDT_1d_01-01-2016_18-01-2026.csv"

df = pd.read_csv(path, sep=",")
df = prepare_data(df)

sel = IndicatorSelector(df, n_mcpt=200, p_threshold=0.05)
df = sel.run()                        # grid search completo
print(sel.top_n(10))                  # top 10 significativos

fig1 = sel.plot_summary(top_n=20)     # dashboard comparativo
plt.show()

fig2 = sel.plot_mc_distributions(top_n=6)  # distribuciones MC
plt.show()