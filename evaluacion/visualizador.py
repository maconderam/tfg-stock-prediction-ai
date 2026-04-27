import mplfinance as mpf
import pandas as pd
import matplotlib as mpl

class Visualizer:
    def __init__(self, data):
        self.data = data.copy()
    
    def update_data(self, data):
        self.data = data.copy()
    
    def plot_candles(self):
        mpf.plot(
            self.data,
            type="candle",
            style="charles",
            volume=False,
            title="Candlestick Chart"
        )

    def plot_with_indicators(self, overlays=[], panels=[]):
        addplots = []

        for col in overlays:
            addplots.append(
                mpf.make_addplot(self.data[col], panel=0)
            )

        panel_index = 1
        for col in panels:
            addplots.append(
                mpf.make_addplot(self.data[col], panel=panel_index, ylabel=col)
            )
            panel_index += 1

        mpf.plot(
            self.data,
            type="candle",
            addplot=addplots,
            volume=False,
            style="charles",
            panel_ratios=(3,) + (1,) * len(panels),
            figsize=(14, 8)
        )