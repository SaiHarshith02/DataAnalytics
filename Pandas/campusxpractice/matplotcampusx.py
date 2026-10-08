# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.2",
#     "numpy>=2.5.3",
#     "pandas>=3.0.6",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="", auto_download=["ipynb"])


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _(np, plt):
    x=np.array([1,2,3,4,5,6,7])
    y=x**2
    plt.plot(y)
    return


@app.cell
def _():
    p_fassion=0.50
    p_electronics=0.40
    p_hd=0.1
    p_clkbyfacion=0.02
    p_clkbyele=0.05
    p_clkbyhd=0.1
    p_eleclked=((p_clkbyele)*p_electronics)/((p_fassion*p_clkbyfacion)+(p_electronics*p_clkbyele)+(p_hd*p_clkbyhd))
    print(p_eleclked)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
