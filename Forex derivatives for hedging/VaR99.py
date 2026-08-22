"""
Value at Risk (VaR) 99% on an FX Exposure Book -- Parametric vs. Monte Carlo

This is a plain-.py mirror of VaR99.ipynb, using PyCharm's "#%%" cell markers
(Scientific Mode). Each "#%%" starts a new cell you can run individually with
Ctrl+Enter (or the green run arrow next to it) inside the Python Console,
and inspect every variable afterwards in the Variables panel below the console.

See VaR99.ipynb for the full write-up (markdown explanations + LaTeX formulas).
This file only carries the code, with short comments.
"""

#%% 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, chi2

plt.rcParams["figure.figsize"] = (10, 4)
plt.ion()  # interactive mode: plt.show() below returns immediately instead of blocking the console
np.random.seed(42)

market_days = 252  # approx. number of trading days in a year, used to annualise daily volatility

#%% 2. Download FX data (Yahoo Finance, no API key needed)
tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]

end_date = pd.Timestamp.today().normalize()
start_date = end_date - pd.DateOffset(years=3)

# Downloaded one ticker at a time (more robust than a single multi-ticker call with yfinance)
series = {
    t: yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"][t]
    for t in tickers
}
prices = pd.DataFrame(series).dropna() # Create dataframe removing missing values
prices.columns = ["EURUSD", "GBPUSD", "USDJPY"]

print(f"{len(prices)} daily observations, {prices.index[0].date()} to {prices.index[-1].date()}")
prices.tail()

#%% 2b. Plot the FX spot rates
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, col in zip(axes, prices.columns):
    ax.plot(prices.index, prices[col], color="b")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("FX spot rates used in this analysis")
fig.tight_layout()
plt.show()

#%% 3. Define the hypothetical FX exposure book (USD reporting currency)
spot = prices.iloc[-1]

notional_eur = 10_000_000   # long EUR receivable
notional_gbp = 6_000_000    # long GBP receivable
notional_jpy = 500_000_000  # short JPY payable

exposures = pd.Series({
    "EURUSD": notional_eur * spot["EURUSD"],            # long EUR -> long EURUSD
    "GBPUSD": notional_gbp * spot["GBPUSD"],            # long GBP -> long GBPUSD
    "USDJPY": notional_jpy / spot["USDJPY"],            # short JPY payable -> long USDJPY (inverted quoting, see notebook Section 3)
}, name="USD exposure")

e = exposures.values  # dollar-delta vector, one entry per FX rate

display_df = exposures.to_frame() # Convert series to dataframe
display_df["% of gross exposure"] = 100 * display_df["USD exposure"] / display_df["USD exposure"].abs().sum()
display_df

#%% 4. Daily log returns
returns = np.log(prices / prices.shift(1)).dropna()

print("Annualised volatility (%):")
print((returns.std() * np.sqrt(market_days) * 100).round(2))

print("\nCorrelation matrix:")
returns.corr().round(2)

#%% 5. Method 1 -- Parametric (Variance-Covariance) VaR
Sigma = returns.cov().values  # daily covariance matrix of log returns

sigma_p = np.sqrt(e @ Sigma @ e)  # portfolio daily P&L volatility, in USD

z_99 = norm.ppf(0.99)
var_parametric = z_99 * sigma_p

gross_exposure = np.abs(e).sum()
print(f"Portfolio daily P&L volatility: ${sigma_p:,.0f}")
print(f"z_0.99 = {z_99:.4f}")
print(f"Parametric 1-day VaR99: ${var_parametric:,.0f}  "
      f"({100 * var_parametric / gross_exposure:.2f}% of gross exposure)")

#%% 6. Method 2 -- Monte Carlo VaR
n_sims = 100_000
L = np.linalg.cholesky(Sigma)

Z = np.random.standard_normal((n_sims, len(e)))
sim_returns = Z @ L.T                 # correlated simulated daily returns
sim_pnl = sim_returns @ e             # simulated portfolio P&L, in USD

var_montecarlo = -np.percentile(sim_pnl, 1)

print(f"Monte Carlo 1-day VaR99 ({n_sims:,} scenarios): ${var_montecarlo:,.0f}  "
      f"({100 * var_montecarlo / gross_exposure:.2f}% of gross exposure)")

#%% 6b. Plot simulated P&L distribution vs. parametric normal pdf
fig, ax = plt.subplots()
ax.hist(sim_pnl, bins=100, density=True, color="#2C6E91", alpha=0.7, label="Simulated P&L")

x = np.linspace(sim_pnl.min(), sim_pnl.max(), 300)
ax.plot(x, norm.pdf(x, 0, sigma_p), color="#B23A48", lw=2, label="Parametric normal pdf")

ax.axvline(-var_montecarlo, color="#B23A48", ls="--", lw=1.5, label=f"VaR99 (MC) = ${var_montecarlo:,.0f}")
ax.set_title("Simulated 1-day portfolio P&L distribution")
ax.set_xlabel("P&L (USD)")
ax.legend()
plt.show()

#%% 7. Compare both methods
comparison = pd.DataFrame({
    "VaR99 (USD)": [var_parametric, var_montecarlo],
    "% of gross exposure": [100 * var_parametric / gross_exposure, 100 * var_montecarlo / gross_exposure],
}, index=["Parametric (Variance-Covariance)", "Monte Carlo"])

comparison.round(2)

#%% 8. Backtesting -- Kupiec Proportion-of-Failures (POF) test
window = 250
pnl_history = returns.values @ e  # realised daily P&L implied by the current book, over the whole history

var_rolling = np.full(len(returns), np.nan)
for t in range(window, len(returns) - 1):
    Sigma_t = np.cov(returns.values[t - window:t].T)
    var_rolling[t] = z_99 * np.sqrt(e @ Sigma_t @ e)

# Align each VaR forecast (made at t, using data up to t) with the *next* day's realised P&L
valid = ~np.isnan(var_rolling[:-1])
dates_valid = returns.index[1:][valid]
pnl_valid = pnl_history[1:][valid]
var_valid = var_rolling[:-1][valid]

exceptions_mask = pnl_valid < -var_valid
n_preds = valid.sum()
n_exceptions = int(exceptions_mask.sum())
exception_rate = n_exceptions / n_preds

print(f"Predictions: {n_preds}, Exceptions: {n_exceptions}, Observed rate: {exception_rate:.2%} (target: 1.00%)")

#%% 8b. Kupiec likelihood-ratio test
p, n, x = 0.01, n_preds, n_exceptions
if x == 0:
    lr_pof = -2 * n * np.log(1 - p)
else:
    lr_pof = -2 * np.log(((1 - p) ** (n - x)) * (p ** x)) \
             + 2 * np.log(((1 - x / n) ** (n - x)) * ((x / n) ** x))

critical_value = chi2.ppf(0.95, df=1)
reject_model = lr_pof > critical_value

print(f"Kupiec LR statistic: {lr_pof:.3f}")
print(f"Chi-squared(1) 95% critical value: {critical_value:.3f}")
print("=> Reject model calibration" if reject_model else "=> Do NOT reject: VaR model is adequately calibrated")

#%% 8c. Plot backtest: realised P&L vs. rolling VaR99
fig, ax = plt.subplots()
ax.plot(dates_valid, pnl_valid, color="#2C6E91", lw=0.8, label="Realised daily P&L")
ax.plot(dates_valid, -var_valid, color="#B23A48", lw=1.2, label="Rolling VaR99 (loss threshold)")
ax.scatter(dates_valid[exceptions_mask], pnl_valid[exceptions_mask],
           color="#B23A48", zorder=5, s=25, label="Exceptions")
ax.set_title("Backtest: realised P&L vs. rolling parametric VaR99")
ax.set_ylabel("USD")
ax.legend()
plt.show()

#%% 9. Keep all figures open when this file is run as a full script
# (not needed when running cell-by-cell in the console, since the console stays
# alive on its own -- this only matters if you hit "Run" on the whole file.)
if __name__ == "__main__":
    plt.show(block=True)
