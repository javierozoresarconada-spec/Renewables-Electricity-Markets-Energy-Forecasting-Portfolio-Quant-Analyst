"""
DV01: Interest Rate Risk Sensitivity on a Treasury Fixed-Income Portfolio

This is a plain-.py mirror of DV01.ipynb, using PyCharm's "#%%" cell markers
(Scientific Mode). Each "#%%" starts a new cell you can run individually with
Ctrl+Enter (or the green run arrow next to it) inside the Python Console,
and inspect every variable afterwards in the Variables panel below the console.

See DV01.ipynb for the full write-up (markdown explanations + LaTeX formulas).
This file only carries the code, with short comments.
"""

#%% 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

plt.rcParams["figure.figsize"] = (10, 4)

#%% 2. Current 10Y US Treasury yield (Yahoo Finance, no API key needed)
tnx = yf.download("^TNX", period="5d", progress=False)["Close"]
current_10y_yield = tnx.iloc[-1].item() / 100  # ^TNX is quoted as yield x 10

print(f"Current 10Y US Treasury yield: {current_10y_yield:.4%}")

#%% 3. Define the hypothetical treasury fixed-income portfolio
portfolio = [
    {"name": "2Y Note",  "face": 20_000_000, "coupon": 0.040, "maturity": 2,  "yield": current_10y_yield - 0.0030},
    {"name": "5Y Note",  "face": 15_000_000, "coupon": 0.043, "maturity": 5,  "yield": current_10y_yield - 0.0010},
    {"name": "10Y Note", "face": 10_000_000, "coupon": 0.047, "maturity": 10, "yield": current_10y_yield},
]

pd.DataFrame(portfolio).set_index("name")

#%% 4. Bond pricing (present value of coupons + face value, semi-annual)
def bond_price(face, coupon_rate, maturity_years, yield_rate, freq=2):
    coupon = face * coupon_rate / freq
    n = int(maturity_years * freq)
    periods = np.arange(1, n + 1)
    discount_factors = 1 / (1 + yield_rate / freq) ** periods
    return np.sum(coupon * discount_factors) + face * discount_factors[-1]


for bond in portfolio:
    bond["price"] = bond_price(bond["face"], bond["coupon"], bond["maturity"], bond["yield"])

pd.DataFrame(portfolio).set_index("name")[["face", "coupon", "maturity", "yield", "price"]]

#%% 5. DV01 via bump-and-reprice, per bond and aggregated (DV01 is additive)
def dv01_bump(face, coupon_rate, maturity_years, yield_rate, freq=2, bump=0.0001):
    p0 = bond_price(face, coupon_rate, maturity_years, yield_rate, freq)
    p1 = bond_price(face, coupon_rate, maturity_years, yield_rate + bump, freq)
    return -(p1 - p0)


for bond in portfolio:
    bond["dv01"] = dv01_bump(bond["face"], bond["coupon"], bond["maturity"], bond["yield"])

dv01_df = pd.DataFrame(portfolio).set_index("name")[["face", "price", "dv01"]]
total_dv01 = dv01_df["dv01"].sum()

print(dv01_df.round(2))
print(f"\nTotal portfolio DV01: ${total_dv01:,.2f} per basis point")

#%% 6. Cross-check: DV01 via modified duration
def modified_duration(face, coupon_rate, maturity_years, yield_rate, freq=2):
    coupon = face * coupon_rate / freq
    n = int(maturity_years * freq)
    periods = np.arange(1, n + 1)
    cashflows = np.full(n, coupon)
    cashflows[-1] += face
    discount_factors = 1 / (1 + yield_rate / freq) ** periods
    pv_cashflows = cashflows * discount_factors
    price = pv_cashflows.sum()
    macaulay_duration = np.sum((periods / freq) * pv_cashflows) / price
    return macaulay_duration / (1 + yield_rate / freq), price


bond = portfolio[-1]  # 10Y Note, as an example
mod_duration, price_check = modified_duration(bond["face"], bond["coupon"], bond["maturity"], bond["yield"])
dv01_analytic = mod_duration * price_check * 0.0001

print(f"{bond['name']}: modified duration = {mod_duration:.3f} years")
print(f"DV01 (bump-and-reprice): ${bond['dv01']:,.2f}")
print(f"DV01 (modified duration formula): ${dv01_analytic:,.2f}")

#%% 7. Using DV01 to size a hedge (10Y swap-equivalent)
hedge_coupon = 0.047
hedge_yield = current_10y_yield
hedge_maturity = 10

hedge_dv01_per_100 = dv01_bump(100, hedge_coupon, hedge_maturity, hedge_yield)
hedge_notional = total_dv01 / hedge_dv01_per_100 * 100

print(f"Hedge instrument (10Y swap-equivalent) DV01 per 100 notional: ${hedge_dv01_per_100:.4f}")
print(f"Hedge notional needed: ${hedge_notional:,.0f} (pay-fixed / short position of this size)")

#%% 8. Keep all figures open when this file is run as a full script
if __name__ == "__main__":
    plt.show(block=True)