# %% IMPORT PACKAGES

import pandas as pd
import numpy as np
import os

# %% READ DATA

# From Excel Assignment 1 IEEE 24 bus data
script_dir = os.path.dirname(os.path.abspath(__file__))
path_A1_data = os.path.join(script_dir, "data", "Assignment1_IEEE24bus_input_data.xlsx")
A1_data_file=pd.read_excel(path_A1_data, sheet_name=None) # Gives back all sheets
gens = A1_data_file["Conventional_generators"]
wind = A1_data_file["Wind_farms"]
demands = A1_data_file["Demands"]
lines = A1_data_file["Transmission_lines"]

# %% STEP 1: MARKET CLEARING PRICE
#
# Model: copper-plate (single node, no network), single hour, price-elastic demand.
#
# Optimization problem (Lecture 2, slides 85-89):
#
#   Maximize   SW = sum_d ( U_d * p_d ) - sum_g ( C_g * p_g )
#   p_d, p_g
#
#   subject to:
#       0 <= p_g <= P_g_max          for every producer g (conventional + wind)   : mu_g_low, mu_g_up
#       0 <= p_d <= P_d_max          for every demand d                            : mu_d_low, mu_d_up
#       sum_d p_d - sum_g p_g = 0                                                  : lambda
#
#   where:
#     U_d  = bid price of demand d      [$/MWh]
#     C_g  = offer price of producer g  [$/MWh]
#     P_g_max = capacity (or day-ahead forecast for wind) of producer g [MW]
#     P_d_max = consumption (max load) of demand d [MW]
#     lambda  = dual variable of the power-balance constraint = market-clearing price

from scipy.optimize import linprog

# --- Step 1.1: build the supply side (offer curve) ---
# Producers = conventional generators (offer price = production cost) + wind farms
# (offer price = 0, quantity = day-ahead forecast, as stated in the assignment).
supplier_names = list(gens["Generator"]) + list(wind["Wind_farm"])
supplier_cost = np.concatenate([
    gens["Production_cost_USD_per_MWh"].to_numpy(),
    np.zeros(len(wind)),
])
supplier_capacity = np.concatenate([
    gens["Capacity_MW"].to_numpy(),
    wind["Day_ahead_forecast_MW"].to_numpy(),
])
n_sup = len(supplier_names)

# --- Step 1.2: build the demand side (bid curve) ---
# Assumption: the demand bid price = curtailment cost (500 $/MWh), which is much
# higher than any generation offer price, as requested by the assignment
# ("use comparatively high values relative to the generation cost").
demand_names = list(demands["Demand"])
demand_bid = demands["Curtailment_cost_USD_per_MWh"].to_numpy()
demand_maxload = demands["Consumption_MW"].to_numpy()
n_dem = len(demand_names)

# --- Step 1.3: decision vector x = [p_sup (n_sup) , p_dem (n_dem)] ---
# scipy.optimize.linprog only MINIMIZES, so we minimize -SW instead of maximizing SW:
#   minimize   sum_g C_g * p_g  -  sum_d U_d * p_d
c = np.concatenate([supplier_cost, -demand_bid])

# Bounds: 0 <= p_sup <= capacity , 0 <= p_dem <= max load
bounds = [(0, cap) for cap in supplier_capacity] + [(0, load) for load in demand_maxload]

# --- Step 1.4: power balance equality constraint ---
#   sum_d p_d - sum_g p_sup = 0
A_eq = np.concatenate([-np.ones(n_sup), np.ones(n_dem)]).reshape(1, -1)
b_eq = np.array([0.0])

# --- Step 1.5: solve the LP ---
result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

if not result.success:
    raise RuntimeError(f"Market-clearing LP did not solve: {result.message}")

p_sup_opt = result.x[:n_sup]
p_dem_opt = result.x[n_sup:]

dispatch_gens = pd.Series(p_sup_opt, index=supplier_names, name="Dispatch_MW")
dispatch_dem = pd.Series(p_dem_opt, index=demand_names, name="Consumption_MW")

# --- Step 1.6: social welfare achieved ---
social_welfare = -result.fun  # result.fun = min(-SW) = -SW*, so SW* = -result.fun

# --- Step 1.7: market-clearing price ---
# Method A: dual variable (Lagrange multiplier) of the power-balance constraint.
# Since we minimized f = -SW subject to (sum_d p_d - sum_sup p_sup) = 0,
# d(min f)/d(b_eq) = -d(SW*)/d(b_eq) = -lambda  =>  lambda = -marginal
price_from_dual = -result.eqlin.marginals[0]

# Method B (KKT verification, Lecture 2 slides 73-77): the market-clearing price
# equals the offer price of the "marginal" producer, i.e. the price-setting unit
# that is dispatched strictly between 0 and its capacity (neither constraint binds).
tol = 1e-6
is_marginal = (p_sup_opt > tol) & (p_sup_opt < supplier_capacity - tol)
marginal_unit = np.array(supplier_names)[is_marginal]
marginal_price = supplier_cost[is_marginal]

print("Dispatch of producers (MW):")
print(dispatch_gens.round(2))
print("\nDispatch of demands (MW):")
print(dispatch_dem.round(2))
print(f"\nTotal social welfare: {social_welfare:,.2f} $")
print(f"Market-clearing price (dual variable, Method A): {price_from_dual:.2f} $/MWh")
print(f"Marginal producer (Method B, KKT check): {marginal_unit} at {marginal_price} $/MWh")

# %% STEP 1: PROFIT OF EACH PRODUCER AND UTILITY OF EACH DEMAND
#
# Once we have the market-clearing price (price_from_dual), the profit/utility of
# every market participant is just arithmetic on the dispatch we already solved for.

# Profit of producer g = revenue - cost = (price - offer_price) * dispatched_quantity
producer_profit = (price_from_dual - supplier_cost) * p_sup_opt

# Utility of demand d = consumption * (bid_price - price), as defined in the assignment
demand_utility = p_dem_opt * (demand_bid - price_from_dual)

producers_summary = pd.DataFrame({
    "Offer_price_USD_per_MWh": supplier_cost,
    "Dispatch_MW": p_sup_opt,
    "Profit_USD": producer_profit,
}, index=supplier_names).round(2)

demands_summary = pd.DataFrame({
    "Bid_price_USD_per_MWh": demand_bid,
    "Consumption_MW": p_dem_opt,
    "Utility_USD": demand_utility,
}, index=demand_names).round(2)

print("\nProducers: offer price, dispatch and profit")
print(producers_summary)
print("\nDemands: bid price, consumption and utility")
print(demands_summary)
print(f"\nCheck -> sum(producer profit) + sum(demand utility) = {producer_profit.sum() + demand_utility.sum():,.2f} $")
print(f"      -> total social welfare                        = {social_welfare:,.2f} $")

# %% STEP 1: VERIFY THE MARKET-CLEARING PRICE VIA THE KKT CONDITIONS
#
# How to read this section: we take the price found above (price_from_dual) and
# check, for EVERY producer and demand, that it is consistent with the KKT
# stationarity + complementary-slackness conditions of the LP -- i.e. that no
# market participant would want to change its dispatch given that price.
#
# For a producer g:  lambda = C_g - mu_g_low + mu_g_up
# For a demand d:     lambda = U_d + mu_d_low - mu_d_up
# where mu_low, mu_up >= 0 are the multipliers of the lower/upper dispatch bounds.
# Complementary slackness requires mu_low * p = 0 and mu_up * (Pbar - p) = 0, i.e.
# a bound's multiplier can only be positive if that same bound is actually active.

# Implied multipliers (only one of the pair can be non-zero for each unit):
mu_sup_up = np.maximum(price_from_dual - supplier_cost, 0.0)   # active if producer is fully dispatched
mu_sup_low = np.maximum(supplier_cost - price_from_dual, 0.0)  # active if producer is not dispatched at all
mu_dem_up = np.maximum(demand_bid - price_from_dual, 0.0)      # active if demand is fully served
mu_dem_low = np.maximum(price_from_dual - demand_bid, 0.0)     # active if demand is not served at all

# Complementary-slackness residuals: should be ~0 (within solver tolerance) if the
# price and dispatch are truly a KKT point.
cs_tol = 1e-3
cs_sup = np.maximum(mu_sup_low * p_sup_opt, mu_sup_up * (supplier_capacity - p_sup_opt))
cs_dem = np.maximum(mu_dem_low * p_dem_opt, mu_dem_up * (demand_maxload - p_dem_opt))

kkt_verified = bool(np.all(cs_sup < cs_tol) and np.all(cs_dem < cs_tol))

print(f"\nMax complementary-slackness residual (producers): {cs_sup.max():.6f}")
print(f"Max complementary-slackness residual (demands):    {cs_dem.max():.6f}")
print(f"KKT conditions verified for price = {price_from_dual:.2f} $/MWh: {kkt_verified}")
