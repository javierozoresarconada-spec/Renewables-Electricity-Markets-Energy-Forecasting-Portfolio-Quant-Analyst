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

# %% STEP 2: COPPER-PLATE, MULTIPLE HOURS (24H) WITH STORAGE
#
# How to read this section: Step 1 cleared a single hour. Step 2 clears 24 hours
# at once (one single LP, not 24 separate ones), because a storage unit couples
# the hours together -- what it does in hour t affects what it can do in hour t+1.
#
# What changes vs. Step 1:
#   - Wind capacity and demand (bid price + max load) now vary hour by hour.
#     The assignment does not give us hourly data, so we build 24-hour profiles
#     ourselves (Step 2.1) and apply them to the Step-1 values (Step 2.2).
#   - A storage unit (e.g. pumped hydro) is added. It bids/offers at zero price
#     (Note 2 of the assignment), so it never appears in the objective function,
#     but it does add new variables (charge/discharge power, energy stored) and
#     new constraints (Eq. 1-4 of the assignment) that link hour t to hour t-1.

# --- Step 2.1: build 24-hour profiles (load, wind, demand bid price) ---
# These are "shape" functions: each returns an array of multipliers with a mean
# of 1, so that "base_value * shape" varies the Step-1 base value hour by hour
# while keeping its daily average unchanged. Written as functions (not inlined)
# because we will reuse the same shapes again in later assignments/steps.

def make_daily_load_profile(hours, morning_peak=9, evening_peak=19, spread=3.0,
                             base=0.6, morning_weight=0.4, evening_weight=0.5):
    """Stylized double-peak daily electricity demand shape (morning + evening
    peaks, low overnight), as two Gaussian bumps on top of a flat base load."""
    profile = (
        base
        + morning_weight * np.exp(-0.5 * ((hours - morning_peak) / spread) ** 2)
        + evening_weight * np.exp(-0.5 * ((hours - evening_peak) / spread) ** 2)
    )
    return profile / profile.mean()


def make_wind_profile(n_hours, seed=1, volatility=0.15, low=0.2, high=1.6):
    """Smooth but non-trivial 24-hour wind-availability shape: a seeded random
    walk (reproducible), clipped so it never collapses to 0 or explodes."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=volatility, size=n_hours)
    walk = 1.0 + np.cumsum(steps - steps.mean())
    walk = np.clip(walk, low, high)
    return walk / walk.mean()


T = 24
hours = np.arange(1, T + 1)

load_shape = make_daily_load_profile(hours)
wind_shape = make_wind_profile(T)
# Assumption: demand bid prices track the load shape (higher during peak hours,
# as requested by the assignment), compressed to +-15% so they always stay far
# above every generator's offer price (max 26.11 $/MWh), exactly as in Step 1.
bid_price_shape = 1.0 + 0.15 * (load_shape - 1.0)

# --- Step 2.2: apply the profiles to the Step-1 base values ---
# Conventional capacity does NOT vary by hour (only wind and demand do, per the
# assignment). n_sup, n_dem, supplier_cost, demand_bid, demand_maxload, gens and
# wind all come from Step 1 above -- nothing is re-read from Excel here.
supplier_capacity_th = np.zeros((n_sup, T))
supplier_capacity_th[: len(gens), :] = gens["Capacity_MW"].to_numpy()[:, None]
supplier_capacity_th[len(gens):, :] = wind["Day_ahead_forecast_MW"].to_numpy()[:, None] * wind_shape[None, :]

demand_bid_th = demand_bid[:, None] * bid_price_shape[None, :]
demand_maxload_th = demand_maxload[:, None] * load_shape[None, :]

# --- Step 2.3: storage (pumped-hydro-like unit) parameters ---
# Not part of the IEEE 24-bus data, so -- per the assignment's own instruction to
# "select reasonable arbitrary values" when data is missing -- we size it on the
# order of a mid-sized conventional unit (assignment's Note 1).
P_ch_max = 150.0    # MW,  charging capacity                        -- Eq. (1)
P_dis_max = 150.0   # MW,  discharging capacity                     -- Eq. (2)
E_max = 600.0       # MWh, energy capacity (4h of full-power duration) -- Eq. (3)
eta_ch = 0.90       # charging efficiency
eta_dis = 0.95      # discharging efficiency (eta_ch < eta_dis < 1, as required)
e0 = 0.5 * E_max    # energy stored at the start of hour 1


# --- Step 2.4: the market-clearing function ---
# We write this ONCE and call it twice (with storage, and with storage disabled),
# instead of duplicating the whole LP -- this guarantees both runs use exactly
# the same market logic, and the comparison in Step 2.5 is apples-to-apples.
def solve_multi_hour_market(supplier_cost, supplier_capacity_th, demand_bid_th, demand_maxload_th,
                             P_ch_max, P_dis_max, E_max, eta_ch, eta_dis, e0):
    """Clear a T-hour copper-plate market with one storage unit.

    Passing P_ch_max = P_dis_max = E_max = 0 removes storage from the market (its
    own bounds then force p_ch = p_dis = e = 0 in every hour), so this same
    function solves both the "with storage" and "without storage" cases.

    Returns a dict with the optimal dispatch, the hourly prices and the social
    welfare achieved.
    """
    n_sup, T = supplier_capacity_th.shape
    n_dem = demand_bid_th.shape[0]

    # Variable layout: [ p_sup (n_sup*T) | p_dem (n_dem*T) | p_ch (T) | p_dis (T) | e (T) ]
    i_sup = 0
    i_dem = i_sup + n_sup * T
    i_ch = i_dem + n_dem * T
    i_dis = i_ch + T
    i_e = i_dis + T
    n_var = i_e + T

    # Objective: minimize sum_t( sum_g C_g p_g,t - sum_d U_d,t p_d,t ). Storage has
    # no bid/offer price, so its variables get a zero coefficient (Note 2).
    c = np.concatenate([
        np.repeat(supplier_cost, T),
        -demand_bid_th.flatten(),
        np.zeros(3 * T),
    ])

    bounds = (
        [(0, cap) for cap in supplier_capacity_th.flatten()]
        + [(0, load) for load in demand_maxload_th.flatten()]
        + [(0, P_ch_max)] * T
        + [(0, P_dis_max)] * T
        + [(0, E_max)] * T
    )

    # Equality constraints: T power-balance rows, then T storage energy-balance rows.
    A_eq = np.zeros((2 * T, n_var))
    b_eq = np.zeros(2 * T)

    for t in range(T):
        # Power balance in hour t:  sum_d p_d,t + p_ch,t = sum_g p_g,t + p_dis,t : lambda_t
        A_eq[t, i_sup + t: i_sup + n_sup * T: T] = -1.0
        A_eq[t, i_dem + t: i_dem + n_dem * T: T] = 1.0
        A_eq[t, i_ch + t] = 1.0
        A_eq[t, i_dis + t] = -1.0

        # Storage energy balance in hour t (Eq. 4): e_t = e_{t-1} + eta_ch*p_ch,t - p_dis,t/eta_dis
        row = T + t
        A_eq[row, i_e + t] = 1.0
        A_eq[row, i_ch + t] = -eta_ch
        A_eq[row, i_dis + t] = 1.0 / eta_dis
        if t == 0:
            b_eq[row] = e0  # e_0 is a fixed parameter, not a variable -> moves to the RHS
        else:
            A_eq[row, i_e + t - 1] = -1.0

    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Multi-hour market-clearing LP did not solve: {result.message}")

    price_t = -result.eqlin.marginals[:T]  # same sign convention as Step 1 (Method A)

    return {
        "p_sup": result.x[i_sup:i_dem].reshape(n_sup, T),
        "p_dem": result.x[i_dem:i_ch].reshape(n_dem, T),
        "p_ch": result.x[i_ch:i_dis],
        "p_dis": result.x[i_dis:i_e],
        "e": result.x[i_e:i_e + T],
        "price_t": price_t,
        "social_welfare": -result.fun,
    }


# --- Step 2.5: solve the 24-hour market, with and without storage ---
with_storage = solve_multi_hour_market(
    supplier_cost, supplier_capacity_th, demand_bid_th, demand_maxload_th,
    P_ch_max, P_dis_max, E_max, eta_ch, eta_dis, e0,
)
without_storage = solve_multi_hour_market(
    supplier_cost, supplier_capacity_th, demand_bid_th, demand_maxload_th,
    P_ch_max=0.0, P_dis_max=0.0, E_max=0.0, eta_ch=eta_ch, eta_dis=eta_dis, e0=0.0,
)

prices_table = pd.DataFrame({
    "Hour": hours,
    "Price_with_storage": with_storage["price_t"],
    "Price_without_storage": without_storage["price_t"],
})
prices_table["Price_difference"] = prices_table["Price_with_storage"] - prices_table["Price_without_storage"]

print("\nHourly market-clearing prices, with vs. without storage ($/MWh):")
print(prices_table.round(2).to_string(index=False))

print(f"\nTotal social welfare with storage:    {with_storage['social_welfare']:,.2f} $")
print(f"Total social welfare without storage: {without_storage['social_welfare']:,.2f} $")
print(f"Welfare gain from adding storage:      {with_storage['social_welfare'] - without_storage['social_welfare']:,.2f} $")

price_range_with = with_storage["price_t"].max() - with_storage["price_t"].min()
price_range_without = without_storage["price_t"].max() - without_storage["price_t"].min()
print(f"\nPrice spread with storage:    {price_range_with:.2f} $/MWh "
      f"(min {with_storage['price_t'].min():.2f}, max {with_storage['price_t'].max():.2f})")
print(f"Price spread without storage: {price_range_without:.2f} $/MWh "
      f"(min {without_storage['price_t'].min():.2f}, max {without_storage['price_t'].max():.2f})")

# %% STEP 2: PROFIT OF EACH PRODUCER AND OF THE STORAGE UNIT, OVER 24 HOURS
#
# Same idea as Step 1's profit/utility section, just summed over the 24 hours
# instead of computed for a single one. Written as functions since we now have
# a genuine repeated pattern (one calculation per hour, for every unit).

def total_profit_over_horizon(price_t, cost, dispatch_gt):
    """Total profit of each producer over T hours: sum_t (price_t - cost) * dispatch_g,t."""
    return dispatch_gt @ price_t - cost * dispatch_gt.sum(axis=1)


def total_utility_over_horizon(price_t, bid_dt, dispatch_dt):
    """Total utility of each demand over T hours: sum_t dispatch_d,t * (bid_d,t - price_t)."""
    return (dispatch_dt * bid_dt).sum(axis=1) - dispatch_dt @ price_t


producer_profit_24h = total_profit_over_horizon(with_storage["price_t"], supplier_cost, with_storage["p_sup"])
demand_utility_24h = total_utility_over_horizon(with_storage["price_t"], demand_bid_th, with_storage["p_dem"])

# Storage profit: revenue from discharging minus the cost of charging, both paid/
# charged at the hourly price -- a "temporal arbitrager" buying energy when it is
# cheap and selling it back when it is expensive (cf. Lecture 3's "spatial
# arbitrager" congestion-rent example, same idea across time instead of space).
storage_profit_24h = float(with_storage["price_t"] @ (with_storage["p_dis"] - with_storage["p_ch"]))

producers_summary_24h = pd.DataFrame({
    "Offer_price_USD_per_MWh": supplier_cost,
    "Total_dispatch_MWh": with_storage["p_sup"].sum(axis=1),
    "Total_profit_USD": producer_profit_24h,
}, index=supplier_names).round(2)

print("\nProducers: total dispatch and profit over 24 hours (with storage)")
print(producers_summary_24h)
print(f"\nTotal storage profit over 24 hours: {storage_profit_24h:,.2f} $")
print(f"Check -> sum(producer profit) + sum(demand utility) + storage profit = "
      f"{producer_profit_24h.sum() + demand_utility_24h.sum() + storage_profit_24h:,.2f} $")
print(f"      -> total social welfare (with storage)                        = "
      f"{with_storage['social_welfare']:,.2f} $")

# %% STEP 2: DOES THE PRICE ALWAYS EQUAL THE MARGINAL PRODUCER'S OFFER PRICE?
#
# Same "marginal unit" check as Step 1 (Method B), repeated for every one of the
# 24 hours, to see whether adding storage ever breaks the usual merit-order rule.

tol = 1e-6
price_matches_generator = np.zeros(T, dtype=bool)
for t in range(T):
    dispatch_t = with_storage["p_sup"][:, t]
    capacity_t = supplier_capacity_th[:, t]
    marginal_t = (dispatch_t > tol) & (dispatch_t < capacity_t - tol)
    if marginal_t.any():
        price_matches_generator[t] = bool(np.any(np.isclose(supplier_cost[marginal_t], with_storage["price_t"][t], atol=1e-2)))

check_table = pd.DataFrame({
    "Hour": hours,
    "Price": with_storage["price_t"].round(2),
    "Matches_a_generator_cost": price_matches_generator,
    "Storage_charging_MW": with_storage["p_ch"].round(2),
    "Storage_discharging_MW": with_storage["p_dis"].round(2),
    "Energy_stored_MWh": with_storage["e"].round(2),
})
print("\nDoes the hourly price equal a generator's offer price?")
print(check_table.to_string(index=False))
print(f"\nHours where the price does NOT match any generator's offer price: "
      f"{int((~price_matches_generator).sum())} / {T}")

# %% STEP 3: NETWORK CONSTRAINTS (NODAL MARKET CLEARING)
#
# How to read this section: Step 3 extends STEP 1 (single hour, no storage --
# not Step 2), replacing the single "copper-plate" system-wide power balance
# with the real IEEE 24-bus transmission network (Lecture 3). Every producer
# and demand is now pinned to a specific node ("Location_node" in the Excel
# data), and power can only move between nodes through transmission lines,
# each with its own capacity limit.

# --- Step 3.1: map every producer/demand to its node, and load line data ---
# Node labels in the data are 1..24; we convert to 0-indexed positions (0..23)
# to use directly as array/matrix indices.
n_nodes = 24
ref_node = 0  # node 1 (index 0) is the reference/slack bus: theta_ref = 0

supplier_node = np.concatenate([
    gens["Location_node"].to_numpy(),
    wind["Location_node"].to_numpy(),
]) - 1
demand_node = demands["Location_node"].to_numpy() - 1

lines_from = lines["From_node"].to_numpy() - 1
lines_to = lines["To_node"].to_numpy() - 1
lines_B = lines["Susceptance_puw"].to_numpy()
lines_cap = lines["Capacity_MW"].to_numpy()
n_lines = len(lines)


# --- Step 3.2: the nodal market-clearing function ---
# Linearized ("DC") power flow (Lecture 3): the flow on line l is
#   f_l = B_l * (theta_from(l) - theta_to(l))                              -- flow definition
# and the single system-wide balance of Step 1 is replaced by ONE balance
# equation PER NODE n:
#   sum_{d at n} p_d  -  sum_{g at n} p_g  +  sum_{l: from=n} f_l  -  sum_{l: to=n} f_l  =  0   : lambda_n
# (demand and outgoing flow are "uses" of power at node n; generation and
# incoming flow are "sources" of power at node n). The dual variable of node
# n's balance equation, lambda_n, is that node's nodal price (LMP).
def solve_nodal_market(supplier_cost, supplier_capacity, supplier_node,
                        demand_bid, demand_maxload, demand_node,
                        lines_from, lines_to, lines_B, lines_cap,
                        n_nodes, ref_node=0):
    """Clear a single-hour market over the full nodal network (DC power flow).

    Extends Step 1 by replacing the single system-wide power balance with one
    balance equation per node, linked by line flows bounded by +-line capacity.
    """
    n_sup = len(supplier_cost)
    n_dem = len(demand_bid)
    n_lines = len(lines_from)

    # Variable layout: [ p_sup (n_sup) | p_dem (n_dem) | theta (n_nodes) | f (n_lines) ]
    i_sup, i_dem = 0, n_sup
    i_theta = i_dem + n_dem
    i_f = i_theta + n_nodes
    n_var = i_f + n_lines

    c = np.concatenate([supplier_cost, -demand_bid, np.zeros(n_nodes), np.zeros(n_lines)])

    bounds = (
        [(0, cap) for cap in supplier_capacity]
        + [(0, load) for load in demand_maxload]
        + [(0, 0) if n == ref_node else (None, None) for n in range(n_nodes)]
        + [(-cap, cap) for cap in lines_cap]
    )

    n_eq = n_nodes + n_lines  # n_nodes power-balance rows, then n_lines flow-definition rows
    A_eq = np.zeros((n_eq, n_var))
    b_eq = np.zeros(n_eq)

    for g in range(n_sup):
        A_eq[supplier_node[g], i_sup + g] = -1.0
    for d in range(n_dem):
        A_eq[demand_node[d], i_dem + d] = 1.0
    for l in range(n_lines):
        A_eq[lines_from[l], i_f + l] = 1.0   # flow leaving its "from" node
        A_eq[lines_to[l], i_f + l] = -1.0    # flow arriving at its "to" node

        row = n_nodes + l
        A_eq[row, i_f + l] = 1.0
        A_eq[row, i_theta + lines_from[l]] = -lines_B[l]
        A_eq[row, i_theta + lines_to[l]] = lines_B[l]

    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Nodal market-clearing LP did not solve: {result.message}")

    return {
        "p_sup": result.x[i_sup:i_dem],
        "p_dem": result.x[i_dem:i_theta],
        "theta": result.x[i_theta:i_f],
        "flow": result.x[i_f:i_f + n_lines],
        "nodal_price": -result.eqlin.marginals[:n_nodes],
        "social_welfare": -result.fun,
    }


# --- Step 3.3: solve the base case and check whether nodal prices are identical ---
nodal_base = solve_nodal_market(
    supplier_cost, supplier_capacity, supplier_node,
    demand_bid, demand_maxload, demand_node,
    lines_from, lines_to, lines_B, lines_cap, n_nodes, ref_node,
)

n_distinct_prices = len(np.unique(nodal_base["nodal_price"].round(2)))
print(f"\nNodal prices: {n_distinct_prices} distinct value(s) across {n_nodes} nodes "
      f"(range {nodal_base['nodal_price'].min():.2f} - {nodal_base['nodal_price'].max():.2f} $/MWh)")
print(f"Nodal social welfare: {nodal_base['social_welfare']:,.2f} $ "
      f"(Step 1 copper-plate social welfare was {social_welfare:,.2f} $)")

# %% STEP 3: SENSITIVITY ANALYSIS ON A TRANSMISSION LINE'S CAPACITY
#
# Identify the most heavily loaded line(s) in the base case, then artificially
# shrink the most-loaded one's capacity to force congestion, and see what
# happens to the nodal prices.

utilization = np.abs(nodal_base["flow"]) / lines_cap
lines_report = pd.DataFrame({
    "From": lines["From_node"], "To": lines["To_node"], "Capacity_MW": lines_cap,
    "Flow_MW": nodal_base["flow"].round(2), "Utilization": utilization.round(3),
}).sort_values("Utilization", ascending=False)
print("\nMost heavily loaded lines in the base case:")
print(lines_report.head(5).to_string(index=False))

most_loaded_line = int(np.argmax(utilization))
stressed_cap = lines_cap.copy()
stressed_cap[most_loaded_line] = 0.3 * lines_cap[most_loaded_line]  # shrink to 30% of its original capacity

nodal_stressed = solve_nodal_market(
    supplier_cost, supplier_capacity, supplier_node,
    demand_bid, demand_maxload, demand_node,
    lines_from, lines_to, lines_B, stressed_cap, n_nodes, ref_node,
)

print(f"\nAfter shrinking line {lines['From_node'].iloc[most_loaded_line]}->"
      f"{lines['To_node'].iloc[most_loaded_line]} from {lines_cap[most_loaded_line]:.0f} MW "
      f"to {stressed_cap[most_loaded_line]:.0f} MW:")
print(f"  Distinct nodal prices: {len(np.unique(nodal_stressed['nodal_price'].round(2)))} "
      f"(range {nodal_stressed['nodal_price'].min():.2f} - {nodal_stressed['nodal_price'].max():.2f} $/MWh)")
print(f"  Social welfare: {nodal_stressed['social_welfare']:,.2f} $ "
      f"(base case: {nodal_base['social_welfare']:,.2f} $)")

# %% STEP 3: ZONAL MARKET CLEARING
#
# Same 2-zone split used in Lecture 3's own example for this exact network:
# Zone 1 = nodes 1-13, Zone 2 = nodes 14-24. The ATC between the zones is the
# total capacity of every line that crosses the zone boundary.

supplier_zone = (supplier_node >= 13).astype(int)  # 0 = zone "1-13", 1 = zone "14-24"
demand_zone = (demand_node >= 13).astype(int)
crosses_zones = (lines_from < 13) != (lines_to < 13)
atc_base = lines_cap[crosses_zones].sum()
print(f"\nLines crossing the zone boundary: {crosses_zones.sum()}, "
      f"ATC (sum of their capacities) = {atc_base:.0f} MW")


def solve_zonal_market(supplier_cost, supplier_capacity, supplier_zone,
                        demand_bid, demand_maxload, demand_zone, atc):
    """Clear a single-hour, 2-zone market: zones 0 and 1, linked by a single
    interconnector with a symmetric Available Transfer Capacity (ATC).

    Extends Step 1 by replacing the single system-wide balance with one
    balance equation per zone, linked by one inter-zonal transfer variable
    bounded by +-atc (no per-line detail within each zone).
    """
    n_sup = len(supplier_cost)
    n_dem = len(demand_bid)
    i_sup, i_dem = 0, n_sup
    i_f = i_dem + n_dem
    n_var = i_f + 1

    c = np.concatenate([supplier_cost, -demand_bid, [0.0]])
    bounds = (
        [(0, cap) for cap in supplier_capacity]
        + [(0, load) for load in demand_maxload]
        + [(-atc, atc)]
    )

    A_eq = np.zeros((2, n_var))
    b_eq = np.zeros(2)
    for g in range(n_sup):
        A_eq[supplier_zone[g], i_sup + g] = -1.0
    for d in range(n_dem):
        A_eq[demand_zone[d], i_dem + d] = 1.0
    A_eq[0, i_f] = 1.0    # zone 0: ... + f01 = 0  (f01 = export from zone 0 to zone 1)
    A_eq[1, i_f] = -1.0   # zone 1: ... - f01 = 0

    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Zonal market-clearing LP did not solve: {result.message}")

    return {
        "p_sup": result.x[i_sup:i_dem],
        "p_dem": result.x[i_dem:i_f],
        "flow_01": result.x[i_f],
        "zonal_price": -result.eqlin.marginals[:2],
        "social_welfare": -result.fun,
    }


zonal_base = solve_zonal_market(
    supplier_cost, supplier_capacity, supplier_zone,
    demand_bid, demand_maxload, demand_zone, atc_base,
)
print(f"Zonal prices at ATC={atc_base:.0f} MW: Zone 1-13 = {zonal_base['zonal_price'][0]:.2f} $/MWh, "
      f"Zone 14-24 = {zonal_base['zonal_price'][1]:.2f} $/MWh")
print(f"Zonal social welfare: {zonal_base['social_welfare']:,.2f} $ "
      f"(nodal: {nodal_base['social_welfare']:,.2f} $, copper-plate: {social_welfare:,.2f} $)")

# --- ATC sensitivity: how do zonal prices react to a tighter/looser interconnector? ---
atc_scenarios = {"0.25x ATC": 0.25 * atc_base, "1x ATC (base)": atc_base, "4x ATC": 4 * atc_base}
atc_rows = []
for label, atc_value in atc_scenarios.items():
    z = solve_zonal_market(supplier_cost, supplier_capacity, supplier_zone,
                            demand_bid, demand_maxload, demand_zone, atc_value)
    atc_rows.append({"Scenario": label, "ATC_MW": atc_value,
                      "Price_zone1_13": z["zonal_price"][0], "Price_zone14_24": z["zonal_price"][1],
                      "Social_welfare": z["social_welfare"]})
print("\nZonal prices for different ATC values:")
print(pd.DataFrame(atc_rows).round(2).to_string(index=False))

# %% STEP 3: NODAL vs. ZONAL -- PROFITS, WELFARE AND EX-POST FEASIBILITY
#
# How to read this section: at the BASE ATC (1650 MW) the network is not
# congested, so nodal and zonal give identical results -- not a very
# informative comparison. To actually see how nodal and zonal frameworks can
# differ, we redo this comparison at the STRESSED ATC (0.25x, from the
# sensitivity table above), where the zone interconnector genuinely binds.
#
# We compare producer profits (conventional vs. renewable) and social welfare
# between the nodal and zonal outcomes, then check whether the zonal
# dispatch -- which ignores individual line limits inside each zone -- is
# actually feasible on the real network, by solving the (non-optimization)
# DC power-flow equations for the flows that dispatch would actually cause.

zonal_stressed = solve_zonal_market(
    supplier_cost, supplier_capacity, supplier_zone,
    demand_bid, demand_maxload, demand_zone, 0.25 * atc_base,
)

is_wind = np.arange(n_sup) >= len(gens)
nodal_producer_profit = (nodal_base["nodal_price"][supplier_node] - supplier_cost) * nodal_base["p_sup"]
zonal_producer_profit = (zonal_stressed["zonal_price"][supplier_zone] - supplier_cost) * zonal_stressed["p_sup"]

profit_comparison = pd.DataFrame({
    "Nodal_profit_USD": [nodal_producer_profit[~is_wind].sum(), nodal_producer_profit[is_wind].sum()],
    "Zonal_profit_USD": [zonal_producer_profit[~is_wind].sum(), zonal_producer_profit[is_wind].sum()],
}, index=["Conventional generators", "Wind farms"]).round(2)
print("\nProducer profit, nodal (uncongested) vs. zonal at a stressed 0.25x ATC:")
print(profit_comparison)
print(f"\nSocial welfare -- nodal: {nodal_base['social_welfare']:,.2f} $, "
      f"zonal (stressed ATC): {zonal_stressed['social_welfare']:,.2f} $, "
      f"difference: {nodal_base['social_welfare'] - zonal_stressed['social_welfare']:,.2f} $")


def compute_dc_power_flow(injections, lines_from, lines_to, lines_B, n_nodes, ref_node=0):
    """Given a FIXED net injection at every node (generation - demand), solve
    the linear DC power-flow equations (Kirchhoff's laws, not an optimization)
    for the voltage angles and the resulting line flows.
    """
    B_mat = np.zeros((n_nodes, n_nodes))
    for l in range(len(lines_from)):
        n, m, b = lines_from[l], lines_to[l], lines_B[l]
        B_mat[n, n] += b
        B_mat[m, m] += b
        B_mat[n, m] -= b
        B_mat[m, n] -= b

    keep = [n for n in range(n_nodes) if n != ref_node]
    theta = np.zeros(n_nodes)
    theta[keep] = np.linalg.solve(B_mat[np.ix_(keep, keep)], injections[keep])

    flow = lines_B * (theta[lines_from] - theta[lines_to])
    return theta, flow


# Net injection at every node implied by the STRESSED ZONAL dispatch (generation - demand)
injection = np.zeros(n_nodes)
for g in range(n_sup):
    injection[supplier_node[g]] += zonal_stressed["p_sup"][g]
for d in range(n_dem):
    injection[demand_node[d]] -= zonal_stressed["p_dem"][d]

_, implied_flow = compute_dc_power_flow(injection, lines_from, lines_to, lines_B, n_nodes, ref_node)
overload = np.abs(implied_flow) - lines_cap
overloaded = overload > 1e-6

print(f"\nEx-post feasibility check of the stressed-ATC zonal dispatch on the real (nodal) network:")
print(f"Lines that would be overloaded: {int(overloaded.sum())} / {n_lines}")
if overloaded.any():
    redispatch_table = pd.DataFrame({
        "From": lines["From_node"][overloaded], "To": lines["To_node"][overloaded],
        "Capacity_MW": lines_cap[overloaded], "Implied_flow_MW": implied_flow[overloaded].round(2),
        "Overload_MW": overload[overloaded].round(2),
    })
    print(redispatch_table.to_string(index=False))
    print(f"Total overload across all lines: {overload[overloaded].sum():,.2f} MW "
          f"-- an approximate lower bound on the re-dispatch this zonal outcome would require.")
