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
