# %% IMPORT PACKAGES

import pandas as pd
import numpy as np
import os
from scipy.optimize import linprog

# %% IMPORT DATA
#
# How to read this section: this file is self-contained. Steps 4-6 build on
# top of Step 1 (the day-ahead copper-plate market) and, for Step 4, on top of
# Step 2 (the 24-hour market with storage) -- so before we can do anything new,
# we re-read the Excel data and re-solve those two markets exactly as in the
# "Electricity_Markets_Step_1_to_3_A1.py" file, to have their results available
# here too.

script_dir = os.path.dirname(os.path.abspath(__file__))
path_A1_data = os.path.join(script_dir, "data", "Assignment1_IEEE24bus_input_data.xlsx")
A1_data_file = pd.read_excel(path_A1_data, sheet_name=None)
gens = A1_data_file["Conventional_generators"]
wind = A1_data_file["Wind_farms"]
demands = A1_data_file["Demands"]
lines = A1_data_file["Transmission_lines"]

# --- Re-derive Step 1: the day-ahead copper-plate market ---
supplier_names = list(gens["Generator"]) + list(wind["Wind_farm"])
supplier_cost = np.concatenate([gens["Production_cost_USD_per_MWh"].to_numpy(), np.zeros(len(wind))])
supplier_capacity = np.concatenate([gens["Capacity_MW"].to_numpy(), wind["Day_ahead_forecast_MW"].to_numpy()])
n_sup = len(supplier_names)
n_gens = len(gens)

demand_names = list(demands["Demand"])
demand_bid = demands["Curtailment_cost_USD_per_MWh"].to_numpy()
demand_maxload = demands["Consumption_MW"].to_numpy()
n_dem = len(demand_names)

c = np.concatenate([supplier_cost, -demand_bid])
bounds = [(0, cap) for cap in supplier_capacity] + [(0, load) for load in demand_maxload]
A_eq = np.concatenate([-np.ones(n_sup), np.ones(n_dem)]).reshape(1, -1)
b_eq = np.array([0.0])

result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
if not result.success:
    raise RuntimeError(f"Day-ahead market-clearing LP did not solve: {result.message}")

p_sup_DA = result.x[:n_sup]
p_dem_DA = result.x[n_sup:]
lambda_DA = -result.eqlin.marginals[0]
social_welfare_DA = -result.fun

print(f"[Step 1 recap] Day-ahead price: {lambda_DA:.2f} $/MWh, social welfare: {social_welfare_DA:,.2f} $")

# --- Re-derive Step 2: the 24-hour market with storage ---
# (needed for Step 4, which analyzes the storage's own equilibrium problem)


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
bid_price_shape = 1.0 + 0.15 * (load_shape - 1.0)

supplier_capacity_th = np.zeros((n_sup, T))
supplier_capacity_th[:n_gens, :] = gens["Capacity_MW"].to_numpy()[:, None]
supplier_capacity_th[n_gens:, :] = wind["Day_ahead_forecast_MW"].to_numpy()[:, None] * wind_shape[None, :]
demand_bid_th = demand_bid[:, None] * bid_price_shape[None, :]
demand_maxload_th = demand_maxload[:, None] * load_shape[None, :]

P_ch_max, P_dis_max, E_max = 150.0, 150.0, 600.0
eta_ch, eta_dis = 0.90, 0.95
e0 = 0.5 * E_max


def solve_multi_hour_market(supplier_cost, supplier_capacity_th, demand_bid_th, demand_maxload_th,
                             P_ch_max, P_dis_max, E_max, eta_ch, eta_dis, e0):
    """Clear a T-hour copper-plate market with one storage unit (see Step 2 for
    the full derivation). Also returns psi_t, the dual variable of the storage's
    OWN energy-balance constraint (its hour-by-hour "water value"), which Step 4
    needs and Step 2 did not.
    """
    n_sup, T = supplier_capacity_th.shape
    n_dem = demand_bid_th.shape[0]

    i_sup, i_dem = 0, n_sup * T
    i_ch, i_dis, i_e = i_dem + n_dem * T, i_dem + n_dem * T + T, i_dem + n_dem * T + 2 * T
    n_var = i_e + T

    c = np.concatenate([np.repeat(supplier_cost, T), -demand_bid_th.flatten(), np.zeros(3 * T)])
    bnds = (
        [(0, cap) for cap in supplier_capacity_th.flatten()]
        + [(0, load) for load in demand_maxload_th.flatten()]
        + [(0, P_ch_max)] * T + [(0, P_dis_max)] * T + [(0, E_max)] * T
    )

    A_eq = np.zeros((2 * T, n_var))
    b_eq = np.zeros(2 * T)
    for t in range(T):
        A_eq[t, i_sup + t: i_sup + n_sup * T: T] = -1.0
        A_eq[t, i_dem + t: i_dem + n_dem * T: T] = 1.0
        A_eq[t, i_ch + t] = 1.0
        A_eq[t, i_dis + t] = -1.0

        row = T + t
        A_eq[row, i_e + t] = 1.0
        A_eq[row, i_ch + t] = -eta_ch
        A_eq[row, i_dis + t] = 1.0 / eta_dis
        if t == 0:
            b_eq[row] = e0
        else:
            A_eq[row, i_e + t - 1] = -1.0

    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if not result.success:
        raise RuntimeError(f"Multi-hour market-clearing LP did not solve: {result.message}")

    return {
        "p_sup": result.x[i_sup:i_dem].reshape(n_sup, T),
        "p_dem": result.x[i_dem:i_ch].reshape(n_dem, T),
        "p_ch": result.x[i_ch:i_dis],
        "p_dis": result.x[i_dis:i_e],
        "e": result.x[i_e:i_e + T],
        "price_t": -result.eqlin.marginals[:T],
        "psi_t": -result.eqlin.marginals[T:2 * T],  # dual of storage's OWN energy-balance constraint
        "social_welfare": -result.fun,
    }


with_storage = solve_multi_hour_market(
    supplier_cost, supplier_capacity_th, demand_bid_th, demand_maxload_th,
    P_ch_max, P_dis_max, E_max, eta_ch, eta_dis, e0,
)
print(f"[Step 2 recap] 24h social welfare with storage: {with_storage['social_welfare']:,.2f} $")

# %% STEP 4: OPTIMIZATION VS. EQUILIBRIUM (STORAGE'S OWN PROFIT-MAXIMIZATION PROBLEM)
#
# How to read this section: the assignment explicitly says "there is no need
# for simulations or coding for this step" -- Step 4 is a derivation, not a
# model to build. We reproduce that derivation here as comments/printouts, and
# ADD (as a bonus, not strictly required) a numerical cross-check against the
# Step 2 solution we already have, to confirm the derivation is actually
# consistent with the data.
#
# The storage owner (taking the 24 hourly prices lambda_t as given -- a
# "price-taker") solves its OWN profit-maximization problem:
#
#   Maximize   sum_t  lambda_t * (p_t^dis - p_t^ch)
#   p_ch,p_dis,e
#
#   subject to (same physical constraints as in the market-clearing LP):
#       0 <= p_t^ch  <= P^ch                                    : mu_ch_low,t, mu_ch_up,t
#       0 <= p_t^dis <= P^dis                                   : mu_dis_low,t, mu_dis_up,t
#       0 <= e_t     <= E                                       : mu_e_low,t, mu_e_up,t
#       e_t = e_{t-1} + eta_ch*p_t^ch - p_t^dis/eta_dis          : psi_t
#
# Its Lagrangian (writing it as "minimize -profit", to match our usual
# convention) gives, by stationarity, for every hour t:
#
#   d/dp_ch,t : lambda_t = mu_ch_low,t - mu_ch_up,t + psi_t * eta_ch
#   d/dp_dis,t: lambda_t = -mu_dis_low,t + mu_dis_up,t + psi_t / eta_dis
#   d/de_t    : psi_t = psi_(t+1) + mu_e_low,t - mu_e_up,t     (psi_(T+1) := 0)
#
# Economic reading:
#   - psi_t is the storage's own "water value": the shadow price (in $/MWh) of
#     having one more MWh of energy stored at the end of hour t.
#   - The third equation says the water value today equals tomorrow's water
#     value, UNLESS the storage is at its energy bounds (0 or E) today, in
#     which case it can jump.
#   - When storage charges with 0 < p_ch,t < P^ch (an interior, non-cornered
#     decision), both mu_ch's are zero, so lambda_t = psi_t * eta_ch.
#   - When it discharges with 0 < p_dis,t < P^dis, lambda_t = psi_t / eta_dis.
#   - It CANNOT do both in the same hour at an interior point: that would
#     require psi_t*eta_ch = psi_t/eta_dis, i.e. eta_ch*eta_dis = 1, which
#     contradicts eta_ch < eta_dis < 1 (assignment's Note 3) unless psi_t = 0.
#     This is exactly WHY the assignment insists on eta_ch != eta_dis: it is
#     what rules out simultaneous charging and discharging without needing
#     binary variables.

print("\n--- Step 4: numerical cross-check of the storage equilibrium KKT conditions ---")
tol = 1e-6
psi_t = with_storage["psi_t"]
price_t = with_storage["price_t"]
p_ch, p_dis, e = with_storage["p_ch"], with_storage["p_dis"], with_storage["e"]

# Note: the simple equalities lambda_t = psi_t*eta_ch (charging) and
# lambda_t = psi_t/eta_dis (discharging) only hold where that decision is
# STRICTLY interior (0 < p < P_max). If charging/discharging is pinned at its
# OWN power bound (P_ch_max or P_dis_max), a nonzero mu_up multiplier is
# expected, and the "residual" below is really that multiplier, not an error.
check_rows = []
for t in range(T):
    if p_ch[t] > tol:
        implied = psi_t[t] * eta_ch
        at_bound = p_ch[t] > P_ch_max - tol
        label = "charging (at P_ch_max)" if at_bound else "charging (interior)"
        check_rows.append((t + 1, label, price_t[t], implied, at_bound))
    elif p_dis[t] > tol:
        implied = psi_t[t] / eta_dis
        at_bound = p_dis[t] > P_dis_max - tol
        label = "discharging (at P_dis_max)" if at_bound else "discharging (interior)"
        check_rows.append((t + 1, label, price_t[t], implied, at_bound))
    else:
        check_rows.append((t + 1, "idle", price_t[t], np.nan, False))

kkt_storage_table = pd.DataFrame(
    check_rows, columns=["Hour", "Storage_action", "Price_lambda_t", "Implied_by_psi_t", "At_power_bound"]
)
kkt_storage_table["Residual"] = (kkt_storage_table["Price_lambda_t"] - kkt_storage_table["Implied_by_psi_t"]).abs()
print(kkt_storage_table.round(4).to_string(index=False))

interior_residual = kkt_storage_table.loc[~kkt_storage_table["At_power_bound"], "Residual"]
print(f"\nMax residual where storage is charging/discharging strictly INTERIOR to its power bound: "
      f"{interior_residual.max(skipna=True):.6f} (should be ~0)")
at_bound_rows = kkt_storage_table[kkt_storage_table["At_power_bound"]]
if len(at_bound_rows):
    print(f"Hour(s) where storage is pinned AT its power bound (residual = that bound's shadow price mu_up, "
          f"not an error): {list(at_bound_rows['Hour'])}, residual = {list(at_bound_rows['Residual'].round(3))}")

interior_e = (e > tol) & (e < E_max - tol)
psi_next = np.roll(psi_t, -1)
psi_next[-1] = 0.0  # psi_(T+1) := 0 (no value to energy left over after the last hour)
psi_residual = np.abs(psi_t - psi_next)[interior_e] if interior_e.any() else np.array([0.0])
print(f"Max residual of psi_t = psi_(t+1) where e_t is strictly interior: "
      f"{psi_residual.max() if len(psi_residual) else float('nan'):.6f} (should be ~0)")
print("Confirms: no hour has BOTH p_ch>0 and p_dis>0 simultaneously: "
      f"{bool(np.all((p_ch > tol) & (p_dis > tol) == False))}")

# %% STEP 5: BALANCING MARKET
#
# How to read this section: builds on STEP 1 (single hour, no storage, no
# network -- not Steps 2/3). We simulate one specific realization of real-time
# events and clear the resulting balancing market.
#
# Assumptions (the assignment leaves the specifics to us):
#   - The outaged unit is G9 (280 MW day-ahead dispatch, node 21) -- a fully
#     dispatched, low-cost unit, so its loss creates a large, meaningful deficit.
#   - Wind farms W1, W2 under-produce by 15%; W3, W4 over-produce by 10%,
#     relative to their OWN day-ahead schedule (both magnitudes as suggested
#     by the assignment text).
#   - The "subset of conventional generators" eligible to provide balancing is
#     every conventional generator EXCEPT the outaged one (11 units).
#   - Demands are inflexible (assignment's own statement): they stay at their
#     day-ahead consumption level.

failed_unit = "G9"
wind_lower = ["W1", "W2"]   # -15% vs. day-ahead forecast
wind_higher = ["W3", "W4"]  # +10% vs. day-ahead forecast
wind_lower_pct, wind_higher_pct = 0.15, 0.10
load_curtailment_cost = 500.0  # $/MWh (same value as the demands' curtailment cost in Step 1)

p_actual = p_sup_DA.copy()
failed_idx = supplier_names.index(failed_unit)
p_actual[failed_idx] = 0.0
for name in wind_lower:
    idx = supplier_names.index(name)
    p_actual[idx] = p_sup_DA[idx] * (1 - wind_lower_pct)
for name in wind_higher:
    idx = supplier_names.index(name)
    p_actual[idx] = p_sup_DA[idx] * (1 + wind_higher_pct)

deviation = p_actual - p_sup_DA  # negative = short vs. schedule, positive = long
system_deviation = deviation.sum()  # negative => system-wide deficit (needs upward regulation)
print(f"\n[Step 5] System-wide deviation from day-ahead schedule: {system_deviation:.2f} MW "
      f"({'deficit' if system_deviation < 0 else 'surplus'} -- needs "
      f"{'upward' if system_deviation < 0 else 'downward'} regulation)")

# --- Balancing offers of the eligible conventional generators ---
eligible = np.array([name in list(gens['Generator']) and name != failed_unit for name in supplier_names])
headroom = np.where(eligible, supplier_capacity - p_sup_DA, 0.0)  # MW they could still increase by
footroom = np.where(eligible, p_sup_DA, 0.0)                      # MW they could still decrease by
up_price = lambda_DA + 0.10 * supplier_cost   # only meaningful where eligible & headroom > 0
down_price = lambda_DA - 0.15 * supplier_cost  # only meaningful where eligible & footroom > 0

n_eligible = int(eligible.sum())
i_rup, i_rdown, i_ls = 0, n_sup, 2 * n_sup
n_var_bal = 2 * n_sup + 1

c_bal = np.concatenate([up_price, -down_price, [load_curtailment_cost]])
bounds_bal = (
    [(0, headroom[g]) if eligible[g] else (0, 0) for g in range(n_sup)]
    + [(0, footroom[g]) if eligible[g] else (0, 0) for g in range(n_sup)]
    + [(0, demand_maxload.sum())]
)
# Balancing power balance: sum(r_up) - sum(r_down) + load_curtailment = -system_deviation : beta
A_eq_bal = np.zeros((1, n_var_bal))
A_eq_bal[0, i_rup:i_rup + n_sup] = 1.0
A_eq_bal[0, i_rdown:i_rdown + n_sup] = -1.0
A_eq_bal[0, i_ls] = 1.0
b_eq_bal = np.array([-system_deviation])

result_bal = linprog(c=c_bal, A_eq=A_eq_bal, b_eq=b_eq_bal, bounds=bounds_bal, method="highs")
if not result_bal.success:
    raise RuntimeError(f"Balancing market LP did not solve: {result_bal.message}")

r_up = result_bal.x[i_rup:i_rup + n_sup]
r_down = result_bal.x[i_rdown:i_rdown + n_sup]
load_curtailed = result_bal.x[i_ls]
# Note the sign convention here is DIFFERENT from the market-clearing LPs above:
# this LP directly minimizes real cost (not "-social welfare"), so the dual is
# already the correctly-signed marginal cost -- no extra sign flip needed.
beta = result_bal.eqlin.marginals[0]

print(f"Balancing price (beta): {beta:.3f} $/MWh (day-ahead price was {lambda_DA:.2f} $/MWh)")
print(f"Load curtailed: {load_curtailed:.2f} MW")
balancing_dispatch = pd.DataFrame({
    "Headroom_MW": headroom, "Footroom_MW": footroom,
    "Up_offer": np.where(eligible, up_price, np.nan), "Down_offer": np.where(eligible, down_price, np.nan),
    "r_up_MW": r_up.round(2), "r_down_MW": r_down.round(2),
}, index=supplier_names)
print(balancing_dispatch[(balancing_dispatch["r_up_MW"] > 1e-6) | (balancing_dispatch["r_down_MW"] > 1e-6)])

# --- Total profit (day-ahead + balancing), one-price vs. two-price schemes ---
# One-price:  every deviation (helping or hurting) is settled at beta.
#   profit_one = (lambda_DA - C_g)*p_DA + (beta - C_g)*(p_actual - p_DA)
# Two-price:  only deviations that WORSEN the system imbalance are settled at
#   beta (a penalty); deviations that HELP are settled at lambda_DA instead
#   (no reward for accidentally helping). Deliberate balancing PROVIDERS
#   (r_up/r_down > 0, i.e. the TSO explicitly activated them) always settle at
#   beta in both schemes -- the distinction only matters for passive,
#   unintentional deviations (the outage and the wind forecast errors).
system_needs_up = system_deviation < 0
is_provider = (r_up > 1e-6) | (r_down > 1e-6)
helps_system = (deviation > 0) if system_needs_up else (deviation < 0)

settle_price_one = np.full(n_sup, beta)  # everyone settled at beta under one-price
settle_price_two = np.where(is_provider | ~helps_system, beta, lambda_DA)  # two-price: helpful passive deviators get lambda_DA

profit_DA = (lambda_DA - supplier_cost) * p_sup_DA
profit_one_price = profit_DA + (settle_price_one - supplier_cost) * deviation
profit_two_price = profit_DA + (settle_price_two - supplier_cost) * deviation

profit_table = pd.DataFrame({
    "DA_dispatch_MW": p_sup_DA.round(2), "Actual_MW": p_actual.round(2), "Deviation_MW": deviation.round(2),
    "Profit_DA_only": profit_DA.round(2),
    "Profit_one_price": profit_one_price.round(2),
    "Profit_two_price": profit_two_price.round(2),
}, index=supplier_names)
print("\nProfit comparison -- day-ahead only vs. one-price vs. two-price settlement:")
print(profit_table[profit_table["Deviation_MW"].abs() > 1e-6].to_string())

# %% STEP 6: RESERVE MARKET
#
# How to read this section: back to STEP 1's plain setup (no outage, no
# storage, no network) -- Step 6 is independent of Step 5's imbalance scenario.
# The TSO procures upward/downward RESERVE first; only afterwards is the
# day-ahead ENERGY market cleared, with each reserve-providing generator's
# available energy capacity reduced by whatever reserve it was awarded
# ("current practice in European electricity markets": sequential clearing).

up_reserve_requirement = 0.15 * demand_maxload.sum()
down_reserve_requirement = 0.10 * demand_maxload.sum()
print(f"\n[Step 6] Upward reserve requirement: {up_reserve_requirement:.2f} MW "
      f"(15% of {demand_maxload.sum():.0f} MW total demand)")
print(f"Downward reserve requirement: {down_reserve_requirement:.2f} MW (10%)")

up_res_cost = gens["Upward_reserve_cost_USD_per_MW"].to_numpy()
down_res_cost = gens["Downward_reserve_cost_USD_per_MW"].to_numpy()
max_up_res = gens["Max_upward_reserve_MW"].to_numpy()
max_down_res = gens["Max_downward_reserve_MW"].to_numpy()

# --- Step 6.1: clear the RESERVE market (all 12 conventional generators eligible) ---
# In addition to each generator's own Max_upward/Max_downward reserve limits,
# we need a JOINT constraint per generator: reserve_up_g + reserve_down_g <=
# Capacity_g. Without it, the reserve market could -- entirely rationally, from
# its own narrow cost-minimization perspective -- award a cheap generator so
# much upward reserve that zero energy capacity is left for it to also provide
# its assigned downward reserve, which is a physical impossibility (this is
# exactly what happens here with G5 if this constraint is left out).
c_res = np.concatenate([up_res_cost, down_res_cost])
bounds_res = [(0, m) for m in max_up_res] + [(0, m) for m in max_down_res]

A_eq_res = np.zeros((2, 2 * n_gens))
A_eq_res[0, :n_gens] = 1.0
A_eq_res[1, n_gens:] = 1.0
b_eq_res = np.array([up_reserve_requirement, down_reserve_requirement])

A_ub_res = np.zeros((n_gens, 2 * n_gens))
b_ub_res = gens["Capacity_MW"].to_numpy()
for g in range(n_gens):
    A_ub_res[g, g] = 1.0
    A_ub_res[g, n_gens + g] = 1.0  # reserve_up_g + reserve_down_g <= Capacity_g

result_res = linprog(c=c_res, A_eq=A_eq_res, b_eq=b_eq_res,
                      A_ub=A_ub_res, b_ub=b_ub_res, bounds=bounds_res, method="highs")
if not result_res.success:
    raise RuntimeError(f"Reserve market LP did not solve: {result_res.message}")

reserve_up = result_res.x[:n_gens]
reserve_down = result_res.x[n_gens:]
rho_up = result_res.eqlin.marginals[0]
rho_down = result_res.eqlin.marginals[1]
print(f"Reserve prices: upward = {rho_up:.3f} $/MW, downward = {rho_down:.3f} $/MW")

# --- Step 6.2: clear the DAY-AHEAD ENERGY market, capacity reduced by awarded reserve ---
supplier_capacity_post_reserve = supplier_capacity.copy()
supplier_capacity_post_reserve[:n_gens] = supplier_capacity[:n_gens] - reserve_up  # headroom held back for up-reserve
supplier_floor_post_reserve = np.zeros(n_sup)
supplier_floor_post_reserve[:n_gens] = reserve_down  # must run at least this much to be able to reduce by reserve_down

bounds_energy_seq = (
    [(supplier_floor_post_reserve[g], supplier_capacity_post_reserve[g]) for g in range(n_sup)]
    + [(0, load) for load in demand_maxload]
)
result_seq = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds_energy_seq, method="highs")
if not result_seq.success:
    raise RuntimeError(f"Post-reserve day-ahead LP did not solve: {result_seq.message}")

lambda_DA_with_reserve = -result_seq.eqlin.marginals[0]
print(f"\nDay-ahead energy price WITHOUT reserve (Step 1): {lambda_DA:.2f} $/MWh")
print(f"Day-ahead energy price WITH reserve procured first (Step 6, sequential): {lambda_DA_with_reserve:.2f} $/MWh")
print(f"Change: {lambda_DA_with_reserve - lambda_DA:+.2f} $/MWh")

# %% STEP 6 (OPTIONAL): U.S.-STYLE JOINT ENERGY-AND-RESERVE CO-OPTIMIZATION
#
# Instead of clearing reserve first and energy second, the U.S. style clears
# BOTH simultaneously in one optimization, linking a generator's energy
# dispatch and its reserve awards through joint capacity constraints:
#   p_g + r_up,g   <= Capacity_g   (must still have headroom left for up-reserve)
#   p_g - r_down,g >= 0            (must still have footroom left for down-reserve)

i_psup, i_pdem = 0, n_sup
i_rup_j, i_rdown_j = i_pdem + n_dem, i_pdem + n_dem + n_gens
n_var_joint = i_rdown_j + n_gens

c_joint = np.concatenate([supplier_cost, -demand_bid, up_res_cost, down_res_cost])
bounds_joint = (
    [(0, cap) for cap in supplier_capacity]
    + [(0, load) for load in demand_maxload]
    + [(0, m) for m in max_up_res]
    + [(0, m) for m in max_down_res]
)

A_eq_joint = np.zeros((3, n_var_joint))
b_eq_joint = np.array([0.0, up_reserve_requirement, down_reserve_requirement])
A_eq_joint[0, i_psup:i_psup + n_sup] = -1.0
A_eq_joint[0, i_pdem:i_pdem + n_dem] = 1.0
A_eq_joint[1, i_rup_j:i_rup_j + n_gens] = 1.0
A_eq_joint[2, i_rdown_j:i_rdown_j + n_gens] = 1.0

A_ub_joint = np.zeros((2 * n_gens, n_var_joint))
b_ub_joint = np.zeros(2 * n_gens)
for g in range(n_gens):
    A_ub_joint[g, i_psup + g] = 1.0
    A_ub_joint[g, i_rup_j + g] = 1.0
    b_ub_joint[g] = supplier_capacity[g]           # p_g + r_up,g <= Capacity_g

    A_ub_joint[n_gens + g, i_psup + g] = -1.0
    A_ub_joint[n_gens + g, i_rdown_j + g] = 1.0
    b_ub_joint[n_gens + g] = 0.0                    # r_down,g - p_g <= 0

result_joint = linprog(c=c_joint, A_eq=A_eq_joint, b_eq=b_eq_joint,
                        A_ub=A_ub_joint, b_ub=b_ub_joint, bounds=bounds_joint, method="highs")
if not result_joint.success:
    raise RuntimeError(f"Joint US-style LP did not solve: {result_joint.message}")

# Sign convention: row 0 (energy balance) is tied to the "-SW" part of the
# objective, so it needs the same sign flip as every other market-clearing LP
# in this project (lambda = -marginal). Rows 1-2 (reserve requirements) are
# tied to the DIRECT (not sign-flipped) reserve-cost part of the objective, so
# -- exactly like the standalone reserve LP in Step 6.1 -- their duals are
# already correctly signed and must NOT be negated again.
lambda_joint = -result_joint.eqlin.marginals[0]
rho_up_joint = result_joint.eqlin.marginals[1]
rho_down_joint = result_joint.eqlin.marginals[2]
p_sup_joint = result_joint.x[i_psup:i_pdem]

print(f"\n[Optional] U.S.-style joint clearing:")
print(f"  Energy price: {lambda_joint:.2f} $/MWh (European sequential: {lambda_DA_with_reserve:.2f} $/MWh)")
print(f"  Reserve prices: up = {rho_up_joint:.3f}, down = {rho_down_joint:.3f} $/MW "
      f"(European sequential: up = {rho_up:.3f}, down = {rho_down:.3f} $/MW)")
print(f"  Social welfare (energy + reserve): {-result_joint.fun:,.2f} $")
max_energy_change = np.abs(p_sup_joint[:n_gens] - result_seq.x[:n_gens]).max()
print(f"  Largest change in any generator's energy dispatch vs. the European sequential result: "
      f"{max_energy_change:.2f} MW")
