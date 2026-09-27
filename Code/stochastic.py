# =============================================================================
# 1. IMPORTS E INIZIALIZZAZIONE
# =============================================================================
import numpy as np
import pandas as pd
from pathlib import Path

from configurations import *
from data_loader import *
from functions import *

# Fissiamo il seed per la riproducibilità dei numeri casuali
np.random.seed(42)

# --- Generazione Matrici Z (Numeri Casuali) per il calcolo Stocastico ---
# Utilizziamo la tecnica delle variabili antitetiche per ridurre la varianza
n_sim_half = CONFIG_STO["NUMBER_SIMULATIONS"] // 2
Z_EQ_HALF = np.random.standard_normal((n_sim_half, CONFIG_STO["HORIZON"]))
Z_EQUITY = np.concatenate((Z_EQ_HALF, -Z_EQ_HALF), axis=0)

Z_PR_HALF = np.random.standard_normal((n_sim_half, CONFIG_STO["HORIZON"]))
Z_PROPERTY = np.concatenate((Z_PR_HALF, -Z_PR_HALF), axis=0)
Z_list = [Z_EQUITY, Z_PROPERTY]

# Inizializziamo il dizionario che conterrà tutti i risultati finali
results = {}

# --- Generazione Matrici Z per il calcolo Deterministico ---
n_sim_half_det = CONFIG_DET["NUMBER_SIMULATIONS"] // 2
Z_EQ_HALF_DET = np.random.standard_normal((n_sim_half_det, CONFIG_DET["HORIZON"]))
Z_EQUITY_DET = np.concatenate((Z_EQ_HALF_DET, -Z_EQ_HALF_DET), axis=0)

Z_PR_HALF_DET = np.random.standard_normal((n_sim_half_det, CONFIG_DET["HORIZON"]))
Z_PROPERTY_DET = np.concatenate((Z_PR_HALF_DET, -Z_PR_HALF_DET), axis=0)
Z_list_det = [Z_EQUITY_DET, Z_PROPERTY_DET]

# =============================================================================
# 2. RUN BASE: STOCASTICO
# =============================================================================
spot_rates = load_eiopa_rates(CONFIG_STO)
q_x = load_mortality_rates(CONFIG_STO)
v_base, fwd_base = rates_calculation(spot_rates)
pop = population_projection(q_x, CONFIG_STO)

fund, asset, comm_m, rd_m, growth_f = fund_simulation(
    fwd_base, Z_list, CONFIG_STO, run_name="Stochastic_Base_Case"
)

bel, duration, pvfp, proxy, mva, bof, leakage, tvog_final, comp = bel_calculation(
    fund, comm_m, rd_m, v_base, pop, CONFIG_STO
)

results["Base_Stochastic"] = {
    "MVA": mva,
    "BEL": bel,
    "PV_Death": comp["PV_Death"],
    "PV_Lapse": comp["PV_Lapse"],
    "PV_Exp": comp["PV_Exp"],
    "PV_Comm": comp["PV_Comm"],
    "PV_Premiums": comp["PV_Premiums"],
    "BOF": bof,
    "PVFP": pvfp,
    "Proxy": proxy,
    "TVOG": tvog_final,
    "Duration": duration
}


# =============================================================================
# 3. MODULO LIFE: SHOCK BIOMETRICI E SPESE
# =============================================================================

# --- Shock di Mortalità (+15% sulle probabilità di decesso) ---
q_x_shock = q_x * 1.15
pop_mort = population_projection(q_x_shock, CONFIG_STO)
bel_mort, dur_mort, pvfp_mort, prox_mort, mva_mort, bof_mort, leak_mort, tvog_mort, comp_mort = bel_calculation(
    fund, comm_m, rd_m, v_base, pop_mort, CONFIG_STO
)
scr_mort = max(0, bof - bof_mort)
results["Mortality_Shock"] = {
    "MVA": mva_mort, "BEL": bel_mort,
    "PV_Death": comp_mort["PV_Death"], "PV_Lapse": comp_mort["PV_Lapse"], "PV_Exp": comp_mort["PV_Exp"], "PV_Comm": comp_mort["PV_Comm"], "PV_Premiums": comp_mort["PV_Premiums"],
    "BOF": bof_mort, "PVFP": pvfp_mort, "Proxy": prox_mort, "TVOG": tvog_mort, "Duration": dur_mort,
    "SCR": scr_mort
}

# --- Shock Catastrofale Life (+0.15% probabilità di decesso nel primo anno) ---
q_x_cat = q_x.copy()
q_x_cat[1] += 0.0015  
pop_cat = population_projection(q_x_cat, CONFIG_STO)
bel_cat, dur_cat, pvfp_cat, prox_cat, mva_cat, bof_cat, leak_cat, tvog_cat, comp_cat = bel_calculation(
    fund, comm_m, rd_m, v_base, pop_cat, CONFIG_STO
)
scr_cat = max(0, bof - bof_cat)
results["Catastrophe_Shock"] = {
    "MVA": mva_cat, "BEL": bel_cat,
    "PV_Death": comp_cat["PV_Death"], "PV_Lapse": comp_cat["PV_Lapse"], "PV_Exp": comp_cat["PV_Exp"], "PV_Comm": comp_cat["PV_Comm"], "PV_Premiums": comp_cat["PV_Premiums"],
    "BOF": bof_cat, "PVFP": pvfp_cat, "Proxy": prox_cat, "TVOG": tvog_cat, "Duration": dur_cat,
    "SCR": scr_cat
}

# --- Shock Spese (+10% spese correnti, +1% inflazione) ---
CONFIG_EXP = CONFIG_STO.copy()
CONFIG_EXP["EXPENSE"] = CONFIG_STO["EXPENSE"] * 1.10
CONFIG_EXP["INFLATION"] = CONFIG_STO["INFLATION"] + 0.01
bel_exp, dur_exp, pvfp_exp, prox_exp, mva_exp, bof_exp, leak_exp, tvog_exp, comp_exp = bel_calculation(
    fund, comm_m, rd_m, v_base, pop, CONFIG_EXP
)
scr_expenses = max(0, bof - bof_exp)
results["Shock_Expenses"] = {
    "MVA": mva_exp, "BEL": bel_exp,
    "PV_Death": comp_exp["PV_Death"], "PV_Lapse": comp_exp["PV_Lapse"], "PV_Exp": comp_exp["PV_Exp"], "PV_Comm": comp_exp["PV_Comm"], "PV_Premiums": comp_exp["PV_Premiums"],
    "BOF": bof_exp, "PVFP": pvfp_exp, "Proxy": prox_exp, "TVOG": tvog_exp, "Duration": dur_exp,
    "SCR": scr_expenses
}


# =============================================================================
# 4. MODULO LIFE: SHOCK SUI RISCATTI (LAPSE)
# =============================================================================

# --- Lapse Up (+50% tasso di riscatto base, max 100%) ---
CONFIG_LAPSE_UP = CONFIG_STO.copy()
CONFIG_LAPSE_UP["LAPSE_RATE"] = min(1.0, CONFIG_STO["LAPSE_RATE"] * 1.5)
pop_l_up = population_projection(q_x, CONFIG_LAPSE_UP)
bel_l_up, dur_l_up, pvfp_l_up, prox_l_up, mva_l_up, bof_l_up, leak_l_up, tvog_l_up, comp_l_up = bel_calculation(
    fund, comm_m, rd_m, v_base, pop_l_up, CONFIG_LAPSE_UP
)
delta_bof_l_up   = max(0, bof - bof_l_up)
results["Lapse_Up_Shock"] = {
    "MVA": mva_l_up, "BEL": bel_l_up,
    "PV_Death": comp_l_up["PV_Death"], "PV_Lapse": comp_l_up["PV_Lapse"], "PV_Exp": comp_l_up["PV_Exp"], "PV_Comm": comp_l_up["PV_Comm"], "PV_Premiums": comp_l_up["PV_Premiums"],
    "BOF": bof_l_up, "PVFP": pvfp_l_up, "Proxy": prox_l_up, "TVOG": tvog_l_up, "Duration": dur_l_up,
    "SCR": delta_bof_l_up
}

# --- Lapse Down (-50% tasso di riscatto base, floor a -20% assoluto) ---
CONFIG_LAPSE_DOWN = CONFIG_STO.copy()
CONFIG_LAPSE_DOWN["LAPSE_RATE"] = max(0.5 * CONFIG_STO["LAPSE_RATE"], CONFIG_STO["LAPSE_RATE"] - 0.2)
pop_l_down = population_projection(q_x, CONFIG_LAPSE_DOWN)
bel_l_down, dur_l_down, pvfp_l_down, prox_l_down, mva_l_down, bof_l_down, leak_l_down, tvog_l_down, comp_l_down = bel_calculation(
    fund, comm_m, rd_m, v_base, pop_l_down, CONFIG_LAPSE_DOWN
)
delta_bof_l_down = max(0, bof - bof_l_down)
results["Lapse_Down_Shock"] = {
    "MVA": mva_l_down, "BEL": bel_l_down,
    "PV_Death": comp_l_down["PV_Death"], "PV_Lapse": comp_l_down["PV_Lapse"], "PV_Exp": comp_l_down["PV_Exp"], "PV_Comm": comp_l_down["PV_Comm"], "PV_Premiums": comp_l_down["PV_Premiums"],
    "BOF": bof_l_down, "PVFP": pvfp_l_down, "Proxy": prox_l_down, "TVOG": tvog_l_down, "Duration": dur_l_down,
    "SCR": delta_bof_l_down
}

# --- Lapse Mass (Uscita istantanea del 40% del portafoglio al t=0) ---
pop_mass = population_projection(q_x, CONFIG_STO, lapse_mass=0.4)
bel_mass, dur_mass, pvfp_mass, prox_mass, mva_mass, bof_mass, leak_mass, tvog_mass, comp_mass = bel_calculation(
    fund, comm_m, rd_m, v_base, pop_mass, CONFIG_STO,
    )
delta_bof_l_mass = max(0, bof - bof_mass)
results["Lapse_Mass_Shock"] = {
    "MVA": mva_mass, "BEL": bel_mass,
    "PV_Death": comp_mass["PV_Death"], "PV_Lapse": comp_mass["PV_Lapse"], "PV_Exp": comp_mass["PV_Exp"], "PV_Comm": comp_mass["PV_Comm"], "PV_Premiums": comp_mass["PV_Premiums"],
    "BOF": bof_mass, "PVFP": pvfp_mass, "Proxy": prox_mass, "TVOG": tvog_mass, "Duration": dur_mass,
    "SCR": delta_bof_l_mass
}

# Il capitale richiesto per il rischio di riscatto è il massimo tra gli scenari Up, Down e Mass
scr_lapse = max(delta_bof_l_up, delta_bof_l_down, delta_bof_l_mass)
print(f"\nSCR LAPSE: {scr_lapse:,.2f} €")


# =============================================================================
# 5. MODULO MARKET: SHOCK DEGLI ASSET (EQUITY & PROPERTY)
# =============================================================================

# --- Equity Shock (Crollo del mercato azionario + symmetric adjustment) ---
eq_shock = 0.39 + CONFIG_STO["SYM_ADJ"]
fund_eq, _, comm_eq, rd_eq, _ = fund_simulation(
    fwd_base, Z_list, CONFIG_STO, asset_shocks=[eq_shock, 0.0], run_name="Equity_Shock"
)
bel_eq, dur_eq, pvfp_eq, prox_eq, mva_eq, bof_eq, leak_eq, tvog_eq, comp_eq = bel_calculation(
    fund_eq, comm_eq, rd_eq, v_base, pop, CONFIG_STO
)
scr_eq = max(0, bof - bof_eq)
results["Equity_Shock"] = {
    "MVA": mva_eq, "BEL": bel_eq,
    "PV_Death": comp_eq["PV_Death"], "PV_Lapse": comp_eq["PV_Lapse"], "PV_Exp": comp_eq["PV_Exp"], "PV_Comm": comp_eq["PV_Comm"], "PV_Premiums": comp_eq["PV_Premiums"],
    "BOF": bof_eq, "PVFP": pvfp_eq, "Proxy": prox_eq, "TVOG": tvog_eq, "Duration": dur_eq,
    "SCR": scr_eq
}

# --- Property Shock (Crollo del mercato immobiliare) ---
prop_shock = 0.25
fund_prop, _, comm_prop, rd_prop, _ = fund_simulation(
    fwd_base, Z_list, CONFIG_STO, asset_shocks=[0.0, prop_shock], run_name="Property_Shock"
)
bel_prop, dur_prop, pvfp_prop, prox_prop, mva_prop, bof_prop, leak_prop, tvog_prop, comp_prop = bel_calculation(
    fund_prop, comm_prop, rd_prop, v_base, pop, CONFIG_STO
)
scr_prop = max(0, bof - bof_prop)
results["Property_Shock"] = {
    "MVA": mva_prop, "BEL": bel_prop,
    "PV_Death": comp_prop["PV_Death"], "PV_Lapse": comp_prop["PV_Lapse"], "PV_Exp": comp_prop["PV_Exp"], "PV_Comm": comp_prop["PV_Comm"], "PV_Premiums": comp_prop["PV_Premiums"],
    "BOF": bof_prop, "PVFP": pvfp_prop, "Proxy": prox_prop, "TVOG": tvog_prop, "Duration": dur_prop,
    "SCR": scr_prop
}


# =============================================================================
# 8. MODULO MARKET: SHOCK DEI TASSI DI INTERESSE
# =============================================================================

# --- Interest Rate Up (Rialzo della curva dei tassi base) ---
spot_rates_up = load_eiopa_rates(CONFIG_STO, sheet_name="Spot_NO_VA_shock_UP")
v_in_up, fwd_in_up = rates_calculation(spot_rates_up)
fund_in_up, _, comm_in_up, rd_in_up, _ = fund_simulation(
    fwd_in_up, Z_list, CONFIG_STO, run_name="Interest_Rate_Up_Shock"
)
bel_in_up, dur_in_up, pvfp_in_up, prox_in_up, mva_in_up, bof_in_up, leak_in_up, tvog_in_up, comp_in_up = bel_calculation(
    fund_in_up, comm_in_up, rd_in_up, v_in_up, pop, CONFIG_STO,
)
delta_bof_in_up   = max(0, bof - bof_in_up)
results["Interest_Rate_Up_Shock"] = {
    "MVA": mva_in_up, "BEL": bel_in_up,
    "PV_Death": comp_in_up["PV_Death"], "PV_Lapse": comp_in_up["PV_Lapse"], "PV_Exp": comp_in_up["PV_Exp"], "PV_Comm": comp_in_up["PV_Comm"], "PV_Premiums": comp_in_up["PV_Premiums"],
    "BOF": bof_in_up, "PVFP": pvfp_in_up, "Proxy": prox_in_up, "TVOG": tvog_in_up, "Duration": dur_in_up,
    "SCR": delta_bof_in_up
}

# --- Interest Rate Down (Ribasso della curva dei tassi base) ---
spot_rates_down = load_eiopa_rates(CONFIG_STO, sheet_name="Spot_NO_VA_shock_DOWN")
v_in_down, fwd_in_down = rates_calculation(spot_rates_down)
fund_in_down, _, comm_in_down, rd_in_down, _ = fund_simulation(
    fwd_in_down, Z_list, CONFIG_STO, run_name="Interest_Rate_Down_Shock"
)
bel_in_down, dur_in_down, pvfp_in_down, prox_in_down, mva_in_down, bof_in_down, leak_in_down, tvog_in_down, comp_in_down = bel_calculation(
    fund_in_down, comm_in_down, rd_in_down, v_in_down, pop, CONFIG_STO
)
delta_bof_in_down = max(0, bof - bof_in_down)
results["Interest_Rate_Down_Shock"] = {
    "MVA": mva_in_down, "BEL": bel_in_down,
    "PV_Death": comp_in_down["PV_Death"], "PV_Lapse": comp_in_down["PV_Lapse"], "PV_Exp": comp_in_down["PV_Exp"], "PV_Comm": comp_in_down["PV_Comm"], "PV_Premiums": comp_in_down["PV_Premiums"],
    "BOF": bof_in_down, "PVFP": pvfp_in_down, "Proxy": prox_in_down, "TVOG": tvog_in_down, "Duration": dur_in_down,
    "SCR": delta_bof_in_down
}

# Il capitale richiesto per il rischio tassi è il massimo tra lo scenario Up e Down
scr_int = max(delta_bof_in_up, delta_bof_in_down)
print(f"\nSCR INTEREST RATE: {scr_int:,.2f} €")


# =============================================================================
# 8. OUTPUT FINALE E AGGREGAZIONE BSCR
# =============================================================================

# Formattazione e stampa della tabella dei risultati dei singoli moduli
df_results = pd.DataFrame(results).T
pd.options.display.float_format = '{:,.2f}'.format
#hide_print = ["PV_Death", "PV_Lapse", "PV_Exp", "PV_Comm", "PV_Premiums"]
hide_print = ["MVA", "PVFP", "Proxy"]
print("\n--- FINAL RESULTS TABLE ---")
print(df_results.drop(columns=hide_print, errors='ignore'))

# --- Aggregazione Market Risk ---
market_risks = np.array([scr_int, scr_eq, scr_prop])
market_corr_mtx = np.array([
    [1.00, 0.50, 0.50],  
    [0.50, 1.00, 0.75],  
    [0.50, 0.75, 1.00]   
])
scr_market = np.sqrt(np.dot(market_risks, np.dot(market_corr_mtx, market_risks)))
print(f"\nSCR MARKET: {scr_market:,.2f} €")

# --- Aggregazione Life Risk ---
life_risks= np.array([scr_mort, scr_expenses, scr_lapse, scr_cat])
life_corr_mtx = np.array([
    [1.00, 0.25, 0.00, 0.25], 
    [0.25, 1.00, 0.50, 0.25], 
    [0.00, 0.50, 1.00, 0.25], 
    [0.25, 0.25, 0.25, 1.00]  
])
scr_life = np.sqrt(np.dot(life_risks, np.dot(life_corr_mtx, life_risks)))
print(f"SCR LIFE: {scr_life:,.2f} €")

# --- Calcolo BSCR e Solvency Ratio ---
mod = np.array([scr_market, scr_life])
corr_bscr = np.array([
    [1.00, 0.25], 
    [0.25, 1.00]  
])
bscr = np.sqrt(np.dot(mod, np.dot(corr_bscr, mod)))

print(f"BSCR: {bscr:,.2f} €")

solvency_ratio = bof / bscr
print(f"Solvency Ratio: {solvency_ratio:.2%}")

# Opzionale: Esporta i risultati in Excel
# df_results.to_excel("Risultati_Modello.xlsx")
