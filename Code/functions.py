# =============================================================================
# MODULO DELLE FUNZIONI ATTUARIALI E FINANZIARIE
# =============================================================================
import numpy as np
import pandas as pd
import openpyxl as px
from pathlib import Path

# =============================================================================
# 1. ELABORAZIONE DELLE CURVE DEI TASSI (RATES CALCULATION)
# =============================================================================
def rates_calculation(spot_curve):
    """
    Converte la curva dei tassi Spot (es. fornita da EIOPA) in fattori di sconto
    discreti e tassi forward continui, necessari per la simulazione Black-Scholes.
    """
    n = len(spot_curve)
    discrete_discount_factors = np.zeros(n)
    continous_fwd_rates = np.zeros(n)
    
    # Il fattore di sconto al tempo 0 è sempre 1 (il valore dei soldi oggi)
    discrete_discount_factors[0] = 1.000
    
    for t in range(1, n):
        # Fattore di sconto discreto: v = 1 / (1 + r)^t
        discrete_discount_factors[t] = 1 / (1 + spot_curve[t])**t
        
        # Tasso forward continuo implicito tra t-1 e t
        continous_fwd_rates[t] = np.log(((1 + spot_curve[t])**t) / ((1 + spot_curve[t-1])**(t-1)))
        
    return discrete_discount_factors, continous_fwd_rates


# =============================================================================
# 2. GENERATORE DI SCENARI ECONOMICI E PROIEZIONE FONDO (FUND SIMULATION)
# =============================================================================
def fund_simulation(fwd_rates, z_list, config, asset_shocks=None, run_name="unknown"):
    """
    Simula l'evoluzione stocastica degli asset e del fondo totale usando il
    Moto Browniano Geometrico, applicando la Martingale Deflation per garantire
    la neutralità al rischio (Market Consistency).
    """
    n_sim = config["NUMBER_SIMULATIONS"]
    horizon = config["HORIZON"]
    premium = config["SINGLE_PREMIUM"] * config["NUMBER_PH"]
    
    weights = np.array(config["WEIGHTS"])
    sigmas = np.array(config["VOLATILITIES"])
    num_assets = len(weights)
    
    # Inizializzazione delle matrici tridimensionali (Asset, Simulazioni, Tempo)
    asset_paths = np.zeros((num_assets, n_sim, horizon + 1))
    fund_paths = np.zeros((n_sim, horizon + 1))
    growth_factors = np.zeros((num_assets, n_sim, horizon))
    
    # Vettore degli shock istantanei (usato per l'Equity/Property risk)
    shocks = np.array(asset_shocks) if asset_shocks is not None else np.zeros(num_assets)

    # Setup del Tempo 0: Investimento del premio iniziale decurtato dell'eventuale shock
    for i in range(num_assets):
        asset_paths[i, :, 0] = (premium * weights[i]) * (1 - shocks[i]) 
        fund_paths[:, 0] += asset_paths[i, :, 0]
        
    # Matrici per tracciare le trattenute di gestione (RD) e le commissioni (COMM)
    comm_matrix = np.zeros((n_sim, horizon))
    rd_matrix = np.zeros((n_sim, horizon))

    # Tracciatore per la Martingale Deflation Cumulativa
    cum_growth_empirical = np.ones((num_assets, n_sim))
    
    # Ciclo temporale di proiezione
    for t in range(horizon):
        f = fwd_rates[t+1] 
        gross_assets = np.zeros((num_assets, n_sim))
        gross_fund = np.zeros(n_sim)
        
        for i in range(num_assets):
            # --- Proiezione Stocastica Grezza ---
            raw_factor = np.exp(f - 0.5 * sigmas[i]**2 + sigmas[i] * z_list[i][:, t])
            
            # --- Martingale Deflation (Correzione Errore Statistico) ---
            test_cum = cum_growth_empirical[i] * raw_factor
            empirical_mean_cum = np.mean(test_cum)
            theoretical_mean_cum = np.mean(cum_growth_empirical[i]) * np.exp(f)
            correction = theoretical_mean_cum / empirical_mean_cum
            
            # Applicazione della correzione
            factor = raw_factor * correction
            cum_growth_empirical[i] = cum_growth_empirical[i] * factor
            
            # --- Salvataggio e Aggregazione ---
            growth_factors[i, :, t] = factor
            gross_assets[i] = asset_paths[i, :, t] * factor
            gross_fund += gross_assets[i]

        # Calcolo dei flussi in uscita dal fondo (Commissioni e Retained Deduction)
        rd_matrix[:, t] = gross_fund * config["RD"]
        comm_matrix[:, t] = gross_fund * config["COMM"]
        
        # Aggiornamento del fondo netto per l'anno successivo
        fund_paths[:, t+1] = gross_fund - rd_matrix[:, t]
        
        # Riapporzionamento degli asset al netto del Management Fee
        for i in range(num_assets):
            asset_paths[i, :, t+1] = gross_assets[i] * (1 - config["RD"])

    # --- CONTROLLO DI QUALITÀ: MARTINGALE TEST ---
    sum_fwd = np.sum(fwd_rates[1:horizon+1])
    v_last = np.exp(-sum_fwd)
    cum_growth = np.prod(growth_factors, axis=2)
    mean_growth = np.mean(cum_growth, axis=1)
    d_value = mean_growth * v_last
    
    errors = np.abs(d_value - 1.0)
    max_error = np.max(errors)
    worst_asset = np.argmax(errors)
    tolleranza = 0.001 # Soglia dello 0.1%

    if max_error > tolleranza:
        print(f"\n⚠️  Warning: Simulations might be insufficient for {run_name}!")
        print(f"   Max Martingale Error: {max_error:.4%} (Asset index {worst_asset})")
    """
    # 1. Fattore di sconto finale (invariato)
    sum_fwd = np.sum(fwd_rates[1:horizon+1])
    v_last = np.exp(-sum_fwd)
    
    # 2. Isoliamo SOLO i fattori di crescita dell'Equity (Asset 0)
    # growth_factors[0] ha dimensione (n_sim, horizon), quindi moltiplichiamo lungo il tempo (axis=1)
    cum_growth_eq = np.prod(growth_factors[0], axis=1) 
    
    # 3. Media su tutti gli scenari e attualizzazione
    mean_growth_eq = np.mean(cum_growth_eq)
    d_value_eq = mean_growth_eq * v_last
    
    # 4. Calcolo dell'errore esclusivo per l'Equity
    error_eq = np.abs(d_value_eq - 1.0)
    tolleranza = 0.00001 # Soglia dello 0.1%

    # 5. Alert mirato
    if error_eq > tolleranza:
        print(f"\n⚠️  Warning: Simulations might be insufficient for {run_name}!")
        print(f"   Equity Martingale Error: {error_eq:.4%}")
    """
    return fund_paths, asset_paths, comm_matrix, rd_matrix, growth_factors


# =============================================================================
# 3. PROIEZIONE DEMOGRAFICA (POPULATION PROJECTION)
# =============================================================================
def population_projection(qx_base, config, lapse_mass=0.00):
    """
    Proietta il numero di assicurati in vita, i decessi e i riscatti (lapses).
    Gestisce l'interazione tra mortalità naturale, riscatti fisiologici e 
    shock di riscatto massivo (lapse_mass).
    """
    horizon = config["HORIZON"]
    alive = np.zeros(horizon + 1)
    deaths = np.zeros(horizon)
    lapses = np.zeros(horizon)
    mass_lapses = np.zeros(horizon)
    
    alive[0] = float(config["NUMBER_PH"])

    for t in range(horizon):
        # 1. Si calcolano prima i decessi sulla popolazione iniziale dell'anno
        deaths[t] = alive[t] * qx_base[t+1]
        
        if t == 0 and lapse_mass > 0.0:
            # 2a. Anno dello Shock Massivo: applicato sui sopravvissuti
            mass_lapses[t] = (alive[t] - deaths[t]) * lapse_mass 
            # I riscatti totali sono la somma dello shock e del riscatto ordinario sulla parte rimanente
            lapses[t] = (alive[t] - deaths[t] - mass_lapses[t]) * config["LAPSE_RATE"] + mass_lapses[t]
            
        elif t == horizon - 1:
            # 2b. Ultimo anno (Scadenza): Tutti i sopravvissuti escono dal fondo
            lapses[t] = alive[t] - deaths[t]
            
        else:
            # 2c. Anni normali: Riscatto ordinario applicato sui sopravvissuti
            lapses[t] = (alive[t] - deaths[t]) * config["LAPSE_RATE"]
            
        # 3. Aggiornamento della popolazione in vita per l'anno successivo
        alive[t+1] = alive[t] - deaths[t] - lapses[t]
        
    return alive, deaths, lapses


# =============================================================================
# 4. VALUTAZIONE ATTUARIALE (BEL E CASHFLOWS)
# =============================================================================
def bel_calculation(fund_paths, comm_matrix, rd_matrix, discounts_factors, population, config):
    """
    Calcola la Best Estimate Liability (BEL), il Time Value of Options and Guarantees 
    (TVOG) e i profitti futuri attesi (PVFP) attualizzando i cashflow della compagnia.
    """
    mva = np.mean(fund_paths[:, 0]) # Valore di mercato degli attivi al tempo 0
    alive, deaths, lapses = population
    n_sim = fund_paths.shape[0]
    horizon = config["HORIZON"]
    
    # Inizializzazione vettori dei risultati per scenario
    bel_scenarios = np.zeros(n_sim)
    d_inflows = np.zeros(n_sim)
    d_outflows = np.zeros(n_sim)
    duration_scenarios = np.zeros(n_sim)
    tvog_scenarios = np.zeros(n_sim)
    proxy_profit = np.zeros(n_sim)

    # Componenti della BEL (per analisi di attribuzione)
    pv_death = np.zeros(n_sim)
    pv_lapse = np.zeros(n_sim)
    pv_expenses = np.zeros(n_sim)
    pv_comm = np.zeros(n_sim)

    for t in range(horizon):
        # Valore della singola quota per assicurato
        single_fund = fund_paths[:, t+1] / config["NUMBER_PH"]

        # Costo extra della garanzia (se il fondo scende sotto il premio versato)
        extra_cost_guarantee = np.maximum(config["SINGLE_PREMIUM"] - single_fund, 0)
        
        # PRESTAZIONI CASO MORTE: Si paga il massimo tra premio versato e valore del fondo
        death_benefits_scenarios = np.maximum(config["SINGLE_PREMIUM"], single_fund)
        death_benefits = deaths[t] * death_benefits_scenarios

        # PRESTAZIONI CASO RISCATTO: Si applica la penale (salvo a scadenza)
        if t == horizon - 1:
            lapse_benefits_scenarios = np.maximum(0, single_fund)
        else:
            lapse_benefits_scenarios = np.maximum(0, single_fund - config["PENALTY"])
            
        lapse_benefits = lapses[t] * lapse_benefits_scenarios
        
        # SPESE E COMMISSIONI (Uscite per la compagnia)
        expenses = alive[t] * config["EXPENSE"] * (1 + config["INFLATION"])**(t+1)
        single_comm = comm_matrix[:, t] / config["NUMBER_PH"]   
        comm = alive[t] * single_comm

        # RETAINED DEDUCTION (Entrate per la compagnia)
        single_rd = rd_matrix[:, t] / config["NUMBER_PH"]
        rd = alive[t] * single_rd

        # Aggregazione Flussi Attualizzati per il calcolo del PVFP (Present Value of Future Profits)
        inflows = rd
        d_inflows += inflows * discounts_factors[t+1]
        
        outflows = expenses + comm
        d_outflows += outflows * discounts_factors[t+1]

        # Scomposizione delle Passività
        pv_death += death_benefits * discounts_factors[t+1]
        pv_lapse += lapse_benefits * discounts_factors[t+1]
        pv_expenses += expenses * discounts_factors[t+1]
        pv_comm += comm * discounts_factors[t+1]
        
        # Le passività totali sono i pagamenti agli assicurati + le spese della compagnia
        liabilities = death_benefits + lapse_benefits + outflows
        bel_scenarios += liabilities * discounts_factors[t+1]

        # Calcolo della Macaulay Duration pesata per i flussi di cassa
        w_duration = (liabilities * discounts_factors[t+1]) * (t+1)
        duration_scenarios += w_duration

        # Calcolo del TVOG (valore attualizzato delle garanzie pagate ai deceduti)
        tvog_scenarios += (extra_cost_guarantee * deaths[t]) * discounts_factors[t+1]

        # Calcolo Proxy Profit (utile solo al primo anno per approssimazioni rapide)
        if t == 0:
            proxy_profit = (inflows - outflows) * discounts_factors[t+1]

    # --- AGGREGAZIONE FINALE (Media sugli scenari) ---
    bel = np.mean(bel_scenarios)
    duration = np.mean(duration_scenarios) / bel
    tvog = np.mean(tvog_scenarios)
    proxy = np.mean(proxy_profit) * duration
    pvfp = np.mean(d_inflows) - np.mean(d_outflows)
    
    # Own Funds (Capitale Proprio o Basic Own Funds)
    bof = mva - bel
    
    # Leakage (Verifica di bilancio: deve essere prossimo a 0)
    leakage = mva - (bel + pvfp)

    bel_components = {
        "PV_Premiums": 0.0,
        "PV_Death": np.mean(pv_death),
        "PV_Lapse": np.mean(pv_lapse),
        "PV_Exp": np.mean(pv_expenses),
        "PV_Comm": np.mean(pv_comm)
    }

    return bel, duration, pvfp, proxy, mva, bof, leakage, tvog, bel_components