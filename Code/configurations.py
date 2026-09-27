# =============================================================================
# FILE DI CONFIGURAZIONE DEI PARAMETRI ATTUARIALI E FINANZIARI
# =============================================================================
import numpy as np

# =============================================================================
# 1. CONFIGURAZIONE DETERMINISTICA (CASO BASE)
# =============================================================================
# Questa configurazione azzera le volatilità del mercato (VOLATILITIES: [0.0, 0.0]).
# Serve per calcolare il valore atteso puro del fondo e la BEL deterministica,
# facendo crescere gli asset esattamente secondo i tassi forward (Risk-Free).

CONFIG_DET = {
    "NUMBER_SIMULATIONS": 10000,  # Numero di scenari (ridondante senza volatilità, ma utile per il codice)
    "SINGLE_PREMIUM": 100000,       # Premio unico versato da ogni singolo assicurato alla stipula (€)
    "WEIGHTS": [0.80, 0.20],        # Asset Allocation iniziale: [80% Equity, 20% Property]
    "VOLATILITIES": [0.0, 0.0],     # Volatilità annua: azzerata per la proiezione deterministica
    "PENALTY": 20,                  # Penale fissa applicata al valore del fondo in caso di riscatto
    "RD": 0.022,                    # Retained Deduction / Management Fee: trattenuta di gestione sul fondo (2.2%)
    "COMM": 0.014,                  # Commissioni annue (1.4%) pagate dalla compagnia
    "NUMBER_PH": 100,               # Numero iniziale di assicurati (PolicyHolders) nel portafoglio
    "LAPSE_RATE": 0.15,             # Tasso di riscatto base annuo (15% degli assicurati vivi esce ogni anno)
    "EXPENSE": 50,                  # Spese di gestione fisse annue per ogni assicurato ancora in vita (€)
    "INFLATION": 0.02,              # Tasso di inflazione annuo (2%) applicato per rivalutare le spese fisse
    "HORIZON": 50,                  # Orizzonte temporale di proiezione del portafoglio (50 anni)
    "SYM_ADJ": 0.079,               # Symmetric Adjustment (7.9%) fornito da EIOPA per lo shock Equity
    "STARTING_AGE": 60,             # Età iniziale della coorte di assicurati
}


# =============================================================================
# 2. CONFIGURAZIONE STOCASTICA (MONTE CARLO / RISK-NEUTRAL)
# =============================================================================
# Questa configurazione attiva i motori di simulazione Monte Carlo inserendo
# le volatilità reali degli asset. È essenziale per il calcolo del TVOG (Costo 
# delle Garanzie) e per valutare gli SCR (Shock di mercato e biometrici).

CONFIG_STO = {
    "NUMBER_SIMULATIONS": 10000,  # Numero di scenari stocastici (i "percorsi" di mercato generati)
    "SINGLE_PREMIUM": 100000,       # Premio unico versato da ogni singolo assicurato (€)
    "WEIGHTS": [0.80, 0.20],        # Asset Allocation iniziale: [80% Equity, 20% Property]
    "VOLATILITIES": [0.2, 0.1],     # Volatilità annua: 20% per il mercato Equity, 10% per il Property
    "PENALTY": 20,                  # Penale applicata in caso di riscatto
    "RD": 0.022,                    # Trattenuta di gestione sul fondo (2.2% annuo)
    "COMM": 0.014,                  # Commissioni annue (1.4%)
    "NUMBER_PH": 100,               # Numero iniziale di assicurati
    "LAPSE_RATE": 0.15,             # Tasso di riscatto base (15% annuo)
    "EXPENSE": 50,                  # Spese di gestione fisse per assicurato vivo (€)
    "INFLATION": 0.02,              # Tasso di inflazione per rivalutare le spese (2%)
    "HORIZON": 50,                  # Orizzonte temporale (50 anni)
    "SYM_ADJ": 0.079,               # Symmetric Adjustment EIOPA (7.9%)
    "STARTING_AGE": 60,             # Età di partenza degli assicurati
}