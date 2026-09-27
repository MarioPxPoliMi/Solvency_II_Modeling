# =============================================================================
# MODULO DI IMPORTAZIONE DATI (DATA LOADER)
# =============================================================================
import numpy as np
import pandas as pd
import openpyxl as px
from pathlib import Path

# =============================================================================
# 1. SETUP DEI PERCORSI E DELLE DIRECTORY
# =============================================================================
# Usa pathlib per trovare automaticamente la cartella "Data" ovunque si trovi
# il progetto sul tuo computer, evitando errori di percorsi assoluti.
root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "Data"


# =============================================================================
# 2. CARICAMENTO DELLE TAVOLE DI MORTALITÀ (ISTAT)
# =============================================================================
def load_mortality_rates(config):
    """
    Legge le tavole di mortalità della popolazione residente (solitamente ISTAT).
    Estrae il vettore q_x (probabilità di decesso entro l'anno) in base all'età
    di partenza e all'orizzonte temporale della simulazione.
    """
    horizon = config["HORIZON"]
    starting_age = config["STARTING_AGE"]

    # Caricamento del file Excel
    df_death_rates = pd.read_excel(data_dir / "Tavole di mortalità della popolazione residente.xlsx")
    
    # Estrazione dei dati:
    # - Le righe partono da 'starting_age + 1' (scartando l'anno 0 di vita)
    # - Si estrae la colonna 3 (indice 3) che contiene i decessi per millennio
    # - Si divide per 1000 per ottenere una probabilità q_x (da 0.0 a 1.0)
    q_x = df_death_rates.iloc[starting_age + 1 : starting_age + 1 + horizon, 3].values.astype(float) / 1000
    
    # Inseriamo un 0.0 al tempo t=0 perché al momento della stipula (t=0) non ci sono decessi
    q_x = np.insert(q_x, 0, 0.0)

    return q_x


# =============================================================================
# 3. CARICAMENTO DELLE CURVE DEI TASSI RISK-FREE (EIOPA)
# =============================================================================
def load_eiopa_rates(config, sheet_name="RFR_spot_no_VA", date_str="20251231"):
    """
    Legge la struttura per scadenza dei tassi privi di rischio pubblicata da EIOPA.
    Tramite il parametro 'sheet_name' può caricare sia la curva base che le 
    curve stressate per il calcolo dell'Interest Rate SCR (es. Spot_NO_VA_shock_UP).
    
    NOTA: Per gli shock, assicurarsi di aver salvato il file Excel originale 
    per forzare il calcolo delle formule nella cache di lettura di Pandas.
    """
    horizon = config["HORIZON"]
    
    # I tassi nei file EIOPA (per la valuta Euro) iniziano solitamente dalla riga 11 (indice 10 in pandas)
    start_index = 10 
    
    # Costruzione dinamica del nome del file (utile se si vuole aggiornare la data di valutazione)
    file_name = f"EIOPA_RFR_{date_str}_Term_Structures.xlsx"
    file_path = data_dir / file_name
    
    # Caricamento senza header per non confondere l'indice delle righe
    df_rates = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    # Estrazione dei dati:
    # - Si prendono le righe dalla 10 fino all'orizzonte scelto (es. 50 anni)
    # - Si estrae la colonna 2 (Colonna C in Excel, solitamente la curva Euro)
    spot_rates = df_rates.iloc[start_index : start_index + horizon, 2].values.astype(float) 
    
    # Inseriamo un tasso 0.0 al tempo t=0 (il tasso spot per scadenza istantanea è per convenzione 0)
    spot_rates = np.insert(spot_rates, 0, 0.0)
    
    return spot_rates