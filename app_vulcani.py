import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from datetime import date

# --- DIZIONARIO VULCANI EUROPEI ---
vulcani = {
    "Cumbre_Vieja_LaPalma": (28.568, -17.833),
    "Etna": (37.751, 14.995),
    "Campi_Flegrei": (40.827, 14.139),
    "Vesuvio": (40.822, 14.426),
    "Stromboli": (38.789, 15.213),
    "Fagradalsfjall": (63.893, -22.271),
    "Katla": (63.633, -19.050),
    "Hekla": (63.983, -19.700),
    "Santorini": (36.404, 25.396),
    "Teide_Tenerife": (28.272, -16.642)
}

st.set_page_config(page_title="Monitoraggio Vulcani AI", page_icon="🌋", layout="wide")

st.title("🌋 Analisi Sismica e Allerta Vulcanica in Tempo Reale")
st.markdown("""
Questa dashboard si collega in diretta ai server sismici europei (EMSC). Scegli un vulcano e un intervallo di date: il sistema scaricherà i dati storici su richiesta, calcolerà l'energia rilasciata e applicherà l'Intelligenza Artificiale per scovare i pattern di pre-eruzione.
""")

# --- MOTORE DI DOWNLOAD E CALCOLO (Nascosto in Cache) ---
# La cache evita di riscaricare i dati se cambiamo solo le opzioni visive
@st.cache_data(show_spinner=False)
def scarica_e_analizza(vulcano, d_inizio, d_fine):
    lat, lon = vulcani[vulcano]
    url_emsc = "https://www.seismicportal.eu/fdsnws/event/1/query"
    
    # Raggio di circa 30km (0.27 gradi)
    params = {
        "latitude": lat, "longitude": lon, "maxradius": 0.27,
        "starttime": f"{d_inizio}T00:00:00",
        "endtime": f"{d_fine}T23:59:59",
        "minmag": 1.0, "format": "text", "limit": 20000
    }
    
    response = requests.get(url_emsc, params=params, timeout=45)
    
    if response.status_code == 204:
        return None, "Nessun terremoto rilevato in questo periodo."
    elif response.status_code != 200:
        return None, f"Errore di comunicazione col server sismico: Status {response.status_code}"
        
    df = pd.read_csv(io.StringIO(response.text), sep='|')
    
    # 1. Pulizia Colonne (Lo Smart Matching)
    df.columns = [col.replace('#', '').strip() for col in df.columns]
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower: col_mapping['time'] = col
        elif 'lat' in col_lower: col_mapping['lat'] = col
        elif 'lon' in col_lower: col_mapping['lon'] = col
        elif 'depth' in col_lower: col_mapping['depth'] = col
        elif 'magnitude' in col_lower: col_mapping['mag'] = col
        
    if len(col_mapping) < 5:
        return None, "Dati incompleti restituiti dal server."
        
    df = df[[col_mapping['time'], col_mapping['lat'], col_mapping['lon'], col_mapping['depth'], col_mapping['mag']]].copy()
    df.columns = ['time', 'lat', 'lon', 'profondita_km', 'magnitudo']
    df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    df = df.sort_values('time')
    
    # 2. Feature Engineering (Calcolo dinamiche magma)
    df.set_index('time', inplace=True)
    df['energia_joule'] = 10 ** (4.8 + 1.5 * df['magnitudo'])
    df['eventi_7g'] = df['magnitudo'].rolling('7D').count()
    df['energia_totale_7g'] = df['energia_joule'].rolling('7D').sum()
    df['profondita_media_7g'] = df['profondita_km'].rolling('7D').mean()
    
    df.reset_index(inplace=True)
    df['trend_profondita'] = df['profondita_media_7g'].diff(periods=7)
    df.fillna(0, inplace=True)
    
    return df, "OK"


# --- INTERFACCIA UTENTE (SIDEBAR) ---
st.sidebar.header("⚙️ 1. Parametri di Ricerca")

# Il blocco form raggruppa le scelte e invia la richiesta solo alla pressione del tasto
with st.sidebar.form("pannello_controllo"):
    vulcano_scelto = st.selectbox("Seleziona il Vulcano:", list(vulcani.keys()))
    
    # SBLOCCO DELLE DATE: Ora partono dal 1980 e non possono superare "oggi"
    data_inizio = st.date_input("Data di Inizio", value=date(2021, 8, 1), min_value=date(1980, 1, 1), max_value=date.today())
    data_fine = st.date_input("Data di Fine", value=date(2022, 1, 1), min_value=date(1980, 1, 1), max_value=date.today())
    
    pulsante_scarica = st.form_submit_button("⬇️ Scarica e Analizza Dati")

st.sidebar.header("🧠 2. Modello AI")
attiva_ai = st.sidebar.checkbox("Mostra Allarmi AI sul grafico", value=False)


# --- ESECUZIONE DEL PROGRAMMA ---
# Inizializza una variabile di memoria per tenere il grafico a schermo anche se si preme solo la checkbox dell'IA
if pulsante_scarica:
    st.session_state['dati_pronti'] = True

if 'dati_pronti' in st.session_state and st.session_state['dati_pronti']:
    
    with st.spinner(f"📡 Connessione al server europeo in corso per {vulcano_scelto.replace('_', ' ')}..."):
        df, messaggio = scarica_e_analizza(vulcano_scelto, data_inizio, data_fine)
        
    if df is None:
        st.warning(messaggio)
    else:
        # 3. Intelligenza Artificiale (Viene calcolata istantaneamente sui dati in memoria)
        colonne_addestramento = ['eventi_7g', 'energia_totale_7g', 'profondita_media_7g', 'trend_profondita']
        X = df[colonne_addestramento].copy()
        
        if len(df) > 50:
            modello = IsolationForest(contamination=0.02, random_state=42, n_estimators=100)
            df['anomalia'] = modello.fit_predict(X)
        else:
            df['anomalia'] = 1 # Dati insufficienti per l'IA
            
        anomalie = df[df['anomalia'] == -1]
        
        # --- METRICHE A SCHERMO ---
        col1, col2 = st.columns(2)
        col1.metric(f"Eventi registrati dal {data_inizio.strftime('%d/%m/%Y')}", f"{len(df):,}")
        
        if attiva_ai:
            if len(df) <= 50:
                col2.metric("Allarmi AI", "Dati insufficienti")
            else:
                col2.metric("Allarmi AI Rilevati", f"{len(anomalie):,}")
        else:
            col2.metric("Allarmi AI", "Spenta")
            
        # --- CREAZIONE DEL GRAFICO ---
        st.subheader("Visualizzazione Dinamiche Sismiche")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        scatter = ax.scatter(
            df['time'], 
            -df['profondita_km'], 
            c=df['magnitudo'], 
            cmap='YlOrRd', 
            s=df['magnitudo']**2 * 3, 
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        plt.colorbar(scatter, label='Magnitudo', ax=ax)
        
        if attiva_ai and not anomalie.empty:
            ax.scatter(
                anomalie['time'], 
                -anomalie['profondita_km'], 
                color='none', 
                s=100, 
                edgecolor='red', 
                linewidth=2.5, 
                zorder=5, 
                label='🚨 Anomalia rilevata da AI'
            )
            ax.legend(loc='lower left')
            
        ax.set_xlabel('Data')
        ax.set_ylabel('Profondità (Km)')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)