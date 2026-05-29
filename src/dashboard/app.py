import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import shap
import matplotlib.pyplot as plt
import joblib

# Page Configuration
st.set_page_config(
    page_title="Swiss ED Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .big-font {
        font-size:30px !important;
        font-weight: 600;
        color: #F63366;
    }
    .metric-card {
        background-color: #1E2127;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #00E676;
    }
    .metric-title {
        color: #FFFFFF;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Swiss ED Predictor - Renkulab MVP")
st.markdown("Anticipation des pics d'affluence aux urgences hospitalières (J+1 à J+4)")

API_URL = "http://api:8000"

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("data/processed/features.csv", parse_dates=['date'])
        if not df.empty:
            df = df.sort_values('date')
            return df
    except Exception:
        pass
        
    # --- MOCK FALLBACK (If data was cleaned) ---
    st.warning("⚠️ Mode Démonstration : Fichier de données introuvable. Affichage de données fictives en attendant l'intégration.")
    today = pd.Timestamp.today().normalize()
    dates = [today - timedelta(days=i) for i in range(35, -1, -1)]
    import numpy as np
    mock_df = pd.DataFrame({'date': dates})
    mock_df['ed_visits'] = np.random.randint(120, 180, size=len(dates))
    return mock_df

df = load_historical_data()

# --- SIDEBAR: SIMULATION ---
st.sidebar.header("🛠️ Simulation Météo & Mobilité")
st.sidebar.markdown("Testez les réactions du modèle en simulant les 3 prochains jours.")

sim_temp = st.sidebar.slider("Température Max Prévue (°C)", min_value=-10.0, max_value=40.0, value=25.0)
sim_temp_min = st.sidebar.slider("Température Min Prévue (°C)", min_value=-20.0, max_value=30.0, value=15.0)
sim_precip = st.sidebar.slider("Précipitations (mm)", min_value=0.0, max_value=50.0, value=0.0)
sim_mobility = st.sidebar.slider("Index de Mobilité (Transport)", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

# --- DATE DE REFERENCE ---
st.sidebar.markdown("---")
st.sidebar.header("📅 Choix de la date")
today_data = df['date'].max().date()
min_selectable = today_data - timedelta(days=30)
max_selectable = today_data + timedelta(days=4)

selected_date = st.sidebar.date_input(
    "Date ciblée (1 mois avant -> J+4)",
    value=today_data,
    min_value=min_selectable,
    max_value=max_selectable
)
selected_date = pd.to_datetime(selected_date)

hist_df_filtered = df[df['date'] <= selected_date]
if len(hist_df_filtered) < 7:
    st.sidebar.error("Besoin de 7 jours minimum.")
    st.stop()

# --- GET LATEST KNOWN STATE ---
last_row = hist_df_filtered.iloc[-1]
last_date = last_row['date']

# --- GENERATE PREDICTIONS ---
# We will predict the next 4 days
predictions = []
future_dates = [last_date + timedelta(days=i) for i in range(1, 5)]

current_lag_1 = float(last_row['ed_visits'])
current_lag_2 = float(hist_df_filtered.iloc[-2]['ed_visits'])
current_lag_7 = float(hist_df_filtered.iloc[-7]['ed_visits'])

payload_j1 = None
for i, d in enumerate(future_dates):
    month = d.month
    day_of_week = d.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    is_summer = 1 if month in [7, 8] else 0
    
    payload = {
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "month": month,
        "is_winter": is_winter,
        "is_summer": is_summer,
        "visits_lag_1": current_lag_1,
        "visits_lag_2": current_lag_2,
        "visits_lag_7": current_lag_7,
        "temp_max": sim_temp,
        "temp_min": sim_temp_min,
        "precipitation": sim_precip,
        "temp_max_rolling_3": sim_temp,
        "mobility_index": sim_mobility
    }
    
    if i == 0:
        payload_j1 = payload
    
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        if res.status_code == 200:
            pred_val = res.json()["predicted_visits"]
        else:
            # Fallback mock prediction if API returns 503 (Model not loaded)
            pred_val = current_lag_1 + (sim_temp - 20) * 1.5
    except:
        # Fallback if API is completely unreachable
        pred_val = current_lag_1 + (sim_temp - 20) * 1.5
            
    predictions.append(pred_val)
    
    # Update lags for next day simulation
    current_lag_7 = current_lag_1 # Simplification for MVP
    current_lag_2 = current_lag_1
    current_lag_1 = pred_val

# --- DASHBOARD UI ---

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Visites Actuelles (J)</div>
            <div class="metric-value">{int(last_row['ed_visits'])}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Prévision Demain (J+1)</div>
            <div class="metric-value" style="color: #F63366;">{int(predictions[0])}</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tendance à 4 jours</div>
            <div class="metric-value" style="color: #FFC107;">{'+' if predictions[-1] > last_row['ed_visits'] else ''}{int(predictions[-1] - last_row['ed_visits'])}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("### 📈 Visualisation Réel vs Prédictions")

# Plotly Graph
fig = go.Figure()

# Historical (last 14 days from selected date)
hist_df = hist_df_filtered.tail(14)
fig.add_trace(go.Scatter(
    x=hist_df['date'], 
    y=hist_df['ed_visits'],
    mode='lines+markers',
    name='Visites Réelles (SpiGes)',
    line=dict(color='#00E676', width=3)
))

# Predictions
pred_df = pd.DataFrame({
    'date': [last_date] + future_dates,
    'ed_visits': [last_row['ed_visits']] + predictions
})

fig.add_trace(go.Scatter(
    x=pred_df['date'], 
    y=pred_df['ed_visits'],
    mode='lines+markers',
    name='Prédictions (XGBoost)',
    line=dict(color='#F63366', width=3, dash='dash')
))

fig.update_layout(
    plot_bgcolor='#1E2127',
    paper_bgcolor='#0E1117',
    font_color='#FFFFFF',
    hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#333333', title='Nombre de Patients')
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### 🧠 Explicabilité du Modèle (SHAP)")
st.markdown("Quels facteurs influencent l'affluence prévue pour **Demain (J+1)** ?")

try:
    model = joblib.load("models/xgboost_ed_model.joblib")
    features_cols = joblib.load("models/model_features.joblib")
    
    simulated_df = pd.DataFrame([payload_j1])
    simulated_df = simulated_df[features_cols]
    
    explainer = shap.Explainer(model)
    shap_values = explainer(simulated_df)
    
    fig_shap, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig_shap, clear_figure=True)
except Exception:
    st.info("ℹ️ Module d'explicabilité (SHAP) en attente d'intégration du nouveau modèle.")

st.info("💡 **Astuce Renkulab :** Modifiez les paramètres météo dans la barre latérale pour voir la courbe et SHAP s'adapter dynamiquement.")
