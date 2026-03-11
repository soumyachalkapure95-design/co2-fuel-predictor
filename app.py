import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO2 & Fuel Predictor",
    page_icon="🚗",
    layout="centered"
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Exo+2:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Exo 2', sans-serif;
            background-color: #0d1117;
            color: #e6edf3;
        }
        .main { background-color: #0d1117; }

        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif;
            color: #58a6ff;
        }

        .title-block {
            text-align: center;
            padding: 2rem 1rem 1rem 1rem;
        }
        .title-block h1 {
            font-size: 2.2rem;
            letter-spacing: 2px;
            color: #58a6ff;
            text-shadow: 0 0 20px #58a6ff55;
        }
        .title-block p {
            color: #8b949e;
            font-size: 1rem;
            margin-top: 0.3rem;
        }

        .result-card {
            background: linear-gradient(135deg, #161b22, #1c2128);
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 1.5rem 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(88, 166, 255, 0.08);
        }
        .result-card h3 {
            font-size: 1rem;
            color: #8b949e;
            margin-bottom: 0.3rem;
            font-family: 'Exo 2', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .result-value {
            font-size: 2.4rem;
            font-weight: 700;
            font-family: 'Orbitron', sans-serif;
        }
        .good  { color: #3fb950; }
        .warn  { color: #d29922; }
        .bad   { color: #f85149; }

        .suggestion-box {
            background: #161b22;
            border-left: 4px solid #58a6ff;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 0.5rem 0;
            font-size: 0.95rem;
            color: #c9d1d9;
        }
        .suggestion-box.good  { border-left-color: #3fb950; }
        .suggestion-box.warn  { border-left-color: #d29922; }
        .suggestion-box.bad   { border-left-color: #f85149; }
        .suggestion-box.info  { border-left-color: #58a6ff; }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stNumberInput"] label {
            color: #8b949e !important;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.7rem 2rem;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
            letter-spacing: 1px;
            width: 100%;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(31, 111, 235, 0.5);
        }

        .divider {
            border: none;
            border-top: 1px solid #21262d;
            margin: 1.5rem 0;
        }

        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD & TRAIN MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("co2_emissions.csv")
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)

    le_vc = LabelEncoder()
    le_tr = LabelEncoder()
    le_ft = LabelEncoder()

    df["Vehicle Class Enc"] = le_vc.fit_transform(df["Vehicle Class"])
    df["Transmission Enc"]  = le_tr.fit_transform(df["Transmission"])
    df["Fuel Type Enc"]     = le_ft.fit_transform(df["Fuel Type"])

    FEATURES = ["Engine Size(L)", "Cylinders", "Vehicle Class Enc", "Transmission Enc", "Fuel Type Enc"]

    X = df[FEATURES]

    co2_model = RandomForestRegressor(n_estimators=100, random_state=42)
    co2_model.fit(X, df["CO2 Emissions(g/km)"])

    fuel_model = RandomForestRegressor(n_estimators=100, random_state=42)
    fuel_model.fit(X, df["Fuel Consumption Comb (L/100 km)"])

    return df, le_vc, le_tr, le_ft, co2_model, fuel_model, FEATURES

df, le_vc, le_tr, le_ft, co2_model, fuel_model, FEATURES = load_and_train()

# ─────────────────────────────────────────────────────────
# LABEL MAPS
# ─────────────────────────────────────────────────────────
FUEL_TYPE_MAP = {
    "D": "D — Diesel",
    "E": "E — Ethanol (E85)",
    "N": "N — Natural Gas (CNG)",
    "X": "X — Regular Gasoline (87 octane)",
    "Z": "Z — Premium Gasoline (91+ octane)"
}

TRANSMISSION_PREFIX_MAP = {
    "A":  "Automatic",
    "AM": "Automated Manual",
    "AS": "Automatic with Select Shift",
    "AV": "Continuously Variable (CVT)",
    "M":  "Manual"
}

def trans_label(code):
    for key in sorted(TRANSMISSION_PREFIX_MAP.keys(), key=len, reverse=True):
        if code.startswith(key):
            return f"{code} — {TRANSMISSION_PREFIX_MAP[key]}"
    return code

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
    <div class="title-block">
        <h1>🚗 CO2 & FUEL PREDICTOR</h1>
        <p>Enter your vehicle details to predict CO2 emissions & fuel consumption</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# USER INPUT FORM
# ─────────────────────────────────────────────────────────
st.markdown("### 🔧 Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    vehicle_options = sorted(df["Vehicle Class"].unique())
    vehicle_class = st.selectbox("🚗 Vehicle Class", vehicle_options)

    trans_options = sorted(df["Transmission"].unique())
    trans_labels  = [trans_label(t) for t in trans_options]
    trans_idx     = st.selectbox("⚙️ Transmission", range(len(trans_options)), format_func=lambda i: trans_labels[i])
    transmission  = trans_options[trans_idx]

with col2:
    fuel_options  = sorted(df["Fuel Type"].unique())
    fuel_labels   = [FUEL_TYPE_MAP.get(f, f) for f in fuel_options]
    fuel_idx      = st.selectbox("⛽ Fuel Type", range(len(fuel_options)), format_func=lambda i: fuel_labels[i])
    fuel_type     = fuel_options[fuel_idx]

    engine_size = st.slider(
        "🔩 Engine Size (L)",
        min_value=float(round(df["Engine Size(L)"].min(), 1)),
        max_value=float(round(df["Engine Size(L)"].max(), 1)),
        value=2.0, step=0.1
    )

cylinders = st.select_slider(
    "🔢 Number of Cylinders",
    options=sorted(df["Cylinders"].unique().astype(int).tolist())
)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────
if st.button("🚀 PREDICT NOW"):

    vc_enc = le_vc.transform([vehicle_class])[0]
    tr_enc = le_tr.transform([transmission])[0]
    ft_enc = le_ft.transform([fuel_type])[0]

    user_data = pd.DataFrame(
        [[engine_size, cylinders, vc_enc, tr_enc, ft_enc]],
        columns=FEATURES
    )

    predicted_co2  = co2_model.predict(user_data)[0]
    predicted_fuel = fuel_model.predict(user_data)[0]
    avg_co2        = df["CO2 Emissions(g/km)"].mean()
    avg_fuel       = df["Fuel Consumption Comb (L/100 km)"].mean()

    # ── RESULTS ──
    st.markdown("### 📊 Prediction Results")

    co2_class  = "good" if predicted_co2 < avg_co2 * 0.85 else ("warn" if predicted_co2 < avg_co2 * 1.15 else "bad")
    fuel_class = "good" if predicted_fuel < avg_fuel * 0.85 else ("warn" if predicted_fuel < avg_fuel * 1.15 else "bad")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
            <div class="result-card">
                <h3>🌿 CO2 Emissions</h3>
                <div class="result-value {co2_class}">{predicted_co2:.1f} <span style="font-size:1rem">g/km</span></div>
                <div style="color:#8b949e; font-size:0.85rem; margin-top:0.4rem">Dataset avg: {avg_co2:.1f} g/km</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="result-card">
                <h3>⛽ Fuel Consumption</h3>
                <div class="result-value {fuel_class}">{predicted_fuel:.1f} <span style="font-size:1rem">L/100km</span></div>
                <div style="color:#8b949e; font-size:0.85rem; margin-top:0.4rem">Dataset avg: {avg_fuel:.1f} L/100km</div>
            </div>
        """, unsafe_allow_html=True)

    # ── GRAPHS ──
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📈 Visual Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#58a6ff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Graph 1: CO2
    co2_vals   = [avg_co2, predicted_co2]
    co2_colors = ["#1f6feb", "#3fb950" if predicted_co2 <= avg_co2 else "#f85149"]
    bars1 = axes[0].bar(["Dataset Avg", "Your Vehicle"], co2_vals, color=co2_colors, edgecolor="#30363d", width=0.45)
    axes[0].set_title("🌿 CO2 Emissions (g/km)", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_ylabel("CO2 (g/km)", color="#8b949e")
    axes[0].set_ylim(0, max(co2_vals) * 1.3)
    axes[0].axhline(avg_co2, color="#58a6ff", linestyle="--", linewidth=1, alpha=0.5, label="Average")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
    for bar, val in zip(bars1, co2_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", color="white", fontweight="bold", fontsize=11)

    # Graph 2: Fuel
    fuel_vals   = [avg_fuel, predicted_fuel]
    fuel_colors = ["#1f6feb", "#3fb950" if predicted_fuel <= avg_fuel else "#f85149"]
    bars2 = axes[1].bar(["Dataset Avg", "Your Vehicle"], fuel_vals, color=fuel_colors, edgecolor="#30363d", width=0.45)
    axes[1].set_title("⛽ Fuel Consumption (L/100km)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylabel("L/100 km", color="#8b949e")
    axes[1].set_ylim(0, max(fuel_vals) * 1.3)
    axes[1].axhline(avg_fuel, color="#58a6ff", linestyle="--", linewidth=1, alpha=0.5, label="Average")
    axes[1].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
    for bar, val in zip(bars2, fuel_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", color="white", fontweight="bold", fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)

    # ── SUGGESTIONS ──
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 💡 Personalized Suggestions")

    # CO2 suggestion
    if predicted_co2 < avg_co2 * 0.85:
        st.markdown('<div class="suggestion-box good">✅ <b>Excellent!</b> Your CO2 emission is well below average. Your vehicle is eco-friendly — keep it up! 🌱</div>', unsafe_allow_html=True)
    elif predicted_co2 < avg_co2:
        st.markdown('<div class="suggestion-box good">🟡 <b>Good.</b> Your CO2 emission is slightly below average. Regular servicing will help maintain this level.</div>', unsafe_allow_html=True)
    elif predicted_co2 < avg_co2 * 1.15:
        st.markdown('<div class="suggestion-box warn">🟠 <b>Above Average CO2.</b> Maintain proper tyre pressure and avoid unnecessary idling to reduce emissions.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="suggestion-box bad">🔴 <b>High CO2 Detected!</b> Consider switching to a hybrid or EV. Regular engine tune-ups and avoiding aggressive acceleration can help significantly.</div>', unsafe_allow_html=True)

    # Fuel suggestion
    if predicted_fuel < avg_fuel * 0.85:
        st.markdown('<div class="suggestion-box good">✅ <b>Great fuel efficiency!</b> You\'re saving money and the planet. 💚</div>', unsafe_allow_html=True)
    elif predicted_fuel < avg_fuel:
        st.markdown('<div class="suggestion-box good">🟡 <b>Below Average Consumption.</b> Maintain steady highway speeds for even better mileage.</div>', unsafe_allow_html=True)
    elif predicted_fuel < avg_fuel * 1.15:
        st.markdown('<div class="suggestion-box warn">🟠 <b>Slightly High Consumption.</b> Try carpooling, combining trips, and checking air filters to improve efficiency.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="suggestion-box bad">🔴 <b>High Fuel Consumption!</b> Consider a smaller engine or more fuel-efficient model. Use cruise control on highways and avoid carrying unnecessary weight.</div>', unsafe_allow_html=True)

    # Fuel type tip
    fuel_tips = {
        "D": ("info", "🛢️ <b>Diesel Tip:</b> Great for long highway drives. Avoid frequent short city trips as diesel engines need to warm up properly."),
        "E": ("info", "🌿 <b>Ethanol (E85) Tip:</b> Eco-friendly but gives ~25% lower mileage than gasoline. Ensure your vehicle is flex-fuel compatible."),
        "N": ("info", "🌿 <b>Natural Gas Tip:</b> CNG is the cleanest burning fuel — excellent eco choice! Ensure CNG stations are available on your regular routes."),
        "X": ("info", "⛽ <b>Regular Gasoline Tip:</b> Cost-effective choice. Do NOT use premium if your engine only requires regular grade."),
        "Z": ("info", "⛽ <b>Premium Gasoline Tip:</b> Needed for high-compression engines. Using regular grade in a premium engine can cause knocking and reduce performance.")
    }
    tip_class, tip_text = fuel_tips.get(fuel_type, ("info", "⛽ Maintain your fuel system regularly for best performance."))
    st.markdown(f'<div class="suggestion-box {tip_class}">{tip_text}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("✅ Prediction complete! Scroll up to see your results.")