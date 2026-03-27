import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import os
import requests
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO2 & Fuel Predictor",
    page_icon="🚗",
    layout="wide"
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
h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #58a6ff; }

.title-block { text-align: center; padding: 2rem 1rem 1rem; }
.title-block h1 { font-size: 2rem; letter-spacing: 2px; color: #58a6ff; text-shadow: 0 0 20px #58a6ff55; }
.title-block p  { color: #8b949e; font-size: 1rem; margin-top: 0.3rem; }

.result-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(88,166,255,0.08);
}
.result-card h3 { font-size: 0.9rem; color: #8b949e; margin-bottom: 0.3rem; font-family: 'Exo 2',sans-serif; text-transform: uppercase; letter-spacing: 1px; }
.result-value   { font-size: 2.2rem; font-weight: 700; font-family: 'Orbitron',sans-serif; }
.good  { color: #3fb950; }
.warn  { color: #d29922; }
.bad   { color: #f85149; }

.suggestion-box { background: #161b22; border-left: 4px solid #58a6ff; border-radius: 8px; padding: 1rem 1.5rem; margin: 0.5rem 0; font-size: 0.95rem; color: #c9d1d9; }
.suggestion-box.good { border-left-color: #3fb950; }
.suggestion-box.warn { border-left-color: #d29922; }
.suggestion-box.bad  { border-left-color: #f85149; }
.suggestion-box.info { border-left-color: #58a6ff; }

.info-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 1.5rem; margin: 0.5rem 0; }
.metric-big { font-size: 2.5rem; font-weight: 700; font-family: 'Orbitron',sans-serif; color: #58a6ff; }

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white; border: none; border-radius: 8px;
    padding: 0.7rem 2rem; font-family: 'Orbitron',sans-serif;
    font-size: 0.85rem; letter-spacing: 1px; width: 100%;
    box-shadow: 0 4px 15px rgba(31,111,235,0.3);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(31,111,235,0.5); }
.divider { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("co2_emissions.csv")
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    return df

bundle     = load_model()
co2_model  = bundle["co2_model"]
fuel_model = bundle["fuel_model"]
le_vc      = bundle["le_vc"]
le_tr      = bundle["le_tr"]
le_ft      = bundle["le_ft"]
FEATURES   = bundle["features"]
avg_co2    = bundle["avg_co2"]
avg_fuel   = bundle["avg_fuel"]
df         = load_data()

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
FUEL_TYPE_MAP = {
    "D": "D — Diesel",
    "E": "E — Ethanol (E85)",
    "N": "N — Natural Gas (CNG)",
    "X": "X — Regular Gasoline (87 octane)",
    "Z": "Z — Premium Gasoline (91+ octane)"
}

TRANSMISSION_PREFIX = {
    "AM": "Automated Manual",
    "AS": "Automatic with Select Shift",
    "AV": "Continuously Variable (CVT)",
    "A":  "Automatic",
    "M":  "Manual"
}

FUEL_PRICES  = {"D": 94, "E": 65, "N": 80, "X": 106, "Z": 112}
HISTORY_FILE = "prediction_history.csv"

def trans_label(code):
    for key in sorted(TRANSMISSION_PREFIX.keys(), key=len, reverse=True):
        if code.startswith(key):
            return f"{code} — {TRANSMISSION_PREFIX[key]}"
    return code

# ─────────────────────────────────────────────────────────
# CSV HISTORY FUNCTIONS
# ─────────────────────────────────────────────────────────
def save_to_history(vehicle_class, transmission, fuel_type, engine_size, cylinders, co2, fuel):
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp","Vehicle Class","Transmission","Fuel Type",
                             "Engine Size(L)","Cylinders","CO2 (g/km)","Fuel (L/100km)"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            vehicle_class, transmission, FUEL_TYPE_MAP.get(fuel_type, fuel_type),
            engine_size, cylinders, round(co2, 2), round(fuel, 2)
        ])

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()

# ─────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚗 CO2 Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home — Predict",
    "📏 Trip Distance Calculator",
    "⛽ Nearby Fuel Stations",
    "📋 Prediction History"
])
st.sidebar.markdown("---")
st.sidebar.markdown("<small style='color:#8b949e'>Built with Streamlit & Random Forest ML</small>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 1 — HOME PREDICT
# ═══════════════════════════════════════════════════════════
if page == "🏠 Home — Predict":

    st.markdown("""
    <div class="title-block">
        <h1>🚗 CO2 & FUEL PREDICTOR</h1>
        <p>Enter your vehicle details to predict CO2 emissions & fuel consumption</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🔧 Vehicle Details")

    col1, col2 = st.columns(2)
    with col1:
        vehicle_options = sorted(df["Vehicle Class"].unique())
        vehicle_class   = st.selectbox("🚗 Vehicle Class", vehicle_options)

        trans_options = sorted(df["Transmission"].unique())
        trans_idx     = st.selectbox("⚙️ Transmission", range(len(trans_options)),
                                     format_func=lambda i: trans_label(trans_options[i]))
        transmission  = trans_options[trans_idx]

    with col2:
        fuel_options = sorted(df["Fuel Type"].unique())
        fuel_idx     = st.selectbox("⛽ Fuel Type", range(len(fuel_options)),
                                    format_func=lambda i: FUEL_TYPE_MAP.get(fuel_options[i], fuel_options[i]))
        fuel_type    = fuel_options[fuel_idx]

        engine_size = st.slider("🔩 Engine Size (L)",
                                float(round(df["Engine Size(L)"].min(), 1)),
                                float(round(df["Engine Size(L)"].max(), 1)),
                                2.0, 0.1)

    cylinders = st.select_slider("🔢 Number of Cylinders",
                                 options=sorted(df["Cylinders"].unique().astype(int).tolist()))

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if st.button("🚀 PREDICT NOW"):
        vc_enc    = le_vc.transform([vehicle_class])[0]
        tr_enc    = le_tr.transform([transmission])[0]
        ft_enc    = le_ft.transform([fuel_type])[0]
        user_data = pd.DataFrame([[engine_size, cylinders, vc_enc, tr_enc, ft_enc]], columns=FEATURES)

        pred_co2  = co2_model.predict(user_data)[0]
        pred_fuel = fuel_model.predict(user_data)[0]

        save_to_history(vehicle_class, transmission, fuel_type, engine_size, cylinders, pred_co2, pred_fuel)

        co2_cls  = "good" if pred_co2  < avg_co2  * 0.85 else ("warn" if pred_co2  < avg_co2  * 1.15 else "bad")
        fuel_cls = "good" if pred_fuel < avg_fuel * 0.85 else ("warn" if pred_fuel < avg_fuel * 1.15 else "bad")

        st.markdown("### 📊 Prediction Results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="result-card">
                <h3>🌿 CO2 Emissions</h3>
                <div class="result-value {co2_cls}">{pred_co2:.1f} <span style="font-size:1rem">g/km</span></div>
                <div style="color:#8b949e;font-size:0.85rem;margin-top:0.4rem">Dataset avg: {avg_co2:.1f} g/km</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="result-card">
                <h3>⛽ Fuel Consumption</h3>
                <div class="result-value {fuel_cls}">{pred_fuel:.1f} <span style="font-size:1rem">L/100km</span></div>
                <div style="color:#8b949e;font-size:0.85rem;margin-top:0.4rem">Dataset avg: {avg_fuel:.1f} L/100km</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📈 Visual Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values(): spine.set_edgecolor("#30363d")

        for ax, vals, title, unit, avg in [
            (axes[0], [avg_co2, pred_co2],  "🌿 CO2 Emissions (g/km)",      "CO2 (g/km)",  avg_co2),
            (axes[1], [avg_fuel, pred_fuel], "⛽ Fuel Consumption (L/100km)", "L/100 km",    avg_fuel)
        ]:
            colors = ["#1f6feb", "#3fb950" if vals[1] <= avg else "#f85149"]
            bars   = ax.bar(["Dataset Avg", "Your Vehicle"], vals, color=colors, edgecolor="#30363d", width=0.45)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#58a6ff")
            ax.set_ylabel(unit, color="#8b949e")
            ax.set_ylim(0, max(vals) * 1.3)
            ax.axhline(avg, color="#58a6ff", linestyle="--", linewidth=1, alpha=0.5, label="Average")
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                        f"{val:.1f}", ha="center", color="white", fontweight="bold", fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 💡 Personalized Suggestions")

        if pred_co2 < avg_co2 * 0.85:
            st.markdown('<div class="suggestion-box good">✅ <b>Excellent!</b> Your CO2 is well below average. Eco-friendly vehicle! 🌱</div>', unsafe_allow_html=True)
        elif pred_co2 < avg_co2:
            st.markdown('<div class="suggestion-box good">🟡 <b>Good.</b> CO2 below average. Regular servicing maintains this.</div>', unsafe_allow_html=True)
        elif pred_co2 < avg_co2 * 1.15:
            st.markdown('<div class="suggestion-box warn">🟠 <b>Above Average CO2.</b> Maintain proper tyre pressure and avoid idling.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="suggestion-box bad">🔴 <b>High CO2!</b> Consider hybrid/EV. Regular tune-ups reduce emissions significantly.</div>', unsafe_allow_html=True)

        if pred_fuel < avg_fuel * 0.85:
            st.markdown('<div class="suggestion-box good">✅ <b>Great fuel efficiency!</b> Saving money & the planet. 💚</div>', unsafe_allow_html=True)
        elif pred_fuel < avg_fuel:
            st.markdown('<div class="suggestion-box good">🟡 <b>Below average consumption.</b> Maintain steady highway speeds for even better mileage.</div>', unsafe_allow_html=True)
        elif pred_fuel < avg_fuel * 1.15:
            st.markdown('<div class="suggestion-box warn">🟠 <b>Slightly high consumption.</b> Check air filters and try combining trips.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="suggestion-box bad">🔴 <b>High fuel consumption!</b> Consider smaller engine. Use cruise control on highways.</div>', unsafe_allow_html=True)

        fuel_tips = {
            "D": ("info", "🛢️ <b>Diesel:</b> Great for long highway drives. Avoid frequent short city trips."),
            "E": ("info", "🌿 <b>Ethanol (E85):</b> Eco-friendly but ~25% lower mileage. Ensure flex-fuel compatibility."),
            "N": ("info", "🌿 <b>Natural Gas:</b> Cleanest burning fuel! Check CNG station availability on your routes."),
            "X": ("info", "⛽ <b>Regular Gasoline:</b> Cost-effective. Don't use premium if engine only needs regular."),
            "Z": ("info", "⛽ <b>Premium Gasoline:</b> Needed for high-compression engines. Using regular causes knocking.")
        }
        tip_cls, tip_txt = fuel_tips.get(fuel_type, ("info", "⛽ Maintain fuel system regularly."))
        st.markdown(f'<div class="suggestion-box {tip_cls}">{tip_txt}</div>', unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 🛣️ Quick Trip Estimate")
        fuel_price = FUEL_PRICES.get(fuel_type, 100)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="info-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">100 km trip costs</div>
                <div class="metric-big">₹{(pred_fuel * fuel_price):.0f}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="info-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">1 litre takes you</div>
                <div class="metric-big">{(100/pred_fuel):.1f}<span style="font-size:1rem"> km</span></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="info-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Monthly cost (1500km)</div>
                <div class="metric-big">₹{(pred_fuel * fuel_price * 15):.0f}</div>
            </div>""", unsafe_allow_html=True)

        st.success("✅ Prediction saved to history! Go to 📋 Prediction History tab to view.")


# ═══════════════════════════════════════════════════════════
# PAGE 2 — TRIP DISTANCE CALCULATOR
# ═══════════════════════════════════════════════════════════
elif page == "📏 Trip Distance Calculator":

    st.markdown("""
    <div class="title-block">
        <h1>📏 TRIP DISTANCE CALCULATOR</h1>
        <p>Find out how far your fuel will take you!</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⛽ Fuel Details")

        fuel_in_tank_str = st.text_input("How many litres of fuel do you have?",
                                          value="40", placeholder="e.g. 40")
        try:
            fuel_in_tank = float(fuel_in_tank_str)
        except:
            fuel_in_tank = 40.0
            st.warning("⚠️ Enter a valid number for fuel in tank.")

        fuel_consumption_str = st.text_input("Your vehicle's fuel consumption (L/100km)",
                                              value=str(round(avg_fuel, 1)),
                                              placeholder="e.g. 10.5",
                                              help="Use predicted value from Home page or enter manually")
        try:
            fuel_consumption = float(fuel_consumption_str)
        except:
            fuel_consumption = avg_fuel
            st.warning("⚠️ Enter a valid number for fuel consumption.")

        fuel_type_trip = st.selectbox("Fuel Type", list(FUEL_TYPE_MAP.keys()),
                                      format_func=lambda x: FUEL_TYPE_MAP[x])

        fuel_price_str = st.text_input("Fuel price per litre (₹)",
                                        value=str(FUEL_PRICES.get(fuel_type_trip, 100)),
                                        placeholder="e.g. 94")
        try:
            fuel_price_inp = float(fuel_price_str)
        except:
            fuel_price_inp = 100.0
            st.warning("⚠️ Enter a valid fuel price.")

    with col2:
        st.markdown("### 🛣️ Trip Details")

        trip_distance_str = st.text_input("Enter trip distance (km) to calculate fuel needed",
                                           value="0", placeholder="e.g. 150")
        try:
            trip_distance = float(trip_distance_str)
        except:
            trip_distance = 0.0
            st.warning("⚠️ Enter a valid trip distance.")

        driving_style = st.selectbox("Driving Style", ["City driving", "Highway driving", "Mixed"])
        style_factor  = {"City driving": 1.15, "Highway driving": 0.90, "Mixed": 1.0}[driving_style]

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if st.button("📏 CALCULATE"):
        adj_consumption = fuel_consumption * style_factor

        st.markdown("### 📊 Results")

        if fuel_in_tank > 0:
            distance_possible = (fuel_in_tank / adj_consumption) * 100
            cost_of_trip      = fuel_in_tank * fuel_price_inp

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="info-card">
                    <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">You can travel</div>
                    <div class="metric-big">{distance_possible:.0f}<span style="font-size:1rem"> km</span></div>
                    <div style="color:#8b949e;font-size:0.8rem">with {fuel_in_tank:.0f}L of fuel</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="info-card">
                    <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Fuel cost</div>
                    <div class="metric-big">₹{cost_of_trip:.0f}</div>
                    <div style="color:#8b949e;font-size:0.8rem">for full tank</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="info-card">
                    <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Cost per km</div>
                    <div class="metric-big">₹{(adj_consumption * fuel_price_inp / 100):.1f}</div>
                    <div style="color:#8b949e;font-size:0.8rem">per kilometer</div>
                </div>""", unsafe_allow_html=True)

        if trip_distance > 0:
            fuel_needed    = (trip_distance / 100) * adj_consumption
            trip_cost      = fuel_needed * fuel_price_inp
            refills_needed = max(0, fuel_needed - fuel_in_tank)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown(f"### 🗺️ For your {trip_distance:.0f} km trip:")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="info-card">
                    <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Fuel needed</div>
                    <div class="metric-big">{fuel_needed:.1f}<span style="font-size:1rem"> L</span></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="info-card">
                    <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Trip cost</div>
                    <div class="metric-big">₹{trip_cost:.0f}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                cls = "good" if refills_needed == 0 else "bad"
                msg = "No refill needed! ✅" if refills_needed == 0 else f"Need {refills_needed:.1f}L more"
                st.markdown(f"""<div class="info-card">
                    <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Refuel status</div>
                    <div class="result-value {cls}" style="font-size:1.2rem">{msg}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 🏙️ Where can you go with your fuel?")
        milestones = {
            "🏪 Local market": 5,
            "🏙️ Nearby city": 50,
            "✈️ Airport": 80,
            "🏔️ Weekend trip": 150,
            "🌊 Beach vacation": 300,
            "🗺️ State border": 500
        }
        dist_can_travel = (fuel_in_tank / adj_consumption) * 100
        cols = st.columns(3)
        for i, (place, dist) in enumerate(milestones.items()):
            reachable = dist_can_travel >= dist
            with cols[i % 3]:
                st.markdown(f"""<div class="info-card" style="text-align:center">
                    <div style="font-size:1.5rem">{place.split()[0]}</div>
                    <div style="font-size:0.85rem;color:#8b949e">{place.split(' ',1)[1]} ({dist} km)</div>
                    <div style="margin-top:0.5rem" class="{'good' if reachable else 'bad'}">
                        {'✅ Reachable' if reachable else '❌ Need refuel'}
                    </div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 3 — NEARBY FUEL STATIONS
# ═══════════════════════════════════════════════════════════
elif page == "⛽ Nearby Fuel Stations":

    st.markdown("""
    <div class="title-block">
        <h1>⛽ NEARBY FUEL STATIONS</h1>
        <p>Find fuel stations near your location</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ✅ FIXED — removed GPS option, only manual search
    st.markdown("### 📍 Enter Your Location")

    st.markdown("""
    <div class="suggestion-box info">
        ℹ️ <b>Note:</b> Since this app runs on Streamlit Cloud (remote servers),
        auto GPS detects the server location (USA) — not your real location!
        Please type your city name below for accurate results. 😊
    </div>""", unsafe_allow_html=True)

    city_input = st.text_input("🔍 Enter your city or area name",
                               placeholder="e.g. Kalaburagi, Karnataka  OR  Mumbai  OR  Bengaluru")

    lat, lon, city_name = None, None, None

    if st.button("🔍 SEARCH FUEL STATIONS"):
        if city_input.strip() == "":
            st.warning("⚠️ Please enter a city name first!")
        else:
            with st.spinner("Searching your location..."):
                try:
                    geo = requests.get(
                        f"https://nominatim.openstreetmap.org/search?q={city_input}&format=json&limit=1",
                        headers={"User-Agent": "CO2FuelPredictor/1.0"},
                        timeout=8
                    ).json()

                    if geo:
                        lat       = float(geo[0]["lat"])
                        lon       = float(geo[0]["lon"])
                        city_name = geo[0].get("display_name", city_input).split(",")[0]
                        st.session_state["lat"]  = lat
                        st.session_state["lon"]  = lon
                        st.session_state["city"] = city_name
                        st.success(f"✅ Found: {city_name} ({lat:.4f}, {lon:.4f})")
                    else:
                        st.error("❌ Location not found! Try: 'Kalaburagi' or 'Kalaburagi, Karnataka'")

                except Exception:
                    st.error("❌ Search failed. Check internet connection and try again.")

    # Load from session if already searched
    if "lat" in st.session_state and lat is None:
        lat       = st.session_state["lat"]
        lon       = st.session_state["lon"]
        city_name = st.session_state.get("city", "")

    # ── Map + Stations
    if lat and lon:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"### 🗺️ Fuel Stations near {city_name}")

        with st.spinner("Finding nearby fuel stations..."):
            try:
                overpass_url   = "http://overpass-api.de/api/interpreter"
                overpass_query = f"""
                [out:json];
                node["amenity"="fuel"](around:10000,{lat},{lon});
                out body;
                """
                response = requests.post(overpass_url, data=overpass_query, timeout=15)
                elements = response.json().get("elements", [])

                if elements:
                    st.success(f"✅ Found {len(elements)} fuel stations within 10 km!")

                    stations = []
                    for el in elements[:20]:
                        tags    = el.get("tags", {})
                        s_lat   = el.get("lat", lat)
                        s_lon   = el.get("lon", lon)
                        dist_km = round(((s_lat - lat)**2 + (s_lon - lon)**2)**0.5 * 111, 2)
                        stations.append({
                            "Name"    : tags.get("name", tags.get("brand", "Fuel Station")),
                            "Brand"   : tags.get("brand", "Unknown"),
                            "Distance": f"{dist_km} km",
                            "dist_val": dist_km,
                            "Lat"     : s_lat,
                            "Lon"     : s_lon,
                            "Address" : tags.get("addr:street", tags.get("addr:city", "—"))
                        })

                    # Sort by distance
                    stations = sorted(stations, key=lambda x: x["dist_val"])

                    # Table
                    station_df = pd.DataFrame(stations)
                    st.dataframe(
                        station_df[["Name", "Brand", "Distance", "Address"]],
                        use_container_width=True,
                        hide_index=True
                    )

                    # Map — Red = You, Green = Fuel Stations
                    map_data = pd.DataFrame({
                        "lat"  : [lat]  + [s["Lat"] for s in stations],
                        "lon"  : [lon]  + [s["Lon"] for s in stations],
                        "size" : [120]  + [50] * len(stations),
                        "color": [[255, 50, 50, 200]] + [[0, 210, 100, 200]] * len(stations)
                    })

                    st.map(
                        map_data,
                        latitude="lat",
                        longitude="lon",
                        size="size",
                        color="color",
                        zoom=12
                    )

                    # Legend
                    st.markdown("""
                    <div class="suggestion-box info" style="display:flex;gap:2rem;">
                        <span>🔴 <b>Red dot</b> = Your Location</span>
                        <span>🟢 <b>Green dots</b> = Fuel Stations</span>
                    </div>""", unsafe_allow_html=True)

                    # Nearest station
                    nearest = stations[0]
                    st.markdown(f"""
                    <div class="suggestion-box good">
                        ✅ <b>Nearest Station:</b> {nearest['Name']} — {nearest['Distance']} away<br>
                        📍 Address: {nearest['Address']}<br>
                        🧭 Brand: {nearest['Brand']}
                    </div>""", unsafe_allow_html=True)

                else:
                    st.warning("⚠️ No fuel stations found within 10 km. Try a nearby bigger city name.")

            except Exception:
                st.error("❌ Could not fetch fuel stations. Check internet and try again.")
                st.info("💡 Alternative: Search 'petrol bunk near me' on Google Maps!")

        # Fuel prices
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 💰 Current Approximate Fuel Prices (India)")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (ft, price) in zip([c1, c2, c3, c4, c5], FUEL_PRICES.items()):
            col.markdown(f"""<div class="info-card" style="text-align:center">
                <div style="font-size:0.75rem;color:#8b949e">{FUEL_TYPE_MAP[ft].split('—')[1].strip()}</div>
                <div class="metric-big" style="font-size:1.5rem">₹{price}</div>
                <div style="font-size:0.7rem;color:#8b949e">per litre</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════
elif page == "📋 Prediction History":

    st.markdown("""
    <div class="title-block">
        <h1>📋 PREDICTION HISTORY</h1>
        <p>All your saved vehicle predictions</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    history = load_history()

    if history.empty:
        st.info("📭 No predictions yet! Go to 🏠 Home and make your first prediction.")
    else:
        st.markdown("### 📊 Your Stats")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="info-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Total Predictions</div>
                <div class="metric-big">{len(history)}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="info-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Avg CO2</div>
                <div class="metric-big">{history['CO2 (g/km)'].mean():.1f}<span style="font-size:1rem"> g/km</span></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="info-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Avg Fuel</div>
                <div class="metric-big">{history['Fuel (L/100km)'].mean():.1f}<span style="font-size:1rem"> L/100km</span></div>
            </div>""", unsafe_allow_html=True)
        with c4:
            best = history.loc[history["CO2 (g/km)"].idxmin(), "Vehicle Class"]
            st.markdown(f"""<div class="info-card">
                <div style="color:#8b949e;font-size:0.8rem;text-transform:uppercase">Best Vehicle Class</div>
                <div style="font-size:1rem;font-weight:600;color:#3fb950;margin-top:0.5rem">{best}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📋 All Predictions")
        st.dataframe(history.sort_values("Timestamp", ascending=False),
                     use_container_width=True, hide_index=True)

        csv_data = history.to_csv(index=False)
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv_data,
            file_name="my_predictions.csv",
            mime="text/csv"
        )

        if len(history) > 1:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("### 📈 CO2 Trend Over Time")
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")
            ax.plot(range(len(history)), history["CO2 (g/km)"],
                    color="#58a6ff", linewidth=2, marker="o", markersize=6)
            ax.axhline(avg_co2, color="#f85149", linestyle="--", linewidth=1,
                       label=f"Dataset avg: {avg_co2:.1f}")
            ax.set_xlabel("Prediction #", color="#8b949e")
            ax.set_ylabel("CO2 (g/km)", color="#8b949e")
            ax.tick_params(colors="#8b949e")
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
            for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All History"):
            os.remove(HISTORY_FILE)
            st.success("✅ History cleared!")
            st.rerun()
