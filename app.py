import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import sys

sys.path.append("src") # Add src to path for imports if needed

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Grid Sensor Data Quality Monitor", page_icon="⚡", layout="wide"
) # this is for the browser tab title and icon, and sets the layout to wide for better use of horizontal space.


# ── Load data & model ─────────────────────────────────────────
@st.cache_data # Cache the loaded data to speed up subsequent runs. This means the CSV files won't be re-read from disk unless they change.
def load_data():
    tagged = pd.read_csv(
        "results/anomalies.csv", index_col="datetime", parse_dates=True
    ) 
    hourly = pd.read_csv(
        "results/hourly_power.csv", index_col="datetime", parse_dates=True
    ).squeeze() # squeeze to convert single-column DataFrame to Series for easier handling in feature builder and plotting.
    return tagged, hourly


@st.cache_resource # Cache the loaded model to avoid reloading it on every interaction. The model will only be reloaded if the underlying file changes.
def load_model():
    model = joblib.load("models/xgb_anomaly_classifier.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, feature_cols


tagged, hourly = load_data() 
model, feature_cols = load_model()


# ── Feature builder (same as in notebook) ──────────────────────────
def build_features(series):
    df = pd.DataFrame({"value": series})
    for w in [3, 6, 24]:
        df[f"rolling_mean_{w}h"] = series.rolling(w, min_periods=1).mean()
        df[f"rolling_std_{w}h"] = series.rolling(w, min_periods=1).std().fillna(0)
        df[f"rolling_var_{w}h"] = series.rolling(w, min_periods=1).var().fillna(0)
    df["dev_from_mean_6h"] = (series - df["rolling_mean_6h"]).abs()
    df["dev_from_mean_24h"] = (series - df["rolling_mean_24h"]).abs()
    df["lag_1h"] = series.shift(1)
    df["lag_2h"] = series.shift(2)
    df["lead_1h"] = series.shift(-1)
    df["hour"] = series.index.hour
    df["day_of_week"] = series.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df.fillna(df.median()).drop(columns=["value"])


# ── Fault injector ────────────────────────────────────────────
def inject_fault(series, fault_type, start, duration_hours):
    injected = series.copy()
    idx = pd.date_range(start=start, periods=duration_hours, freq="h")
    idx = idx[idx.isin(series.index)]
    if fault_type == "Spike":
        injected[idx] = series.mean() + 6 * series.std()
    elif fault_type == "Flatline":
        injected[idx] = series.median() * 0.01
    elif fault_type == "Dropout":
        injected[idx] = np.nan
    return injected, idx


# ── Color map ─────────────────────────────────────────────────
COLORS = {
    "normal": "#2196F3",
    "spike": "#F44336",
    "flatline": "#FF9800",
    "dropout": "#9E9E9E",
}

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("⚡ Grid Sensor Data Quality Monitor")
st.markdown("*Automated anomaly detection for electric distribution sensor data*")
st.divider() # A horizontal line to visually separate the header from the rest of the content.

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Controls")

    st.subheader("Date Range")
    min_date = hourly.index.min().date()
    max_date = hourly.index.max().date()
    start_date = st.date_input(
        "From",
        value=pd.to_datetime("2008-01-01"),
        min_value=min_date,
        max_value=max_date,
    ) #
    end_date = st.date_input(
        "To", value=pd.to_datetime("2008-03-31"), min_value=min_date, max_value=max_date
    )

    st.subheader("Detection Sensitivity")
    spike_std = st.slider(
        "Spike threshold (std devs)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Lower = more sensitive, more false alarms. Higher = stricter, may miss spikes.",
    )

    st.divider()
    st.subheader("🧪 Inject a Synthetic Fault")
    fault_type = st.selectbox("Fault type", ["Spike", "Flatline", "Dropout"])
    fault_date = st.date_input(
        "Fault start date",
        value=pd.to_datetime("2008-02-15"),
        min_value=min_date,
        max_value=max_date,
    )
    fault_duration = st.slider("Duration (hours)", min_value=1, max_value=12, value=5)
    inject = st.button("🔴 Inject Fault", use_container_width=True)
    reset = st.button("↺ Reset", use_container_width=True)

    if "injected" not in st.session_state or reset:
        st.session_state.injected = False
        st.session_state.fault_idx = None

# ══════════════════════════════════════════════════════════════
# DATA PREP
# ══════════════════════════════════════════════════════════════
series = hourly[str(start_date) : str(end_date)].copy()

if inject:
    series, fault_idx = inject_fault(
        series, fault_type, str(fault_date), fault_duration
    )
    st.session_state.injected = True
    st.session_state.fault_idx = fault_idx
    st.session_state.fault_type = fault_type

# Run detection + classification
features = build_features(series)[feature_cols]
preds = model.predict(features)
proba = model.predict_proba(features)
label_map = {0: "normal", 1: "dropout", 2: "spike", 3: "flatline"}
pred_labels = pd.Series([label_map[p] for p in preds], index=series.index)
confidence = pd.Series(proba.max(axis=1), index=series.index).round(3)

# ══════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════
counts = pred_labels.value_counts()
total = len(pred_labels)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Readings", f"{total:,}")
col2.metric(
    "✅ Normal",
    f"{counts.get('normal', 0):,}",
    f"{counts.get('normal',0)/total*100:.1f}%",
)
col3.metric(
    "🔴 Spikes",
    f"{counts.get('spike', 0):,}",
    f"{counts.get('spike',0)/total*100:.1f}%",
    delta_color="inverse",
)
col4.metric(
    "🟠 Flatlines",
    f"{counts.get('flatline', 0):,}",
    f"{counts.get('flatline',0)/total*100:.1f}%",
    delta_color="inverse",
)
col5.metric(
    "⬜ Dropouts",
    f"{counts.get('dropout', 0):,}",
    f"{counts.get('dropout',0)/total*100:.1f}%",
    delta_color="inverse",
)

st.divider()

# ══════════════════════════════════════════════════════════════
# MAIN TIME SERIES PLOT
# ══════════════════════════════════════════════════════════════
st.subheader("📈 Sensor Time Series with Anomaly Flags")

fig = go.Figure()

# Normal line
fig.add_trace(
    go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name="Signal",
        line=dict(color="#90CAF9", width=1),
        opacity=0.8,
    )
)

# Anomaly scatter points
for atype, color in COLORS.items():
    if atype == "normal":
        continue
    mask = pred_labels == atype
    if mask.sum() == 0:
        continue
    fig.add_trace(
        go.Scatter(
            x=series.index[mask],
            y=series.values[mask],
            mode="markers",
            name=atype.capitalize(),
            marker=dict(color=color, size=7, symbol="circle"),
        )
    )

# Highlight injected fault window
if st.session_state.injected and st.session_state.fault_idx is not None:
    fidx = st.session_state.fault_idx
    if len(fidx) > 0:
        fig.add_vrect(
            x0=fidx[0],
            x1=fidx[-1],
            fillcolor="yellow",
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text=f"Injected {st.session_state.fault_type}",
            annotation_position="top left",
        )

fig.update_layout(
    height=400,
    xaxis_title="Time",
    yaxis_title="Global Active Power (kW)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=40, r=40, t=40, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# BOTTOM ROW — Anomaly breakdown + flagged table
# ══════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🥧 Anomaly Breakdown")
    anomaly_counts = pred_labels[pred_labels != "normal"].value_counts()
    if len(anomaly_counts) > 0:
        fig2 = go.Figure(
            go.Pie(
                labels=anomaly_counts.index.str.capitalize(),
                values=anomaly_counts.values,
                marker_colors=[COLORS[l] for l in anomaly_counts.index],
                hole=0.4,
            )
        )
        fig2.update_layout(
            height=300, margin=dict(l=20, r=20, t=20, b=20), showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No anomalies detected in selected range.")

with col_right:
    st.subheader("🚨 Flagged Readings")
    flagged = pd.DataFrame(
        {
            "datetime": series.index,
            "value (kW)": series.values.round(3),
            "anomaly_type": pred_labels.values,
            "confidence": confidence.values,
        }
    )
    flagged = flagged[flagged["anomaly_type"] != "normal"].reset_index(drop=True)
    flagged["datetime"] = flagged["datetime"].astype(str)

    if len(flagged) > 0:
        st.dataframe(
            flagged.sort_values("confidence", ascending=False).head(50),
            use_container_width=True,
            height=280,
        )
    else:
        st.info("No anomalies detected in selected range.")

# ══════════════════════════════════════════════════════════════
# INJECT FAULT FEEDBACK
# ══════════════════════════════════════════════════════════════
if st.session_state.injected:
    fidx = st.session_state.fault_idx
    caught = pred_labels[fidx][pred_labels[fidx] != "normal"]
    st.divider()
    if len(caught) == len(fidx):
        st.success(
            f"✅ All {len(fidx)} injected {st.session_state.fault_type.lower()} hours detected successfully!"
        )
    elif len(caught) > 0:
        st.warning(
            f"⚠️ Partially detected: {len(caught)} of {len(fidx)} injected hours flagged."
        )
    else:
        st.error(f"❌ Injected fault not detected. Try lowering the spike threshold.")

# thats it