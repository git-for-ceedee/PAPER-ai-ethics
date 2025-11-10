# -------------------
# Pick-a-Path AI Ethics Analytics Dashboard
# Visualise user metrics and log data
# -------------------

import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import timedelta

# --- Helper functions ---
def trend_arrow(curr, prev, higher_is_better=True):
    if prev is None or pd.isna(prev):
        return "↔︎"
    if higher_is_better:
        return "↑" if curr > prev else ("↓" if curr < prev else "↔︎")
    else:
        return "↓" if curr < prev else ("↑" if curr > prev else "↔︎")

def colourise(value, good_thresh=None, warn_thresh=None, invert=False):
    # Traffic-light colouring for numeric metrics.
    if value is None or pd.isna(value):
        return f"<span style='color:#999'>n/a</span>"
    v = float(value)
    if good_thresh is None and warn_thresh is None:
        good_thresh, warn_thresh = (0.33, 0.66)
    if invert:
        good = v <= good_thresh
        warn = good_thresh < v <= warn_thresh
    else:
        good = v >= good_thresh
        warn = warn_thresh <= v < good_thresh
    if good:
        colour = "#0a7f2e"
    elif warn:
        colour = "#b07000"
    else:
        colour = "#b00020"
    return f"<span style='color:{colour}'><b>{v:.1f}</b></span>"

# --- Setup ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_DIR / "user_metrics_log.csv"

st.set_page_config(page_title="PAPE Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Pick-a-Path AI Ethics Analytics Dashboard")

# --- Load and clean data ---
cols = [
    "timestamp",
    "session_id",
    "event_type",
    "story_key",
    "user_industry",
    "user_role",
    "user_topic",
    "user_query",
    "level",
    "latency_ms",
    "flesch_ease",
    "fk_grade",
    "repeat_sim",
    "choices_count",
    "details",
]

if not LOG_FILE.exists():
    st.warning("⚠️ No log file found yet — run the main app to generate data.")
    st.stop()

df = pd.read_csv(LOG_FILE, header=None, names=cols, encoding="utf-8")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp")

# Convert numeric fields
num_cols = ["latency_ms", "flesch_ease", "fk_grade", "repeat_sim", "choices_count", "level"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Filters ---
with st.expander("Filters"):
    f_role = st.selectbox("Role", ["(all)"] + sorted(df["user_role"].dropna().unique().tolist()))
    f_ind  = st.selectbox("Industry", ["(all)"] + sorted(df["user_industry"].dropna().unique().tolist()))
    f_top  = st.selectbox("Topic", ["(all)"] + sorted(df["user_topic"].dropna().unique().tolist()))

def apply_filters(d):
    if f_role != "(all)":
        d = d[d["user_role"] == f_role]
    if f_ind != "(all)":
        d = d[d["user_industry"] == f_ind]
    if f_top != "(all)":
        d = d[d["user_topic"] == f_top]
    return d

df = apply_filters(df)

# --- Define analysis windows (7-day current vs previous) ---
if not df["timestamp"].isna().all():
    now = df["timestamp"].max()
    curr_start = now - timedelta(days=7)
    prev_start = now - timedelta(days=14)

    df_curr = df[df["timestamp"] >= curr_start]
    df_prev = df[(df["timestamp"] >= prev_start) & (df["timestamp"] < curr_start)]
else:
    df_curr = df.copy()
    df_prev = pd.DataFrame(columns=df.columns)

# --- Summary Metrics ---
st.header("Summary Metrics")

total_gens_curr = len(df_curr[df_curr["event_type"].isin(["story_started", "story_step"])])
failures_curr = len(df_curr[df_curr["event_type"] == "generation_failure"])
failure_rate_curr = (failures_curr / total_gens_curr * 100) if total_gens_curr else 0

avg_latency_curr = df_curr["latency_ms"].mean()
avg_flesch_curr = df_curr["flesch_ease"].mean()
avg_repeat_curr = df_curr["repeat_sim"].mean()

# Compare with previous window
total_gens_prev = len(df_prev[df_prev["event_type"].isin(["story_started", "story_step"])])
failures_prev = len(df_prev[df_prev["event_type"] == "generation_failure"])
failure_rate_prev = (failures_prev / total_gens_prev * 100) if total_gens_prev else None

avg_latency_prev = df_prev["latency_ms"].mean() if not df_prev.empty else None
avg_flesch_prev = df_prev["flesch_ease"].mean() if not df_prev.empty else None
avg_repeat_prev = df_prev["repeat_sim"].mean() if not df_prev.empty else None

# Arrows (directional)
arrow_fail = trend_arrow(failure_rate_curr, failure_rate_prev, higher_is_better=False)
arrow_lat  = trend_arrow(avg_latency_curr, avg_latency_prev, higher_is_better=False)
arrow_read = trend_arrow(avg_flesch_curr, avg_flesch_prev, higher_is_better=True)
arrow_rep  = trend_arrow(avg_repeat_curr, avg_repeat_prev, higher_is_better=False)

# Colours
fail_col = colourise(failure_rate_curr, good_thresh=5, warn_thresh=10, invert=True)
lat_col  = colourise(avg_latency_curr, good_thresh=1200, warn_thresh=2500, invert=True)
read_col = colourise(avg_flesch_curr, good_thresh=60, warn_thresh=40)
rep_col  = colourise(avg_repeat_curr, good_thresh=0.4, warn_thresh=0.7, invert=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"**❌ Failure Rate**<br>{fail_col}% {arrow_fail}", unsafe_allow_html=True)
c2.markdown(f"**⏱ Avg Latency**<br>{lat_col} ms {arrow_lat}", unsafe_allow_html=True)
c3.markdown(f"**🧠 Avg Readability (FRE)**<br>{read_col} {arrow_read}", unsafe_allow_html=True)
c4.markdown(f"**🔁 Repeat Similarity**<br>{rep_col} {arrow_rep}", unsafe_allow_html=True)

# --- Charts ---
st.subheader("Theme (Topic) Distribution")
topic_counts = df[df["event_type"] == "story_started"]["user_topic"].value_counts()
st.bar_chart(topic_counts)

st.subheader("User Journey Depth (Levels Reached)")
depth_df = df.groupby("story_key")["level"].max().reset_index(name="max_level")
st.line_chart(depth_df["max_level"])

st.subheader("Latency Over Time (ms)")
st.line_chart(df_curr.set_index("timestamp")["latency_ms"])

st.subheader("Reading Ease (FRE) by Scene")
st.bar_chart(df_curr["flesch_ease"].dropna())

st.subheader("Repeat Similarity Between Scenes")
st.bar_chart(df_curr["repeat_sim"].dropna())

# --- Event breakdown ---
st.subheader("Event Type Counts")
event_counts = df["event_type"].value_counts()
st.dataframe(event_counts)

# --- Raw data + download ---
with st.expander("🔍 Raw Log Data"):
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data", data=csv, file_name="PAPE_filtered_log.csv", mime="text/csv")


