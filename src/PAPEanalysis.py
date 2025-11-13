# -------------------
# Pick-a-Path AI Ethics Analytics Dashboard
# Visualise user metrics and log data
# -------------------

import pandas as pd
import streamlit as st
import json
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path
from datetime import datetime, timedelta
import time

# --- Setup ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="PAPE Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Pick-a-Path AI Ethics Analytics Dashboard")

# Column order used for logs (must match PAPEui logging)
LOG_COLUMNS = [
    "timestamp","session_id","event_type","story_key",
    "user_industry","user_role","user_topic","user_query",
    "level","latency_ms","flesch_ease","fk_grade","repeat_sim","choices_count","details",
]

# Spreadsheet settings read from Streamlit secrets (fallback names)
SPREADSHEET_NAME = st.secrets.get("GSHEET_SPREADSHEET_NAME", "PAPE User Metrics")
WORKSHEET_NAME   = st.secrets.get("GSHEET_WORKSHEET_NAME", "logs")
GSCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# --- Auto-refresh settings ---
# Initialize session state
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = time.time()
if "refresh_paused" not in st.session_state:
    st.session_state.refresh_paused = False
if "last_data_load" not in st.session_state:
    st.session_state.last_data_load = datetime.now()

# Refresh controls in sidebar
with st.sidebar:
    st.header("⚙️ Refresh Settings")
    refresh_interval = st.slider(
        "Refresh interval (seconds)",
        min_value=30,
        max_value=600,
        value=30,
        step=30,
        help="How often the dashboard should auto-refresh data"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏸️ Pause" if not st.session_state.refresh_paused else "▶️ Resume"):
            st.session_state.refresh_paused = not st.session_state.refresh_paused
    with col2:
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh_time = time.time()
            st.session_state.last_data_load = datetime.now()
            st.rerun()

    st.caption(f"{'⏸️ Auto-refresh paused' if st.session_state.refresh_paused else '✅ Auto-refresh active'}")
    st.caption(f"Last updated: {st.session_state.last_data_load.strftime('%H:%M:%S')}")

    # Show countdown progress
    if not st.session_state.refresh_paused:
        elapsed = time.time() - st.session_state.last_refresh_time
        remaining = max(0, refresh_interval - elapsed)
        progress = min(1.0, elapsed / refresh_interval)
        st.progress(progress, text=f"Next refresh in {int(remaining)}s")

    # Close dashboard button
    st.divider()
    if st.button("❌ Close Dashboard", use_container_width=True):
        st.markdown("""
            <script>
                window.close();
            </script>
        """, unsafe_allow_html=True)
        st.success("✅ Dashboard session ended. You can now close this tab.")
        st.caption("If the tab didn't close automatically, please close it manually.")

# Auto-refresh logic
if not st.session_state.refresh_paused:
    elapsed_time = time.time() - st.session_state.last_refresh_time
    if elapsed_time >= refresh_interval:
        st.session_state.last_refresh_time = time.time()
        st.session_state.last_data_load = datetime.now()
        time.sleep(0.1)  # Brief pause to allow progress bar to complete
        st.rerun()

# --------------------------------
# Functions
def trend_arrow(curr, prev, higher_is_better=True):
    if prev is None or pd.isna(prev):
        return "↔︎"
    if higher_is_better:
        return "↑" if curr > prev else ("↓" if curr < prev else "↔︎")
    else:
        return "↓" if curr < prev else ("↑" if curr > prev else "↔︎")

def colourise(value, good_thresh=None, warn_thresh=None, invert=False, suffix=""):
    # Traffic-light colouring for numeric metrics.
    if value is None or pd.isna(value):
        return f'<span style="color:#999">n/a</span>'
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
    return f'<span style="color:{colour}"><b>{v:.1f}{suffix}</b></span>'

# -----------------------
@st.cache_resource
def get_gsheet_client():
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        st.error("Missing GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets.")
        st.stop()
    try:
        info = dict(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    except Exception as e:
        st.error(f"Malformed GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
        st.stop()
    creds = Credentials.from_service_account_info(info, scopes=GSCOPE)
    gc = gspread.authorize(creds)
    return gc

# -----------------------
@st.cache_resource
def open_worksheet(spreadsheet_name: str = SPREADSHEET_NAME, worksheet_name: str = WORKSHEET_NAME):
    gc = get_gsheet_client()
    try:
        sh = gc.open(spreadsheet_name)
    except gspread.SpreadsheetNotFound:
        # try to create (requires the service account to have permission)
        sh = gc.create(spreadsheet_name)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=20)
        # write header row
        ws.append_row(LOG_COLUMNS, value_input_option="USER_ENTERED")
    return ws

# -----------------------
def load_logs_from_sheet():
    """
    Return a pandas DataFrame loaded from the Google Sheet.
    Columns and types normalised to expected names.
    """
    try:
        ws = open_worksheet(SPREADSHEET_NAME, WORKSHEET_NAME)
        data = ws.get_all_values()
        if not data or len(data) <= 1:
            # empty or only header
            return pd.DataFrame(columns=LOG_COLUMNS)
        df = pd.DataFrame(data[1:], columns=data[0])
        # convert numeric fields safely
        for c in ["latency_ms", "flesch_ease", "fk_grade", "repeat_sim", "choices_count", "level"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Failed to load logs from Google Sheets: {e}")
        return pd.DataFrame(columns=LOG_COLUMNS)

# -----------------------
def apply_filters(d):
    if f_role != "(all)":
        d = d[d["user_role"] == f_role]
    if f_ind != "(all)":
        d = d[d["user_industry"] == f_ind]
    if f_top != "(all)":
        d = d[d["user_topic"] == f_top]
    return d

# -----------------------
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

# Load logs from Google Sheets (live) with loading indicator
with st.spinner("Loading data from Google Sheets..."):
    df = load_logs_from_sheet()
    if df.empty:
        st.warning("⚠️ No logs found in Google Sheets yet. Generate a few stories in the main app first.")
        st.stop()
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
fail_col = colourise(failure_rate_curr, good_thresh=5, warn_thresh=10, invert=True, suffix="%")
lat_col  = colourise(avg_latency_curr, good_thresh=1200, warn_thresh=2500, invert=True, suffix=" ms")
read_col = colourise(avg_flesch_curr, good_thresh=60, warn_thresh=40)
rep_col  = colourise(avg_repeat_curr, good_thresh=0.4, warn_thresh=0.7, invert=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"**❌ Failure Rate**<br>{fail_col} {arrow_fail}", unsafe_allow_html=True)
c2.markdown(f"**⏱ Avg Latency**<br>{lat_col} {arrow_lat}", unsafe_allow_html=True)
c3.markdown(f"**🧠 Avg Readability (FRE)**<br>{read_col} {arrow_read}", unsafe_allow_html=True)
c4.markdown(f"**🔁 Repeat Similarity**<br>{rep_col} {arrow_rep}", unsafe_allow_html=True)

# --- Charts ---
st.subheader("Theme (Topic) Distribution")
topic_counts = df[df["event_type"] == "story_started"]["user_topic"].value_counts()
st.bar_chart(topic_counts)

st.subheader("Latency Over Time (ms)")
latency_over_time = df[df["latency_ms"].notna()].set_index("timestamp")["latency_ms"]
st.line_chart(latency_over_time)

st.subheader("Reading Ease (FRE) Over Time")
fre_over_time = df[df["flesch_ease"].notna()].set_index("timestamp")["flesch_ease"]
st.line_chart(fre_over_time)
st.caption("FRE guide: 80–100 very easy • 60–80 easy/moderate • 30–60 hard • 0–30 very hard")

# turn values into 0-100 scale for better visualisation
st.subheader("Repeat Similarity Over Time")
rep_over_time = df[df["repeat_sim"].notna()].set_index("timestamp")["repeat_sim"]
rep_pct = (rep_over_time * 100).clip(0, 100)
st.line_chart(rep_pct)
st.caption("Repeat similarity: 0 = unique scenes, 100 = near-identical text. Lower is better.")

# --- Event breakdown ---
st.subheader("Event Type Counts")
event_counts = df["event_type"].value_counts()
pretty_labels = {
    "choice_made": "Number of click-throughs",
    "story_started": "Stories started",
    "story_completed": "Stories completed",
    "generation_failure": "Generation failures",
    "generation_error": "Generation errors",
}
ec_df = event_counts.rename(index=lambda k: pretty_labels.get(k, k))
st.dataframe(ec_df)

st.subheader("Deepest depth reached (per story)")
depth_df = df.groupby("story_key")["level"].max().reset_index(name="max_level")
st.dataframe(depth_df)
depth_distribution = depth_df["max_level"].value_counts().sort_index()
st.bar_chart(depth_distribution)
st.caption("X-axis: Maximum level reached | Y-axis: Number of stories")

# Also show an aggregate:
st.caption(f"Average deepest depth: {depth_df['max_level'].mean():.2f}")


# --- Raw data + download ---
with st.expander("🔍 Raw Log Data"):
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data", data=csv, file_name="PAPE_filtered_log.csv", mime="text/csv")


