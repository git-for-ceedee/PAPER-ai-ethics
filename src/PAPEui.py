# --------------------------------
# Pick-a-Path AI Ethics
# Create front end for the Pick-a-Path AI Ethics project
# --------------------------------

from pathlib import Path
import os, sqlite3, textwrap
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
import json
import requests
from datetime import datetime
import csv
import time
import uuid
import textstat
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
from PAPEsecurity import (
    validate_chunk_ids,
    validate_integer_range,
    validate_float_range,
    validate_string_length,
    detect_prompt_injection,
    sanitise_llm_input,
    validate_api_key
)
from PAPErate_limiter import RateLimiter, estimate_cost

# --------------------------------
# Settings and constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_DIR / "ragEthics.db"
INDEX_DIR = PROJECT_DIR / "vectorstore"
INDEX_PATH   = PROJECT_DIR / "vectorstore" / "chunks.faiss"
IDS_CSV      = PROJECT_DIR / "vectorstore" / "chunk_ids.csv"
# Column order used for logs 
LOG_COLUMNS = [
    "timestamp","session_id","event_type","story_key",
    "user_industry","user_role","user_topic","user_query",
    "level","latency_ms","flesch_ease","fk_grade","repeat_sim","choices_count","details",
]
# Spreadsheet settings read from Streamlit secrets (used for logging to gsheet)
SPREADSHEET_NAME = st.secrets.get("GSHEET_SPREADSHEET_NAME", "PAPE User Metrics")
WORKSHEET_NAME   = st.secrets.get("GSHEET_WORKSHEET_NAME", "logs")
GSCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# other tuning variables:
topic_boost = 0.15  # %boost to add to topics which are on-theme
per_doc_cap = 3  # Cap number of chunks per document to avoid repetition
max_levels = 3  # Maximum number of levels to generate

# --------------------------------
# Functions

def get_openai_client():
    # Initialise OpenAI client with security validation.
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    # SECURITY: Validate API key
    is_valid, error_msg = validate_api_key(api_key)
    if not is_valid:
        st.error(f"Invalid OpenAI API key: {error_msg}")
        st.error("Add a valid API key to .streamlit/secrets.toml or set OPENAI_API_KEY environment variable.")
        raise RuntimeError("Missing or invalid OPENAI_API_KEY")

    return OpenAI(api_key=api_key)

# --------------------------------
@st.cache_resource(show_spinner=False)
def get_rate_limiter():
    # Initialise rate limiter (cached across sessions).
    return RateLimiter(
        max_requests_per_minute=20,
        max_requests_per_hour=100,
        max_tokens_per_minute=40000,
        max_cost_per_day=10.0  # $10/day limit
    )

# --------------------------------
@st.cache_resource(show_spinner=False)
def get_gsheet_client():
    raw = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        st.error("Missing GOOGLE_SERVICE_ACCOUNT_JSON in Streamlit secrets.")
        st.stop()
    info = dict(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=GSCOPE)
    # st.caption(f"Service account in use: {creds.service_account_email}") - used for debugging accounts in google
    gc = gspread.authorize(creds)
    return gc

# --------------------------------
@st.cache_resource(show_spinner=False)
def open_worksheet(spreadsheet_name: str = SPREADSHEET_NAME, worksheet_name: str = WORKSHEET_NAME):
    gc = get_gsheet_client()
    try:
        sh = gc.open(spreadsheet_name)
    except gspread.SpreadsheetNotFound:
        # If the service account was given Editor rights, create will work
        sh = gc.create(spreadsheet_name)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=20)
        # Add header row if worksheet newly created
        ws.append_row(LOG_COLUMNS, value_input_option="USER_ENTERED")
    return ws

# --------------------------------
def load_logs_from_sheet():
    try:
        gc = get_gsheet_client()
        sh = gc.open(SPREADSHEET_NAME)
        ws = sh.worksheet(WORKSHEET_NAME)
        data = ws.get_all_values()
        if not data:
            return pd.DataFrame(columns=LOG_COLUMNS)
        df = pd.DataFrame(data[1:], columns=data[0]) if len(data) > 1 else pd.DataFrame(columns=data[0])
        # convert numeric types
        for c in ["latency_ms","flesch_ease","fk_grade","repeat_sim","choices_count","level"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Failed to load logs from Google Sheets: {e}")
        return pd.DataFrame(columns=LOG_COLUMNS)


# --------------------------------
def validate_required_files():
    # Validate that all required files and directories exist and contain data
    missing = []
    errors = []
    
    # Check database file
    if not DB_PATH.exists():
        missing.append(f"Database: {DB_PATH.name}")
        errors.append(f"Database file not found: `{DB_PATH}`\n   Run: `python -m src.PAPEsetup_database` to create it.")
    else:
        # Check if database has chunks
        try:
            conn = sqlite3.connect(DB_PATH)
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
            conn.close()
            if chunk_count == 0:
                missing.append("Database: No chunks")
                errors.append(f"Database exists but contains no chunks.\n   Run: `python -m src.PAPEdata_load` to ingest documents.")
        except Exception as e:
            missing.append("Database: Invalid or corrupted")
            errors.append(f"Database file exists but cannot be read: {str(e)}\n   You may need to recreate it.")
    
    # Check vectorstore directory
    if not INDEX_DIR.exists():
        missing.append(f"Directory: {INDEX_DIR.name}/")
        errors.append(f"Vectorstore directory not found: `{INDEX_DIR}`\n   This should be created when you build the FAISS index.")
    
    # Check FAISS index file
    if not INDEX_PATH.exists():
        missing.append(f"Index: {INDEX_PATH.name}")
        errors.append(f"FAISS index not found: `{INDEX_PATH}`\n   Run data ingestion to build the index.")
    else:
        # Check if index is not empty
        try:
            index = faiss.read_index(str(INDEX_PATH))
            if index.ntotal == 0:
                missing.append("Index: Empty")
                errors.append(f"FAISS index exists but is empty.\n   Run: `python -m src.PAPEdata_load` to rebuild the index.")
        except Exception as e:
            missing.append("Index: Corrupted")
            errors.append(f"FAISS index exists but is corrupted: {str(e)}\n   You may need to rebuild it.")
    
    # Check chunk IDs CSV
    if not IDS_CSV.exists():
        missing.append(f"IDs: {IDS_CSV.name}")
        errors.append(f"Chunk IDs file not found: `{IDS_CSV}`\n   This should be created when you build the FAISS index.")
    else:
        # Check if CSV has data
        try:
            ids_df = pd.read_csv(str(IDS_CSV))
            if ids_df.empty or "id" not in ids_df.columns:
                missing.append("IDs: Empty or invalid")
                errors.append(f"Chunk IDs file exists but is empty or invalid.\n   Run: `python -m src.PAPEdata_load` to rebuild it.")
        except Exception as e:
            missing.append("IDs: Corrupted")
            errors.append(f"Chunk IDs file exists but is corrupted: {str(e)}\n   You may need to rebuild it.")
    
    return missing, errors

# --------------------------------
def log_event_gsheet(
    event_type,
    story_key=None,
    user_industry=None,
    user_role=None,
    user_topic=None,
    user_query=None,
    details=None,
    level=None,
    latency_ms=None,
    flesch_ease=None,
    fk_grade=None,
    repeat_sim=None,
    choices_count=None,
    session_id=None,
):
    row = [
        pd.Timestamp.now().isoformat(timespec="seconds"),
        session_id or st.session_state.get("session_id"),
        event_type or "",
        story_key or "",
        user_industry or "",
        user_role or "",
        user_topic or "",
        user_query or "",
        "" if level is None else level,
        "" if latency_ms is None else latency_ms,
        "" if flesch_ease is None else flesch_ease,
        "" if fk_grade is None else fk_grade,
        "" if repeat_sim is None else repeat_sim,
        "" if choices_count is None else choices_count,
        details or "",
    ]
    try:
        ws = open_worksheet(SPREADSHEET_NAME, WORKSHEET_NAME)
        # If sheet is empty, write headers first
        vals = ws.get_all_values()
        if not vals:
            ws.append_row(LOG_COLUMNS, value_input_option="USER_ENTERED")
        ws.append_row(row, value_input_option="USER_ENTERED", table_range="A1")
    except Exception as e:
        # Fallback to local CSV if Sheets unavailable
        PROJECT_DIR = Path(__file__).resolve().parent.parent
        fallback = PROJECT_DIR / "user_metrics_log.csv"
        with open(fallback, "a", newline="", encoding="utf-8") as f:
            import csv
            csv.writer(f).writerow(row)
        # Optional: non-fatal warning
        st.warning(f"Google Sheets logging failed; wrote to CSV instead. ({e})")

# --------------------------------
def safe_readability(text: str):
    """Return (flesch_ease, fk_grade) or (None, None) if textstat missing."""
    if not text or not text.strip() or not textstat:
        return (None, None)
    try:
        return (
            float(textstat.flesch_reading_ease(text)),
            float(textstat.flesch_kincaid_grade(text)),
        )
    except Exception:
        return (None, None)

# --------------------------------
def cosine_repeat(prev_text: str, new_text: str):
    # Rough repeat score 0..1 using bag-of-words cosine; None if sklearn missing.
    if not prev_text or not new_text or not CountVectorizer or not cosine_similarity:
        return None
    try:
        vect = CountVectorizer(min_df=1, stop_words=None)
        X = vect.fit_transform([prev_text, new_text])
        sim = cosine_similarity(X)[0,1]
        return float(sim)
    except Exception:
        return None

# --------------------------------
@st.cache_resource(show_spinner=False) # streamlit caches output to avoid reloading the index every time
def load_index():
    # Load the FAISS index and chunk IDs. Assumes files have been validated.
    index = faiss.read_index(str(INDEX_PATH))
    ids = pd.read_csv(str(IDS_CSV))["id"].tolist()
    return index, ids

# --------------------------------
def fetch_chunks(chunk_ids):
    """
    Fetch chunks from database with security validation.

    Args:
        chunk_ids: List of chunk IDs to fetch

    Returns:
        DataFrame with chunk data
    """
    if not chunk_ids:
        return pd.DataFrame() # error handling if no chunk_ids

    # SECURITY: Validate chunk IDs to prevent SQL injection
    is_valid, error_msg = validate_chunk_ids(chunk_ids)
    if not is_valid:
        st.error(f"Security error: {error_msg}")
        return pd.DataFrame()

    # Use parameterised query to prevent SQL injection (?,?,?... etc)
    placeholders = ",".join("?" * len(chunk_ids))
    conn = sqlite3.connect(DB_PATH)
    rows = pd.read_sql_query(
        f"""
        SELECT c.id as chunk_id, c.text, d.title, d.origin, d.path_or_url, d.source_type
        FROM chunk c
        JOIN document d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        conn,
        params=tuple(chunk_ids))
    conn.close()
    # Preserve request order
    order = {cid:i for i,cid in enumerate(chunk_ids)}
    rows["__ord"] = rows["chunk_id"].map(order)
    return rows.sort_values("__ord")

# --------------------------------
@st.cache_resource(show_spinner=False) # cache the embedder to avoid reloading it every time
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2") # create a new transformer to generate sentence embeddings

# --------------------------------
def compose_prompt_with_profile(query, rows, user_industry, user_role, user_topic, temperature=0.5):
    # Compose the LLM prompt and include user profile so the story is tailored.
    context_blocks = "\n\n".join(
        f"[Source: {r['title']} | {r['origin'] or r['path_or_url']}]\n{r['text']}"
        for _, r in rows.iterrows()
    )
    profile = f"Industry: {user_industry}\nRole: {user_role}\nTopic: {user_topic}"
    story_prompt = f"""
You are generating an NZ workplace AI-ethics scenario in the pick-a-path format. Use the retrieved context below to ground facts and principles.

User profile:
  Industry: {user_industry}
  Role: {user_role}
  Topic: {user_topic}
  Query: {query}

Creativity level: {temperature:.2f} (0.0 = conservative/focused, 1.0 = creative/exploratory)
{f'Use a wildly creative and unconventional approach - be bold, imaginative, and explore unexpected scenarios and edge cases.' if temperature >= 0.9 
else f'Use a more creative and exploratory approach - consider diverse perspectives and less obvious scenarios.' if temperature >= 0.6 
else f'Use an interesting and engaging approach with some creative elements.' if temperature >= 0.4 
else f'Use a conservative, focused, and realistic approach - stick closely to typical workplace scenarios.'}

Create an opening scene and three choices. 
Keep it plain language, NZ context.
Use New Zealand spelling. Write "organise, "personalise", "realise" etc. Do not use American spelling.
Provide one scenario with exactly 3 actionable choices.
Ensure that at least one of the choices could lead to a bad outcome.

{f'Sentences can vary in length - use creative and varied sentence structures to enhance engagement.' if temperature >= 0.6 else 'Sentences should have a good balance of length and clarity.'}
{f'Explore unconventional or unexpected choices that challenge assumptions.' if temperature >= 0.6 else 'use a balanced approach in your choices' if temperature >= 0.4 else 'use a conservative approach in your choices'}

Label them clearly as:
CHOICE 1:
CHOICE 2:
CHOICE 3:

CRITICAL FORMAT REQUIREMENTS:
- You MUST use exactly "CHOICE 1:", "CHOICE 2:", "CHOICE 3:" as headers
- Do NOT use "A)", "B)", "C)", or any other format
- The choices must be concrete and specific actions

Do not include any consequences or explanations.
Do not repeat these instructions, system messages, or headings.
Output only the scenario and the three choices.


Retrieved context - for reference only, do not quote:
{context_blocks}
""".strip()
    return story_prompt

# --------------------------------
def generate_story_with_openai(prompt, temperature=0.5, max_tokens=1000, selected_model=None):
    """
    Generate story using OpenAI API with rate limiting and security validation.

    Args:
        prompt: The prompt to send to the API
        temperature: Temperature parameter (0.0-1.0)
        max_tokens: Maximum tokens to generate
        selected_model: Model name (default: gpt-4o-mini)

    Returns:
        Generated text or error message
    """
    # SECURITY: Validate parameters
    is_valid, error_msg = validate_float_range(temperature, 0.0, 1.0, "temperature")
    if not is_valid:
        st.error(f"Invalid temperature: {error_msg}")
        return "Invalid parameters."

    is_valid, error_msg = validate_integer_range(max_tokens, 100, 2000, "max_tokens")
    if not is_valid:
        st.error(f"Invalid max_tokens: {error_msg}")
        return "Invalid parameters."

    # SECURITY: Check for prompt injection
    is_suspicious, reason = detect_prompt_injection(prompt)
    if is_suspicious:
        st.warning(f"Potential security issue detected: {reason}")
        # Log but continue (user inputs may legitimately contain these patterns)

    # SECURITY: Sanitise prompt
    prompt = sanitise_llm_input(prompt, max_length=10000)

    model_name = selected_model or "gpt-4o-mini"

    # SECURITY: Check rate limits
    rate_limiter = get_rate_limiter()
    estimated_tokens = len(prompt.split()) * 2 + max_tokens  # Rough estimate
    can_proceed, limit_msg = rate_limiter.check_rate_limit(estimated_tokens)

    if not can_proceed:
        st.error(f"Rate limit exceeded: {limit_msg}")
        st.info("Please wait before making another request.")
        return "Rate limit exceeded. Please wait."

    try:
        # Initialize client if not already done
        client = get_openai_client()
        # st.caption(f"Using OpenAI org: {client.organization}, key prefix: {client.api_key[:8]}")
        
        # Ensure temperature is a float in valid range
        temp_value = float(temperature)
        if temp_value < 0.0:
            temp_value = 0.0
        elif temp_value > 2.0:  # OpenAI allows up to 2.0 for some models
            temp_value = 2.0
        
        # Debug: Log temperature being used
        import logging
        logging.info(f"OpenAI API call: temperature={temp_value:.2f}, max_tokens={max_tokens}, model={model_name}")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp_value,  # Use validated float value
            max_tokens=int(max_tokens)
        )

        # Record successful request for rate limiting (not too many requests to openAI)
        usage = response.usage
        total_tokens = usage.total_tokens if usage else estimated_tokens
        cost = estimate_cost(model_name, usage.prompt_tokens if usage else 0,
                           usage.completion_tokens if usage else 0)
        rate_limiter.record_request(total_tokens, cost)

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return "Could not generate scenario."


# --------------------------------
def build_query(user_industry: str, user_role: str, user_topic: str, user_query_free: str) -> str:
    
    # Turn sidebar selections into a retrieval query string.
    # Lightly map 'user_topic' to common keywords to help retrieval.
    # map topics to helpful keywords (expand as needed)
    topic_boosts = {
        "Accountability": ["accountability", "responsibility", "accountable", "accounting"],
        "Employment Impact": ["employment", "workforce", "displacement", "reskilling"],
        "Ethical Wellbeing": ["ethical wellbeing", "wellbeing", "hauora"],
        "Fairness": ["fairness", "bias", "discrimination", "equity"],
        "Human Oversight": ["human oversight", "human in the loop", "human review"],
        "Māori Data Sovereignty": ["Māori", "Maori", "data sovereignty", "CARE Principles", "Te Tiriti"],
        "Privacy": ["privacy", "consent", "IPP", "data minimisation", "de-identification"],
        "Security": ["security", "cyber", "encryption", "breach"],
        "Transparency": ["transparency", "explainability", "interpretability", "auditability"],
        "Other": [],
    }
    boosts = topic_boosts.get(user_topic, []) # find a list of keywords for the topic
    bits = [ # the user options they entered in the UI
        f"industry:{user_industry}",
        f"role:{user_role}",
        f"topic:{user_topic}",
    ]
    if user_query_free.strip():
        bits.append(user_query_free.strip())
    # fold in boosts to help retrieval 
    if boosts:
        bits.append(" ".join(boosts))
    # Final query is a friendly text blob; embedder will turn it into a vector
    return " | ".join(bits)

# --------------------------------
def get_chunk_ids_for_theme(theme_name: str) -> set[int]:
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.chunk_id
        FROM chunk_theme c
        JOIN theme t ON t.id = c.theme_id
        WHERE LOWER(t.name) = LOWER(?)
    """, (theme_name,))
    ids = {row[0] for row in cur.fetchall()}
    conn.close()
    return ids

# --------------------------------
def clean_for_choices(text: str) -> str:
    # Light clean so regex anchors behave well. normalise newlines
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # drop fenced code blocks (reduce false positives)
    t = re.sub(r"```.+?```", "", t, flags=re.DOTALL)
    # collapse multiple blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

# --------------------------------
def parse_scenario_choices(scenario_text: str):
    # Parse exactly three choices from model output, robust to a few formats.
    
    s = clean_for_choices(scenario_text)

    # ------- Primary: CHOICE N: (line-anchored, order-preserving) -------
    # Capture text after each header up to the next header or end of string.
    pattern_choiceN = re.compile(
        r"(?im)^\s*CHOICE\s+([1-3])\s*:\s*(.+?)(?=^\s*CHOICE\s+[1-3]\s*:|\Z)",
        flags=re.DOTALL
    )
    buckets = { "1": None, "2": None, "3": None }
    for m in pattern_choiceN.finditer(s):
        idx, body = m.group(1), m.group(2).strip()
        if idx in buckets and not buckets[idx]:
            buckets[idx] = body

    if all(buckets.values()):
        return [buckets["1"], buckets["2"], buckets["3"]]

    # ------- Fallback 1: A) / B) / C) (line-anchored only) -------
    pattern_abc = re.compile(
        r"(?ms)^[ \t]*A\)\s*(.+?)(?=^[ \t]*[BC]\)|\Z)"
        r"|^[ \t]*B\)\s*(.+?)(?=^[ \t]*C\)|\Z)"
        r"|^[ \t]*C\)\s*(.+?)(?=^\Z)",
    )
    # Find all and then reassemble in A,B,C order
    A = re.search(r"(?ms)^[ \t]*A\)\s*(.+?)(?=^[ \t]*[BC]\)|\Z)", s)
    B = re.search(r"(?ms)^[ \t]*B\)\s*(.+?)(?=^[ \t]*C\)|\Z)", s)
    C = re.search(r"(?ms)^[ \t]*C\)\s*(.+?)(?=^\Z)", s)
    if A and B and C:
        return [A.group(1).strip(), B.group(1).strip(), C.group(1).strip()]

    # ------- Fallback 2: 1) / 2) / 3) (line-anchored only) -------
    one = re.search(r"(?ms)^[ \t]*1\)\s*(.+?)(?=^[ \t]*[23]\)|\Z)", s)
    two = re.search(r"(?ms)^[ \t]*2\)\s*(.+?)(?=^[ \t]*3\)|\Z)", s)
    three = re.search(r"(?ms)^[ \t]*3\)\s*(.+?)\s*\Z", s)
    if one and two and three:
        return [one.group(1).strip(), two.group(1).strip(), three.group(1).strip()]

    # Nothing reliable found
    return []

# --------------------------------
def generate_consequence_prompt(original_scenario, user_choices, user_profile, context_blocks, temperature=0.5):
    # Follow on from prevoious prompt
    # note: removed items as it was not returning a good response (no flow on in the story)

    choices_bullets = "\n".join([f"- {c}" for c in user_choices]) if user_choices else "- (no choices recorded)"
    consequence_prompt = f"""
You are continuing a pick-a-path AI ethics training scenario for NZ workplaces.
The story continues from the ORIGINAL SCENARIO: {original_scenario} and the user choices - {choices_bullets}.
User profile: {user_profile}

Creativity level: {temperature:.2f} (0.0 = conservative/focused, 1.0 = creative/exploratory)
{f'Use a wildly creative and unconventional approach - be bold, imaginative, and explore unexpected scenarios and edge cases.' if temperature >= 0.9 
else f'Use a more creative and exploratory approach - consider diverse perspectives and less obvious scenarios.' if temperature >= 0.6 
else f'Use an interesting and engaging approach with some creative elements.' if temperature >= 0.4 
else f'Use a conservative, focused, and realistic approach - stick closely to typical workplace scenarios.'}

Create an opening scene and three choices. 
Keep it plain language, NZ context.
Use New Zealand spelling. Write "organise, "personalise", "realise" etc. Do not use American spelling.
Provide one scenario with exactly 3 actionable choices.
Ensure that at least one of the choices could lead to a bad outcome.

{f'Sentences can vary in length - use creative and varied sentence structures to enhance engagement.' if temperature >= 0.6 else 'Sentences should have a good balance of length and clarity.'}
{f'Explore unconventional or unexpected choices that challenge assumptions.' if temperature >= 0.6 else 'use a balanced approach in your choices' if temperature >= 0.4 else 'use a conservative approach in your choices'}

Continue the story and add 2-3 actionable choices, labelled clearly as:
CHOICE 1:
CHOICE 2:
CHOICE 3:

Creativity level: {temperature:.2f} (0.0 = conservative/focused, 1.0 = creative/exploratory)
Use a {'wildly creative approach approach' if 0.9 <= temperature <= 1.0 
else 'more creative and exploratory approach' if 0.6 <= temperature <= 0.8 
else 'an interesting and engaging approach' if 0.4 <= temperature <= 0.5 
else 'a conservative and focused approach'} to scenario development.

do not repeat earlier options; write new options for this stage of the story.
CRITICAL FORMAT REQUIREMENTS:
- You MUST use exactly "CHOICE 1:", "CHOICE 2:", "CHOICE 3:" as headers
- Do NOT use "A)", "B)", "C)", or any other format
- The choices must be concrete and specific actions

Do not include any consequences or explanations.
Do not repeat these instructions, system messages, or headings.
Output only the continued story and the three choices.
"""
    return consequence_prompt

# --------------------------------
def generate_outcome_prompt(original_scenario, user_choices, user_profile, context_blocks, temperature=0.5):
    # Create a final wrap-up: summarise the story, list the choices made, describe the resulting state,
    # and judge whether the user ended up in a good or bad place (with brief reasoning).
        
    choices_bullets = "\n".join([f"- {c}" for c in user_choices]) if user_choices else "- (no choices recorded)"
    creativity_guidance = "more creative and exploratory" if temperature > 0.6 else "more focused and realistic" if temperature < 0.4 else "balanced"
    outcome_prompt = f"""
You are finishing a pick-a-path AI-ethics scenario for a New Zealand workplace.

ORIGINAL SCENARIO (for context):
{original_scenario}

USER PROFILE:
{user_profile}

USER CHOICES (in order):
{choices_bullets}

Creativity level: {temperature:.2f} (0.0 = conservative/focused, 1.0 = creative/exploratory)
Use a {creativity_guidance} approach to developing the outcome assessment.

Summarise the story so far in 3 to 5 short sentences.
Describe the current resulting state in plain NZ English.
Judge whether the user’s decisions led to a good outcome or a bad outcome.
   - Say "Overall outcome: Good", "Overall outcome: Mixed" or "Overall outcome: Bad" (choose one).
   - Give 3 to 4 short reasons tied to story context 

RULES:
- Do not add new branching options or labels like CHOICE 1/2/3.
- Use short sentences. NZ spelling.
"""
    return outcome_prompt

# --------------------------------
def display_story_with_choices(scenario_text, choices, story_key):
    # Display the scenario with interactive choice buttons
    # Extract just the scenario part (before CHOICE 1:)
    scenario_only = re.split(r'(?im)^\s*CHOICE\s+1\s*:', scenario_text)[0].strip() # match CHOICE 1: with optional whitespace, also stops it seeing a) as a choice but as part of a sentence when it's in a sentence
    
    # If the story is finished, just show the text and return
    finished = st.session_state.get(story_key, {}).get('is_finished', False)
    if finished:
        st.write(scenario_only)
        return
    
    # show the current sccene details
    st.subheader("Current scene")
    st.markdown(scenario_only.replace("\n\n", "\n\n&nbsp;\n\n"))

    # If no choices, try a one-shot repair using the model's own text
    if not choices:
        repair_prompt = f"""
    Rewrite the text below into the required format.
    Rules:
    - Keep the same content, but write short, plain NZ English sentences.
    - Then output exactly three concrete choices labelled CHOICE 1:, CHOICE 2:, CHOICE 3:.
    - No extra headings, no 'Task:' lines, no markdown #.

    TEXT:
    {scenario_text}
    """
        with st.spinner("Creating options…"):
            try:
                repaired = generate_story_with_openai(repair_prompt, temperature=max(0.3, temperature-0.2), max_tokens=max_tokens, selected_model=openai_model)
                new_choices = parse_scenario_choices(repaired)
                if new_choices:
                    st.session_state[story_key]['scenario'] = repaired
                    st.session_state[story_key]['choices'] = []  # clear old
                    st.session_state[story_key]['choices'] = new_choices
                    # re-render with repaired content
                    scenario_only = re.split(r'(?im)^\s*CHOICE\s+1\s*:', scenario_text)[0].strip()
                    choices = new_choices
                else:
                    st.warning("Sorry, something weird happened - no choices generated! Please click Start New Story to try again.")
                    return
            except Exception as e:
                st.error(f"Repair failed: {e}")
                return

  
    # Handle single choice case - regenerate scenario
    if len(choices) == 1:
        st.warning("Only one choice found - this is not a proper scenario. Regenerating scenario...")
        # Force regeneration by clearing the scenario
        st.session_state[story_key]['scenario'] = None
        st.session_state[story_key]['choices'] = None
        st.session_state[story_key]['waiting_for_consequence'] = False
        st.rerun()
        return
    
    # Handle 2-3 choices dynamically
    if len(choices) < 2:
        st.warning(f"Only found {len(choices)} choices, but need at least 2. Full response:")
        return
    
    # Display the choices automatically (not in dropdown)
    st.subheader("Your Options:")
    for i, choice in enumerate(choices):
        st.write(f"**Choice {i+1}:**", choice)
    
    # Show choice buttons dynamically
    st.subheader("What do you choose?")
    
    # Create columns based on number of choices
    if len(choices) == 2:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Choice 1", key=f"{story_key}_lvl{st.session_state[story_key]['current_level']}_choice_1"):
                st.session_state[story_key]['selected_choice'] = choices[0]
                st.session_state[story_key]['choice_made'] = True
                st.rerun()
        with col2:
            if st.button("Choice 2", key=f"{story_key}_lvl{st.session_state[story_key]['current_level']}_choice_2"):
                st.session_state[story_key]['selected_choice'] = choices[1]
                st.session_state[story_key]['choice_made'] = True
                st.rerun()
    else:  # 3 choices
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Choice 1", key=f"{story_key}_lvl{st.session_state[story_key]['current_level']}_choice_1"):
                st.session_state[story_key]['selected_choice'] = choices[0]
                st.session_state[story_key]['choice_made'] = True
                st.rerun()
        with col2:
            if st.button("Choice 2", key=f"{story_key}_lvl{st.session_state[story_key]['current_level']}_choice_2"):
                st.session_state[story_key]['selected_choice'] = choices[1]
                st.session_state[story_key]['choice_made'] = True
                st.rerun()
        with col3:
            if st.button("Choice 3", key=f"{story_key}_lvl{st.session_state[story_key]['current_level']}_choice_3"):
                st.session_state[story_key]['selected_choice'] = choices[2]
                st.session_state[story_key]['choice_made'] = True
                st.rerun()


# --------------------------------
# One session id per browser session
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    CountVectorizer = None
    cosine_similarity = None

# ----------------------------------------------------------------------------------------------------------------------------
# --- UI ---
st.set_page_config(page_title="Pick-a-Path AI Ethics", page_icon="🎲", layout="wide")
st.title("🎲 Pick-a-Path Ethics & Results")

# --------------------------------
# Initialise LLM and Security Features
try:
    client = get_openai_client()
    # st.caption(f"Using OpenAI org: {client.organization}, key prefix: {client.api_key[:8]}")


    _ = client.models.list()
    st.sidebar.success("✅ Connected to LLM API")
except Exception as e:
    st.sidebar.error(f"❌ OpenAI connection failed: {e}")
    st.error("Cannot proceed without valid API connection. Please check your API key.")
    st.stop()

# Initialise rate limiter
rate_limiter = get_rate_limiter()

# Validate required files at startup
missing_files, file_errors = validate_required_files()
if missing_files:
    st.error("**Required files are missing!**")
    st.write("The application cannot run without these files:")
    for error in file_errors:
        st.text(error)
    
    st.markdown("---")
    st.info("""
    **Setup Instructions:**
    
    1. **Create database:** Run `python -m src.PAPEsetup_database`
    2. **Load structured data:** Run `python -m src.PAPEload_structured_data`
    3. **Ingest documents:** Run `python -m src.PAPEdata_load` (this will also build the FAISS index)
    4. **Tag themes (optional):** Run `python -m src.PAPEtag_themes`
    
    After completing these steps, refresh this page.
    """)
    st.stop()  # Stop execution if files are missing

with st.sidebar:
    st.header("Query")
    
    user_industry = st.selectbox(
        "What industry are you in?", 
        options=[
        "I feel lucky",
        "Banking", 
        "Government", 
        "Financial Advice",
        "Financial Management", 
        "Insurance", 
        "Investment", 
        "IT",
        "Other Financial Services"],
        index=0,
    )
    
    user_role = st.selectbox(
        "What role are you in? (please select the closest option)", 
        options=[
        "Any role will do",
        "Administrator",
        "Bank Teller",
        "Data Scientist",
        "Data Security",
        "Developer",
        "Financial Advisor",
        "Financial Manager",
        "Government Role",
        "Governance",
        "Insurance Agent",
        "Investment Advisor", 
        "IT role",
        "Other - unusual role"],
        index=0,
    )
    # selected_role = st.selectbox("What industry are you in?", options=user_role, index=0)

    #if not selected_role:  # Empty string is not ok
    #    st.warning("Please select an option")
    #    st.stop()
    
    user_topic = st.selectbox("What area would you like to explore?",  options=[
        "Random Ethics Topic",
        "Accountability",
        "Employment Impact",
        "Ethical Wellbeing",
        "Fairness",
        "Human Oversight",
        "Māori Data Sovereignty",
        "Privacy",
        "Security",
        "Transparency",
        "Other",        
        ],
        index=0,
    )
    
    #if not user_topic:  # Empty string is not ok
    #    st.warning("Please select a topic")
    #    st.stop()
    
    # Free text add-on (optional) to let them be specific
    user_query_free = st.text_input(
        "Optional: add a short description (e.g., 'monitoring staff messages')",
        value="",
        max_chars=500,  # SECURITY: Limit input length
        help="This is appended to the query to sharpen retrieval. Max 500 characters."
    )

    # SECURITY: Validate and sanitise user input
    if user_query_free:
        is_valid, error_msg = validate_string_length(user_query_free, 500, "query")
        if not is_valid:
            st.warning(error_msg)
            user_query_free = user_query_free[:500]

        # Check for potential prompt injection
        is_suspicious, reason = detect_prompt_injection(user_query_free)
        if is_suspicious:
            st.warning(f"⚠️ Your input contains suspicious patterns: {reason}")
            st.info("Please rephrase your query or it may not work as expected.")
    
    top_k = st.slider("Number of retrieved references (lots can overwhelm)", 1, 10, 5) # min 1, max 10, default 5
    temperature = st.slider("Creativity of answers", 0.0, 1.0, 0.5, 0.05)
    
    # Show temperature indicator
    if temperature < 0.3:
        temp_label = "🔵 Conservative"
    elif temperature < 0.6:
        temp_label = "🟢 Balanced"
    elif temperature < 0.8:
        temp_label = "🟡 Creative"
    else:
        temp_label = "🔴 Very Creative"
    st.caption(f"Temperature: {temperature:.2f} - {temp_label}")
    
    max_tokens = st.slider("Story size approx.(words)", 300, 700, 500, 50)

    # Add feedback form button in sidebar
    st.divider()
    feedback_form_url = "https://docs.google.com/forms/d/e/1FAIpQLSdCMB-O98aGLRIYkiAJqkCEnGU8NcHP47v9g0IJBaK1-fYVYQ/viewform?usp=header"
    # Button navigates directly to feedback form
    st.link_button(
        "I'm done & ready to give feedback",
        feedback_form_url,
        type="secondary",
        use_container_width=True
    )

    st.divider()
    show_debug_prompt = st.checkbox("Show full prompt", value=False)
   
    # Model selection with default "gpt-4o-mini"
    st.divider()
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
    default_model = "gpt-4o-mini"
    default_index = model_options.index(default_model) if default_model in model_options else 0
    
    openai_model = st.selectbox(
        "Model", 
        model_options, 
        index=default_index,
        key="model_selectbox"
    )

    # Display rate limiter stats
    st.subheader("API Usage")
    rate_limiter = get_rate_limiter()
    stats = rate_limiter.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Requests/min", f"{stats['requests_last_minute']}/{stats['max_requests_per_minute']}")
        st.metric("Tokens/min", f"{stats['tokens_last_minute']}/{stats['max_tokens_per_minute']}")
    with col2:
        st.metric("Requests/hour", f"{stats['requests_last_hour']}/{stats['max_requests_per_hour']}")
        st.metric("Daily cost", f"${stats['daily_cost']:.4f}/${stats['max_cost_per_day']:.2f}")

   

colL, colR = st.columns([1,1])

with st.spinner("Loading index…"):
    index, ids = load_index()
    embedder = get_embedder()

# added this so that the story continues instead of restarting
story_key = f"story_{user_industry}_{user_role}_{user_topic}"
if story_key not in st.session_state:
    st.session_state[story_key] = {
        'scenario': None,
        'choices': None,
        'consequences': [],
        'user_choices': [],
        'current_level': 0,
        'choice_made': False,
        'selected_choice': None,
        'waiting_for_consequence': False,
        'max_levels': max_levels,
        # store latest retrieved evidence rows so we can show them after reruns
        'evidence_rows': None,
        'context_blocks': ''
    }


if st.button("Generate a story"):
    # Build a query from sidebar selections
    query = build_query(user_industry, user_role, user_topic, user_query_free)
    qvec = embedder.encode([query], convert_to_numpy=True).astype(np.float32) # convert query to a numerical vector via embedding module
    # becomes a 2D NumPy array with shape (1, D), where D is the embedding dimension
    qvec /= (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12) # normalise to unit length 

    #adding in ability to rank the returned items according to the topic selected
    pool = min(max(top_k * 5, 20), 100) # get a larger pool to allow re-ranking. min max it to suit query size e.g., k=3 -> pool=15; cap at 100
    D, I = index.search(qvec, pool) # D: scores, I: indeces of the top-K chunks
    
    # Guard invalid indices and keep only valid hits
    cand_pairs = []
    for idx, score in zip(I[0].tolist(), D[0].tolist()):
        if idx is None or idx < 0:
            continue
        # Index to your persisted ids list
        try:
            cand_id = ids[idx]
            cand_pairs.append((cand_id, float(score)))
        except IndexError:
            # Ignore out-of-range ids; can happen if ids / index are out of sync
            continue

    # If nothing valid, bail early
    if not cand_pairs:
        st.warning("No valid matches returned from the vector index.")
        rows = pd.DataFrame()
        ranked = []
        rows_all = pd.DataFrame()
    else:
        # Drop weak matches using a simple relative threshold.
        # Tweak 'keep_fraction' up for stricter filtering
        scores_only = [s for _, s in cand_pairs] # extract the list of D from cand_pairs, use _ to skip I
        lo, hi = min(scores_only), max(scores_only)
        if hi > lo:
            keep_fraction = 0.25  # keep top quartile of the *returned* score range
            thr = lo + keep_fraction * (hi - lo)
            cand_pairs = [(cid, s) for cid, s in cand_pairs if s >= thr] # filter to keep only items above the threshold
        
        # pick the theme name from UI (map Topic selectbox to theme rows)
        topic_to_theme = {
            "Accountability": "accountability",
            "Employment Impact": "employment_impact",  
            "Ethical Wellbeing": "ethical_wellbeing",
            "Fairness": "fairness",
            "Human Oversight": "human_oversight",
            "Māori Data Sovereignty": "maori_data",
            "Privacy": "privacy",
            "Transparency": "transparency",
            "Security": "security",
            "Other": None,
        }
        selected_theme = topic_to_theme.get(user_topic)
        themed_ids = get_chunk_ids_for_theme(selected_theme) if selected_theme else set()

        # Normalise FAISS scores to 0..1 so boosts behave the same whether FAISS scores are tight or wide
        scores_only = np.array([s for _, s in cand_pairs], dtype=np.float32)
        smin, smax = float(scores_only.min()), float(scores_only.max())
        rng = max(smax - smin, 1e-6)
        norm_pairs = [(cid, (s - smin) / rng) for cid, s in cand_pairs]

        # Dynamic, tunable topic boost (stronger base match → slightly more boost)
        boosted = []
        for cid, s_norm in norm_pairs:
            if cid in themed_ids:
                # Boost slides between 1.075x and 1.225x as s_norm goes 0→1 (with topic_boost=0.15)
                factor = 1.0 + topic_boost * (0.5 + 0.5 * s_norm)
                boosted.append((cid, s_norm * factor))
            else:
                boosted.append((cid, s_norm))

        # Sort by boosted score and keep top_k
        boosted.sort(key=lambda x: x[1], reverse=True)
        ranked = boosted[:top_k]

        # Fetch metadata for ranked IDs 
        rows_all = fetch_chunks([cid for cid, _ in ranked])
        if not rows_all.empty:
            rows_all["score"] = [s for _, s in ranked]  # keep score alignment
        else:
            rows_all = pd.DataFrame()

    by_doc = {}
    filtered_rows = []
    
    for row in rows_all.itertuples(index=False):
        # Combine title and path/url as a document key
        key = (row.title, row.path_or_url)
        count = by_doc.get(key, 0)
        if count < per_doc_cap:
            filtered_rows.append(row)
            by_doc[key] = count + 1

    # Keep the same number of rows as top_k if possible
    rows = pd.DataFrame(filtered_rows)[:top_k]

    # Save retrieved evidence rows and context for follow-on story sections after a choice
    st.session_state[story_key]['evidence_rows'] = rows.to_dict('records')
    st.session_state[story_key]['context_blocks'] = "\n\n".join(
        f"[Source: {r['title']} | {r['origin'] or r['path_or_url']}]\n{r['text']}"
        for _, r in rows.iterrows()
    )

    # Extract top_ids for downstream consistency
    top_ids = list(rows.chunk_id) if "chunk_id" in rows.columns else []

    # Compose a prompt with user profile
    story_prompt = compose_prompt_with_profile(
        query,
        rows, 
        user_industry, 
        user_role, 
        user_topic,
        temperature=temperature
    )

    # Display the prompt (moved to left column for better layout)
    with colL:
        # Do not show the prompt by default; only display if the debug toggle is on
        if show_debug_prompt:
            st.subheader("Prompt (grounded)")
            st.code(story_prompt, language="markdown")

        # ----------------------------------------------------------------------------------------------------------------------------
        # --- Story Management ---

        if story_key not in st.session_state:
            st.session_state[story_key] = {
                'scenario': None,
                'choices': None,
                'consequences': [],
                'user_choices': [],
                'current_level': 0,
                'choice_made': False,
                'selected_choice': None,
                'waiting_for_consequence': False,
                'max_levels': max_levels  # Limit to 4 levels
            }

        # add to event log
        log_event_gsheet(
            "story_generated",
            story_key,
            user_industry,
            user_role,
            user_topic,
            user_query_free,
            f"Level={st.session_state[story_key]['current_level']}"
        )   

        # Generate new scenario when button is clicked
        with st.spinner("Generating scenario..."):
            try:
                t0 = time.perf_counter()
                scenario = generate_story_with_openai(story_prompt, temperature=temperature, max_tokens=max_tokens, selected_model=openai_model)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                scene_only = re.split(r'(?im)^\s*CHOICE\s+1\s*:', scenario)[0].strip()
                ease, grade = safe_readability(scene_only)
                choices = parse_scenario_choices(scenario)
                if not scenario or not choices:
                        log_event_gsheet(
                        "generation_failure",
                        story_key=story_key,
                        user_industry=user_industry,
                        user_role=user_role,
                        user_topic=user_topic,
                        user_query=user_query_free,
                        details="Initial generation blank or no choices",
                        level=0,
                        )
                else:
                    log_event_gsheet(
                        "story_started",
                        story_key=story_key,
                        user_industry=user_industry,
                        user_role=user_role,
                        user_topic=user_topic,
                        user_query=user_query_free,
                        details="Initial story generated",
                        level=st.session_state[story_key]['current_level'],   # likely 0
                        latency_ms=elapsed_ms,
                        flesch_ease=ease,
                        fk_grade=grade,
                        choices_count=len(choices) if choices else 0,
                    )

                # Store the initial story & choices so we can go back to them if needed

                st.session_state[story_key]['origin_scenario'] = scenario
                st.session_state[story_key]['origin_choices']  = choices
                # log the story for analytics
                origin_scene_only = re.split(r'(?im)^\s*CHOICE\s+1\s*:', scenario)[0].strip()
                st.session_state[story_key]['origin_scene_only'] = origin_scene_only
                st.session_state[story_key]['scenes_only'] = [origin_scene_only]  # running list


                # Reset the story state for a fresh start
                st.session_state[story_key]['scenario'] = scenario
                st.session_state[story_key]['choices'] = []  # clear old
                st.session_state[story_key]['choices'] = choices
                st.session_state[story_key]['consequences'] = []
                st.session_state[story_key]['user_choices'] = []
                st.session_state[story_key]['current_level'] = 0
                st.session_state[story_key]['choice_made'] = False
                st.session_state[story_key]['selected_choice'] = None
                st.session_state[story_key]['waiting_for_consequence'] = False
                st.session_state[story_key]['is_finished'] = False
                # Store the context for consequence generation
                st.session_state[story_key]['context_blocks'] = "\n\n".join([f"[Source: {r['title']} | {r['origin'] or r['path_or_url']}]\n{r['text']}" for _, r in rows.iterrows()])
                st.session_state[story_key]['origin_scenario'] = scenario
                # Store temperature so it's available for consequence/outcome generation
                st.session_state[story_key]['temperature'] = temperature
                st.session_state[story_key]['max_tokens'] = max_tokens

            except Exception as e:
                st.error(f"Generation error: {e}")
                scenario = None
                choices = None
        
    
# Handle choice consequences or final wrap-up BEFORE displaying scenario
current_level = st.session_state[story_key]['current_level']
if (st.session_state[story_key]['choice_made']
    and not st.session_state[story_key]['waiting_for_consequence']):
    
    # add choice to log
    log_event_gsheet(
        "choice_made",
        story_key=story_key,
        user_industry=user_industry,
        user_role=user_role,
        user_topic=user_topic,
        user_query=user_query_free,
        details=st.session_state[story_key]['selected_choice'],
        level=st.session_state[story_key]['current_level'],
        )


    
    selected_choice = st.session_state[story_key]['selected_choice']
    st.session_state[story_key]['user_choices'].append(selected_choice)

    # If we still have levels to go, generate another branching scene
    if current_level < st.session_state[story_key]['max_levels'] - 1:
        st.subheader("🎯 Your Choice")
        st.write(f"**You chose:** {selected_choice}")

        # Put the app into a “busy” state for this branch
        st.session_state[story_key]['waiting_for_consequence'] = True

        # Build the follow-on prompt from the scene text (without old CHOICE lines)
        selected_choice = st.session_state[story_key]['selected_choice']
        previous_scenario = st.session_state[story_key]['scenario']  # capture now
        scenario_for_prompt = re.split(r'(?im)^\s*CHOICE\s+1\s*:', previous_scenario)[0].strip()

        # Use stored temperature from story generation, or fall back to current slider value
        story_temperature = st.session_state[story_key].get('temperature', temperature)
        story_max_tokens = st.session_state[story_key].get('max_tokens', max_tokens)
        
        consequence_prompt = generate_consequence_prompt(
            scenario_for_prompt,
            [selected_choice],
            f"Industry: {user_industry}\nRole: {user_role}\nTopic: {user_topic}",
            st.session_state[story_key].get('context_blocks', ''),
            temperature=story_temperature
        )

        # Display the consequence prompt if debug mode is enabled
        if show_debug_prompt:
            st.subheader("Consequence Prompt")
            st.code(consequence_prompt, language="markdown")

        with st.spinner("Generating consequences..."):
            try:
                t_start = time.perf_counter()
                consequence = generate_story_with_openai(
                    consequence_prompt,
                    temperature=story_temperature,
                    max_tokens=story_max_tokens,
                    selected_model=openai_model
                )
                elapsed_ms = int((time.perf_counter() - t_start) * 1000)

                if not consequence:
                    st.warning("No new consequence generated — reusing previous story.")
                else:
                    # --- Compute metrics *after* we have the new text ---
                    consequence_scene_only = re.split(r'(?im)^\s*CHOICE\s+1\s*:', consequence)[0].strip()
                    ease, grade = safe_readability(consequence_scene_only)

                    prev_list = st.session_state[story_key].get('scenes_only', [])
                    prev_scene_only = prev_list[-1] if prev_list else ""
                    sim = cosine_repeat(prev_scene_only, consequence_scene_only)
                    st.session_state[story_key].setdefault('scenes_only', []).append(consequence_scene_only)

                    new_choices = parse_scenario_choices(consequence)

                    # Log failure or step with metrics
                    if not new_choices:
                        log_event_gsheet(
                            "generation_failure",
                            story_key=story_key,
                            user_industry=user_industry,
                            user_role=user_role,
                            user_topic=user_topic,
                            user_query=user_query_free,
                            details="Empty consequence or no choices",
                            level=st.session_state[story_key]['current_level'],
                            latency_ms=elapsed_ms,
                            flesch_ease=ease,
                            fk_grade=grade,
                            repeat_sim=sim,
                            choices_count=0,
                        )
                    else:
                        log_event_gsheet(
                            "story_step",
                            story_key=story_key,
                            user_industry=user_industry,
                            user_role=user_role,
                            user_topic=user_topic,
                            user_query=user_query_free,
                            details="Consequence generated",
                            level=st.session_state[story_key]['current_level'] + 1,
                            latency_ms=elapsed_ms,
                            flesch_ease=ease,
                            fk_grade=grade,
                            repeat_sim=sim,
                            choices_count=len(new_choices),
                        )

                        # Update state (always overwrite)
                        st.session_state[story_key]['consequences'].append({
                            'choice': selected_choice,
                            'consequence': consequence,
                            'new_choices': new_choices,
                            'level': current_level + 1
                        })
                        st.session_state[story_key]['scenario'] = consequence
                        st.session_state[story_key]['choices'] = new_choices
                        st.session_state[story_key]['current_level'] += 1

            except Exception as e:
                st.error(f"Consequence generation error: {e}")

        # Reset click flags — do this outside the try so it always runs
        st.session_state[story_key]['choice_made'] = False
        st.session_state[story_key]['selected_choice'] = None
        st.session_state[story_key]['waiting_for_consequence'] = False

        
    else:
        # We reached the maximum depth: produce FINAL OUTCOME instead of a new scene with choices
        st.session_state[story_key]['waiting_for_consequence'] = True
       

        # Use stored temperature from story generation, or fall back to current slider value
        story_temperature = st.session_state[story_key].get('temperature', temperature)
        story_max_tokens = st.session_state[story_key].get('max_tokens', max_tokens)
        
        outcome_prompt = generate_outcome_prompt(
            st.session_state[story_key]['scenario'],
            st.session_state[story_key]['user_choices'],  # include all choices made
            f"Industry: {user_industry}\nRole: {user_role}\nTopic: {user_topic}",
            st.session_state[story_key].get('context_blocks', ''),
            temperature=story_temperature
        )

        # Display the outcome prompt if debug mode is enabled
        if show_debug_prompt:
            st.subheader("Outcome Prompt")
            st.code(outcome_prompt, language="markdown")

        with st.spinner("Wrapping up your story..."):
            try:
                outcome_text = generate_story_with_openai(
                    outcome_prompt,
                    temperature=max(0.2, story_temperature - 0.1),  # often nicer a touch more deterministic
                    max_tokens=story_max_tokens,
                    selected_model=openai_model
                )
                
                # strip any stray choices the model added
                outcome_text_clean = re.split(r'(?im)^\s*CHOICE\s+1\s*:', outcome_text)[0].strip()
                # No more choices after final outcome
                st.session_state[story_key]['scenario'] = outcome_text_clean
                st.session_state[story_key]['choices'] = []
                st.session_state[story_key]['is_finished'] = True
                final_level = st.session_state[story_key]['current_level'] + 1  # human-readable count
                log_event_gsheet(
                    "story_completed",
                    story_key=story_key,
                    user_industry=user_industry,
                    user_role=user_role,
                    user_topic=user_topic,
                    user_query=user_query_free,
                    details="Final outcome shown",
                    level=final_level,
                )

            except Exception as e:
                st.error(f"Outcome generation error: {e}")
                st.session_state[story_key]['is_finished'] = True  # end anyway if model fails
                final_level = st.session_state[story_key]['current_level'] + 1  # human-readable count
                log_event_gsheet(
                    "story_completed",
                    story_key=story_key,
                    user_industry=user_industry,
                    user_role=user_role,
                    user_topic=user_topic,
                    user_query=user_query_free,
                    details="Final outcome shown",
                    level=final_level,
                )


        # Reset click flags
        st.session_state[story_key]['choice_made'] = False
        st.session_state[story_key]['selected_choice'] = None
        st.session_state[story_key]['waiting_for_consequence'] = False

#--------------------------------
# The RENDERER - the section which renders the scenraio to the screen
# Display the current scenario if it exists
if st.session_state[story_key]['scenario']:
    st.subheader("🎲 Your AI Ethics Scenario")
    
    # Show story progress
    current_level = st.session_state[story_key]['current_level']  # 0-based
    max_levels = st.session_state[story_key]['max_levels']
    progress_value = (current_level + 1) / max_levels
    st.progress(progress_value, text=f"Level {current_level + 1} of {max_levels}")
    
    # Display scenario with choices - always use current session state values
    if st.session_state[story_key].get('is_finished', False): # if we are finished (default set to false)
        # Final summary only
        scenario_only = re.split(r'(?im)^\s*CHOICE\s+1\s*:',
                                 st.session_state[story_key]['scenario'])[0].strip()
        st.write(scenario_only)
        st.success("🎉 **End of Story!** You've reached the maximum story depth. This scenario is complete.")
        
        with st.expander("Show how you got here"):
            # Original opening scene
            orig = st.session_state[story_key].get('origin_scenario', '')
            orig_scene = re.split(r'(?im)^\s*CHOICE\s+1\s*:', orig)[0].strip()
            st.markdown("**Original opening scene**")
            st.write(orig_scene if orig_scene else "_No original scene saved._")

            # Choices made (in order)
            path = st.session_state[story_key].get('user_choices', [])
            st.markdown("**Choices you made**")
            if path:
                for i, c in enumerate(path, 1):
                    st.write(f"{i}. {c}")
            else:
                st.write("_No choices recorded._")
    else:
        # Normal branch-with-choices render
        display_story_with_choices(
            st.session_state[story_key]['scenario'],
            st.session_state[story_key]['choices'],
            story_key
        )
    
    # Display retrieved evidence section below the scenario (optional expander)
    with st.expander("Retrieved evidence"):
        rows_to_show = None
        if st.session_state[story_key].get('evidence_rows'):
            rows_to_show = pd.DataFrame(st.session_state[story_key]['evidence_rows'])
        
        if rows_to_show is None or rows_to_show.empty:
            st.info("No evidence found for this query. Try adjusting the topic or add a short description.")
        else:
            for _, r in rows_to_show.iterrows():
                with st.expander(f"{r['title']}  ·  {r['origin'] or r['source_type']}"):
                    st.write(f"**Source:** `{r['path_or_url']}`")
                    snippet = textwrap.shorten(" ".join(r["text"].split()), width=900, placeholder=" …")
                    st.write(snippet)
    
    # Add story controls
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Start New Story", key=f"{story_key}_restart"):
            # Completely delete the story state to force fresh start
            if story_key in st.session_state:
                del st.session_state[story_key]
            st.rerun()
    
    with col2:
        if current_level > 0 and st.button("⬅️ Go Back", key=f"{story_key}_back"):
            # Go back to previous level
            if st.session_state[story_key]['consequences']:
                prev_consequence = st.session_state[story_key]['consequences'].pop()
                st.session_state[story_key]['current_level'] -= 1
                st.session_state[story_key]['user_choices'].pop()
                
                if st.session_state[story_key]['consequences']:
                    # Go back to previous consequence
                    prev = st.session_state[story_key]['consequences'][-1]
                    st.session_state[story_key]['scenario'] = prev['consequence']
                    st.session_state[story_key]['choices'] = prev['new_choices']
                else:
                    # Go back to original scenario
                    st.session_state[story_key]['scenario'] = st.session_state[story_key]['origin_scenario']
                    st.session_state[story_key]['choices'] = st.session_state[story_key]['origin_choices']
                
                # Reset choice state
                st.session_state[story_key]['choice_made'] = False
                st.session_state[story_key]['selected_choice'] = None
                st.session_state[story_key]['waiting_for_consequence'] = False
    
    with col3:
        if st.session_state[story_key]['user_choices']:
            st.write(f"**Choices made:** {len(st.session_state[story_key]['user_choices'])}")
else:
    st.info("<-- Make your selections then click 'Generate a story' to start your journey.")
