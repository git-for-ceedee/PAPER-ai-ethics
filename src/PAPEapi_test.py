# --------------------------------
# Pick-a-Path AI Ethics
# API test
# --------------------------------


import os
from openai import OpenAI

# Try to read your API key from Streamlit secrets if available, or from environment variables
try:
    import streamlit as st
    api_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    api_key = None

api_key = api_key or os.getenv("OPENAI_API_KEY")

if not api_key:
    print("No OpenAI API key found. Add it to .streamlit/secrets.toml or set it as an environment variable.")
    exit(1)

client = OpenAI(api_key=api_key)

try:
    models = client.models.list()
    print("✅ OpenAI connection OK")
    print(f"Available models (first few): {[m.id for m in models.data[:10]]}")
except Exception as e:
    print(f"OpenAI check failed: {e}")
