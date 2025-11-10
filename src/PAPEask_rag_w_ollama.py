# --------------------------------
# Pick-a-Path AI Ethics
# test interface to ask query of the db
# --------------------------------


# ask_rag.py
import os, sqlite3, argparse, textwrap
from pathlib import Path
import logging
import ollama
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import subprocess, json

from src.PAPElogging_config import setup_logging


PROJECT_DIR = Path(__file__).resolve().parent.parent  # the parent directory of this file
DB_PATH = PROJECT_DIR / "ragEthics.db"
INDEX_DIR = PROJECT_DIR / "vectorstore"
INDEX_PATH = INDEX_DIR / "chunks.faiss"
IDS_CSV = PROJECT_DIR / "vectorstore/chunk_ids.csv"

# ________________________________________________________
# Standard logging header 
logger = logging.getLogger(__name__)

# ________________________________________________________

def load_index():
    if not (INDEX_PATH.exists() and IDS_CSV.exists() and DB_PATH.exists()):
        raise SystemExit("Missing index or DB. Run your ingestion + build_faiss_index first.")
    index = faiss.read_index(str(INDEX_PATH))
    ids = pd.read_csv(str(IDS_CSV))["id"].tolist()
    return index, ids

def fetch_chunks(chunk_ids):
    if not chunk_ids:
        return pd.DataFrame() # error handling if no chunk_ids
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

def main():
    ap = argparse.ArgumentParser(description="Query the ethics RAG index")
    ap.add_argument("query", nargs="+", help="Enter your topic/question e.g., privacy monitoring in bank branches")
    ap.add_argument("-k", type=int, default=5, help="Top-k chunks to return")
    ap.add_argument("--prompt", action="store_true", help="Print a composed LLM prompt at the end")
    ap.add_argument("--generate", action="store_true", help="Generate a story using the LLM")
    args = ap.parse_args()
    query = " ".join(args.query).strip()

    # Validate required files exist before proceeding
    if not (INDEX_PATH.exists() and IDS_CSV.exists() and DB_PATH.exists()):
        print("ERROR: Required files are missing!")
        missing = []
        if not DB_PATH.exists():
            missing.append(f"Database: {DB_PATH}")
        if not INDEX_PATH.exists():
            missing.append(f"Index: {INDEX_PATH}")
        if not IDS_CSV.exists():
            missing.append(f"Chunk IDs: {IDS_CSV}")
        print("\nMissing files:")
        for item in missing:
            print(f"  - {item}")
        print("\nSetup instructions:")
        print("  1. Create database: python -m src.PAPEsetup_database")
        print("  2. Load structured data: python -m src.PAPEload_structured_data")
        print("  3. Ingest documents: python -m src.PAPEdata_load")
        raise SystemExit(1)

    print(f"Query: {query}")
    index, ids = load_index()
    embedder = SentenceTransformer("all-MiniLM-L6-v2") # initialise the converter so we can use it on the next line
    queryvector = embedder.encode([query], convert_to_numpy=True).astype(np.float32) # convert text into numerical vector representations (embeddings)
    # a numpy array - each row representing a vector, each column representing a dimension
    # axis=1 means normalise each row (vector) to have a unit length 
    # to ensure they are treated equally regardless of actual length
    # keepdims=True means keep the dimensions of the array
    # 1e-12 is a small number added to avoid division by zero
    # /= divide each vector in the array by it's normalised magnitude.
    queryvector /= (np.linalg.norm(queryvector, axis=1, keepdims=True) + 1e-12) # normalise the query


    # search the index for the top-k results
    print(f"Searching index for Top-k results: {args.k}")
    # search using the FAISS index for the top-k results
    # D contains the distances between the query vector and the k nearest neighbors. 
    # I contains the indexes of the k nearest neighbors in the index
    D, I = index.search(queryvector, args.k) 
    top_ids = [ids[i] for i in I[0]] # get the ids of the top-k results

    rows = fetch_chunks(top_ids) # create an array of the chunks returned
    for i, r in rows.iterrows(): # iterate over the rows of the array
        # print the source type, origin, title, path_or_url, and text
        src = f"{r['source_type']} | {r['origin'] or 'unknown origin'}"
        print("\n" + "—"*72) # print a separator line
        print(f"{r['title']}  [{src}]")
        print(f"->   {r['path_or_url']}")
        snippet = textwrap.shorten(" ".join(r["text"].split()), width=600, placeholder=" …") # shorten the text to 600 characters
        print(f"\n{snippet}")

    
    # Build context for prompting/generation if requested
    story_prompt = None
    if args.prompt or args.generate:
        context_blocks = "\n\n".join(
            f"[Source: {r['title']} | {r['origin'] or r['path_or_url']}]\n{r['text']}"
            for _, r in rows.iterrows()
        )
        story_prompt = f"""
You are generating an NZ workplace AI-ethics scenario in the pick-a-path format. Use the retrieved context below to ground facts and principles.
Create an opening scene and three decision options. Keep it approximately 200 to 300 words, plain language, NZ context.
Use new zealand spelling throughout. absolutely NO use of z in words that are '...ise'. if using dashes, only use n-dash; do not use m-dash anywhere. Sentences should be no longer than 30 words. Use business language. 
only provide one scenario with 3 choices. 
User query: {query}

Retrieved context (verbatim — cite specific principles or passages when relevant):
{context_blocks}
""".strip()

    if args.prompt and story_prompt:
        print("\n" + "="*72)
        print("Copy-paste this into your LLM to generate a grounded story:")
        print(story_prompt)

    if args.generate and story_prompt:
        # Call local Phi-3 Mini via Ollama; ensure you've run:  ollama pull phi3:mini
        try:
            result = subprocess.run(
                ["ollama", "run", "phi3:mini"],
                input=story_prompt,
                text=True,
                capture_output=True,
                encoding="utf-8",   # force UTF-8, had issues with maori macrons in text
                errors="replace"    # replace bad chars if needed
            )
            if result.returncode != 0:
                print("\n[Generation error]")
                print(result.stderr.strip() or "Ollama returned a non-zero exit code.")
            else:
                print("\n" + "="*72)
                print("Generated Story (phi3:mini):\n")
                print(result.stdout.strip())
        except FileNotFoundError:
            print("\n[Generation error] Could not find the 'ollama' command. Is Ollama installed and on your PATH?")



# only run if this file is called directly, not imported as part of the package
if __name__ == "__main__":
    setup_logging(log_to_file=True, log_dir=PROJECT_DIR / "logs")
    main()



