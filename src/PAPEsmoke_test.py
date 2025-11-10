# --------------------------------
# Pick-a-Path AI Ethics
# Smoke test if data in db is querying correctly
# --------------------------------

from pathlib import Path
import sqlite3, faiss, numpy as np
from sentence_transformers import SentenceTransformer
import logging
from src.PAPElogging_config import setup_logging


PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_DIR / "ragEthics.db"
INDEX_PATH   = PROJECT_DIR / "vectorstore" / "chunks.faiss"
IDS_CSV      = PROJECT_DIR / "vectorstore" / "chunk_ids.csv"
DEFAULT_K = 5  # Default number of chunks to return

# ________________________________________________________
# Standard logging header 
logger = logging.getLogger(__name__)

#_________________________________________________________
# Error handling for missing files/directories
def ensure_env():
    missing = [p for p in [DB_PATH, INDEX_PATH, IDS_CSV] if not p.exists()]
    if missing:
        print("Required files or folders not found:")
        for p in missing: print(f"   - {p}")
        print("Tip: run the setup script first: python -m src.PAPEsetup_database")
        raise SystemExit(1)

def load_index_and_ids():
    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        raise SystemExit(f"Failed to read FAISS index at {INDEX_PATH}: {e}")
    try:
        id_map = np.loadtxt(IDS_CSV, delimiter=",", dtype=int, skiprows=1).reshape(-1)
    except Exception as e:
        raise SystemExit(f"Failed to read ID map at {IDS_CSV}: {e}")
    if index.ntotal != len(id_map):
        raise SystemExit(f"Index size ({index.ntotal}) != id_map size ({len(id_map)}). "
              "Rebuild with: python -m src.PAPEsetup_database")
    return index, id_map

#_________________________________________________________
# Main function to query the database and return top k chunks

def top_k(query: str, k: int = None):
    # Query the database and return top k chunks. If k is None, prompts user for input.
    # Ask the user how many chunks to return if k not provided
    if k is None:
        while True:
            user_input = input(f"How many chunks would you like returned? (default {DEFAULT_K}): ")
            if user_input.strip() == "":
                k = DEFAULT_K  # Default if blank
                break
            try:
                k = int(user_input)
                if k > 0:
                    break  # Valid integer, exit loop
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter a whole number.")
    
    # Ensure required files exist
    ensure_env() 
    # load faiss + id mapping
    index, id_map = load_index_and_ids() 
    
    # embed the query with the SAME model used for the index
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = model.encode([query], convert_to_numpy=True).astype("float32") # FAISS needs float32 not 64 to minimise processing time
    if q.ndim == 1:  # if single query, add a new axis to make it 2D
        q = q.reshape(1, -1)  # shape (1, embedding_dim)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)  # normalise to unit length for cosine measurement

    # search
    k = max(1, min(k, index.ntotal))  # ensure k is not larger than the number of indexed items
    D, I = index.search(q, k) # scores cosine IP similarity
    hit_ids = id_map[I[0]]
    if hit_ids.size == 0: # error handling if no hits
        logger.warning("No results found for that query.")                                # ← CHANGED
        return  
    
    # fetch chunks from sqlite and print snippets
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            qmarks = ",".join("?" * len(hit_ids))
            rows = cur.execute(
                f"""SELECT id,
                document_id, 
                chunk_index, 
                text FROM chunk WHERE id IN ({qmarks})""",
                tuple(map(int, hit_ids))
            ).fetchall()
    except sqlite3.OperationalError as e: # error handling for db issues
        raise SystemExit(f"Database error: {e}. The schema may be missing. "
              "Rebuild with: python -m src.PAPEsetup_database")

    if not rows: # error handling for no rows returned
        logger.warning("No matching chunk rows returned from the database.")              # ← CHANGED
        return

    # keep the same order as FAISS results
    by_id = {r[0]: r for r in rows}
    print(f"\nQuery: {query}\nTop {k} hits:")
    for rank, cid in enumerate(hit_ids, 1):
        row = by_id.get(int(cid)) # defensive check if DB had partial rows
        if not row:  # error handling for missing row
            continue
        _, doc_id, chunk_idx, text = by_id[int(cid)]
        snippet = (text[:220] + "…") if len(text) > 220 else text
        print(f"\n#{rank}  (chunk {chunk_idx} of doc {doc_id})")
        print(snippet)

# only run if this file is called directly, not imported as part of the package
if __name__ == "__main__":
    from src.PAPElogging_config import setup_logging
    setup_logging()
    top_k("transparency and human oversight in NZ finance", k=DEFAULT_K)
