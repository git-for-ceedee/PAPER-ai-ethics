# --------------------------------
# Pick-a-Path AI Ethics
# Ingest data into RAG db
# --------------------------------

# packages needed to ingest the data in several formats
import os, re, hashlib, sqlite3
from pathlib import Path
import logging

import pandas as pd
import faiss, numpy as np
import fitz  # PyMuPDF for PDF handling
import trafilatura  # for web scraping
from sentence_transformers import SentenceTransformer
from docx import Document as Docx

from src.PAPEingest_web_sources import ingest_web_urls
from src.PAPEsetup_database import init_db, assert_data_dir
from src.PAPElogging_config import setup_logging
from src.PAPEsecurity import validate_file_path, sanitize_filename

# ________________________________________________________
# This script ingests local PDF and DOCX files and extracts URLs from a DOCX file.
# It also provides utilities for web scraping and text chunking.

PROJECT_DIR = Path(__file__).resolve().parent.parent  # the parent directory of this file
DB_PATH = PROJECT_DIR / "ragEthics.db"
DATA_DIR = PROJECT_DIR / "data" / "raw_cases"
DOCX_WITH_URLS = DATA_DIR / "AIML339_Sources_of_data.docx"
INDEX_DIR = PROJECT_DIR / "vectorstore"
INDEX_DIR.mkdir(exist_ok=True)  # ensure the index directory exists
URLS_FILE = PROJECT_DIR / "data" / "raw_cases" / "urls.txt"  # plain text, one URL per line
SUPPORTED_EXTS = {".pdf", ".docx"}

# ________________________________________________________
# Standard logging header 
logger = logging.getLogger(__name__)

# ________________________________________________________
# polite crawling
USER_AGENT = "EthicsStoryRAG/1.0 (+https://example.local)"
CRAWL_DELAY_S = 1.0

# (optional) set these for testing; reduce processing time by ignoring urls and the index function
# Toggles (change to True/False as needed)
DO_INGEST_WEB = True     # crawl and ingest URLs
DO_BUILD_INDEX = True     # build FAISS index from DB
DO_DB_COMPLETE_REBUILD = True  # drop and recreate the DB
# ________________________________________________________

# util: load text from a pdf
def load_pdf(path):
    """
    Load text from a PDF file with security checks.

    Args:
        path: Path to the PDF file

    Returns:
        Tuple of (title, text)

    Raises:
        ValueError: If PDF is encrypted or cannot be read
    """
    text = ""
    with fitz.open(path) as doc: # use fitz function to open the pdf file
        # SECURITY: Skip encrypted PDFs instead of trying to decrypt them
        if doc.is_encrypted:
            logger.warning("Skipping encrypted PDF (security risk): %s", path)
            raise ValueError(f"PDF is encrypted and cannot be processed: {path}. "
                           "Please decrypt the file manually before ingestion.")

        for page in doc:
            text += page.get_text("text") + "\n"
        title = os.path.basename(path) # get the extisting doc's name
        if not text.strip():  # if the text is empty, raise an error
            raise ValueError(f"Failed to extract text from {path} - empty text")
    return title, text

# util: load text from a docx file
def load_docx(path):
    try:
        doc = Docx(path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        title = os.path.basename(path) # get the extisting doc's name
    except Exception as e:
        raise ValueError(f"Failed to read {path}: {e}")
    return title, text

# util: load text from a web page
def load_web(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ValueError(f"Failed to find&download {url}")
    text = trafilatura.extract(downloaded, include_comments=False) # removes ads etc
    if text is None:
        raise ValueError(f"Failed - no text on {url}")
    title = url.split("/")[-1] or "index"
    return title, text

# util: chunking the test
def chunks_by_paragraph(text, max_chars=300):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()] #strip spaces off the ends, split by newlines&spaces into paras
    Stored_chunks = []
    current_chunk = ""
    for p in paras:
        if len(current_chunk)+len(p)+2 <= max_chars: # if the current chunk is shorter than the max
            current_chunk = f"{current_chunk}\n\n{p}" if current_chunk else p # if the current chunk is not empty, add a newline before the next paragraph
        else:
            Stored_chunks.append(current_chunk); current_chunk = p # start a new chunk
    if current_chunk: Stored_chunks.append(current_chunk) # add the last chunk if it exists
    return Stored_chunks

# Create unique ID hash values to ensure load data is not duplicated and loaded correctly
def hashIDsum(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest() #SHA-1 format is 40chars

# ________________________________________________________

# add document to db
def insert_document(meta):
    conn = sqlite3.connect(DB_PATH); # connect to the db
    c = conn.cursor() # activate cursor
    #***check table format is correct
    c.execute("""INSERT INTO document( 
              source_type,
              path_or_url,
              title,
              origin,
              published_date,
              notes)
                 VALUES(?,?,?,?,?,?)""",
              (meta.get("source_type"), 
               meta.get("path_or_url"), 
               meta.get("title"),
               meta.get("origin") or None, 
               meta.get("published_date") or None, 
               meta.get("notes") or None))
    doc_id = c.lastrowid; conn.commit(); conn.close(); return doc_id


# add chunks of each doc to the db
def insert_chunks(doc_id: int, chunk_texts: list[str], db_path: str = str(DB_PATH)) -> int:
    # Batch chunk insert for a single document 
    # Returns number of inserts (duplicates are ignored by UNIQUE(hash)).
    
    rows = [] # empty array to hold rows of data to insert (chunks)
    for i, t in enumerate(chunk_texts): # ennumerate provides the index & text chunk
        if not t or not t.strip(): # check if empty
            continue
        rows.append((
            doc_id,
            i,
            t,
            len(t.split()),
            hashIDsum(t)
        ))

    if not rows:
        logging.info("No non-empty chunks to insert for document_id=%s", doc_id)
        return 0

    # ignore if duplicate chunk
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        with conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO chunk
                (document_id, chunk_index, text, tokens, hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows
            )
        cur = conn.execute("SELECT COUNT(*) FROM chunk WHERE document_id = ?", (doc_id,))
        total = cur.fetchone()[0] # run above statement, return the count
        logging.info("Chunks now stored for doc %s: %s", doc_id, total)
        return total
    finally:
        conn.close()


# ________________________________________________________

# build the vector index
def build_faiss_index():
    if DO_BUILD_INDEX:
        conn = sqlite3.connect(DB_PATH)
        # check if the chunks table has data
        try:
            df = pd.read_sql_query("SELECT id, text FROM chunk ORDER BY id", conn)
        except Exception as e:
            conn.close()
            print(f"❌ Error reading chunk table: {e}")
            raise SystemExit("Ensure the database is set up correctly.")
        conn.close()

        if df.empty: 
            raise SystemExit("No chunks found in the database. Please insert document and chunk first.")

        print(f"Embedding {len(df)} chunks...")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2") # employ a standard model for text embeddings
            embeds = model.encode(df["text"].tolist(), # encode the text into vector embeddings
                                show_progress_bar=True, convert_to_numpy=True) # convert to numpy array
            embeds = embeds.astype(np.float32) # FAISS needs float32
            # Calculate the Euclidean length of each embedding vector along axis=1 (per-row)
            # Adds a tiny value (1e-12) to prevent division by zero in case a vector has zero length:
            norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12 
            #  Normalise the embeddings to unit length (L2 normalisation). Normalised vectors allow dot product similarity to approximate cosine similarity:
            embeds = embeds / norms
            index = faiss.IndexFlatIP(embeds.shape[1]) # creates index search ti perform an exact *brute-force) search using inner product (IP) of vector between the 2 items as the distance, shape = #dimensions
            index.add(embeds) 
        except Exception as e:
            print(f"Embedding/index build failed: {e}")
            logger.error("Embedding/index build failed: %s", e)
            return       

        try:
            faiss.write_index(index, os.path.join(INDEX_DIR, "chunks.faiss")) # save to a file
            df[["id"]].to_csv(os.path.join(INDEX_DIR, "chunk_ids.csv"), index=False) # relates the chunk IDs from the chunk table to the embeddings IDs
        except Exception as e:
            print(f"Failed to write FAISS/ID files: {e}") 
            logger.error("Failed to write FAISS/ID files: %s", e)
        logger.info("FAISS index written to %s", INDEX_DIR / "chunks.faiss")
        print("FAISS index written safely to vectorstore/chunks.faiss")

# ________________________________________________________

# Main execution block
# This is the entry point for the script when run directly.
if __name__ == "__main__": # only do this if the .py file is called directly, not imported
    from src.PAPElogging_config import setup_logging
    setup_logging(log_to_file=True, log_dir=PROJECT_DIR / "logs")  # setup logging to file
    logger.info("Setting up the RAG database...")
    print("Setting up the RAG database...")
    assert_data_dir()  # ensure the data directory exists
   
    # ingest local PDFs / DOCX
    print("Ingesting local files...")
    logger.info("Ingesting local files...")
    files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        raise SystemExit(f"No PDF/DOCX found under: {DATA_DIR}")

    for path in files:
        try:
            # SECURITY: Validate file path to prevent path traversal attacks
            is_valid, error_msg = validate_file_path(path, DATA_DIR)
            if not is_valid:
                print(f"Security error: {error_msg}. Skipping.")
                logger.error("Path traversal attempt blocked: %s - %s", path, error_msg)
                continue

            ext = os.path.splitext(path)[1].lower() #find the ext of the document
            if not os.path.exists(path):
                print(f"File {path} does not exist. Skipping.")
                logger.warning("File %s does not exist. Skipping.", path)
                continue

            if ext == ".pdf": # ingest pdf files
                title, text = load_pdf(path)
                meta = {"source_type":"pdf",
                        "path_or_url":str(path), # convert to string for db
                        "title":title,
                        "origin":"Local PDF files"} 
            elif ext == ".docx": # ingest docx files
                title, text = load_docx(path)
                meta = {"source_type":"docx",
                        "path_or_url": str(path),
                        "title":title,
                        "origin":"Curated finance cases"}
            else:
                continue

            # basic cleaning, then chunk & write
            text = re.sub(r"[ \t]+", " ", text).strip() #replace spaces/tabs with one space, strip whitespace
            if not text:
                print(f"No text extracted from {path}. Skipping.")
                logger.warning("No text extracted from %s. Skipping.", path)
                continue
            print(type(meta["path_or_url"]), meta["path_or_url"]) # *** testing loading path_or_url correctly

            doc_id = insert_document(meta) # insert the document metadata into the db
            chunk_list = chunks_by_paragraph(text) 
            if not chunk_list:
                print(f"No chunks created from {path}. Skipping.")
                logger.warning("No chunks created from %s. Skipping.", path)
                continue

            insert_chunks(doc_id, chunk_list) # insert chunks into db
            print(f"Loaded {title} with {len(chunk_list)} chunks.") 
            logger.info("Loaded %s with %d chunks.", title, len(chunk_list))
        except Exception as e:
            raise SystemExit(f"Skipped {path} due to error: {e}")
            

    # ingest URLs 
    if DO_INGEST_WEB:
        ingested = ingest_web_urls(
            urls_file=URLS_FILE if URLS_FILE.exists() else None,
            docx_with_urls=DOCX_WITH_URLS if DOCX_WITH_URLS.exists() else None,
            insert_document=insert_document,
            insert_chunks=lambda doc_id, chunks: insert_chunks(doc_id, chunks),
            make_chunks=chunks_by_paragraph,
        )
        print(f"Ingested {ingested} web documents.")
        logger.info("Ingested %d web documents.", ingested)

    # build vector index
    if DO_BUILD_INDEX:
        print("Building FAISS index...")
        build_faiss_index()