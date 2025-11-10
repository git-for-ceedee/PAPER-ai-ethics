# --------------------------------
# Pick-a-Path AI Ethics
# Create db 
# --------------------------------

# ________________________________________________________
# This script sets up a database for the Pick-a-Path AI Ethics project

# packages needed to ingest the data in several formats
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
# from src.PAPEingest_web_sources import ingest_web_urls
from docx import Document as Docx
from pathlib import Path
import logging
from src.PAPElogging_config import setup_logging  # centralised logging config

PROJECT_DIR = Path(__file__).resolve().parent.parent  # the parent directory of this file
DB_PATH = PROJECT_DIR / "ragEthics.db"
DATA_DIR = PROJECT_DIR / "data" / "raw_cases"
INDEX_DIR = PROJECT_DIR / "vectorstore"
INDEX_DIR.mkdir(parents=True, exist_ok=True)  # ensure the index directory exists
DO_DB_COMPLETE_REBUILD = False  # set to True to drop and recreate tables

# ________________________________________________________
# Standard logging header 
logger = logging.getLogger(__name__)

# ________________________________________________________


# DB setup
def assert_data_dir(): # check the directory exists
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Tip: put PDFs/DOCX under data/raw_cases or update DATA_DIR variable.")
        logger.warning("Data directory not found: %s", DATA_DIR)

def init_db():
    # Delete and recreate DB file if doing a full rebuild
    if DO_DB_COMPLETE_REBUILD and DB_PATH.exists():
        logging.warning("Deleting existing DB: %s", DB_PATH)
        DB_PATH.unlink() # delete the existing db file
    
    
    conn = sqlite3.connect(DB_PATH) # connect to the db
    conn.execute("PRAGMA foreign_keys = ON;") # FK enforcement has to be turned on in sqlite
    conn.execute("PRAGMA journal_mode = WAL;")  # nicer concurrency for read/write
    
    with conn:  # 'with' allows db rollback automatically if commit fails
        c = conn.cursor() # activate cursor in the db
                        

        # =========================
        # Tables (entities & lookups)
        # =========================

        c.execute("""CREATE TABLE IF NOT EXISTS sector (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL UNIQUE COLLATE NOCASE
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS actor (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL UNIQUE COLLATE NOCASE,
            kind      TEXT               
        )""") # kind e.g., 'regulator','bank','insurer','govt agency','platform', 'ngo'

        c.execute("""CREATE TABLE IF NOT EXISTS theme (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL UNIQUE COLLATE NOCASE
        )""") #  broader topical categories to analyse or filter by (e.g., privacy, fairness, transparency, Māori Data Sovereignty, employment impact)

        c.execute("""CREATE TABLE IF NOT EXISTS ethical_issue (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL UNIQUE COLLATE NOCASE
        )""") #  specific issues or principles explicitly raised (e.g., consent, data minimisation, human oversight)
              # Often map more directly to frameworks/charters or concrete compliance points

        c.execute("""CREATE TABLE IF NOT EXISTS source (
            id        INTEGER PRIMARY KEY,
            title     TEXT,           
            publisher TEXT,           
            year      INTEGER,
            url       TEXT,
            citation  TEXT              
        )""") # publisher e.g. 'Stats NZ','OPC','RBNZ','IFSO', citation = full citation if exists

        c.execute("""CREATE TABLE IF NOT EXISTS mechanism (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL UNIQUE COLLATE NOCASE
        )""") # e.g. 'automated risk rating'

        c.execute("""CREATE TABLE IF NOT EXISTS outcome (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL UNIQUE COLLATE NOCASE       
        )""") 

        # =========================
        # Cases (top-level entity)
        # =========================

        c.execute("""CREATE TABLE IF NOT EXISTS ai_case (
            id                INTEGER PRIMARY KEY,
            code              TEXT UNIQUE,      
            title             TEXT NOT NULL,
            summary           TEXT,             
            sector_id         INTEGER,    
            date_start_year  INTEGER CHECK (date_start_year  BETWEEN 1800 AND 2100),
            date_start_month INTEGER CHECK (date_start_month BETWEEN 1 AND 12)  NULL,
            date_start_day   INTEGER CHECK (date_start_day   BETWEEN 1 AND 31)  NULL,
            date_end_year    INTEGER CHECK (date_end_year    BETWEEN 1800 AND 2100),
            date_end_month   INTEGER CHECK (date_end_month   BETWEEN 1 AND 12)  NULL,
            date_end_day     INTEGER CHECK (date_end_day     BETWEEN 1 AND 31)  NULL,           
            ethical_choice    TEXT,             
            intended_goal     TEXT,             
            confidence_score  REAL,            
            notes             TEXT,
            FOREIGN KEY (sector_id) REFERENCES sector(id) ON DELETE CASCADE
        )""") # summary = short overview of the case, dates YYYY-MM-DD if known, confidernce_score = confidence in the case details (0.0-1.0)

        # =========================
        # Linking tables (many-to-many relationships)
        # =========================
        c.execute("""CREATE TABLE IF NOT EXISTS ai_case_mechanism (
            ai_case_id   INTEGER NOT NULL,
            mechanism_id INTEGER NOT NULL,
            detail       TEXT,                
            PRIMARY KEY (ai_case_id, mechanism_id),
            FOREIGN KEY (ai_case_id) REFERENCES ai_case(id) ON DELETE CASCADE,
            FOREIGN KEY (mechanism_id) REFERENCES mechanism(id) ON DELETE CASCADE
        )""")  # detail = verbatim line from case is available (richer text)

        # =========================
        c.execute("""CREATE TABLE IF NOT EXISTS ai_case_outcome (
            id            INTEGER PRIMARY KEY,
            ai_case_id    INTEGER NOT NULL,
            outcome_id    INTEGER,             
            polarity      TEXT NOT NULL CHECK (polarity IN ('good','bad','mixed','ongoing', 'unresolved')),
            detail        TEXT NOT NULL,       
            FOREIGN KEY (ai_case_id) REFERENCES ai_case(id) ON DELETE CASCADE,
            FOREIGN KEY (outcome_id) REFERENCES outcome(id) ON DELETE CASCADE
        )""")
        
        # =========================
        c.execute("""CREATE TABLE IF NOT EXISTS ai_case_theme (
            ai_case_id  INTEGER NOT NULL,
            theme_id    INTEGER NOT NULL,
            PRIMARY KEY (ai_case_id, theme_id),
            FOREIGN KEY (ai_case_id) REFERENCES ai_case(id) ON DELETE CASCADE,
        FOREIGN KEY (theme_id) REFERENCES theme(id) ON DELETE CASCADE
        )""")

        # =========================
        c.execute("""CREATE TABLE IF NOT EXISTS ai_case_issue (
            ai_case_id     INTEGER NOT NULL,
            ethical_issue_id INTEGER NOT NULL,
            PRIMARY KEY (ai_case_id, ethical_issue_id),
            FOREIGN KEY (ai_case_id) REFERENCES ai_case(id) ON DELETE CASCADE,
            FOREIGN KEY (ethical_issue_id) REFERENCES ethical_issue(id) ON DELETE CASCADE
        )""")

        # =========================
        c.execute("""CREATE TABLE IF NOT EXISTS ai_case_actor (
            ai_case_id  INTEGER NOT NULL,
            actor_id    INTEGER NOT NULL,
            role        TEXT,                  
            notes       TEXT,                  
            PRIMARY KEY (ai_case_id, actor_id),
            FOREIGN KEY (ai_case_id) REFERENCES ai_case(id) ON DELETE CASCADE,
            FOREIGN KEY (actor_id) REFERENCES actor(id) ON DELETE CASCADE
        )""") # role e.g., 'implementer','regulator','affected group', notes are optional

        # =========================
        c.execute("""CREATE TABLE IF NOT EXISTS ai_case_source (
            ai_case_id  INTEGER NOT NULL,
            source_id   INTEGER NOT NULL,
            evidence    TEXT,                  
            PRIMARY KEY (ai_case_id, source_id),
            FOREIGN KEY (ai_case_id) REFERENCES ai_case(id) ON DELETE CASCADE,
            FOREIGN KEY (source_id) REFERENCES source(id) ON DELETE CASCADE
        )""") # evidence: quote/notes on how the source supports the case if available

        # =========================
        # RAG layer: documents & chunks
        # =========================
        c.execute("""CREATE TABLE IF NOT EXISTS document (
            id            INTEGER PRIMARY KEY,
            source_type   TEXT NOT NULL,       
            path_or_url   TEXT NOT NULL UNIQUE,
            title         TEXT,
            origin        TEXT,               
            published_date TEXT,
            notes         TEXT
        )""") # source e.g. 'pdf','docx','web','db', origin e.g., host domain or organisation

        c.execute("""CREATE TABLE IF NOT EXISTS chunk (
            id            INTEGER PRIMARY KEY,
            document_id   INTEGER NOT NULL,
            chunk_index   INTEGER NOT NULL,
            text          TEXT NOT NULL,
            tokens        INTEGER, 
            hash          TEXT UNIQUE,
            FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE
        )""") # tokens = number of tokens in the chunk
        
        c.execute("""CREATE TABLE IF NOT EXISTS chunk_theme (
            chunk_id INTEGER NOT NULL,
            theme_id INTEGER NOT NULL,
            PRIMARY KEY (chunk_id, theme_id),
            FOREIGN KEY (chunk_id) REFERENCES chunk(id) ON DELETE CASCADE,
            FOREIGN KEY (theme_id) REFERENCES theme(id) ON DELETE CASCADE
        )""")

        # link chunks back to cases (provenance)
        c.execute("""CREATE TABLE IF NOT EXISTS ai_case_chunk (
            ai_case_id  INTEGER NOT NULL,
            chunk_id    INTEGER NOT NULL,
            relation    TEXT,                  
            PRIMARY KEY (ai_case_id, chunk_id),
            FOREIGN KEY (ai_case_id) REFERENCES ai_case(id) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id) REFERENCES chunk(id) ON DELETE CASCADE
        )""") # relation e.g., 'case-description','evidence','mechanism','outcome'

        # =========================
        # indices for speed
        # =========================
        
        c.execute("""CREATE INDEX IF NOT EXISTS idx_chunk_doc        ON chunk(document_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_chunk_theme_chunk ON chunk_theme(chunk_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_chunk_theme_theme ON chunk_theme(theme_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_sector      ON ai_case(sector_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_start_y   ON ai_case(date_start_year)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_end_y     ON ai_case(date_end_year)""")
        # optional for if filtering by months/days
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_start_ym  ON ai_case(date_start_year, date_start_month)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_end_ym    ON ai_case(date_end_year,   date_end_month)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_conf        ON ai_case(confidence_score)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_document_origin  ON document(origin)""")
        # including junction tables
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_theme_case  ON ai_case_theme(ai_case_id)""");
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_theme_theme ON ai_case_theme(theme_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_issue_case  ON ai_case_issue(ai_case_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_issue_issue ON ai_case_issue(ethical_issue_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_actor_case  ON ai_case_actor(ai_case_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_actor_actor ON ai_case_actor(actor_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_mech_case   ON ai_case_mechanism(ai_case_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_outcome_case ON ai_case_outcome(ai_case_id)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_case_source_case ON ai_case_source(ai_case_id)""")

    conn.commit(); 
    conn.close()
    logger.info("Database initialised at %s", DB_PATH)
    print("Database initialised.")

# ---------- Entry point ----------
if __name__ == "__main__":
    setup_logging(log_to_file=True, log_dir=PROJECT_DIR / "logs")
    logger.info("Starting DB setup…")
    assert_data_dir()   # optional
    init_db()

