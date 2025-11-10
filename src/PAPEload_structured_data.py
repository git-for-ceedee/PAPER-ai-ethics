# --------------------------------
# Pick-a-Path AI Ethics
# Load structured CSV data into database
# --------------------------------

import sqlite3
import pandas as pd
from pathlib import Path
import logging
from src.PAPElogging_config import setup_logging

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "ragEthics.db"
DATA_DIR = PROJECT_DIR / "data" / "datafiles"

# Standard logging header 
logger = logging.getLogger(__name__)

def load_structured_data():
    """Load all CSV data into the database"""
    
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}. Run PAPEsetup_database.py first.")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    
    try:
        with conn:
            # Load lookup tables first (no dependencies)
            logger.info("Loading lookup tables...")
            load_sectors(conn)
            load_actors(conn)
            load_themes(conn)
            load_mechanisms(conn)
            load_ethical_issues(conn)
            load_sources(conn)
            load_outcomes(conn)
            
            # Load main AI cases
            logger.info("Loading AI cases...")
            load_ai_cases(conn)
            
            # Load relationship tables
            logger.info("Loading relationship tables...")
            load_case_actors(conn)
            load_case_themes(conn)
            load_case_mechanisms(conn)
            load_case_issues(conn)
            load_case_sources(conn)
            load_case_outcomes(conn)
            
            logger.info("✅ All structured data loaded successfully!")
            return True
            
    except Exception as e:
        logger.error(f"Error loading structured data: {e}")
        return False
    finally:
        conn.close()

def load_sectors(conn):
    """Load sectors from CSV"""
    csv_path = DATA_DIR / "PAPEsector.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
        for _, row in df.iterrows():
            conn.execute("INSERT OR IGNORE INTO sector (id, name) VALUES (?, ?)", 
                        (int(row['id']), str(row['name'])))
        logger.info(f"Loaded {len(df)} sectors")

def load_actors(conn):
    """Load actors from CSV"""
    csv_path = DATA_DIR / "PAPEactor.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("INSERT OR IGNORE INTO actor (id, name, kind) VALUES (?, ?, ?)", 
                        (int(row['id']), str(row['name']), str(row.get('kind', None)) if pd.notna(row.get('kind')) else None))
        logger.info(f"Loaded {len(df)} actors")

def load_themes(conn):
    """Load themes from CSV"""
    csv_path = DATA_DIR / "PAPEtheme.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("INSERT OR IGNORE INTO theme (id, name) VALUES (?, ?)", 
                        (int(row['id']), str(row['name'])))
        logger.info(f"Loaded {len(df)} themes")

def load_mechanisms(conn):
    """Load mechanisms from CSV"""
    csv_path = DATA_DIR / "PAPEmechanism.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("INSERT OR IGNORE INTO mechanism (id, name) VALUES (?, ?)", 
                        (int(row['id']), str(row['name'])))
        logger.info(f"Loaded {len(df)} mechanisms")

def load_ethical_issues(conn):
    """Load ethical issues from CSV"""
    csv_path = DATA_DIR / "PAPEethical_issue.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("INSERT OR IGNORE INTO ethical_issue (id, name) VALUES (?, ?)", 
                        (int(row['id']), str(row['name'])))
        logger.info(f"Loaded {len(df)} ethical issues")

def load_sources(conn):
    """Load sources from CSV"""
    csv_path = DATA_DIR / "PAPEsource.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("INSERT OR IGNORE INTO source (id, title, publisher, year, url, citation) VALUES (?, ?, ?, ?, ?, ?)", 
                        (int(row['id']), 
                         str(row['title']) if pd.notna(row.get('title')) else None, 
                         str(row['publisher']) if pd.notna(row.get('publisher')) else None, 
                         int(row['year']) if pd.notna(row.get('year')) else None,
                         str(row.get('url', None)) if pd.notna(row.get('url')) else None, 
                         str(row.get('citation', None)) if pd.notna(row.get('citation')) else None))
        logger.info(f"Loaded {len(df)} sources")

def load_outcomes(conn):
    """Load outcomes from CSV"""
    csv_path = DATA_DIR / "PAPEoutcome.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("INSERT OR IGNORE INTO outcome (id, name) VALUES (?, ?)", 
                        (int(row['id']), str(row['name'])))
        logger.info(f"Loaded {len(df)} outcomes")

def load_ai_cases(conn):
    """Load AI cases from CSV"""
    csv_path = DATA_DIR / "PAPEai_case.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Parse dates - handle YYYY-MM format
            date_start_year = None
            date_start_month = None
            date_end_year = None
            date_end_month = None
            
            if pd.notna(row.get('date_start')):
                date_parts = str(row['date_start']).split('-')
                if len(date_parts) >= 1:
                    date_start_year = int(date_parts[0])
                if len(date_parts) >= 2:
                    date_start_month = int(date_parts[1])
            
            if pd.notna(row.get('date_end')):
                date_parts = str(row['date_end']).split('-')
                if len(date_parts) >= 1:
                    date_end_year = int(date_parts[0])
                if len(date_parts) >= 2:
                    date_end_month = int(date_parts[1])
            
            conn.execute("""INSERT OR IGNORE INTO ai_case 
                (id, code, title, summary, sector_id, 
                 date_start_year, date_start_month, 
                 date_end_year, date_end_month,
                 ethical_choice, intended_goal, confidence_score, notes) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                (int(row['id']), str(row['code']), str(row['title']), str(row['summary']), 
                 int(row.get('sector_id')) if pd.notna(row.get('sector_id')) else None,
                 date_start_year, date_start_month, date_end_year, date_end_month,
                 str(row.get('ethical_choice')) if pd.notna(row.get('ethical_choice')) else None, 
                 str(row.get('intended_goal')) if pd.notna(row.get('intended_goal')) else None, 
                 float(row.get('confidence_score')) if pd.notna(row.get('confidence_score')) else None, 
                 str(row.get('notes')) if pd.notna(row.get('notes')) else None))
        logger.info(f"Loaded {len(df)} AI cases")

def load_case_actors(conn):
    """Load case-actor relationships"""
    csv_path = DATA_DIR / "PAPEai_case_actor.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("""INSERT OR IGNORE INTO ai_case_actor 
                (ai_case_id, actor_id, role) VALUES (?, ?, ?)""", 
                (int(row['ai_case_id']), int(row['actor_id']), str(row.get('role', None)) if pd.notna(row.get('role')) else None))
        logger.info(f"Loaded {len(df)} case-actor relationships")

def load_case_themes(conn):
    """Load case-theme relationships"""
    csv_path = DATA_DIR / "PAPEai_case_theme.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("""INSERT OR IGNORE INTO ai_case_theme 
                (ai_case_id, theme_id) VALUES (?, ?)""", 
                (int(row['ai_case_id']), int(row['theme_id'])))
        logger.info(f"Loaded {len(df)} case-theme relationships")

def load_case_mechanisms(conn):
    """Load case-mechanism relationships"""
    csv_path = DATA_DIR / "PAPEai_case_mechanism.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("""INSERT OR IGNORE INTO ai_case_mechanism 
                (ai_case_id, mechanism_id, detail) VALUES (?, ?, ?)""", 
                (int(row['ai_case_id']), int(row['mechanism_id']), str(row.get('detail', None)) if pd.notna(row.get('detail')) else None))
        logger.info(f"Loaded {len(df)} case-mechanism relationships")

def load_case_issues(conn):
    """Load case-ethical issue relationships"""
    csv_path = DATA_DIR / "PAPEai_case_issue.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("""INSERT OR IGNORE INTO ai_case_issue 
                (ai_case_id, ethical_issue_id) VALUES (?, ?)""", 
                (int(row['ai_case_id']), int(row['ethical_issue_id'])))
        logger.info(f"Loaded {len(df)} case-issue relationships")

def load_case_sources(conn):
    """Load case-source relationships"""
    csv_path = DATA_DIR / "PAPEai_case_source.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("""INSERT OR IGNORE INTO ai_case_source 
                (ai_case_id, source_id) VALUES (?, ?)""", 
                (int(row['ai_case_id']), int(row['source_id'])))
        logger.info(f"Loaded {len(df)} case-source relationships")

def load_case_outcomes(conn):
    """Load case-outcome relationships"""
    csv_path = DATA_DIR / "PAPEai_case_outcome.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            conn.execute("""INSERT OR IGNORE INTO ai_case_outcome 
                (ai_case_id, outcome_id) VALUES (?, ?)""", 
                (int(row['ai_case_id']), int(row['outcome_id'])))
        logger.info(f"Loaded {len(df)} case-outcome relationships")

def verify_data_loaded(conn):
    """Verify that data was loaded correctly"""
    tables_to_check = [
        ('sector', 'sectors'),
        ('actor', 'actors'), 
        ('theme', 'themes'),
        ('mechanism', 'mechanisms'),
        ('ethical_issue', 'ethical issues'),
        ('source', 'sources'),
        ('outcome', 'outcomes'),
        ('ai_case', 'AI cases')
    ]
    
    print("\n📊 Data Loading Summary:")
    print("=" * 50)
    
    for table, description in tables_to_check:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{description.capitalize()}: {count}")
    
    # Check relationships
    relationship_tables = [
        ('ai_case_actor', 'case-actor relationships'),
        ('ai_case_theme', 'case-theme relationships'),
        ('ai_case_mechanism', 'case-mechanism relationships'),
        ('ai_case_issue', 'case-issue relationships'),
        ('ai_case_source', 'case-source relationships'),
        ('ai_case_outcome', 'case-outcome relationships')
    ]
    
    print("\n🔗 Relationship Data:")
    print("-" * 30)
    for table, description in relationship_tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{description.capitalize()}: {count}")
        except:
            print(f"{description.capitalize()}: 0 (table not found)")

if __name__ == "__main__":
    setup_logging(log_to_file=True, log_dir=PROJECT_DIR / "logs")
    logger.info("Starting structured data loading...")
    
    success = load_structured_data()
    
    if success:
        # Verify the data was loaded
        conn = sqlite3.connect(DB_PATH)
        verify_data_loaded(conn)
        conn.close()
        
        print("\n[SUCCESS] Structured data loading completed successfully!")
        print("You can now run the UI to see your AI ethics cases in action.")
    else:
        print("\n[ERROR] Data loading failed. Check the logs for details.")
