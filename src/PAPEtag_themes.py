# --------------------------------
# Pick-a-Path AI Ethics — Theme Tagger (rules-based)
# --------------------------------


from pathlib import Path
import re, sqlite3, logging
from src.PAPElogging_config import setup_logging

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "ragEthics.db"
logger = logging.getLogger(__name__)

# NZ-relevant, lightweight patterns (you can tune later)
THEMES = {
  "accountability": [r"accountability", r"responsibility", r"accountable", r"accounting"],
  "employment_impact": [r"employment", r"workforce", r"displacement", r"reskilling"],
  "ethical_wellbeing": [r"ethical wellbeing", r"wellbeing", r"hauora"],
  "fairness":       [r"\bfair(ness)?\b", r"\bbias(ed|es)?\b", r"discriminat", r"\bequity\b"],
  "human_oversight":[r"human\s*oversight", r"human.?in.?the.?loop", r"human\s*review"],
  "maori_data":     [r"M[āa]ori", r"CARE\s*Principles", r"Te\s*Tiriti", r"data\s*sovereignty"],
  "privacy":        [r"\bprivacy\b", r"\bIPP(s)?\b", r"\bconsent\b", r"de-?identif", r"\bP(\.?|rivacy)\s*Act\s*2020\b"],
  "transparency":   [r"transparen", r"explainab", r"interpretab", r"auditab", r"\breason\b"],
  "security":       [r"\bsecurity\b", r"\bcyber\b", r"\bencryption\b", r"\bbreach\b"],
  "wellbeing":      [r"well-?being", r"\bhauora\b", r"\bwellbeing\b"],
}

# --------------------------------
def upsert_theme(conn, name: str) -> int:
    conn.execute("INSERT OR IGNORE INTO theme(name) VALUES (?)", (name,))
    return conn.execute("SELECT id FROM theme WHERE name = ?", (name,)).fetchone()[0]

# --------------------------------
def ensure_chunk_theme_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS chunk_theme (
      chunk_id INTEGER NOT NULL,
      theme_id INTEGER NOT NULL,
      PRIMARY KEY (chunk_id, theme_id),
      FOREIGN KEY (chunk_id) REFERENCES chunk(id) ON DELETE CASCADE,
      FOREIGN KEY (theme_id) REFERENCES theme(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_chunk_theme_chunk ON chunk_theme(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_chunk_theme_theme ON chunk_theme(theme_id);
    """)

# --------------------------------
# Main function
# --------------------------------
def main():
    setup_logging(log_to_file=True, log_dir=PROJECT_DIR / "logs")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON;")
    with conn:
        ensure_chunk_theme_table(conn)
        # make sure themes exist
        theme_ids = {name: upsert_theme(conn, name) for name in THEMES.keys()}
        # fetch chunks
        rows = conn.execute("SELECT id, text FROM chunk").fetchall()
        n_links = 0
        for cid, txt in rows: # chunkID & text
            low = (txt or "").lower()
            for name, patterns in THEMES.items(): # theme name & word search patterns
                if any(re.search(p, low) for p in patterns):
                    conn.execute(
                        "INSERT OR IGNORE INTO chunk_theme(chunk_id, theme_id) VALUES (?, ?)",
                        (cid, theme_ids[name])
                    )
                    n_links += 1
        logger.info("Applied %d chunk→theme links", n_links)
        # small summary
        for name, tid in theme_ids.items():
            cnt = conn.execute(
                "SELECT COUNT(*) FROM chunk_theme WHERE theme_id = ?", (tid,)
            ).fetchone()[0]
            logger.info("Theme '%s': %d chunks", name, cnt)

# --------------------------------
if __name__ == "__main__":
    main()