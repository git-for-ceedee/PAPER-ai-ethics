# --------------------------------
# Pick-a-Path AI Ethics
# Ingest data into RAG db
# --------------------------------

# packages needed to ingest the data in several formats
import os, re, hashlib, sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
import fitz  # PyMuPDF for PDF handling
from docx import Document as Docx
import trafilatura, requests
from pathlib import Path
import time
import urllib.parse as urlparse
import urllib.robotparser as robotparser
import tldextract
import trafilatura
import requests
import re

# Locate the db
DB_PATH = "ragEthics.db"
INDEX_DIR = "vectorstore" # directory to be created to store the vector chunk db
os.makedirs(INDEX_DIR, exist_ok=True)
DATA_DIR = Path(r"C:\Users\Ceede\OneDrive\CD_VicUni\AIML339\Pick_a_path_AI_ethics\data\raw_cases") # r makes it a raw string to cope with /
DOCX_WITH_URLS = r"C:\Users\Ceede\OneDrive\CD_VicUni\AIML339\Pick_a_path_AI_ethics\data\raw_cases\AIML339_Sources_of_data.docx"
URLS_FILE = "urls.txt"  # plain text, one URL per line
SUPPORTED_EXTS = {".pdf", ".docx"}

# polite crawling
USER_AGENT = "EthicsStoryRAG/1.0 (+https://example.local)"
CRAWL_DELAY_S = 1.0

# (optional) restrict to trusted domains
# WHITELIST_SUFFIXES = (".govt.nz", ".org.nz", ".parliament.nz", ".justice.govt.nz", ".privacy.org.nz")


# ________________________________________________________

# util: load text from a pdf
def load_pdf(path):
    doc = fitz.open(path) # use fitz function to open the pdf file 
    if doc.is_encrypted:  # check if the PDF is encrypted
        doc.authenticate("")  # try to decrypt with an empty password
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    title = os.path.basename(path) # get the extisting doc's name
    if not text.strip():  # if the text is empty, raise an error
        raise ValueError(f"Failed to extract text from {path} - empty text")
    return title, text

# util: load text from a docx file
def load_docx(path):
    doc = Docx(path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    title = os.path.basename(path) # get the extisting doc's name
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
def chunks_by_paragraph(text, max_chars=1800):
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
def hashIDsum(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ________________________________________________________

# DB setup
def init_db():
    conn = sqlite3.connect(DB_PATH) # connect to the db
    c = conn.cursor() # activate cursor
    c.execute("""CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY, 
              source_type TEXT, 
              path_or_url TEXT, 
              title TEXT,
              origin TEXT, 
              published_date TEXT, 
              date_range TEXT, 
              sector TEXT, 
              notes TEXT
    )""") #source_type: pdf, docx, web, etc | origin: web links, etc   
    c.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, 
              document_id INTEGER, 
              chunk_index INTEGER,
              text TEXT, 
              section TEXT, 
              themes TEXT, 
              tokens INTEGER, 
              hash TEXT UNIQUE
    )""") #themes = ethics topics, tokens = number of tokens in the chunk
    conn.commit(); conn.close()

# add document to db
def insert_document(meta):
    conn = sqlite3.connect(DB_PATH); # connect to the db
    c = conn.cursor() # activate cursor
    c.execute("""INSERT INTO documents(source_type,path_or_url,title,origins,published_date,date_range,sector,notes)
                 VALUES(?,?,?,?,?,?,?,?)""",
              (meta.get("source_type"), meta.get("path_or_url"), meta.get("title"),
               meta.get("origin"), meta.get("published_date"), meta.get("date_range"), meta.get("sector"), meta.get("notes")))
    doc_id = c.lastrowid; conn.commit(); conn.close(); return doc_id

# add chunks of each doc to the db
def insert_chunks(doc_id, chunk_texts, section=None, themes=None):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    for i, t in enumerate(chunk_texts):
        try:
            c.execute("""INSERT OR IGNORE INTO chunks(document_id,chunk_index,text,section,themes,tokens,hash)
                         VALUES(?,?,?,?,?,?,?)""",
                      (doc_id, i, t, section, themes, len(t.split()), hashIDsum(t)))
        except sqlite3.IntegrityError:
            pass
    conn.commit(); conn.close()
# ________________________________________________________
# URL processing section
URL_REGEX = re.compile(r'(https?://[^\s)>\]]+)', re.IGNORECASE)

def normalise_url(u: str) -> str:
    p = urlparse.urlsplit(u.strip())
    p = p._replace(fragment="") #strip any bits that are not eh base URL
    if not p.scheme:  # if no scheme, assume http
        p = p._replace(scheme="http") 
    return urlparse.urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, p.query, ""))

def robots_ok(u: str, user_agent: str = USER_AGENT) -> bool:
    parts = urlparse.urlsplit(u)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt" #checks the url/robots.txt file
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, u)
    except Exception: # Many sites omit robots.txt — default allow
        return True

def get_domain(u: str) -> str: # makes a string out of the domiain and suffix 
                            # e.g. justice.gov.nz, so that everything is grouped under that domain, not tv.justice.govt nz beta.justice.govt.nz etc
    ext = tldextract.extract(u)
    return ".".join(part for part in [ext.domain, ext.suffix] if part)

# def url_allowed_by_whitelist(u: str) -> bool:
#    if not WHITELIST_SUFFIXES:  # empty tuple/list means allow all
#        return True
#    host = urlparse.urlsplit(u).netloc.lower()
#    return host.endswith(WHITELIST_SUFFIXES)


def fetch_and_extract(u: str) -> tuple[str, str]:
    headers = {"User-Agent": USER_AGENT} #let the website know who is asking
    resp = requests.get(u, headers=headers, timeout=20)
    resp.raise_for_status() # set an error if it returns 404 etc.
    html = resp.text

    # trafilatura extracts main content and strips boilerplate/ads
    cleaned = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=False, url=u)
    if cleaned: #is not empty
        meta = trafilatura.extract_metadata(html, url=u) or {}
        title = (meta.get("title") or u).strip()
        return title, cleaned.strip()
    # fallback if no content found it will return minimal info
    return u, html

def urls_from_docx(docx_path: str) -> list[str]:
    doc = Docx(docx_path)
    urls = set()

    # raw URLs in text
    for para in doc.paragraphs:
        for m in URL_REGEX.findall(para.text or ""): #find all the url's in the text or return empty string
            urls.add(normalise_url(m))

    # embedded hyperlinks via 'relationships' - stored in word docs as /hyperlink - embedded clickable links
    for rel in doc.part.rels.values():
        if rel.reltype.endswith("/hyperlink") and rel._target:
            tgt = str(rel._target)
            if tgt.lower().startswith(("http://", "https://")):
                urls.add(normalise_url(tgt))

    return sorted(urls)
# ________________________________________________________



# build the vector index
def build_faiss_index():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, text FROM chunks ORDER BY id", conn)
    conn.close()
    if df.empty: 
        print("No chunks found in the database. Please insert documents and chunks first.")
        return

    print(f"Embedding {len(df)} chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2") # employ a standard model for text embeddings
    embs = model.encode(df["text"].tolist(), # encode the text into vector embeddings
                        show_progress_bar=True, convert_to_numpy=True) # convert to numpy array
    index = faiss.IndexFlatL2(embs.shape[1]) # creates index search ti perform an exact *brute-force) search using euclidian (L2) distance, shape = #dimensions
    index.add(embs.astype(np.float32)) # faiss requires float32,  so convert the embeddings to float32
    faiss.write_index(index, os.path.join(INDEX_DIR, "chunks.faiss")) # save to a file
    df[["id"]].to_csv(os.path.join(INDEX_DIR, "chunk_ids.csv"), index=False) # relates the chunk IDs from the chunks table to the embeddings IDs
    print("✅ FAISS index written to vectorstore/chunks.faiss")

if __name__ == "__main__": # only do this is the .py file is called directly, not imported
    print("Setting up the RAG database...")
    init_db()

    # ingest local PDFs / DOCX
    print("Ingesting local files...")
    files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        print(f"No PDF/DOCX found under: {DATA_DIR}")
        raise SystemExit

    for path in files:
        ext = os.path.splitext(path)[1].lower() #find the ext of the document
        if not os.path.exists(path):
            print(f"File {path} does not exist. Skipping.")
            continue
        if ext == ".pdf": # ingest pdf files
            title, text = load_pdf(path)
            meta = {"source_type":"pdf",
                    "path_or_url":path,
                    "title":title,
                    "origin":"NZ Govt/AI Forum"} # ***
        elif ext == ".docx": # ingest docx files
            title, text = load_docx(path)
            meta = {"source_type":"docx","path_or_url":path,"title":title,"origin":"Curated finance cases"}
        else:
            continue

        # basic cleaning, then chunk & write
        text = re.sub(r"[ \t]+", " ", text).strip() #replace spaces/tabs with one space, strip whitespace
        doc_id = insert_document(meta) # insert the document metadata into the db
        insert_chunks(doc_id, chunks_by_paragraph(text)) # insert the chunks into the db
        print(f"Loaded {title} with {len(chunks_by_paragraph(text))} chunks.") 

# _______________________________________
# Run the web scraping and URL processing
    all_urls = set()
    print("Ingesting URLs...")

# ingest URLs from the .txt file
    if os.path.exists(URLS_FILE):
        print(f"Ingesting URLs from {URLS_FILE} ...")
        with open(URLS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    all_urls.add(normalise_url(s))
    else:
        print("No urls.txt found; skipping text URL list.")

# ingest from the DOCX file with URLs
    if DOCX_WITH_URLS and os.path.exists(DOCX_WITH_URLS):
        print(f"Extracting URLs from DOCX: {DOCX_WITH_URLS}")
        extracted = urls_from_docx(DOCX_WITH_URLS)
        print(f"Found {len(extracted)} URLs in DOCX.")
        all_urls.update(extracted)
    else:
        print("No DOCX URL source found; skipping DOCX URL ingestion.")

# ingest unique URLs from the file
    last_domain = None
    for raw_url in sorted(all_urls):
        url = normalise_url(raw_url)

        if not url.startswith(("http://", "https://")):
            print(f"Skipping non-http URL: {url}")
            continue
        #if not url_allowed_by_whitelist(url):
        #    print(f"Skipping non-whitelisted domain: {url}")
        #    continue
        if not robots_ok(url):
            print(f"Disallowed by robots.txt: {url}")
            continue

        # polite delay between domains
        dom = get_domain(url)
        if last_domain and dom != last_domain:
            time.sleep(CRAWL_DELAY_S)
        last_domain = dom

        try:
            title, text = fetch_and_extract(url)
            if not text or len(text) < 400:
                print(f"Too little extractable text at: {url}")
                continue

            text = re.sub(r"[ \t]+", " ", text).strip()
            meta = {
                "source_type": "web",
                "path_or_url": url,
                "title": title[:300],
                "origin": dom,
                "published_date": None,
                "sector": None,
                "notes": "ingested-from-urls"
            }
            doc_id = insert_document(meta)
            chunk_list = chunks_by_paragraph(text)
            insert_chunks(doc_id, chunk_list)
            print(f"Ingested URL OK : {url} ({len(chunk_list)} chunks)")
        except Exception as e:
            print(f"Skipped {url}: {e}")

    # build vector index
    build_faiss_index()