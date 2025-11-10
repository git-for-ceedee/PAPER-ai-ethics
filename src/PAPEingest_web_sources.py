# --------------------------------
# Pick-a-Path AI Ethics
# Ingest URLs data into RAG db
# --------------------------------

# src/ingest_web_sources.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable, Sequence
import re, time
import urllib.parse as urlparse
import urllib.robotparser as robotparser
import requests
import trafilatura
import tldextract
from docx import Document as Docx
from bs4 import BeautifulSoup
import logging
from src.PAPElogging_config import setup_logging
from src.PAPEsecurity import validate_url, sanitise_url


# ________________________________________________________
# Web crawling constants (kept local to avoid circular imports)
USER_AGENT = "EthicsStoryRAG/1.0"
CRAWL_DELAY_S = 1.0
URL_REGEX = re.compile(r"(https?://[^\s)>\]]+)", re.IGNORECASE)
PROJECT_DIR = Path(__file__).resolve().parent.parent  # the parent directory of this file

# ________________________________________________________
# Standard logging header 
logger = logging.getLogger(__name__)
# ________________________________________________________

# ________________________________________________________
# Functions for URL processing and web scraping
def normalise_url(u: str) -> str:
    p = urlparse.urlsplit(u.strip())
    p = p._replace(fragment="") #strip any bits that are not eh base URL
    if not p.scheme:  # if no scheme, assume https
        p = p._replace(scheme="https") 
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

def read_urls(path: Path) -> Iterable[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    yield s
    except FileNotFoundError:
        return []

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

# def fetch_text(url: str) -> str | None:
#     try:
#         r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
#         if not r.ok:
#             return None
#         text = trafilatura.extract(r.text)
#         return text.strip() if text else None
#     except requests.RequestException:
#         return None

def ingest_web_urls(
    urls_file: Path | None,
    docx_with_urls: Path | None,
    insert_document: Callable[[dict], int],
    insert_chunks: Callable[[int, Sequence[str]], None],
    make_chunks: Callable[[str], Sequence[str]],
) -> int:
    
    # Read URLs from urls_file and/or docx_with_urls, fetch text, and write via callbacks.
    # Returns count of successfully ingested URLs.
    
    # gather URLs
    urls: set[str] = set()
    if urls_file:
        urls.update(read_urls(Path(urls_file)))
    if docx_with_urls and Path(docx_with_urls).exists():
        urls.update(urls_from_docx(Path(docx_with_urls)))
    if not urls:
        logger.warning("No URLs found in the provided sources.")
        return 0

    ingested = 0
    for raw in sorted(urls):
        url = normalise_url(raw)
        if not url.startswith(("http://", "https://")):
            logger.warning("Skipping URL with invalid scheme: %s", url)
            continue

        # SECURITY: Validate URL to prevent SSRF attacks
        is_valid, error_msg = validate_url(url, allow_private_ips=False)
        if not is_valid:
            logger.warning("Skipping invalid/unsafe URL: %s - %s", url, error_msg)
            time.sleep(CRAWL_DELAY_S)
            continue

        # Sanitise URL
        url = sanitise_url(url)

        if not robots_ok(url):
            logger.info("Skipping URL blocked by robots.txt: %s", url)
            time.sleep(CRAWL_DELAY_S)
            continue

        try:
            title, extracted = fetch_and_extract(url) 
        except Exception as e:
            logger.error(f"Failed to fetch for %s: %s, {url}: {e}")
            time.sleep(CRAWL_DELAY_S)
            continue
        if not extracted or len(extracted) < 400:
            time.sleep(CRAWL_DELAY_S)
            continue

        meta = {
            "source_type": "url",
            "path_or_url":  str(url),  # convert to string for db
            "title": title[:300],
            "origin": get_domain(url),
            "published_date": None,
            "notes": "ingested-from-urls",
        }
        doc_id = insert_document(meta)
        insert_chunks(doc_id, make_chunks(extracted))
        ingested += 1
        time.sleep(CRAWL_DELAY_S)
    return ingested

# ________________________________________________________
# URL processing section

def get_domain(u: str) -> str: # makes a string out of the domiain and suffix 
                            # e.g. justice.gov.nz, so that everything is grouped under that domain, not tv.justice.govt nz beta.justice.govt.nz etc
    ext = tldextract.extract(u)
    return ".".join(part for part in [ext.domain, ext.suffix] if part)

def fetch_and_extract(u: str) -> tuple[str, str]:
    # Fetch a URL and return (title, cleaned_text). Raises RuntimeError on fetch errors.
    # Returns (u, None) if content extraction fails.
    headers = {"User-Agent": USER_AGENT} #let the website know who is asking

    try:
        downloadedUrl = trafilatura.fetch_url(u, no_ssl=False)
    except Exception as e:
        # Network/SSL/etc. Fetch failed
        raise RuntimeError(f"fetch_url failed: {e}")

    if not downloadedUrl:
        # e.g., robots.txt disallowed, paywalled, or transient block
        logger.warning("No content downloaded (robots/paywall/blocked?): %s", u)
        return u, None

    # trafilatura extracts main content and strips boilerplate/ads
    try:
        cleaned = trafilatura.extract(downloadedUrl, 
                                    include_comments=False, 
                                    include_tables=False, 
                                    favor_recall=False, 
                                    )
    except Exception as e:
        logger.warning("extract() failed for %s: %s", u, e)
        return u, None

    # Title: try trafilatura metadata, then fall back to <title>, then URL
    title = None
    try:
        meta = trafilatura.extract_metadata(downloadedUrl) or {}
        title = (meta.get("title") or "").strip() or None
    except Exception:
        pass

    if not title:
        try:
            soup = BeautifulSoup(downloadedUrl, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            title = None

    title = title or u

    if not cleaned:
        logger.warning("No content extracted from %s", u)
        return title, None

    return title, cleaned.strip()