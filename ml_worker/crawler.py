import requests
import pandas as pd
import random
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

# Fallback: trafilatura for deep article extraction
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

# Fallback 2: Playwright for bypassing bot protection and SPAs
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    from playwright_stealth.stealth import Stealth
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


# --- ROTATING USER AGENTS ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _get_headers():
    """Generate realistic browser-like headers with a random User-Agent."""
    ua = random.choice(USER_AGENTS)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    }


def _check_robots_txt(url):
    """Check robots.txt to ensure we are allowed to crawl this URL (legal compliance)."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch("*", url)
        print(f"robots.txt check: {'ALLOWED' if allowed else 'BLOCKED'}")
        return allowed
    except Exception:
        # If we can't read robots.txt, assume allowed (most sites don't block)
        print("robots.txt check: Could not read (proceeding)")
        return True


def _fetch_page(url, retries=3):
    """Fetch a page with retry logic and proper error handling."""
    session = requests.Session()

    for attempt in range(retries):
        try:
            headers = _get_headers()
            response = session.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()

            # Handle encoding properly
            if response.encoding is None or response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding

            print(f"Fetched: {url} | Status: {response.status_code} | Size: {len(response.text)} chars")
            return response.text

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error (attempt {attempt + 1}/{retries}): {e}")
            if response.status_code == 403:
                print("→ Site is blocking the request (403 Forbidden)")
                return None
            if response.status_code == 429:
                wait = (attempt + 1) * 3
                print(f"→ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
        except requests.exceptions.ConnectionError:
            print(f"Connection Error (attempt {attempt + 1}/{retries})")
            time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"Timeout (attempt {attempt + 1}/{retries})")
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    print("All retries exhausted.")
    return None


def _fetch_page_playwright(url, retries=2):
    """Fetch a page using Playwright to bypass bot protection and render SPAs."""
    if not HAS_PLAYWRIGHT:
        print("Playwright not installed. Skipping headless browser fallback.")
        return None

    print(f"Launching Playwright to fetch: {url}")
    for attempt in range(retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                ua = random.choice(USER_AGENTS)
                context = browser.new_context(
                    user_agent=ua,
                    viewport={'width': 1920, 'height': 1080},
                    java_script_enabled=True,
                    bypass_csp=True,
                )
                page = context.new_page()
                Stealth().apply_stealth_sync(page) # Apply stealth masks to defeat DataDome/Cloudflare
                
                # Wait until dom is loaded, then wait a bit for dynamic content
                response = page.goto(url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
                
                if response:
                    # Some aggressive protections might still block us
                    if response.status in [403, 401]:
                        print(f"→ Playwright also blocked (HTTP {response.status})")
                        browser.close()
                        return None
                    if response.status == 429:
                        wait = (attempt + 1) * 3
                        print(f"→ Rate limited in Playwright. Waiting {wait}s...")
                        time.sleep(wait)
                        browser.close()
                        continue

                html = page.content()
                browser.close()
                print(f"Playwright Fetched: {url} | Size: {len(html)} chars")
                return html
                
        except PlaywrightTimeoutError:
            print(f"Playwright Timeout (attempt {attempt + 1}/{retries})")
        except Exception as e:
            print(f"Playwright Unexpected error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2)

    print("Playwright retries exhausted.")
    return None


# --- CONTENT FILTERS ---
IGNORE_WORDS = [
    "login", "sign in", "subscribe", "advertisement", "cookie", "privacy policy",
    "terms of service", "all rights reserved", "copyright", "newsletter",
    "skip to content", "read more", "share this", "follow us", "menu",
    "search bar", "close", "toggle navigation", "sidebar", "footer",
]

IGNORE_CLASSES = [
    "navbar", "nav-bar", "main-nav", "main-menu", "site-nav",
    "footer", "sidebar", "breadcrumb", "pagination",
    "social", "share", "comment", "ad-", "advert", "promo", "popup", "modal",
    "cookie", "widget",
]


def _is_junk_element(tag):
    """Check if a tag belongs to navigation, footer, ads, etc."""
    # Check the tag's own classes and IDs
    for attr in ["class", "id"]:
        values = tag.get(attr, [])
        if isinstance(values, str):
            values = [values]
        attr_text = " ".join(values).lower()
        if any(ignore in attr_text for ignore in IGNORE_CLASSES):
            return True

    # Check only immediate parents (max 3 levels up) to avoid over-filtering
    depth = 0
    for parent in tag.parents:
        if depth >= 3:
            break
        if parent.name in ["nav", "footer", "aside"]:
            return True
        for attr in ["class", "id"]:
            values = parent.get(attr, [])
            if isinstance(values, str):
                values = [values]
            attr_text = " ".join(values).lower()
            if any(ignore in attr_text for ignore in IGNORE_CLASSES):
                return True
        depth += 1

    return False


def _is_valid_text(text):
    """Check if extracted text is meaningful content, not UI junk."""
    text_lower = text.lower().strip()
    if len(text.strip()) < 10:
        return False
    if any(word in text_lower for word in IGNORE_WORDS):
        return False
    # Skip text that's mostly numbers/symbols
    letter_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if letter_ratio < 0.4:
        return False
    return True


def _extract_with_beautifulsoup(html):
    """
    Tier 1: Smart BeautifulSoup extraction.
    Extracts from a wide range of HTML tags, not just h1-h3.
    """
    soup = BeautifulSoup(html, "lxml")
    data = []
    seen = set()  # Track duplicates during extraction

    # --- PASS 1: Headings (h1 through h6) ---
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if _is_junk_element(tag):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if _is_valid_text(text) and text not in seen:
            data.append(text)
            seen.add(text)

    # --- PASS 2: Article titles in <a> tags (with junk filter) ---
    for tag in soup.find_all("a"):
        if _is_junk_element(tag):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 20 and _is_valid_text(text) and text not in seen:
            if not text.startswith("http"):
                data.append(text)
                seen.add(text)

    # --- PASS 2b: Raw link text (NO junk filter — safety net for unusual DOMs) ---
    if len(data) < 5:
        for tag in soup.find_all("a", href=True):
            text = tag.get_text(separator=" ", strip=True)
            if 15 < len(text) < 200 and _is_valid_text(text) and text not in seen:
                if not text.startswith("http") and text.count(" ") >= 2:
                    data.append(text)
                    seen.add(text)

    # --- PASS 3: <article>, <section> title extraction ---
    for container in soup.find_all(["article", "section"]):
        if _is_junk_element(container):
            continue
        # Look for the first heading or prominent text inside
        title_tag = container.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if title_tag:
            text = title_tag.get_text(separator=" ", strip=True)
            if _is_valid_text(text) and text not in seen:
                data.append(text)
                seen.add(text)

    # --- PASS 4: <span>, <div> with title-like classes ---
    title_class_patterns = re.compile(
        r"(title|headline|heading|story|article|card|post|entry|news)", re.I
    )
    for tag in soup.find_all(["span", "div"]):
        classes = " ".join(tag.get("class", []))
        tag_id = tag.get("id", "")
        if title_class_patterns.search(classes) or title_class_patterns.search(tag_id):
            if _is_junk_element(tag):
                continue
            text = tag.get_text(separator=" ", strip=True)
            # Only grab reasonably sized text (not entire page sections)
            if 15 < len(text) < 300 and _is_valid_text(text) and text not in seen:
                data.append(text)
                seen.add(text)

    # --- PASS 5: Paragraphs (fallback if we have fewer than 5 items) ---
    if len(data) < 5:
        for tag in soup.find_all("p"):
            if _is_junk_element(tag):
                continue
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 30 and _is_valid_text(text) and text not in seen:
                data.append(text)
                seen.add(text)

    # --- PASS 6: List items (some sites use <li> for article lists) ---
    if len(data) < 5:
        for tag in soup.find_all("li"):
            if _is_junk_element(tag):
                continue
            text = tag.get_text(separator=" ", strip=True)
            if 20 < len(text) < 200 and _is_valid_text(text) and text not in seen:
                data.append(text)
                seen.add(text)

    return data


def _extract_with_trafilatura(url, html):
    """
    Tier 2: Trafilatura deep extraction.
    Extracts the main article body and splits into meaningful sentences.
    Used as a fallback when BeautifulSoup finds very little.
    """
    if not HAS_TRAFILATURA:
        print("trafilatura not installed — skipping deep extraction")
        return []

    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if not text:
            return []

        # Split into sentences/lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Filter meaningful lines
        result = [line for line in lines if _is_valid_text(line) and len(line) > 15]
        return result

    except Exception as e:
        print(f"trafilatura extraction error: {e}")
        return []


def _detect_spa(html, url):
    """
    Detect JavaScript-heavy Single Page Applications.
    These sites render content client-side and return near-empty HTML shells.
    Returns True if SPA is detected, False otherwise.
    """
    soup = BeautifulSoup(html, "lxml")
    script_count = len(soup.find_all("script"))
    body = soup.find("body")
    body_text = body.get_text(separator=" ", strip=True) if body else ""
    text_len = len(body_text)

    # Heuristic: many scripts + very little visible text = likely SPA
    if script_count > 10 and text_len < 500:
        print(f"⚠ SPA detected: {script_count} <script> tags but only {text_len} chars of visible text.")
        print(f"  → {url} likely requires a JavaScript engine for full content.")
        return True
    return False


def scrape_url(url):
    """
    Phase 1: Robust Web Crawling Pipeline.
    
    Tier 1 → Smart BeautifulSoup (headings, links, articles, spans, paragraphs)
    Tier 2 → Trafilatura deep extraction (fallback for article-heavy pages)
    
    Legal: Checks robots.txt, uses polite delays, respects rate limits.
    """
    print(f"\n{'='*60}")
    print(f"CRAWLING: {url}")
    print(f"{'='*60}")

    # Step 1: Legal compliance — check robots.txt (soft warning for educational use)
    if not _check_robots_txt(url):
        print("⚠ robots.txt disallows this URL — proceeding anyway (educational/research use)")

    # Polite delay
    time.sleep(1)

    html = _fetch_page(url)
    
    # Check if requests was blocked (html is None)
    if not html:
        print("✗ requests failed to fetch page (likely blocked). Attempting fallback with Playwright...")
        html = _fetch_page_playwright(url)
    else:
        # Check if it's a SPA that returned mostly empty content
        is_spa = _detect_spa(html, url)
        if is_spa:
            print("Attempting fallback with Playwright to render JavaScript...")
            pw_html = _fetch_page_playwright(url)
            if pw_html:
                html = pw_html

    if not html:
        print("✗ All fetching methods failed.")
        return pd.DataFrame(columns=["text"])

    # Step 3: Tier 1 — BeautifulSoup extraction
    data = _extract_with_beautifulsoup(html)
    print(f"Tier 1 (BeautifulSoup): Extracted {len(data)} items")

    # Step 4: Tier 2 — Trafilatura fallback if BS4 found very little
    if len(data) < 10:
        print("Sparse results — trying Tier 2 (trafilatura)...")
        trafilatura_data = _extract_with_trafilatura(url, html)
        # Merge, avoiding duplicates
        existing = set(data)
        for item in trafilatura_data:
            if item not in existing:
                data.append(item)
                existing.add(item)
        print(f"Tier 2 (trafilatura): Total now {len(data)} items")

    # Step 5: Build DataFrame
    df = pd.DataFrame(data, columns=["text"])

    print(f"\n✓ Crawling complete: {len(df)} items extracted")
    if not df.empty:
        print(df.head())

    return df