"""
EDGAR enrichment: fills blank IPO fields from SEC filings.

Fills (only when the cell is blank - existing values are never overwritten):
    SEC S-1 filing date      earliest S-1 / F-1 filed within 400 days before the IPO
    Filed Range Low/High ($) first S-1/A (or S-1/F-1) that states a price range
    Offer Price ($)          parsed from the final prospectus (424B4 / 424B1)
    Shares Offered (M)       parsed from the final prospectus (424B4 / 424B1)

Every fill (and every conflict with a value you already had) is written to a
log with the SEC URL so a person can spot-check it. Dates come straight from
EDGAR metadata (reliable). Prices / share counts are regex-parsed from filing
text (usually right, but marked "review").

SEC rules: requests must send a descriptive User-Agent with contact info, and
stay under 10 requests/second. This module sleeps between calls and caches
every response on disk so reruns are fast and polite.

Not covered: first-day return (not in EDGAR), and filings before ~1996
(e.g. Amgen 1983) which are not on EDGAR at all.
"""
import difflib
import hashlib
import html
import json
import logging
import re
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger("ipo_pipeline.edgar")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

FINAL_FORMS = {"424B4", "424B1"}
REG_FORMS = {"S-1", "S-1/A", "F-1", "F-1/A"}
ORIGINAL_REG_FORMS = {"S-1", "F-1"}
S1_WINDOW_DAYS = 400          # ignore S-1s from an earlier, abandoned attempt
FINAL_PROSPECTUS_WINDOW = (-3, 7)   # days relative to IPO date
MAX_RANGE_DOCS = 4

RANGE_RE = re.compile(
    r"(?:between|range of|from)\s+(?:approximately\s+)?\$\s?(\d+(?:\.\d+)?)\s*(?:and|to|-|\u2013)\s*\$?\s?(\d+(?:\.\d+)?)"
    r"\s+per\s+(?:share|ADS|American depositary share|unit)", re.I)
PRICE_RES = [
    re.compile(r"initial public offering price[^$]{0,120}\$\s?(\d+(?:\.\d+)?)\s*per", re.I),
    re.compile(r"\$\s?(\d+\.\d{2})\s+per\s+(?:share|ADS|American depositary share)", re.I),
]
SHARES_RES = [
    re.compile(r"(?:we|the company) (?:are|is) offering\s+([\d,]{7,})\s+(?:shares|ADS|American)", re.I),
    re.compile(r"offering\s+([\d,]{7,})\s+(?:shares|ADS|American)", re.I),
    re.compile(r"([\d,]{7,})\s+(?:shares of (?:our )?(?:Class A )?common stock|ordinary shares|American Depositary Shares|ADSs)", re.I),
]


# ----------------------------------------------------------------------------
class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path, min_interval: float = 0.15):
        import requests  # imported lazily so the base pipeline works without it
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with contact info, e.g. "
                "--sec-user-agent 'Acme Corp data-team@acme.com'")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, url: str, as_json: bool = False):
        key = self.cache / hashlib.sha1(url.encode()).hexdigest()
        if key.exists():
            txt = key.read_text(encoding="utf-8", errors="ignore")
            return json.loads(txt) if as_json else txt
        for attempt in range(4):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = self.session.get(url, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            key.write_text(r.text, encoding="utf-8")
            return r.json() if as_json else r.text
        log.warning("Giving up on %s", url)
        return None


# ----------------------------------------------------------------------------
def _norm_name(s: str) -> str:
    s = re.sub(r"\b(inc|corp|corporation|ltd|plc|holdings?|therapeutics|pharmaceuticals?|the|nv)\b", " ", s.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def resolve_cik(client: EdgarClient, ticker, company):
    data = client.get(TICKERS_URL, as_json=True) or {}
    rows = list(data.values())
    if isinstance(ticker, str) and ticker:
        for r in rows:
            if r["ticker"].upper() == ticker.upper():
                return int(r["cik_str"]), "ticker"
    if isinstance(company, str) and company:
        names = {_norm_name(r["title"]): r for r in rows}
        hit = difflib.get_close_matches(_norm_name(company), list(names), n=1, cutoff=0.85)
        if hit:
            return int(names[hit[0]]["cik_str"]), "name"
    return None, "not_found"


def list_filings(client: EdgarClient, cik: int):
    """All filings for a CIK: the recent block plus older archive pages."""
    main = client.get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"), as_json=True)
    if not main:
        return []
    blocks = [main["filings"]["recent"]]
    for f in main["filings"].get("files", []):
        extra = client.get(SUBMISSIONS_URL.format(name=f["name"]), as_json=True)
        if extra:
            blocks.append(extra)
    out = []
    for b in blocks:
        for form, date, acc, doc in zip(b["form"], b["filingDate"], b["accessionNumber"], b["primaryDocument"]):
            out.append({"form": form, "date": pd.Timestamp(date), "acc": acc.replace("-", ""), "doc": doc, "cik": cik})
    return out


def filing_url(f) -> str:
    return ARCHIVE_URL.format(cik=f["cik"], acc=f["acc"], doc=f["doc"])


def filing_text(client: EdgarClient, f, limit=60000) -> str:
    raw = client.get(filing_url(f)) or ""
    txt = html.unescape(re.sub(r"<[^>]+>", " ", raw)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", txt)[:limit]


# ----------------------------------------------------------------------------
# Parsers (pure functions - easy to unit test)
def parse_range(text):
    m = RANGE_RE.search(text)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if 0 < lo <= hi:
            return lo, hi
    return None


def parse_offer_price(text):
    for rx in PRICE_RES:
        m = rx.search(text[:20000])
        if m:
            return float(m.group(1))
    return None


def parse_shares_m(text):
    head = text[:8000]
    for rx in SHARES_RES:
        m = rx.search(head)
        if m:
            return int(m.group(1).replace(",", "")) / 1e6
    return None


# ----------------------------------------------------------------------------
def select_filings(filings, ipo_date):
    final = [f for f in filings if f["form"] in FINAL_FORMS
             and FINAL_PROSPECTUS_WINDOW[0] <= (f["date"] - ipo_date).days <= FINAL_PROSPECTUS_WINDOW[1]]
    final.sort(key=lambda f: abs((f["date"] - ipo_date).days))
    reg = [f for f in filings if f["form"] in REG_FORMS
           and 0 <= (ipo_date - f["date"]).days <= S1_WINDOW_DAYS]
    reg.sort(key=lambda f: f["date"])
    originals = [f for f in reg if f["form"] in ORIGINAL_REG_FORMS]
    return (final[0] if final else None), (originals[0] if originals else None), reg


def enrich(df: pd.DataFrame, user_agent: str, cache_dir="edgar_cache"):
    """Return (df_with_fills, fills_log_df). Only rows flagged is_conventional_ipo are touched."""
    client = EdgarClient(user_agent, Path(cache_dir))
    fills = []
    df = df.copy()
    watch = ["SEC S-1 filing date", "Filed Range Low ($)", "Filed Range High ($)", "Offer Price ($)", "Shares Offered (M)"]

    def log_fill(idx, field, old, new, url, conf, note=""):
        fills.append({
            "row_number": df.loc[idx, "#"] if "#" in df.columns else idx + 1,
            "company": df.loc[idx, "Company"], "field": field,
            "old_value": old, "new_value": new, "source_url": url,
            "confidence": conf, "note": note,
        })

    def put(idx, field, value, url, conf):
        cur = df.at[idx, field]
        if pd.isna(cur):
            df.at[idx, field] = value
            log_fill(idx, field, None, value, url, conf, "filled")
        elif isinstance(value, float) and abs(float(cur) - value) > 1e-6:
            log_fill(idx, field, cur, value, url, conf, "CONFLICT - existing value kept, please review")

    todo = df[df["is_conventional_ipo"] & df[watch].isna().any(axis=1)]
    log.info("EDGAR enrichment: %d rows with blank fields", len(todo))

    for idx, r in todo.iterrows():
        ipo_date = r["IPO Date"] if pd.notna(r["IPO Date"]) else r["Deal Date"]
        if pd.isna(ipo_date):
            log_fill(idx, "-", None, None, "", "n/a", "skipped: no IPO date")
            continue
        try:
            cik, how = resolve_cik(client, r["Ticker"], r["Company"])
            if cik is None:
                log_fill(idx, "-", None, None, "", "n/a", "skipped: company not found on EDGAR (delisted/renamed or pre-1996?)")
                continue
            filings = list_filings(client, cik)
            final, first_s1, reg = select_filings(filings, ipo_date)
        except Exception as e:  # network hiccup on one row must not kill the run
            log_fill(idx, "-", None, None, "", "n/a", f"error: {e}")
            continue

        if first_s1 is not None:
            put(idx, "SEC S-1 filing date", first_s1["date"], filing_url(first_s1), "high (EDGAR metadata)")
        else:
            log_fill(idx, "SEC S-1 filing date", None, None, "", "n/a", "no S-1/F-1 found in window")

        for f in reg[:MAX_RANGE_DOCS]:
            rng = parse_range(filing_text(client, f))
            if rng:
                put(idx, "Filed Range Low ($)", rng[0], filing_url(f), "review (regex)")
                put(idx, "Filed Range High ($)", rng[1], filing_url(f), "review (regex)")
                break

        if final is not None:
            text = filing_text(client, final)
            price, shares = parse_offer_price(text), parse_shares_m(text)
            if price is not None:
                put(idx, "Offer Price ($)", price, filing_url(final), "review (regex)")
            if shares is not None:
                put(idx, "Shares Offered (M)", shares, filing_url(final), "review (regex)")
        else:
            log_fill(idx, "Offer Price ($)", None, None, "", "n/a", "no 424B4/424B1 near IPO date")

    cols = ["row_number", "company", "field", "old_value", "new_value", "source_url", "confidence", "note"]
    return df, pd.DataFrame(fills, columns=cols)