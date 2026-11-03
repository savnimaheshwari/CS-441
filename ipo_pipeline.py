#!/usr/bin/env python3
"""
IPO data pipeline: initial dataset -> FINAL dataset + Change Log.

    ingest -> clean -> classify -> [EDGAR enrich] -> exclude -> reconcile -> correct -> derive
           -> validate -> export

"""
import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("ipo_pipeline")
T = pd.Timestamp

# ----------------------------------------------------------------------------
# Config - tune these without touching the logic
# ----------------------------------------------------------------------------
DATE_COLS = ["Deal Date", "IPO Date", "SEC S-1 filing date"]
NUM_COLS = [
    "Deal Size ($M)", "Post Valuation ($M)", "Total Funding ($M)",
    "Valuation at IPO ($M)", "Money Raised at IPO ($M)", "IPO Share Price ($)",
    "Num of Funding Rounds (PRE-IPO)", "Filed Range Low ($)", "Filed Range High ($)",
    "Offer Price ($)", "Shares Offered (M)", "First-Day Return Rate",
]
REQUIRED_COLS = ["Companies", "Deal Date", "Deal Size ($M)", "Offer Price ($)"]

# The 16 columns kept in the final dataset, then the derived columns appended after them
FINAL_COLS = [
    "Companies", "Deal Date", "Post Valuation ($M)", "Investors", "HQ Location", "Total Funding ($M)",
    "Money Raised at IPO ($M)", "IPO Share Price ($)", "IPO Date", "Num of Funding Rounds (PRE-IPO)",
    "SEC S-1 filing date", "Filed Range Low ($)", "Filed Range High ($)", "Shares Offered (M)",
    "Lead Underwriters", "Status"]
DERIVED_COLS = ["Pricing Outcome (calc)", "price_vs_range", "days_s1_to_ipo", "offer_vs_midpoint_pct",
                "implied_proceeds_$M", "underwriter_count"]

# Rows matching any of these (in Status / Underwriters) are NOT conventional IPOs
NON_IPO_PATTERN = re.compile(r"spin-?off|not an ipo|not a nasdaq ipo|reverse merger|uplisting|uplist", re.I)
VERIFY_PATTERN = re.compile(r"verify|not confirmed|not found|not stated|not verifiable", re.I)
FIRST_DAY_SENTENCE = re.compile(r"[^.;]*first-day return[^.;]*[.;]?", re.I)  # first-day return is not modelled

DEAL_SIZE_TOLERANCE = 0.15      # allow for over-allotment / rounding
DEAL_VS_IPO_DATE_DAYS = 5       # Deal Date vs IPO Date drift
STALE_S1_DAYS = 365

# Externally verified fixes that cannot be derived from the initial dataset itself.
# (company-name prefix, column, value expected in the initial data, corrected value, source)
CORRECTIONS = [
    ("Innoviva", "SEC S-1 filing date", T("2000-03-21"), T("2004-06-10"),
     "Theravance S-1 filed 6/10/2004 (SEC EDGAR); the 2000 date does not match the 2004 IPO."),
    ("Innoviva", "Lead Underwriters", "Not confirmed in sources pulled",
     "Merrill Lynch, Lehman Brothers, Credit Suisse First Boston, Thomas Weisel Partners",
     "S-1/A cover page."),
    ("U.S. Stem Cell", "IPO Date", T("2015-12-02"), T("2008-02-19"),
     "Trading began 2/19/2008 (company press release); 12/2/2015 is unrelated."),
    ("U.S. Stem Cell", "SEC S-1 filing date", None, T("2007-02-13"),
     "Bioworld IPO table lists 'date filed' 2/13/07."),
    ("BeyondSpring", "SEC S-1 filing date", None, T("2016-11-15"),
     "F-1 filed 11/15/2016 (SEC EDGAR). The $20-22 range was first set 2/28/2017."),
]
# Judgement call - see --stem-cell-range. Renaissance Capital reports the ORIGINAL filing as $14-16;
# the initial dataset's own note gives a $6-8 preliminary range. The outcome (below) is the same.
STEM_CELL_ORIGINAL_RANGE = [
    ("U.S. Stem Cell", "Filed Range Low ($)", 6.0, 14.0,
     "Original filing 3.6M shares at $14-16 (Renaissance Capital); the range was cut later. Outcome (below) "
     "is the same with the $6-8 preliminary range in the initial data's note."),
    ("U.S. Stem Cell", "Filed Range High ($)", 8.0, 16.0, "Same source as Filed Range Low."),
]

# Underwriter count per company (team-supplied, in final-dataset row order). Rule: count EVERY named
# underwriting firm in 'Lead Underwriters', including co-managers and 'also book-runners' in
# parentheses (Regenxbio is 3 per the team's list). Companies not listed here fall back to a text
# parser that applies the same rule - check those counts.
UW_COUNTS = {
    "Legend": 3, "10x": 4, "Instil": 4, "Kyverna": 4, "Caribou": 3, "Adaptive": 5, "Nkarta": 4,
    "AtaiBeckley": 7, "Rapid": 4, "Kiniksa": 4, "Aligos": 4, "Intellia": 4, "Precision": 4,
    "Regenxbio": 3, "Ultragenyx": 4, "Innoviva": 4, "Keros": 4, "Protagonist": 3, "CytomX": 4,
    "Spero": 4, "AnaptysBio": 2, "MeiraGTx": 1, "Twist": 4, "Omeros": 6, "Enanta": 4,
    "Vaccinex": 3, "Tengion": 2, "TransCode": 1, "Lantern": 3, "GlobeImmune": 1, "Genprex": 1,
    "U.S. Stem": 1, "BeyondSpring": 1}

LOG_COLS = ["row_number", "company", "field", "old_value", "new_value", "change_type", "source", "status"]
_MONTH = "January|February|March|April|May|June|July|August|September|October|November|December"


def _blank(v):
    return v is None or (not isinstance(v, str) and pd.isna(v)) or (isinstance(v, str) and v.strip() == "")


def _same(a, b):
    if _blank(a) and _blank(b):
        return True
    if _blank(a) or _blank(b):
        return False
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(np.isclose(a, b))
    if isinstance(a, pd.Timestamp) or isinstance(b, pd.Timestamp):
        return pd.Timestamp(a) == pd.Timestamp(b)
    return str(a).strip() == str(b).strip()


def _round_half_away(x, nd=1):
    """Round like Excel (half away from zero); pandas/numpy round half to even (-6.25 -> -6.2)."""
    f = 10 ** nd
    return np.sign(x) * np.floor(np.abs(x) * f + 0.5 + 1e-9) / f


def _py(v):
    if _blank(v):
        return None
    return v.item() if isinstance(v, np.generic) else v


def _entry(df, idx, field, old, new, change_type, source, status):
    return {"row_number": df.loc[idx, "#"] if "#" in df.columns else idx + 1,
            "company": df.loc[idx, "Company"], "field": field, "old_value": _py(old), "new_value": _py(new),
            "change_type": change_type, "source": source, "status": status}


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------
def load(path: Path, sheet=0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet)
    elif suffix == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:          # Excel on Windows often saves CSV as cp1252
            df = pd.read_csv(path, encoding="cp1252")
    else:
        df = pd.read_csv(path, sep="\t")
    # Excel exports often contain non-breaking spaces in headers ("IPO\xa0Date") - normalise them
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\xa0", " ")).strip() for c in df.columns]
    # Drop blank rows and footnote rows at the bottom of the sheet (no company or no row number)
    before = len(df)
    keep = df["Companies"].notna() & ~df["Companies"].astype(str).str.match(r"^\s*Note\b", case=False)
    if "#" in df.columns:
        keep &= pd.to_numeric(df["#"], errors="coerce").notna()
    df = df[keep].reset_index(drop=True)
    if len(df) != before:
        log.info("Dropped %d blank/footnote rows", before - len(df))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path)
    return df


def _to_number(col: pd.Series) -> pd.Series:
    """Handle CSV text like '2,972.58', '$20.00', '(1.5)' and '60.9%' (percent -> decimal)."""
    if pd.api.types.is_numeric_dtype(col):
        return col
    t = col.astype("string").str.strip()
    pct = t.str.endswith("%").fillna(False)
    t = t.str.replace(r"[,$%\s]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    out = pd.to_numeric(t, errors="coerce")
    return out.where(~pct, out / 100)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Tolerate absent optional columns so downstream checks don't crash
    for c in DATE_COLS + NUM_COLS + FINAL_COLS + ["Status", "Notes (S-1 Filing)", "Lead Underwriters", "Source / Notes"]:
        if c not in df.columns:
            df[c] = pd.NA
    # Trim whitespace in text columns and turn empty strings into NaN
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], format="mixed", errors="coerce")
    for c in NUM_COLS:
        df[c] = _to_number(df[c])
    # Split "Legend Biotech USA (NAS: LEGN)" -> name + exchange + ticker
    parts = df["Companies"].str.extract(r"^(?P<company>.*?)\s*(?:\((?P<exchange>[A-Z]+):\s*(?P<ticker>[A-Z.]+)\))?$")
    df["Company"] = parts["company"]
    df["Exchange"] = parts["exchange"]
    df["Ticker"] = parts["ticker"]
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    uw = df["Lead Underwriters"].fillna("")
    haystack = df["Status"].fillna("") + " | " + uw
    no_uw = uw.str.match(r"^None\b")           # "None (spin-off ...)" = no underwriters at all
    df["is_conventional_ipo"] = ~(haystack.str.contains(NON_IPO_PATTERN) | no_uw)
    df["exclude_reason"] = ""
    df.loc[~df["is_conventional_ipo"], "exclude_reason"] = \
        "Not a conventional underwritten IPO (spin-off / reverse merger / uplisting)"
    notes = df["Source / Notes"].fillna("").str.replace(FIRST_DAY_SENTENCE, " ", regex=True)
    df["needs_verification"] = notes.str.contains(VERIFY_PATTERN)
    return df


def mark_unclassifiable(df: pd.DataFrame) -> pd.DataFrame:
    """A conventional IPO still cannot be used if its pricing outcome cannot be computed."""
    df = df.copy()
    price = df["Offer Price ($)"].fillna(df["IPO Share Price ($)"])
    for idx in df.index[df["is_conventional_ipo"]]:
        gaps = [n for n, v in [("Filed Range Low", df.at[idx, "Filed Range Low ($)"]),
                               ("Filed Range High", df.at[idx, "Filed Range High ($)"]),
                               ("Offer Price", price[idx])] if pd.isna(v)]
        if gaps:
            df.at[idx, "exclude_reason"] = "Pricing outcome cannot be classified: missing " + ", ".join(gaps)
    return df


def reconcile(df: pd.DataFrame):
    """IPO Share Price := Offer Price; fix Money Raised from Deal Size; blank S-1 date from the notes."""
    df, out = df.copy(), []
    for idx, r in df.iterrows():
        offer, ipo_px = r["Offer Price ($)"], r["IPO Share Price ($)"]
        px = offer if pd.notna(offer) else ipo_px
        if pd.notna(offer) and pd.notna(ipo_px) and not np.isclose(offer, ipo_px):
            out.append(_entry(df, idx, "IPO Share Price ($)", ipo_px, offer, "reconcile",
                              "Used 'Offer Price ($)': it conflicts with 'IPO Share Price ($)'.", "applied"))
        elif pd.isna(ipo_px) and pd.notna(offer):
            out.append(_entry(df, idx, "IPO Share Price ($)", None, offer, "reconcile",
                              "Blank in 'IPO Share Price ($)'; filled from 'Offer Price ($)'.", "applied"))
        df.at[idx, "IPO Share Price ($)"] = px

        mr, ds, sh = r["Money Raised at IPO ($M)"], r["Deal Size ($M)"], r["Shares Offered (M)"]
        implied = px * sh if pd.notna(px) and pd.notna(sh) else np.nan
        if pd.isna(mr) and pd.notna(ds):
            df.at[idx, "Money Raised at IPO ($M)"] = ds
            out.append(_entry(df, idx, "Money Raised at IPO ($M)", None, ds, "reconcile",
                              "Blank; filled from 'Deal Size ($M)'.", "applied"))
        elif pd.notna(mr) and pd.notna(ds) and pd.notna(implied) and implied > 0 \
                and abs(mr - implied) / implied > DEAL_SIZE_TOLERANCE \
                and abs(ds - implied) / implied <= DEAL_SIZE_TOLERANCE:
            df.at[idx, "Money Raised at IPO ($M)"] = ds
            out.append(_entry(df, idx, "Money Raised at IPO ($M)", mr, ds, "reconcile",
                              f"Money Raised is {abs(mr - implied) / implied:.0%} off shares x price "
                              f"(${implied:.1f}M); 'Deal Size ($M)' agrees, so it was used.", "applied"))

        notes = r["Notes (S-1 Filing)"]
        if pd.isna(r["SEC S-1 filing date"]) and not _blank(notes):
            m = re.search(rf"({_MONTH})\s+(\d{{1,2}}),\s*(\d{{4}})", str(notes))
            if m:
                dt = pd.to_datetime(m.group(0), format="%B %d, %Y")
                df.at[idx, "SEC S-1 filing date"] = dt
                out.append(_entry(df, idx, "SEC S-1 filing date", None, dt, "reconcile",
                                  "Date taken from 'Notes (S-1 Filing)'.", "applied"))
    return df, out


def apply_corrections(df: pd.DataFrame, stem_cell_range="original"):
    df, out = df.copy(), []
    fixes = CORRECTIONS + (STEM_CELL_ORIGINAL_RANGE if stem_cell_range == "original" else [])
    for prefix, col, old, new, source in fixes:
        hit = df.index[df["Companies"].str.startswith(prefix).fillna(False)]
        if len(hit) != 1:
            log.warning("Correction skipped: %d rows match '%s'", len(hit), prefix)
            continue
        idx = hit[0]
        cur = df.at[idx, col]
        if _same(cur, new):
            status = "already in input"
        else:
            status = "applied" if _same(cur, old) else "applied (WARNING: unexpected input value)"
            df.at[idx, col] = new
        out.append(_entry(df, idx, col, cur, new, "correction", source, status))
    return df, out


def _uw_count(row):
    name, text = str(row["Companies"]), row["Lead Underwriters"]
    for k, v in UW_COUNTS.items():
        if name.startswith(k):
            return v
    if _blank(text):
        return np.nan
    # fallback: every named firm outside AND inside parentheses (role labels such as 'co-mgr',
    # 'lead mgr', 'also underwriters', 'joint book-runners' are stripped)
    names = []
    text = re.sub(r"\((a unit of|a division of)[^)]*\)", " ", str(text), flags=re.I)   # parent company, not an extra firm
    for chunk in re.split(r"[(),;]", text):
        chunk = re.sub(r"\b(also (book-runners|underwriters)|co-?m(an)?ag(e)?rs?|co-mgr|lead mgr|lead manager|"
                       r"joint book-runners|sole (book-running manager|underwriter|manager)|a unit of|a division of)\b",
                       " ", chunk, flags=re.I)
        chunk = re.sub(r"\s+-\s*$", "", chunk.strip()).strip(" -")
        if chunk and not chunk.lower().startswith("none"):
            names.append(chunk)
    return len(names)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    low, high, px = df["Filed Range Low ($)"], df["Filed Range High ($)"], df["IPO Share Price ($)"]
    pos = pd.Series(pd.NA, index=df.index, dtype="string")
    known = low.notna() & high.notna() & px.notna()
    pos[known & (px < low)] = "below"
    pos[known & (px > high)] = "above"
    pos[known & (px >= low) & (px <= high)] = "within"
    df["price_vs_range"] = pos
    df["Pricing Outcome (calc)"] = pos.str.capitalize()
    mid = (low + high) / 2
    df["offer_vs_midpoint_pct"] = _round_half_away(((px / mid) - 1).where(known & (mid > 0)) * 100, 1)   # PERCENT, 1 dp
    df["implied_proceeds_$M"] = _round_half_away(df["Shares Offered (M)"] * px, 2)
    df["ipo_year"] = df["IPO Date"].dt.year
    df["days_s1_to_ipo"] = (df["IPO Date"] - df["SEC S-1 filing date"]).dt.days
    df["underwriter_count"] = df.apply(_uw_count, axis=1)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long table of issues that REMAIN in the final dataset."""
    issues = []

    def add(idx, check, detail):
        issues.append({"row_number": df.loc[idx, "#"] if "#" in df.columns else idx + 1,
                       "company": df.loc[idx, "Company"], "check": check, "detail": detail})

    for idx, r in df.iterrows():
        ds, mr = r["Deal Size ($M)"], r["Money Raised at IPO ($M)"]
        if pd.notna(r["implied_proceeds_$M"]) and pd.notna(ds) and ds > 0:
            diff = abs(r["implied_proceeds_$M"] - ds) / ds
            if diff > DEAL_SIZE_TOLERANCE:
                add(idx, "deal_size_mismatch",
                    f"shares x price = {r['implied_proceeds_$M']:.1f}M vs Deal Size {ds:.1f}M ({diff:.0%} off)")
        if pd.notna(mr) and pd.notna(ds) and ds > 0 and abs(mr - ds) / ds > DEAL_SIZE_TOLERANCE:
            add(idx, "money_raised_mismatch", f"Money Raised {mr}M vs Deal Size {ds}M")
        if pd.notna(r["IPO Date"]) and pd.notna(r["SEC S-1 filing date"]) and r["IPO Date"] < r["SEC S-1 filing date"]:
            add(idx, "ipo_before_s1", f"IPO {r['IPO Date']:%Y-%m-%d} precedes S-1 {r['SEC S-1 filing date']:%Y-%m-%d}")
        if pd.notna(r["IPO Date"]) and pd.notna(r["Deal Date"]):
            gap = abs((r["IPO Date"] - r["Deal Date"]).days)
            if gap > DEAL_VS_IPO_DATE_DAYS:
                add(idx, "deal_vs_ipo_date", f"Deal Date and IPO Date differ by {gap} days")
        if pd.notna(r["days_s1_to_ipo"]) and r["days_s1_to_ipo"] > STALE_S1_DAYS:
            add(idx, "stale_s1", f"S-1 filed {int(r['days_s1_to_ipo'])} days before IPO")
        if pd.notna(r["IPO Date"]) and r["IPO Date"] > pd.Timestamp.today():
            add(idx, "future_ipo_date", f"IPO Date {r['IPO Date']:%Y-%m-%d} is in the future")
        lo, hi = r["Filed Range Low ($)"], r["Filed Range High ($)"]
        if pd.notna(lo) and pd.notna(hi):
            if lo > hi:
                add(idx, "range_inverted", f"Low {lo} > High {hi}")
            elif lo == hi:
                add(idx, "range_degenerate", f"Filed range low = high = ${lo:g}")
        for col in ["Filed Range Low ($)", "Filed Range High ($)", "IPO Share Price ($)", "SEC S-1 filing date"]:
            if pd.isna(r[col]):
                add(idx, "missing_field", col)
        if r["needs_verification"]:
            add(idx, "flagged_in_notes", "Source / Notes says to verify (first-day-return notes ignored)")
    dup = df[df.duplicated(subset=["Ticker", "IPO Date"], keep=False) & df["Ticker"].notna()]
    for idx in dup.index:
        add(idx, "duplicate", "Duplicate Ticker + IPO Date")
    return pd.DataFrame(issues, columns=["row_number", "company", "check", "detail"])


def _write_sheet(writer, df, name, date_fmt="yyyy-mm-dd"):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    df.to_excel(writer, sheet_name=name, index=False)
    ws = writer.sheets[name]
    header_fill = PatternFill("solid", start_color="1F3864")
    for cell in ws[1]:
        cell.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Arial", size=10)
            if hasattr(cell.value, "year"):
                cell.number_format = date_fmt
    for i, col in enumerate(df.columns, start=1):
        sample = df[col].astype(str).head(200)
        width = min(max(len(str(col)), sample.str.len().max() if len(sample) else 0) + 2, 60)
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = "A2"
    if len(df):
        ws.auto_filter.ref = ws.dimensions


def export(df: pd.DataFrame, final: pd.DataFrame, excluded: pd.DataFrame, issues: pd.DataFrame,
           changes: pd.DataFrame, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = final[FINAL_COLS + DERIVED_COLS]
    excl = excluded[["#", "Companies", "exclude_reason", "Status", "Lead Underwriters"]] \
        if "#" in excluded.columns else excluded[["Companies", "exclude_reason", "Status", "Lead Underwriters"]]
    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "rows_in": int(len(df)),
        "excluded_not_ipo": int(excluded["exclude_reason"].str.startswith("Not a conventional").sum()),
        "excluded_unclassifiable": int(excluded["exclude_reason"].str.startswith("Pricing").sum()),
        "final_rows": int(len(final)),
        "price_vs_range": final["price_vs_range"].value_counts().to_dict(),
        "changes_logged": int(len(changes)),
        "changes_by_type": changes["change_type"].value_counts().to_dict() if len(changes) else {},
        "issues_total": int(len(issues)),
        "issues_by_check": issues["check"].value_counts().to_dict() if len(issues) else {},
        "median_days_s1_to_ipo": float(final["days_s1_to_ipo"].median()),
    }
    rows = [("Run at", summary["run_at"]), ("Rows in (initial dataset)", summary["rows_in"]),
            ("Excluded: not a conventional IPO", summary["excluded_not_ipo"]),
            ("Excluded: pricing outcome unclassifiable", summary["excluded_unclassifiable"]),
            ("Rows in final dataset", summary["final_rows"])]
    rows += [(f"Priced {k} range", v) for k, v in summary["price_vs_range"].items()]
    rows += [("Median days S-1 to IPO", summary["median_days_s1_to_ipo"]), ("Changes logged", summary["changes_logged"])]
    rows += [(f"Changes: {k}", v) for k, v in summary["changes_by_type"].items()]
    rows += [("Issues remaining", summary["issues_total"])]
    rows += [(f"Issues: {k}", v) for k, v in summary["issues_by_check"].items()]
    summary_df = pd.DataFrame(rows, columns=["Metric", "Value"])

    xlsx_path = outdir / "ipo_final_dataset.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
        _write_sheet(writer, summary_df, "Summary")
        _write_sheet(writer, dataset, "Dataset", date_fmt="m/d/yyyy")
        _write_sheet(writer, excl, "Excluded")
        _write_sheet(writer, changes, "Change Log")
        _write_sheet(writer, issues, "Issues")
    csv_df = dataset.copy()
    for c in DATE_COLS:      # 6/5/2020 style, no zero padding
        csv_df[c] = csv_df[c].map(lambda d: "" if pd.isna(d) else f"{d.month}/{d.day}/{d.year}")
    csv_df.to_csv(outdir / "ipo_final_dataset.csv", index=False)
    summary["output_file"] = str(xlsx_path)
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _fills_to_changes(fills: pd.DataFrame) -> pd.DataFrame:
    """Convert edgar_enrich's Fills_Log into the unified Change Log format."""
    if fills is None or not len(fills):
        return pd.DataFrame(columns=LOG_COLS)
    kind = np.where(fills["note"].str.startswith("filled"), "edgar_fill",
                    np.where(fills["note"].str.startswith("CONFLICT"), "edgar_conflict", "edgar_skipped"))
    return pd.DataFrame({"row_number": fills["row_number"], "company": fills["company"], "field": fills["field"],
                         "old_value": fills["old_value"], "new_value": fills["new_value"], "change_type": kind,
                         "source": fills["source_url"].where(fills["source_url"].astype(bool), fills["note"]),
                         "status": fills["confidence"]})[LOG_COLS]


# ----------------------------------------------------------------------------
def build(df_raw: pd.DataFrame, stem_cell_range="original", enrich_fn=None):
    """Pure function used by main(): returns (final, excluded, changes, all_rows)."""
    df = classify(clean(df_raw))
    edgar = pd.DataFrame(columns=LOG_COLS)
    if enrich_fn is not None:
        df, fills = enrich_fn(df)
        edgar = _fills_to_changes(fills)
    df = mark_unclassifiable(df)
    excluded = df[df["exclude_reason"] != ""].reset_index(drop=True)
    kept = df[df["exclude_reason"] == ""].reset_index(drop=True)
    kept, c1 = reconcile(kept)
    kept, c2 = apply_corrections(kept, stem_cell_range)
    changes = pd.concat([edgar, pd.DataFrame(c1 + c2, columns=LOG_COLS)], ignore_index=True)
    final = derive(kept)
    return final, excluded, changes, df


def main():
    ap = argparse.ArgumentParser(description="IPO data pipeline: initial dataset -> final dataset + change log")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--sheet", default=0, help="Excel sheet name or index (default: first sheet)")
    ap.add_argument("--outdir", default=Path("output"), type=Path)
    ap.add_argument("--stem-cell-range", choices=["original", "preliminary"], default="original",
                    help="U.S. Stem Cell filed range: 'original' = $14-16 (S-1), 'preliminary' = $6-8 (initial data)")
    ap.add_argument("--enrich", action="store_true",
                    help="fill blank S-1 date / price range / offer price / shares from SEC EDGAR (before exclusion)")
    ap.add_argument("--sec-user-agent", default=os.environ.get("SEC_USER_AGENT", ""),
                    help="required with --enrich, e.g. 'Your Name you@school.edu' (or set SEC_USER_AGENT)")
    ap.add_argument("--cache-dir", default="edgar_cache", type=Path)
    ap.add_argument("--fail-on-issues", action="store_true",
                    help="exit non-zero if any issues remain (useful in CI/cron alerting)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    enrich_fn = None
    if args.enrich:
        from edgar_enrich import enrich
        enrich_fn = lambda d: enrich(d, args.sec_user_agent, args.cache_dir)

    raw = load(args.input, args.sheet)
    final, excluded, changes, all_rows = build(raw, args.stem_cell_range, enrich_fn)
    issues = validate(final)
    summary = export(all_rows, final, excluded, issues, changes, args.outdir)
    log.info("Initial %d rows -> excluded %d -> final %d | outcomes %s", summary["rows_in"], len(excluded),
             len(final), summary["price_vs_range"])
    log.info("Summary: %s", json.dumps(summary, default=str))
    if args.fail_on_issues and len(issues):
        raise SystemExit(f"{len(issues)} data-quality issues found - see the Issues sheet in {summary['output_file']}")


if __name__ == "__main__":
    main()