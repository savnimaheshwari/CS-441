#!/usr/bin/env python3
"""
IPO data pipeline: ingest -> clean -> classify -> derive -> validate -> export.

Usage:
    python ipo_pipeline.py --input IPO_data.xlsx --outdir output/
    python ipo_pipeline.py --input IPO_data.xlsx --sheet "Sheet1" --outdir output/
    python ipo_pipeline.py --input IPO_data.xlsx --enrich --sec-user-agent "Acme Corp you@acme.com"

Input: .xlsx (default), also .csv / .tsv.   Requires: pandas>=2.0, openpyxl

Output (in --outdir): ipo_pipeline_output_<timestamp>.xlsx with sheets
    Summary     run counts and breakdowns
    Clean       conventional underwritten IPOs, analysis-ready, with derived fields
    Excluded    spin-offs, reverse mergers, uplistings, etc. (not real IPOs)
    Issues      one row per data-quality issue (row, company, check, detail)
    Fills_Log   (with --enrich) every field filled + source URL, plus conflicts and skips
plus run_summary.json for monitoring / alerting.
"""
import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger("ipo_pipeline")

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

# Rows matching any of these (in Status / Notes / Underwriters) are NOT conventional IPOs
NON_IPO_PATTERN = re.compile(
    r"spin-?off|not an ipo|not a nasdaq ipo|reverse merger|uplisting",
    re.I,
)
VERIFY_PATTERN = re.compile(r"verify|not confirmed|not found|not stated|not verifiable", re.I)

DEAL_SIZE_TOLERANCE = 0.15      # allow for over-allotment / rounding
DEAL_VS_IPO_DATE_DAYS = 5       # Deal Date vs IPO Date drift
MAX_PLAUSIBLE_RETURN = 3.0      # +300% day-one
MIN_PLAUSIBLE_RETURN = -0.95


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
    for c in DATE_COLS + NUM_COLS + ["Status", "Notes (S-1 Filing)", "Lead Underwriters", "Source / Notes"]:
        if c not in df.columns:
            df[c] = pd.NA
    # Trim whitespace in text columns and turn empty strings into NaN
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)

    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="mixed", errors="coerce")
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = _to_number(df[c])

    # Split "Legend Biotech USA (NAS: LEGN)" -> name + exchange + ticker
    parts = df["Companies"].str.extract(r"^(?P<company>.*?)\s*(?:\((?P<exchange>[A-Z]+):\s*(?P<ticker>[A-Z.]+)\))?$")
    df["Company"] = parts["company"]
    df["Exchange"] = parts["exchange"]
    df["Ticker"] = parts["ticker"]
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    haystack = (
        df.get("Status", pd.Series("", index=df.index)).fillna("") + " | "
        + df.get("Lead Underwriters", pd.Series("", index=df.index)).fillna("")
    )
    # Lead underwriters starting with "None (" means no underwriters at all
    no_uw = df.get("Lead Underwriters", pd.Series("", index=df.index)).fillna("").str.match(r"^None\b")
    df["is_conventional_ipo"] = ~(haystack.str.contains(NON_IPO_PATTERN) | no_uw)
    note_col = df.get("Source / Notes", pd.Series("", index=df.index)).fillna("")
    df["needs_verification"] = note_col.str.contains(VERIFY_PATTERN)
    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    low, high, offer = df["Filed Range Low ($)"], df["Filed Range High ($)"], df["Offer Price ($)"]
    pos = pd.Series(pd.NA, index=df.index, dtype="string")
    known = low.notna() & high.notna() & offer.notna()
    pos[known & (offer < low)] = "below"
    pos[known & (offer > high)] = "above"
    pos[known & (offer >= low) & (offer <= high)] = "within"
    df["price_vs_range"] = pos

    mid = (low + high) / 2
    df["offer_vs_midpoint_pct"] = ((offer / mid) - 1).where(known & (mid > 0))
    df["implied_proceeds_$M"] = df["Shares Offered (M)"] * offer
    df["ipo_year"] = df["IPO Date"].dt.year
    df["days_s1_to_ipo"] = (df["IPO Date"] - df["SEC S-1 filing date"]).dt.days
    df["first_day_return_pct"] = df["First-Day Return Rate"] * 100
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long table of issues. Only conventional IPOs get IPO-specific checks."""
    issues = []

    def add(idx, check, detail):
        issues.append({
            "row_number": df.loc[idx, "#"] if "#" in df.columns else idx + 1,
            "company": df.loc[idx, "Company"],
            "check": check,
            "detail": detail,
        })

    for idx, r in df.iterrows():
        if not r["is_conventional_ipo"]:
            continue

        # 1. Offer price vs implied proceeds
        if pd.notna(r["implied_proceeds_$M"]) and pd.notna(r["Deal Size ($M)"]) and r["Deal Size ($M)"] > 0:
            diff = abs(r["implied_proceeds_$M"] - r["Deal Size ($M)"]) / r["Deal Size ($M)"]
            if diff > DEAL_SIZE_TOLERANCE:
                add(idx, "deal_size_mismatch",
                    f"shares x price = {r['implied_proceeds_$M']:.1f}M vs Deal Size {r['Deal Size ($M)']:.1f}M ({diff:.0%} off)")

        # 2. Money Raised at IPO vs Deal Size
        mr = r.get("Money Raised at IPO ($M)")
        if pd.notna(mr) and pd.notna(r["Deal Size ($M)"]) and r["Deal Size ($M)"] > 0:
            if abs(mr - r["Deal Size ($M)"]) / r["Deal Size ($M)"] > DEAL_SIZE_TOLERANCE:
                add(idx, "money_raised_mismatch", f"Money Raised {mr}M vs Deal Size {r['Deal Size ($M)']}M")

        # 3. Date logic
        if pd.notna(r["IPO Date"]) and pd.notna(r["SEC S-1 filing date"]) and r["IPO Date"] < r["SEC S-1 filing date"]:
            add(idx, "ipo_before_s1", f"IPO {r['IPO Date']:%Y-%m-%d} precedes S-1 {r['SEC S-1 filing date']:%Y-%m-%d}")
        if pd.notna(r["IPO Date"]) and pd.notna(r["Deal Date"]):
            gap = abs((r["IPO Date"] - r["Deal Date"]).days)
            if gap > DEAL_VS_IPO_DATE_DAYS:
                add(idx, "deal_vs_ipo_date", f"Deal Date and IPO Date differ by {gap} days")
        if pd.notna(r["days_s1_to_ipo"]) and r["days_s1_to_ipo"] > 365:
            add(idx, "stale_s1", f"S-1 filed {int(r['days_s1_to_ipo'])} days before IPO")
        if pd.notna(r["IPO Date"]) and r["IPO Date"] > pd.Timestamp.today():
            add(idx, "future_ipo_date", f"IPO Date {r['IPO Date']:%Y-%m-%d} is in the future")

        # 4. Price/range logic
        lo, hi = r["Filed Range Low ($)"], r["Filed Range High ($)"]
        if pd.notna(lo) and pd.notna(hi) and lo > hi:
            add(idx, "range_inverted", f"Low {lo} > High {hi}")
        if pd.notna(r["IPO Share Price ($)"]) and pd.notna(r["Offer Price ($)"]) \
                and abs(r["IPO Share Price ($)"] - r["Offer Price ($)"]) > 0.01:
            add(idx, "price_fields_disagree",
                f"IPO Share Price {r['IPO Share Price ($)']} vs Offer Price {r['Offer Price ($)']}")

        # 5. Return plausibility
        ret = r["First-Day Return Rate"]
        if pd.notna(ret) and not (MIN_PLAUSIBLE_RETURN <= ret <= MAX_PLAUSIBLE_RETURN):
            add(idx, "implausible_first_day_return", f"First-Day Return Rate = {ret} (expected decimal, e.g. 0.35 = +35%)")

        # 6. Missing key fields / flagged notes
        for col in ["Filed Range Low ($)", "Filed Range High ($)", "First-Day Return Rate", "SEC S-1 filing date"]:
            if pd.isna(r[col]):
                add(idx, "missing_field", col)
        if r["needs_verification"]:
            add(idx, "flagged_in_notes", "Source / Notes says to verify")

    # Duplicate checks across all rows
    dup = df[df.duplicated(subset=["Ticker", "IPO Date"], keep=False) & df["Ticker"].notna()]
    for idx in dup.index:
        add(idx, "duplicate", "Duplicate Ticker + IPO Date")

    return pd.DataFrame(issues, columns=["row_number", "company", "check", "detail"])


def _write_sheet(writer, df, name):
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
                cell.number_format = "yyyy-mm-dd"
    for i, col in enumerate(df.columns, start=1):
        sample = df[col].astype(str).head(200)
        width = min(max(len(str(col)), sample.str.len().max() if len(sample) else 0) + 2, 60)
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = "A2"
    if len(df):
        ws.auto_filter.ref = ws.dimensions


def export(df: pd.DataFrame, issues: pd.DataFrame, outdir: Path, fills=None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    clean_df = df[df["is_conventional_ipo"]].drop(columns=["is_conventional_ipo"])
    excluded = df[~df["is_conventional_ipo"]]

    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "rows_in": int(len(df)),
        "conventional_ipos": int(len(clean_df)),
        "excluded_non_ipos": int(len(excluded)),
        "issues_total": int(len(issues)),
        "fields_filled": int(fills["note"].str.startswith("filled").sum()) if fills is not None and len(fills) else 0,
        "fill_conflicts": int(fills["note"].str.startswith("CONFLICT").sum()) if fills is not None and len(fills) else 0,
        "issues_by_check": issues["check"].value_counts().to_dict() if len(issues) else {},
        "price_vs_range": clean_df["price_vs_range"].value_counts().to_dict(),
        "median_first_day_return_pct": (
            round(float(clean_df["first_day_return_pct"].median()), 2)
            if clean_df["first_day_return_pct"].notna().any() else None
        ),
    }

    summary_rows = [("Run at", summary["run_at"]), ("Rows in", summary["rows_in"]),
                    ("Conventional IPOs", summary["conventional_ipos"]),
                    ("Excluded (not IPOs)", summary["excluded_non_ipos"]),
                    ("Issues found", summary["issues_total"]),
                    ("Median first-day return (%)", summary["median_first_day_return_pct"])]
    summary_rows += [(f"Issues: {k}", v) for k, v in summary["issues_by_check"].items()]
    summary_rows += [(f"Priced {k} range", v) for k, v in summary["price_vs_range"].items()]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    xlsx_path = outdir / f"ipo_pipeline_output_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", datetime_format="yyyy-mm-dd") as writer:
        _write_sheet(writer, summary_df, "Summary")
        _write_sheet(writer, clean_df, "Clean")
        _write_sheet(writer, excluded, "Excluded")
        _write_sheet(writer, issues, "Issues")
        if fills is not None and len(fills):
            _write_sheet(writer, fills, "Fills_Log")
    summary["output_file"] = str(xlsx_path)
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="IPO data pipeline")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--sheet", default=0, help="Excel sheet name or index (default: first sheet)")
    ap.add_argument("--outdir", default=Path("output"), type=Path)
    ap.add_argument("--enrich", action="store_true",
                    help="fill blank S-1 date / price range / offer price / shares from SEC EDGAR")
    ap.add_argument("--sec-user-agent", default=os.environ.get("SEC_USER_AGENT", ""),
                    help="required with --enrich, e.g. 'Acme Corp data-team@acme.com' (or set SEC_USER_AGENT)")
    ap.add_argument("--cache-dir", default="edgar_cache", type=Path)
    ap.add_argument("--fail-on-issues", action="store_true",
                    help="exit non-zero if any issues are found (useful in CI/cron alerting)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load(args.input, args.sheet)
    df = clean(df)
    df = classify(df)
    fills = None
    if args.enrich:
        from edgar_enrich import enrich
        df, fills = enrich(df, args.sec_user_agent, args.cache_dir)
    df = derive(df)
    issues = validate(df)
    summary = export(df, issues, args.outdir, fills)

    log.info("Summary: %s", json.dumps(summary, default=str))
    if args.fail_on_issues and len(issues):
        raise SystemExit(f"{len(issues)} data-quality issues found - see the Issues sheet in {summary['output_file']}")


if __name__ == "__main__":
    main()