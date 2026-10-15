# Next steps discussed with TA 
T-test for statistical significance
Regression + random forest 
CV


# IPO Data Pipeline

Cleans and checks a hand-compiled table of biotech IPOs, and optionally fills blank fields from SEC EDGAR.

It reads your spreadsheet export, **never edits it**, and writes a new Excel workbook with the cleaned data, the rows that aren't real IPOs, a list of data-quality problems, and a log of anything filled in automatically.

## Files

| File | Purpose |
|---|---|
| `ipo_pipeline.py` | Main script: load, clean, classify, derive, validate, export |
| `edgar_enrich.py` | Optional SEC EDGAR step, used only with `--enrich` |
| `Dataset.csv` (yours) | Input data |

Keep the `.py` files in the same folder.

## Setup

Requires Python 3.9+ and pandas 2.0 or newer.

```
python3 -m pip install pandas openpyxl requests
```

(`requests` is only needed for `--enrich`.)

## Usage

```
# Checks only (no internet needed)
python3 ipo_pipeline.py --input "Dataset.csv"

# Checks + fill blanks from SEC EDGAR
python3 ipo_pipeline.py --input "Dataset.csv" --enrich --sec-user-agent "Your Name you@school.edu"
```

| Option | Meaning |
|---|---|
| `--input` | Required. `.csv`, `.tsv` or `.xlsx` |
| `--sheet` | Sheet name or index for `.xlsx` (default: first sheet) |
| `--outdir` | Output folder (default: `output/`) |
| `--enrich` | Fill blank fields from SEC EDGAR |
| `--sec-user-agent` | Required with `--enrich`: your name and email. The SEC rejects requests without it. Can also be set as the `SEC_USER_AGENT` environment variable |
| `--cache-dir` | Where downloaded SEC responses are cached (default: `edgar_cache/`) |
| `--fail-on-issues` | Exit with an error if any issues are found (useful for scheduled jobs) |

## What the script does, step by step

### 1. Load
- Reads CSV (tries UTF-8, falls back to Windows cp1252), TSV or Excel.
- Normalises header names: removes hidden non-breaking spaces (e.g. `IPO Date`) and extra whitespace.
- Drops blank rows and footnote rows at the bottom ("Note: ..."), i.e. rows with no company or no row number.
- Stops with an error if required columns are missing (`Companies`, `Deal Date`, `Deal Size ($M)`, `Offer Price ($)`).

### 2. Clean
- Adds any optional columns that are absent so later checks don't crash.
- Trims whitespace; turns empty strings into blanks.
- Parses dates in mixed formats (`6/5/2020`, `1-Oct-04`); unparseable dates become blank.
- Converts numeric columns, handling CSV text such as `2,972.58`, `$23.00`, `(1.5)` and `60.9%` (percent becomes a decimal, 0.609).
- Splits `Legend Biotech USA (NAS: LEGN)` into `Company`, `Exchange` and `Ticker`.

### 3. Classify
Marks each row as a conventional underwritten IPO or not. A row is **excluded** if its `Status` or `Lead Underwriters` mentions a spin-off, "not an IPO", reverse merger or uplisting, or if the lead underwriters begin with "None". Excluded rows go to their own sheet instead of the analysis set (in the current dataset: Aptevo, Circle8, Dyadic, GRAIL, Hepion, Insmed).

It also flags `needs_verification` when `Source / Notes` contains wording like "verify" or "not confirmed".

### 4. Enrich from SEC EDGAR (only with `--enrich`)
For real IPOs with a blank in any of these fields: `SEC S-1 filing date`, `Filed Range Low/High ($)`, `Offer Price ($)`, `Shares Offered (M)`.

1. **Find the company** on EDGAR by ticker, then by fuzzy name match.
2. **Pull its filing list** from the SEC submissions API.
3. **Pick filings** relative to the IPO date:
   - Final prospectus (424B4 or 424B1) filed from 3 days before to 7 days after the IPO date.
   - S-1 or F-1: the earliest original filing within 400 days before the IPO (this ignores older abandoned attempts).
   - Price range: the first S-1/A, S-1, F-1 or F-1/A (up to four checked) that states a range.
4. **Fill** only blank cells. The S-1 date comes from EDGAR metadata (reliable). Prices and share counts are read from filing text with pattern matching (usually right, but verify).
5. **Never overwrites** your values. If EDGAR's number differs from yours, your value stays and the difference is logged as a `CONFLICT`. (Conflicts are checked for numeric fields only.)

Responses are cached on disk and requests are rate-limited, so reruns are fast and SEC-friendly. A failure on one company is logged and the run continues.

### 5. Derive
Adds columns to the clean data:

| Column | Meaning |
|---|---|
| `price_vs_range` | `below`, `within` or `above` the filed range |
| `offer_vs_midpoint_pct` | Offer price relative to the range midpoint |
| `implied_proceeds_$M` | Shares offered × offer price |
| `ipo_year` | Year of the IPO |
| `days_s1_to_ipo` | Days between S-1 filing and IPO |
| `first_day_return_pct` | `First-Day Return Rate` × 100 |

### 6. Validate
Writes one row per problem to the Issues sheet. Checks apply to real IPOs only (except duplicates).

| Check | Flags when |
|---|---|
| `deal_size_mismatch` | Shares × price differs from Deal Size by more than 15% |
| `money_raised_mismatch` | Money Raised at IPO differs from Deal Size by more than 15% |
| `ipo_before_s1` | IPO date is earlier than the S-1 date |
| `deal_vs_ipo_date` | Deal Date and IPO Date differ by more than 5 days |
| `stale_s1` | S-1 filed more than 365 days before the IPO |
| `future_ipo_date` | IPO date is in the future |
| `range_inverted` | Filed range low is above high |
| `price_fields_disagree` | `IPO Share Price` and `Offer Price` differ |
| `implausible_first_day_return` | Below -95% or above +300% (probably a unit error) |
| `missing_field` | Blank filed range, first-day return or S-1 date |
| `flagged_in_notes` | Notes say to verify |
| `duplicate` | Same ticker and IPO date appear twice |

The thresholds are constants at the top of `ipo_pipeline.py`.

### 7. Export
Writes `output/ipo_pipeline_output_<timestamp>.xlsx` with these sheets (Arial, styled header, filters, frozen top row):

| Sheet | Contents |
|---|---|
| `Summary` | Row counts, issues by type, price-vs-range counts, median first-day return |
| `Clean` | Real IPOs with the derived columns |
| `Excluded` | Rows that are not conventional IPOs |
| `Issues` | Every data-quality problem found |
| `Fills_Log` | With `--enrich`: each fill or conflict with old value, new value, SEC link, confidence and notes |

It also writes `output/run_summary.json` for monitoring.

## Limitations

- **First-day return is not filled.** EDGAR does not have it. Those cells stay blank and show up as `missing_field` issues.
- **Old filings:** deals before about 1996 (e.g. Amgen 1983) are not on EDGAR.
- **Renamed or delisted companies** may not be found by ticker or name and are logged as skipped.
- **Prices and share counts are regex-parsed** from filing text. Spot-check a sample using the links in `Fills_Log`.
- The script flags problems but does not fix your source data. Review the Issues and Fills_Log sheets and update your sheet yourself.

## Troubleshooting

| Message | Fix |
|---|---|
| `the following arguments are required: --input` | Add `--input "yourfile.csv"` |
| `No such file or directory` | Wrong path. Drag the file into Terminal to paste its full path |
| `Missing required columns` | Header names differ from the expected ones; check row 1 of your sheet |
| `No module named 'edgar_enrich'` | `edgar_enrich.py` must be in the same folder as `ipo_pipeline.py` |
| `No module named 'requests'` (or pandas, openpyxl) | `python3 -m pip install <name>` |
| `SEC requires a User-Agent with contact info` | Pass `--sec-user-agent "Name email@domain"` |
| `format="mixed"` error | `python3 -m pip install -U pandas` |

## Automating it

Run the same command on a schedule (cron, Windows Task Scheduler or a GitHub Action) against a freshly downloaded copy of the sheet. Add `--fail-on-issues` if you want the job to fail when new problems appear.
