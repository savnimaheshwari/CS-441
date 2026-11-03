# IPO Data Pipeline

Turns the hand-compiled **initial** biotech IPO table (`Dataset.csv`, 44 rows) into the **final** modeling dataset (33 IPOs) with a change log. The input file is never edited.

## Files

| File | Purpose |
|---|---|
| `ipo_pipeline.py` | Main script. Self-contained, this is all you need to build the final dataset |
| `edgar_enrich.py` | Optional. Only needed with `--enrich` (fills blanks from SEC EDGAR) |
| `ipo_rf_analysis.py` | Optional. Nested-CV random forest that reads the pipeline's output |
| `Dataset.csv` | Input (not in the repo if private) |

## Setup

Python 3.9+, `pip install pandas openpyxl` (`requests` only for `--enrich`).

## Usage

```
python3 ipo_pipeline.py --input Dataset.csv --outdir output/
```

| Option | Meaning |
|---|---|
| `--input` | Required. `.csv`, `.tsv` or `.xlsx` |
| `--outdir` | Output folder (default `output/`) |
| `--sheet` | Sheet name or index for `.xlsx` |
| `--stem-cell-range` | U.S. Stem Cell filed range: `original` ($14-16, default) or `preliminary` ($6-8) |
| `--enrich` | Fill blank S-1 date / range / price / shares from SEC EDGAR (needs `edgar_enrich.py`) |
| `--sec-user-agent` | Required with `--enrich`, e.g. `"Your Name you@school.edu"` |
| `--fail-on-issues` | Exit with an error if any issues remain |

## What it does

1. **Load and clean:** fixes headers, drops blank/footnote rows, parses dates and numbers.
2. **Exclude (44 to 33):** removes spin-offs, reverse mergers and uplistings (6), and IPOs with no filed range or offer price (5). Each exclusion is listed with its reason.
3. **Reconcile:** `IPO Share Price` is taken from `Offer Price`; Money Raised is checked against Deal Size; a blank S-1 date is read from the notes column.
4. **Correct:** applies externally verified fixes (the `CORRECTIONS` table at the top of the script).
5. **Derive:** adds the outcome and derived columns below.
6. **Validate and export:** writes the data-quality issues and the workbook.

Every change in steps 3-4 (and any EDGAR fill) goes to the **Change Log** with old value, new value and source. Values that can't be verified are left blank, never estimated.

## Output

`output/ipo_final_dataset.xlsx`

| Sheet | Contents |
|---|---|
| `Summary` | Row counts (initial, excluded, final), outcome counts, issues by check |
| `Dataset` | The final dataset |
| `Excluded` | Removed rows and why |
| `Change Log` | Every reconciled, corrected or filled value |
| `Issues` | Data-quality problems still in the final dataset |

Also written: `ipo_final_dataset.csv` (the Dataset sheet) and `run_summary.json`.

**Dataset columns:** the 16 original columns, then:

| Column | Meaning |
|---|---|
| `Pricing Outcome (calc)` / `price_vs_range` | Above / Within / Below the filed range (capitalized / lowercase) |
| `days_s1_to_ipo` | IPO date minus S-1 filing date |
| `offer_vs_midpoint_pct` | Offer price vs. range midpoint, in percent |
| `implied_proceeds_$M` | Shares offered x offer price |
| `underwriter_count` | Number of named underwriters (team-supplied counts in `UW_COUNTS`) |

First-Day Return and Valuation at IPO are left out of the final dataset because they are only known after pricing.

## Notes

- Thresholds, the corrections table and the underwriter counts are constants at the top of `ipo_pipeline.py`.
- With `--enrich`, EDGAR runs before exclusion, so a row it fills can move into the final dataset. EDGAR prices and share counts are regex-parsed, so spot-check them using the URLs in the Change Log. Deals before about 1996 (e.g. Amgen) are not on EDGAR.
- To run the model: `python3 ipo_rf_analysis.py --input output/ipo_final_dataset.xlsx --out results.json`.