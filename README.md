# Northwind Traders: three business questions

15.561 · Homework 3 · MIT Sloan, Fall 2026

An analysis of the Northwind Traders database (a specialty-foods wholesaler, July 1996 – May 1998), published as a single web page. Every number, chart and table on the page is computed by the Python scripts in this repository.

**Live page:** https://sianm34mit.github.io/sianm34-15561-hw2/ (served from the `docs/` folder by GitHub Pages)

**Repository:** https://github.com/sianm34MIT/sianm34-15561-hw2

## The questions

| | Question | Script |
|---|---|---|
| Q1 | What is driving Northwind's revenue growth, and how dependent is it on a few customers? | `src/q1_growth_drivers.py` |
| Q2 | How much of Northwind's revenue rests on a few products, and how much of it is about to disappear? | `src/q2_product_risk.py` |
| Q3 | Is there a relationship between seller and sales? Would customers buy more if each had a dedicated seller? | `src/q3_relationship_selling.py` |

The page also has an Overview and four "Explore" tabs (products, suppliers, employees, orders & customers), produced by `src/build_overview.py`.

## Reproduce the results

Requires Python 3.10 or newer.

```bash
git clone https://github.com/sianm34MIT/sianm34-15561-hw2.git
cd sianm34-15561-hw2
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_all.py
```

`run_all.py` runs the steps below in order. Each script can also be run on its own (from the repository root).

| Step | Script | Output |
|---|---|---|
| 1 | `src/build_overview.py` | `site/data/overview.json`: Overview and Explore tabs |
| 2 | `src/q1_growth_drivers.py` | `site/data/q1_growth.json` |
| 3 | `src/q2_product_risk.py` | `site/data/q2_products.json` |
| 4 | `src/q3_relationship_selling.py` | `site/data/q3_relationships.json` (about 1–2 minutes: 1,000 reshuffles and 2,000 bootstrap samples, fixed random seed) |
| 5 | `src/build_site.py` | `docs/index.html`: inserts all four JSON files into `site/template.html` |

Open `docs/index.html` in a browser to view the page locally. The charts load Chart.js 4.4.1 from cdnjs, so an internet connection is needed for them to appear.

Because the random seeds are fixed, rerunning the scripts reproduces the published figures exactly.

## Checking the numbers

`audit/audit.py` recomputes about 45 headline figures directly from the raw workbook, without using any of the analysis code, and compares them with the JSON outputs:

```bash
python audit/audit.py
```

It ends with `FAILURES: none` when everything matches.

## Data

`data/northwind.xlsx` is an export of the course copy of the Northwind Database Google Sheet. Decisions made when preparing it:

- **Revenue** is net of discounts: `UnitPrice × Quantity × (1 − Discount)` from the Orders Details sheet, using the price charged on each order rather than the Product sheet's list price (they differ on 658 order lines). Freight is excluded.
- **Partial periods.** Orders run from 4 July 1996 to 6 May 1998. May 1998 (14 orders over 6 days) is left out of time-series charts, and comparisons use matching calendar months rather than annual totals.
- **Product sheet helper columns** (`Cheap`, `NewPrice`, `IsCheap` and two others) are not part of the Northwind schema and are dropped.
- **Customer IDs.** On 4 orders (10248, 10249, 10260, 10808; $5.2k), the CustomerID doesn't match the ship-to company. They are kept as recorded. Reassigning them changes only one figure (new customers in Q1: 10 instead of 9).
- **Country** means the customer's registered country, not the ship-to country.
- **Late** means shipped after the required date. 21 unshipped orders are excluded from shipping measures.

## Repository layout

```
data/northwind.xlsx      source data
src/northwind.py         loading and shared calculations (order lines, order totals)
src/build_overview.py    Overview and Explore tabs
src/q1_growth_drivers.py Q1 analysis
src/q2_product_risk.py   Q2 analysis
src/q3_relationship_selling.py  Q3 analysis
src/build_site.py        builds docs/index.html from site/template.html and the JSON outputs
src/run_all.py           runs everything in order
site/template.html       page layout, styles and chart code
site/data/*.json         analysis outputs used by the page
docs/index.html          the published page (GitHub Pages)
audit/audit.py           independent check of headline figures
```
