import json

import pandas as pd

from build_overview import r, records
from northwind import (
    LAST_FULL_MONTH, ROOT, build_order_lines, load_tables,
)

OUT_JSON = ROOT / "site" / "data" / "q2_products.json"

# Premium = top decile of list price.
PREMIUM_QUANTILE = 0.9
STAR_PRODUCT = "Côte de Blaye"
# Most recent three full months, used for the current run-rate.
RECENT = (pd.Period("1998-02", freq="M"), LAST_FULL_MONTH)


def product_table(t, lines):
    p = t["products"].merge(
        t["categories"][["CategoryID", "CategoryName"]], on="CategoryID"
    ).merge(
        t["suppliers"][["SupplierID", "CompanyName"]]
        .rename(columns={"CompanyName": "Supplier"}), on="SupplierID"
    )
    recent = lines[lines["Month"].between(*RECENT)]
    sales = lines.groupby("ProductID").agg(
        Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
        Orders=("OrderID", "nunique"), Customers=("CustomerID", "nunique"),
        LastSale=("OrderDate", "max"),
    )
    p = p.merge(sales, on="ProductID", how="left")
    p["RecentRevenue"] = p["ProductID"].map(
        recent.groupby("ProductID")["Revenue"].sum()
    ).fillna(0)
    p["Share"] = p["Revenue"] / p["Revenue"].sum()
    p["RecentShare"] = p["RecentRevenue"] / recent["Revenue"].sum()
    threshold = p["Price"].quantile(PREMIUM_QUANTILE)
    p["Premium"] = p["Price"] >= threshold
    p["Group"] = p.apply(
        lambda row: "Discontinued" if row["Discontinued"]
        else ("Premium" if row["Premium"] else "Other"), axis=1
    )
    p = p.sort_values("Revenue", ascending=False).reset_index(drop=True)
    p["Rank"] = p.index + 1
    p["CumShare"] = p["Share"].cumsum()
    return p, threshold


def quarterly_shares(lines, products):
    full = lines[lines["Month"] <= LAST_FULL_MONTH].copy()
    full["Quarter"] = full["OrderDate"].dt.to_period("Q")
    premium_ids = set(products.loc[products["Premium"], "ProductID"])
    full["IsPremium"] = full["ProductID"].isin(premium_ids)
    q = full.groupby("Quarter").apply(
        lambda d: pd.Series({
            "Revenue": d["Revenue"].sum(),
            "PremiumShare": d.loc[d["IsPremium"], "Revenue"].sum()
            / d["Revenue"].sum(),
            "DiscontinuedShare": d.loc[d["Discontinued"], "Revenue"].sum()
            / d["Revenue"].sum(),
        }),
        include_groups=False,
    ).reset_index()
    q["Months"] = q["Quarter"].map(
        full.groupby(full["OrderDate"].dt.to_period("Q"))["Month"].nunique()
    )
    q["Quarter"] = q["Quarter"].astype(str)
    return q


def star_product(lines):
    full = lines[lines["Month"] <= LAST_FULL_MONTH]
    star = full[full["ProductName"] == STAR_PRODUCT]
    months = pd.period_range(full["Month"].min(), LAST_FULL_MONTH, freq="M")
    total = full.groupby("Month")["Revenue"].sum().reindex(months)
    star_m = star.groupby("Month")["Revenue"].sum().reindex(months,
                                                            fill_value=0)
    monthly = pd.DataFrame({
        "Month": months.astype(str), "Revenue": star_m.values,
        "ShareOfMonth": (star_m / total).values,
    })
    all_star = lines[lines["ProductName"] == STAR_PRODUCT]
    buyers = all_star.groupby("Customer")["Revenue"].sum().sort_values(
        ascending=False)
    return {
        "monthly": records(monthly),
        "monthsWithoutSales": int((star_m == 0).sum()),
        "monthsTotal": int(len(months)),
        "maxShareOfMonth": r(monthly["ShareOfMonth"].max(), 4),
        "orders": int(all_star["OrderID"].nunique()),
        "customers": int(all_star["CustomerID"].nunique()),
        "top3Buyers": buyers.head(3).index.tolist(),
        "top3BuyerShare": r(buyers.head(3).sum() / buyers.sum(), 4),
    }


def category_exposure(products):
    c = products.groupby("CategoryName").apply(
        lambda d: pd.Series({
            "Active": d.loc[~d["Discontinued"], "Revenue"].sum(),
            "DiscontinuedRev": d.loc[d["Discontinued"], "Revenue"].sum(),
            "ActiveProducts": int((~d["Discontinued"]).sum()),
            "DiscontinuedProducts": int(d["Discontinued"].sum()),
        }),
        include_groups=False,
    ).reset_index()
    c["DiscontinuedShare"] = c["DiscontinuedRev"] / (
        c["Active"] + c["DiscontinuedRev"])
    return c.sort_values("DiscontinuedRev", ascending=False)


def customer_exposure(lines):
    c = lines.groupby(["CustomerID", "Customer", "CustomerCountry"]).apply(
        lambda d: pd.Series({
            "Total": d["Revenue"].sum(),
            "DiscontinuedRev": d.loc[d["Discontinued"], "Revenue"].sum(),
        }),
        include_groups=False,
    ).reset_index()
    c["DiscontinuedShare"] = c["DiscontinuedRev"] / c["Total"]
    return c.sort_values("DiscontinuedRev", ascending=False)


def supplier_exposure(products):
    s = products[products["Discontinued"]].groupby("Supplier").agg(
        Products=("ProductName", lambda x: ", ".join(x)),
        Revenue=("Revenue", "sum"),
        RecentRevenue=("RecentRevenue", "sum"),
    ).reset_index()
    return s.sort_values("Revenue", ascending=False)


def main():
    t = load_tables()
    lines = build_order_lines(t)
    products, threshold = product_table(t, lines)
    disc = products[products["Discontinued"]]
    prem = products[products["Premium"]]
    recent = lines[lines["Month"].between(*RECENT)]
    recent_disc = recent.loc[recent["Discontinued"], "Revenue"].sum()
    n_months = len(pd.period_range(*RECENT, freq="M"))
    cats = category_exposure(products)
    custs = customer_exposure(lines)

    data = {
        "premiumThreshold": r(threshold),
        "recentWindow": [str(RECENT[0]), str(RECENT[1])],
        "kpis": {
            "top10Share": r(products["Share"].head(10).sum(), 4),
            "productsForHalf": int((products["CumShare"] < 0.5).sum() + 1),
            "premiumCount": int(len(prem)),
            "premiumShare": r(prem["Share"].sum(), 4),
            "premiumUnitShare": r(prem["Units"].sum()
                                  / products["Units"].sum(), 4),
            "starShare": r(products.loc[
                products["ProductName"] == STAR_PRODUCT, "Share"].iloc[0], 4),
            "discontinuedCount": int(len(disc)),
            "discontinuedShare": r(disc["Share"].sum(), 4),
            "discontinuedRecentShare": r(recent_disc
                                         / recent["Revenue"].sum(), 4),
            "annualRevenueAtRisk": r(recent_disc / n_months * 12),
            "discontinuedSoldOut": int((disc["UnitsInStock"] == 0).sum()),
            "premiumDiscontinued": disc.loc[disc["Premium"],
                                            "ProductName"].tolist(),
            "customersExposed": int((custs["DiscontinuedRev"] > 0).sum()),
            "customersOverQuarter": int(
                (custs["DiscontinuedShare"] > 0.25).sum()),
            "customersTotal": int(len(custs)),
        },
        "products": records(products[[
            "Rank", "ProductID", "ProductName", "CategoryName", "Supplier",
            "Price", "Revenue", "Share", "CumShare", "RecentRevenue",
            "RecentShare", "Units", "Customers", "UnitsInStock",
            "UnitsOnOrder", "ReorderLevel", "Premium", "Discontinued",
            "Group", "LastSale",
        ]]),
        "quarterly": records(quarterly_shares(lines, products)),
        "star": star_product(lines),
        "categories": records(cats),
        "customers": records(custs.head(12)),
        "suppliers": records(supplier_exposure(products)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1, default=str))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
