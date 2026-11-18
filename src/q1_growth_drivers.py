import json

import numpy as np
import pandas as pd

from build_overview import r, records
from northwind import (
    LAST_FULL_MONTH, ROOT, build_order_lines, build_order_summary,
    load_tables,
)

OUT_JSON = ROOT / "site" / "data" / "q1_growth.json"

# Same ten calendar months, one year apart, so seasonality cancels out.
BEFORE = ("1996-07-01", "1997-04-30")
AFTER = ("1997-07-01", "1998-04-30")


def in_window(dates, window):
    start, end = (pd.Timestamp(d) for d in window)
    return (dates >= start) & (dates <= end + pd.Timedelta(days=1)
                               - pd.Timedelta(seconds=1))


def period_metrics(orders, lines):
    revenue = orders["Revenue"].sum()
    n_orders = len(orders)
    customers = orders["CustomerID"].nunique()
    return {
        "revenue": revenue,
        "orders": n_orders,
        "customers": customers,
        "ordersPerCustomer": n_orders / customers,
        "avgOrderValue": revenue / n_orders,
        "linesPerOrder": len(lines) / n_orders,
        "unitsPerLine": lines["Quantity"].mean(),
        "pricePerUnit": lines["GrossRevenue"].sum() / lines["Quantity"].sum(),
        "discountRate": lines["DiscountAmount"].sum()
        / lines["GrossRevenue"].sum(),
    }


def decompose(before, after):
    factors = {
        "Active customers": after["customers"] / before["customers"],
        "Orders per customer": (after["ordersPerCustomer"]
                                / before["ordersPerCustomer"]),
        "Average order value": (after["avgOrderValue"]
                                / before["avgOrderValue"]),
    }
    total_log = np.log(after["revenue"] / before["revenue"])
    delta = after["revenue"] - before["revenue"]
    steps = []
    for name, factor in factors.items():
        share = np.log(factor) / total_log
        steps.append({
            "driver": name,
            "change": r(factor - 1, 4),
            "shareOfGrowth": r(share, 4),
            "dollars": r(share * delta),
        })
    return steps


def monthly_drivers(orders):
    full = orders[orders["Month"] <= LAST_FULL_MONTH]
    m = full.groupby("Month").agg(
        Orders=("OrderID", "count"),
        Customers=("CustomerID", "nunique"),
        Revenue=("Revenue", "sum"),
    ).reset_index()
    m["AvgOrderValue"] = m["Revenue"] / m["Orders"]
    m["Month"] = m["Month"].astype(str)
    return m


def customer_contribution(t, orders):
    first_order = orders.groupby("CustomerID")["OrderDate"].min()
    before = orders[in_window(orders["OrderDate"], BEFORE)]
    after = orders[in_window(orders["OrderDate"], AFTER)]
    c = pd.DataFrame({
        "Before": before.groupby("CustomerID")["Revenue"].sum(),
        "After": after.groupby("CustomerID")["Revenue"].sum(),
        "OrdersBefore": before.groupby("CustomerID")["OrderID"].count(),
        "OrdersAfter": after.groupby("CustomerID")["OrderID"].count(),
    }).fillna(0)
    c["Change"] = c["After"] - c["Before"]
    c = c.join(t["customers"].set_index("CustomerID")[
        ["CompanyName", "Country"]])
    c["FirstOrder"] = first_order.reindex(c.index)

    def segment(row):
        if row["Before"] > 0 and row["After"] > 0:
            return "Returning"
        if row["After"] > 0 and row["FirstOrder"] > pd.Timestamp(BEFORE[1]):
            return "New"
        if row["After"] > 0:
            return "Returning after a gap"
        return "Lapsed"

    c["Segment"] = c.apply(segment, axis=1)
    c = c.sort_values("Change", ascending=False).reset_index()
    c["CumShareOfGrowth"] = c["Change"].cumsum() / c["Change"].sum()

    seg = c.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Before=("Before", "sum"), After=("After", "sum"),
        OrdersBefore=("OrdersBefore", "sum"),
        OrdersAfter=("OrdersAfter", "sum"),
    ).reset_index()
    seg["Change"] = seg["After"] - seg["Before"]
    seg["ShareOfGrowth"] = seg["Change"] / seg["Change"].sum()
    seg["ShareOfAfter"] = seg["After"] / seg["After"].sum()
    return c, seg


def country_contribution(t, orders):
    """Revenue change by the customer's registered country."""
    country = t["customers"].set_index("CustomerID")["Country"]
    o = orders.assign(Country=orders["CustomerID"].map(country))
    before = o[in_window(o["OrderDate"], BEFORE)]
    after = o[in_window(o["OrderDate"], AFTER)]
    c = pd.DataFrame({
        "Before": before.groupby("Country")["Revenue"].sum(),
        "After": after.groupby("Country")["Revenue"].sum(),
    }).fillna(0)
    c["Change"] = c["After"] - c["Before"]
    c["Growth"] = np.where(c["Before"] > 0, c["After"] / c["Before"] - 1,
                           np.nan)
    c["ShareOfGrowth"] = c["Change"] / c["Change"].sum()
    return c.sort_values("Change", ascending=False).reset_index()


def main():
    t = load_tables()
    lines = build_order_lines(t)
    orders = build_order_summary(t, lines)

    o_b = orders[in_window(orders["OrderDate"], BEFORE)]
    o_a = orders[in_window(orders["OrderDate"], AFTER)]
    l_b = lines[in_window(lines["OrderDate"], BEFORE)]
    l_a = lines[in_window(lines["OrderDate"], AFTER)]
    before = period_metrics(o_b, l_b)
    after = period_metrics(o_a, l_a)

    cust, seg = customer_contribution(t, orders)
    top10 = cust["Change"].head(10).sum() / cust["Change"].sum()
    n_half = int((cust["CumShareOfGrowth"] < 0.5).sum() + 1)

    data = {
        "windows": {"before": BEFORE, "after": AFTER},
        "before": {k: r(v, 4) for k, v in before.items()},
        "after": {k: r(v, 4) for k, v in after.items()},
        "decomposition": decompose(before, after),
        "monthly": records(monthly_drivers(orders)),
        "segments": records(seg),
        "customers": records(cust[[
            "CustomerID", "CompanyName", "Country", "Segment", "Before",
            "After", "OrdersBefore", "OrdersAfter", "Change",
            "CumShareOfGrowth",
        ]]),
        "countries": records(country_contribution(t, orders)),
        "concentration": {
            "top10Share": r(top10, 4),
            "customersForHalf": n_half,
            "customersGrowing": int((cust["Change"] > 0).sum()),
            "customersShrinking": int((cust["Change"] < 0).sum()),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1, default=str))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
