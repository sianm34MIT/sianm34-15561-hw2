import json
from pathlib import Path

import numpy as np
import pandas as pd

from northwind import (
    LAST_FULL_MONTH, ROOT, build_order_lines, build_order_summary,
    load_tables,
)

OUT_JSON = ROOT / "site" / "data" / "overview.json"


def r(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


def records(df):
    out = df.replace({np.nan: None}).to_dict(orient="records")
    for row in out:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = round(float(v), 2)
            elif isinstance(v, (np.bool_,)):
                row[k] = bool(v)
            elif isinstance(v, pd.Timestamp):
                row[k] = v.strftime("%Y-%m-%d")
    return out


def overview_section(t, lines, orders):
    monthly = (
        lines[lines["Month"] <= LAST_FULL_MONTH]
        .groupby("Month")
        .agg(Revenue=("Revenue", "sum"), Orders=("OrderID", "nunique"))
        .reset_index()
    )
    monthly["Month"] = monthly["Month"].astype(str)
    # Exponential growth trend: straight-line fit to log revenue.
    x = np.arange(len(monthly))
    slope, intercept = np.polyfit(x, np.log(monthly["Revenue"]), 1)
    monthly["Trend"] = np.exp(intercept + slope * x)
    growth = {
        "monthlyRate": r(np.exp(slope) - 1, 4),
        "annualRate": r(np.exp(slope * 12) - 1, 4),
    }
    by_cat = (
        lines.groupby("CategoryName")["Revenue"].sum()
        .sort_values(ascending=False).reset_index()
    )
    by_year = orders.groupby(orders["OrderDate"].dt.year).agg(
        Orders=("OrderID", "count"), Revenue=("Revenue", "sum"),
        FirstDate=("OrderDate", "min"), LastDate=("OrderDate", "max"),
    ).reset_index().rename(columns={"OrderDate": "Year"})
    by_year["Months"] = (
        orders.groupby(orders["OrderDate"].dt.year)["Month"].nunique().values
    )
    return {
        "kpis": {
            "revenue": r(lines["Revenue"].sum()),
            "discounts": r(lines["DiscountAmount"].sum()),
            "orders": int(orders["OrderID"].nunique()),
            "avgOrder": r(orders["Revenue"].mean()),
            "customersActive": int(orders["CustomerID"].nunique()),
            "customers": int(len(t["customers"])),
            "products": int(len(t["products"])),
            "suppliers": int(len(t["suppliers"])),
            "employees": int(len(t["employees"])),
            "firstOrder": orders["OrderDate"].min().strftime("%Y-%m-%d"),
            "lastOrder": orders["OrderDate"].max().strftime("%Y-%m-%d"),
        },
        "monthly": records(monthly),
        "growth": growth,
        "byCategory": records(by_cat),
        "byYear": records(by_year),
        "tables": [
            {"name": "Orders", "rows": len(t["orders"]),
             "grain": "One row per order"},
            {"name": "Orders Details", "rows": len(t["order_details"]),
             "grain": "One row per product within an order"},
            {"name": "Customers", "rows": len(t["customers"]),
             "grain": "One row per customer company"},
            {"name": "Product", "rows": len(t["products"]),
             "grain": "One row per product"},
            {"name": "Categories", "rows": len(t["categories"]),
             "grain": "One row per product category"},
            {"name": "Suppliers", "rows": len(t["suppliers"]),
             "grain": "One row per supplier company"},
            {"name": "Employees", "rows": len(t["employees"]),
             "grain": "One row per employee"},
            {"name": "Shippers", "rows": len(t["shippers"]),
             "grain": "One row per shipping company"},
        ],
    }


def products_section(t, lines):
    p = t["products"].merge(
        t["categories"][["CategoryID", "CategoryName"]], on="CategoryID"
    ).merge(
        t["suppliers"][["SupplierID", "CompanyName"]]
        .rename(columns={"CompanyName": "Supplier"}), on="SupplierID"
    )
    sales = lines.groupby("ProductID").agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        Orders=("OrderID", "nunique"),
        AvgDiscount=("Discount", "mean"),
        AvgSellPrice=("UnitPrice", "mean"),
    )
    p = p.merge(sales, on="ProductID", how="left")
    p["RevenueShare"] = p["Revenue"] / p["Revenue"].sum()
    active = ~p["Discontinued"]
    p["NeedsReorder"] = active & (
        p["UnitsInStock"] + p["UnitsOnOrder"] <= p["ReorderLevel"]
    )
    p["BelowReorder"] = active & (p["UnitsInStock"] <= p["ReorderLevel"])
    p["OutOfStock"] = active & (p["UnitsInStock"] == 0)

    by_cat = p.groupby("CategoryName").agg(
        Products=("ProductID", "count"),
        Discontinued=("Discontinued", "sum"),
        AvgPrice=("Price", "mean"),
        Revenue=("Revenue", "sum"),
        Units=("Units", "sum"),
        StockUnits=("UnitsInStock", "sum"),
    ).reset_index().sort_values("Revenue", ascending=False)
    by_cat["RevenueShare"] = by_cat["Revenue"] / by_cat["Revenue"].sum()
    by_cat["AvgDiscount"] = by_cat["CategoryName"].map(
        lines.groupby("CategoryName")["Discount"].mean()
    )

    bins = [0, 10, 20, 30, 40, 50, 75, 100, 300]
    labels = ["$0–10", "$10–20", "$20–30", "$30–40", "$40–50",
              "$50–75", "$75–100", "$100+"]
    hist = pd.cut(p["Price"], bins=bins, labels=labels, right=False)
    price_hist = hist.value_counts().reindex(labels).reset_index()
    price_hist.columns = ["Band", "Products"]

    top = p.sort_values("Revenue", ascending=False)
    cum = top["Revenue"].cumsum() / top["Revenue"].sum()
    n80 = int((cum < 0.8).sum() + 1)

    cols = ["ProductID", "ProductName", "CategoryName", "Supplier", "Price",
            "Units", "Orders", "Revenue", "RevenueShare", "AvgDiscount",
            "UnitsInStock", "UnitsOnOrder", "ReorderLevel", "Discontinued",
            "NeedsReorder", "BelowReorder"]
    return {
        "kpis": {
            "products": int(len(p)),
            "active": int(active.sum()),
            "discontinued": int(p["Discontinued"].sum()),
            "medianPrice": r(p["Price"].median()),
            "minPrice": r(p["Price"].min()),
            "maxPrice": r(p["Price"].max()),
            "belowReorder": int(p["BelowReorder"].sum()),
            "needsReorder": int(p["NeedsReorder"].sum()),
            "outOfStock": int(p["OutOfStock"].sum()),
            "productsFor80": n80,
            "top10Share": r(top["Revenue"].head(10).sum()
                            / top["Revenue"].sum(), 4),
            "discontinuedRevenueShare": r(
                p.loc[p["Discontinued"], "Revenue"].sum()
                / p["Revenue"].sum(), 4),
        },
        "byCategory": records(by_cat),
        "priceHist": records(price_hist),
        "table": records(top[cols]),
    }


def suppliers_section(t, lines):
    s = t["suppliers"].copy()
    prod = t["products"].merge(
        t["categories"][["CategoryID", "CategoryName"]], on="CategoryID"
    )
    pagg = prod.groupby("SupplierID").agg(
        Products=("ProductID", "count"),
        Discontinued=("Discontinued", "sum"),
        Categories=("CategoryName",
                    lambda x: ", ".join(sorted(x.unique()))),
        AvgPrice=("Price", "mean"),
    )
    sales = lines.groupby("SupplierID").agg(
        Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
        Orders=("OrderID", "nunique"), Customers=("CustomerID", "nunique"),
    )
    s = s.merge(pagg, on="SupplierID", how="left").merge(
        sales, on="SupplierID", how="left"
    )
    s["RevenueShare"] = s["Revenue"] / s["Revenue"].sum()
    s = s.sort_values("Revenue", ascending=False)

    by_country = s.groupby("Country").agg(
        Suppliers=("SupplierID", "count"), Products=("Products", "sum"),
        Revenue=("Revenue", "sum"),
    ).reset_index().sort_values("Revenue", ascending=False)
    by_country["RevenueShare"] = (
        by_country["Revenue"] / by_country["Revenue"].sum()
    )

    shares = s["RevenueShare"].fillna(0)
    cols = ["SupplierID", "CompanyName", "Country", "Products",
            "Discontinued", "Categories", "AvgPrice", "Units", "Orders",
            "Customers", "Revenue", "RevenueShare"]
    return {
        "kpis": {
            "suppliers": int(len(s)),
            "countries": int(s["Country"].nunique()),
            "top5Share": r(shares.head(5).sum(), 4),
            "hhi": r((shares ** 2).sum() * 10000, 0),
            "singleProduct": int((s["Products"] == 1).sum()),
            "medianProducts": r(s["Products"].median(), 1),
            "withDiscontinued": int((s["Discontinued"] > 0).sum()),
        },
        "byCountry": records(by_country),
        "table": records(s[cols]),
    }


def employees_section(t, lines, orders):
    e = t["employees"].copy()
    o = orders.copy()
    agg = o.groupby("EmployeeID").agg(
        Orders=("OrderID", "count"),
        Revenue=("Revenue", "sum"),
        AvgOrder=("Revenue", "mean"),
        Customers=("CustomerID", "nunique"),
        Countries=("ShipCountry", "nunique"),
        ShippedOrders=("Shipped", "sum"),
        LateOrders=("Late", "sum"),
        AvgDaysToShip=("DaysToShip", "mean"),
        FirstOrder=("OrderDate", "min"),
    )
    disc = lines.groupby("EmployeeID").agg(
        AvgDiscount=("Discount", "mean"),
        DiscountedShare=("Discount", lambda x: (x > 0).mean()),
        DiscountGiven=("DiscountAmount", "sum"),
    )
    e = e.merge(agg, on="EmployeeID", how="left").merge(
        disc, on="EmployeeID", how="left"
    )
    e["LateRate"] = e["LateOrders"] / e["ShippedOrders"]
    e["RevenueShare"] = e["Revenue"] / e["Revenue"].sum()
    names = e.set_index("EmployeeID")["Name"]
    e["Manager"] = e["ReportsTo"].map(names)
    e = e.sort_values("Revenue", ascending=False)

    monthly = (
        o[o["Month"] <= LAST_FULL_MONTH]
        .assign(Quarter=lambda d: d["OrderDate"].dt.to_period("Q").astype(str))
        .groupby(["EmployeeID", "Quarter"])["Revenue"].sum()
        .unstack(fill_value=0)
    )
    quarterly = []
    for emp_id, row in monthly.iterrows():
        quarterly.append({
            "EmployeeID": int(emp_id), "Name": names[emp_id],
            "values": [r(v) for v in row.values],
        })

    cols = ["EmployeeID", "Name", "Title", "City", "Country", "HireDate",
            "Manager", "Orders", "Revenue", "RevenueShare", "AvgOrder",
            "Customers", "Countries", "AvgDiscount", "DiscountedShare",
            "LateRate", "LateOrders", "ShippedOrders", "AvgDaysToShip"]
    return {
        "kpis": {
            "employees": int(len(e)),
            "salesReps": int((e["Title"] == "Sales Representative").sum()),
            "offices": sorted(e["Country"].unique().tolist()),
            "topShare": r(e["RevenueShare"].max(), 4),
            "topName": e.iloc[0]["Name"],
            "lateRate": r(o["Late"].sum() / o["Shipped"].sum(), 4),
            "avgDaysToShip": r(o["DaysToShip"].mean(), 1),
        },
        "quarters": [str(c) for c in monthly.columns],
        "quarterly": quarterly,
        "table": records(e[cols]),
    }


def orders_customers_section(t, lines, orders):
    full = orders[orders["Month"] <= LAST_FULL_MONTH]
    ship = orders.merge(
        t["shippers"].rename(columns={"ShipperID": "ShipVia",
                                      "CompanyName": "Shipper"}),
        on="ShipVia",
    ).groupby("Shipper").agg(
        Orders=("OrderID", "count"),
        AvgFreight=("Freight", "mean"),
        TotalFreight=("Freight", "sum"),
        AvgDaysToShip=("DaysToShip", "mean"),
        Late=("Late", "sum"),
        Shipped=("Shipped", "sum"),
    ).reset_index()
    ship["LateRate"] = ship["Late"] / ship["Shipped"]
    ship["FreightPctOfRevenue"] = ship["Shipper"].map(
        orders.merge(t["shippers"].rename(
            columns={"ShipperID": "ShipVia", "CompanyName": "Shipper"}),
            on="ShipVia").groupby("Shipper")
        .apply(lambda d: d["Freight"].sum() / d["Revenue"].sum(),
               include_groups=False)
    )

    days = orders["DaysToShip"].dropna()
    day_bins = [0, 3, 7, 14, 21, 28, 60]
    day_labels = ["0–2", "3–6", "7–13", "14–20", "21–27", "28+"]
    days_hist = pd.cut(days, bins=day_bins, labels=day_labels, right=False)
    days_hist = days_hist.value_counts().reindex(day_labels).reset_index()
    days_hist.columns = ["Days", "Orders"]

    cust = lines.groupby(["CustomerID", "Customer", "CustomerCountry"]).agg(
        Revenue=("Revenue", "sum"), Orders=("OrderID", "nunique"),
    ).reset_index().sort_values("Revenue", ascending=False)
    cust["Share"] = cust["Revenue"] / cust["Revenue"].sum()
    by_country = lines.groupby("CustomerCountry").agg(
        Revenue=("Revenue", "sum"), Customers=("CustomerID", "nunique"),
        Orders=("OrderID", "nunique"),
    ).reset_index().sort_values("Revenue", ascending=False)
    by_country["Share"] = by_country["Revenue"] / by_country["Revenue"].sum()

    return {
        "kpis": {
            "orders": int(len(orders)),
            "unshipped": int((~orders["Shipped"]).sum()),
            "lateRate": r(orders["Late"].sum() / orders["Shipped"].sum(), 4),
            "medianDaysToShip": r(days.median(), 1),
            "avgItemsPerOrder": r(
                lines.groupby("OrderID")["ProductID"].count().mean(), 1),
            "discountedLines": r((lines["Discount"] > 0).mean(), 4),
            "customerCountries": int(t["customers"]["Country"].nunique()),
            "top10CustomerShare": r(cust["Share"].head(10).sum(), 4),
            "noOrderCustomers": t["customers"].loc[
                ~t["customers"]["CustomerID"].isin(orders["CustomerID"]),
                "CompanyName"].tolist(),
            "monthsFull": int(full["Month"].nunique()),
        },
        "shippers": records(ship),
        "daysHist": records(days_hist),
        "topCustomers": records(cust.head(15)),
        "byCountry": records(by_country),
    }


def main():
    t = load_tables()
    lines = build_order_lines(t)
    orders = build_order_summary(t, lines)
    data = {
        "overview": overview_section(t, lines, orders),
        "products": products_section(t, lines),
        "suppliers": suppliers_section(t, lines),
        "employees": employees_section(t, lines, orders),
        "ordersCustomers": orders_customers_section(t, lines, orders),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1, default=str))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
