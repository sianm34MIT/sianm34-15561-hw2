"""Load the Northwind workbook and build shared order tables."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "northwind.xlsx"

# Orders after this date are an incomplete final month (May 1998).
LAST_FULL_MONTH = pd.Period("1998-04", freq="M")

# Leftover helper columns in the Product sheet, unrelated to the dataset.
PRODUCT_EXTRA_COLS = [
    "Cheap", "NewPrice", "Cheap.1", "IsCheap", "Unnamed: 14", "NewPrice.1",
]


def load_tables(path=DATA_PATH):
    """Read the workbook sheets used in the analysis.

    Drops the Categories ``Picture`` column and the Product sheet helper
    columns, and adds a ``Name`` (first + last) column to employees.

    Args:
        path (pathlib.Path): Northwind workbook (.xlsx).

    Returns:
        dict[str, pandas.DataFrame]: Tables keyed by ``categories``,
            ``customers``, ``employees``, ``order_details``, ``orders``,
            ``shippers``, ``products`` and ``suppliers``.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    tables = {
        "categories": sheets["Categories"].drop(columns=["Picture"]),
        "customers": sheets["Customers"],
        "employees": sheets["Employees"],
        "order_details": sheets["Orders Details"],
        "orders": sheets["Orders"],
        "shippers": sheets["Shippers"],
        "products": sheets["Product"].drop(
            columns=PRODUCT_EXTRA_COLS, errors="ignore"
        ),
        "suppliers": sheets["Suppliers"],
    }
    tables["employees"]["Name"] = (
        tables["employees"]["FirstName"] + " "
        + tables["employees"]["LastName"]
    )
    return tables


def build_order_lines(t):
    """Return one row per order line with revenue and lookups attached.

    Revenue is net of discount at the charged price:
    ``UnitPrice * Quantity * (1 - Discount)``.

    Args:
        t (dict[str, pandas.DataFrame]): Tables from ``load_tables``.

    Returns:
        pandas.DataFrame: Order details plus ``GrossRevenue``,
            ``Revenue``, ``DiscountAmount``, order, product, category,
            supplier and customer columns, and ``Month`` (monthly
            Period).
    """
    lines = t["order_details"].copy()
    lines["GrossRevenue"] = lines["UnitPrice"] * lines["Quantity"]
    lines["Revenue"] = lines["GrossRevenue"] * (1 - lines["Discount"])
    lines["DiscountAmount"] = lines["GrossRevenue"] - lines["Revenue"]

    orders = t["orders"][
        ["OrderID", "CustomerID", "EmployeeID", "OrderDate", "ShipVia"]
    ]
    products = t["products"][
        ["ProductID", "ProductName", "SupplierID", "CategoryID",
         "Discontinued"]
    ]
    lines = (
        lines.merge(orders, on="OrderID", how="left")
        .merge(products, on="ProductID", how="left")
        .merge(t["categories"][["CategoryID", "CategoryName"]],
               on="CategoryID", how="left")
        .merge(t["suppliers"][["SupplierID", "CompanyName", "Country"]]
               .rename(columns={"CompanyName": "Supplier",
                                "Country": "SupplierCountry"}),
               on="SupplierID", how="left")
        .merge(t["customers"][["CustomerID", "CompanyName", "Country"]]
               .rename(columns={"CompanyName": "Customer",
                                "Country": "CustomerCountry"}),
               on="CustomerID", how="left")
    )
    lines["Month"] = lines["OrderDate"].dt.to_period("M")
    return lines


def build_order_summary(t, lines):
    """Return one row per order with totals and shipping measures.

    Args:
        t (dict[str, pandas.DataFrame]): Tables from ``load_tables``.
        lines (pandas.DataFrame): Output of ``build_order_lines``.

    Returns:
        pandas.DataFrame: Orders sheet plus ``Revenue``, ``Items``
            (units), ``DaysToShip``, ``Shipped``, ``Late`` (shipped
            after the required date) and ``Month`` (monthly Period).
    """
    totals = lines.groupby("OrderID").agg(
        Revenue=("Revenue", "sum"), Items=("Quantity", "sum")
    )
    orders = t["orders"].merge(totals, on="OrderID", how="left")
    orders["DaysToShip"] = (orders["ShippedDate"] - orders["OrderDate"]).dt.days
    orders["Shipped"] = orders["ShippedDate"].notna()
    orders["Late"] = orders["Shipped"] & (
        orders["ShippedDate"] > orders["RequiredDate"]
    )
    orders["Month"] = orders["OrderDate"].dt.to_period("M")
    return orders
