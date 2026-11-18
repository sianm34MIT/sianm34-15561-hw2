"""Independent recomputation of headline figures from the raw workbook."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
x = pd.read_excel(ROOT / "data" / "northwind.xlsx", sheet_name=None)
O, D, P = x["Orders"], x["Orders Details"], x["Product"]
C, E, S, CAT, SH = (x["Customers"], x["Employees"], x["Suppliers"],
                    x["Categories"], x["Shippers"])
J = {n: json.loads((ROOT / "site" / "data" / f"{n}.json").read_text())
     for n in ["overview", "q1_growth", "q2_products", "q3_relationships"]}

fails = []


def check(label, ours, theirs, tol=0.005):
    ok = (ours == theirs) if not isinstance(ours, float) else (
        abs(ours - theirs) <= tol * max(1, abs(ours)))
    print(("OK  " if ok else "FAIL"), label, ours, theirs)
    if not ok:
        fails.append(label)


# ---- raw data sanity
print("dup detail keys", D.duplicated(["OrderID", "ProductID"]).sum(),
      "| neg qty", (D.Quantity <= 0).sum(), "| neg price", (D.UnitPrice <= 0).sum())
print("discount values", sorted(D.Discount.round(4).unique()))
print("null OrderDate", O.OrderDate.isna().sum(), "| null Required",
      O.RequiredDate.isna().sum())
un = O[O.ShippedDate.isna()]
print("unshipped", len(un), un.OrderDate.min().date(), un.OrderDate.max().date())
cc = O.merge(C[["CustomerID", "Country"]], on="CustomerID")
print("ship country != customer country:", (cc.ShipCountry != cc.Country).sum())
print("ShippedDate < OrderDate", (O.ShippedDate < O.OrderDate).sum())

# ---- revenue
D["rev"] = D.UnitPrice * D.Quantity * (1 - D.Discount)
D["gross"] = D.UnitPrice * D.Quantity
ov = J["overview"]["overview"]["kpis"]
check("total revenue", round(D.rev.sum(), 2), ov["revenue"])
check("discounts", round((D.gross - D.rev).sum(), 2), ov["discounts"])
check("orders", O.OrderID.nunique(), ov["orders"])
check("active customers", O.CustomerID.nunique(), ov["customersActive"])
check("first order", str(O.OrderDate.min().date()), ov["firstOrder"])
check("last order", str(O.OrderDate.max().date()), ov["lastOrder"])
may = O[O.OrderDate >= "1998-05-01"]
print("May 1998 orders", len(may), "days spanned", may.OrderDate.dt.day.max())

# list vs charged price
m = D.merge(P[["ProductID", "Price"]], on="ProductID")
print("lines where charged != list", (~np.isclose(m.UnitPrice, m.Price)).sum())

# ---- products
pr = D.groupby("ProductID").rev.sum()
p = P.set_index("ProductID")
pk = J["q2_products"]["kpis"]
share = pr.sort_values(ascending=False) / pr.sum()
check("top10 product share", round(share.head(10).sum(), 4), pk["top10Share"])
check("products for half", int((share.cumsum() < .5).sum() + 1), pk["productsForHalf"])
thr = P.Price.quantile(.9)
prem = P[P.Price >= thr].ProductID
check("premium count", len(prem), pk["premiumCount"])
check("premium share", round(pr[prem].sum() / pr.sum(), 4), pk["premiumShare"])
cdb = P.loc[P.ProductName == "Côte de Blaye", "ProductID"].iloc[0]
check("CdB share", round(pr[cdb] / pr.sum(), 4), pk["starShare"])
disc = P[P.Discontinued].ProductID
check("discontinued count", len(disc), pk["discontinuedCount"])
check("discontinued share", round(pr[disc].sum() / pr.sum(), 4), pk["discontinuedShare"])
check("discontinued sold out", int((P[P.Discontinued].UnitsInStock == 0).sum()), pk["discontinuedSoldOut"])
DO = D.merge(O[["OrderID", "OrderDate", "CustomerID", "EmployeeID"]], on="OrderID")
rec = DO[(DO.OrderDate >= "1998-02-01") & (DO.OrderDate < "1998-05-01")]
rd = rec[rec.ProductID.isin(disc)].rev.sum()
check("discontinued recent share", round(rd / rec.rev.sum(), 4), pk["discontinuedRecentShare"])
check("revenue at risk/yr", round(rd / 3 * 12, 2), pk["annualRevenueAtRisk"])
lastsale = DO[DO.ProductID.isin(disc)].groupby("ProductID").OrderDate.max()
print("discontinued last sales", {p.loc[i, "ProductName"]: str(v.date()) for i, v in lastsale.items()})
active = P[~P.Discontinued]
below = active[active.UnitsInStock <= active.ReorderLevel]
check("below reorder", len(below), J["overview"]["products"]["kpis"]["belowReorder"])
mp = P[P.CategoryID == CAT.loc[CAT.CategoryName == "Meat/Poultry", "CategoryID"].iloc[0]].ProductID
check("meat disc share", round(pr[mp[mp.isin(disc)]].sum() / pr[mp].sum(), 4),
      round([c for c in J["q2_products"]["categories"] if c["CategoryName"] == "Meat/Poultry"][0]["DiscontinuedShare"], 4))

# ---- suppliers
sr = D.merge(P[["ProductID", "SupplierID"]], on="ProductID").groupby("SupplierID").rev.sum().sort_values(ascending=False)
sk = J["overview"]["suppliers"]["kpis"]
check("suppliers countries", S.Country.nunique(), sk["countries"])
check("top5 supplier share", round(sr.head(5).sum() / sr.sum(), 4), sk["top5Share"])

# ---- employees
orev = DO.groupby("OrderID").rev.sum()
O2 = O.assign(rev=O.OrderID.map(orev))
er = O2.groupby("EmployeeID").rev.sum()
emp = {r["EmployeeID"]: r for r in J["overview"]["employees"]["table"]}
for i, v in er.items():
    check(f"employee {i} revenue", round(v, 2), round(emp[i]["Revenue"], 2))
sh = O2[O2.ShippedDate.notna()]
late = sh.ShippedDate > sh.RequiredDate
check("late rate", round(late.mean(), 4), J["overview"]["employees"]["kpis"]["lateRate"])
print("late orders per employee", sh[late].groupby("EmployeeID").size().to_dict())

# ---- Q1 windows
def win(df, a, b):
    return df[(df.OrderDate >= a) & (df.OrderDate <= b)]
A, B = win(O2, "1996-07-01", "1997-04-30"), win(O2, "1997-07-01", "1998-04-30")
q = J["q1_growth"]
check("Q1 before rev", round(A.rev.sum(), 2), round(q["before"]["revenue"], 2))
check("Q1 after rev", round(B.rev.sum(), 2), round(q["after"]["revenue"], 2))
check("Q1 cust before", A.CustomerID.nunique(), q["before"]["customers"])
check("Q1 cust after", B.CustomerID.nunique(), q["after"]["customers"])
first = O2.groupby("CustomerID").OrderDate.min()
new = [c for c in B.CustomerID.unique() if first[c] > pd.Timestamp("1997-04-30")]
seg = {s["Segment"]: s for s in q["segments"]}
check("Q1 new customers", len(new), seg["New"]["Customers"])
chg = (B.groupby("CustomerID").rev.sum().sub(A.groupby("CustomerID").rev.sum(), fill_value=0)).sort_values(ascending=False)
check("Q1 top10 share of growth", round(chg.head(10).sum() / chg.sum(), 4), q["concentration"]["top10Share"])
print("Q1 customers shrinking", (chg < 0).sum(), "growing", (chg > 0).sum())
cA = A.groupby("ShipCountry").rev.sum(); cB = B.groupby("ShipCountry").rev.sum()
cg = cB.sub(cA, fill_value=0).sort_values(ascending=False)
print("Q1 US+Germany share of growth", round(cg[["USA", "Germany"]].sum() / cg.sum(), 3))

# ---- Q3
o3 = O2.sort_values(["CustomerID", "OrderDate", "OrderID"])
prev = o3.groupby("CustomerID").EmployeeID.shift()
same = (o3.EmployeeID == prev)[prev.notna()]
check("Q3 same seller rate", round(same.mean(), 4), J["q3_relationships"]["natural"]["sameSellerRate"])
top20 = O2.groupby("CustomerID").rev.sum().sort_values(ascending=False).head(20).index
check("Q3 sellers per top account", round(O2[O2.CustomerID.isin(top20)].groupby("CustomerID").EmployeeID.nunique().mean(), 1),
      J["q3_relationships"]["sellersPerTopAccount"])
rec_top = rec[rec.CustomerID.isin(top20) & rec.ProductID.isin(disc)].rev.sum() / 3 * 12
check("Q3 top20 discontinued run-rate", round(rec_top, 2), J["q3_relationships"]["topDiscontinuedRunRate"])

print("\nFAILURES:", fails or "none")
