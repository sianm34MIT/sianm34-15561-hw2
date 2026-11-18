import json

import numpy as np
import pandas as pd
from scipy import stats

from build_overview import r, records
from northwind import (
    LAST_FULL_MONTH, ROOT, build_order_lines, build_order_summary,
    load_tables,
)

OUT_JSON = ROOT / "site" / "data" / "q3_relationships.json"

SEED = 561
N_SHUFFLES = 1000
N_BOOTSTRAP = 2000
MIN_ORDERS = 5          # customers needed for a meaningful loyalty measure
TOP_N = 20              # accounts proposed for the pilot
RECENT = (pd.Period("1998-02", freq="M"), LAST_FULL_MONTH)
SCENARIO_LIFTS = [0.05, 0.10]


def top_seller_share(sellers):
    counts = sellers.value_counts()
    return counts.iloc[0] / counts.sum()


def loyalty_test(orders, rng):
    eligible = orders.groupby("CustomerID")["OrderID"].transform("count")
    sub = orders[eligible >= MIN_ORDERS]
    observed = sub.groupby("CustomerID")["EmployeeID"].apply(
        top_seller_share).mean()
    shuffled_means = []
    for _ in range(N_SHUFFLES):
        shuffled = sub.assign(EmployeeID=rng.permutation(
            sub["EmployeeID"].values))
        shuffled_means.append(
            shuffled.groupby("CustomerID")["EmployeeID"]
            .apply(top_seller_share).mean())
    shuffled_means = np.array(shuffled_means)
    hist, edges = np.histogram(shuffled_means, bins=24)
    return {
        "customers": int(sub["CustomerID"].nunique()),
        "observed": r(observed, 4),
        "randomMean": r(shuffled_means.mean(), 4),
        "pValue": r((shuffled_means >= observed).mean(), 3),
        "hist": [{"from": r(edges[i], 4), "to": r(edges[i + 1], 4),
                  "count": int(hist[i])} for i in range(len(hist))],
    }


def repeat_orders(orders):
    o = orders.sort_values(["CustomerID", "OrderDate", "OrderID"]).copy()
    o["PrevSeller"] = o.groupby("CustomerID")["EmployeeID"].shift()
    o["NextDate"] = o.groupby("CustomerID")["OrderDate"].shift(-1)
    o["DaysToNext"] = (o["NextDate"] - o["OrderDate"]).dt.days
    rep = o[o["PrevSeller"].notna()].copy()
    rep["SameSeller"] = (rep["EmployeeID"] == rep["PrevSeller"]).astype(float)
    rep["LogRevenue"] = np.log(rep["Revenue"])
    rep["LogItems"] = np.log(rep["Items"])
    return rep


def within_customer_effect(df, outcome):
    y = df[outcome] - df.groupby("CustomerID")[outcome].transform("mean")
    x = df["SameSeller"] - df.groupby("CustomerID")["SameSeller"].transform(
        "mean")
    return (x * y).sum() / (x * x).sum()


def bootstrap_effect(df, outcome, rng):
    groups = [g for _, g in df.groupby("CustomerID")]
    draws = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, len(groups), len(groups))
        sample = pd.concat(
            [groups[i].assign(CustomerID=j) for j, i in enumerate(pick)])
        draws.append(within_customer_effect(sample, outcome))
    return np.percentile(draws, [2.5, 97.5])


def natural_experiment(rep, rng):
    seller_mix = rep["EmployeeID"].value_counts(normalize=True)
    results = {
        "repeatOrders": int(len(rep)),
        "sameSellerOrders": int(rep["SameSeller"].sum()),
        "sameSellerRate": r(rep["SameSeller"].mean(), 4),
        "randomRate": r((seller_mix ** 2).sum(), 4),
        "effects": [],
    }
    for outcome, label, is_log in [
        ("LogRevenue", "Order value", True),
        ("LogItems", "Units per order", True),
        ("DaysToNext", "Days until next order", False),
    ]:
        df = rep.dropna(subset=[outcome])
        est = within_customer_effect(df, outcome)
        lo, hi = bootstrap_effect(df, outcome, rng)
        conv = (lambda v: np.exp(v) - 1) if is_log else (lambda v: v)
        results["effects"].append({
            "outcome": label, "unit": "%" if is_log else "days",
            "estimate": r(conv(est), 4), "low": r(conv(lo), 4),
            "high": r(conv(hi), 4), "n": int(len(df)),
        })
    return results


def top_accounts(t, lines, orders):
    recent_lines = lines[lines["Month"].between(*RECENT)]
    n_months = len(pd.period_range(*RECENT, freq="M"))
    rev = orders.groupby("CustomerID")["Revenue"].sum().sort_values(
        ascending=False)
    top_ids = rev.head(TOP_N).index
    names = t["employees"].set_index("EmployeeID")["Name"]
    rows = []
    for cid in top_ids:
        o = orders[orders["CustomerID"] == cid]
        counts = o["EmployeeID"].value_counts()
        cust_lines = lines[lines["CustomerID"] == cid]
        recent = recent_lines[recent_lines["CustomerID"] == cid]
        rows.append({
            "CustomerID": cid,
            "Customer": cust_lines["Customer"].iloc[0],
            "Country": cust_lines["CustomerCountry"].iloc[0],
            "Revenue": rev[cid],
            "Orders": int(len(o)),
            "Sellers": int(counts.size),
            "TopSeller": names[counts.index[0]],
            "TopSellerShare": counts.iloc[0] / counts.sum(),
            "AnnualRunRate": recent["Revenue"].sum() / n_months * 12,
            "DiscontinuedRunRate": recent.loc[recent["Discontinued"],
                                              "Revenue"].sum()
            / n_months * 12,
            "DiscontinuedShare": cust_lines.loc[cust_lines["Discontinued"],
                                                "Revenue"].sum()
            / cust_lines["Revenue"].sum(),
        })
    acc = pd.DataFrame(rows)
    return acc, rev.head(TOP_N).sum() / rev.sum()


# Matched historical windows used to measure how much account order counts
# swing on their own, without any intervention.
POWER_WINDOWS = {
    6: (("1997-05-01", "1997-11-01"), ("1997-11-01", "1998-05-01")),
    10: (("1996-07-01", "1997-05-01"), ("1997-07-01", "1998-05-01")),
}
POWER_SIZES = [20, 40, 89]


def detectable_lift(orders):
    """Smallest lift in orders per account a two-arm pilot could detect.

    Accounts are randomized 50/50; the outcome is the change in each
    account's order count between two equal windows (log scale), so the
    noise is measured from how much accounts varied on their own.
    """
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    ranked = orders.groupby("CustomerID")["Revenue"].sum().sort_values(
        ascending=False).index

    def count(window):
        start, end = (pd.Timestamp(d) for d in window)
        sel = orders[(orders["OrderDate"] >= start)
                     & (orders["OrderDate"] < end)]
        return sel.groupby("CustomerID")["OrderID"].count()

    out = []
    for months, (w_before, w_after) in POWER_WINDOWS.items():
        before, after = count(w_before), count(w_after)
        for n in POWER_SIZES:
            ids = ranked[:n]
            change = (np.log1p(after.reindex(ids).fillna(0))
                      - np.log1p(before.reindex(ids).fillna(0)))
            mde = z * change.std() * np.sqrt(4 / len(ids))
            out.append({"months": months, "accounts": int(len(ids)),
                        "detectableLift": r(np.exp(mde) - 1, 4)})
    return out


def main():
    rng = np.random.default_rng(SEED)
    t = load_tables()
    lines = build_order_lines(t)
    orders = build_order_summary(t, lines)

    loyalty = loyalty_test(orders, rng)
    rep = repeat_orders(orders)
    natural = natural_experiment(rep, rng)
    acc, top_share = top_accounts(t, lines, orders)
    power = detectable_lift(orders)

    run_rate = acc["AnnualRunRate"].sum()
    scenarios = [{"lift": lift, "annualGain": r(run_rate * lift)}
                 for lift in SCENARIO_LIFTS]
    point = natural["effects"][0]
    scenarios.append({"lift": point["estimate"],
                      "annualGain": r(run_rate * point["estimate"]),
                      "isEstimate": True})

    data = {
        "loyalty": loyalty,
        "natural": natural,
        "accounts": records(acc),
        "topN": TOP_N,
        "topShare": r(top_share, 4),
        "topRunRate": r(run_rate),
        "topDiscontinuedRunRate": r(acc["DiscontinuedRunRate"].sum()),
        "topWithDiscontinued": int((acc["DiscontinuedShare"] > 0).sum()),
        "sellersPerTopAccount": r(acc["Sellers"].mean(), 1),
        "scenarios": scenarios,
        "power": power,
        "sellers": int(t["employees"].shape[0]),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1, default=str))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
