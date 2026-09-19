# ============================================================
# 3.3 — Distance matrix preview
# Raw per-period pair distances (rows = period start year, columns = genre_a|genre_b).
# Paste this as its own notebook cell, after `pair_result = GenrePairAnalyzer(...).analyze()`.
# ============================================================
dist = pair_result.pair_distances

display(Markdown(
    f"Shape: **{dist.shape[0]} periods × {dist.shape[1]} pairs** — "
    "non-null cells per period:"
))
display(dist.notna().sum(axis=1).to_frame("non_null_pairs"))

display(Markdown("First 6 pair columns:"))
display(dist.iloc[:, :6].round(2))


# ============================================================
# 3.4 — Top movers: concrete distance changes
# Per genre pair: first/last observed centroid distance, absolute and relative
# change, and a direction label. Sorted by |relative change| so the biggest
# movers come first. Reflects whichever subset pair_result was built from
# (small / medium / large).
# ============================================================
import pandas as pd

rows = []
for col in pair_result.pair_distances.columns:
    vals = pair_result.pair_distances[col].dropna()
    if len(vals) < 2 or vals.iloc[0] == 0:
        continue
    first, last = vals.iloc[0], vals.iloc[-1]
    abs_change = last - first
    rel_pct = abs_change / first * 100
    rows.append({
        "pair": col,
        "first_period": int(vals.index[0]),
        "last_period": int(vals.index[-1]),
        "n_periods": len(vals),
        "first_dist": round(first, 2),
        "last_dist": round(last, 2),
        "abs_change": round(abs_change, 2),
        "rel_change_pct": round(rel_pct, 1),
        "direction": (
            "converging" if last < first
            else ("diverging" if last > first else "stable")
        ),
    })

movers = (
    pd.DataFrame(rows)
    .sort_values("rel_change_pct", key=lambda s: s.abs(), ascending=False)
    .reset_index(drop=True)
)

display(Markdown(
    f"**{len(movers)} pairs** with \u22652 observed periods. "
    "Top movers by |relative change|:"
))
display(movers.head(15))
