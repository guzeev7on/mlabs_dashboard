from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import date as Date

# ── Field aliases: canonical name → possible API field names ──────────────────
_ALIASES: dict[str, list[str]] = {
    "market_id": ["conditionId", "market", "marketId", "condition_id"],
    "title":     ["title", "question", "name", "marketTitle", "market_slug"],
    "slug":      ["slug", "eventSlug", "market_slug"],
    "outcome":   ["outcome", "outcomeName", "outcomeIndex"],
    "price":     ["price", "tradePrice", "avgPrice"],
    "size":      ["size", "shares", "contractsAmount", "sharesAmount"],
    "usdc_size": ["usdcSize", "amount", "usdcAmount", "cashPayout"],
    "side":      ["side", "tradeType"],
    "timestamp": ["timestamp", "createdAt", "time", "created_at"],
}


def _first_col(df: pd.DataFrame, canonical: str) -> str | None:
    for alias in _ALIASES.get(canonical, []):
        if alias in df.columns:
            return alias
    return None


# ── Parse raw list → normalised DataFrame ────────────────────────────────────

def parse_trades(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    out = pd.DataFrame(index=df.index)

    # Timestamp & date
    ts_col = _first_col(df, "timestamp")
    if ts_col:
        ts = pd.to_numeric(df[ts_col], errors="coerce")
        if ts.median() > 1e12:          # milliseconds → seconds
            ts = ts / 1000
        out["timestamp"] = ts
        out["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.normalize().dt.tz_localize(None)
    else:
        out["timestamp"] = np.nan
        out["date"] = pd.NaT

    # Market identifier
    mid = _first_col(df, "market_id")
    out["market_id"] = df[mid].astype(str) if mid else df.index.astype(str)

    # Event title (fallback to market_id)
    tc = _first_col(df, "title")
    out["title"] = df[tc].fillna(out["market_id"]) if tc else out["market_id"]

    # Slug
    slc = _first_col(df, "slug")
    out["slug"] = df[slc].astype(str) if slc else ""

    # Outcome
    oc = _first_col(df, "outcome")
    out["outcome"] = df[oc].astype(str) if oc else "Unknown"

    # Price (0–1 scale); if looks like cents → divide by 100
    pc = _first_col(df, "price")
    if pc:
        p = pd.to_numeric(df[pc], errors="coerce")
        out["price"] = p.where(p <= 1.0, p / 100.0)
    else:
        out["price"] = np.nan

    # Contract count
    sc = _first_col(df, "size")
    out["contracts"] = pd.to_numeric(df[sc], errors="coerce") if sc else np.nan

    # USDC size (derived if not present)
    uc = _first_col(df, "usdc_size")
    if uc:
        out["usdc_size"] = pd.to_numeric(df[uc], errors="coerce")
    else:
        out["usdc_size"] = out["contracts"] * out["price"]

    # Side
    sdc = _first_col(df, "side")
    out["side"] = df[sdc].str.upper() if sdc else "UNKNOWN"

    return out.dropna(subset=["market_id"])


# ── Date filter ───────────────────────────────────────────────────────────────

def apply_date_filter(df: pd.DataFrame, start: Date | None, end: Date | None) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= df["date"] >= pd.Timestamp(start)
    if end:
        mask &= df["date"] <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[mask].copy()


# ── Per-event analytics ───────────────────────────────────────────────────────
#
# For each event (market) the user may have traded several outcomes.
# Intersection = min contracts across all outcomes  (guaranteed resolved units).
# Profit = intersection × (Σ avg_prices − 1.0)
#   • price_sum > 1 → net avg sell > $1 per pair → profit positive
#   • price_sum < 1 → positions still cheap / net bought
# ROI   = profit / (intersection × price_sum)

def event_analytics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    needed = {"market_id", "outcome", "contracts", "price"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    work = df.dropna(subset=["contracts", "price"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["px_c"] = work["price"] * work["contracts"]

    # Aggregate per market × outcome
    agg = (
        work.groupby(["market_id", "outcome"])
        .agg(
            title=("title", "first"),
            slug=("slug", "first"),
            total_contracts=("contracts", "sum"),
            px_c_sum=("px_c", "sum"),
            usdc_volume=("usdc_size", "sum"),
            trade_count=("contracts", "count"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    agg["avg_price"] = agg["px_c_sum"] / agg["total_contracts"]

    rows = []
    for market_id, grp in agg.groupby("market_id"):
        title = grp["title"].iloc[0]
        slug = grp["slug"].iloc[0]
        outcomes = grp["outcome"].tolist()
        vols = grp["total_contracts"].tolist()
        prices = grp["avg_price"].tolist()
        usdc_vols = grp["usdc_volume"].tolist()
        last_date = grp["last_date"].max()
        total_trades = grp["trade_count"].sum()

        total_vol = sum(vols)
        total_usdc = sum(usdc_vols)

        if len(outcomes) < 2:
            # Single outcome — no intersection
            detail = f"{outcomes[0]}: {vols[0]:.1f} @ {prices[0]:.4f}"
            rows.append({
                "market_id": market_id,
                "title": title,
                "slug": slug,
                "outcomes_detail": detail,
                "total_contracts": total_vol,
                "total_usdc": total_usdc,
                "intersection": 0.0,
                "price_sum": prices[0],
                "profit_usdc": 0.0,
                "roi_pct": 0.0,
                "trade_count": int(total_trades),
                "last_date": last_date,
                "n_outcomes": 1,
            })
            continue

        intersection = min(vols)
        price_sum = sum(prices)
        profit = intersection * (price_sum - 1.0)
        cost = intersection * price_sum
        roi = (profit / cost * 100) if cost > 0 else 0.0

        detail_parts = [f"{o}: {v:.1f} @ {p:.4f}" for o, v, p in zip(outcomes, vols, prices)]
        rows.append({
            "market_id": market_id,
            "title": title,
            "slug": slug,
            "outcomes_detail": " | ".join(detail_parts),
            "total_contracts": total_vol,
            "total_usdc": total_usdc,
            "intersection": intersection,
            "price_sum": price_sum,
            "profit_usdc": profit,
            "roi_pct": roi,
            "trade_count": int(total_trades),
            "last_date": last_date,
            "n_outcomes": len(outcomes),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("profit_usdc", ascending=False).reset_index(drop=True)
    return result


# ── Daily volume stats ────────────────────────────────────────────────────────

def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    d = (
        df.groupby("date")
        .agg(
            usdc_volume=("usdc_size", "sum"),
            contracts_traded=("contracts", "sum"),
            trade_count=("contracts", "count"),
        )
        .reset_index()
        .sort_values("date")
    )
    d["cum_volume"] = d["usdc_volume"].cumsum()
    return d


# ── Daily profit series (attributed to last-trade date per event) ─────────────

def parse_rebates(raw: list) -> pd.DataFrame:
    """Parse MAKER_REBATE records into a daily DataFrame."""
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    out = pd.DataFrame(index=df.index)
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts.median() > 1e12:
        ts = ts / 1000
    out["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.normalize().dt.tz_localize(None)
    out["rebate_usdc"] = pd.to_numeric(df["usdcSize"], errors="coerce").fillna(0)
    return out.dropna(subset=["date"]).sort_values("date")


def rebate_daily(rebates_df: pd.DataFrame, trades_vol_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily rebates with daily trade volume; compute ratio."""
    if rebates_df.empty:
        return pd.DataFrame()
    reb = rebates_df.groupby("date").agg(rebate_usdc=("rebate_usdc", "sum")).reset_index()
    if not trades_vol_df.empty:
        merged = trades_vol_df[["date", "usdc_volume"]].merge(reb, on="date", how="outer").sort_values("date")
        merged["rebate_usdc"] = merged["rebate_usdc"].fillna(0)
        merged["usdc_volume"] = merged["usdc_volume"].fillna(0)
    else:
        merged = reb.rename(columns={})
        merged["usdc_volume"] = 0
    merged["ratio_pct"] = (merged["rebate_usdc"] / merged["usdc_volume"] * 100).where(merged["usdc_volume"] > 0)
    return merged


def daily_profit(event_df: pd.DataFrame, vol_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if event_df.empty or "last_date" not in event_df.columns:
        return pd.DataFrame()
    d = (
        event_df.groupby("last_date")
        .agg(daily_profit=("profit_usdc", "sum"))
        .reset_index()
        .rename(columns={"last_date": "date"})
        .sort_values("date")
    )
    d["cum_profit"] = d["daily_profit"].cumsum()
    if vol_df is not None and not vol_df.empty:
        d = d.merge(vol_df[["date", "usdc_volume"]], on="date", how="left")
        d["roi_pct"] = (d["daily_profit"] / d["usdc_volume"] * 100).where(d["usdc_volume"] > 0)
    return d
