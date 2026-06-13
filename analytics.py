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
    out["side"] = df[sdc].astype(str).str.upper() if sdc else "UNKNOWN"

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

def _fifo_match_outcome(trades: pd.DataFrame) -> dict:
    long_lots: list[dict[str, float]] = []
    short_lots: list[dict[str, float]] = []
    realized_pnl = 0.0
    matched_qty = 0.0
    matched_volume = 0.0

    for row in trades.sort_values(["timestamp", "date"]).itertuples():
        side = str(row.side).upper()
        qty = float(row.contracts)
        price = float(row.price)
        if qty <= 0 or np.isnan(qty) or np.isnan(price):
            continue

        if side == "BUY":
            remaining = qty
            while remaining > 1e-12 and short_lots:
                lot = short_lots[0]
                close_qty = min(remaining, lot["qty"])
                realized_pnl += close_qty * (lot["price"] - price)
                matched_qty += close_qty
                matched_volume += close_qty * (lot["price"] + price)
                lot["qty"] -= close_qty
                remaining -= close_qty
                if lot["qty"] <= 1e-12:
                    short_lots.pop(0)
            if remaining > 1e-12:
                long_lots.append({"qty": remaining, "price": price})

        elif side == "SELL":
            remaining = qty
            while remaining > 1e-12 and long_lots:
                lot = long_lots[0]
                close_qty = min(remaining, lot["qty"])
                realized_pnl += close_qty * (price - lot["price"])
                matched_qty += close_qty
                matched_volume += close_qty * (lot["price"] + price)
                lot["qty"] -= close_qty
                remaining -= close_qty
                if lot["qty"] <= 1e-12:
                    long_lots.pop(0)
            if remaining > 1e-12:
                short_lots.append({"qty": remaining, "price": price})

    return {
        "realized_pnl": realized_pnl,
        "matched_qty": matched_qty,
        "matched_volume": matched_volume,
        "long_lots": long_lots,
        "short_lots": short_lots,
    }


def _consume_lots(lots: list[dict[str, float]], qty: float) -> tuple[float, float]:
    remaining = qty
    consumed_qty = 0.0
    consumed_value = 0.0

    for lot in lots:
        if remaining <= 1e-12:
            break
        take_qty = min(remaining, lot["qty"])
        consumed_qty += take_qty
        consumed_value += take_qty * lot["price"]
        remaining -= take_qty

    return consumed_qty, consumed_value


def _lot_qty(lots: list[dict[str, float]]) -> float:
    return sum(lot["qty"] for lot in lots)


def _lot_value(lots: list[dict[str, float]]) -> float:
    return sum(lot["qty"] * lot["price"] for lot in lots)

def event_analytics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    needed = {"market_id", "outcome", "contracts", "price", "side"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    work = df.dropna(subset=["contracts", "price"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["side"] = work["side"].astype(str).str.upper()
    work = work[work["side"].isin(["BUY", "SELL"])].copy()
    if work.empty:
        return pd.DataFrame()

    work["trade_value"] = work["contracts"] * work["price"]

    rows = []
    for market_id, grp in work.groupby("market_id", sort=False):
        title = grp["title"].iloc[0]
        slug = grp["slug"].iloc[0]
        last_date = grp["date"].max()
        total_contracts = grp["contracts"].sum()
        total_usdc = grp["trade_value"].sum()
        total_trades = len(grp)

        outcome_rows: list[dict] = []
        realized_leg_pnl = 0.0
        leg_matched_qty = 0.0
        leg_matched_volume = 0.0

        for outcome, outcome_grp in grp.groupby("outcome", sort=False):
            matched = _fifo_match_outcome(outcome_grp)
            realized_leg_pnl += matched["realized_pnl"]
            leg_matched_qty += matched["matched_qty"]
            leg_matched_volume += matched["matched_volume"]

            buys = outcome_grp[outcome_grp["side"] == "BUY"]
            sells = outcome_grp[outcome_grp["side"] == "SELL"]
            buy_qty = buys["contracts"].sum()
            sell_qty = sells["contracts"].sum()
            buy_value = buys["trade_value"].sum()
            sell_value = sells["trade_value"].sum()
            long_qty = _lot_qty(matched["long_lots"])
            short_qty = _lot_qty(matched["short_lots"])
            long_value = _lot_value(matched["long_lots"])
            short_value = _lot_value(matched["short_lots"])

            outcome_rows.append({
                "outcome": outcome,
                "buy_qty": buy_qty,
                "buy_value": buy_value,
                "sell_qty": sell_qty,
                "sell_value": sell_value,
                "long_lots": matched["long_lots"],
                "short_lots": matched["short_lots"],
                "long_qty": long_qty,
                "long_value": long_value,
                "short_qty": short_qty,
                "short_value": short_value,
            })

        n_outcomes = len(outcome_rows)
        complete_buy_qty = 0.0
        complete_buy_cost = 0.0
        complete_buy_pnl = 0.0
        complete_buy_price_sum = np.nan
        complete_sell_qty = 0.0
        complete_sell_proceeds = 0.0
        complete_sell_pnl = 0.0
        complete_sell_price_sum = np.nan

        if n_outcomes >= 2:
            complete_buy_qty = min(row["long_qty"] for row in outcome_rows)
            if complete_buy_qty > 1e-12:
                for row in outcome_rows:
                    _, value = _consume_lots(row["long_lots"], complete_buy_qty)
                    complete_buy_cost += value
                complete_buy_pnl = complete_buy_qty - complete_buy_cost
                complete_buy_price_sum = complete_buy_cost / complete_buy_qty

            complete_sell_qty = min(row["short_qty"] for row in outcome_rows)
            if complete_sell_qty > 1e-12:
                for row in outcome_rows:
                    _, value = _consume_lots(row["short_lots"], complete_sell_qty)
                    complete_sell_proceeds += value
                complete_sell_pnl = complete_sell_proceeds - complete_sell_qty
                complete_sell_price_sum = complete_sell_proceeds / complete_sell_qty

        complete_pnl = complete_buy_pnl + complete_sell_pnl
        profit = realized_leg_pnl + complete_pnl
        pnl_capital = complete_buy_cost + complete_sell_qty
        roi = (profit / total_usdc * 100) if total_usdc > 0 else 0.0

        detail_parts = []
        unmatched_long_qty = 0.0
        unmatched_short_qty = 0.0
        for row in outcome_rows:
            buy_avg = row["buy_value"] / row["buy_qty"] if row["buy_qty"] > 0 else np.nan
            sell_avg = row["sell_value"] / row["sell_qty"] if row["sell_qty"] > 0 else np.nan
            long_after_sets = max(row["long_qty"] - complete_buy_qty, 0.0)
            short_after_sets = max(row["short_qty"] - complete_sell_qty, 0.0)
            unmatched_long_qty += long_after_sets
            unmatched_short_qty += short_after_sets
            buy_txt = f"buy {row['buy_qty']:.1f} @ {buy_avg:.4f}" if row["buy_qty"] > 0 else "buy 0"
            sell_txt = f"sell {row['sell_qty']:.1f} @ {sell_avg:.4f}" if row["sell_qty"] > 0 else "sell 0"
            net_txt = f"net long {long_after_sets:.1f}" if long_after_sets > 1e-9 else ""
            if short_after_sets > 1e-9:
                net_txt = f"net short {short_after_sets:.1f}"
            if not net_txt:
                net_txt = "net flat"
            detail_parts.append(f"{row['outcome']}: {buy_txt}, {sell_txt}, {net_txt}")

        rows.append({
            "market_id": market_id,
            "title": title,
            "slug": slug,
            "outcomes_detail": " | ".join(detail_parts),
            "total_contracts": total_contracts,
            "total_usdc": total_usdc,
            "leg_matched_qty": leg_matched_qty,
            "leg_matched_volume": leg_matched_volume,
            "leg_pnl_usdc": realized_leg_pnl,
            "complete_buy_qty": complete_buy_qty,
            "complete_buy_price_sum": complete_buy_price_sum,
            "complete_buy_pnl_usdc": complete_buy_pnl,
            "complete_sell_qty": complete_sell_qty,
            "complete_sell_price_sum": complete_sell_price_sum,
            "complete_sell_pnl_usdc": complete_sell_pnl,
            "complete_pnl_usdc": complete_pnl,
            "pnl_capital": pnl_capital,
            "unmatched_long_qty": unmatched_long_qty,
            "unmatched_short_qty": unmatched_short_qty,
            "intersection": complete_buy_qty + complete_sell_qty,
            "price_sum": np.nan,
            "profit_usdc": profit,
            "roi_pct": roi,
            "trade_count": int(total_trades),
            "last_date": last_date,
            "n_outcomes": n_outcomes,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("profit_usdc", ascending=False).reset_index(drop=True)
    return result


# ── Daily volume stats ────────────────────────────────────────────────────────

def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    work = df.dropna(subset=["contracts", "price"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["trade_value"] = work["contracts"] * work["price"]
    d = (
        work.groupby("date")
        .agg(
            usdc_volume=("trade_value", "sum"),
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


def parse_closed_positions(raw: list) -> pd.DataFrame:
    """Normalize Data API closed-position rows."""
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    out = pd.DataFrame(index=df.index)
    ts = pd.to_numeric(df.get("timestamp"), errors="coerce")
    if ts.median() > 1e12:
        ts = ts / 1000

    out["timestamp"] = ts
    out["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.normalize().dt.tz_localize(None)
    out["market_id"] = df.get("conditionId", "").astype(str)
    out["title"] = df.get("title", "").astype(str)
    out["slug"] = df.get("slug", "").astype(str)
    out["event_slug"] = df.get("eventSlug", "").astype(str)
    out["outcome"] = df.get("outcome", "").astype(str)
    out["avg_price"] = pd.to_numeric(df.get("avgPrice"), errors="coerce")
    out["total_bought"] = pd.to_numeric(df.get("totalBought"), errors="coerce").fillna(0)
    out["realized_pnl"] = pd.to_numeric(df.get("realizedPnl"), errors="coerce").fillna(0)
    out["resolved_price"] = pd.to_numeric(df.get("curPrice"), errors="coerce")
    return out.dropna(subset=["date", "market_id", "resolved_price"])


def expiration_analytics(closed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate outcome-level closed positions into settled markets.

    A closed position can also mean that the user sold out before resolution.
    Exact 0/1 current prices distinguish markets that have actually settled.
    """
    if closed_df.empty:
        return pd.DataFrame()

    rows = []
    for market_id, grp in closed_df.groupby("market_id"):
        prices = grp["resolved_price"].dropna()
        if prices.empty or not prices.apply(lambda p: np.isclose(p, 0.0) or np.isclose(p, 1.0)).all():
            continue

        winners = grp.loc[np.isclose(grp["resolved_price"], 1.0), "outcome"].tolist()
        details = [
            f"{row.outcome}: settle {row.resolved_price:.0f}, PnL {row.realized_pnl:+.4f}"
            for row in grp.itertuples()
        ]
        rows.append({
            "market_id": market_id,
            "date": grp["date"].max(),
            "title": grp["title"].iloc[0],
            "slug": grp["slug"].iloc[0],
            "event_slug": grp["event_slug"].iloc[0],
            "winner": ", ".join(winners) if winners else "Нет выигрышного исхода",
            "expiration_pnl": grp["realized_pnl"].sum(),
            "total_bought": grp["total_bought"].sum(),
            "outcomes_detail": " | ".join(details),
            "n_outcomes": len(grp),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("date").reset_index(drop=True)
    return result


def expiration_daily(expiration_df: pd.DataFrame) -> pd.DataFrame:
    if expiration_df.empty:
        return pd.DataFrame()
    daily = (
        expiration_df.groupby("date")
        .agg(
            daily_pnl=("expiration_pnl", "sum"),
            events=("market_id", "count"),
            profitable=("expiration_pnl", lambda s: int((s > 0).sum())),
            losing=("expiration_pnl", lambda s: int((s < 0).sum())),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()
    return daily


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
