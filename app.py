from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    apply_date_filter,
    daily_profit,
    daily_stats,
    event_analytics,
    parse_trades,
    parse_rebates,
    rebate_daily,
)
from fetcher import DEFAULT_ADDRESS, get_all_trades, get_rebates

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Polymarket Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    [data-testid="metric-container"] {
        background: #1a1b2e;
        border: 1px solid #2d2e4a;
        border-radius: 10px;
        padding: 14px 18px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #6366f1; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Боковая панель ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Настройки")
    address = st.text_input("Адрес кошелька", value=DEFAULT_ADDRESS)

    st.divider()
    st.subheader("Период")
    all_trades = st.checkbox("Все трейды (без фильтра)", value=True)

    start_date: date | None = None
    end_date: date | None = None
    if not all_trades:
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("С", value=date.today() - timedelta(days=90))
        with c2:
            end_date = st.date_input("По", value=date.today())

    st.divider()
    refresh = st.button("🔄 Загрузить / Обновить", type="primary", use_container_width=True)
    st.caption("Данные кэшируются на 5 мин.")

# ── Загрузка данных ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _cached_fetch(addr: str) -> list:
    return get_all_trades(addr)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_rebates(addr: str) -> list:
    return get_rebates(addr)


if "raw" not in st.session_state or refresh:
    with st.spinner("Загружаем трейды и rebates…"):
        try:
            st.session_state["raw"]     = _cached_fetch(address)
            st.session_state["raw_reb"] = _cached_rebates(address)
            st.session_state["fetch_addr"] = address
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")
            st.stop()

raw: list = st.session_state.get("raw", [])
if not raw:
    st.warning("Трейды не найдены. Проверьте адрес кошелька и нажмите «Обновить».")
    st.stop()

df_full     = parse_trades(raw)
df          = apply_date_filter(df_full, start_date, end_date)
reb_full    = parse_rebates(st.session_state.get("raw_reb", []))
reb_df      = apply_date_filter(reb_full, start_date, end_date)

# ── Вкладки ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Аналитика событий",
    "📈 Market rebates",
    "📉 Вкладка 3",
    "🗂️ Вкладка 4",
    "⚙️ Вкладка 5",
])

# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 1 — Аналитика событий
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Аналитика событий")

    if df.empty:
        st.warning("Нет трейдов в выбранном периоде.")
        st.stop()

    ev_df = event_analytics(df)
    day_df = daily_stats(df)
    profit_df = daily_profit(ev_df, day_df)

    # ── KPI ───────────────────────────────────────────────────────────────────
    n_events     = len(ev_df)
    total_vol    = ev_df["total_usdc"].sum()      if not ev_df.empty else 0.0
    total_contr  = ev_df["total_contracts"].sum()  if not ev_df.empty else 0.0
    total_profit = ev_df["profit_usdc"].sum()      if not ev_df.empty else 0.0
    total_cost   = (ev_df["intersection"] * ev_df["price_sum"]).sum() if not ev_df.empty else 0.0
    overall_roi  = (total_profit / total_cost * 100) if total_cost > 0 else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Трейдов",           f"{len(df):,}")
    k2.metric("Уникальных событий", f"{n_events:,}")
    k3.metric("Объём (USDC)",      f"${total_vol:,.2f}")
    k4.metric("Объём (контракты)", f"{total_contr:,.1f}")
    k5.metric("Прибыль (intersection)", f"${total_profit:,.2f}",
              delta=f"{total_profit:+.2f}")
    k6.metric("Общий ROI",         f"{overall_roi:.2f}%")

    st.divider()

    # ── Базовые параметры графиков ─────────────────────────────────────────────
    _PLOT_BASE = dict(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#d1d5db",
    )
    _MARGIN_DEFAULT = dict(l=0, r=0, t=36, b=0)

    # ── Графики — ряд 1: объём и количество трейдов ────────────────────────────
    st.subheader("Дневная статистика")

    if not day_df.empty:
        c_left, c_right = st.columns(2)

        with c_left:
            fig = go.Figure()
            fig.add_bar(
                x=day_df["date"], y=day_df["usdc_volume"],
                name="Объём USDC", marker_color="#6366f1",
            )
            fig.update_layout(
                title="Дневной объём (USDC)",
                xaxis_title="Дата", yaxis_title="USDC",
                height=340, margin=_MARGIN_DEFAULT, **_PLOT_BASE,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            fig2 = go.Figure()
            fig2.add_bar(
                x=day_df["date"], y=day_df["trade_count"],
                name="Трейды", marker_color="#10b981",
            )
            fig2.update_layout(
                title="Количество трейдов в день",
                xaxis_title="Дата", yaxis_title="Трейды",
                height=340, margin=_MARGIN_DEFAULT, **_PLOT_BASE,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Графики — ряд 2: прибыль и ROI по дням ────────────────────────────────
    if not profit_df.empty:
        c_profit, c_roi = st.columns(2)

        with c_profit:
            fig3 = go.Figure()
            fig3.add_bar(
                x=profit_df["date"],
                y=profit_df["daily_profit"],
                name="Дневная прибыль (USDC)",
                marker_color=profit_df["daily_profit"].apply(
                    lambda v: "#10b981" if v >= 0 else "#ef4444"
                ),
                yaxis="y",
            )
            fig3.add_scatter(
                x=profit_df["date"],
                y=profit_df["cum_profit"],
                name="Накопленная прибыль",
                line=dict(color="#f59e0b", width=2),
                yaxis="y2",
                mode="lines+markers",
                marker=dict(size=4),
            )
            fig3.update_layout(
                title="Прибыль по дням (intersection)",
                xaxis_title="Дата",
                yaxis=dict(title="Дневная прибыль (USDC)"),
                yaxis2=dict(title="Накоп. прибыль (USDC)", overlaying="y", side="right"),
                legend=dict(orientation="h", y=-0.28),
                height=380, margin=_MARGIN_DEFAULT, **_PLOT_BASE,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c_roi:
            if "roi_pct" in profit_df.columns and profit_df["roi_pct"].notna().any():
                fig_roi = go.Figure()
                roi_data = profit_df.dropna(subset=["roi_pct"])
                fig_roi.add_bar(
                    x=roi_data["date"],
                    y=roi_data["roi_pct"],
                    name="ROI %",
                    marker_color=roi_data["roi_pct"].apply(
                        lambda v: "#10b981" if v >= 0 else "#ef4444"
                    ),
                )
                fig_roi.update_layout(
                    title="ROI по дням (%)",
                    xaxis_title="Дата",
                    yaxis_title="ROI %",
                    height=380, margin=_MARGIN_DEFAULT, **_PLOT_BASE,
                )
                st.plotly_chart(fig_roi, use_container_width=True)
            else:
                st.info("Недостаточно данных для ROI по дням.")

    st.divider()

    # ── Таблица событий ────────────────────────────────────────────────────────
    st.subheader("Разбивка по событиям")

    if ev_df.empty:
        st.info("Не удалось рассчитать аналитику — см. раздел «Отладка» ниже.")
    else:
        multi_side  = ev_df[ev_df["n_outcomes"] >= 2].copy()
        single_side = ev_df[ev_df["n_outcomes"] <  2].copy()

        if not multi_side.empty:
            st.markdown("**События с торгами по обоим исходам** (intersection значим)")
            disp = multi_side[[
                "title", "slug", "outcomes_detail", "trade_count",
                "total_contracts", "total_usdc",
                "intersection", "price_sum",
                "profit_usdc", "roi_pct",
            ]].copy()
            disp.columns = [
                "Событие", "Slug", "Исходы (контракты @ ср. цена)", "Трейды",
                "Контракты всего", "Объём (USDC)",
                "Intersection", "Price sum",
                "Прибыль (USDC)", "ROI %",
            ]
            disp["Прибыль (USDC)"] = disp["Прибыль (USDC)"].map(lambda x: f"${x:,.4f}")
            disp["ROI %"]          = disp["ROI %"].map(lambda x: f"{x:.2f}%")
            disp["Price sum"]      = disp["Price sum"].map(lambda x: f"{x:.4f}")
            disp["Контракты всего"]= disp["Контракты всего"].map(lambda x: f"{x:,.2f}")
            disp["Intersection"]   = disp["Intersection"].map(lambda x: f"{x:,.2f}")
            disp["Объём (USDC)"]   = disp["Объём (USDC)"].map(lambda x: f"${x:,.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)


    # ── Отладка ────────────────────────────────────────────────────────────────
    with st.expander("🔍 Отладка / Данные API"):
        st.write(f"**Всего трейдов загружено:** {len(raw)}")
        st.write(f"**После фильтра по дате:** {len(df)}")
        st.write(f"**Колонки DataFrame:** {list(df.columns)}")
        if raw:
            st.write("**Пример записи (первый трейд):**")
            st.json(raw[0])

# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 2 — Maker rebates
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Maker Rebates")

    rb = rebate_daily(reb_df, day_df)

    if rb.empty:
        st.info("Нет данных о rebates в выбранном периоде.")
    else:
        total_reb   = rb["rebate_usdc"].sum()
        total_vol2  = rb["usdc_volume"].sum()
        avg_ratio   = (total_reb / total_vol2 * 100) if total_vol2 > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Всего rebates (USDC)", f"${total_reb:,.4f}")
        m2.metric("Объём трейдов (USDC)", f"${total_vol2:,.2f}")
        m3.metric("Средняя доля rebate", f"{avg_ratio:.4f}%")

        st.divider()

        _PLOT2 = dict(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#d1d5db",
            margin=dict(l=0, r=0, t=36, b=0),
        )

        # График 1 — Объём торгов по дням
        fig_v = go.Figure()
        fig_v.add_bar(
            x=rb["date"], y=rb["usdc_volume"],
            name="Объём (USDC)", marker_color="#6366f1",
        )
        fig_v.update_layout(
            title="Объём торгов по дням (USDC)",
            xaxis_title="Дата", yaxis_title="USDC",
            height=340, **_PLOT2,
        )
        st.plotly_chart(fig_v, use_container_width=True)

        # График 2 — Rebates по дням
        fig_r = go.Figure()
        fig_r.add_bar(
            x=rb["date"], y=rb["rebate_usdc"],
            name="Rebate (USDC)", marker_color="#10b981",
        )
        fig_r.update_layout(
            title="Maker Rebates по дням (USDC)",
            xaxis_title="Дата", yaxis_title="USDC",
            height=340, **_PLOT2,
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # График 3 — Доля rebate от объёма
        fig_ratio = go.Figure()
        ratio_data = rb.dropna(subset=["ratio_pct"])
        fig_ratio.add_bar(
            x=ratio_data["date"], y=ratio_data["ratio_pct"],
            name="Rebate / Объём %", marker_color="#f59e0b",
        )
        fig_ratio.update_layout(
            title="Доля Rebate от объёма торгов (%)",
            xaxis_title="Дата", yaxis_title="Rebate / Объём, %",
            height=340, **_PLOT2,
        )
        st.plotly_chart(fig_ratio, use_container_width=True)

# ── Заглушки для остальных вкладок ────────────────────────────────────────────
for _tab, _label in [(tab3, "Вкладка 3"), (tab4, "Вкладка 4"), (tab5, "Вкладка 5")]:
    with _tab:
        st.header(_label)
        st.info("🚧 В разработке")
