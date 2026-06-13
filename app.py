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
    expiration_analytics,
    expiration_daily,
    parse_closed_positions,
    parse_trades,
    parse_rebates,
    rebate_daily,
)
from fetcher import DEFAULT_ADDRESS, get_all_trades, get_closed_positions, get_rebates

logging.basicConfig(level=logging.INFO)

ENABLE_EXPIRATION_PNL = False

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

def _incremental_closed_positions(addr: str) -> list:
    return get_closed_positions(addr)


if refresh:
    _cached_fetch.clear()
    _cached_rebates.clear()


if (
    "raw" not in st.session_state
    or (ENABLE_EXPIRATION_PNL and "raw_closed" not in st.session_state)
    or st.session_state.get("fetch_addr") != address
    or refresh
):
    spinner_text = (
        "Загружаем трейды, rebates и закрытые позиции…"
        if ENABLE_EXPIRATION_PNL
        else "Загружаем трейды и rebates…"
    )
    with st.spinner(spinner_text):
        try:
            st.session_state["raw"]        = _cached_fetch(address)
            st.session_state["raw_reb"]    = _cached_rebates(address)
            if ENABLE_EXPIRATION_PNL:
                st.session_state["raw_closed"] = _incremental_closed_positions(address)
            else:
                st.session_state["raw_closed"] = []
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
closed_full = (
    parse_closed_positions(st.session_state.get("raw_closed", []))
    if ENABLE_EXPIRATION_PNL
    else pd.DataFrame()
)

# ── Вкладки ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Аналитика событий",
    "📈 Market rebates",
    "📉 В разработке",
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
    overall_roi  = (total_profit / total_vol * 100) if total_vol > 0 else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Трейдов",           f"{len(df):,}")
    k2.metric("Уникальных событий", f"{n_events:,}")
    k3.metric("Объём (USDC)",      f"${total_vol:,.2f}")
    k4.metric("Объём (контракты)", f"{total_contr:,.1f}")
    k5.metric("PnL стратегии", f"${total_profit:,.2f}",
              delta=f"{total_profit:+.2f}")
    k6.metric("ROI от объёма",     f"{overall_roi:.2f}%")

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
                title="PnL стратегии по дням",
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
        st.markdown("**События с FIFO PnL и complete-set PnL**")
        disp = ev_df[[
            "title", "slug", "outcomes_detail", "trade_count",
            "total_contracts", "total_usdc",
            "leg_pnl_usdc", "complete_buy_qty", "complete_buy_price_sum",
            "complete_buy_pnl_usdc", "complete_sell_qty",
            "complete_sell_price_sum", "complete_sell_pnl_usdc",
            "profit_usdc", "roi_pct", "unmatched_long_qty",
            "unmatched_short_qty",
        ]].copy()
        disp.columns = [
            "Событие", "Slug", "Исходы", "Трейды",
            "Контракты всего", "Объём (USDC)",
            "PnL ног", "Buy sets", "Buy price sum",
            "PnL buy sets", "Sell sets",
            "Sell price sum", "PnL sell sets",
            "Итоговый PnL", "ROI %", "Остаток long",
            "Остаток short",
        ]
        money_cols = ["PnL ног", "PnL buy sets", "PnL sell sets", "Итоговый PnL"]
        for col in money_cols:
            disp[col] = disp[col].map(lambda x: f"${x:+,.4f}")
        disp["ROI %"]           = disp["ROI %"].map(lambda x: f"{x:.2f}%")
        disp["Buy price sum"]   = disp["Buy price sum"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        disp["Sell price sum"]  = disp["Sell price sum"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        disp["Контракты всего"] = disp["Контракты всего"].map(lambda x: f"{x:,.2f}")
        disp["Buy sets"]        = disp["Buy sets"].map(lambda x: f"{x:,.2f}")
        disp["Sell sets"]       = disp["Sell sets"].map(lambda x: f"{x:,.2f}")
        disp["Остаток long"]    = disp["Остаток long"].map(lambda x: f"{x:,.2f}")
        disp["Остаток short"]   = disp["Остаток short"].map(lambda x: f"{x:,.2f}")
        disp["Объём (USDC)"]    = disp["Объём (USDC)"].map(lambda x: f"${x:,.2f}")
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

# ══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 3 — PnL рассчитанных рынков
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not ENABLE_EXPIRATION_PNL:
        st.header("В разработке")
        st.info(
            "PnL экспираций временно отключен. "
            "Чтобы вернуть расчет, установите ENABLE_EXPIRATION_PNL = True в app.py."
        )
    else:
        st.header("PnL от экспираций")
        st.caption(
            "Итог события = сумма realizedPnl всех исходов из Data API /closed-positions. "
            "Это полный реализованный PnL события к экспирации; показываются только "
            "рассчитанные рынки с финальной ценой 0 или 1."
        )

        exp_df = apply_date_filter(expiration_analytics(closed_full), start_date, end_date)
        exp_daily = expiration_daily(exp_df)

        if exp_df.empty:
            st.info("Нет рассчитанных рынков в выбранном периоде.")
        else:
            total_exp_pnl = exp_df["expiration_pnl"].sum()
            profitable = int((exp_df["expiration_pnl"] > 0).sum())
            losing = int((exp_df["expiration_pnl"] < 0).sum())
            win_rate = profitable / len(exp_df) * 100

            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("PnL экспираций", f"${total_exp_pnl:,.2f}", delta=f"{total_exp_pnl:+.2f}")
            e2.metric("Рассчитанных событий", f"{len(exp_df):,}")
            e3.metric("Прибыльных", f"{profitable:,}")
            e4.metric("Убыточных", f"{losing:,}")
            e5.metric("Win rate", f"{win_rate:.1f}%")

            st.divider()

            _PLOT3 = dict(
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="#d1d5db",
                margin=dict(l=0, r=0, t=36, b=0),
            )

            fig_exp = go.Figure()
            fig_exp.add_bar(
                x=exp_daily["date"],
                y=exp_daily["daily_pnl"],
                name="Дневной PnL",
                marker_color=exp_daily["daily_pnl"].apply(
                    lambda value: "#10b981" if value >= 0 else "#ef4444"
                ),
                yaxis="y",
            )
            fig_exp.add_scatter(
                x=exp_daily["date"],
                y=exp_daily["cum_pnl"],
                name="Накопленный PnL",
                line=dict(color="#f59e0b", width=2),
                mode="lines+markers",
                marker=dict(size=4),
                yaxis="y2",
            )
            fig_exp.update_layout(
                title="PnL рассчитанных событий по дням",
                xaxis_title="Дата расчета",
                yaxis=dict(title="Дневной PnL (USDC)"),
                yaxis2=dict(title="Накопленный PnL (USDC)", overlaying="y", side="right"),
                legend=dict(orientation="h", y=-0.28),
                height=420,
                **_PLOT3,
            )
            st.plotly_chart(fig_exp, use_container_width=True)

            st.subheader("Разбивка по событиям")
            exp_display = exp_df.sort_values("date", ascending=False)[[
                "date",
                "title",
                "slug",
                "winner",
                "expiration_pnl",
                "total_bought",
                "outcomes_detail",
            ]].copy()
            exp_display.columns = [
                "Дата расчета",
                "Событие",
                "Slug",
                "Победитель",
                "PnL (USDC)",
                "Куплено контрактов",
                "Исходы",
            ]
            exp_display["PnL (USDC)"] = exp_display["PnL (USDC)"].map(lambda value: f"${value:+,.4f}")
            exp_display["Куплено контрактов"] = exp_display["Куплено контрактов"].map(
                lambda value: f"{value:,.2f}"
            )
            st.dataframe(exp_display, use_container_width=True, hide_index=True)

            with st.expander("Проверка источника"):
                st.write(f"**Строк closed-positions загружено:** {len(st.session_state.get('raw_closed', []))}")
                st.write(f"**Рассчитанных рынков после фильтра:** {len(exp_df)}")
                st.write("**Основной запрос:**")
                st.code(
                    "GET https://data-api.polymarket.com/closed-positions"
                    "?user=<wallet>&limit=50&offset=0&sortBy=TIMESTAMP&sortDirection=DESC"
                )
                st.write("**Аудит конкретного события по conditionId:**")
                st.code(
                    "GET https://data-api.polymarket.com/activity"
                    "?user=<wallet>&market=<conditionId>&limit=500&sortDirection=ASC"
                )

# ── Заглушки для остальных вкладок ────────────────────────────────────────────
for _tab, _label in [(tab4, "Вкладка 4"), (tab5, "Вкладка 5")]:
    with _tab:
        st.header(_label)
        st.info("🚧 В разработке")
