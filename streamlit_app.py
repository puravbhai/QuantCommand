import datetime
import html
import json
import math
import os
import time

from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

try:  # MASSIVE_API_KEY and ANTHROPIC_API_KEY can live in a .env file next to this script
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Enforce clean dark-mode config baseline for Streamlit
os.makedirs(".streamlit", exist_ok=True)
config_path = os.path.join(".streamlit", "config.toml")
with open(config_path, "w") as f:
    f.write("""
[theme]
base = "dark"
primaryColor = "#3b82f6"
backgroundColor = "#09090b"
secondaryBackgroundColor = "#121215"
textColor = "#e4e4e7"
""")

MARKET_TZ = ZoneInfo("America/New_York")
RISK_FREE_RATE = 0.045
HV_WINDOW = 30  # trading days of returns used for historical volatility
SESSION_MINUTES = 390  # 9:30 to 16:00 ET

# Page Configuration (Wide Layout for 3 cards per row)
st.set_page_config(
    page_title="Quant Command | v1.11.1",
    page_icon="⚡",
    layout="wide"
)

# --- Data Source Chosen by the Switch in the Toolbar (Massive needs MASSIVE_API_KEY) ---
DATA_SOURCE = st.session_state.get("data_source") or "Yahoo"
if DATA_SOURCE == "Massive" and not os.getenv("MASSIVE_API_KEY", "").strip():
    DATA_SOURCE = "Yahoo"

# --- Custom HTML/CSS Styling (Seamless Unified Card Architecture) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

        :root {
            --bg: #09090b; --card-bg: #121215; --border: #27272a;
            --accent: #3b82f6; --success: #10b981; --warning: #f59e0b; --danger: #ef4444;
            --text-main: #e4e4e7; --text-muted: #71717a;
            --sma50-color: #3b82f6; --sma200-color: #f59e0b;
        }

        .stApp {
            background-color: var(--bg);
            background-image: radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.15), transparent 40%);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
        }

        .header { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid var(--border); padding-bottom: 25px; margin-bottom: 30px; }
        .brand h1 { font-size: 2rem; font-weight: 800; letter-spacing: -1px; margin: 0; background: linear-gradient(to right, #fff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .brand .sub { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--accent); text-transform: uppercase; letter-spacing: 2px; margin-top: 8px; font-weight: 700; }
        .sys-info { text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; }
        .last-refresh { color: var(--accent); font-weight: 700; }

        .market-section-title { font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; color: var(--accent); margin-bottom: 15px; display: flex; align-items: center; gap: 8px; justify-content: center; }

        /* Unified Single-Card Shell Styling */
        .card-wrapper {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 30px;
            box-shadow: 0 20px 40px -20px rgba(0,0,0,0.8);
            transition: border-color 0.3s ease;
        }
        [class*="st-key-qcard-"] {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 30px;
            padding-bottom: 8px;
            box-shadow: 0 20px 40px -20px rgba(0,0,0,0.8);
            transition: border-color 0.3s ease;
        }
        [class*="st-key-qcard-buy-"] { border-color: var(--success); box-shadow: 0 0 30px -10px rgba(16, 185, 129, 0.15); }
        [class*="st-key-qcard-exit-"] { border-color: var(--danger); box-shadow: 0 0 30px -10px rgba(239, 68, 68, 0.15); }

        /* High-Visibility Earnings Alert Pill */
        .earnings-pill {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: rgba(245, 158, 11, 0.15);
            color: var(--warning);
            border: 1px solid var(--warning);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            font-weight: 800;
            padding: 3px 8px;
            border-radius: 12px;
            margin-left: 8px;
            vertical-align: middle;
        }

        /* Seamless Expander Integration inside the Single Card */
        [data-testid="stExpander"] {
            border: none !important;
            border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
            background-color: rgba(0, 0, 0, 0.25) !important;
            border-radius: 0px !important;
            margin: 0 !important;
            box-shadow: none !important;
        }
        [data-testid="stExpander"] details summary {
            background-color: transparent !important;
            padding: 12px 20px !important;
        }
        [data-testid="stExpander"] details summary p, 
        [data-testid="stExpander"] details summary span,
        [data-testid="stExpander"] summary svg {
            color: var(--text-main) !important;
            fill: var(--text-main) !important;
            font-weight: 700 !important;
            font-size: 0.75rem !important;
            letter-spacing: 0.5px;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            padding: 0 20px 15px 20px !important;
            background-color: rgba(0, 0, 0, 0.3) !important;
        }

        /* Force Selectbox & Dropdown Menus to Absolute Pitch Black (#000000) */
        div[data-baseweb="select"] > div {
            background-color: #000000 !important;
            color: #ffffff !important;
            border-color: var(--border) !important;
            border-radius: 6px !important;
        }
        div[data-baseweb="select"] span, div[data-baseweb="select"] div {
            color: #ffffff !important;
        }
        div[data-baseweb="popover"], ul[data-baseweb="menu"] {
            background-color: #000000 !important;
            border: 1px solid var(--border) !important;
        }
        li[data-baseweb="option"] {
            background-color: #000000 !important;
            color: #e4e4e7 !important;
        }
        li[data-baseweb="option"]:hover, li[data-baseweb="option"][aria-selected="true"] {
            background-color: #27272a !important;
            color: #ffffff !important;
        }

        .hud-section { padding: 15px 20px; border-bottom: 1px solid rgba(255,255,255,0.03); }
        .hud-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .hud-item { display: flex; flex-direction: column; gap: 4px; }
        .hud-label { font-size: 0.55rem; color: var(--text-muted); font-weight: 800; text-transform: uppercase; letter-spacing: 1.2px; }
        .hud-value { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; }

        .action-banner { margin: 15px 20px; padding: 10px; text-align: center; border-radius: 8px; font-weight: 800; text-transform: uppercase; font-size: 0.8rem; }
        .b-buy { background: rgba(16, 185, 129, 0.1); color: var(--success); border: 2px solid var(--success); }
        .b-hold { background: rgba(245, 158, 11, 0.1); color: var(--warning); border: 1px solid var(--warning); }
        .b-sell { background: rgba(239, 68, 68, 0.1); color: var(--danger); border: 2px solid var(--danger); }

        .logic-content { padding: 10px 0; font-size: 0.8rem; line-height: 1.5; color: var(--text-main); }

        .c-bull { color: var(--success); }
        .c-bear { color: var(--danger); }
        .c-neut { color: var(--warning); }
        .c-accent { color: var(--accent); }

        .hud-label[title] { cursor: help; text-decoration: underline dotted rgba(255,255,255,0.25); text-underline-offset: 3px; }
        .chart-legend { display: flex; gap: 14px; flex-wrap: wrap; padding: 10px 20px 0 20px; font-size: 0.68rem; color: var(--text-muted); }
        .chart-legend span { display: inline-flex; align-items: center; gap: 6px; }
        .chart-legend i { display: inline-block; width: 14px; height: 2px; border-radius: 1px; }
        .muted { color: var(--text-muted); }
        .ai-view { padding: 6px 0; font-size: 0.8rem; line-height: 1.5; color: var(--text-main); }
        .ai-top { display: flex; align-items: center; gap: 10px; font-size: 0.7rem; }
        .ai-pill { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; padding: 1px 8px;
                   border: 1px solid currentColor; border-radius: 999px; margin-left: 4px; }
        .ai-top .ai-pill { margin-left: 0; font-size: 0.75rem; }
        .ai-headline { font-weight: 600; color: #fff; margin: 8px 0 10px 0; }
        .ai-term p { margin: 3px 0 10px 0; }
        .ai-lists { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .ai-lists ul { margin: 3px 0 10px 0; padding-left: 16px; }
        .ai-foot { font-size: 0.66rem; color: var(--text-muted); border-top: 1px solid var(--border); padding-top: 6px; }
        [class*="st-key-qcard-"] [data-testid="stButtonGroup"] { padding: 6px 16px 0 16px; }
        [class*="st-key-qcard-"] [data-testid="stButtonGroup"] button { min-height: 26px; padding: 2px 7px; }
        [class*="st-key-qcard-"] [data-testid="stButtonGroup"] button p { font-size: 0.68rem; }
        [class*="st-key-qcard-"] [data-testid="stMetricValue"] { font-size: 1.15rem; font-family: 'JetBrains Mono', monospace; }
        [class*="st-key-qcard-"] [data-testid="stMetricLabel"] p { font-size: 0.7rem; color: var(--text-muted); }
        [class*="st-key-qcard-"] [data-testid="stMetricDelta"] { font-size: 0.7rem; }
        .signal-reason { margin: -6px 20px 4px 20px; text-align: center; font-size: 0.75rem; color: var(--text-muted); }
        .panel-title { margin: 14px 0 6px 0; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; color: var(--accent); }
        .trade-sentence { font-size: 0.85rem; line-height: 1.5; margin-bottom: 8px; }
        .check-row { padding: 3px 0 3px 4px; }
        .market-strip { display: flex; gap: 34px; flex-wrap: wrap; justify-content: center; align-items: baseline; margin: -12px 0 22px 0; font-family: 'JetBrains Mono', monospace; }
        .m-item { display: inline-flex; align-items: baseline; gap: 8px; cursor: default; }
        .m-name { font-size: 0.7rem; font-weight: 800; letter-spacing: 1.5px; color: var(--text-muted); }
        .m-val { font-size: 1rem; font-weight: 700; color: var(--text-main); }
        .m-chg { font-size: 0.7rem; font-weight: 700; }
        .m-note { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
        .summary-strip { display: flex; gap: 22px; flex-wrap: wrap; justify-content: center; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; margin: 6px 0 22px 0; }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown(f"""
    <div class="header">
        <div class="brand"><h1>QUANT COMMAND</h1><div class="sub">v1.11.1 | Signals · Strike Finder · Wheel Backtest</div></div>
        <div class="sys-info">
            <div>SOURCE: {DATA_SOURCE.upper()}</div>
            <div>CLOCK: {datetime.datetime.now().strftime('%H:%M:%S')}</div>
            <div>STATUS: <span class="last-refresh">ACTIVE</span></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Watchlist Tickers
tickers = ['NOW', 'NVDA', 'META', 'MSFT', 'AMZN', 'AVGO', 'CRM', 'ACN', 'SOFI', 'MP', 'AAPL', 'TSLA', 'COST', 'QQQ']


# --- Helper Normal CDF for Delta Estimation ---
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --- Black-Scholes Helpers (no dividends) ---
def bs_price(S, K, T, sigma, contract_type, r=RISK_FREE_RATE):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if contract_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_theta_per_day(S, K, T, sigma, contract_type, r=RISK_FREE_RATE):
    # Value the option loses per calendar day (returned as a positive number)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    decay = S * math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) * sigma / (2 * math.sqrt(T))
    carry = r * K * math.exp(-r * T) * (norm_cdf(d2) if contract_type == "call" else -norm_cdf(-d2))
    return (decay + carry) / 365.0


def option_value(S, K, T, sigma, contract_type):
    # Black-Scholes value before expiry, intrinsic value at expiry
    if T <= 0:
        return max(S - K, 0.0) if contract_type == "call" else max(K - S, 0.0)
    return bs_price(S, K, T, sigma, contract_type)


def strike_for_delta(S, target_delta, T, sigma, contract_type, r=RISK_FREE_RATE):
    # Invert call delta N(d1) = delta, or put delta N(d1) - 1 = -delta, for the strike
    target_delta = min(max(target_delta, 0.01), 0.99)
    d1 = NormalDist().inv_cdf(target_delta if contract_type == "call" else 1.0 - target_delta)
    return S * math.exp((r + 0.5 * sigma ** 2) * T - d1 * sigma * math.sqrt(T))


# --- Annualized Historical Volatility from Daily Log Returns ---
def historical_volatility(close, window=HV_WINDOW):
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * math.sqrt(252)


# --- Massive (formerly Polygon.io) REST Client, Used When the Page's Data Source Switch Is Set to Massive ---
MASSIVE_BASE = "https://api.massive.com"
YAHOO_PERIODS = {1: "1y", 2: "2y", 5: "5y", 10: "10y"}


class MassiveAccessError(Exception):
    """The key is missing, not entitled to this data, or rate limited: stop asking for the rest of the run."""


def massive_api_key():
    return os.getenv("MASSIVE_API_KEY", "").strip()


def massive_get(path_or_url, params=None):
    url = path_or_url if path_or_url.startswith("http") else MASSIVE_BASE + path_or_url
    # Key goes in a header rather than the URL so it never lands in logs
    resp = requests.get(url, params=params, headers={"Authorization": f"Bearer {massive_api_key()}"}, timeout=10)
    if resp.status_code in (401, 403, 429):
        raise MassiveAccessError(f"Massive returned HTTP {resp.status_code} for {path_or_url}")
    resp.raise_for_status()
    return resp.json()


def massive_paged(path, params, max_pages=5):
    data = massive_get(path, params)
    results = list(data.get("results") or [])
    for _ in range(max_pages - 1):
        if not data.get("next_url"):
            break
        data = massive_get(data["next_url"])
        results.extend(data.get("results") or [])
    return results


def massive_bars_frame(sym, multiplier, timespan, start, end):
    results = massive_paged(f"/v2/aggs/ticker/{sym}/range/{multiplier}/{timespan}/{start}/{end}",
                            {"adjusted": "true", "sort": "asc", "limit": 50000})
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(MARKET_TZ)
    df.index = pd.DatetimeIndex(idx.dt.normalize() if timespan == "day" else idx)
    return df[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=600)
def massive_stock_bars(sym, multiplier, timespan, start, end):
    return massive_bars_frame(sym, multiplier, timespan, start, end)


def massive_daily_bars(sym, years):
    end = datetime.date.today()
    return massive_stock_bars(sym, 1, "day", (end - datetime.timedelta(days=365 * years + 7)).isoformat(),
                              end.isoformat())


def load_daily_history(sym, years, source):
    """Daily OHLCV bars from the chosen source. Returns (bars, source actually used); falls back to Yahoo."""
    yahoo_cols = ["Open", "High", "Low", "Close", "Volume"]
    if source == "Massive" and massive_api_key():
        try:
            df = massive_daily_bars(sym, years)
            if not df.empty:
                # Plans cap how far back Massive goes (e.g. 2 years): fill anything older from Yahoo
                wanted_start = df.index[-1] - pd.DateOffset(years=years)
                if df.index[0] > wanted_start + pd.Timedelta(days=10):
                    older = yf.Ticker(sym).history(period=YAHOO_PERIODS[years], interval="1d")
                    older = older[older.index < df.index[0]][yahoo_cols]
                    if not older.empty:
                        older.index = older.index.tz_convert(MARKET_TZ)
                        return pd.concat([older, df]), "Massive"
                return df, "Massive"
        except Exception:
            pass
    return yf.Ticker(sym).history(period=YAHOO_PERIODS[years], interval="1d"), "Yahoo"


@st.cache_data(ttl=300)
def fetch_massive_option_by_delta(underlying, contract_type, target_delta, dte_days, current_price):
    # Chain snapshot carries Massive's own greeks, IV and NBBO quote for every contract
    try:
        today_date = datetime.date.today()
        target_date = today_date + datetime.timedelta(days=dte_days)
        rows = massive_paged(f"/v3/snapshot/options/{underlying}", {
            "contract_type": contract_type,
            "expiration_date.gte": max(today_date, target_date - datetime.timedelta(days=7)).isoformat(),
            "expiration_date.lte": (target_date + datetime.timedelta(days=7)).isoformat(),
            "limit": 250,
        }, max_pages=8)
        rows = [r for r in rows if (r.get("greeks") or {}).get("delta") is not None]
        if not rows:
            return None

        exp_str = min({r["details"]["expiration_date"] for r in rows},
                      key=lambda x: abs((datetime.date.fromisoformat(x) - target_date).days))
        rows = [r for r in rows if r["details"]["expiration_date"] == exp_str]
        best = min(rows, key=lambda r: abs(abs(r["greeks"]["delta"]) - target_delta))

        quote = best.get("last_quote") or {}
        bid, ask = float(quote.get("bid") or 0), float(quote.get("ask") or 0)
        premium = float(quote.get("midpoint") or ((bid + ask) / 2 if bid > 0 and ask > 0 else 0)
                        or (best.get("day") or {}).get("close") or 0)
        if premium <= 0:
            return None
        exp_date = datetime.date.fromisoformat(exp_str)
        return {
            "strike": float(best["details"]["strike_price"]),
            "premium": premium,
            "bid": bid,
            "ask": ask,
            "delta": float(best["greeks"]["delta"]),
            "expiry": exp_date,
            "dte": (exp_date - today_date).days,
            "theta": abs(float(best["greeks"].get("theta") or 0)),
            "iv": float(best.get("implied_volatility") or 0),
            "ticker": best["details"].get("ticker", ""),
            "is_live": True
        }
    except Exception:
        return None


@st.cache_data(ttl=86400)
def massive_trade_contract(underlying, contract_type, entry_iso, target_exp_iso, model_strike):
    # The real listed contract nearest the backtest's expiry and model strike, as it existed on the entry date
    entry = datetime.date.fromisoformat(entry_iso)
    target_exp = datetime.date.fromisoformat(target_exp_iso)
    contracts = massive_paged("/v3/reference/options/contracts", {
        "underlying_ticker": underlying,
        "contract_type": contract_type,
        "as_of": entry_iso,
        "expiration_date.gte": (entry + datetime.timedelta(days=1)).isoformat(),
        "expiration_date.lte": (target_exp + datetime.timedelta(days=7)).isoformat(),
        "strike_price.gte": round(model_strike * 0.8, 2),
        "strike_price.lte": round(model_strike * 1.2, 2),
        "limit": 1000,
    }, max_pages=3)
    if not contracts:
        return None
    exp_str = min({c["expiration_date"] for c in contracts},
                  key=lambda x: (abs((datetime.date.fromisoformat(x) - target_exp).days), x))
    best = min((c for c in contracts if c["expiration_date"] == exp_str),
               key=lambda c: abs(float(c["strike_price"]) - model_strike))
    return {"ticker": best["ticker"], "strike": float(best["strike_price"]), "expiration": exp_str}


@st.cache_data(ttl=86400)
def massive_option_closes(option_ticker, start_iso, end_iso):
    # Daily closing trade price per date (trade-based bars: no bid/ask, and illiquid days may be missing)
    df = massive_bars_frame(option_ticker, 1, "day", start_iso, end_iso)
    return {} if df.empty else {ts.date(): float(c) for ts, c in df["Close"].items()}


# --- Expander that Reports Whether It Is Open (newer Streamlit), so Hidden Content Can Skip Work ---
def lazy_expander(label, key):
    try:
        expander = st.expander(label, key=key, on_change="rerun")
        return expander, expander.open
    except TypeError:  # older Streamlit without tracked expanders: always render the contents
        return st.expander(label), True


# --- Fraction of Today's Session Elapsed (1.0 once the bar is a completed day) ---
def session_fraction(last_bar_ts):
    now = datetime.datetime.now(MARKET_TZ)
    if last_bar_ts.date() != now.date():
        return 1.0
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    elapsed = (now - market_open).total_seconds() / (SESSION_MINUTES * 60)
    return min(1.0, max(0.1, elapsed))


# --- Earnings Report Timing from Yahoo's Scheduled Timestamp ---
def earnings_timing(ts):
    # Yahoo stamps each report with its scheduled time in US/Eastern; midnight means no time was given
    ts = ts.tz_convert(MARKET_TZ) if ts.tzinfo else ts
    minutes = ts.hour * 60 + ts.minute
    if minutes == 0:
        return "Time Not Confirmed"
    if minutes < 9 * 60 + 30:
        return "Before Market Open"
    if minutes >= 16 * 60:
        return "After Market Close"
    return "During Market Hours"


# --- Yahoo Finance Upcoming Earnings Fetcher ---
@st.cache_data(ttl=3600)
def fetch_yahoo_upcoming_earnings(sym):
    if sym == 'QQQ':
        return None, "N/A", "N/A"

    try:
        ticker_obj = yf.Ticker(sym)
        earnings_df = ticker_obj.get_earnings_dates(limit=8)
        if earnings_df is not None and not earnings_df.empty:
            today = datetime.date.today()
            future_stamps = [d for d in earnings_df.index if d.date() >= today]
            if future_stamps:
                next_stamp = min(future_stamps)
                next_date = next_stamp.date()
                eps_est = "N/A"
                if "EPS Estimate" in earnings_df.columns:
                    match_row = earnings_df[earnings_df.index.date == next_date]
                    if not match_row.empty and "EPS Estimate" in match_row.columns:
                        val = match_row["EPS Estimate"].iloc[0]
                        if not pd.isna(val):
                            eps_est = f"${val:.2f}"
                return next_date, eps_est, earnings_timing(next_stamp)

        cal = ticker_obj.calendar
        if cal is not None:
            if isinstance(cal, dict) and "Earnings Date" in cal:
                dates = cal["Earnings Date"]
                if dates:
                    return dates[0].date() if hasattr(dates[0], 'date') else dates[0], "N/A", "Time Not Confirmed"
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    d = cal.loc["Earnings Date"].iloc[0]
                    return d.date() if hasattr(d, 'date') else d, "N/A", "Time Not Confirmed"
                elif "Earnings Date" in cal.columns:
                    d = cal["Earnings Date"].iloc[0]
                    return d.date() if hasattr(d, 'date') else d, "N/A", "Time Not Confirmed"
    except Exception:
        pass

    return None, "N/A", "N/A"


# --- Yahoo Finance Past & Upcoming Earnings Dates (for flagging backtest trades) ---
@st.cache_data(ttl=86400)
def fetch_earnings_history(sym):
    try:
        earnings_df = yf.Ticker(sym).get_earnings_dates(limit=28)  # ~5 years of quarterly reports for the 5Y backtest
        if earnings_df is not None and not earnings_df.empty:
            return sorted({d.date() for d in earnings_df.index})
    except Exception:
        pass
    return []


# --- Yahoo Finance Options Chain Fetcher by Exact Delta Scan ---
@st.cache_data(ttl=300)
def fetch_yahoo_option_by_delta(underlying, contract_type, target_delta, dte_days, current_price):
    try:
        ticker_obj = yf.Ticker(underlying)
        expirations = ticker_obj.options
        if not expirations:
            return None

        today_date = datetime.date.today()
        target_date = today_date + datetime.timedelta(days=dte_days)
        closest_exp = min(expirations,
                          key=lambda x: abs((datetime.datetime.strptime(x, "%Y-%m-%d").date() - target_date).days))
        exp_date = datetime.datetime.strptime(closest_exp, "%Y-%m-%d").date()
        actual_dte = (exp_date - today_date).days

        opt_chain = ticker_obj.option_chain(closest_exp)
        chain_df = opt_chain.calls if contract_type.lower() == 'call' else opt_chain.puts

        if chain_df.empty or 'impliedVolatility' not in chain_df.columns:
            return None

        chain_df = chain_df[
            (chain_df['impliedVolatility'] > 0.01) & ((chain_df['bid'] > 0) | (chain_df['lastPrice'] > 0))]
        if chain_df.empty:
            return None

        # Time to the expiration actually found, which may differ from the requested Friday
        T = max(actual_dte, 1) / 365.0
        r = RISK_FREE_RATE

        best_row = None
        best_delta = None
        min_diff = float('inf')

        for idx, row in chain_df.iterrows():
            K = float(row['strike'])
            sigma = float(row['impliedVolatility'])

            if sigma > 0 and T > 0:
                d1 = (math.log(current_price / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                calc_delta = norm_cdf(d1) if contract_type.lower() == 'call' else norm_cdf(d1) - 1.0

                diff = abs(abs(calc_delta) - target_delta)
                if diff < min_diff:
                    min_diff = diff
                    best_row = row
                    best_delta = calc_delta

        if best_row is not None:
            K = float(best_row['strike'])
            sigma = float(best_row['impliedVolatility'])
            bid = float(best_row.get('bid', 0))
            ask = float(best_row.get('ask', 0))
            last_price = float(best_row.get('lastPrice', 0))
            mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else last_price

            if mid_price > 0:
                return {
                    "strike": K,
                    "premium": mid_price,
                    "bid": bid,
                    "ask": ask,
                    "delta": best_delta,
                    "expiry": exp_date,
                    "dte": actual_dte,
                    "theta": bs_theta_per_day(current_price, K, T, sigma, contract_type.lower()),
                    "iv": sigma,
                    "ticker": best_row.get('contractSymbol', ''),
                    "is_live": True
                }
    except Exception:
        pass
    return None


# --- Complete Full-Cycle Wheel Backtest Engine with Configurable 6M or 5Y Lookback ---
# Strikes are solved from the target delta and premiums priced with Black-Scholes, using the
# stock's own trailing historical volatility on each entry date scaled by today's IV/HV ratio
# (a stand-in for the implied volatility and skew the market would have charged).
@st.cache_data(ttl=600)
def run_full_wheel_backtest(sym, initial_strategy, target_delta, dte_days, use_five_year, iv_ratio=1.0,
                            earnings_dates=(), take_profit=False, stop_loss=False, source="Yahoo"):
    try:
        # Pull extra history so the volatility window is already warmed up on day one
        full_df, _ = load_daily_history(sym, 10 if use_five_year else 1, source)
        if full_df.empty:
            return None, None, None, None
        full_df['HV'] = historical_volatility(full_df['Close'])
        lookback = pd.DateOffset(years=5) if use_five_year else pd.DateOffset(months=6)
        hist_df = full_df[full_df.index >= full_df.index[-1] - lookback].dropna(subset=['HV'])
        if len(hist_df) < 2 or (hist_df.index[-1] - hist_df.index[0]).days < dte_days + 7:
            return None, None, None, None

        initial_price = float(hist_df['Close'].iloc[0])
        initial_investment = initial_price * 100.0
        T = max(dte_days, 1) / 365.0

        if initial_strategy == "Cash-Secured Put":
            cash = initial_investment
            shares = 0.0
            active_mode = "PUT"
        else:
            cash = 0.0
            shares = 100.0
            active_mode = "CALL"

        i = 0
        trade_records = []
        daily_records = []
        event_markers = []
        total_trades = 0
        wins = 0
        earnings_trades = 0
        real_priced_trades = 0
        # With Massive selected, each trade uses the real listed contract and its daily closing prices
        use_real_options = source == "Massive" and bool(massive_api_key())
        massive_note = ""
        massive_since = None
        access_failures = 0
        if use_real_options:
            try:
                # Only ask for option prices inside the plan's history window
                massive_since = massive_daily_bars(sym, 10 if use_five_year else 1).index[0].date()
            except Exception:
                use_real_options = False

        while i < len(hist_df) - 1:
            s_start = float(hist_df['Close'].iloc[i])
            sigma = float(hist_df['HV'].iloc[i]) * iv_ratio
            entry_date = hist_df.index[i]
            entry_date_str = entry_date.strftime('%b %d, %Y')

            initial_wheel_val = cash + (shares * s_start if shares > 0 else 0)
            if active_mode == "CALL":
                event_markers.append({
                    "Date": entry_date,
                    "Balance": initial_wheel_val,
                    "Event": "Sell Call Option",
                    "Symbol": "square",
                    "Color": "#3b82f6"
                })
            else:
                event_markers.append({
                    "Date": entry_date,
                    "Balance": initial_wheel_val,
                    "Event": "Sell PUT Option",
                    "Symbol": "diamond",
                    "Color": "#a855f7"
                })

            if active_mode == "CALL":
                if shares == 0:
                    cash -= s_start * 100.0
                    shares = 100.0
                    event_markers.append({
                        "Date": entry_date,
                        "Balance": cash + (shares * s_start),
                        "Event": "Buy Stock",
                        "Symbol": "triangle-up",
                        "Color": "#10b981"
                    })
                contract_type = "call"
            else:
                if shares > 0:
                    cash += s_start * 100.0
                    shares = 0.0
                    event_markers.append({
                        "Date": entry_date,
                        "Balance": cash,
                        "Event": "sell Stock",
                        "Symbol": "triangle-down",
                        "Color": "#ef4444"
                    })
                contract_type = "put"

            strike = strike_for_delta(s_start, target_delta, T, sigma, contract_type)
            est_prem = bs_price(s_start, strike, T, sigma, contract_type)

            # DTE is calendar days: expire on the last trading day on or before entry + DTE
            # (an expiry that lands on a weekend or holiday moves to the prior session)
            target_exp = entry_date + pd.Timedelta(days=dte_days)

            real_closes = None
            if use_real_options and entry_date.date() >= massive_since:
                try:
                    contract = massive_trade_contract(sym, contract_type, entry_date.date().isoformat(),
                                                      target_exp.date().isoformat(), round(strike, 2))
                    if contract:
                        closes = massive_option_closes(contract["ticker"], entry_date.date().isoformat(),
                                                       contract["expiration"])
                        if closes.get(entry_date.date(), 0) > 0:
                            real_closes = closes
                            strike = contract["strike"]
                            est_prem = closes[entry_date.date()]
                            target_exp = pd.Timestamp(contract["expiration"]).tz_localize(hist_df.index.tz)
                            real_priced_trades += 1
                    access_failures = 0
                except MassiveAccessError as err:
                    # Rate limited, or repeatedly not entitled: model-price the rest of the run
                    access_failures += 1
                    if "429" in str(err) or access_failures >= 5:
                        use_real_options = False
                        massive_note = str(err)
                except Exception:
                    pass
            cash += est_prem * 100.0

            exp_idx = int(hist_df.index.searchsorted(target_exp, side='right')) - 1
            exp_idx = min(max(exp_idx, i + 1), len(hist_df) - 1)
            exp_date = hist_df.index[exp_idx]

            closed_early = False
            stopped_out = False
            buyback = 0.0
            for d_idx in range(i, exp_idx + 1):
                d_price = float(hist_df['Close'].iloc[d_idx])
                d_date = hist_df.index[d_idx]
                d_sigma = float(hist_df['HV'].iloc[d_idx]) * iv_ratio
                opt_val = option_value(d_price, strike, (exp_date - d_date).days / 365.0, d_sigma, contract_type)
                if real_closes and d_date < exp_date and real_closes.get(d_date.date(), 0) > 0:
                    opt_val = real_closes[d_date.date()]

                # Buy the option back once it has lost half its value (50% of max profit captured)
                if take_profit and i < d_idx < exp_idx and opt_val <= 0.5 * est_prem:
                    buyback = opt_val
                    cash -= buyback * 100.0
                    opt_val = 0.0
                    closed_early = True
                    exp_idx = d_idx
                    exp_date = d_date

                # Stop loss: buy the option back once it is worth twice the premium collected (avoids assignment)
                elif stop_loss and d_idx > i and opt_val >= 2.0 * est_prem:
                    buyback = opt_val
                    cash -= buyback * 100.0
                    opt_val = 0.0
                    stopped_out = True
                    exp_idx = d_idx
                    exp_date = d_date

                # The short option is marked to market, so drawdowns show up on the day they happen
                daily_records.append({
                    "Date": d_date,
                    "Wheel Balance": cash + (shares * d_price if shares > 0 else 0) - opt_val * 100.0,
                    "BuyHold Balance": initial_investment * (d_price / initial_price),
                    "Stock Price": d_price
                })
                if closed_early or stopped_out:
                    break

            s_end = float(hist_df['Close'].iloc[exp_idx])
            total_trades += 1

            if closed_early:
                wins += 1
                result = "Closed at 50% Max Profit"
                exp_wheel_val = cash + (shares * s_end if shares > 0 else 0)
                event_markers.append({
                    "Date": exp_date,
                    "Balance": exp_wheel_val,
                    "Event": "Closed at 50% profit",
                    "Symbol": "star",
                    "Color": "#22d3ee"
                })
            elif stopped_out:
                result = "Stopped Out at 2× Premium (Kept Position)"
                exp_wheel_val = cash + (shares * s_end if shares > 0 else 0)
                event_markers.append({
                    "Date": exp_date,
                    "Balance": exp_wheel_val,
                    "Event": "Stopped out (2× premium)",
                    "Symbol": "x",
                    "Color": "#f97316"
                })
            elif active_mode == "CALL" and s_end > strike:
                cash += strike * 100.0
                shares = 0.0
                result = "Assigned (Called Away) ➔ Switch to Put"
                active_mode = "PUT"
                exp_wheel_val = cash
                event_markers.append({
                    "Date": exp_date,
                    "Balance": exp_wheel_val,
                    "Event": "Option exercised",
                    "Symbol": "triangle-down",
                    "Color": "#ef4444"
                })
                event_markers.append({
                    "Date": exp_date,
                    "Balance": exp_wheel_val,
                    "Event": "sell Stock",
                    "Symbol": "triangle-down",
                    "Color": "#ef4444"
                })
            elif active_mode == "PUT" and s_end < strike:
                cash -= strike * 100.0
                shares = 100.0
                result = "Assigned (Bought Stock) ➔ Switch to Call"
                active_mode = "CALL"
                exp_wheel_val = cash + (shares * s_end)
                event_markers.append({
                    "Date": exp_date,
                    "Balance": exp_wheel_val,
                    "Event": "Option exercised",
                    "Symbol": "triangle-up",
                    "Color": "#10b981"
                })
                event_markers.append({
                    "Date": exp_date,
                    "Balance": exp_wheel_val,
                    "Event": "Buy Stock",
                    "Symbol": "triangle-up",
                    "Color": "#10b981"
                })
            else:
                wins += 1
                result = "Expired OTM (Retained Shares)" if active_mode == "CALL" else "Expired OTM (Retained Cash)"
                exp_wheel_val = cash + (shares * s_end if shares > 0 else 0)
                event_markers.append({
                    "Date": exp_date,
                    "Balance": exp_wheel_val,
                    "Event": "Option expired",
                    "Symbol": "circle",
                    "Color": "#10b981"
                })

            current_portfolio_value = cash + (shares * s_end if shares > 0 else 0)

            # Real premiums run richer when an earnings report falls inside the option's life
            spans_earnings = any(entry_date.date() < e <= exp_date.date() for e in earnings_dates)
            if spans_earnings:
                earnings_trades += 1

            trade_records.append({
                "Trade #": total_trades,
                "Entry Date": entry_date_str,
                "Expiry Date": exp_date.strftime('%b %d, %Y'),
                "Stock Entry": f"${s_start:.2f}",
                "Stock Expiry": f"${s_end:.2f}",
                "Strike": f"${strike:.2f}",
                "Model Vol": f"{sigma * 100:.0f}%",
                "Pricing": "Massive" if real_closes else "Model",
                "Earnings": "⚠️ Spans earnings" if spans_earnings else "",
                "Result": result,
                "Premium": f"${est_prem * 100:,.2f}",
                "Buyback": f"${buyback * 100:,.2f}" if closed_early or stopped_out else "",
                "Investment Balance": f"${current_portfolio_value:,.2f}"
            })

            i = exp_idx + 1
            if i >= len(hist_df):
                break

        final_price = float(hist_df['Close'].iloc[-1])
        final_portfolio_value = cash + (shares * final_price if shares > 0 else 0)

        total_return_pct = ((final_portfolio_value - initial_investment) / initial_investment) * 100.0
        buy_hold_return_pct = ((final_price - initial_price) / initial_price) * 100.0
        win_ratio = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        metrics = {
            "initial_investment": initial_investment,
            "final_value": final_portfolio_value,
            "total_return_pct": total_return_pct,
            "buy_hold_return_pct": buy_hold_return_pct,
            "win_ratio": win_ratio,
            "total_trades": total_trades,
            "earnings_trades": earnings_trades,
            "real_priced_trades": real_priced_trades,
            "massive_note": massive_note,
            "massive_since": massive_since
        }

        trades_df = pd.DataFrame(trade_records)
        daily_df = pd.DataFrame(daily_records).drop_duplicates(subset=["Date"])
        for col, name in (("Wheel Balance", "max_drawdown_pct"), ("BuyHold Balance", "bh_max_drawdown_pct")):
            metrics[name] = float((daily_df[col] / daily_df[col].cummax() - 1.0).min() * 100.0)
        events_df = pd.DataFrame(event_markers)
        return metrics, trades_df, daily_df, events_df
    except Exception:
        return None, None, None, None


# --- Streamlit Modal Dialog for Charts with Vertical Grid Lines & Bottom Title ---
@st.dialog("Simulated Trades Breakdown & Performance Chart", width="large")
def show_trade_log_dialog(sym, trades_df, daily_df, events_df, strategy_name):
    st.markdown(f"### 📈 Wheel Backtest Performance: **{sym}** ({strategy_name} Start)")
    st.markdown("Chronological performance comparison with trade event icons, vertical tracking lines, and ledger.")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
        subplot_titles=("", "")
    )

    fig.add_trace(go.Scatter(x=daily_df["Date"], y=daily_df["Wheel Balance"], mode='lines', name='Wheel Strategy',
                             line=dict(color='#3b82f6', width=2)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=daily_df["Date"], y=daily_df["BuyHold Balance"], mode='lines', name='Buy & Hold Benchmark',
                   line=dict(color='#f59e0b', width=1.5, dash='dash')), row=1, col=1)

    if events_df is not None and not events_df.empty:
        for event_name in events_df["Event"].unique():
            subset = events_df[events_df["Event"] == event_name]
            fig.add_trace(go.Scatter(
                x=subset["Date"],
                y=subset["Balance"],
                mode='markers',
                name=event_name,
                marker=dict(
                    symbol=subset["Symbol"].tolist(),
                    color=subset["Color"].tolist(),
                    size=12,
                    line=dict(width=1, color='#ffffff')
                )
            ), row=1, col=1)

        for _, row in events_df.iterrows():
            fig.add_vline(x=row["Date"], line_width=1, line_dash="dot", line_color="rgba(255, 255, 255, 0.2)")

    fig.add_trace(go.Scatter(x=daily_df["Date"], y=daily_df["Stock Price"], mode='lines', name='Stock Price',
                             line=dict(color='#10b981', width=1.5)), row=2, col=1)

    fig.update_layout(
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        margin=dict(l=20, r=20, t=20, b=50),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color='#e4e4e7', size=9)),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#71717a', size=9)),
        xaxis2=dict(showgrid=False, zeroline=False, tickfont=dict(color='#71717a', size=9),
                    title=dict(text="Portfolio Investment Balance (Top) & Underlying Stock Price (Bottom)",
                               font=dict(size=11, color='#a1a1aa'))),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False,
                   tickfont=dict(color='#71717a', size=9), side='right'),
        yaxis2=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False,
                    tickfont=dict(color='#71717a', size=9), side='right')
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("#### 📋 Chronological Trade Ledger")
    st.dataframe(trades_df, use_container_width=True, hide_index=True)


# --- Market Data Fetching Function ---
# Price, prior closes, volume and indicators all come from the same yfinance daily bars, so they
# describe one point in time. During market hours the last bar is today's live bar.
@st.cache_data(ttl=600)
def fetch_ticker_data(sym, source="Yahoo"):
    df, used_source = load_daily_history(sym, 2, source)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not df.empty:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['SMA_5'] = ta.sma(df['Close'], length=5)
        df['HV'] = historical_volatility(df['Close'])

        cp = float(df['Close'].iloc[-1])
        sma50 = float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else cp
        sma200 = float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else cp
        sma5 = float(df['SMA_5'].iloc[-1]) if not pd.isna(df['SMA_5'].iloc[-1]) else cp
        rsi_val = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
        hv = float(df['HV'].iloc[-1]) if not pd.isna(df['HV'].iloc[-1]) else 0.30

        h1 = float(df['Close'].iloc[-2]) if len(df) > 1 else cp
        h2 = float(df['Close'].iloc[-3]) if len(df) > 2 else cp
        vol = float(df['Volume'].iloc[-1])
        avg_vol = float(df['Volume'].iloc[-11:-1].mean()) if len(df) >= 11 else vol
        # Compare a partial day against the volume normally traded by this point in the session
        expected_vol = avg_vol * session_fraction(df.index[-1])
        rvol = vol / expected_vol if expected_vol > 0 else 1.0

        chart_df = df[['Close', 'SMA_50', 'SMA_200']].tail(90).copy()
        chart_df['SMA_200'] = chart_df['SMA_200'].bfill().ffill()
        chart_df.columns = ['Price', '50 SMA', '200 SMA']

        return cp, sma50, sma200, sma5, rsi_val, h1, h2, rvol, hv, chart_df, used_source
    return None


# --- Price Chart Data for the Card's Range Buttons ---
CHART_RANGES = ["1D", "1W", "1M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y"]
INTRADAY_RANGES = {"1D": ("1d", "5m"), "1W": ("5d", "30m")}
DAILY_LOOKBACK = {"1M": pd.DateOffset(months=1), "3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6),
                  "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2), "3Y": pd.DateOffset(years=3),
                  "4Y": pd.DateOffset(years=4), "5Y": pd.DateOffset(years=5)}


@st.cache_data(ttl=3600)
def fetch_daily_chart_history(sym, source="Yahoo"):
    # Extra years so the 200-day average already exists at the start of a 5-year window
    df, _ = load_daily_history(sym, 10, source)
    if df.empty:
        return None
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    return df[['Close', 'SMA_50', 'SMA_200']]


@st.cache_data(ttl=300)
def fetch_intraday_chart_history(sym, period, interval, source="Yahoo"):
    if source == "Massive" and massive_api_key():
        try:
            minutes = int(interval[:-1])
            end = datetime.date.today()
            df = massive_stock_bars(sym, minutes, "minute", (end - datetime.timedelta(days=10)).isoformat(),
                                    end.isoformat())
            if not df.empty:
                # Regular session only, then the last 1 or 5 trading days
                df = df.between_time("09:30", "15:59")
                sessions = sorted(set(df.index.date))[-(1 if period == "1d" else 5):]
                df = df[np.isin(df.index.date, sessions)]
                if not df.empty:
                    return df[['Close']]
        except Exception:
            pass
    df = yf.Ticker(sym).history(period=period, interval=interval)
    return None if df.empty else df[['Close']]


def chart_data_for_range(sym, chart_range, source="Yahoo"):
    if chart_range in INTRADAY_RANGES:
        return fetch_intraday_chart_history(sym, *INTRADAY_RANGES[chart_range], source)
    df = fetch_daily_chart_history(sym, source)
    if df is None:
        return None
    return df[df.index >= df.index[-1] - DAILY_LOOKBACK[chart_range]]


# --- Market Strip: Major Indices and the VIX (always from Yahoo) ---
MARKET_INDICES = [("^DJI", "DOW", "Dow Jones Industrial Average"),
                  ("^IXIC", "NASDAQ", "Nasdaq Composite"),
                  ("^VIX", "VIX", "CBOE Volatility Index: the market's expected S&P 500 volatility over the next 30 days")]


@st.cache_data(ttl=300)
def fetch_index_quote(sym):
    try:
        closes = yf.Ticker(sym).history(period="5d", interval="1d")['Close'].dropna()
        last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
        return last, last - prev, (last / prev - 1.0) * 100.0
    except Exception:
        return None


def vix_regime(vix):
    return "calm" if vix < 15 else ("normal" if vix < 20 else ("elevated" if vix < 30 else "high fear"))


# --- 30-Day At-the-Money Implied Volatility for a Card ---
def fetch_atm_iv(sym, cp, source):
    opt = fetch_massive_option_by_delta(sym, "call", 0.5, 30, cp) if source == "Massive" else None
    opt = opt or fetch_yahoo_option_by_delta(sym, "call", 0.5, 30, cp)
    return opt["iv"] if opt and opt.get("iv", 0) > 0 else None


# --- Signal Engine (shared by the cards, the summary strip, filters and sorting) ---
def compute_signals(data):
    cp, sma50, sma200, sma5, rsi_val, h1, h2, rvol, hv, chart_df, used_source = data

    slope_up = cp > sma5
    days_above = sum(1 for p in [cp, h1, h2] if p > sma50)
    rule3 = "SECURE" if days_above == 3 else "TESTING"
    structural_decay = sum(1 for p in [cp, h1, h2] if p < sma50) == 3

    is_buy = (cp > sma50 and rvol > 1.0 and slope_up and rule3 == "SECURE")
    is_sell = structural_decay and (sma50 < sma200)
    regime = "EXIT / CASH" if is_sell else (
        "HIGH CONVICTION BUY" if is_buy and rvol > 1.5 else ("SPECULATIVE BUY" if is_buy else "HOLD / MONITOR"))

    score = int(
        max(0, min(100, (40 if cp > sma50 else 10) + (20 if rvol > 1.2 else 0) + (20 if cp > sma200 else 0) + 20)))

    # Plain-English conditions behind the label, shown in the "Why this signal?" checklist
    buy_checks = [
        (cp > sma50, f"Price above its 50-day average (${cp:.2f} vs ${sma50:.2f})"),
        (rule3 == "SECURE", f"Closed above the 50-day average 3 days in a row ({days_above} of 3)"),
        (slope_up, f"Price above its 5-day average (short-term momentum, ${sma5:.2f})"),
        (rvol > 1.0, f"Volume above normal ({rvol:.2f}× the 10-day average)"),
    ]
    exit_checks = [
        (structural_decay, "Closed below the 50-day average 3 days in a row"),
        (sma50 < sma200, f"50-day average below the 200-day (${sma50:.2f} vs ${sma200:.2f})"),
    ]

    if is_sell:
        reason = "Below the 50-day for 3 days while the 50-day is under the 200-day"
    elif is_buy:
        reason = f"Uptrend confirmed on {rvol:.1f}× normal volume"
    else:
        missing = [text.split(" (")[0].lower() for ok, text in buy_checks if not ok]
        reason = "Not a buy yet: needs " + missing[0] if missing else "Watching for confirmation"

    return {
        "regime": regime,
        "b_class": "b-sell" if is_sell else ("b-buy" if is_buy else "b-hold"),
        "card_tone": "buy" if regime == "HIGH CONVICTION BUY" else ("exit" if regime == "EXIT / CASH" else "neutral"),
        "is_buy": is_buy,
        "is_sell": is_sell,
        "score": score,
        "reason": reason,
        "buy_checks": buy_checks,
        "exit_checks": exit_checks,
    }


def days_until_earnings(sym):
    next_date, _, _ = fetch_yahoo_upcoming_earnings(sym)
    return (next_date - today).days if next_date else None


def hud_item(label, tip, value, css_class="", style=""):
    return (f'<div class="hud-item"><span class="hud-label" title="{tip}">{label}</span>'
            f'<span class="hud-value {css_class}" style="{style}">{value}</span></div>')


# --- AI View: Claude's Read on Sentiment, Short-Term and Long-Term Outlook (needs ANTHROPIC_API_KEY) ---
AI_MODEL = os.getenv("QC_AI_MODEL", "claude-opus-5-5")
AI_CACHE_HOURS = 4
AI_OUTLOOKS = ["Bullish", "Neutral", "Bearish"]
AI_VIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": AI_OUTLOOKS},
        "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
        "headline": {"type": "string"},
        "short_term": {
            "type": "object",
            "properties": {"outlook": {"type": "string", "enum": AI_OUTLOOKS}, "summary": {"type": "string"}},
            "required": ["outlook", "summary"],
            "additionalProperties": False,
        },
        "long_term": {
            "type": "object",
            "properties": {"outlook": {"type": "string", "enum": AI_OUTLOOKS}, "summary": {"type": "string"}},
            "required": ["outlook", "summary"],
            "additionalProperties": False,
        },
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "options_angle": {"type": "string"},
    },
    "required": ["sentiment", "confidence", "headline", "short_term", "long_term", "catalysts", "risks",
                 "options_angle"],
    "additionalProperties": False,
}
AI_SYSTEM_PROMPT = """You are a buy-side equity and options analyst writing a short briefing for an experienced \
retail trader who sells covered calls and cash-secured puts (the wheel).

You get a snapshot from the trader's dashboard: price, trend and momentum indicators, volatility, earnings date, \
recent returns, and recent news headlines. Form a view from that data plus what you know about the company, its \
sector and its business model.

- sentiment: your overall read right now. confidence reflects how strongly the data agrees.
- headline: one sentence, the single most important thing about this stock right now.
- short_term: the next 1 to 4 weeks. Lean on the trend, momentum, volume, volatility, news and any earnings date in \
the window. Mention concrete price levels from the data where useful. 2 to 3 sentences.
- long_term: 6 to 12 months and beyond. Lean on the business: competitive position, growth drivers, valuation \
concerns, sector trends. 2 to 3 sentences.
- catalysts and risks: 2 to 3 short items each, specific to this stock.
- options_angle: one or two sentences on what the current setup means for selling calls or puts (for example \
whether premiums look rich versus realized volatility, or whether earnings fall inside a typical 30-day trade).

Be specific and plain-spoken, and disagree with the dashboard's signal when the data supports that. Your knowledge \
of company fundamentals has a training cutoff, so treat the dashboard numbers and headlines as the current truth \
and do not state recent financial figures you are not given. This is analysis, not a recommendation to trade."""


@st.cache_data(ttl=1800)
def fetch_news_headlines(sym, limit=8):
    # yfinance has returned news in two shapes; read either
    try:
        items = yf.Ticker(sym).news or []
    except Exception:
        return []
    headlines = []
    for item in items[:limit]:
        content = item.get("content") or item
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else content.get("publisher")
        when = content.get("pubDate") or content.get("providerPublishTime")
        if isinstance(when, (int, float)):
            when = datetime.datetime.fromtimestamp(when, MARKET_TZ).date().isoformat()
        elif isinstance(when, str):
            when = when[:10]
        headlines.append(" | ".join(str(x) for x in (when, publisher, title) if x))
    return headlines


def ai_view_context(s, data, sig, atm_iv, next_earnings_date, earnings_time):
    """Plain-text snapshot of everything the card knows, sent to Claude as the user message."""
    cp, sma50, sma200, sma5, rsi_val, h1, h2, rvol, hv, chart_df, used_source = data
    lines = [f"Ticker: {s}", f"Date: {today.isoformat()}",
             f"Price: ${cp:.2f} ({(cp - h1) / h1 * 100:+.2f}% today)",
             f"50-day average: ${sma50:.2f} | 200-day average: ${sma200:.2f} | 5-day average: ${sma5:.2f}",
             f"RSI(14): {rsi_val:.0f} | Volume vs 10-day average: {rvol:.2f}x",
             f"Dashboard signal: {sig['regime']} (score {sig['score']}/100). {sig['reason']}"]

    daily = fetch_daily_chart_history(s, DATA_SOURCE)
    if daily is not None and len(daily) > 260:
        closes = daily["Close"]
        last = closes.iloc[-1]
        rets = [f"{label} {(last / closes.iloc[-1 - n] - 1) * 100:+.1f}%"
                for label, n in (("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)) if len(closes) > n]
        year = closes.iloc[-252:]
        lines.append("Returns: " + ", ".join(rets))
        lines.append(f"52-week range: ${year.min():.2f} to ${year.max():.2f} "
                     f"({(last / year.max() - 1) * 100:.1f}% from the high)")

    vol_line = f"30-day historical volatility: {hv * 100:.0f}%"
    if atm_iv:
        vol_line += (f" | 30-day at-the-money implied volatility: {atm_iv * 100:.0f}% "
                     f"(IV/HV {atm_iv / hv:.2f}x) | implied 30-day move: ±${cp * atm_iv * math.sqrt(30 / 365):.2f}")
    lines.append(vol_line)

    if next_earnings_date:
        lines.append(f"Next earnings: {next_earnings_date.isoformat()} ({earnings_time}), "
                     f"{(next_earnings_date - today).days} days away")
    else:
        lines.append("Next earnings: none scheduled (ETF or unconfirmed)")

    headlines = fetch_news_headlines(s)
    lines.append("Recent headlines:\n" + ("\n".join(f"- {h}" for h in headlines) if headlines else "- none available"))
    return "\n".join(lines)


@st.cache_data(ttl=AI_CACHE_HOURS * 3600, show_spinner=False)
def generate_ai_view(sym, cache_slot, _context):
    # Cached per ticker and time slot (the context changes every tick, so it is left out of the cache key);
    # errors are raised, so failures are never cached
    import anthropic

    client = anthropic.Anthropic(max_retries=2, timeout=90.0)
    response = client.messages.create(
        model=AI_MODEL,
        max_tokens=4000,
        system=AI_SYSTEM_PROMPT,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": AI_VIEW_SCHEMA}},
        messages=[{"role": "user", "content": _context}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to answer this one.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The answer was cut off. Try Refresh.")
    text = next(b.text for b in response.content if b.type == "text")
    view = json.loads(text)
    view["generated_at"] = datetime.datetime.now(MARKET_TZ).strftime("%b %d, %I:%M %p ET")
    view["model"] = response.model
    return view


def ai_error_message(err):
    try:
        import anthropic
    except ImportError:
        return "The Anthropic SDK isn't installed. Run `pip install anthropic` and restart the app."
    if isinstance(err, anthropic.AuthenticationError):
        return "Your ANTHROPIC_API_KEY was rejected. Check the key in your .env file."
    if isinstance(err, anthropic.PermissionDeniedError):
        return "This API key doesn't have access to the model. Check it in the Claude Console."
    if isinstance(err, anthropic.RateLimitError):
        return "Rate limited by the Claude API. Wait a minute and try again."
    if isinstance(err, anthropic.BadRequestError) and "credit" in str(err).lower():
        return "Your Claude API credit balance is too low. Add credits under Billing in the Claude Console."
    if isinstance(err, anthropic.APIConnectionError):
        return "Couldn't reach the Claude API. Check your internet connection."
    return f"AI view failed: {err}"


def render_ai_view(view):
    def esc(x):
        return html.escape(str(x))

    def pill(outlook):
        tone = {"Bullish": "c-bull", "Bearish": "c-bear"}.get(outlook, "c-neut")
        return f'<span class="ai-pill {tone}">{esc(outlook)}</span>'

    def bullets(items):
        return "".join(f"<li>{esc(i)}</li>" for i in items)

    st.markdown(f"""
        <div class="ai-view">
            <div class="ai-top">{pill(view['sentiment'])}<span class="muted">confidence: {esc(view['confidence'].lower())}</span></div>
            <div class="ai-headline">{esc(view['headline'])}</div>
            <div class="ai-term"><span class="hud-label">Short term · 1–4 weeks</span> {pill(view['short_term']['outlook'])}
                <p>{esc(view['short_term']['summary'])}</p></div>
            <div class="ai-term"><span class="hud-label">Long term · 6–12+ months</span> {pill(view['long_term']['outlook'])}
                <p>{esc(view['long_term']['summary'])}</p></div>
            <div class="ai-lists">
                <div><span class="hud-label">Catalysts</span><ul>{bullets(view['catalysts'])}</ul></div>
                <div><span class="hud-label">Risks</span><ul>{bullets(view['risks'])}</ul></div>
            </div>
            <div class="ai-term"><span class="hud-label">Options angle</span><p>{esc(view['options_angle'])}</p></div>
            <div class="ai-foot">Generated {esc(view['generated_at'])} by {esc(view['model'])} from the data on this card
                and recent headlines. Not financial advice.</div>
        </div>
    """, unsafe_allow_html=True)


# --- Reusable Card Rendering Function ---
def render_ticker_card(s, is_custom=False):
    data = fetch_ticker_data(s, DATA_SOURCE)
    if not data:
        st.error(f"Unable to fetch data for symbol: {s}")
        return
    cp, sma50, sma200, sma5, rsi_val, h1, h2, rvol, hv, chart_df, used_source = data
    sig = compute_signals(data)
    regime, score = sig["regime"], sig["score"]

    next_earnings_date, eps_estimate, earnings_time = fetch_yahoo_upcoming_earnings(s)
    days_to_earnings = (next_earnings_date - today).days if next_earnings_date else None

    earnings_pill_html = ""
    if days_to_earnings is not None and 0 <= days_to_earnings <= 30:
        earnings_pill_html = (f'<span class="earnings-pill" title="Earnings report in {days_to_earnings} days">'
                              f'🔔 {next_earnings_date.strftime("%b %d")}</span>')

    key_prefix = "custom_" if is_custom else f"grid_{s}_"
    score_tip = ("Trend score, 30 to 100. +40 above the 50-day average (else +10), +20 volume over 1.2× normal, "
                 "+20 above the 200-day average, +20 base.")

    # A keyed container really encloses the chart, HUD and expanders; the key drives the border CSS
    with st.container(key=f"qcard-{sig['card_tone']}-{key_prefix}{s}"):
        st.markdown(f"""
            <div style="padding: 20px 20px 0 20px; display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h2 style="margin:0; font-size: 1.8rem; font-weight:800; letter-spacing: -1px; display: flex; align-items: center;">
                        {s} {earnings_pill_html}
                    </h2>
                    <span style="font-family:'JetBrains Mono'; font-size:1.3rem; font-weight:700; color: #fff;">${cp:.2f}</span>
                    <span title="Change since the previous close" style="font-family:'JetBrains Mono'; font-size:0.8rem; font-weight:700; display:block; margin-top:2px; color: {'var(--success)' if cp >= h1 else 'var(--danger)'};">
                        {('▲' if cp >= h1 else '▼')} {abs(cp - h1):.2f} ({((cp - h1) / h1) * 100:.2f}%)
                    </span>
                </div>
                <div style="text-align: right;" title="{score_tip}">
                    <span class="hud-label">SCORE ⓘ</span>
                    <div style="font-family:'JetBrains Mono'; font-size:1.1rem; font-weight:800; color: {'var(--success)' if score > 75 else 'var(--warning)'};">{score}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        chart_range = st.segmented_control(
            "Chart range", CHART_RANGES, default="3M", key=f"{key_prefix}chart_range_{s}",
            label_visibility="collapsed") or "3M"
        intraday = chart_range in INTRADAY_RANGES
        range_df = chart_data_for_range(s, chart_range, DATA_SOURCE)
        avg_note = "today's level" if intraday else ""
        st.markdown(f"""
            <div class="chart-legend">
                <span><i style="background:#ffffff"></i>Price</span>
                <span><i style="background:var(--sma50-color)"></i>50-day avg {avg_note}</span>
                <span><i style="background:var(--sma200-color)"></i>200-day avg {avg_note}</span>
                <span class="muted" style="margin-left:auto">data: {used_source}{' (Massive unavailable)' if DATA_SOURCE == 'Massive' and used_source != 'Massive' else ''}</span>
            </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        if range_df is not None:
            fig.add_trace(go.Scatter(x=range_df.index, y=range_df['Close'], mode='lines', name='Price',
                                     line=dict(color='#ffffff', width=2 if len(range_df) < 400 else 1.2)))
            if intraday:
                # Daily averages don't apply bar by bar intraday, so show where they sit today,
                # but only when nearby, so a distant level doesn't flatten the day's price line
                lo, hi = range_df['Close'].min() * 0.98, range_df['Close'].max() * 1.02
                for level, color in ((sma50, '#3b82f6'), (sma200, '#f59e0b')):
                    if lo <= level <= hi:
                        fig.add_hline(y=level, line=dict(color=color, width=1.2, dash='dot'))
            else:
                fig.add_trace(go.Scatter(x=range_df.index, y=range_df['SMA_50'], mode='lines', name='50-day avg',
                                         line=dict(color='#3b82f6', width=1.5)))
                fig.add_trace(go.Scatter(x=range_df.index, y=range_df['SMA_200'], mode='lines', name='200-day avg',
                                         line=dict(color='#f59e0b', width=1.5)))
        # Hide weekends (and overnight hours intraday) so the line has no flat gaps
        rangebreaks = [dict(bounds=["sat", "mon"])]
        if intraday:
            rangebreaks.append(dict(bounds=[16, 9.5], pattern="hour"))

        fig.update_layout(
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            margin=dict(l=10, r=10, t=10, b=10),
            height=240,
            showlegend=False,
            hovermode='x unified',
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#71717a', size=9),
                       rangebreaks=rangebreaks),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False,
                       tickfont=dict(color='#71717a', size=9), side='right', tickprefix='$')
        )
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False}, key=f"{key_prefix}chart_{s}")

        above_50 = cp > sma50
        rsi_note = " overbought" if rsi_val >= 70 else (" oversold" if rsi_val <= 30 else "")
        three_day = sig["buy_checks"][1][0]
        atm_iv = fetch_atm_iv(s, cp, DATA_SOURCE)
        iv_hv = atm_iv / hv if atm_iv and hv > 0 else None
        expected_move = cp * atm_iv * math.sqrt(30 / 365) if atm_iv else 0.0
        hud_html = "".join([
            hud_item("Trend", "Is the price above or below its 50-day average? Above is an uptrend (Weinstein stage 2), below a downtrend (stage 4).",
                     "Uptrend" if above_50 else "Downtrend", "c-bull" if above_50 else "c-bear"),
            hud_item("Volume vs avg", "Today's volume compared with the 10-day average, adjusted for how much of the session has passed. Above 1.0× means busier than normal.",
                     f"{rvol:.2f}×", "c-bull" if rvol > 1.5 else "c-neut"),
            hud_item("50-day avg", "Average closing price over the last 50 trading days.",
                     f"${sma50:.2f}", style="color:var(--sma50-color)"),
            hud_item("3-day hold", "Has the stock closed above its 50-day average for 3 days in a row? Filters out one-day fake breakouts.",
                     "Confirmed" if three_day else "Not yet", "c-bull" if three_day else "c-neut"),
            hud_item("RSI (14)", "Relative Strength Index, 0 to 100. Above 70 is often called overbought, below 30 oversold.",
                     f"{rsi_val:.0f}{rsi_note}", "c-neut" if rsi_note else ""),
            hud_item("200-day avg", "Average closing price over the last 200 trading days. The long-term trend line.",
                     f"${sma200:.2f}", style="color:var(--sma200-color)"),
            hud_item("Implied vol (30d)", "Volatility the options market is pricing in, from the at-the-money option about 30 days out. "
                     "Higher IV means richer premiums and bigger expected swings.",
                     f"{atm_iv * 100:.0f}%" if atm_iv else "n/a", "c-neut" if iv_hv and iv_hv >= 1.3 else ""),
            hud_item("IV vs hist. vol", "Implied vol divided by the stock's actual 30-day volatility. Above about 1.3× options are "
                     "pricing more movement than usual (rich premiums, often before earnings); below 1.0× they look cheap.",
                     f"{iv_hv:.2f}× ({hv * 100:.0f}% HV)" if iv_hv else f"{hv * 100:.0f}% HV",
                     "c-neut" if iv_hv and iv_hv >= 1.3 else ""),
            hud_item("Expected move (30d)", "One standard deviation move implied by IV over 30 days: roughly a 2-in-3 chance "
                     "the price stays inside this range.",
                     f"±${expected_move:.2f}" if atm_iv else "n/a"),
        ])
        st.markdown(f"""
            <div class="action-banner {sig['b_class']}">{regime}</div>
            <div class="signal-reason">{sig['reason']}</div>
            <div class="hud-section"><div class="hud-grid">{hud_html}</div></div>
        """, unsafe_allow_html=True)

        opt_expander, opt_open = lazy_expander("🎯 OPTIONS: STRIKE FINDER & BACKTEST", f"{key_prefix}opt_exp_{s}")
        with opt_expander:
            # Option chains and the backtest are only fetched while this expander is open
            if opt_open:
                opt_strategy = st.radio(
                    "Strategy", ["Covered Call", "Cash-Secured Put"], horizontal=True, key=f"{key_prefix}strat_{s}",
                    help="Covered call: you own 100 shares and sell a call above the price. "
                         "Cash-secured put: you hold cash and sell a put below the price, agreeing to buy if it falls.")

                days_until_first_friday = (4 - today.weekday()) % 7
                if days_until_first_friday == 0 and datetime.datetime.now().hour >= 16:
                    days_until_first_friday = 7

                first_friday = today + datetime.timedelta(days=days_until_first_friday)
                friday_dates = [first_friday + datetime.timedelta(weeks=i) for i in range(5)]
                friday_options = {f"{f.strftime('%b %d')} ({(f - today).days}d)": (f - today).days for f in friday_dates}

                selected_friday_label = st.select_slider(
                    "Expiration",
                    options=list(friday_options.keys()),
                    key=f"{key_prefix}friday_slider_{s}",
                    help="The next five Friday expirations, with calendar days to expiry (DTE)."
                )

                target_dte = friday_options[selected_friday_label]

                target_delta_val = st.slider(
                    "Target delta", min_value=0, max_value=100, value=20, step=1, format="%d",
                    key=f"{key_prefix}delta_{s}",
                    help="Roughly the chance the option finishes in the money (you get assigned). "
                         "20 ≈ 20% chance: further from the price, less premium, less risk.")
                target_delta = target_delta_val / 100.0

                contract_type = "call" if opt_strategy == "Covered Call" else "put"

                live_opt = None
                if DATA_SOURCE == "Massive":
                    live_opt = fetch_massive_option_by_delta(s, contract_type, target_delta, target_dte, cp)
                live_source = "Massive" if live_opt else "Yahoo"
                if not live_opt:
                    live_opt = fetch_yahoo_option_by_delta(s, contract_type, target_delta, target_dte, cp)

                if live_opt and live_opt["premium"] > 0:
                    rec_strike = live_opt["strike"]
                    est_prem = live_opt["premium"]
                    est_theta = live_opt["theta"]
                    rec_delta = live_opt["delta"]
                    exp_date = live_opt["expiry"]
                    opt_dte = live_opt["dte"]
                    data_source_tag = f"Live {live_source} option chain (bid/ask midpoint)"
                    if DATA_SOURCE == "Massive" and live_source == "Yahoo":
                        data_source_tag += ", Massive chain unavailable"
                else:
                    # Black-Scholes using the stock's 30-day historical volatility in place of implied volatility
                    T = max(target_dte, 1) / 365.0
                    rec_strike = strike_for_delta(cp, target_delta, T, hv, contract_type)
                    est_prem = bs_price(cp, rec_strike, T, hv, contract_type)
                    est_theta = bs_theta_per_day(cp, rec_strike, T, hv, contract_type)
                    rec_delta = target_delta if contract_type == "call" else -target_delta
                    exp_date = today + datetime.timedelta(days=target_dte)
                    opt_dte = target_dte
                    data_source_tag = f"Model estimate (Black-Scholes, {hv * 100:.0f}% hist. vol), no live quote"

                capital_base = rec_strike if opt_strategy == "Cash-Secured Put" else cp
                annual_roi = (est_prem / capital_base) * (365 / max(opt_dte, 1)) * 100
                breakeven = cp - est_prem if contract_type == "call" else rec_strike - est_prem

                st.markdown('<div class="panel-title">Suggested trade</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="trade-sentence">Sell 1 <b>{s} ${rec_strike:.2f} {contract_type}</b> expiring '
                    f'<b>{exp_date.strftime("%b %d")}</b> for about <b>${est_prem * 100:,.0f}</b>. '
                    f'About <b>{abs(rec_delta) * 100:.0f}%</b> chance of assignment.</div>',
                    unsafe_allow_html=True)
                if opt_dte != target_dte:
                    st.caption(f"No listed expiry on your Friday; using the nearest one ({opt_dte} DTE).")

                oc1, oc2, oc3 = st.columns(3)
                oc1.metric("Strike", f"${rec_strike:.2f}", help="The strike whose delta is closest to your target.")
                oc2.metric("Premium", f"${est_prem:.2f}",
                           help=f"Per share. One contract covers 100 shares, so about ${est_prem * 100:,.0f}.")
                oc3.metric("Ann. ROI", f"{annual_roi:.1f}%",
                           help="Premium ÷ capital tied up, scaled to a year, assuming you could repeat this "
                                "trade all year. Capital is the strike for puts, the share price for calls.")
                oc4, oc5, oc6 = st.columns(3)
                oc4.metric("Delta", f"{rec_delta:+.2f}", help="Actual delta of this strike.")
                oc5.metric("Theta", f"-${est_theta:.2f}/d",
                           help="How much the option's value decays per day right now, in your favor as the seller.")
                oc6.metric("Break-even", f"${breakeven:.2f}",
                           help="Covered call: the share price where premium offsets a drop. "
                                "Put: the effective price you'd pay for shares if assigned.")
                st.caption(f"Source: {data_source_tag}")

                # --- IV/HV Ratio from a ~30-Day Option at the Target Delta (captures vol premium and skew) ---
                iv_ratio = 1.0
                ref_opt = None
                if DATA_SOURCE == "Massive":
                    ref_opt = fetch_massive_option_by_delta(s, contract_type, target_delta, 30, cp)
                if not ref_opt:
                    ref_opt = fetch_yahoo_option_by_delta(s, contract_type, target_delta, 30, cp)
                if ref_opt and ref_opt["iv"] > 0 and hv > 0:
                    iv_ratio = min(3.0, max(0.5, ref_opt["iv"] / hv))

                st.markdown('<div class="panel-title">Backtest: the wheel with these settings</div>',
                            unsafe_allow_html=True)
                bc1, bc2, bc3 = st.columns(3)
                use_five_year = bc1.toggle(
                    "5 years (vs 6 months)", value=False, key=f"{key_prefix}five_year_{s}",
                    help="Test over the last 5 years instead of the last 6 months.")
                take_profit = bc2.toggle(
                    "Close at 50% profit", value=False, key=f"{key_prefix}take_profit_{s}",
                    help="Buy the option back once it has lost half its value, then sell the next one.")
                stop_loss = bc3.toggle(
                    "Stop at 2× premium", value=False, key=f"{key_prefix}stop_loss_{s}",
                    help="Buy the option back if it grows to twice the premium you collected, taking the loss "
                         "instead of being assigned. You keep your shares (calls) or cash (puts) and sell a new option.")

                # --- REACTIVE AUTO-EXECUTION BACKTEST (Instant refresh on 5Y toggle, 50% exit, DTE, or Delta) ---
                # First Massive run makes ~2 API calls per trade (cached for a day afterwards)
                with st.spinner("Running backtest..."):
                    metrics, trades_df, daily_df, events_df = run_full_wheel_backtest(
                        s, opt_strategy, target_delta, target_dte, use_five_year, iv_ratio,
                        fetch_earnings_history(s), take_profit, stop_loss, DATA_SOURCE)

                if metrics and trades_df is not None:
                    period_label = "5 years" if use_five_year else "6 months"
                    beat = metrics['total_return_pct'] - metrics['buy_hold_return_pct']
                    st.markdown(
                        f'<div class="trade-sentence">Over the last <b>{period_label}</b>, {metrics["total_trades"]} trades, '
                        f'the wheel <b class="{"c-bull" if beat >= 0 else "c-bear"}">'
                        f'{"beat" if beat >= 0 else "trailed"} buy &amp; hold by {abs(beat):.1f} points</b>.</div>',
                        unsafe_allow_html=True)

                    pnl = metrics['final_value'] - metrics['initial_investment']
                    m1, m2 = st.columns(2)
                    m1.metric("Wheel return", f"{metrics['total_return_pct']:+.1f}%",
                              # Sign goes before the $ so Streamlit colors and arrows the delta correctly
                              f"{'+' if pnl >= 0 else '-'}${abs(pnl):,.0f}",
                              help="Starting with 100 shares' worth of capital.")
                    m2.metric("Buy & hold return", f"{metrics['buy_hold_return_pct']:+.1f}%",
                              help="Simply owning 100 shares for the same period.")
                    m3, m4 = st.columns(2)
                    dd_gap = metrics['max_drawdown_pct'] - metrics['bh_max_drawdown_pct']
                    m3.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.1f}%",
                              f"{dd_gap:+.1f} pts vs buy & hold ({metrics['bh_max_drawdown_pct']:.1f}%)",
                              help="The biggest fall from a previous high along the way. Green means a "
                                   "shallower fall than simply holding the stock, red a deeper one.")
                    m4.metric("Win ratio", f"{metrics['win_ratio']:.0f}%",
                              help="Share of trades that expired worthless or closed at 50% profit. "
                                   "Assignments and stop-outs count as losses. Not the same as making money.")

                    if st.button("📋 View chart & trade log", key=f"{key_prefix}log_btn_{s}", width='stretch'):
                        show_trade_log_dialog(s, trades_df, daily_df, events_df, opt_strategy)

                    vol_note = (f"Option prices modeled at {hv * 100:.0f}% hist. vol × {iv_ratio:.2f} IV/HV ratio"
                                if ref_opt else "Option prices modeled at hist. vol (no live chain for an IV/HV ratio)")
                    if ref_opt and days_to_earnings is not None and 0 <= days_to_earnings <= 30:
                        vol_note += "; today's IV includes the upcoming earnings, so the ratio may run high"
                    if metrics["earnings_trades"]:
                        vol_note += (f". ⚠️ {metrics['earnings_trades']} of {metrics['total_trades']} trades spanned "
                                     f"earnings, where real premiums were likely higher.")
                    if DATA_SOURCE == "Massive":
                        since = metrics["massive_since"]
                        vol_note = (f"{metrics['real_priced_trades']} of {metrics['total_trades']} trades priced from real "
                                    f"Massive option prices"
                                    + (f" (your plan's history starts {since:%b %Y})" if since else "")
                                    + "; the rest use the model. ") + vol_note
                        if metrics["massive_note"]:
                            vol_note += f" Massive stopped part-way: {metrics['massive_note']}."
                    st.caption(vol_note)
                else:
                    st.warning("Not enough price history for this backtest window.")

        with st.expander("🧭 WHY THIS SIGNAL?"):
            rows = "".join(f'<div class="check-row">{"✅" if ok else "⬜"} {text}</div>'
                           for ok, text in sig["buy_checks"])
            exit_rows = "".join(f'<div class="check-row">{"🔴" if ok else "⬜"} {text}</div>'
                                for ok, text in sig["exit_checks"])
            st.markdown(f"""
                <div class="logic-content">
                    <b>Buy</b> needs all four (high conviction if volume is over 1.5×):{rows}
                    <br><b>Exit</b> needs both:{exit_rows}
                </div>
            """, unsafe_allow_html=True)

        with st.expander("📅 EARNINGS"):
            if next_earnings_date:
                e_date_str = next_earnings_date.strftime("%b %d, %Y")
                d_remain = (next_earnings_date - today).days

                earnings_df = pd.DataFrame([{
                    "Event Date": e_date_str,
                    "Timing": earnings_time,
                    "Status": f"In {d_remain} Days" if d_remain >= 0 else "Pending Report",
                    "EPS Est.": eps_estimate
                }])

                st.dataframe(
                    earnings_df,
                    width='stretch',
                    hide_index=True
                )
            else:
                st.markdown(
                    "<span style='font-size:0.75rem; color:var(--text-muted);'>No upcoming earnings scheduled (Index ETF / Unconfirmed).</span>",
                    unsafe_allow_html=True)

        ai_expander, ai_open = lazy_expander("🤖 AI VIEW: SENTIMENT & OUTLOOK", f"{key_prefix}ai_exp_{s}")
        with ai_expander:
            if ai_open:
                if not os.getenv("ANTHROPIC_API_KEY", "").strip():
                    st.info("Add ANTHROPIC_API_KEY to the .env file next to main.py (create the key in the Claude "
                            "Console) and restart the app to turn on AI views.")
                else:
                    # Nothing is sent to Claude until the button is clicked; the view then stays until Refresh
                    ai_key = f"{key_prefix}ai_{s}"
                    view = st.session_state.get(ai_key)
                    clicked = st.button(
                        "↻ Refresh" if view else "✨ Generate AI view", key=f"{ai_key}_btn",
                        help=f"Asks {AI_MODEL} for a read on this stock using the numbers on this card and recent "
                             f"headlines. Each new view is one API call (a few cents). Views are shared between "
                             f"cards and reused for up to {AI_CACHE_HOURS} hours.")
                    if clicked:
                        # A refresh moves to a new cache slot so it really asks again
                        slot = (int(time.time() // (AI_CACHE_HOURS * 3600)), st.session_state.get(f"{ai_key}_n", 0))
                        if view:
                            st.session_state[f"{ai_key}_n"] = slot[1] + 1
                            slot = (slot[0], slot[1] + 1)
                        try:
                            with st.spinner(f"Asking Claude about {s}..."):
                                view = generate_ai_view(s, slot, ai_view_context(
                                    s, data, sig, atm_iv, next_earnings_date, earnings_time))
                            st.session_state[ai_key] = view
                            st.rerun()  # redraw so the button reads Refresh
                        except Exception as err:
                            st.error(ai_error_message(err))
                    if view:
                        render_ai_view(view)


# --- Global Date Reference ---
today = datetime.date.today()

# --- Market Strip: Dow, Nasdaq and the VIX with Today's Change ---
strip_items = []
for idx_sym, idx_name, idx_tip in MARKET_INDICES:
    quote = fetch_index_quote(idx_sym)
    if not quote:
        strip_items.append(f'<span class="m-item" title="{idx_tip}"><span class="m-name">{idx_name}</span> <span class="m-val">n/a</span></span>')
        continue
    last, chg, pct = quote
    # A rising VIX means rising fear, so it is colored the opposite way to the indices
    up_is_good = idx_sym != "^VIX"
    tone = "c-bull" if (chg >= 0) == up_is_good else "c-bear"
    extra = f' <span class="m-note">{vix_regime(last)}</span>' if idx_sym == "^VIX" else ""
    strip_items.append(
        f'<span class="m-item" title="{idx_tip}"><span class="m-name">{idx_name}</span> '
        f'<span class="m-val">{last:,.2f}</span> '
        f'<span class="m-chg {tone}">{"▲" if chg >= 0 else "▼"} {abs(chg):,.2f} ({pct:+.2f}%)</span>{extra}</span>')
st.markdown(f'<div class="market-strip">{"".join(strip_items)}</div>', unsafe_allow_html=True)

# --- Toolbar: Look Up Any Ticker, Filter and Sort the Watchlist ---
t_search, t_filter, t_sort, t_source = st.columns([2, 2, 1, 1])
with t_search:
    custom_sym_input = st.text_input("Look up any ticker", value="", placeholder="e.g. AAPL, SPY, AMD",
                                     key="custom_ticker_input").upper().strip()
with t_filter:
    view_filter = st.segmented_control(
        "Show", ["All", "Buy signals", "Exit signals", "Earnings ≤ 30d"], default="All", key="view_filter")
with t_sort:
    sort_by = st.selectbox("Sort by", ["Watchlist order", "Score (high first)"], key="sort_by")
with t_source:
    st.segmented_control(
        "Data source", ["Yahoo", "Massive"], default="Yahoo", key="data_source",
        help="Yahoo is free. Massive uses your MASSIVE_API_KEY for stock bars, the live option chain, and real "
             "historical option prices in the backtest. Anything Massive can't provide falls back to Yahoo.")

if st.session_state.get("data_source") == "Massive" and not massive_api_key():
    st.warning("Massive is selected but no MASSIVE_API_KEY was found (environment or .env file), so Yahoo is "
               "being used.")

if custom_sym_input:
    c_col1, c_col2, c_col3 = st.columns([1, 2, 1])
    with c_col2:
        render_ticker_card(custom_sym_input, is_custom=True)
    st.markdown("<br>", unsafe_allow_html=True)

# --- Summary Strip: Signal Counts Across the Watchlist ---
watch_signals = {}
for s in tickers:
    data = fetch_ticker_data(s, DATA_SOURCE)
    if data:
        watch_signals[s] = compute_signals(data)

n_buy = sum(1 for v in watch_signals.values() if v["is_buy"])
n_exit = sum(1 for v in watch_signals.values() if v["is_sell"])
earnings_soon = [s for s in tickers if (d := days_until_earnings(s)) is not None and 0 <= d <= 30]
st.markdown(f"""
    <div class="summary-strip">
        <span class="c-bull">● {n_buy} buy</span>
        <span class="c-neut">● {len(watch_signals) - n_buy - n_exit} hold</span>
        <span class="c-bear">● {n_exit} exit</span>
        <span class="muted">🔔 earnings within 30 days: {', '.join(earnings_soon) if earnings_soon else 'none'}</span>
    </div>
""", unsafe_allow_html=True)

# --- Render Ticker Grid (3 Cards Per Row) ---
shown = [s for s in tickers if s in watch_signals]
if view_filter == "Buy signals":
    shown = [s for s in shown if watch_signals[s]["is_buy"]]
elif view_filter == "Exit signals":
    shown = [s for s in shown if watch_signals[s]["is_sell"]]
elif view_filter == "Earnings ≤ 30d":
    shown = [s for s in shown if s in earnings_soon]
if sort_by == "Score (high first)":
    shown.sort(key=lambda s: -watch_signals[s]["score"])

st.markdown('<div class="market-section-title">📊 Watchlist</div>', unsafe_allow_html=True)

if not shown:
    st.info("No watchlist tickers match this filter right now.")

grid_cols = st.columns(3)

for idx, s in enumerate(shown):
    with grid_cols[idx % 3]:
        render_ticker_card(s, is_custom=False)

for s in tickers:
    if s not in watch_signals:
        st.error(f"Unable to fetch data for symbol: {s}")
