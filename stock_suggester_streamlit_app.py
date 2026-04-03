import math
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

    class _DummyCache:
        def __call__(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    class _DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyStreamlit:
        cache_data = _DummyCache()
        session_state: Dict[str, Any] = {}
        sidebar = _DummyContext()

        def set_page_config(self, *args, **kwargs):
            return None

        def columns(self, spec):
            count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
            return [_DummyContext() for _ in range(count)]

        def expander(self, *args, **kwargs):
            return _DummyContext()

        def spinner(self, *args, **kwargs):
            return _DummyContext()

        def stop(self):
            raise SystemExit

        def __getattr__(self, name):
            def method(*args, **kwargs):
                return None
            return method

    st = _DummyStreamlit()


if STREAMLIT_AVAILABLE:
    st.set_page_config(
        page_title="Stock Suggestion Engine",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )


class MarketDataProvider(ABC):
    @abstractmethod
    def get_historical_data(self, symbols: List[str], benchmark: str, period: str, interval: str) -> Dict[str, pd.DataFrame]:
        pass


class YFinanceProvider(MarketDataProvider):
    def get_historical_data(self, symbols: List[str], benchmark: str, period: str, interval: str) -> Dict[str, pd.DataFrame]:
        if yf is None:
            return {}

        all_symbols = list(dict.fromkeys(symbols + [benchmark]))
        raw = yf.download(
            tickers=all_symbols,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        result: Dict[str, pd.DataFrame] = {}
        if raw.empty:
            return result

        if not isinstance(raw.columns, pd.MultiIndex):
            df = raw.copy().dropna(how="all")
            if not df.empty:
                result[all_symbols[0]] = df
            return result

        for symbol in all_symbols:
            if symbol in raw.columns.get_level_values(0):
                df = raw[symbol].copy().dropna(how="all")
                if not df.empty:
                    result[symbol] = df

        return result


class DemoProvider(MarketDataProvider):
    def __init__(self):
        self.seed = 42
        np.random.seed(self.seed)

    def get_historical_data(self, symbols: List[str], benchmark: str, period: str, interval: str) -> Dict[str, pd.DataFrame]:
        # Generate deterministic demo data for symbols and benchmark
        all_symbols = list(dict.fromkeys(symbols + [benchmark]))
        result = {}
        # Generate ~252 trading days of data
        num_days = 252
        dates = pd.date_range(end=pd.Timestamp.now(), periods=num_days, freq='D')

        for symbol in all_symbols:
            # Generate synthetic OHLCV with realistic volatility
            base_price = np.random.uniform(100, 1000)
            prices = [base_price]
            for _ in range(num_days - 1):
                change = np.random.normal(0, 0.015)  # ~1.5% daily volatility
                new_price = max(prices[-1] * (1 + change), 1)  # Prevent negative prices
                prices.append(new_price)

            closes = np.array(prices)
            highs = closes * (1 + np.abs(np.random.normal(0, 0.02, num_days)))
            lows = closes * (1 - np.abs(np.random.normal(0, 0.02, num_days)))
            opens = closes * (1 + np.random.normal(0, 0.005, num_days))
            volumes = np.random.randint(100000, 10000000, num_days)

            df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes
            }, index=dates)
            result[symbol] = df

        return result


def get_provider(provider_type: str) -> MarketDataProvider:
    if provider_type == 'YFINANCE':
        return YFinanceProvider()
    elif provider_type == 'DEMO':
        return DemoProvider()
    else:
        return DemoProvider()  # Safe fallback


@dataclass
class StockSuggestion:
    symbol: str
    direction: str
    score: float
    setup_type: str
    last_close: float
    entry: float
    stop_loss: float
    target: float
    rr: float
    liquidity_score: float
    trend_score: float
    rs_score: float
    volume_score: float
    setup_score: float
    rr_score: float
    notes: str


DEFAULT_SYMBOLS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "INFY.NS",
    "TCS.NS",
    "AXISBANK.NS",
    "LT.NS",
    "BHARTIARTL.NS",
    "TATAMOTORS.NS",
    "KOTAKBANK.NS",
    "SUNPHARMA.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "HINDUNILVR.NS",
]
DEFAULT_BENCHMARK = "^NSEI"
HISTORY_FILE = "stock_suggester_history.csv"
SETTINGS_FILE = "stock_suggester_settings.json"

SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "INFY.NS": "IT",
    "TCS.NS": "IT",
    "AXISBANK.NS": "Banking",
    "LT.NS": "Infrastructure",
    "BHARTIARTL.NS": "Telecom",
    "TATAMOTORS.NS": "Auto",
    "KOTAKBANK.NS": "Banking",
    "SUNPHARMA.NS": "Pharma",
    "BAJFINANCE.NS": "Finance",
    "MARUTI.NS": "Auto",
    "HINDUNILVR.NS": "FMCG",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def indian_number(n: float) -> str:
    if pd.isna(n):
        return "-"
    if abs(n) >= 1e7:
        return f"₹{n / 1e7:.2f} Cr"
    if abs(n) >= 1e5:
        return f"₹{n / 1e5:.2f} L"
    return f"₹{n:,.0f}"


def get_status_text(score: float) -> str:
    if score >= 85:
        return "High conviction"
    if score >= 75:
        return "Trade-ready"
    if score >= 70:
        return "Good candidate"
    return "Watch carefully"


@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(provider_type: str, symbols: Tuple[str, ...], benchmark: str, period: str, interval: str) -> Tuple[Dict[str, pd.DataFrame], str]:
    provider = get_provider(provider_type)
    try:
        data = provider.get_historical_data(list(symbols), benchmark, period, interval)
        if data and all(not df.empty for df in data.values()):
            mode = 'live'
        else:
            # Fallback to demo if live data is empty
            demo_provider = DemoProvider()
            data = demo_provider.get_historical_data(list(symbols), benchmark, period, interval)
            mode = 'demo'
    except Exception as e:
        # Fallback to demo on any error
        demo_provider = DemoProvider()
        data = demo_provider.get_historical_data(list(symbols), benchmark, period, interval)
        mode = 'demo'
    return data, mode


@st.cache_data(ttl=3600, show_spinner=False)
def get_single_symbol_data(provider_type: str, symbol: str, period: str, interval: str) -> pd.DataFrame:
    data, _ = download_price_data(provider_type, (symbol,), symbol, period, interval)
    return data.get(symbol, pd.DataFrame())


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["VOL20"] = out["Volume"].rolling(20).mean()

    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift()).abs()
    low_close = (out["Low"] - out["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    out["HH20"] = out["High"].rolling(20).max()
    out["LL20"] = out["Low"].rolling(20).min()
    out["RET20"] = out["Close"].pct_change(20) * 100.0
    out["RET10"] = out["Close"].pct_change(10) * 100.0
    out["VOL_RATIO"] = out["Volume"] / out["VOL20"]
    return out.dropna().copy()


def score_stock(symbol: str, df: pd.DataFrame, bench_df: pd.DataFrame, direction_mode: str = "Long Only") -> Optional[StockSuggestion]:
    if len(df) < 80 or len(bench_df) < 80:
        return None

    s = enrich(df)
    b = enrich(bench_df)
    if len(s) < 30 or len(b) < 30:
        return None

    row = s.iloc[-1]
    prev = s.iloc[-2]
    bench_row = b.iloc[-1]

    close = safe_float(row["Close"])
    high = safe_float(row["High"])
    low = safe_float(row["Low"])
    ema20 = safe_float(row["EMA20"])
    ema50 = safe_float(row["EMA50"])
    hh20 = safe_float(row["HH20"])
    ll20 = safe_float(row["LL20"])
    atr = safe_float(row["ATR14"])
    vol_ratio = safe_float(row["VOL_RATIO"], default=0.0)
    ret20 = safe_float(row["RET20"], default=0.0)
    bench_ret20 = safe_float(bench_row["RET20"], default=0.0)

    if any(pd.isna(x) for x in [close, high, low, ema20, ema50, hh20, ll20, atr]):
        return None

    traded_value = safe_float(row["VOL20"] * row["Close"], default=0.0)
    liquidity_score = clamp((math.log10(max(traded_value, 1.0)) - 5.7) * 6.0, 0.0, 20.0)

    long_trend = close > ema20 > ema50
    short_trend = close < ema20 < ema50
    ema_gap_pct = abs(ema20 - ema50) / close * 100.0
    trend_strength = clamp((ema_gap_pct / 3.0) * 20.0, 0.0, 20.0)

    rs_value = ret20 - bench_ret20
    rs_long_score = clamp((rs_value + 5.0) * 2.0, 0.0, 20.0)
    rs_short_score = clamp((-rs_value + 5.0) * 2.0, 0.0, 20.0)
    volume_score = clamp(vol_ratio * 7.5, 0.0, 15.0)

    near_breakout = close >= hh20 * 0.97 and close <= hh20 * 1.01
    near_breakdown = close <= ll20 * 1.03 and close >= ll20 * 0.99
    breakout_clean = close > safe_float(prev["High"])
    breakdown_clean = close < safe_float(prev["Low"])

    setup_score_long = 0.0
    if long_trend:
        setup_score_long += 6.0
        if near_breakout:
            setup_score_long += 5.0
        if breakout_clean:
            setup_score_long += 2.0
        if close > safe_float(prev["Close"]):
            setup_score_long += 2.0
    setup_score_long = clamp(setup_score_long, 0.0, 15.0)

    setup_score_short = 0.0
    if short_trend:
        setup_score_short += 6.0
        if near_breakdown:
            setup_score_short += 5.0
        if breakdown_clean:
            setup_score_short += 2.0
        if close < safe_float(prev["Close"]):
            setup_score_short += 2.0
    setup_score_short = clamp(setup_score_short, 0.0, 15.0)

    long_entry = hh20 * 1.002
    long_stop = max(ema20, close - 1.2 * atr)
    if long_stop >= long_entry:
        long_stop = close - 1.1 * atr
    long_risk = long_entry - long_stop
    long_target = long_entry + 2.0 * long_risk
    long_rr = (long_target - long_entry) / long_risk if long_risk > 0 else 0.0
    long_rr_score = clamp(long_rr * 5.0, 0.0, 10.0)

    short_entry = ll20 * 0.998
    short_stop = min(ema20, close + 1.2 * atr)
    if short_stop <= short_entry:
        short_stop = close + 1.1 * atr
    short_risk = short_stop - short_entry
    short_target = short_entry - 2.0 * short_risk
    short_rr = (short_entry - short_target) / short_risk if short_risk > 0 else 0.0
    short_rr_score = clamp(short_rr * 5.0, 0.0, 10.0)

    long_score = liquidity_score + trend_strength + rs_long_score + volume_score + setup_score_long + long_rr_score
    short_score = liquidity_score + trend_strength + rs_short_score + volume_score + setup_score_short + short_rr_score

    pick_long = direction_mode in ("Both", "Long Only")
    pick_short = direction_mode in ("Both", "Short Only")

    notes_long: List[str] = []
    if long_trend:
        notes_long.append("Trend aligned above EMA20/EMA50")
    if rs_value > 0:
        notes_long.append("Outperforming benchmark")
    if vol_ratio >= 1:
        notes_long.append("Volume above 20-bar average")
    if near_breakout:
        notes_long.append("Near 20-bar breakout zone")

    notes_short: List[str] = []
    if short_trend:
        notes_short.append("Trend aligned below EMA20/EMA50")
    if rs_value < 0:
        notes_short.append("Underperforming benchmark")
    if vol_ratio >= 1:
        notes_short.append("Volume above 20-bar average")
    if near_breakdown:
        notes_short.append("Near 20-bar breakdown zone")

    candidates: List[StockSuggestion] = []

    if pick_long and long_risk > 0 and long_score >= 50:
        candidates.append(
            StockSuggestion(
                symbol=symbol,
                direction="LONG",
                score=round(long_score, 1),
                setup_type="Breakout / Trend Continuation",
                last_close=round(close, 2),
                entry=round(long_entry, 2),
                stop_loss=round(long_stop, 2),
                target=round(long_target, 2),
                rr=round(long_rr, 2),
                liquidity_score=round(liquidity_score, 1),
                trend_score=round(trend_strength, 1),
                rs_score=round(rs_long_score, 1),
                volume_score=round(volume_score, 1),
                setup_score=round(setup_score_long, 1),
                rr_score=round(long_rr_score, 1),
                notes=" • ".join(notes_long) if notes_long else "Basic trend setup",
            )
        )

    if pick_short and short_risk > 0 and short_score >= 50:
        candidates.append(
            StockSuggestion(
                symbol=symbol,
                direction="SHORT",
                score=round(short_score, 1),
                setup_type="Breakdown / Weakness Continuation",
                last_close=round(close, 2),
                entry=round(short_entry, 2),
                stop_loss=round(short_stop, 2),
                target=round(short_target, 2),
                rr=round(short_rr, 2),
                liquidity_score=round(liquidity_score, 1),
                trend_score=round(trend_strength, 1),
                rs_score=round(rs_short_score, 1),
                volume_score=round(volume_score, 1),
                setup_score=round(setup_score_short, 1),
                rr_score=round(short_rr_score, 1),
                notes=" • ".join(notes_short) if notes_short else "Basic weakness setup",
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda x: x.score)


def build_suggestions(provider_type: str, symbols: List[str], benchmark: str, period: str, interval: str, direction_mode: str) -> Tuple[List[StockSuggestion], str, Dict[str, pd.DataFrame]]:
    data, mode = download_price_data(provider_type, tuple(symbols), benchmark, period, interval)
    if benchmark not in data:
        return [], mode, data

    bench_df = data[benchmark]
    results: List[StockSuggestion] = []
    for symbol in symbols:
        df = data.get(symbol)
        if df is None or df.empty:
            continue
        suggestion = score_stock(symbol, df, bench_df, direction_mode=direction_mode)
        if suggestion:
            results.append(suggestion)

    results.sort(key=lambda x: x.score, reverse=True)
    return results, mode, data


def build_whatsapp_message(best: StockSuggestion, capital: float = 200000, risk_pct: float = 1.0) -> str:
    risk_per_share = abs(best.entry - best.stop_loss)
    rupee_risk = capital * (risk_pct / 100.0)
    qty = int(rupee_risk // risk_per_share) if risk_per_share > 0 else 0
    lines = [
        "Daily Stock Alert",
        "",
        f"Best Pick: {best.symbol} ({best.direction})",
        f"Setup: {best.setup_type}",
        f"Entry: ₹{best.entry:,.2f}",
        f"Stop Loss: ₹{best.stop_loss:,.2f}",
        f"Target: ₹{best.target:,.2f}",
        f"Reward:Risk: {best.rr:.2f}",
        f"Suggested Qty: {qty}",
        f"Score: {best.score}/100",
        f"Notes: {best.notes}",
    ]
    return "\n".join(lines)


def build_copy_ready_message(best: StockSuggestion) -> str:
    lines = [
        "Today's final stock pick",
        "",
        f"Symbol: {best.symbol}",
        f"Direction: {best.direction}",
        f"Entry: {best.entry}",
        f"Stop Loss: {best.stop_loss}",
        f"Target: {best.target}",
        f"Score: {best.score}/100",
        f"Why: {best.notes}",
    ]
    return "\n".join(lines)


def send_whatsapp_via_twilio(to_number: str, body: str) -> Tuple[bool, str]:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

    if not account_sid or not auth_token:
        return False, "Missing Twilio credentials in environment variables."

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    payload = {
        "From": from_number,
        "To": f"whatsapp:{to_number}",
        "Body": body,
    }

    try:
        response = requests.post(url, data=payload, auth=(account_sid, auth_token), timeout=30)
        if 200 <= response.status_code < 300:
            return True, "WhatsApp alert sent successfully."
        return False, f"Twilio error {response.status_code}: {response.text}"
    except Exception as exc:
        return False, f"Request failed: {exc}"


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_pick_to_history(suggestion: StockSuggestion, capital: float, risk_pct: float) -> None:
    risk_per_share = abs(suggestion.entry - suggestion.stop_loss)
    rupee_risk = capital * (risk_pct / 100.0)
    qty = int(rupee_risk // risk_per_share) if risk_per_share > 0 else 0
    row = {
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        **asdict(suggestion),
        "capital": capital,
        "risk_pct": risk_pct,
        "suggested_qty": qty,
    }
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)


def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            return pd.read_json(SETTINGS_FILE, typ="series").to_dict()
        except Exception:
            return {}
    return {}


def save_settings(payload: dict) -> None:
    pd.Series(payload).to_json(SETTINGS_FILE)


def build_sector_summary(items: List[StockSuggestion]) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append({
            "Sector": SECTOR_MAP.get(item.symbol, "Other"),
            "Score": item.score,
            "Symbol": item.symbol,
        })
    if not rows:
        return pd.DataFrame(columns=["Sector", "Avg Score", "Candidates"])
    df = pd.DataFrame(rows)
    grouped = df.groupby("Sector", as_index=False).agg(**{"Avg Score": ("Score", "mean"), "Candidates": ("Symbol", "count")})
    grouped["Avg Score"] = grouped["Avg Score"].round(1)
    return grouped.sort_values(["Avg Score", "Candidates"], ascending=[False, False])


def backtest_suggestions(items: List[StockSuggestion], symbols_data: Dict[str, pd.DataFrame], lookahead_bars: int = 10) -> pd.DataFrame:
    results = []
    for item in items:
        df = symbols_data.get(item.symbol)
        if df is None or df.empty or len(df) < lookahead_bars + 2:
            continue
        test_df = df.tail(lookahead_bars)
        hit_target = False
        hit_stop = False
        exit_price = safe_float(test_df["Close"].iloc[-1], default=item.entry)
        for _, row in test_df.iterrows():
            row_high = safe_float(row["High"], default=exit_price)
            row_low = safe_float(row["Low"], default=exit_price)
            if item.direction == "LONG":
                if row_low <= item.stop_loss:
                    hit_stop = True
                    exit_price = item.stop_loss
                    break
                if row_high >= item.target:
                    hit_target = True
                    exit_price = item.target
                    break
            else:
                if row_high >= item.stop_loss:
                    hit_stop = True
                    exit_price = item.stop_loss
                    break
                if row_low <= item.target:
                    hit_target = True
                    exit_price = item.target
                    break
        pnl_pct = (((exit_price - item.entry) / item.entry) * 100.0) if item.direction == "LONG" else (((item.entry - exit_price) / item.entry) * 100.0)
        results.append({
            "Symbol": item.symbol,
            "Direction": item.direction,
            "Score": item.score,
            "Outcome": "Target" if hit_target else "Stop" if hit_stop else "Open",
            "Exit Price": round(exit_price, 2),
            "PnL %": round(pnl_pct, 2),
        })
    return pd.DataFrame(results)


def render_console_fallback() -> None:
    print("Stock Suggestion Engine")
    print("=" * 24)
    print("This runtime does not have Streamlit installed, so the web dashboard cannot be launched here.")
    print("The core logic loaded successfully and self-tests passed.")
    if yf is None:
        print("yfinance is also unavailable in this environment, so live market download is disabled.")
    print("Run this app in an environment with these packages installed:")
    print("  pip install streamlit yfinance plotly pandas numpy requests")
    print("Then launch with:")
    print("  streamlit run stock_suggester_streamlit_app.py")


def run_self_tests() -> None:
    demo = StockSuggestion(
        symbol="TEST.NS",
        direction="LONG",
        score=88.5,
        setup_type="Breakout / Trend Continuation",
        last_close=123.45,
        entry=125.0,
        stop_loss=120.0,
        target=135.0,
        rr=2.0,
        liquidity_score=18.0,
        trend_score=16.0,
        rs_score=17.0,
        volume_score=12.0,
        setup_score=14.0,
        rr_score=10.0,
        notes="Trend aligned above EMA20/EMA50",
    )
    msg = build_whatsapp_message(demo, capital=200000, risk_pct=1.0)
    assert "Daily Stock Alert" in msg
    assert "Best Pick: TEST.NS (LONG)" in msg
    assert "Suggested Qty: 400" in msg
    assert "Reward:Risk: 2.00" in msg
    copy_msg = build_copy_ready_message(demo)
    assert "Today's final stock pick" in copy_msg
    assert "Symbol: TEST.NS" in copy_msg
    assert clamp(25, 0, 20) == 20
    assert clamp(-1, 0, 20) == 0
    assert safe_float("3.5") == 3.5
    assert np.isnan(safe_float(None))
    assert indian_number(150000) == "₹1.50 L"
    assert indian_number(25000000) == "₹2.50 Cr"
    assert get_status_text(90) == "High conviction"
    assert get_status_text(76) == "Trade-ready"
    assert get_status_text(71) == "Good candidate"
    assert get_status_text(65) == "Watch carefully"
    assert not build_sector_summary([demo]).empty


run_self_tests()


if not STREAMLIT_AVAILABLE:
    render_console_fallback()
else:
    st.title("📈 Smart Stock Suggestion Engine")
    st.caption("Scans a fixed liquid-stock universe, ranks trade-ready setups, and highlights the best candidate with entry, stop loss, and target.")

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .hero-card {
            padding: 1.1rem 1.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(23,37,84,0.95), rgba(30,64,175,0.90));
            color: white;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .status-card {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(148,163,184,0.3);
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
            font-size: 0.9rem;
            color: #374151;
        }
        .pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            font-size: 0.85rem;
            margin-top: 0.35rem;
        }
        .rank-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.8rem;
        }
        .mini-title {
            font-size: 0.82rem;
            color: #475569;
            margin-bottom: 0.2rem;
        }
        .value-lg {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
        }
        .badge-long {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(22,163,74,0.12);
            color: #166534;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-short {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(220,38,38,0.12);
            color: #991b1b;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .score-chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: rgba(30,64,175,0.1);
            color: #1d4ed8;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .mini-chart-card {
            padding: 0.75rem 0.9rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(148,163,184,0.20);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    saved_settings = load_settings()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Open page", ["Dashboard", "History", "Backtest", "Settings", "WhatsApp Alerts"], index=0)

        st.header("Scanner Settings")
        preset_options = ["Focused", "Balanced", "Aggressive"]
        preset_default = saved_settings.get("preset", "Balanced")
        preset = st.selectbox("Dashboard preset", preset_options, index=preset_options.index(preset_default) if preset_default in preset_options else 1)
        symbols_text = st.text_area("Symbols (comma-separated Yahoo Finance tickers)", value=saved_settings.get("symbols_text", ", ".join(DEFAULT_SYMBOLS)), height=180)
        benchmark = st.text_input("Benchmark", value=saved_settings.get("benchmark", DEFAULT_BENCHMARK))
        period_options = ["6mo", "1y", "2y"]
        period_default = saved_settings.get("period", "1y")
        period = st.selectbox("Lookback period", period_options, index=period_options.index(period_default) if period_default in period_options else 1)
        interval_options = ["1d", "1h"]
        interval_default = saved_settings.get("interval", "1d")
        interval = st.selectbox("Interval", interval_options, index=interval_options.index(interval_default) if interval_default in interval_options else 0)
        direction_options = ["Long Only", "Short Only", "Both"]
        direction_default = saved_settings.get("direction_mode", "Long Only")
        direction_mode = st.selectbox("Direction", direction_options, index=direction_options.index(direction_default) if direction_default in direction_options else 0)
        provider_options = ["YFINANCE", "DEMO"]
        provider_default = saved_settings.get("provider", "YFINANCE")
        provider = st.selectbox("Data Provider", provider_options, index=provider_options.index(provider_default) if provider_default in provider_options else 0)
        top_n = st.slider("Show top candidates", min_value=1, max_value=10, value=int(saved_settings.get("top_n", 5)))
        show_score_chart = st.toggle("Show score chart", value=bool(saved_settings.get("show_score_chart", True)))
        show_breakdown_table = st.toggle("Show breakdown table", value=bool(saved_settings.get("show_breakdown_table", True)))
        compact_table = st.toggle("Compact ranked table", value=bool(saved_settings.get("compact_table", False)))
        card_view = st.toggle("Card view for ranked stocks", value=bool(saved_settings.get("card_view", True)))
        show_mini_charts = st.toggle("Show mini charts for top picks", value=bool(saved_settings.get("show_mini_charts", True)))
        auto_refresh = st.toggle("Auto refresh every 60s", value=bool(saved_settings.get("auto_refresh", False)))
        refresh = st.button("Run Scanner", type="primary")
        if st.button("Save current settings"):
            save_settings({
                "preset": preset,
                "symbols_text": symbols_text,
                "benchmark": benchmark,
                "period": period,
                "interval": interval,
                "direction_mode": direction_mode,
                "provider": provider,
                "top_n": top_n,
                "show_score_chart": show_score_chart,
                "show_breakdown_table": show_breakdown_table,
                "compact_table": compact_table,
                "card_view": card_view,
                "show_mini_charts": show_mini_charts,
                "whatsapp_number": saved_settings.get("whatsapp_number", "+919999999999"),
                "auto_refresh": auto_refresh,
            })
            st.success("Settings saved locally.")

    preset_threshold = {"Focused": 75, "Balanced": 60, "Aggressive": 50}[preset]
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]

    if refresh or "suggestions" not in st.session_state:
        with st.spinner("Scanning market and ranking stocks..."):
            all_results, data_mode, data = build_suggestions(provider, symbols, benchmark, period, interval, direction_mode)
            st.session_state["suggestions"] = [s for s in all_results if s.score >= preset_threshold] or all_results
            st.session_state["data_mode"] = data_mode
            st.session_state["data_info"] = {
                "provider": provider,
                "mode": data_mode,
                "requested_symbols": len(symbols),
                "available_data": len(data),
                "benchmark_available": benchmark in data
            }

    suggestions: List[StockSuggestion] = st.session_state.get("suggestions", [])
    if not suggestions:
        st.warning("No trade-ready stocks found for the current rules. Try switching interval, direction, or stock universe.")
        st.stop()

    history_df = load_history()
    best = suggestions[0]
    top_results = suggestions[:top_n]
    status_text = get_status_text(best.score)
    data_info = st.session_state.get("data_info", {})

    if auto_refresh:
        st.caption("Auto refresh is enabled. Refreshing in Streamlit normally requires an additional helper package or browser refresh.")

    if page == "History":
        st.subheader("Saved Picks History")
        if history_df.empty:
            st.info("No saved picks yet. Save the final pick from the Dashboard page.")
        else:
            ordered = history_df.sort_values("saved_at", ascending=False)
            st.dataframe(ordered, use_container_width=True, hide_index=True)
            if px is not None and {"saved_at", "score", "symbol"}.issubset(history_df.columns):
                hist_chart = history_df.copy()
                hist_chart["saved_at"] = pd.to_datetime(hist_chart["saved_at"], errors="coerce")
                hist_chart = hist_chart.dropna(subset=["saved_at"])
                if not hist_chart.empty:
                    fig_hist = px.line(hist_chart.sort_values("saved_at"), x="saved_at", y="score", color="symbol", title="Saved Pick Scores Over Time")
                    fig_hist.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_hist, use_container_width=True)
        st.stop()

    if page == "Backtest":
        st.subheader("Backtest Snapshot")
        bt_data, _ = download_price_data(provider, tuple([s.symbol for s in top_results]), benchmark, period, interval)
        bt_df = backtest_suggestions(top_results, bt_data, lookahead_bars=10)
        if bt_df.empty:
            st.info("Not enough data to generate a quick backtest snapshot yet.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Trades Tested", len(bt_df))
            with c2:
                win_rate = (bt_df["Outcome"].eq("Target").mean() * 100.0) if len(bt_df) else 0.0
                st.metric("Target Hit %", f"{win_rate:.1f}%")
            with c3:
                st.metric("Avg PnL %", f"{bt_df['PnL %'].mean():.2f}%")
            st.dataframe(bt_df, use_container_width=True, hide_index=True)
            if px is not None:
                fig_bt = px.bar(bt_df, x="Symbol", y="PnL %", color="Outcome", title="Quick Backtest PnL %")
                fig_bt.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_bt, use_container_width=True)
        st.stop()

    if page == "Settings":
        st.subheader("Settings Overview")
        current_settings = {
            "preset": preset,
            "benchmark": benchmark,
            "period": period,
            "interval": interval,
            "direction_mode": direction_mode,
            "provider": provider,
            "top_n": top_n,
            "show_score_chart": show_score_chart,
            "show_breakdown_table": show_breakdown_table,
            "compact_table": compact_table,
            "card_view": card_view,
            "show_mini_charts": show_mini_charts,
            "auto_refresh": auto_refresh,
            "symbols_count": len(symbols),
            "status_text": status_text,
        }
        st.json(current_settings)
        st.info("Use the sidebar to change values, then click 'Save current settings'.")
        st.stop()

    if page == "WhatsApp Alerts":
        st.subheader("WhatsApp Alerts")
        capital_preview = 200000
        risk_preview = 1.0
        whatsapp_message = build_whatsapp_message(best, capital=capital_preview, risk_pct=risk_preview)
        wa1, wa2 = st.columns([1.2, 1])
        with wa1:
            st.text_area("Preview message", value=whatsapp_message, height=240)
        with wa2:
            whatsapp_number = st.text_input("WhatsApp number in E.164 format", value=saved_settings.get("whatsapp_number", "+919999999999"))
            st.caption("Example: +919876543210")
            if st.button("Send test WhatsApp alert"):
                ok, msg = send_whatsapp_via_twilio(whatsapp_number, whatsapp_message)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            if st.button("Save WhatsApp number"):
                merged = load_settings()
                merged["whatsapp_number"] = whatsapp_number
                save_settings(merged)
                st.success("WhatsApp number saved locally.")
        st.stop()

    st.markdown(
        f"""
        <div class="hero-card">
            <div style="font-size:0.95rem; opacity:0.9;">🏆 Best Pick Right Now</div>
            <div style="font-size:2rem; font-weight:700; margin-top:0.25rem;">{best.symbol} — {best.direction}</div>
            <div class="pill">{best.setup_type}</div>
            <div style="margin-top:0.9rem; font-size:1rem; line-height:1.6;">{best.notes}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col1:
        st.metric("Score", f"{best.score}/100")
    with col2:
        st.metric("Last Close", f"₹{best.last_close:,.2f}")
    with col3:
        st.metric("Entry", f"₹{best.entry:,.2f}")
    with col4:
        st.metric("Stop Loss", f"₹{best.stop_loss:,.2f}")
    with col5:
        st.metric("Target", f"₹{best.target:,.2f}")

    st.markdown(
        f"""
        <div class="status-card">
            <strong>Data Provider Status:</strong><br>
            Provider: {data_info.get('provider', 'Unknown')} | Mode: {data_info.get('mode', 'unknown').title()}<br>
            Requested Symbols: {data_info.get('requested_symbols', 0)} | Data Available: {data_info.get('available_data', 0)} | Benchmark: {'Yes' if data_info.get('benchmark_available') else 'No'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("### Quick Read")
        st.markdown(f"""
- **Direction:** {best.direction}
- **Reward : Risk:** {best.rr:.2f}
- **Setup:** {best.setup_type}
- **Screening preset:** {preset}
        """)
    with col_b:
        st.markdown("### Decision Hint")
        if best.score >= 85:
            st.success("High-conviction candidate. Best suited when price is still near entry.")
        elif best.score >= 70:
            st.info("Good candidate. Check that price has not already moved too far from entry.")
        else:
            st.warning("Moderate setup. Better used as a watch candidate than an immediate trade.")

    st.divider()
    st.subheader("Sector Summary")
    sector_df = build_sector_summary(top_results)
    if not sector_df.empty:
        sec1, sec2 = st.columns([1, 1.2])
        with sec1:
            st.dataframe(sector_df, use_container_width=True, hide_index=True)
        with sec2:
            if px is not None:
                fig_sector = px.bar(sector_df, x="Sector", y="Avg Score", text="Candidates", title="Sector Strength by Average Score")
                fig_sector.update_layout(height=320, margin=dict(l=20, r=20, t=45, b=20))
                st.plotly_chart(fig_sector, use_container_width=True)

    st.divider()
    st.subheader("Top Suggested Stocks")
    summary_df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Symbol": s.symbol,
            "Direction": s.direction,
            "Score": s.score,
            "Setup": s.setup_type,
            "Last Close": s.last_close,
            "Entry": s.entry,
            "Stop Loss": s.stop_loss,
            "Target": s.target,
            "RR": s.rr,
            "Notes": s.notes,
        }
        for i, s in enumerate(top_results)
    ])

    if card_view:
        for i, s in enumerate(top_results, start=1):
            badge_class = "badge-long" if s.direction == "LONG" else "badge-short"
            st.markdown(
                f"""
                <div class="rank-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
                        <div>
                            <div class="mini-title">Rank #{i}</div>
                            <div class="value-lg">{s.symbol}</div>
                            <div style="margin-top:0.3rem;">
                                <span class="{badge_class}">{s.direction}</span>
                                <span class="score-chip">Score {s.score}/100</span>
                            </div>
                            <div style="margin-top:0.55rem; color:#334155;">{s.setup_type}</div>
                            <div style="margin-top:0.45rem; color:#475569; line-height:1.5;">{s.notes}</div>
                        </div>
                        <div style="min-width:240px; text-align:left; color:#0f172a;">
                            <div><strong>Entry:</strong> ₹{s.entry:,.2f}</div>
                            <div><strong>SL:</strong> ₹{s.stop_loss:,.2f}</div>
                            <div><strong>Target:</strong> ₹{s.target:,.2f}</div>
                            <div><strong>RR:</strong> {s.rr:.2f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        if compact_table:
            st.dataframe(summary_df[["Rank", "Symbol", "Direction", "Score", "Entry", "Stop Loss", "Target", "RR"]], use_container_width=True, hide_index=True)
        else:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if show_score_chart and px is not None:
        chart_df = summary_df[["Symbol", "Score", "Direction"]].copy()
        fig = px.bar(chart_df, x="Symbol", y="Score", color="Direction", text="Score", title="Top Candidate Scores")
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    if show_mini_charts and go is not None:
        st.subheader("Mini Charts — Top 3 Picks")
        mini_cols = st.columns(min(3, len(top_results)))
        for idx, s in enumerate(top_results[:3]):
            with mini_cols[idx]:
                mini_df = get_single_symbol_data(provider, s.symbol, period, interval)
                st.markdown(f"<div class='mini-chart-card'><strong>{s.symbol}</strong> <span class='score-chip'>Score {s.score}/100</span></div>", unsafe_allow_html=True)
                if mini_df is not None and not mini_df.empty:
                    mini_source = mini_df.tail(30).reset_index()
                    mini_time_col = mini_source.columns[0]
                    mini_fig = go.Figure(data=[go.Candlestick(x=mini_source[mini_time_col], open=mini_source["Open"], high=mini_source["High"], low=mini_source["Low"], close=mini_source["Close"], name=s.symbol)])
                    mini_fig.add_hline(y=s.entry, line_dash="dash", line_width=1)
                    mini_fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False, showlegend=False)
                    st.plotly_chart(mini_fig, use_container_width=True)
                else:
                    st.caption("No price data available.")

    st.subheader("Final Pick Decision Card")
    price_df = get_single_symbol_data(provider, best.symbol, period, interval)
    if price_df is not None and not price_df.empty and go is not None:
        chart_source = price_df.tail(60).reset_index()
        time_col = chart_source.columns[0]
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(x=chart_source[time_col], open=chart_source["Open"], high=chart_source["High"], low=chart_source["Low"], close=chart_source["Close"], name=best.symbol))
        fig_price.add_trace(go.Bar(x=chart_source[time_col], y=chart_source["Volume"], name="Volume", opacity=0.22, yaxis="y2"))
        fig_price.add_hrect(y0=min(best.entry, best.target), y1=max(best.entry, best.target), fillcolor="rgba(34,197,94,0.12)", line_width=0)
        fig_price.add_hrect(y0=min(best.entry, best.stop_loss), y1=max(best.entry, best.stop_loss), fillcolor="rgba(239,68,68,0.10)", line_width=0)
        fig_price.add_hline(y=best.entry, line_dash="dash", line_width=2, annotation_text="Entry")
        fig_price.add_hline(y=best.stop_loss, line_dash="dot", line_width=2, annotation_text="Stop Loss")
        fig_price.add_hline(y=best.target, line_dash="dot", line_width=2, annotation_text="Target")
        fig_price.add_trace(go.Scatter(x=[chart_source[time_col].iloc[-1]], y=[best.entry], mode="markers+text", text=["BUY" if best.direction == "LONG" else "SELL"], textposition="top center", marker=dict(size=12, symbol="diamond"), name="Signal"))
        fig_price.update_layout(title=f"Candlestick Chart — {best.symbol}", height=500, margin=dict(l=20, r=20, t=50, b=20), xaxis_rangeslider_visible=False, yaxis=dict(title="Price"), yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, rangemode="tozero"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), bargap=0.05)
        st.plotly_chart(fig_price, use_container_width=True)
    elif px is not None and price_df is not None and not price_df.empty:
        chart_source = price_df.tail(60).reset_index()
        time_col = chart_source.columns[0]
        fig_price = px.line(chart_source, x=time_col, y="Close", title=f"Recent Price View — {best.symbol}")
        fig_price.update_layout(height=320, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(fig_price, use_container_width=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f"""
### {best.symbol}
**Direction:** {best.direction}  
**Entry:** ₹{best.entry:,.2f}  
**Stop Loss:** ₹{best.stop_loss:,.2f}  
**Target:** ₹{best.target:,.2f}  
**Setup:** {best.setup_type}  
**Why selected:** {best.notes}
        """)
    with right:
        risk_per_share = abs(best.entry - best.stop_loss)
        reward_per_share = abs(best.target - best.entry)
        st.metric("Risk / share", f"₹{risk_per_share:,.2f}")
        st.metric("Reward / share", f"₹{reward_per_share:,.2f}")
        capital = st.number_input("Capital for sizing", min_value=10000, value=200000, step=10000)
        risk_pct = st.slider("Risk per trade (%)", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
        rupee_risk = capital * (risk_pct / 100.0)
        qty = int(rupee_risk // risk_per_share) if risk_per_share > 0 else 0
        st.metric("Suggested quantity", max(qty, 0))
        if st.button("Save final pick to history"):
            save_pick_to_history(best, capital, risk_pct)
            st.success("Final pick saved to history.")

    st.divider()
    st.subheader("Why This Stock Ranked High")
    breakdown_df = pd.DataFrame({
        "Factor": ["Liquidity", "Trend", "Relative Strength", "Volume", "Setup Quality", "Risk/Reward"],
        "Score": [best.liquidity_score, best.trend_score, best.rs_score, best.volume_score, best.setup_score, best.rr_score],
        "Max": [20, 20, 20, 15, 15, 10],
    })
    st.bar_chart(breakdown_df.set_index("Factor")["Score"])
    if show_breakdown_table:
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.subheader("Action Plan")
    st.info("Use this app as a decision-support scanner. It narrows the market to a few trade-ready candidates. " f"Current dashboard status: {status_text}. Final execution should still respect your capital, order type, and risk rules.")

    with st.expander("Copy-ready final stock message", expanded=False):
        st.code(build_copy_ready_message(best), language="text")
