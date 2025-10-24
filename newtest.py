# backtest_hybrid_momentum_intraday_pyramiding.py
import os
import math
import time
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# =========================
# Konfiguration / Defaults
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # mehrere Symbole
PRIMARY_INTERVAL = Client.KLINE_INTERVAL_8HOUR
INTRADAY_INTERVAL = Client.KLINE_INTERVAL_1HOUR

# Backtest Zeitfenster (UTC)
END_DAYS_AGO = 0        # 0 = bis jetzt
START_DAYS_AGO = 120    # 120 Tage zurück (beispielhaft)
# Du kannst alternativ start/end als ISO setzen:
# START_AT = "2024-06-01 00:00:00"
# END_AT   = "2024-10-01 00:00:00"

# Portfolio
START_EQUITY = 10000.0
MAX_PORTFOLIO_EXPOSURE = 0.9  # maximal 90% des Portfolios investiert

# Gebühren/Marktausführung
TAKER_FEE = 0.001  # 0.1%
MAKER_FEE = 0.0007 # 0.07%
HALF_SPREAD_PCT = 0.0002  # 2 bps je Seite -> 4 bps Round-trip
MAX_SLIPPAGE_PCT = 0.001   # Kappe Slippage auf 10 bps
VOL_CAP_FRAC = 0.02        # max 2% des 8h-Dollarvolumens pro Fill

# Strategie: Hybrid Momentum + Intraday Entries + Pyramiding
MOM_WINDOW = 20            # 8h-Momentum-Fenster (20*8h ~ 160h)
MOM_THRESH = 0.04          # +4% Momentum Bedingung (Hybrid-Trendfilter)
EMA_TREND = 200            # 200-EMA auf 8h als Trendfilter
ATR_WINDOW = 14            # ATR für Stop/Risikogröße
RISK_PER_TRADE = 0.01      # 1% vom Portfolio pro initialem Trade riskieren
PYRAMID_MAX_LAYERS = 3     # inkl. Erstposition
PYRAMID_STEP_PCT = 0.02    # neue Layer alle +2% vom letzten Entry
TRAIL_MULT_ATR = 2.5       # Trail-Stop (ATR-Multiplikator) nach erstem Lift
BREAKEVEN_AFTER_PCT = 0.015 # ab +1.5% ab Entry SL auf BE ziehen

# Intraday Entry-Filter (1h) NUR wenn 8h-Bedingungen erfüllt:
INTRADAY_RSI_W = 14
INTRADAY_RSI_BUY = 55      # RSI kreuzt > 55 als Trigger (prozyklisch)
INTRADAY_CONFIRM_WIN = 6   # betrachte die letzten 6 Stunden für Trigger

# Walk-Forward
DO_WALK_FORWARD = False
WF_TRAIN_DAYS = 60
WF_TEST_DAYS = 30

PLOT = True

# ================
# Hilfsdatenklassen
# ================
@dataclass
class PositionLayer:
    entry_price: float
    qty: float

@dataclass
class PositionState:
    layers: List[PositionLayer]
    stop_price: float
    last_add_price: float

# =========================
# API / Datenbeschaffung
# =========================
def get_time_range():
    now = datetime.now(timezone.utc)
    end = now - timedelta(days=END_DAYS_AGO)
    start = now - timedelta(days=START_DAYS_AGO)
    return start, end

def load_klines(client: Client, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Lädt lückenlos Klines im Intervall [start, end] (UTC)."""
    out = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while True:
        try:
            kl = client.get_klines(symbol=symbol, interval=interval, startTime=start_ms, endTime=end_ms, limit=1000)
        except BinanceAPIException as e:
            logging.error(f"{symbol} get_klines error: {e}")
            time.sleep(1)
            continue
        if not kl:
            break
        out.extend(kl)
        last_open = kl[-1][0]
        # Schritt weiter
        start_ms = last_open + 1
        # API rate friendly
        time.sleep(0.05)
        if start_ms >= end_ms:
            break

    if not out:
        raise RuntimeError(f"Keine Klines für {symbol} {interval}")

    df = pd.DataFrame(out, columns=["open_time","open","high","low","close","volume","close_time",
                                    "qav","num_trades","taker_base","taker_quote","ignore"])
    df["open"]   = df["open"].astype(float)
    df["high"]   = df["high"].astype(float)
    df["low"]    = df["low"].astype(float)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    # Dollar-Volumen näherungsweise:
    df["dollar_vol"] = df["volume"] * df["close"]
    return df

# =========================
# Indikatoren / Feature-Bau
# =========================
def add_primary_features(df: pd.DataFrame) -> pd.DataFrame:
    df["mom"] = df["close"].pct_change(MOM_WINDOW)  # 8h Momentum
    df["ema200"] = EMAIndicator(df["close"], window=EMA_TREND).ema_indicator()
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW)
    df["atr"] = atr.average_true_range()
    # einfache Datenqualitäts-Checks
    df = df.dropna().copy()
    return df

def intraday_trigger(df_1h: pd.DataFrame, t: pd.Timestamp) -> bool:
    """Einfacher Intraday-Trigger: In letzter INTRADAY_CONFIRM_WIN Stunden kreuzt RSI von <50 auf >INTRADAY_RSI_BUY."""
    sub = df_1h.loc[:t].tail(INTRADAY_CONFIRM_WIN + 1).copy()
    if len(sub) < INTRADAY_CONFIRM_WIN + 1:
        return False
    rsi = RSIIndicator(sub["close"], window=INTRADAY_RSI_W).rsi()
    if rsi.isna().any():
        return False
    # Kreuzungskriterium
    prev = rsi.iloc[-2]
    cur  = rsi.iloc[-1]
    return (prev < 50) and (cur > INTRADAY_RSI_BUY)

# =========================
# Ausführung / Ausführungsmodell
# =========================
def execution_price(side: str, mid: float, qty_usd: float, window_dollar_vol: float) -> float:
    """
    Simulierter Ausführungspreis:
      mid +/- half_spread +/- slippage
    Slippage ~ min(MAX_SLIPPAGE_PCT, qty_usd / (0.1 * window_dollar_vol))
    """
    half_spread = HALF_SPREAD_PCT
    # Liquiditätsbasierte Slippage
    denom = max(1e-9, 0.1 * window_dollar_vol)
    slip = min(MAX_SLIPPAGE_PCT, qty_usd / denom)
    if side == "buy":
        px = mid * (1 + half_spread + slip)
    else:
        px = mid * (1 - half_spread - slip)
    return px

def fee_rate(is_maker: bool) -> float:
    return MAKER_FEE if is_maker else TAKER_FEE

def cap_by_volume(desired_qty_usd: float, window_dollar_vol: float) -> float:
    cap = VOL_CAP_FRAC * window_dollar_vol
    return float(min(desired_qty_usd, cap))

# =========================
# Risiko / Positionslogik
# =========================
def initial_position_size(equity: float, stop_dist: float, price: float) -> float:
    """
    Positionsgröße in USD aus RISK_PER_TRADE und Stop-Distanz (in $).
    Riskiere R% vom Equity; Menge so, dass Verlust bis Stop = R% * Equity.
    """
    risk_usd = RISK_PER_TRADE * equity
    if stop_dist <= 0:
        return 0.0
    qty = risk_usd / stop_dist
    usd = qty * price
    return max(0.0, usd)

def update_trailing_stop(pos: PositionState, atr_val: float, last_price: float):
    # Trail auf max(pos.stop, last_price - k*ATR)
    trail = last_price - TRAIL_MULT_ATR * atr_val
    pos.stop_price = max(pos.stop_price, trail)

# =========================
# Metriken / Auswertung
# =========================
def compute_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    ret = equity_curve.pct_change().fillna(0.0)
    # 8h Bars -> pro Jahr ~ 3 Bars/Tag * 365 = 1095
    ann_factor = 1095
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (ann_factor/len(ret)) - 1 if len(ret) > 0 else 0
    vol = ret.std() * np.sqrt(ann_factor)
    sharpe = (ret.mean() * ann_factor) / vol if vol > 1e-9 else 0.0
    neg = ret[ret < 0]
    dd = (equity_curve / equity_curve.cummax() - 1).min()
    sortino = (ret.mean() * ann_factor) / (neg.std() * np.sqrt(ann_factor)) if len(neg)>0 and neg.std()>1e-9 else 0.0
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": dd,
    }

# =========================
# Backtest Kern
# =========================
def backtest_symbol(
    df_8h: pd.DataFrame,
    df_1h: pd.DataFrame,
    equity_alloc: float
) -> Tuple[pd.Series, Dict[str, float], pd.DataFrame]:
    """
    Backtest je Symbol mit:
     - Hybrid Momentum (8h momentum + über EMA200)
     - Intraday 1h Trigger
     - ATR-Stop, Trail, Breakeven
     - Pyramiding (max Layers, Step%), Risiko-gedeckelt
    """
    df8 = add_primary_features(df_8h.copy())
    # wir laufen über 8h-Zeitachse
    equity = equity_alloc
    cash = equity_alloc
    position: PositionState = None  # None oder PositionState
    max_layers = PYRAMID_MAX_LAYERS

    records = []
    equity_curve = []

    for t, row in df8.iterrows():
        price_mid = row["close"]
        dollar_vol = max(1e-9, row["dollar_vol"])
        atr_val = row["atr"]

        # aktualisiere equity aus Marktwert
        mkt_value = 0.0
        if position:
            total_qty = sum(l.qty for l in position.layers)
            mkt_value = total_qty * price_mid
        equity_now = cash + mkt_value
        equity_curve.append((t, equity_now))

        # Exit-Logik (Stops / Trail)
        if position:
            # Stop/Trail prüfen
            if price_mid <= position.stop_price:
                # simulierte Sell-Exec
                total_qty = sum(l.qty for l in position.layers)
                notional = total_qty * price_mid
                notional = cap_by_volume(notional, dollar_vol)
                px = execution_price("sell", price_mid, notional, dollar_vol)
                fee = fee_rate(False)  # konservativ: Taker
                proceeds = notional * (1 - fee) * (px / price_mid)  # px anstelle von mid
                cash += proceeds
                records.append({"time": t, "side": "SELL_STOP", "price": px, "qty": total_qty, "equity": cash})
                position = None
                # nach Exit Equity neu setzen
                equity_now = cash
                equity_curve[-1] = (t, equity_now)
                continue
            else:
                # Trailing Stop ggf. nachziehen
                update_trailing_stop(position, atr_val, price_mid)
                # Break-even Shift
                be_trigger = position.layers[0].entry_price * (1 + BREAKEVEN_AFTER_PCT)
                if price_mid > be_trigger and position.stop_price < position.layers[0].entry_price:
                    position.stop_price = position.layers[0].entry_price

        # Entry-Logik
        in_trend = (row["close"] > row["ema200"]) and (row["mom"] >= MOM_THRESH)
        if in_trend:
            # Intraday-Trigger prüfen
            if intraday_trigger(df_1h, t):
                # initiale Positionsgröße (USD)
                stop_dist = ATR_WINDOW and atr_val * 2.0 or (row["close"] * 0.02)
                init_usd = initial_position_size(equity_now, stop_dist, row["close"])
                # Portfolio-Exposure-Cap
                portfolio_exposure_after = (equity_now - cash + init_usd) / equity_now
                if portfolio_exposure_after > MAX_PORTFOLIO_EXPOSURE:
                    init_usd = max(0.0, MAX_PORTFOLIO_EXPOSURE * equity_now - (equity_now - cash))
                init_usd = cap_by_volume(init_usd, dollar_vol)

                if init_usd > 5.0 and cash >= init_usd:
                    # Buy zu simuliertem Exec-Preis
                    px = execution_price("buy", price_mid, init_usd, dollar_vol)
                    fee = fee_rate(False)  # Taker
                    eff_usd = init_usd * (1 + fee)  # Gebühren einpreisen
                    qty = (init_usd / px)
                    cash -= eff_usd
                    position = PositionState(
                        layers=[PositionLayer(entry_price=px, qty=qty)],
                        stop_price=px - 2.0 * atr_val,  # initial SL 2*ATR
                        last_add_price=px
                    )
                    records.append({"time": t, "side": "BUY", "price": px, "qty": qty, "equity": cash})

        # Pyramiding (Add-Ons nur im Gewinn, bis max Layers)
        if position and len(position.layers) < max_layers:
            target_add_px = position.last_add_price * (1 + PYRAMID_STEP_PCT)
            if price_mid >= target_add_px:
                # Add-On Volumen proportional zum initialen Risk (defensiv 50% des ersten USD)
                first_usd = position.layers[0].qty * position.layers[0].entry_price
                add_usd = min(first_usd * 0.5, cash)  # nicht überziehen
                # Exposure Cap + Volumen Cap
                eq_now = cash + sum(l.qty for l in position.layers) * price_mid
                exposure_after = (eq_now - cash + add_usd) / eq_now
                if exposure_after > MAX_PORTFOLIO_EXPOSURE:
                    add_usd = max(0.0, MAX_PORTFOLIO_EXPOSURE * eq_now - (eq_now - cash))
                add_usd = cap_by_volume(add_usd, dollar_vol)

                if add_usd > 5.0 and cash >= add_usd:
                    px = execution_price("buy", price_mid, add_usd, dollar_vol)
                    fee = fee_rate(False)
                    eff = add_usd * (1 + fee)
                    qty_add = add_usd / px
                    cash -= eff
                    position.layers.append(PositionLayer(entry_price=px, qty=qty_add))
                    position.last_add_price = px
                    # SL mind. nicht tiefer als 2*ATR unter letztem Entry
                    position.stop_price = max(position.stop_price, px - 2.0 * atr_val)
                    records.append({"time": t, "side": "PYRAMID", "price": px, "qty": qty_add, "equity": cash})

    # Close am Ende (Mark-to-Market -> realisieren wir nicht; Equity ist cash+MTM)
    eq_series = pd.Series({t: e for t, e in equity_curve})
    trades_df = pd.DataFrame(records)
    metrics = compute_metrics(eq_series)
    return eq_series, metrics, trades_df

# =========================
# Walk-Forward (optional)
# =========================
def walk_forward(df8: pd.DataFrame, df1: pd.DataFrame, equity_alloc: float):
    """
    Einfache Walk-Forward: rollierende Fenster (Train wird hier nicht 'gelernt',
    sondern nur Zeitsegmente zur Evaluierung genutzt).
    """
    start = df8.index.min()
    end = df8.index.max()
    cur = start + timedelta(days=WF_TRAIN_DAYS)
    all_eq = []
    while cur + timedelta(days=WF_TEST_DAYS) < end:
        test_start = cur
        test_end = cur + timedelta(days=WF_TEST_DAYS)
        sub8 = df8.loc[test_start:test_end].copy()
        sub1 = df1.loc[test_start:test_end].copy()
        if len(sub8) < 10 or len(sub1) < 10:
            cur += timedelta(days=WF_TEST_DAYS)
            continue
        eq, _, _ = backtest_symbol(sub8, sub1, equity_alloc)
        all_eq.append(eq)
        cur += timedelta(days=WF_TEST_DAYS)
    if not all_eq:
        return None
    # Kette die Equity-Segmente (naiv)
    res = pd.concat(all_eq).groupby(level=0).last().sort_index()
    return res

# =========================
# Hauptlauf (Multi-Asset)
# =========================
def main():
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        logging.warning("⚠️  Keine API Keys gefunden (.env). Historische Klines gehen mit öffentlichen Endpunkten, aber ratelimitiert.")
    client = Client(api_key, api_secret, testnet=True)

    start, end = get_time_range()
    logging.info(f"Backtest-Fenster (UTC): {start} -> {end}")

    per_symbol = START_EQUITY / len(COINS)
    portfolio_equity = None
    results: Dict[str, Dict] = {}

    for sym in COINS:
        logging.info(f"==> Lade Daten für {sym}")
        df8 = load_klines(client, sym, PRIMARY_INTERVAL, start, end)
        df1 = load_klines(client, sym, INTRADAY_INTERVAL, start, end)

        if DO_WALK_FORWARD:
            logging.info(f"[WF] {sym}")
            eq = walk_forward(df8, df1, per_symbol)
            if eq is None or len(eq) == 0:
                logging.warning(f"[WF] Kein Ergebnis für {sym}")
                continue
            met = compute_metrics(eq)
            tr = pd.DataFrame()
        else:
            eq, met, tr = backtest_symbol(df8, df1, per_symbol)

        results[sym] = {"equity": eq, "metrics": met, "trades": tr}

        if portfolio_equity is None:
            portfolio_equity = eq.rename(sym).to_frame()
        else:
            portfolio_equity = portfolio_equity.join(eq.rename(sym), how="outer")

    # Portfolio zusammenführen (Summation über Symbole)
    portfolio_equity = portfolio_equity.fillna(method="ffill").fillna(method="bfill")
    portfolio_equity["PORT"] = portfolio_equity.sum(axis=1)

    port_met = compute_metrics(portfolio_equity["PORT"])

    # Ausgabe
    print("\n=== METRIKEN pro Symbol ===")
    for sym, r in results.items():
        print(f"{sym}: " + ", ".join([f"{k}={v:.4f}" for k, v in r["metrics"].items()]))
    print("\n=== PORTFOLIO ===")
    print(", ".join([f"{k}={v:.4f}" for k, v in port_met.items()]))

    # Optional Plot
    if PLOT:
        fig, ax = plt.subplots()
        (portfolio_equity["PORT"]/START_EQUITY).plot(ax=ax, label="Portfolio")
        for sym in COINS:
            (portfolio_equity[sym]/(START_EQUITY/len(COINS))).plot(ax=ax, alpha=0.4, label=sym)
        ax.set_title("Equity (normalisiert)")
        ax.legend()
        plt.show()

        # Drawdown
        eq = portfolio_equity["PORT"]
        dd = eq/eq.cummax() - 1
        dd.plot(title="Portfolio Drawdown")
        plt.show()

    # Trades je Symbol (für Analyse in CSV)
    for sym, r in results.items():
        td = r["trades"]
        if td is not None and len(td) > 0:
            out = f"trades_{sym}.csv"
            td.to_csv(out, index=False)
            logging.info(f"Trades gespeichert: {out}")

if __name__ == "__main__":
    main()
