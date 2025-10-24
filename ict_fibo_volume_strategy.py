import os
import math
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# =========================================================
# Daten-Beschaffung
# =========================================================

def fetch_ohlcv_ccxt(symbol: str = "BTC/USDT", timeframe: str = "15m", limit: int = 2000) -> pd.DataFrame:
    """
    Holt OHLCV-Daten über CCXT (falls installiert).
    Gibt ein DataFrame mit Spalten: time, open, high, low, close, volume (UTC ms -> Datetime)
    """
    try:
        import ccxt
    except ImportError as e:
        raise RuntimeError("CCXT ist nicht installiert. Installiere mit: pip install ccxt") from e

    ex = ccxt.binance()
    # ex.set_sandbox_mode(True)  # nur wenn du explizit die Binance-Sandbox nutzen willst
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """
    Erwartet CSV mit Spalten: time, open, high, low, close, volume
    time kann Timestamp (ms) oder ISO-String sein.
    """
    df = pd.read_csv(path)
    if np.issubdtype(df["time"].dtype, np.number):
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]].copy()


# =========================================================
# Indikatoren & Hilfen
# =========================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def volume_filter(df: pd.DataFrame, ma_len: int = 20, mul: float = 1.2) -> pd.Series:
    """
    True, wenn aktuelles Volumen > mul * Volumen-SMA(ma_len)
    """
    vol_ma = df["volume"].rolling(ma_len, min_periods=1).mean()
    return df["volume"] > (mul * vol_ma)

def swings(df: pd.DataFrame, lb: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Markiert Swing High/Low (einfach): Hoch ist größer als lb links & rechts, Low analog.
    Gibt zwei bool-Serien zurück (is_swing_high, is_swing_low).
    """
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(lb, n - lb):
        if high[i] == max(high[i - lb:i + lb + 1]):
            sh[i] = True
        if low[i] == min(low[i - lb:i + lb + 1]):
            sl[i] = True
    return pd.Series(sh, index=df.index), pd.Series(sl, index=df.index)

def last_swing_high_low(df: pd.DataFrame, i: int, sh: pd.Series, sl: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    Sucht rückwärts bis Index i den letzten Swing High/Low.
    """
    idxs = np.arange(0, i + 1)
    sh_pos = idxs[sh.iloc[: i + 1].values]
    sl_pos = idxs[sl.iloc[: i + 1].values]
    last_sh = int(sh_pos[-1]) if len(sh_pos) else None
    last_sl = int(sl_pos[-1]) if len(sl_pos) else None
    hi = float(df["high"].iloc[last_sh]) if last_sh is not None else None
    lo = float(df["low"].iloc[last_sl]) if last_sl is not None else None
    return hi, lo, last_sh, last_sl

def fib_levels(high: float, low: float) -> Dict[str, float]:
    """
    Klassische Retracement-Levels (0..1) inkl. 0.5 Equilibrium.
    """
    diff = high - low
    return {
        "0.0": low,
        "0.236": low + 0.236 * diff,
        "0.382": low + 0.382 * diff,
        "0.5": low + 0.5 * diff,         # Equilibrium
        "0.618": low + 0.618 * diff,
        "0.786": low + 0.786 * diff,
        "1.0": high
    }

def equilibrium(high: float, low: float) -> float:
    return (high + low) / 2.0

def detect_fvg(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    ICT FVG (3-Kerzen-Logik)
    Bullish FVG an Bar i, wenn: high[i-2] < low[i]
    Bearish FVG an Bar i, wenn: low[i-2] > high[i]
    Gibt bool-Serien (bull_fvg, bear_fvg) und DataFrames mit FVG-Bereich (oben/unten) zurück.
    """
    high = df["high"]
    low = df["low"]
    bull = (high.shift(2) < low)
    bear = (low.shift(2) > high)

    # Bereiche
    bull_top = low
    bull_bottom = high.shift(2)
    bear_top = low.shift(2)
    bear_bottom = high

    bull_box = pd.DataFrame({"top": bull_top, "bottom": bull_bottom})
    bear_box = pd.DataFrame({"top": bear_top, "bottom": bear_bottom})
    # Nur dort, wo wirklich FVG
    bull_box.loc[~bull, ["top", "bottom"]] = np.nan
    bear_box.loc[~bear, ["top", "bottom"]] = np.nan

    return bull.fillna(False), bear.fillna(False), bull_box, bear_box

def liquidity_sweep(df: pd.DataFrame, sh: pd.Series, sl: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Einfache Sweep-Heuristik:
    - Bearish Sweep: High bricht letztes Swing High, Close < letztem Swing High
    - Bullish Sweep: Low bricht letztes Swing Low,  Close > letztem Swing Low
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    idx = df.index

    bear_sw = np.zeros(len(df), dtype=bool)
    bull_sw = np.zeros(len(df), dtype=bool)

    last_sh_price = None
    last_sl_price = None
    for i in range(len(df)):
        if sh.iloc[i]:
            last_sh_price = highs[i]
        if sl.iloc[i]:
            last_sl_price = lows[i]

        if last_sh_price is not None:
            if highs[i] > last_sh_price and closes[i] < last_sh_price:
                bear_sw[i] = True
        if last_sl_price is not None:
            if lows[i] < last_sl_price and closes[i] > last_sl_price:
                bull_sw[i] = True

    return pd.Series(bear_sw, index=idx), pd.Series(bull_sw, index=idx)

# =========================================================
# Signal-Generierung (ICT + Fibo + Volumen)
# =========================================================

def generate_signals(df: pd.DataFrame,
                     fast_ema_len: int = 8,
                     slow_ema_len: int = 20,
                     vol_ma_len: int = 20,
                     vol_mul: float = 1.2) -> pd.DataFrame:
    """
    Erzeugt Long/Short-Signale basierend auf:
    - EMA Crossover (Richtung)
    - FVG in dieselbe Richtung
    - Liquidity Sweep als Trigger
    - Trend/Equilibrium Filter via letztem Swing
    - Volumen-Filter
    """
    out = df.copy()
    out["ema_fast"] = ema(out["close"], fast_ema_len)
    out["ema_slow"] = ema(out["close"], slow_ema_len)
    out["ema_dir_up"] = out["ema_fast"] > out["ema_slow"]
    out["ema_dir_dn"] = out["ema_fast"] < out["ema_slow"]

    sh, sl = swings(out, lb=3)
    out["swing_high"] = sh
    out["swing_low"] = sl

    bear_sw, bull_sw = liquidity_sweep(out, sh, sl)
    out["bear_sweep"] = bear_sw
    out["bull_sweep"] = bull_sw

    bull_fvg, bear_fvg, bull_box, bear_box = detect_fvg(out)
    out["bull_fvg"] = bull_fvg
    out["bear_fvg"] = bear_fvg
    out[["bull_fvg_top", "bull_fvg_bottom"]] = bull_box[["top", "bottom"]]
    out[["bear_fvg_top", "bear_fvg_bottom"]] = bear_box[["top", "bottom"]]

    out["vol_ok"] = volume_filter(out, ma_len=vol_ma_len, mul=vol_mul)

    # Equilibrium Filter: nutze letztes Swing-Paar bis i
    eq_list = []
    prem_disc_list = []  # "discount" (unter 50%) / "premium" (über 50%) / None
    for i in range(len(out)):
        hi, lo, hi_i, lo_i = last_swing_high_low(out, i, sh, sl)
        if hi is not None and lo is not None:
            eq = equilibrium(hi, lo)
            eq_list.append(eq)
            prem_disc_list.append("discount" if out["close"].iloc[i] < eq else "premium")
        else:
            eq_list.append(np.nan)
            prem_disc_list.append(None)
    out["equilibrium"] = eq_list
    out["pd_zone"] = prem_disc_list

    # Long, wenn:
    #  - ema_dir_up
    #  - bull_fvg
    #  - bull_sweep (Liquidity unter Low, Close wieder drüber)
    #  - in Discount-Zone
    #  - Volumen-Bestätigung
    out["long_signal"] = (
        out["ema_dir_up"] &
        (out["bull_fvg"] | out["bull_sweep"]) & #FVG ODER Sweep reicht
        (out["pd_zone"] == "discount")
    )

    # Short, wenn:
    #  - ema_dir_dn
    #  - bear_fvg
    #  - bear_sweep
    #  - in Premium-Zone
    #  - Volumen-Bestätigung
    out["short_signal"] = (
        out["ema_dir_dn"] &
        (out["bear_fvg"] | out["bear_sweep"]) &
        (out["pd_zone"] == "premium")
    )

    return out


# =========================================================
# Einfacher Backtest
# =========================================================

def backtest(df: pd.DataFrame,
             rr: float = 1.5,
             sl_buffer_pct: float = 0.0005,
             risk_per_trade: float = 100.0) -> Dict[str, float]:
    """
    Simpler 1-Trade-zur-Zeit Backtest:
    - Entry: FVG-Mittenpreis (wenn verfügbar), sonst Close
    - SL: knapp unter/über dem gesweepten Swing (mit kleinem Buffer)
    - TP: RR-Multiplikator (z.B. 1.5R)
    - Positionsgröße anhand Risikobetrag (risk_per_trade / SL-Distanz)
    """
    equity = 0.0
    wins = 0
    losses = 0
    trades = 0

    in_pos = False
    side = None
    entry = None
    sl = None
    tp = None
    size = None

    for i in range(5, len(df)):  # start später, um Indikator-Warmup zu skippen
        row = df.iloc[i]

        # Falls Position offen, prüfe TP/SL (High/Low-Berührung)
        if in_pos:
            hi = row["high"]
            lo = row["low"]

            if side == "long":
                hit_tp = hi >= tp
                hit_sl = lo <= sl
            else:
                hit_tp = lo <= tp
                hit_sl = hi >= sl

            if hit_tp or hit_sl:
                trades += 1
                pnl = (tp - entry) * size if hit_tp else (sl - entry) * size
                equity += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                in_pos = False
                side = None
                entry = sl = tp = size = None
                continue  # nächste Kerze

        # Wenn keine Position offen, nach Signalen schauen
        # LONG
        if (not in_pos) and row.get("long_signal", False):
            # Entry am FVG-Mid (falls existiert), sonst Close
            fvg_top = row.get("bull_fvg_top", np.nan)
            fvg_bottom = row.get("bull_fvg_bottom", np.nan)
            if np.isfinite(fvg_top) and np.isfinite(fvg_bottom):
                entry_price = (fvg_top + fvg_bottom) / 2.0
            else:
                entry_price = row["close"]

            # SL unter letztem Swing Low
            # Finde den letzten Swing Low vor i
            sl_price = None
            for j in range(i, max(0, i - 50), -1):
                if df["swing_low"].iloc[j]:
                    sl_price = float(df["low"].iloc[j])
                    break
            if sl_price is None:
                continue  # kein valider SL

            sl_price *= (1.0 - sl_buffer_pct)

            risk_per_unit = entry_price - sl_price
            if risk_per_unit <= 0:
                continue
            position_size = risk_per_trade / risk_per_unit
            take_profit = entry_price + rr * risk_per_unit

            # Position setzen
            in_pos = True
            side = "long"
            entry = entry_price
            sl = sl_price
            tp = take_profit
            size = position_size
            continue

        # SHORT
        if (not in_pos) and row.get("short_signal", False):
            fvg_top = row.get("bear_fvg_top", np.nan)
            fvg_bottom = row.get("bear_fvg_bottom", np.nan)
            if np.isfinite(fvg_top) and np.isfinite(fvg_bottom):
                entry_price = (fvg_top + fvg_bottom) / 2.0
            else:
                entry_price = row["close"]

            # SL über letztem Swing High
            sl_price = None
            for j in range(i, max(0, i - 50), -1):
                if df["swing_high"].iloc[j]:
                    sl_price = float(df["high"].iloc[j])
                    break
            if sl_price is None:
                continue

            sl_price *= (1.0 + sl_buffer_pct)

            risk_per_unit = sl_price - entry_price
            if risk_per_unit <= 0:
                continue
            position_size = risk_per_trade / risk_per_unit
            take_profit = entry_price - rr * risk_per_unit

            in_pos = True
            side = "short"
            entry = entry_price
            sl = sl_price
            tp = take_profit
            size = position_size
            continue

    winrate = (wins / trades * 100.0) if trades > 0 else 0.0
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate_pct": winrate,
        "equity_change_usd": equity,
    }


# =========================================================
# Beispiel-Nutzung
# =========================================================

def run_example(use_ccxt: bool = True, csv_path: Optional[str] = None,
                symbol: str = "BTC/USDT", timeframe: str = "1m", limit: int = 5000):
    if use_ccxt:
        df = fetch_ohlcv_ccxt(symbol=symbol, timeframe=timeframe, limit=limit)
    else:
        if not csv_path:
            raise ValueError("Bitte csv_path angeben oder use_ccxt=True setzen.")
        df = load_csv(csv_path)

    # Indikatoren & Signale
    data = generate_signals(df,
                            fast_ema_len=8,
                            slow_ema_len=20,
                            vol_ma_len=20,
                            vol_mul=1.2)

    # Backtest
    stats = backtest(data,
                     rr=1.5,
                     sl_buffer_pct=0.0005,
                     risk_per_trade=100.0)

    print("==== Backtest-Ergebnis ====")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    # Beispiel: CCXT holen (Internet notwendig). Sonst CSV nutzen.
    run_example(use_ccxt=True, symbol="BTC/USDT", timeframe="1h", limit=1500)
    # Für CSV:
    # run_example(use_ccxt=False, csv_path="./btc_1h.csv")
