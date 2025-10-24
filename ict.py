import pandas as pd
import ccxt
import numpy as np
from datetime import datetime

# =====================================================
# 📘 1. Daten holen (von Binance über CCXT)
# =====================================================
def fetch_data(symbol="BTC/USDT", timeframe="15m", limit=1000):
    exchange = ccxt.binance()
    exchange.load_markets()
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df


# =====================================================
# ⚙️ 2. Indikatoren & ICT-Konzepte berechnen
# =====================================================
def compute_indicators(df):
    # Exponentielle gleitende Durchschnitte
    df["ema_fast"] = df["close"].ewm(span=8).mean()
    df["ema_slow"] = df["close"].ewm(span=20).mean()

    # Struktur: Trendrichtung
    df["bull_trend"] = df["ema_fast"] > df["ema_slow"]

    # Fair Value Gap (FVG)
    # FVG entsteht, wenn die Low der aktuellen Kerze > High der vorigen + eine Lücke
    df["bull_fvg"] = (df["low"] > df["high"].shift(1))
    df["bear_fvg"] = (df["high"] < df["low"].shift(1))

    # Liquidity Sweep (z. B. Tief rausgenommen, aber wieder drüber geschlossen)
    df["bull_sweep"] = (df["low"] < df["low"].shift(1)) & (df["close"] > df["low"].shift(1))
    df["bear_sweep"] = (df["high"] > df["high"].shift(1)) & (df["close"] < df["high"].shift(1))

    # Combine into ICT Signals
    df["long_signal"] = df["bull_trend"] & (df["bull_fvg"] | df["bull_sweep"])
    df["short_signal"] = (~df["bull_trend"]) & (df["bear_fvg"] | df["bear_sweep"])

    return df


# =====================================================
# 💰 3. Einfacher Backtest
# =====================================================
def backtest(df, initial_balance=10000, risk_per_trade=0.01):
    balance = initial_balance
    position = None
    entry_price = 0
    wins, losses = 0, 0

    for i in range(1, len(df)):
        row = df.iloc[i]

        # LONG Einstieg
        if row["long_signal"] and position is None:
            position = "long"
            entry_price = row["close"]

        # SHORT Einstieg
        elif row["short_signal"] and position is None:
            position = "short"
            entry_price = row["close"]

        # Exit-Logik: Wenn Trend dreht oder Gegensignal kommt
        elif position == "long" and row["short_signal"]:
            change = (row["close"] - entry_price) / entry_price
            balance *= (1 + change * (risk_per_trade * 10))
            wins += change > 0
            losses += change <= 0
            position = None

        elif position == "short" and row["long_signal"]:
            change = (entry_price - row["close"]) / entry_price
            balance *= (1 + change * (risk_per_trade * 10))
            wins += change > 0
            losses += change <= 0
            position = None

    winrate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    return {
        "start_balance": initial_balance,
        "end_balance": round(balance, 2),
        "trades": wins + losses,
        "wins": wins,
        "losses": losses,
        "winrate_%": round(winrate, 2),
    }


# =====================================================
# 🚀 4. Hauptfunktion
# =====================================================
def run(symbol="BTC/USDT", timeframe="15m", limit=1500):
    print(f"Lade {limit} Kerzen ({timeframe}) für {symbol} ...")
    df = fetch_data(symbol, timeframe, limit)
    df = compute_indicators(df)
    stats = backtest(df)
    print("==== Backtest-Ergebnis ====")
    for k, v in stats.items():
        print(f"{k}: {v}")


# =====================================================
# 🏁 Startpunkt
# =====================================================
if __name__ == "__main__":
    run(symbol="BTC/USDT", timeframe="15min", limit=1500)
