# backtest_multi_coin_pyramiding.py


import os
import time
import math
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from binance.client import Client
from dotenv import load_dotenv

# --- Konfiguration ---
COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]   # Liste der Handelspaare
INTERVAL = Client.KLINE_INTERVAL_8HOUR     # 8-Stunden Chart
LOOKBACK = 500                             # Anzahl Kerzen für historische Daten
START_CAPITAL_USDT = 10000                 # Paper-Kapital
MAX_POS_PER_COIN = 3                       # Maximal Pyramiding-Ebenen pro Coin
PYRAMID_STEP_PCT = 0.02                    # Abstand zwischen Ebenen (z. B. 2 %)
MOMENTUM_WINDOW = 20                       # Fenster für Momentum-Berechnung
INTRADAY_LOOKBACK = 3                      # z. B. für Intraday Einstieg (3 Kerzen)
MIN_MOMENTUM_PCT = 0.05                    # Mindest-Momentum (z. B. +5 % innerhalb Fenster) für Einstieg
STOP_LOSS_PCT = 0.03                       # Stop-Loss Beispiel 3 %
TAKE_PROFIT_PCT = 0.06                     # Take-Profit Beispiel 6 %
FEE_PCT = 0.001                            # angenommen Binance Spot Fee ~0,1%

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# API Keys laden (wenn gebraucht, aber beim reinen Backtest kann ohne Handel)
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(API_KEY, API_SECRET, testnet=True)

# Hilfsfunktion: historische Daten laden
def get_historical(symbol, interval, lookback):
    """Lädt historische Kerzen (OHLCV) für ein Symbol."""
    logging.info(f"Lade {lookback} Kerzen für {symbol} im Interval {interval}")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["time"] = pd.to_datetime(df["open_time"], unit='ms')
    df = df.set_index("time")
    return df

# Hilfsfunktion: Momentum berechnen
def compute_momentum(df, window):
    """Berechnet Momentum als prozentuale Veränderung über window Kerzen."""
    df["mom"] = df["close"].pct_change(periods=window)
    return df

# Backtest Logik: Pyramiding Entry + Intraday + Momentum
def backtest_symbol(df, capital_usdt):
    """Backtest für ein einzelnes Symbol mit Pyramiding."""
    cash = capital_usdt
    pos_size_usdt = 0
    positions = []  # Liste von offenen Positionen: dict mit entry_price, level
    equity_curve = []

    df = compute_momentum(df, MOMENTUM_WINDOW)
    # ggf. Intraday Filter: z. B. wenn letzte INTRADAY_LOOKBACK Kerzen starken Move …
    for idx in range(LOOKBACK - 1):
        row = df.iloc[idx]
        # Prüfen ob Einstieg möglich ist
        if row["mom"] is None or math.isnan(row["mom"]):
            continue
        if row["mom"] >= MIN_MOMENTUM_PCT and len(positions) == 0:
            # Einstieg Level 1
            entry_price = row["close"]
            pos_size_usdt = cash * 0.10  # Beispiel: 10 % des Kapitals
            qty = pos_size_usdt / entry_price
            positions.append({"entry_price": entry_price, "level": 1, "qty": qty})
            cash -= pos_size_usdt
            logging.info(f"{row.name} ↑ Einstieg {symbol} @ {entry_price:.4f}, Qty={qty:.6f}")
        # Pyramiding: wenn bereits Position offen und Preissprung …
        elif len(positions) > 0 and len(positions) < MAX_POS_PER_COIN:
            last_pos = positions[-1]
            if row["close"] >= last_pos["entry_price"] * (1 + PYRAMID_STEP_PCT):
                # neue Pyramiding-Stufe
                entry_price = row["close"]
                pos_size_usdt = cash * 0.10
                qty = pos_size_usdt / entry_price
                positions.append({"entry_price": entry_price, "level": last_pos["level"] + 1, "qty": qty})
                cash -= pos_size_usdt
                logging.info(f"{row.name} ↑ Pyramiding Level {last_pos['level'] + 1} @ {entry_price:.4f}")
        # Risikokontrolle: Stop-Loss / Take-Profit prüfen
        new_positions = []
        for pos in positions:
            current_price = row["close"]
            entry_price = pos["entry_price"]
            if current_price <= entry_price * (1 - STOP_LOSS_PCT):
                # Stop-Loss auslösen
                cash += pos["qty"] * current_price * (1 - FEE_PCT)
                logging.info(f"{row.name} ↓ Stop-Loss ausgelöst @ {current_price:.4f}")
            elif current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                # Take-Profit realisieren
                cash += pos["qty"] * current_price * (1 - FEE_PCT)
                logging.info(f"{row.name} ↑ Take-Profit realisiert @ {current_price:.4f}")
            else:
                new_positions.append(pos)
        positions = new_positions
        # Gesamte Positionsbewertung
        pos_value = sum(pos["qty"] * row["close"] for pos in positions)
        equity = cash + pos_value
        equity_curve.append(equity)
    # Rückgabe: End­kapital & Equity-Verlauf
    return equity_curve

# Multi-Coin Backtest
def run_backtest():
    results = {}
    for symbol in COINS:
        df = get_historical(symbol, INTERVAL, LOOKBACK)
        equity_curve = backtest_symbol(df.copy(), START_CAPITAL_USDT / len(COINS))
        results[symbol] = equity_curve
    # Ausgabe
    for symbol, curve in results.items():
        final = curve[-1] if len(curve) > 0 else None
        logging.info(f"Ergebnis {symbol}: Endkapital ~ {final:.2f} USDT")
    # Hier könnte man Kurven kombinieren, plotten etc.
    return results

if __name__ == "__main__":
    results = run_backtest()
    # Beispielplot (optional)
    import matplotlib.pyplot as plt
    for symbol, curve in results.items():
        plt.plot(curve, label=symbol)
    plt.legend()
    plt.title("Equity Curves der Strategie")
    plt.xlabel("Zeit-Schritt")
    plt.ylabel("USDT Equity")
    plt.show()
