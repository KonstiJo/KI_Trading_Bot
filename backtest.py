import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import os

# === Einstellungen ===
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR  # 1-Stunden-Chart
FAST_MA = 10
SLOW_MA = 50
START_PERIOD = "7 day ago UTC"

# === Setup ===
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret, testnet=True)

# === Daten laden ===
print(f"Lade {SYMBOL}-Kerzen ({INTERVAL}) für {START_PERIOD}...")
klines = client.get_historical_klines(SYMBOL, INTERVAL, START_PERIOD)
print(f"Geladene Kerzen: {len(klines)}")

df = pd.DataFrame(klines, columns=["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"])
df["close"] = df["close"].astype(float)

# === EMAs berechnen ===
df["fast_ema"] = df["close"].ewm(span=FAST_MA).mean()
df["slow_ema"] = df["close"].ewm(span=SLOW_MA).mean()

# === Backtest Simulation ===
in_position = False
entry_price = 0
profit = 0

for i in range(1, len(df)):
    fast_prev, fast_cur = df["fast_ema"].iloc[i-1], df["fast_ema"].iloc[i]
    slow_prev, slow_cur = df["slow_ema"].iloc[i-1], df["slow_ema"].iloc[i]
    price = df["close"].iloc[i]

    # Kaufsignal
    if not in_position and fast_prev <= slow_prev and fast_cur > slow_cur:
        entry_price = price
        in_position = True
        print(f"[BUY] @ {price:.2f}")

    # Verkaufssignal
    elif in_position and fast_prev >= slow_prev and fast_cur < slow_cur:
        trade_profit = price - entry_price
        profit += trade_profit
        print(f"[SELL] @ {price:.2f} | Gewinn: {trade_profit:.2f}")
        in_position = False

# === Ergebnisse ===
print("=================================")
print(f"Gesamtergebnis (fiktiv): {profit:.2f} USDT")
print(f"Anzahl Kerzen: {len(df)}")
print(f"Parameter: EMA({FAST_MA}/{SLOW_MA}), Zeitraum: {START_PERIOD}")
