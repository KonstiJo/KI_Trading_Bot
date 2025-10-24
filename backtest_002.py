import os
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

# === Einstellungen ===
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR
FAST_MA = 8
SLOW_MA = 20
TREND_MA = 50   # Trendfilter: nur handeln wenn Preis über dieser MA liegt
START_PERIOD = "30 day ago UTC"
STOP_LOSS_PCT = 0.03  # z. B. 3%
TAKE_PROFIT_PCT = 0.05  # optional: Gewinn mitnehmen bei +5%

# === Setup ===
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret, testnet=True)

# === Daten laden ===
print(f"Lade {SYMBOL} Daten für {START_PERIOD} im Interval {INTERVAL}")
klines = client.get_historical_klines(SYMBOL, INTERVAL, START_PERIOD)
df = pd.DataFrame(klines, columns=["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"])
df["close"] = df["close"].astype(float)

# === Indikatoren berechnen ===
df["fast_ema"] = df["close"].ewm(span=FAST_MA).mean()
df["slow_ema"] = df["close"].ewm(span=SLOW_MA).mean()
df["trend_ma"] = df["close"].ewm(span=TREND_MA).mean()

# === Backtest Simulation ===
in_position = False
entry_price = 0.0
profit = 0.0
wins = 0
losses = 0
total_trades = 0

for i in range(1, len(df)):
    fast_prev, fast_cur = df["fast_ema"].iloc[i-1], df["fast_ema"].iloc[i]
    slow_prev, slow_cur = df["slow_ema"].iloc[i-1], df["slow_ema"].iloc[i]
    trend = df["trend_ma"].iloc[i]
    price = df["close"].iloc[i]

    # Bedingung: Preis liegt über Trend-MA (nur Long) => Trendfilter
    if not in_position and price > trend and fast_prev <= slow_prev and fast_cur > slow_cur:
        # Einstieg
        entry_price = price
        in_position = True
        total_trades += 1
        #print(f"[BUY] @ {price:.2f}")

    elif in_position:
        # Stop-Loss Check
        if price <= entry_price * (1 - STOP_LOSS_PCT):
            trade_profit = price - entry_price
            profit += trade_profit
            losses += 1
            in_position = False
            #print(f"[SELL via Stop] @ {price:.2f} | Ergebnis: {trade_profit:.2f}")
        # Take-Profit Check
        elif price >= entry_price * (1 + TAKE_PROFIT_PCT):
            trade_profit = price - entry_price
            profit += trade_profit
            wins += 1
            in_position = False
            #print(f"[SELL via TP] @ {price:.2f} | Ergebnis: {trade_profit:.2f}")
        # Crossover Verkaufsignal
        elif fast_prev >= slow_prev and fast_cur < slow_cur:
            trade_profit = price - entry_price
            profit += trade_profit
            if trade_profit > 0:
                wins += 1
            else:
                losses += 1
            in_position = False
            total_trades += 1
            #print(f"[SELL via Crossover] @ {price:.2f} | Ergebnis: {trade_profit:.2f}")

# Ergebnis auswerten
print("=================================")
print(f"Trades insgesamt: {total_trades}")
print(f"Wins: {wins} | Losses: {losses}")
print(f"Winrate: { (wins / total_trades * 100) if total_trades>0 else 0:.2f}%")
print(f"Gesamtergebnis (fiktiv): {profit:.2f} USDT")
print(f"Parameter: EMA({FAST_MA}/{SLOW_MA}), TrendMA={TREND_MA}, STOP {STOP_LOSS_PCT*100:.1f}%, TP {TAKE_PROFIT_PCT*100:.1f}%")
