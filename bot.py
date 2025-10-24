import os
import time
import math
import logging
from datetime import datetime, timedelta

# Drittanbieter-Bibliotheken
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import requests

# --- Konfiguration ---
# Strategie-Parameter (Moving Averages)
SYMBOL = "BTCUSDT"                        # Handels-Paar (BTC/USDT als Beispiel im Testnet)
FAST_MA = 10                              # Zeitraum für schnellen gleitenden Durchschnitt (z.B. 10 Kerzen)
SLOW_MA = 50                              # Zeitraum für langsamen gleitenden Durchschnitt (z.B. 50 Kerzen)
INTERVAL = Client.KLINE_INTERVAL_1MINUTE  # Intervall der Kerzen: 1-Minuten-Chart

# Risikomanagement
TRADE_AMOUNT_USDT = 100       # Einsatz pro Trade (wir nutzen ~100 USDT vom Testnet-Guthaben)
STOP_LOSS_PCT = 0.02          # Stop-Loss: 2% unter Einstiegsprice verkauft der Bot (Begrenzung des Verlustes)
MAX_DAILY_LOSS_USD = 10       # Maximaler erlaubter Verlust pro Tag (z.B. 10 USDT). Bot stoppt danach.

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Laden der API-Schlüssel aus .env
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not API_KEY or not API_SECRET:
    logging.error("API-Schlüssel nicht gefunden. Bitte .env Datei mit BINANCE_API_KEY und BINANCE_API_SECRET erstellen.")
    exit(1)

# Binance Client initialisieren (Testnet)
client = Client(API_KEY, API_SECRET, testnet=True)

try:
    # Testweise Kontostand abrufen, um Verbindung und Keys zu prüfen
    account_info = client.get_account()
    balances = {bal['asset']: float(bal['free']) for bal in account_info.get('balances', [])}
    usdt_balance = balances.get('USDT', 0.0)
    logging.info(f"🚀 Verbunden mit Binance Testnet. Verfügbares Test-Guthaben: {usdt_balance:.2f} USDT")
except BinanceAPIException as e:
    logging.error(f"Fehler bei API-Verbindung: {e.message}. Überprüfen Sie die API-Schlüssel.")
    exit(1)

# Handelsstatus Variablen
in_position = False
entry_price = None
daily_loss = 0.0

# --- FUNKTIONEN ---

def check_for_signal(fast_ma_prev, fast_ma_cur, slow_ma_prev, slow_ma_cur, in_position):
    """
    Überprüft die Überkreuzung der gleitenden Durchschnitte.
    Gibt 'BUY', 'SELL' oder None zurück.
    """
    buy_signal = (fast_ma_prev <= slow_ma_prev) and (fast_ma_cur > slow_ma_cur) and not in_position
    sell_signal = (fast_ma_prev >= slow_ma_prev) and (fast_ma_cur < slow_ma_cur) and in_position
    if buy_signal:
        return "BUY"
    if sell_signal:
        return "SELL"
    return None


def send_telegram_alert(message: str):
    """Sende eine Nachricht an Telegram (optional)."""
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            params = {"chat_id": chat_id, "text": message}
            requests.get(url, params=params, timeout=5)
        except Exception as e:
            logging.error(f"Telegram Alert fehlgeschlagen: {e}")


# --- HAUPTPROGRAMM (wird nur ausgeführt, wenn Datei direkt gestartet wird) ---
if __name__ == "__main__":
    logging.info(f"Starte Trading-Bot für {SYMBOL} - Strategie: {FAST_MA}/{SLOW_MA} MA Crossover")
    logging.info("Drücken Sie STRG+C zum Beenden des Bots.")

    while True:
        try:
            # 1. Marktdaten abrufen
            klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=SLOW_MA + 10)
            df = pd.DataFrame(
                klines,
                columns=["open_time", "open", "high", "low", "close", "volume",
                         "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"]
            )
            df["close"] = df["close"].astype(float)

            # 2. Indikatoren berechnen (EMAs)
            df["fast_ema"] = df["close"].ewm(span=FAST_MA).mean()
            df["slow_ema"] = df["close"].ewm(span=SLOW_MA).mean()

            fast_prev, fast_cur = df["fast_ema"].iloc[-2], df["fast_ema"].iloc[-1]
            slow_prev, slow_cur = df["slow_ema"].iloc[-2], df["slow_ema"].iloc[-1]
            last_price = df["close"].iloc[-1]

            # 3. Stop-Loss prüfen
            if in_position and entry_price:
                if last_price <= entry_price * (1 - STOP_LOSS_PCT):
                    logging.warning(
                        f"❗️ Stop-Loss ausgelöst! Aktueller Preis {last_price:.4f} <= {STOP_LOSS_PCT*100:.1f}% unter Entry {entry_price:.4f}. Verkaufe..."
                    )
                    try:
                        sell_quantity = TRADE_AMOUNT_USDT / entry_price
                        order = client.order_market_sell(symbol=SYMBOL, quantity=round(sell_quantity, 6))
                    except BinanceAPIException as e:
                        logging.error(f"Verkaufsorder fehlgeschlagen: {e.message}")
                    in_position = False
                    loss = (last_price - entry_price) * (TRADE_AMOUNT_USDT / entry_price)
                    daily_loss += loss
                    logging.info(f"Verkauft (Stop-Loss) {SYMBOL} @ {last_price:.4f}. Verlust: {loss:.2f} USDT.")
                    send_telegram_alert(f"Stop-Loss verkauft {SYMBOL} @ {last_price:.4f}, Verlust {loss:.2f} USDT")
                    entry_price = None
                    time.sleep(60)
                    continue

            # 4. Handelssignal prüfen
            signal = check_for_signal(fast_prev, fast_cur, slow_prev, slow_cur, in_position)

            if signal == "BUY":
                logging.info(f"📈 BUY-Signal erkannt (Preis={last_price:.4f}). Platziere Kauforder...")
                try:
                    buy_quantity = TRADE_AMOUNT_USDT / last_price
                    order = client.order_market_buy(symbol=SYMBOL, quantity=round(buy_quantity, 6))
                except BinanceAPIException as e:
                    logging.error(f"Kauforder fehlgeschlagen: {e.message}")
                else:
                    in_position = True
                    entry_price = last_price
                    logging.info(f"Gekauft {SYMBOL} @ {entry_price:.4f} (Menge ~{buy_quantity:.6f}).")
                    send_telegram_alert(f"Gekauft {SYMBOL} @ {entry_price:.4f}")

            elif signal == "SELL":
                logging.info(f"📉 SELL-Signal erkannt (Preis={last_price:.4f}). Verkaufe Position...")
                try:
                    sell_quantity = TRADE_AMOUNT_USDT / entry_price
                    order = client.order_market_sell(symbol=SYMBOL, quantity=round(sell_quantity, 6))
                except BinanceAPIException as e:
                    logging.error(f"Verkaufsorder fehlgeschlagen: {e.message}")
                else:
                    in_position = False
                    exit_price = last_price
                    profit = (exit_price - entry_price) * (TRADE_AMOUNT_USDT / entry_price)
                    if profit < 0:
                        daily_loss += profit
                    logging.info(f"Verkauft {SYMBOL} @ {exit_price:.4f}. Ergebnis: {profit:+.2f} USDT.")
                    send_telegram_alert(f"Verkauft {SYMBOL} @ {exit_price:.4f}, Ergebnis: {profit:+.2f} USDT")
                    entry_price = None

            # 5. Tagesverlust prüfen
            if daily_loss <= -MAX_DAILY_LOSS_USD:
                logging.error(f"Maximaler Tagesverlust erreicht ({daily_loss:.2f} USDT). Bot pausiert.")
                send_telegram_alert(f"Tagesverlust-Limit erreicht: {daily_loss:.2f} USDT. Bot pausiert.")
                break

        except Exception as e:
            logging.error(f"Unerwarteter Fehler: {e}")
            time.sleep(60)
            continue

        time.sleep(60)
