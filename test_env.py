import os
from dotenv import load_dotenv

# .env laden
load_dotenv()

# Variablen holen
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
tg_token = os.getenv("TELEGRAM_TOKEN")
tg_chat  = os.getenv("TELEGRAM_CHAT_ID")

print("BINANCE_API_KEY:", "✓ gesetzt" if api_key else "✗ FEHLT")
print("BINANCE_API_SECRET:", "✓ gesetzt" if api_secret else "✗ FEHLT")
print("TELEGRAM_TOKEN:", "✓ gesetzt" if tg_token else "(optional / leer)")
print("TELEGRAM_CHAT_ID:", "✓ gesetzt" if tg_chat else "(optional / leer)")
