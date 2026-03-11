# KI_Trading_Bot

A collection of simple Python scripts for back‑testing and running
basic moving‑average and ICT‑style trading strategies against the
Binance testnet. The repository is educational in nature and contains
various experiments (`bot.py`, `newtest.py`, etc.) along with a few
utilities for data loading and strategy evaluation.

---

## 🧩 Project Structure

```
/                  <- root of the workspace
  backtest.py       # simple EMA crossover backtester
  backtest_002.py   # variant with trend filter & TP/SL
  bot.py            # live/testnet trading bot using EMA crossover
  bot_1.py          # older copy of above (kept for reference)
  ict.py            # minimal ICT‑style strategy example
  ict_fibo_volume_strategy.py  # more elaborate indicator toolbox + backtest
  newtest.py        # large hybrid/momentum+intraday backtester
  tetet.py          # another experimental multi‑coin pyramider
  test_env.py       # quick check for required environment variables
  tests/            # pytest unit tests
      test_bot.py
  requirements.txt  # Python dependencies (loose versions)
```

> 💡 **Note:** Several of the scripts print output or fetch live data when
> executed directly. Importing the modules no longer triggers network
> calls thanks to `if __name__ == "__main__"` guards added during
> cleanup.

---

## 🚀 Getting Started

The examples assume you are running inside a Python 3.12+ environment.
You can use a virtual environment, conda, or the provided dev container.

### 1. Install dependencies

```bash
cd /workspaces/KI_Trading_Bot
python -m pip install -r requirements.txt
```

The `requirements.txt` file uses open version specifiers so that you
can install the latest compatible packages. If you prefer a clean
install you can also run:

```bash
python -m pip install python-binance ccxt pandas numpy python-dotenv pytest requests ta matplotlib
```

> 📦 `ta` and `matplotlib` are optional; some scripts will warn if they
> are not available.

### 2. Configure credentials

Create a `.env` file in the project root with the following keys:

```
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
TELEGRAM_TOKEN=optional (for alerts)
TELEGRAM_CHAT_ID=optional
```

Use this repository only on the Binance **testnet** – real funds are not
supported and requests may be rate‑limited.

You can verify the environment by running:

```bash
python test_env.py
```

### 3. Running examples

All scripts are standalone Python files. When executed they will print
information about their progress (e.g. "== running backtest.py ==") and
a run‑time summary.

#### Web GUI (optional)

A simple Flask application (`app.py`) provides a web interface to start
and stop the various scripts and view their logs. To use it:

```bash
python -m pip install -r requirements.txt   # Flask is included
python app.py
```

Point your browser at `http://127.0.0.1:5000/` and you will see a
control panel listing each script with start/stop buttons and log links.
Logs accumulate in `logs/` (ignored by git).

---

Below are the most common entry points along with usage notes:

#### backtest.py

```bash
python backtest.py
```

*Simple EMA crossover backtester.*

- Downloads the last 7 days of BTCUSDT 1‑h candles from Binance testnet.
- Calculates 10‑/50‑EMA, simulates buy/sell on crossovers, and prints each
  trade as well as the net P/L. Sample output snippet:

```
== running backtest.py ==
Lade BTCUSDT-Kerzen (1h) für 7 day ago UTC...
Geladene Kerzen: 168
[BUY] @ 71471.13
[SELL] @ 72479.98 | Gewinn: 1008.85
...
Gesamtergebnis (fiktiv): 562.60 USDT
```

#### backtest_002.py

```bash
python backtest_002.py
```

Variant with a trend filter (50‑EMA), optional take‑profit/stop‑loss and
basic win/loss stats. Configurable parameter constants appear at the top
of the file.

#### bot.py / bot_1.py

```bash
python bot.py
# or
python bot_1.py
```

Live trading bot that places orders on the Binance **testnet**. Usage:

1. Ensure `.env` has valid API keys for the testnet.
2. Start the script; it will log connection status and then poll the
   market every minute.
3. The bot uses a 10/50 EMA crossover strategy with a 2 % stop‑loss and
   an optional daily loss limit.
4. Stop with Ctrl+C or let the daily loss limit halt execution.

Example log:

```
== running bot.py ==
2026-03-11 11:25:00,123 [INFO] Starte Trading-Bot für BTCUSDT - Strategie: 10/50 MA Crossover
2026-03-11 11:25:00,124 [INFO] 🚀 Verbunden mit Binance Testnet. Verfügbares Test-Guthaben: 1000.00 USDT
2026-03-11 11:26:00,130 [INFO] 📈 BUY-Signal erkannt (Preis=71000.1234). Platziere Kauforder...
```

> Note: `bot_1.py` is a duplicate kept for reference; prefer `bot.py`.

#### ict.py

```
python ict.py
```

Minimal ICT‑style example that fetches 15 m OHLCV data via CCXT,
computes a handful of indicators (EMAs, FVG/sweep) and runs a very basic
backtest. Useful for learning the indicator logic.

#### ict_fibo_volume_strategy.py

```
python ict_fibo_volume_strategy.py
```

More elaborate utility module exposing helper functions. Running it
retrieves live data and prints backtest stats similar to the snippet
above. Modify the call at the bottom for CSV input or different symbol.

#### newtest.py

```
python newtest.py
```

Advanced hybrid momentum + pyramiding backtest spanning multiple coins.
Download 8‑h and 1‑h data, compute ATR/RSI, and simulate trailing
stops. The script prints per‑symbol and portfolio metrics, and can
optionally plot equity curves if `matplotlib` is installed.

#### tetet.py

```
python tetet.py
```

Experimental multi‑coin pyramiding backtest for quick visualization of
strategy performance. Logging output shows entry/exit events.

#### highest_profit_factor.py

A tiny demonstration of the profit factor concept. Run it to see the
`inf` case when losses are zero.

---

### 4. Command‑line options

Most scripts have hard‑coded parameters; feel free to edit the constants
at the top of the file. You can also open them in an editor and add
e.g. `argparse` if you want command‑line flags.

### 5. Troubleshooting

- **missing module** errors: install the package via pip; the repository
  provides a liberal `requirements.txt`.
- **API errors**: make sure testnet API keys are valid and your IP isn’t
  rate‑limited.  You can enable debug‑level logging by setting
  `logging.basicConfig(level=logging.DEBUG)`.
- **no output when running**: ensure you are executing the script in the
  project root so that relative imports work and the `.env` is visible.

---

(The rest of the README remains unchanged.)

### 4. Running tests

A small suite of unit tests is located in `tests/test_bot.py`. Execute
with:

```bash
pytest -q
```

They exercise the crossover signal logic and confirm the import
workflow remains side‑effect free.

---

## 🛠️ Development Notes

- All market‑data access is currently done via the Binance REST API or
  CCXT where noted.  No cache or rate‑limit handling is implemented.
- The code is intentionally simple; performance, error handling and
  proper order sizing are **not** production‑grade.
- Feel free to experiment: add new indicators, swap data sources, or
  convert backtests to use `pandas` vectorisation for speed.

---

## 📄 License

This repository is provided under the MIT license. Use at your own risk.

Enjoy! 👋

