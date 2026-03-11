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

Here are a few of the most useful entry points:

| Script | Description |
|--------|-------------|
| `python backtest.py` | EMA crossover backtest (prints trades & P/L) |
| `python backtest_002.py` | Extended backtester with trend filter, TP/SL |
| `python bot.py` | Live/testnet execution bot (EMA crossover) |
| `python ict_fibo_volume_strategy.py` | Toolbox + quick backtest example |
| `python newtest.py` | Advanced hybrid/momentum/intraday backtester |
| `python tetet.py` | Experimental multi‑coin pyramiding backtest |

The `bot.py` file will run indefinitely and place orders on Binance
(testnet) until stopped with Ctrl+C. It logs to the console and can send
Telegram alerts if configured.

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

