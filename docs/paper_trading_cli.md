# Terminal Paper Trading CLI

The CLI keeps paper trading fully local: signals are generated, executed, and
logged on the same machine where the command is launched. No workloads are
delegated to the frontend or any cloud service.

---

## Installation

```bash
pip install ./pytrader\ sdk
```

The installation registers the `pytrader-paper` console script.

---

## Usage

```bash
pytrader-paper \
  --strategy macd_crossover \
  --symbols OGDC,HBL,LUCK \
  --config '{"fast_period":10,"slow_period":22,"signal_period":7}' \
  --token YOUR_LOCAL_TOKEN \
  --capital 1500000 \
  --position-notional 250000 \
  --cycle-minutes 10 \
  --metrics-path logs/cli/metrics.csv \
  --trades-path logs/cli/trades.csv
```

### Required arguments

| Flag        | Description                                                                                  |
|-------------|----------------------------------------------------------------------------------------------|
| `--strategy`| Registered template (`sma_momentum`, `dual_sma_momentum`, `rsi_momentum`, `macd_crossover`, `bollinger_mean_reversion`, `vwap_reversion`, `testbots_momentum`, `buying_on_up`, `buying_on_down`, `ml_layer`) |
| `--symbols` | Comma or space separated PSX tickers                                                         |
| `--token`   | Local PyPSX token (never transmitted anywhere else)                                          |

### Optional arguments

| Flag                 | Description                                                                           |
|----------------------|---------------------------------------------------------------------------------------|
| `--config`           | JSON string or path to JSON file with strategy parameters                             |
| `--capital`          | Initial paper capital (default 1,000,000)                                             |
| `--position-notional`| Target notional per trade                                                             |
| `--cycle-minutes`    | Polling cadence for new data                                                          |
| `--max-cycles`       | Run for N cycles then stop (omit to run indefinitely)                                 |
| `--bot-id`           | Custom identifier appended to log filenames                                           |
| `--metrics-path`     | CSV path for rolling metrics (defaults to `logs/paper_cli/<bot>-<ts>-metrics.csv`)    |
| `--trades-path`      | CSV path for trade blotter (defaults to `logs/paper_cli/<bot>-<ts>-trades.csv`)       |
| `--cold-start`       | Skip warm-start replay and begin from the next live batch                             |
| `--detailed`         | Show verbose cycle-by-cycle report instead of the compact dashboard                   |
| `--log-dir`          | Directory for supplementary logs (defaults to `logs/`)                                |

---

## Screen Layout

1. **Top**: Cash, equity, realized/unrealized PnL, exposure.
2. **Middle**: Per-symbol line items (qty, avg cost, last price, unrealized PnL).
3. **Bottom**: Rolling list of the most recent trades plus strategy messages.

All output is plain text to keep terminals clean and scroll-friendly.

---

## Logs

Both `metrics_path` and `trades_path` are CSV files that can be tailed or opened
in notebooks for further analysis. See `docs/log_formats.md` for a detailed
column reference.

