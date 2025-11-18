# Paper Trading Modes: Warm-Start vs Cold-Start

PyTrader supports two initialization modes for live paper trading, allowing you to choose how the engine initializes when starting mid-day or at market open.

## Overview

### Warm-Start (Default)
**What it does:** Replays all available data from today's market open until the current time, simulating what would have happened if the bot had been running since market open.

**Use cases:**
- Development and testing
- Evaluating strategy performance mid-day
- Continuous runs where you want to see full-day context

**Example:**
```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")
bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC", "HBL"],
    warm_start=True  # Default
)
```

### Cold-Start
**What it does:** Starts fresh from the current time with live data only. No historical replay - begins trading immediately with whatever portfolio state exists (usually cash only).

**Use cases:**
- True user sessions started mid-day
- When you want to simulate "starting now" scenario
- Testing how a bot performs when launched at arbitrary times

**Example:**
```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")
bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC", "HBL"],
    warm_start=False  # Cold-start mode
)
```

## Account Persistence

PyTrader maintains persistent account state across sessions, similar to Alpaca's paper trading model:

- **Cash** persists day-to-day
- **Positions** carry over until closed
- **Portfolio value** updates continuously
- **Trade history** is maintained

Account state is stored in `user_data/accounts/{user_id}.json` and is automatically saved after each live cycle.

### Resetting an Account

Account state is managed by the backend. To reset your account, stop and restart the bot with the desired initial cash amount.

## Logging Examples

### Warm-Start Mode
```
[MODE: LIVE] [WARM START] [runall-live] Starting live paper trading session for ['OGDC', 'HBL']
[runall-live] [WARM START] Replaying today's data from open to now...
[runall-live] ⚠️  WARM START (WHAT-IF SCENARIO): Simulating historical trades from market open (09:30) to current time (12:00) (step 15m)
[runall-live] [Warm progress: 75%] Processed up to 11:45
[runall-live] [WARM START COMPLETE] Portfolio equity = 954,217 | Return -4.58% at 12:00
[runall-live] [LIVE] Switching to live mode...
[runall-live] [LIVE] Waiting for next 15m batch at 12:15...
```

### Cold-Start Mode
```
[MODE: LIVE] [COLD START] [runall-live] Starting live paper trading session for ['OGDC', 'HBL']
[runall-live] [COLD START] Starting fresh from current time with live data only. Cash: 1,000,000 | Equity: 1,000,000
[runall-live] [LIVE] Waiting for next 15m batch at 12:15...
```

## API Parameters

### `warm_start` (bool)
- `True` (default): Enable warm-start mode - replay today's data first
- `False`: Cold-start mode - begin fresh from current time

### `user_id` (str, optional)
Specify user ID for account persistence. Useful for running multiple paper trading sessions.

### `initial_cash` (float, optional)
Set initial cash amount. Only applies on first run.

## Examples

### Developer Replay (Warm-Start)
```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")
bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC", "HBL"],
    warm_start=True  # Replay today's data first
)
```

### True User Session (Cold-Start)
```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")
bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC", "HBL"],
    warm_start=False  # Start fresh from current time
)
```

### Multiple User Sessions
```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")

# User 1
bot1 = client.start_bot(
    strategy="sma",
    symbols=["OGDC"],
    user_id="user1"
)

# User 2 (separate account)
bot2 = client.start_bot(
    strategy="sma",
    symbols=["OGDC"],
    user_id="user2"
)
```

## How It Works

### Warm-Start Flow
1. Load existing account state (if available)
2. Replay historical 15-minute batches from market open to now
3. Execute simulated trades during replay
4. Save account state after warm-start completes
5. Switch to live mode and continue with real-time data

### Cold-Start Flow
1. Load existing account state (if available, or use defaults)
2. Skip historical replay
3. Begin live trading immediately from current time
4. Save account state after each live cycle

## Notes

- Account state persists across sessions - your portfolio carries over day-to-day
- Warm-start trades are simulated "what-if" scenarios - they don't represent actual execution
- Cold-start mode matches real-world scenarios where you start trading mid-day
- Account state is automatically managed by the backend
- Each bot maintains its own portfolio state

