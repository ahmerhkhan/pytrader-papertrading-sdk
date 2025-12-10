# Testing Telemetry Fix

## Changes Made

### 1. Enhanced Error Logging (`pytrader/telemetry.py`)
- Added detailed HTTP error messages with response body
- Added timeout error handling
- Errors now show which endpoint failed and why

### 2. Verbose Telemetry Logging (`pytrader/trader_core/execution/telemetry.py`)
- Added success message (logs 10% of successful pushes to avoid spam)
- Added detailed error printing with stack traces
- Errors now print to console with clear formatting

## How to Test

1. **Restart your bot:**
   ```bash
   python buying_down_strategy_v2.py
   ```

2. **Watch for telemetry messages:**
   - Success: `[buying-down-bot] ✓ Telemetry pushed successfully`
   - Errors: Will show detailed stack trace with endpoint and error details

3. **Check web app after 1-2 cycles:**
   - Bot details page should show equity curve
   - Positions should appear
   - Logs should be visible

## Expected Output

### If Working:
```
2025-12-10 14:30:00 | INFO     | ✅ Cycle complete
[buying-down-bot] ✓ Telemetry pushed successfully
```

### If Failing:
```
================================================================================
TELEMETRY ERROR: HTTP 500: <error details>
================================================================================
<stack trace>
================================================================================
```

## Common Issues & Fixes

### Issue 1: Foreign Key Constraint
**Error:** `foreign key constraint fails (token, bot_id) REFERENCES bots`
**Fix:** Already fixed - `/log_event` now auto-creates bot

### Issue 2: CHECK Constraint  
**Error:** `CHECK constraint failed: level`
**Fix:** Already fixed - levels converted to uppercase

### Issue 3: Timeout
**Error:** `Timeout calling /portfolio_update`
**Fix:** Increase timeout or check backend health

### Issue 4: Invalid Positions Data
**Error:** `JSON serialization error`
**Fix:** Check that positions don't contain non-serializable objects

## Next Steps

1. Run the bot and watch terminal for telemetry messages
2. If you see errors, copy the full error message
3. Check backend Render logs for server-side errors
4. Verify Supabase connection is working

## Debugging Commands

```bash
# Check if backend is up
curl -H "X-PyTrader-Token: dev-token" https://pytrader-backend.onrender.com/health

# Check if bot is registered
curl -H "X-PyTrader-Token: dev-token" https://pytrader-backend.onrender.com/bots

# Check bot portfolio
curl -H "X-PyTrader-Token: dev-token" https://pytrader-backend.onrender.com/portfolio/buying-down-bot
```

