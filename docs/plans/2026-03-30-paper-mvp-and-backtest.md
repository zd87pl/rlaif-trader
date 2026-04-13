# Paper Trading MVP and Backtest Cleanup Plan

Goal: make the paper-trading path internally coherent enough to simulate signal execution and provide minimally useful backtest metrics.

Architecture: keep the existing broker / OMS / scheduler / pipeline layout, but add compatibility shims and missing methods so the core loop can operate in paper mode without real external services.

Tech Stack: Python, pytest, existing execution modules, monkeypatch-based smoke tests.

## Task 1: Fix execution-layer API mismatches
- main.py should initialize PaperBroker and TradingScheduler with the constructor signatures they actually expose.
- Add missing convenience methods needed by the pipeline and CLI status path.

## Task 2: Complete paper broker + OMS compatibility surface
- Add PaperBroker helpers like is_connected and flatten_all.
- Add OMS helpers used by scheduler and status/reporting paths.

## Task 3: Add risk engine compatibility helpers
- Add lightweight status and check_risk-style adapter methods so scheduler and CLI can call into the existing validation model.

## Task 4: Stabilize scheduler paper path
- Add start() wrapper.
- Make pre_market_analysis call pipeline.analyze() when available.
- Make execute_signals route through OMS.execute_signal() when not in dry-run mode.
- Make monitor_positions use available OMS state methods.

## Task 5: Improve backtest output
- Add minimal metrics derived from generated trade decisions.
- Ensure backtest returns a stable summary even when real data is unavailable.

## Task 6: Protect with smoke tests
- Add tests for scheduler paper execution and backtest summary behavior.
