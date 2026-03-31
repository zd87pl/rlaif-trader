# Pipeline Stabilization Implementation Plan

> For Hermes: use the writing-plans and systematic-debugging approach. Prioritize one runnable golden path over feature expansion.

Goal: make the repo's main orchestration path coherent enough to initialize, report status, and execute a mocked analysis flow without crashing.

Architecture: keep the existing module structure, but harden TradingPipeline so it adapts to real component APIs and degrades gracefully when optional dependencies or credentials are missing. Add a thin verification test suite around the pipeline golden path instead of trying to complete the whole trading system.

Tech Stack: Python, pytest, monkeypatch, existing src/ package modules.

---

## Scope

This execution pass focuses on Phase 1 stabilization only:
- fix pipeline/component API mismatches
- export missing config helpers
- make initialization resilient to missing optional dependencies
- add smoke tests for status and analyze

Explicitly out of scope for this pass:
- real broker execution
- full backtesting correctness
- full RLAIF training loop
- doc-wide cleanup

---

## Task 1: Export config helpers consistently

Objective: make `src.utils` export the helpers already implemented in `src/utils/config.py`.

Files:
- Modify: `src/utils/__init__.py`

Steps:
1. Export `get_settings` from `src.utils`.
2. Keep existing exports intact.
3. Verify imports used by `deployment/api/main.py` resolve.

Verification:
- `python3 - <<'PY'\nfrom src.utils import get_settings\nprint(callable(get_settings))\nPY`

## Task 2: Harden TradingPipeline initialization

Objective: stop pipeline construction from crashing when optional components are unavailable.

Files:
- Modify: `main.py`

Steps:
1. Initialize logging in a safe order.
2. Add helper logic for optional component initialization.
3. Adapt `_init_data` to real constructor signatures.
4. Adapt `_init_features` to real constructor signatures.
5. Make `_init_foundation_model`, `_init_agents`, `_init_options`, `_init_execution`, `_init_rlaif`, and `_init_scheduler` degrade gracefully instead of hard-failing.
6. Ensure live broker construction uses credentials/settings when required; paper mode must work without external credentials.

Verification:
- `python3 -m compileall main.py src`

## Task 3: Fix the pipeline golden-path method mismatches

Objective: make `TradingPipeline.status()` and `TradingPipeline.analyze()` operate against the actual component APIs.

Files:
- Modify: `main.py`

Steps:
1. Add helper methods for:
   - fetching price data
   - preprocessing with the actual preprocessor API
   - computing technical, sentiment, and fundamental features with graceful fallback
2. Replace calls to non-existent methods (`get_bars`, `process`, `compute`, `get_chain`, etc.) with compatible logic.
3. For unavailable upstream data (news/fundamentals), return explicit placeholder payloads rather than crashing.
4. Keep `status()` truthful about what is loaded vs unavailable.
5. Ensure `analyze_options()` tolerates differing options module APIs and unavailable data sources.

Verification:
- a mocked pipeline analyze path returns a dict with symbol, features, agents, manager_decision, and recommended_trades.

## Task 4: Add regression smoke tests for the pipeline

Objective: protect the stabilization work with a minimal but real test suite.

Files:
- Create: `tests/test_pipeline.py`

Steps:
1. Add a status smoke test using monkeypatched init methods to avoid heavy dependencies.
2. Add an analyze smoke test using fake data client, fake preprocessor, fake feature engines, fake manager, and fake options analyst.
3. Keep tests fast and dependency-light.

Verification:
- `pytest tests/test_pipeline.py -q`

## Task 5: Run focused verification and capture remaining gaps

Objective: validate the pass and identify the next engineering frontier.

Files:
- No required file changes unless minor follow-up is needed.

Steps:
1. Run compile checks.
2. Run the new smoke tests.
3. If failures appear, fix root cause only.
4. Summarize what now works and what still needs the next pass.

Verification:
- `python3 -m compileall main.py src`
- `pytest tests/test_pipeline.py -q`
