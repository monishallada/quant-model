"""Gate 2 diagnostic: decompose signal edge vs fill-cost drag.

Reruns Engine A and Engine B under the trend signal over the full range with
ZERO-COST fills (mid fills, no slippage). Comparing to the Gate 2 runs (60%
toward worse side + 2% slippage per leg) separates "the directional signal
has no edge" from "the edge exists but transaction costs eat it".
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from catalyst.core.config import load_config
from catalyst.backtest.backtester import Backtester
from catalyst.risk.gate import FixedFractionalGate
from catalyst.runners.backtest_runner import format_result
from catalyst.runners.gate2_runner import build_engine, build_signal, load_catalysts
from catalyst.data.alpaca_history import AlpacaDailyBars
from catalyst.data.cache import ParquetCache
from catalyst.data.iv_history import IVRankProvider
from catalyst.data.thetadata_client import ThetaDataClient
from catalyst.data.thetadata_historical import ThetaDataHistorical

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

cfg = load_config("backtest", overrides={
    "execution.fill_model.spread_fill_fraction": 0.0,
    "execution.fill_model.slippage_pct_of_premium": 0.0,
})
start = date.fromisoformat(cfg.backtest.start_date)
end = date.fromisoformat(cfg.backtest.end_date)

cache = ParquetCache(cfg.data.cache_dir)
client = ThetaDataClient(cfg.data.thetadata)
data = ThetaDataHistorical(cfg.data, client=client, cache=cache,
                           history_provider=AlpacaDailyBars(cfg.data.alpaca, cache))
catalysts = load_catalysts(cfg, cache, start, end)
iv_rank = IVRankProvider(cfg.data, client, cache,
                         lookback_days=cfg.engines.engine_a.iv_rank_lookback_days)
for sym in sorted({c.symbol for c in catalysts}):
    iv_rank.prepare(sym, start, end)

out = Path("results/gate2_diagnostic")
out.mkdir(parents=True, exist_ok=True)
for engine_name in ("engine_a", "engine_b"):
    bt = Backtester(
        cfg=cfg, data=data,
        strategies=[build_engine(engine_name, cfg, iv_rank)],
        signal=build_signal("trend", cfg),
        catalysts=catalysts,
        gate=FixedFractionalGate(),
        label=f"{engine_name}+trend-ZEROCOST-full",
    )
    result = bt.run(start, end)
    print(format_result(result))
    (out / f"{engine_name}+trend-zerocost.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2))
