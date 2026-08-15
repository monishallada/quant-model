# Working in this repository

## The workflow, end to end

You describe a strategy. It gets built, tested, and — only if the evidence
supports it — promoted to paper, then to live. Each step is gated by the step
before it, and the gates read evidence rather than intent.

```
  you describe it
        |
  [1] read the brief      design_brief          what has already been measured
  [2] scaffold            new_strategy          contract satisfied from line one
  [3] implement plan()                          the idea itself, nothing else
  [4] backtest            deploy --mode backtest
        |                 standard report: real + zero cost, 70/30 OOS,
        |                 N>100 flag, concentration check, verdict
        |                 CANDIDATE verdict  ->  validated  (written by pipeline)
  [5] paper               deploy --mode paper   typed `yes`, real Alpaca paper
        |                 a real session       ->  paper_tested (written by the session)
  [6] YOUR APPROVAL + Schwab keys
        |
  [7] live                deploy --mode live    validated AND paper_tested,
                                                code unchanged since validation,
                                                typed `LIVE`
```

**Promotion is earned, never declared.** `validated` is written only by the
pipeline, from the verdict its own report produced. `paper_tested` is written
only by an actual session against a real paper account — a `--dry-run` does not
grant it. Neither can be set from config, a CLI flag, or the strategy's own
source. The ledger lives in `results/<name>/status.json` with the evidence
trail, and it fingerprints the strategy's source: **edit the file after
validation and live eligibility is withdrawn automatically**, because the
evidence no longer describes the code that would run.

A failing re-run also withdraws a prior pass, so a strategy cannot be validated
once and then quietly changed.

### The commands

```bash
# what do we already know? (read this before designing anything)
uv run python -m catalyst.runners.design_brief
uv run python -m catalyst.runners.design_brief --tags options costs

# scaffold — registration is automatic, anything in active/ is discovered
uv run python -m catalyst.runners.new_strategy --name my_idea --cadence scheduled

# test — real + zero cost, 70/30 OOS, N>100, concentration, verdict
uv run python -m catalyst.runners.deploy_runner --mode backtest --strategy my_idea

# paper — typed confirmation; --dry-run reaches the gate and transmits nothing
uv run python -m catalyst.runners.deploy_runner --mode paper --strategy my_idea --dry-run
uv run python -m catalyst.runners.deploy_runner --mode paper --strategy my_idea

# archive when done — never deletes, stays runnable
uv run python -m catalyst.runners.archive_strategy --name my_idea --verdict "..."
```

### What I need from you to describe a strategy

The more of this you specify, the less I have to assume:

- **The bet.** What inefficiency or behaviour is it exploiting?
- **Instrument and structure.** Long calls/puts, spreads, shares.
- **Universe.** Which names, and why those.
- **Entry trigger.** Catalyst-driven, scheduled, or a cross-sectional rank.
- **Tenor and hold.** DTE at entry, how long it is held.
- **Exit.** Fixed date, profit target, stop — or deliberately none.
- **Sizing intent.** The RiskManager has final say, but tell me the fraction
  you have in mind so it can be configured rather than guessed.

If something is ambiguous I will pick the option consistent with the measured
evidence and say which assumption I made.

---

## Adding a strategy

Two steps. Nothing else in the repository changes.

**1. Drop one file in `src/catalyst/strategies/active/`** implementing the
`Strategy` interface:

```python
from catalyst.core.interfaces import Cadence, Opportunity, Strategy, StrategyContext
from catalyst.strategies.registry import StrategyMeta, register

class MyStrategy(Strategy):
    name = "my_strategy"
    cadence = Cadence.SCHEDULED          # CATALYST | SCHEDULED | DAILY

    def opportunities(self, session, ctx) -> list[Opportunity]:
        """What would you look at today? No chain access, no trade decision."""

    def plan(self, opp, ctx) -> ProposedTrade | None:
        """Unsized trade intent, or None. You do NOT choose quantity."""

register(StrategyMeta(name="my_strategy", module=__name__), build)
```

**2. Put its parameters in `config/base.yaml`.** No numbers in code.

Then run it:

```bash
uv run python -m catalyst.runners.deploy_runner --mode backtest --strategy my_strategy
```

### What you may not do

These are enforced by tests in `tests/core_invariants/`, not by convention:

- **No strategy may import `catalyst.risk`, `catalyst.costs`, `catalyst.brokers`
  or `catalyst.execution`.** If you cannot import the risk layer you cannot
  bypass it.
- **No strategy may call `place_order`.** There is exactly one path to a
  broker, and it runs through `execution/engine.py`.
- **No strategy chooses its own quantity.** `ProposedTrade` has no quantity
  field. The `RiskManager` sizes every position.
- **No strategy prices its own fill.** Return intent; `costs/` prices it.

A strategy that needs one of these is not blocked by policy — it is telling you
the pipeline is missing a feature. Add it to the pipeline, where every strategy
gets it.

### Cadence

| Cadence | Opportunities come from | Example |
|---|---|---|
| `CATALYST` | screener catalyst hits | Engines A–D |
| `SCHEDULED` | every N sessions | v9 long options, v8 index VRP |
| `DAILY` | one cross-sectional slot per session | v5 alpha, pairs |

Catalyst engines should subclass `CatalystStrategy`, which preserves the
original `evaluate(catalyst, chain, signal, as_of)` shape.

---

## Archiving

One command. It never deletes.

```bash
uv run python -m catalyst.runners.archive_strategy \
    --name long_options --verdict "no alpha; leveraged beta only"
```

It moves the module to `strategies/archive/<name>/`, moves results to
`results/archive/<name>/`, writes a `strategy.json` stub (date tested, verdict,
key metrics, the Engine C baseline it was measured against), and drops it from
active runs.

**Archived strategies stay runnable through the current pipeline.** That is the
difference between an archive and a graveyard: when the pipeline improves, old
results can be regenerated and compared apples-to-apples instead of trusted
from a stale artifact.

---

## What the shared pipeline guarantees

Every strategy travels one fixed path:

```
screener → strategy → RiskManager → cost model → exits → metrics → report
```

There is no flag, config key, or argument that removes a stage. The pipeline
calls the strategy, never the reverse.

Every run produces the same report, and no strategy can decline a check:

| Check | Why it exists |
|---|---|
| Average monthly return, first | The project's headline convention |
| Real **and** zero-cost runs | Separates "no signal" from "signal eaten by friction" — the distinction that decided v3 and v4 |
| 70/30 chronological OOS split | In-sample numbers here have been wrong often enough to be worthless alone |
| N > 100 flag | Engine C looked like +$125/trade at N=85 and was gross-negative at N=926 |
| Top-3 concentration | A "winning" strategy resting on three trades is a lottery ticket |
| Verdict vs Engine C baseline (~1%/yr) | Ranked against the archived reference, not against hope |

The verdict is computed on the **out-of-sample** segment. Ranking on training
data is how a curve-fit gets promoted.

### Core invariants

`tests/core_invariants/` property-tests the guarantees every future strategy
inherits for free:

1. The RiskManager cash floor is never breached, under randomized order streams.
2. No fill is ever better than NBBO — asserted in code, not assumed.
3. The out-of-sample split never leaks.
4. There is exactly one code path to a broker.

---

## Multi-engine backtesting

Every backtest runs on **all available engines**, and the report shows each
engine's numbers side by side.

| Engine | Status | What it independently verifies |
|---|---|---|
| `native` | always | Reference. Owns screener, RiskManager sizing, NBBO cost model, exits, cash ledger. |
| `nautilus` | ready | NautilusTrader recomputes position P&L from the same fills using its own Rust/Python accounting. Cross-checks the **ledger**, not the signal. No Docker needed. |
| `lean` | needs Docker | QuantConnect LEAN: independent event loop with corporate actions (**splits**), early assignment/exercise, dividends, margin — the two real gaps in the native engine. |

### Why two engines, precisely

**Divergence is the finding.** Every wrong number this project has produced came
from accounting or data handling — the v8 credit double-count (+4.01%/mo
reported against a true +0.07%), the v9 split mismatch, the row-count time base.
Each was caught only because someone happened to notice an internal
inconsistency. A second engine computing the same P&L a different way turns
that from luck into a check.

**Agreement is weak evidence.** It means neither implementation miscomputed P&L
from the same fills. It does not validate the signal, the data, or whether those
fills were achievable. Two engines agreeing on a wrong input agree perfectly.

**A divergence downgrades the verdict to `HOLD`** — a number two engines
disagree about is not a result, so the report refuses to rank it.

The sharpest check is **final equity**: two engines summing the same fills must
land on the same dollar. Monthly return and max drawdown may differ slightly by
marking *frequency* (native marks daily and sees intra-hold swings; a fill-replay
engine steps at exits), which is curve shape rather than arithmetic.

An unavailable engine is **shown as unavailable**, never omitted — otherwise
"agreed" and "never ran" look identical on the page.

### Enabling LEAN

```bash
open -a Docker                                        # start the daemon
uv run python -m catalyst.runners.lean_setup --convert  # ThetaData -> LEAN format
```

The converter uses the **same ThetaData key and the same snapshots** the native
engine consumes — one subscription, one source of truth for prices. That is what
makes a divergence attributable to engine behaviour rather than to two different
price feeds.


## Deployment

`--mode` selects **which DataSource and Broker are injected, and nothing else.**
Strategy, risk, costs and exits are byte-for-byte identical in all three modes.

| Mode | Data | Broker | Gate |
|---|---|---|---|
| `backtest` | ThetaData historical | SimulatedBroker | none — cannot reach an account |
| `paper` | live | Alpaca **paper** endpoint | pre-flight + typed `yes` |
| `live` | live | Schwab (real money) | validated **and** paper-tested + typed `LIVE` |

```bash
# safe: reaches the gate, transmits nothing
uv run python -m catalyst.runners.deploy_runner --mode paper --strategy long_options --dry-run
```

**Live is hard-gated.** A strategy that is not marked `validated` *and*
`paper_tested` is refused before a broker is even constructed — so a missing
credential can never mask the real reason. `--yes` is not honoured in live
mode; real money requires an interactive typed confirmation.

### Moving to Schwab

Switching from Alpaca paper to Schwab live is a **config change, not a code
change**. Add to `.env`:

```
SCHWAB_CLIENT_ID=...
SCHWAB_CLIENT_SECRET=...
SCHWAB_REFRESH_TOKEN=...
SCHWAB_ACCOUNT_HASH=...
```

`SchwabBroker` implements the same `Broker` interface, so strategy, risk, cost
and exit code never learn which broker they are talking to. OAuth refresh, the
21-character space-padded OSI symbol format and the Schwab order payload are
all contained inside `brokers/schwab.py`.

Note: Schwab refresh tokens expire roughly every 7 days and must be re-issued.

### Safety in every mode

- **Kill switch** — `touch .catalyst_kill` halts new entries on the next cycle.
  It is a file on disk on purpose: stopping a running system must not require
  the running system to cooperate. Exits are never blocked; abandoning open
  risk is a different kind of accident.
- **Reconciliation** — broker state is read before every action. Internal
  memory is never trusted.
- **The RiskManager is authoritative in every mode.** Paper is not a lighter
  mode; a safety layer that only runs in production rots before it is needed.

---

## Before you commit

```bash
uv run pytest -q
```

If a core invariant test fails, do not adjust the test. It is describing the
reason to trust every number in this repository.
