"""Option symbol formatting: canonical OptionKey ⇄ broker/vendor formats.

Every broker adapter converts at its own boundary; nothing outside adapters
ever handles a broker-specific symbol string.

Formats:
- OSI 21-char (Schwab / thinkorswim): root left-justified space-padded to 6,
  then YYMMDD, C/P, strike x1000 zero-padded to 8. "AAPL  240419C00150000"
- Compact OSI (Alpaca / OCC tape): same without root padding. "AAPL240419C00150000"
"""

from __future__ import annotations

import re
from datetime import date

from catalyst.core.models import OptionKey, OptionRight

_COMPACT_RE = re.compile(r"^([A-Z.]{1,6})(\d{6})([CP])(\d{8})$")


def _strike_thousandths(strike: float) -> int:
    # round() guards float artifacts (e.g. 149.999999) before formatting
    return round(strike * 1000)


def to_osi(key: OptionKey, pad_root: bool = True) -> str:
    """Format as OSI. ``pad_root=True`` gives the Schwab 21-char form,
    False the Alpaca/OCC compact form."""
    root = key.underlying.upper()
    if pad_root:
        root = f"{root:<6}"
    return (
        f"{root}"
        f"{key.expiry.strftime('%y%m%d')}"
        f"{key.right.value}"
        f"{_strike_thousandths(key.strike):08d}"
    )


def to_alpaca_symbol(key: OptionKey) -> str:
    return to_osi(key, pad_root=False)


def to_schwab_symbol(key: OptionKey) -> str:
    return to_osi(key, pad_root=True)


def parse_osi(symbol: str) -> OptionKey:
    """Parse either OSI form (padded or compact) back to the canonical key."""
    compact = symbol.replace(" ", "")
    m = _COMPACT_RE.match(compact)
    if not m:
        raise ValueError(f"Unparseable OSI option symbol: {symbol!r}")
    root, ymd, right, strike_str = m.groups()
    yy, mm, dd = int(ymd[0:2]), int(ymd[2:4]), int(ymd[4:6])
    return OptionKey(
        underlying=root,
        expiry=date(2000 + yy, mm, dd),
        right=OptionRight(right),
        strike=int(strike_str) / 1000.0,
    )
