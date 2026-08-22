"""SEC EDGAR Form 4 insider-transaction adapter (backend-side).

This module lives under ``edge/data/`` and is therefore BEHIND the loader
wall: research code (signals/features/regime/research/runners) must never
import it directly — the AST gateway scan enforces that. Research reaches
this data only through :class:`edge.data.loader.EdgeDataLoader`.

POINT-IN-TIME CONVENTION (non-negotiable, shared by every edge adapter)
-----------------------------------------------------------------------
Every output frame carries two columns:

- ``asof_date`` — the data's OWN date (for a Form 4 row, the transaction
  date; for an index/submissions row, the EDGAR filing date).
- ``available_at`` — the tz-aware America/New_York datetime at which a live
  trader could FIRST have seen the datum. All downstream feature joins key
  on ``available_at <= decision_ts``.

Publication schedule encoded here:

- A filing's ``available_at`` is its EDGAR ``acceptanceDateTime`` (an ET
  wall-clock instant) **plus 5 minutes** of dissemination lag
  (``EdgarConfig.availability_lag_minutes``). EDGAR accepts Section 16
  filings until 22:00 ET, so a Form 4 accepted 20:30 ET is available the
  SAME evening at 20:35 ET, and one accepted 10:05 ET is available intraday
  at 10:10 ET.
- The quarterly ``form.idx`` files carry no acceptance times, so rows
  parsed from them get the CONSERVATIVE placeholder ``available_at`` of
  ``index_available_hour_et`` (default 06:00 ET) on the day AFTER the
  filing date — later than any real acceptance time. The placeholder is
  replaced by the true acceptance-based time once the filing itself is
  fetched (:meth:`EdgarForm4Client.fetch_filing`).
- Note a subtlety the acceptance time already absorbs: a transaction
  executed on day T may be FILED days later (Form 4s are due within two
  business days, and late filings happen). ``asof_date`` stays the
  transaction date; ``available_at`` stays the filing's acceptance time —
  so a late-filed transaction is correctly invisible until it was actually
  public.

Data paths
----------
- (a) :func:`parse_ownership_xml` — the ``ownershipDocument`` XML embedded
  in every Form 3/4/5 (schema per the real public AAPL Form 4 filings).
- (b) Backfill — quarterly form index files
  (``.../edgar/full-index/{year}/QTR{q}/form.idx``), fixed-width or
  pipe-delimited; :func:`backfill_quarters` plans the quarters,
  :func:`parse_form_index` parses one file, and
  :meth:`EdgarForm4Client.backfill_filings` orchestrates the whole
  index->filings run (cache-first, so the cache is the resume checkpoint).
- (c) Incremental — ``https://data.sec.gov/submissions/CIK{10digit}.json``
  parallel arrays; :func:`parse_submissions_json`.

Fetch discipline: every request carries the mandatory SEC ``User-Agent``
(``EdgarConfig.user_agent``) and is throttled to at most
``max_requests_per_second`` (default 8, SEC's published fair-access limit).
Immutable payloads (accepted filings, completed-quarter indexes) are cached
as parquet under ``<cache_root>`` (``data_cache/edge/edgar_form4/`` in
production), cache-first; the submissions JSON mutates daily and is never
cached. Tests drive the adapter with canned fixtures and an injected HTTP
getter — NO network in any test code path.
"""

from __future__ import annotations

import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from edge.core.events import MARKET_TZ

logger = logging.getLogger(__name__)

#: Source name — cache lives under ``data_cache/edge/<SOURCE_NAME>/``.
SOURCE_NAME = "edgar_form4"

#: Transaction codes with a defined open-market sign for the dollar-flow
#: feature. P = open-market purchase (+), S = open-market sale (−). Every
#: other code (A = award/grant, F = tax withholding, G = gift, M = option
#: exercise, ...) is not an open-market conviction trade and contributes 0.
CODE_PURCHASE = "P"
CODE_SALE = "S"

#: Columns of a transactions frame (plus the two point-in-time columns).
TRANSACTION_COLUMNS = [
    "symbol",
    "issuer_cik",
    "insider_name",
    "insider_cik",
    "is_director",
    "is_officer",
    "officer_title",
    "is_ten_pct_owner",
    "security_title",
    "transaction_code",
    "shares",
    "price",
    "shares_owned_after",
    "direct",
    "asof_date",
    "available_at",
]


class EdgarConfig(BaseModel):
    """Adapter knobs. Defaults encode SEC's published access rules; nothing
    here is buried in code paths, and everything is overridable per env."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: SEC REQUIRES a descriptive User-Agent with a contact address; requests
    #: without one are rejected (403). Sent on every request, no exceptions.
    user_agent: str = "quant-research contact: sensordyme@gmail.com"
    #: SEC fair-access limit is 10 req/s; we throttle below it at 8.
    max_requests_per_second: float = Field(default=8.0, gt=0.0)
    #: Dissemination lag added to acceptanceDateTime -> available_at.
    availability_lag_minutes: int = Field(default=5, ge=0)
    #: Conservative placeholder hour (ET, day AFTER filing date) for rows
    #: sourced from quarterly indexes, which carry no acceptance times.
    index_available_hour_et: int = Field(default=6, ge=0, le=23)
    archives_base_url: str = "https://www.sec.gov/Archives/"
    full_index_base_url: str = "https://www.sec.gov/Archives/edgar/full-index"
    submissions_base_url: str = "https://data.sec.gov/submissions"
    #: SEC's ticker->CIK mapping file (mutable — refreshed daily, never cached).
    company_tickers_url: str = "https://www.sec.gov/files/company_tickers.json"
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_attempts: int = Field(default=5, gt=0)
    retry_backoff_seconds: float = Field(default=2.0, ge=0.0)

    @field_validator("user_agent")
    @classmethod
    def _require_contact(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("SEC requires a non-empty User-Agent with contact info")
        return value


class Form4Transaction(BaseModel):
    """One non-derivative transaction row from an ownershipDocument."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    issuer_cik: str
    insider_name: str
    insider_cik: str | None = None
    is_director: bool = False
    is_officer: bool = False
    officer_title: str | None = None
    is_ten_pct_owner: bool = False
    security_title: str
    transaction_date: date
    #: Single-letter SEC code: P, S, A, F, G, M, ...
    transaction_code: str
    shares: float = Field(ge=0.0)
    #: 0.0 when the filing omits price (common for awards / footnoted prices).
    price: float = Field(ge=0.0)
    shares_owned_after: float = Field(ge=0.0)
    #: True = direct ownership ("D"), False = indirect ("I").
    direct: bool = True


class Form4Filing(BaseModel):
    """A parsed full-submission ``.txt``: acceptance instant + transactions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accession_number: str | None = None
    acceptance_datetime: datetime
    transactions: tuple[Form4Transaction, ...]


class BackfillSummary(BaseModel):
    """Result of one :meth:`EdgarForm4Client.backfill_filings` run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: date
    end: date
    #: Filings served (cache or network) per requested symbol, counted by
    #: the ISSUER CIK the index row carried.
    filings_by_symbol: dict[str, int]
    #: Index filenames whose full-submission ``.txt`` failed to parse
    #: (logged loudly and skipped — never silently dropped).
    failed: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# (a) ownershipDocument XML
# ---------------------------------------------------------------------------


def _text(elem: ET.Element | None) -> str | None:
    """Text of ``elem`` or of its ``<value>`` child (EDGAR wraps most leaf
    values in <value> so footnoteId siblings can ride along)."""
    if elem is None:
        return None
    value = elem.find("value")
    node = value if value is not None else elem
    return node.text.strip() if node.text and node.text.strip() else None


def _flag(elem: ET.Element | None) -> bool:
    """EDGAR booleans appear as 1/0 or true/false across schema versions."""
    text = _text(elem)
    return text is not None and text.lower() in ("1", "true")


def _number(elem: ET.Element | None) -> float:
    """Numeric leaf; missing/footnote-only values parse as 0.0 (documented:
    awards routinely omit price — a missing price is not an error)."""
    text = _text(elem)
    return float(text) if text is not None else 0.0


def parse_ownership_xml(raw: bytes) -> list[Form4Transaction]:
    """Parse an EDGAR ``ownershipDocument`` (Form 3/4/5 XML) into transactions.

    Emits one :class:`Form4Transaction` per ``nonDerivativeTransaction``.
    The derivative table (options/RSU legs) is deliberately excluded from
    v1 — the dollar-flow feature is defined on open-market common-stock
    trades. Documents can list several ``reportingOwner`` blocks (joint
    filers); the FIRST is used, matching the dominant single-owner case.

    Raises:
        ValueError: the bytes are not a parseable ownershipDocument.
    """
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"unparseable ownershipDocument XML: {exc}") from exc
    if root.tag != "ownershipDocument":
        raise ValueError(f"expected <ownershipDocument> root, got <{root.tag}>")

    issuer = root.find("issuer")
    symbol = _text(issuer.find("issuerTradingSymbol")) if issuer is not None else None
    issuer_cik = _text(issuer.find("issuerCik")) if issuer is not None else None
    if not symbol or not issuer_cik:
        raise ValueError("ownershipDocument missing issuer symbol/CIK")

    owner = root.find("reportingOwner")
    if owner is None:
        raise ValueError("ownershipDocument missing reportingOwner")
    owner_id = owner.find("reportingOwnerId")
    name = _text(owner_id.find("rptOwnerName")) if owner_id is not None else None
    insider_cik = _text(owner_id.find("rptOwnerCik")) if owner_id is not None else None
    if not name:
        raise ValueError("reportingOwner missing rptOwnerName")
    rel = owner.find("reportingOwnerRelationship")
    is_director = _flag(rel.find("isDirector")) if rel is not None else False
    is_officer = _flag(rel.find("isOfficer")) if rel is not None else False
    officer_title = _text(rel.find("officerTitle")) if rel is not None else None
    is_ten_pct = _flag(rel.find("isTenPercentOwner")) if rel is not None else False

    transactions: list[Form4Transaction] = []
    for txn in root.iter("nonDerivativeTransaction"):
        coding = txn.find("transactionCoding")
        amounts = txn.find("transactionAmounts")
        post = txn.find("postTransactionAmounts")
        nature = txn.find("ownershipNature")
        code = _text(coding.find("transactionCode")) if coding is not None else None
        txn_date = _text(txn.find("transactionDate"))
        if code is None or txn_date is None:
            raise ValueError("nonDerivativeTransaction missing code or date")
        ownership = (
            _text(nature.find("directOrIndirectOwnership")) if nature is not None else None
        )
        transactions.append(
            Form4Transaction(
                symbol=symbol,
                issuer_cik=issuer_cik,
                insider_name=name,
                insider_cik=insider_cik,
                is_director=is_director,
                is_officer=is_officer,
                officer_title=officer_title,
                is_ten_pct_owner=is_ten_pct,
                security_title=_text(txn.find("securityTitle")) or "",
                transaction_date=date.fromisoformat(txn_date),
                transaction_code=code,
                shares=_number(amounts.find("transactionShares") if amounts is not None else None),
                price=_number(
                    amounts.find("transactionPricePerShare") if amounts is not None else None
                ),
                shares_owned_after=_number(
                    post.find("sharesOwnedFollowingTransaction") if post is not None else None
                ),
                direct=(ownership or "D").upper() != "I",
            )
        )
    return transactions


# ---------------------------------------------------------------------------
# Availability (the point-in-time clock of this source)
# ---------------------------------------------------------------------------


def parse_acceptance_datetime(value: str) -> datetime:
    """Parse an EDGAR acceptance instant into a tz-aware ET datetime.

    Two wire formats exist:

    - SGML header ``<ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS`` — an ET wall-clock
      time by EDGAR specification; the ET zone is attached explicitly.
    - ISO-8601 from the submissions JSON (offset or ``Z`` suffix) —
      converted to ET. A NAIVE ISO string is treated as ET wall-clock (the
      same convention as the SGML header — EDGAR's operational timezone).
    """
    value = value.strip()
    if re.fullmatch(r"\d{14}", value):
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=MARKET_TZ)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=MARKET_TZ)
    return parsed.astimezone(MARKET_TZ)


def available_at_from_acceptance(
    acceptance: datetime, lag_minutes: int = EdgarConfig().availability_lag_minutes
) -> datetime:
    """``acceptanceDateTime`` (ET) + dissemination lag -> ``available_at``.

    A filing accepted 20:30 ET is available the same evening at 20:35 ET;
    one accepted 10:05 ET is available intraday at 10:10 ET.

    Raises:
        ValueError: ``acceptance`` is naive — ambiguity is never guessed
            here; parse wire strings with :func:`parse_acceptance_datetime`,
            which documents its naive-input convention.
    """
    if acceptance.tzinfo is None or acceptance.utcoffset() is None:
        raise ValueError("acceptance must be tz-aware; use parse_acceptance_datetime")
    return acceptance.astimezone(MARKET_TZ) + timedelta(minutes=lag_minutes)


def _index_placeholder_available_at(filed: date, hour_et: int) -> datetime:
    """Conservative available_at for index rows lacking acceptance times:
    early morning ET on the day AFTER the filing date — later than any real
    same-day acceptance (Section 16 cutoff is 22:00 ET)."""
    return datetime.combine(filed + timedelta(days=1), dtime(hour=hour_et), tzinfo=MARKET_TZ)


# ---------------------------------------------------------------------------
# (b) Backfill: quarterly form index files
# ---------------------------------------------------------------------------


def backfill_quarters(start: date, end: date) -> list[tuple[int, int]]:
    """The backfill plan: every ``(year, quarter)`` overlapping inclusive
    ``[start, end]``, in chronological order."""
    if start > end:
        raise ValueError(f"empty range: start {start} is after end {end}")
    quarters: list[tuple[int, int]] = []
    year, quarter = start.year, (start.month - 1) // 3 + 1
    while (year, quarter) <= (end.year, (end.month - 1) // 3 + 1):
        quarters.append((year, quarter))
        quarter += 1
        if quarter == 5:
            year, quarter = year + 1, 1
    return quarters


def backfill_index_urls(
    start: date, end: date, base_url: str = EdgarConfig().full_index_base_url
) -> list[str]:
    """The backfill plan as URLs: one ``form.idx`` URL per quarter overlapping
    inclusive ``[start, end]``, in chronological order."""
    return [
        f"{base_url.rstrip('/')}/{year}/QTR{quarter}/form.idx"
        for year, quarter in backfill_quarters(start, end)
    ]


def parse_company_tickers(payload: dict[str, Any]) -> dict[str, int]:
    """Parse SEC's ``company_tickers.json`` (``{"0": {"cik_str": 320193,
    "ticker": "AAPL", ...}, ...}``) into an upper-cased ``ticker -> cik``
    map. Duplicate tickers keep the FIRST (SEC orders by market cap, so the
    primary listing wins)."""
    out: dict[str, int] = {}
    for row in payload.values():
        ticker = str(row.get("ticker", "")).strip().upper()
        if ticker and ticker not in out:
            out[ticker] = int(row["cik_str"])
    return out


#: One fixed-width ``form.idx`` data row, anchored from the RIGHT: the CIK
#: (digits), ISO Date Filed, and File Name columns are unambiguous, so the
#: form type / company split is whatever 2+-space runs remain on the left.
#: Header-label offsets are NOT trusted: real EDGAR files (e.g. 2019 QTR1)
#: pad the DATA columns wider than the header labels, so label-derived
#: bounds truncate fields mid-value.
_INDEX_ROW_RE = re.compile(
    r"^(?P<form>\S(?:.*?\S)?)\s{2,}"
    r"(?P<company>\S(?:.*?\S)?)\s{2,}"
    r"(?P<cik>\d+)\s{2,}"
    r"(?P<filed>\d{4}-\d{2}-\d{2})\s{2,}"
    r"(?P<filename>\S+)\s*$"
)


def parse_form_index(
    raw: bytes,
    forms: Sequence[str] = (("4",)),
    index_available_hour_et: int = EdgarConfig().index_available_hour_et,
) -> pd.DataFrame:
    """Parse one EDGAR quarterly ``form.idx`` into a filings frame.

    Handles BOTH on-disk formats: the classic fixed-width layout (each data
    row split by the right-anchored :data:`_INDEX_ROW_RE` — real files pad
    data columns wider than the header labels, so header-derived offsets
    are never trusted) and the pipe-delimited variant. Rows are filtered to
    ``forms`` (default: exactly ``"4"`` — amendments ``"4/A"`` are opt-in
    because a naive join would double-count the original).

    Returns columns: ``form_type, company, cik, date_filed, filename,
    asof_date, available_at``. ``asof_date`` is the filing date;
    ``available_at`` is the conservative next-morning placeholder documented
    in the module header (the true acceptance-based time replaces it when
    the filing is fetched).
    """
    text = raw.decode("latin-1")
    lines = text.splitlines()
    header_i = next(
        (
            i
            for i, line in enumerate(lines)
            if "Form Type" in line and "CIK" in line and "Date Filed" in line
        ),
        None,
    )
    if header_i is None:
        raise ValueError("form.idx header line ('Form Type ... CIK ... Date Filed') not found")
    header = lines[header_i]
    wanted = {f.strip() for f in forms}

    records: list[dict[str, Any]] = []
    unmatched = 0
    if "|" in header:
        names = [h.strip() for h in header.split("|")]
        for line in lines[header_i + 1 :]:
            if "|" not in line or set(line.strip()) <= {"-"}:
                continue
            parts = [p.strip() for p in line.split("|")]
            row = dict(zip(names, parts))
            records.append(row)
    else:
        for line in lines[header_i + 1 :]:
            stripped = line.strip()
            if not stripped or set(stripped) <= {"-"}:
                continue
            m = _INDEX_ROW_RE.match(line)
            if m is None:
                unmatched += 1
                continue
            records.append(
                {
                    "Form Type": m["form"],
                    "Company Name": m["company"],
                    "CIK": m["cik"],
                    "Date Filed": m["filed"],
                    "File Name": m["filename"],
                }
            )
    if unmatched:
        # Loud, and fatal when pervasive: a layout change must never
        # silently shrink the index.
        logger.warning("form.idx: %d non-blank rows did not match the row pattern", unmatched)
        if unmatched > max(10, len(records) // 100):
            raise ValueError(
                f"form.idx layout drift: {unmatched} unmatched rows vs "
                f"{len(records)} parsed — refusing a silently-shrunken index"
            )

    rows: list[dict[str, Any]] = []
    for row in records:
        form_type = row.get("Form Type", "")
        if form_type not in wanted:
            continue
        filed = date.fromisoformat(row["Date Filed"])
        rows.append(
            {
                "form_type": form_type,
                "company": row["Company Name"],
                "cik": int(row["CIK"]),
                "date_filed": filed,
                "filename": row.get("File Name") or row.get("Filename", ""),
                "asof_date": filed,
                "available_at": _index_placeholder_available_at(
                    filed, index_available_hour_et
                ),
            }
        )
    columns = ["form_type", "company", "cik", "date_filed", "filename", "asof_date", "available_at"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# (c) Incremental: data.sec.gov submissions JSON
# ---------------------------------------------------------------------------


def submissions_url(cik: int, base_url: str = EdgarConfig().submissions_base_url) -> str:
    """``https://data.sec.gov/submissions/CIK{10-digit-zero-padded}.json``."""
    return f"{base_url.rstrip('/')}/CIK{int(cik):010d}.json"


def parse_submissions_json(
    payload: dict[str, Any],
    forms: Sequence[str] = (("4",)),
    lag_minutes: int = EdgarConfig().availability_lag_minutes,
) -> pd.DataFrame:
    """Parse the parallel arrays of a submissions JSON into a filings frame.

    ``filings.recent`` holds equal-length arrays (``accessionNumber``,
    ``filingDate``, ``acceptanceDateTime``, ``form``, ``primaryDocument``,
    ...); row *i* across all arrays is one filing. Filtered to ``forms``.

    Returns columns: ``cik, symbol, accession_number, form, primary_document,
    acceptance_datetime (ET), asof_date`` (the EDGAR filing date) and
    ``available_at`` (acceptance + lag).
    """
    recent = payload.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    acceptances = recent.get("acceptanceDateTime", [])
    form_types = recent.get("form", [])
    primary_docs = recent.get("primaryDocument", [""] * len(accessions))
    lengths = {len(accessions), len(filing_dates), len(acceptances), len(form_types)}
    if len(lengths) > 1:
        raise ValueError(f"submissions parallel arrays have mismatched lengths: {lengths}")

    tickers = payload.get("tickers") or []
    symbol = tickers[0] if tickers else None
    cik = int(payload["cik"]) if "cik" in payload else None
    wanted = {f.strip() for f in forms}

    rows: list[dict[str, Any]] = []
    for i, form_type in enumerate(form_types):
        if form_type not in wanted:
            continue
        acceptance = parse_acceptance_datetime(acceptances[i])
        rows.append(
            {
                "cik": cik,
                "symbol": symbol,
                "accession_number": accessions[i],
                "form": form_type,
                "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                "acceptance_datetime": acceptance,
                "asof_date": date.fromisoformat(filing_dates[i]),
                "available_at": available_at_from_acceptance(acceptance, lag_minutes),
            }
        )
    columns = [
        "cik",
        "symbol",
        "accession_number",
        "form",
        "primary_document",
        "acceptance_datetime",
        "asof_date",
        "available_at",
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Full-submission .txt (backfill fetch target: header + embedded XML)
# ---------------------------------------------------------------------------

_ACCEPTANCE_RE = re.compile(rb"<ACCEPTANCE-DATETIME>\s*(\d{14})")
_ACCESSION_RE = re.compile(rb"ACCESSION NUMBER:\s*(\S+)")
_XML_RE = re.compile(rb"<XML>\s*(.*?)\s*</XML>", re.DOTALL)


def parse_filing_txt(raw: bytes) -> Form4Filing:
    """Parse a full-submission ``.txt`` (the file the quarterly index points
    at): the SGML header's ``<ACCEPTANCE-DATETIME>`` (ET) plus the embedded
    ``ownershipDocument`` XML between ``<XML>...</XML>``."""
    acceptance_m = _ACCEPTANCE_RE.search(raw)
    if acceptance_m is None:
        raise ValueError("filing txt missing <ACCEPTANCE-DATETIME> header")
    xml_m = _XML_RE.search(raw)
    if xml_m is None:
        raise ValueError("filing txt missing embedded <XML> ownershipDocument")
    accession_m = _ACCESSION_RE.search(raw)
    return Form4Filing(
        accession_number=accession_m.group(1).decode("ascii") if accession_m else None,
        acceptance_datetime=parse_acceptance_datetime(acceptance_m.group(1).decode("ascii")),
        transactions=tuple(parse_ownership_xml(xml_m.group(1))),
    )


def transactions_frame(
    transactions: Sequence[Form4Transaction],
    acceptance: datetime,
    lag_minutes: int = EdgarConfig().availability_lag_minutes,
) -> pd.DataFrame:
    """One filing's transactions as a point-in-time frame.

    ``asof_date`` is each transaction's own date; ``available_at`` is the
    FILING's acceptance + lag for every row (a late-filed transaction stays
    invisible until it was actually public).
    """
    available = available_at_from_acceptance(acceptance, lag_minutes)
    rows = []
    for txn in transactions:
        row = txn.model_dump()
        row["asof_date"] = row.pop("transaction_date")
        row["available_at"] = available
        rows.append(row)
    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)


# ---------------------------------------------------------------------------
# HTTP client: mandatory User-Agent, built-in throttle, parquet cache
# ---------------------------------------------------------------------------


class _Response(Protocol):
    status_code: int
    text: str

    @property
    def content(self) -> bytes: ...

    def json(self) -> Any: ...


@runtime_checkable
class HttpGetter(Protocol):
    """The one slice of ``httpx.Client`` used — tests inject a fake with
    canned bodies; production injects a real client. Headers are passed per
    request so the mandatory User-Agent is visible at every call site."""

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> _Response: ...


class EdgarForm4Client:
    """Throttled, cache-first EDGAR client for Form 4 data.

    Contract: every HTTP request carries ``config.user_agent`` and is spaced
    to at most ``config.max_requests_per_second``. Immutable payloads
    (accepted filings, quarter indexes) are parquet-cached under
    ``cache_root`` and served without touching HTTP on a hit; the mutable
    submissions JSON is never cached. ``sleep``/``monotonic`` are injectable
    so tests exercise the throttle without real waiting — and with fixture
    getters, without any network.
    """

    def __init__(
        self,
        client: HttpGetter,
        cache_root: str | Path,
        config: EdgarConfig | None = None,
        *,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        self._http = client
        self._cache_root = Path(cache_root)
        self._config = config or EdgarConfig()
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_allowed: float | None = None

    @classmethod
    def build(
        cls, cache_root: str | Path | None = None, config: EdgarConfig | None = None
    ) -> EdgarForm4Client:
        """Production constructor: real httpx client, default cache root
        ``<repo>/data_cache/edge/edgar_form4/``. Never used by tests."""
        import httpx

        config = config or EdgarConfig()
        if cache_root is None:
            repo_root = Path(__file__).resolve().parents[4]
            cache_root = repo_root / "data_cache" / "edge" / SOURCE_NAME
        client = httpx.Client(
            headers={"User-Agent": config.user_agent},
            timeout=config.request_timeout_seconds,
            follow_redirects=True,
        )
        return cls(client, cache_root, config)

    # -- fetch paths --------------------------------------------------------

    def fetch_submissions(self, cik: int, forms: Sequence[str] = (("4",))) -> pd.DataFrame:
        """Incremental path: today's submissions JSON for one CIK, filtered
        to Form 4 rows. NOT cached — the file mutates with every filing."""
        resp = self._get(submissions_url(cik, self._config.submissions_base_url))
        return parse_submissions_json(
            resp.json(), forms=forms, lag_minutes=self._config.availability_lag_minutes
        )

    def fetch_quarter_index(
        self, year: int, quarter: int, forms: Sequence[str] = (("4",))
    ) -> pd.DataFrame:
        """Backfill path: one quarter's ``form.idx``, filtered to Form 4.

        Cache-first per (year, quarter, forms): completed quarters are
        immutable. NOTE: only backfill completed quarters — the CURRENT
        quarter's index grows daily and would be frozen stale by this cache.
        """
        tag = "-".join(sorted(f.replace("/", "A") for f in forms))
        path = self._cache_root / "form_index" / f"{year}-QTR{quarter}-{tag}.parquet"
        cached = self._read_cache(path)
        if cached is not None:
            return cached
        url = f"{self._config.full_index_base_url.rstrip('/')}/{year}/QTR{quarter}/form.idx"
        frame = parse_form_index(
            self._get(url).content,
            forms=forms,
            index_available_hour_et=self._config.index_available_hour_et,
        )
        _write_atomic(path, frame)
        return frame

    def fetch_filing(self, filename: str) -> pd.DataFrame:
        """One filing's transactions frame, from an index ``filename`` (e.g.
        ``edgar/data/320193/0000320193-26-000012.txt``). Cache-first per
        accession — accepted filings never change."""
        accession = Path(filename).stem
        path = self._cache_root / "filings" / f"accession={accession}.parquet"
        cached = self._read_cache(path)
        if cached is not None:
            return cached
        url = self._config.archives_base_url.rstrip("/") + "/" + filename.lstrip("/")
        filing = parse_filing_txt(self._get(url).content)
        frame = transactions_frame(
            filing.transactions,
            filing.acceptance_datetime,
            lag_minutes=self._config.availability_lag_minutes,
        )
        _write_atomic(path, frame)
        return frame

    def fetch_company_ciks(self, tickers: Sequence[str]) -> dict[str, int]:
        """Resolve ``tickers`` to CIKs via SEC's ``company_tickers.json``.

        NOT cached — the mapping file mutates as listings change. Raises
        ``KeyError`` naming every unresolvable ticker: a silently-missing
        symbol would silently exclude an issuer from the backfill.
        """
        resp = self._get(self._config.company_tickers_url)
        mapping = parse_company_tickers(resp.json())
        wanted = {t.strip().upper() for t in tickers}
        missing = sorted(wanted - mapping.keys())
        if missing:
            raise KeyError(f"tickers not in SEC company_tickers.json: {missing}")
        return {t: mapping[t] for t in sorted(wanted)}

    def backfill_filings(
        self,
        cik_by_symbol: Mapping[str, int],
        start: date,
        end: date,
        forms: Sequence[str] = (("4",)),
        *,
        today: date | None = None,
        progress_every: int = 250,
    ) -> BackfillSummary:
        """Backfill: fetch every ``forms`` filing by the given issuers with
        ``date_filed`` in inclusive ``[start, end]`` into the parquet cache.

        Drives the module's own backfill path — quarter indexes via
        :meth:`fetch_quarter_index`, then each filing via
        :meth:`fetch_filing`. Both are cache-first per immutable payload, so
        THE CACHE IS THE CHECKPOINT: an interrupted run resumes exactly
        where it stopped when re-invoked with the same arguments, refetching
        nothing already on disk.

        Index rows are filtered to the ISSUER CIKs in ``cik_by_symbol``
        (a Form 4 appears in the index once per filer CIK — issuer and
        reporting owner(s) — so the issuer-CIK row uniquely identifies the
        filing) and deduplicated by accession. A filing whose ``.txt`` fails
        to parse is logged loudly, skipped, and reported in ``failed`` —
        one malformed document must not abort a multi-thousand-filing run,
        but it is never silently dropped. HTTP failures (retries exhausted)
        still raise — see :meth:`_get`.

        Raises:
            ValueError: ``start > end``, or ``end`` falls in the CURRENT
                (incomplete) quarter — its index grows daily and would be
                frozen stale by the cache (see :meth:`fetch_quarter_index`).
        """
        today = today or date.today()
        quarters = backfill_quarters(start, end)
        if quarters and quarters[-1] >= (today.year, (today.month - 1) // 3 + 1):
            raise ValueError(
                f"backfill end {end} falls in the current (incomplete) quarter "
                f"{quarters[-1]}; its form.idx would be cached stale"
            )
        symbol_by_cik = {int(cik): symbol for symbol, cik in cik_by_symbol.items()}
        counts: dict[str, int] = {symbol: 0 for symbol in cik_by_symbol}
        failed: list[str] = []
        seen_accessions: set[str] = set()
        planned: list[tuple[str, str]] = []  # (filename, symbol)
        for year, quarter in quarters:
            index = self.fetch_quarter_index(year, quarter, forms)
            mask = (
                index["cik"].isin(symbol_by_cik)
                & (index["date_filed"] >= start)
                & (index["date_filed"] <= end)
            )
            for row in index[mask].itertuples(index=False):
                accession = Path(row.filename).stem
                if accession in seen_accessions:
                    continue
                seen_accessions.add(accession)
                planned.append((row.filename, symbol_by_cik[int(row.cik)]))
        logger.info(
            "edgar backfill: %d filings planned across %d quarters (%s..%s)",
            len(planned), len(quarters), start, end,
        )
        for i, (filename, symbol) in enumerate(planned, start=1):
            try:
                self.fetch_filing(filename)
            except ValueError as exc:
                logger.warning("edgar backfill: unparseable filing %s: %s", filename, exc)
                failed.append(filename)
                continue
            counts[symbol] += 1
            if progress_every and i % progress_every == 0:
                logger.info("edgar backfill: %d/%d filings done", i, len(planned))
        return BackfillSummary(
            start=start, end=end, filings_by_symbol=counts, failed=tuple(failed)
        )

    # -- plumbing -----------------------------------------------------------

    def _read_cache(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 — unreadable cache is a miss
            logger.warning("unreadable cache %s: %s — refetching", path, exc)
            return None

    def _throttle(self) -> None:
        """Space requests to at most ``max_requests_per_second``: each send
        is scheduled no earlier than the previous send + 1/rate."""
        interval = 1.0 / self._config.max_requests_per_second
        now = self._monotonic()
        scheduled = now if self._next_allowed is None else max(now, self._next_allowed)
        if scheduled > now:
            self._sleep(scheduled - now)
        self._next_allowed = scheduled + interval

    def _get(self, url: str) -> _Response:
        """Throttled GET with the mandatory User-Agent; bounded retries on
        429/5xx; raises on failure — a silently-empty insider frame would
        poison every feature built on it, so failures are LOUD."""
        cfg = self._config
        headers = {"User-Agent": cfg.user_agent}
        last_status: int | None = None
        for attempt in range(cfg.max_attempts):
            self._throttle()
            resp = self._http.get(url, headers=headers)
            last_status = resp.status_code
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = cfg.retry_backoff_seconds * (2**attempt)
                logger.warning("EDGAR %s -> HTTP %d; retrying in %.0fs", url, resp.status_code, wait)
                self._sleep(wait)
                continue
            raise RuntimeError(f"EDGAR {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
        raise RuntimeError(f"EDGAR {url} exhausted retries (last HTTP {last_status})")


def _write_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Parquet via tmp + replace so a crash or concurrent writer can never
    leave a torn file at the final path (same discipline as alpaca_ticks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


#: Buyer-identity placeholder for transactions frames that carry NO insider
#: identity columns (``insider_cik``/``insider_name``): every officer buy
#: collapses onto this one identity, so the distinct-buyer count degrades to
#: a conservative 0-or-1 rather than overcounting.
_UNIDENTIFIED_BUYER = "<unidentified>"


def _buyer_identity(frame: pd.DataFrame) -> pd.Series:
    """One identity string per transaction row: ``insider_cik`` when present
    and non-blank, else ``insider_name``, else the shared unidentified
    placeholder. Preferring the CIK makes the distinct count robust to name
    spelling drift across filings ('SMITH JOHN' vs 'Smith John A')."""

    def column(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series([None] * len(frame), index=frame.index, dtype=object)
        text = frame[name].astype(object).astype(str).str.strip()
        blank = frame[name].isna() | (text == "") | (text.str.lower() == "none")
        return text.mask(blank)

    cik = column("insider_cik")
    name = column("insider_name")
    return cik.where(cik.notna(), name).fillna(_UNIDENTIFIED_BUYER)


def insider_features(transactions: pd.DataFrame, window_days: int = 21) -> pd.DataFrame:
    """Per-symbol, per-day insider features from a transactions frame.

    Input: any concatenation of :func:`transactions_frame` outputs (columns
    ``symbol, transaction_code, shares, price, is_officer, asof_date,
    available_at`` are required; ``insider_cik``/``insider_name`` are
    optional and feed the distinct-buyer count below).

    Output columns:

    - ``net_insider_dollars_{window_days}d`` — signed open-market dollars
      (+P shares*price, −S shares*price; every other code contributes 0 —
      awards, withholding, and exercises are not conviction trades) summed
      over the trailing ``window_days`` CALENDAR days ``(t-Nd, t]`` ending
      at each event day.
    - ``officer_buy`` — True on days where an OFFICER made an open-market
      purchase (code P) — the classic high-signal insider event.
    - ``officer_buyers_{window_days}d`` — DISTINCT officers with an
      open-market purchase (code P) inside the same trailing window,
      identified by ``insider_cik`` (falling back to ``insider_name``).
      Frames carrying neither identity column cannot distinguish buyers:
      every officer buy collapses onto one placeholder identity, so the
      count degrades to a conservative 0-or-1, never an overcount.
    - ``asof_date`` / ``available_at`` — rows exist only on event days;
      ``available_at`` is the MAX filing availability across every
      transaction inside the row's window (conservative: the windowed sum
      is not knowable until its LAST contributing filing was public, and a
      late-filed transaction drags the whole window's availability later).
    """
    value_col = f"net_insider_dollars_{window_days}d"
    buyers_col = f"officer_buyers_{window_days}d"
    out_columns = [
        "symbol",
        "asof_date",
        "available_at",
        value_col,
        "officer_buy",
        buyers_col,
    ]
    if transactions.empty:
        return pd.DataFrame(columns=out_columns)

    frame = transactions.copy()
    code = frame["transaction_code"].astype(str).str.upper()
    dollars = frame["shares"].astype(float) * frame["price"].astype(float)
    frame["signed_dollars"] = dollars.where(code == CODE_PURCHASE, 0.0) - dollars.where(
        code == CODE_SALE, 0.0
    )
    frame["officer_buy"] = frame["is_officer"].astype(bool) & (code == CODE_PURCHASE)
    # Identity only matters on officer-buy rows; everything else is masked out
    # so a director's or seller's identity never inflates the buyer count.
    frame["buyer_id"] = _buyer_identity(frame).where(frame["officer_buy"])

    out_frames: list[pd.DataFrame] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        daily = (
            group.groupby("asof_date", sort=True)
            .agg(
                signed_dollars=("signed_dollars", "sum"),
                officer_buy=("officer_buy", "any"),
                available_at=("available_at", "max"),
                buyer_ids=("buyer_id", lambda s: frozenset(s.dropna())),
            )
            .reset_index()
        )
        idx = pd.DatetimeIndex(pd.to_datetime(daily["asof_date"]))
        # Trailing window (t - window_days, t], matching pandas' closed='right'
        # time rolling — computed via searchsorted so the dollar sum, the
        # availability max, and the distinct-buyer union use the IDENTICAL
        # window (and datetimes never pass through lossy float64).
        window_starts = idx - pd.Timedelta(days=window_days)
        lows = idx.searchsorted(window_starts, side="right")
        net = [
            float(daily["signed_dollars"].iloc[lo : i + 1].sum())
            for i, lo in enumerate(lows)
        ]
        avail = [daily["available_at"].iloc[lo : i + 1].max() for i, lo in enumerate(lows)]
        buyers = [
            len(frozenset().union(*daily["buyer_ids"].iloc[lo : i + 1]))
            for i, lo in enumerate(lows)
        ]
        out_frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "asof_date": daily["asof_date"],
                    "available_at": avail,
                    value_col: net,
                    "officer_buy": daily["officer_buy"],
                    buyers_col: buyers,
                }
            )
        )
    return pd.concat(out_frames, ignore_index=True)[out_columns]
