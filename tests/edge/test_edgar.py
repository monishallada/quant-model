"""EDGAR Form 4 adapter tests: canned fixtures only, NO network.

Fixtures mirror the real public EDGAR wire formats — the ownershipDocument
XML schema of AAPL Form 4 filings, the quarterly ``form.idx`` layouts
(fixed-width and pipe), the ``data.sec.gov`` submissions parallel arrays,
and the full-submission ``.txt`` SGML wrapper. Insider names and figures
are synthetic. Every date predates the 2026-02-22 lockbox start.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from edge.data.feeds.edgar_form4 import (
    EdgarConfig,
    EdgarForm4Client,
    available_at_from_acceptance,
    backfill_index_urls,
    backfill_quarters,
    insider_features,
    parse_acceptance_datetime,
    parse_company_tickers,
    parse_filing_txt,
    parse_form_index,
    parse_ownership_xml,
    parse_submissions_json,
    submissions_url,
    transactions_frame,
)

ET_TZ = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Fixture: ownershipDocument XML (structure of the real AAPL Form 4 filings;
# schemaVersion X0508, <value> wrappers, footnoteId siblings, empty leaves).
# ---------------------------------------------------------------------------

OFFICER_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0508</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-02-06</periodOfReport>
    <notSubjectToSection16>0</notSubjectToSection16>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0009999901</rptOwnerCik>
            <rptOwnerName>DOE JANE A</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerAddress>
            <rptOwnerStreet1>ONE APPLE PARK WAY</rptOwnerStreet1>
            <rptOwnerCity>CUPERTINO</rptOwnerCity>
            <rptOwnerState>CA</rptOwnerState>
            <rptOwnerZipCode>95014</rptOwnerZipCode>
        </reportingOwnerAddress>
        <reportingOwnerRelationship>
            <isDirector>0</isDirector>
            <isOfficer>1</isOfficer>
            <isTenPercentOwner>0</isTenPercentOwner>
            <isOther>0</isOther>
            <officerTitle>Senior Vice President, CFO</officerTitle>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-02-06</value></transactionDate>
            <deemedExecutionDate></deemedExecutionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>S</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionTimeliness><value></value></transactionTimeliness>
            <transactionAmounts>
                <transactionShares><value>15000</value></transactionShares>
                <transactionPricePerShare><value>228.8701</value><footnoteId id="F1"/></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>110450</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-02-06</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>A</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>8000</value></transactionShares>
                <transactionPricePerShare><footnoteId id="F2"/></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>118450</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>I</value></directOrIndirectOwnership>
                <natureOfOwnership><value>By Trust</value></natureOfOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
    <footnotes>
        <footnote id="F1">Weighted average sale price; range 228.10 to 229.55.</footnote>
        <footnote id="F2">Restricted stock units granted under the 2022 Plan.</footnote>
    </footnotes>
    <ownerSignature>
        <signatureName>/s/ Example Attorney-in-Fact</signatureName>
        <signatureDate>2026-02-06</signatureDate>
    </ownerSignature>
</ownershipDocument>
"""

# Director purchase; booleans in the true/false style some filers emit.
DIRECTOR_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
    <schemaVersion>X0508</schemaVersion>
    <documentType>4</documentType>
    <periodOfReport>2026-02-10</periodOfReport>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>AAPL</issuerTradingSymbol>
    </issuer>
    <reportingOwner>
        <reportingOwnerId>
            <rptOwnerCik>0009999902</rptOwnerCik>
            <rptOwnerName>ROE RICHARD B</rptOwnerName>
        </reportingOwnerId>
        <reportingOwnerRelationship>
            <isDirector>true</isDirector>
            <isOfficer>false</isOfficer>
            <isTenPercentOwner>false</isTenPercentOwner>
            <isOther>false</isOther>
        </reportingOwnerRelationship>
    </reportingOwner>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <securityTitle><value>Common Stock</value></securityTitle>
            <transactionDate><value>2026-02-10</value></transactionDate>
            <transactionCoding>
                <transactionFormType>4</transactionFormType>
                <transactionCode>P</transactionCode>
                <equitySwapInvolved>0</equitySwapInvolved>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>5000</value></transactionShares>
                <transactionPricePerShare><value>214.25</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>62500</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
            <ownershipNature>
                <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
            </ownershipNature>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>
"""

# ---------------------------------------------------------------------------
# Fixture: quarterly form index (both layouts). Column offsets in the fixed-
# width variant come from the header labels, exactly as on EDGAR.
# ---------------------------------------------------------------------------

_IDX_ROWS = [
    ("10-K", "ACME MANUFACTURING CO", "1234567", "2026-01-15", "edgar/data/1234567/0001234567-26-000004.txt"),
    ("4", "APPLE INC", "320193", "2026-01-08", "edgar/data/320193/0000320193-26-000012.txt"),
    ("4/A", "APPLE INC", "320193", "2026-01-09", "edgar/data/320193/0000320193-26-000013.txt"),
    ("8-K", "BETA CORP", "7654321", "2026-01-20", "edgar/data/7654321/0007654321-26-000002.txt"),
    ("4", "BETA CORP", "7654321", "2026-02-11", "edgar/data/7654321/0007654321-26-000003.txt"),
]


def _fixed_line(form: str, company: str, cik: str, filed: str, filename: str) -> str:
    return form.ljust(12) + company.ljust(62) + cik.ljust(12) + filed.ljust(12) + filename


FORM_IDX_FIXED = (
    "Description:           Master Index of EDGAR Dissemination Feed by Form Type\n"
    "Last Data Received:    March 31, 2026\n"
    "Comments:              webmaster@sec.gov\n"
    "Anonymous FTP:         ftp://ftp.sec.gov/edgar/\n"
    "\n\n\n"
    + _fixed_line("Form Type", "Company Name", "CIK", "Date Filed", "File Name")
    + "\n"
    + "-" * 130
    + "\n"
    + "\n".join(_fixed_line(*row) for row in _IDX_ROWS)
    + "\n"
).encode("latin-1")

# The REAL 2019+ layout: DATA columns are padded WIDER than the header
# labels (form-type column is 17 wide, the header's label spacing implies
# 12). Header-derived offsets would truncate CIK/date mid-value — the bug
# the right-anchored row regex exists to prevent.
def _drifted_line(form: str, company: str, cik: str, filed: str, filename: str) -> str:
    return form.ljust(17) + company.ljust(62) + cik.ljust(12) + filed.ljust(12) + filename


FORM_IDX_FIXED_DRIFTED = (
    "Description:           Master Index of EDGAR Dissemination Feed by Form Type\n"
    "Last Data Received:    March 31, 2019\n"
    "\n\n\n"
    + _fixed_line("Form Type", "Company Name", "CIK", "Date Filed", "File Name")
    + "\n"
    + "-" * 141
    + "\n"
    + "\n".join(_drifted_line(*row) for row in _IDX_ROWS)
    + "\n"
).encode("latin-1")

FORM_IDX_PIPE = (
    "Description:           Master Index of EDGAR Dissemination Feed by Form Type\n"
    "Last Data Received:    March 31, 2026\n"
    "\n"
    "CIK|Company Name|Form Type|Date Filed|Filename\n"
    + "-" * 90
    + "\n"
    + "\n".join(f"{cik}|{company}|{form}|{filed}|{fn}" for form, company, cik, filed, fn in _IDX_ROWS)
    + "\n"
).encode("latin-1")

# ---------------------------------------------------------------------------
# Fixture: data.sec.gov submissions JSON (parallel arrays).
# Row 0: accepted 2026-02-11T01:30Z == 2026-02-10 20:30 ET (evening filing).
# Row 2: accepted 2026-02-06T15:05Z == 2026-02-06 10:05 ET (intraday filing).
# ---------------------------------------------------------------------------

SUBMISSIONS_PAYLOAD: dict[str, Any] = {
    "cik": "320193",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "exchanges": ["Nasdaq"],
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-26-000030",
                "0000320193-26-000029",
                "0000320193-26-000028",
            ],
            "filingDate": ["2026-02-10", "2026-02-09", "2026-02-06"],
            "acceptanceDateTime": [
                "2026-02-11T01:30:00.000Z",
                "2026-02-09T16:31:12.000-05:00",
                "2026-02-06T15:05:00.000Z",
            ],
            "form": ["4", "10-Q", "4"],
            "primaryDocument": [
                "xslF345X05/wk-form4_a.xml",
                "aapl-10q.htm",
                "xslF345X05/wk-form4_b.xml",
            ],
        }
    },
}

# ---------------------------------------------------------------------------
# Fixture: full-submission .txt (SGML header + embedded ownershipDocument).
# Accepted 20:30 ET on 2026-02-10 — the evening-availability case.
# ---------------------------------------------------------------------------

FILING_TXT = (
    b"-----BEGIN PRIVACY-ENHANCED MESSAGE-----\n"
    b"<SEC-DOCUMENT>0000320193-26-000012.txt : 20260210\n"
    b"<SEC-HEADER>0000320193-26-000012.hdr.sgml : 20260210\n"
    b"<ACCEPTANCE-DATETIME>20260210203000\n"
    b"ACCESSION NUMBER:\t\t0000320193-26-000012\n"
    b"CONFORMED SUBMISSION TYPE:\t4\n"
    b"PUBLIC DOCUMENT COUNT:\t\t1\n"
    b"<DOCUMENT>\n"
    b"<TYPE>4\n"
    b"<SEQUENCE>1\n"
    b"<FILENAME>wk-form4_1.xml\n"
    b"<DESCRIPTION>FORM 4\n"
    b"<TEXT>\n"
    b"<XML>\n" + DIRECTOR_XML + b"</XML>\n"
    b"</TEXT>\n"
    b"</DOCUMENT>\n"
    b"-----END PRIVACY-ENHANCED MESSAGE-----\n"
)


# ---------------------------------------------------------------------------
# Fake HTTP plumbing (no network anywhere).
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, content: bytes = b"", payload: Any = None, status_code: int = 200):
        self.status_code = status_code
        self._payload = payload
        self.content = content if content else json.dumps(payload).encode()
        self.text = self.content[:500].decode("latin-1", "replace")

    def json(self) -> Any:
        return self._payload if self._payload is not None else json.loads(self.content)


class FakeGetter:
    """Canned routes; records (url, headers) of every request."""

    def __init__(self, routes: dict[str, FakeResponse]):
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, *, headers: dict[str, str] | None = None) -> FakeResponse:
        self.calls.append((url, headers))
        return self.routes[url]


def _client(routes: dict[str, FakeResponse], tmp_path, sleeps=None) -> EdgarForm4Client:
    return EdgarForm4Client(
        FakeGetter(routes),
        cache_root=tmp_path,
        sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
        monotonic=lambda: 0.0,
    )


# ---------------------------------------------------------------------------
# (a) ownershipDocument parsing
# ---------------------------------------------------------------------------


def test_parse_officer_sale_and_award() -> None:
    txns = parse_ownership_xml(OFFICER_XML)
    assert len(txns) == 2
    sale, award = txns

    assert sale.symbol == "AAPL"
    assert sale.issuer_cik == "0000320193"
    assert sale.insider_name == "DOE JANE A"
    assert sale.insider_cik == "0009999901"
    assert sale.is_officer and not sale.is_director and not sale.is_ten_pct_owner
    assert sale.officer_title == "Senior Vice President, CFO"
    assert sale.transaction_code == "S"
    assert sale.transaction_date == date(2026, 2, 6)
    assert sale.shares == 15000.0
    assert sale.price == pytest.approx(228.8701)
    assert sale.shares_owned_after == 110450.0
    assert sale.direct is True

    # Award: code A, price omitted (footnote only) -> 0.0, held indirectly.
    assert award.transaction_code == "A"
    assert award.shares == 8000.0
    assert award.price == 0.0
    assert award.shares_owned_after == 118450.0
    assert award.direct is False


def test_parse_director_purchase_truefalse_flags() -> None:
    (txn,) = parse_ownership_xml(DIRECTOR_XML)
    assert txn.is_director is True
    assert txn.is_officer is False
    assert txn.officer_title is None
    assert txn.is_ten_pct_owner is False
    assert txn.transaction_code == "P"
    assert txn.shares == 5000.0
    assert txn.price == pytest.approx(214.25)
    assert txn.shares_owned_after == 62500.0
    assert txn.direct is True


def test_parse_ownership_xml_rejects_non_ownership_document() -> None:
    with pytest.raises(ValueError):
        parse_ownership_xml(b"not xml at all")
    with pytest.raises(ValueError):
        parse_ownership_xml(b"<html><body>404</body></html>")


# ---------------------------------------------------------------------------
# (b) backfill: quarterly form index
# ---------------------------------------------------------------------------


def _assert_form4_rows(frame: pd.DataFrame) -> None:
    assert list(frame["form_type"]) == ["4", "4"]
    assert list(frame["cik"]) == [320193, 7654321]
    assert list(frame["date_filed"]) == [date(2026, 1, 8), date(2026, 2, 11)]
    assert list(frame["filename"]) == [
        "edgar/data/320193/0000320193-26-000012.txt",
        "edgar/data/7654321/0007654321-26-000003.txt",
    ]
    assert list(frame["asof_date"]) == list(frame["date_filed"])


def test_form_index_fixed_width() -> None:
    frame = parse_form_index(FORM_IDX_FIXED)
    _assert_form4_rows(frame)
    assert frame.loc[0, "company"] == "APPLE INC"


def test_form_index_pipe_format() -> None:
    frame = parse_form_index(FORM_IDX_PIPE)
    _assert_form4_rows(frame)


def test_form_index_real_layout_data_wider_than_header_labels() -> None:
    """Real EDGAR files pad data columns wider than the header labels; the
    parser must not truncate CIK/date by trusting header offsets."""
    frame = parse_form_index(FORM_IDX_FIXED_DRIFTED)
    _assert_form4_rows(frame)
    assert frame.loc[0, "company"] == "APPLE INC"


def test_form_index_refuses_pervasive_layout_drift() -> None:
    garbage = "\n".join(f"unsplittable_row_without_columns_{i}" for i in range(12))
    raw = (
        _fixed_line("Form Type", "Company Name", "CIK", "Date Filed", "File Name")
        + "\n"
        + "-" * 130
        + "\n"
        + garbage
        + "\n"
    ).encode("latin-1")
    with pytest.raises(ValueError, match="layout drift"):
        parse_form_index(raw)


def test_form_index_amendments_opt_in() -> None:
    frame = parse_form_index(FORM_IDX_FIXED, forms=("4", "4/A"))
    assert list(frame["form_type"]) == ["4", "4/A", "4"]


def test_form_index_available_at_is_conservative_next_morning() -> None:
    frame = parse_form_index(FORM_IDX_FIXED)
    first = frame.loc[0]
    # No acceptance time in the index -> conservative placeholder: 06:00 ET
    # the day AFTER the filing date, later than ANY real same-day acceptance
    # (Section 16 cutoff is 22:00 ET).
    assert first["available_at"] == datetime(2026, 1, 9, 6, 0, tzinfo=ET_TZ)
    latest_possible_acceptance = datetime(2026, 1, 8, 22, 0, tzinfo=ET_TZ)
    assert first["available_at"] > latest_possible_acceptance


def test_backfill_quarters_spans_range() -> None:
    assert backfill_quarters(date(2025, 11, 15), date(2026, 2, 10)) == [
        (2025, 4),
        (2026, 1),
    ]
    assert backfill_quarters(date(2026, 1, 8), date(2026, 2, 11)) == [(2026, 1)]
    with pytest.raises(ValueError):
        backfill_quarters(date(2026, 2, 1), date(2026, 1, 1))


def test_backfill_index_urls_spans_quarters() -> None:
    urls = backfill_index_urls(date(2025, 11, 15), date(2026, 2, 10))
    assert urls == [
        "https://www.sec.gov/Archives/edgar/full-index/2025/QTR4/form.idx",
        "https://www.sec.gov/Archives/edgar/full-index/2026/QTR1/form.idx",
    ]
    with pytest.raises(ValueError):
        backfill_index_urls(date(2026, 2, 1), date(2026, 1, 1))


# ---------------------------------------------------------------------------
# (c) incremental: submissions JSON parallel arrays
# ---------------------------------------------------------------------------


def test_submissions_url_pads_cik_to_ten_digits() -> None:
    assert submissions_url(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"


def test_parse_submissions_filters_parallel_arrays_to_form4() -> None:
    frame = parse_submissions_json(SUBMISSIONS_PAYLOAD)
    assert list(frame["form"]) == ["4", "4"]  # the 10-Q row is dropped
    assert list(frame["accession_number"]) == [
        "0000320193-26-000030",
        "0000320193-26-000028",
    ]
    assert list(frame["symbol"]) == ["AAPL", "AAPL"]
    assert list(frame["cik"]) == [320193, 320193]
    assert list(frame["asof_date"]) == [date(2026, 2, 10), date(2026, 2, 6)]


def test_evening_acceptance_available_same_evening() -> None:
    """A filing accepted 20:30 ET must be available the SAME evening, 20:35 ET."""
    frame = parse_submissions_json(SUBMISSIONS_PAYLOAD)
    row = frame.iloc[0]  # accepted 2026-02-11T01:30Z == 2026-02-10 20:30 ET
    assert row["acceptance_datetime"] == datetime(2026, 2, 10, 20, 30, tzinfo=ET_TZ)
    assert row["available_at"] == datetime(2026, 2, 10, 20, 35, tzinfo=ET_TZ)
    # Same evening: the ET availability date equals the filing's asof_date.
    assert pd.Timestamp(row["available_at"]).date() == row["asof_date"] == date(2026, 2, 10)


def test_intraday_acceptance_available_intraday() -> None:
    """A filing accepted 10:05 ET must be available intraday at 10:10 ET."""
    frame = parse_submissions_json(SUBMISSIONS_PAYLOAD)
    row = frame.iloc[1]  # accepted 2026-02-06T15:05Z == 2026-02-06 10:05 ET
    assert row["acceptance_datetime"] == datetime(2026, 2, 6, 10, 5, tzinfo=ET_TZ)
    assert row["available_at"] == datetime(2026, 2, 6, 10, 10, tzinfo=ET_TZ)


def test_acceptance_parsing_and_naive_rejection() -> None:
    # SGML 14-digit header form is ET wall clock by EDGAR specification.
    assert parse_acceptance_datetime("20260210203000") == datetime(
        2026, 2, 10, 20, 30, tzinfo=ET_TZ
    )
    # available_at_from_acceptance never guesses a timezone.
    with pytest.raises(ValueError):
        available_at_from_acceptance(datetime(2026, 2, 10, 20, 30))


# ---------------------------------------------------------------------------
# Client: mandatory User-Agent, throttle, cache-first parquet
# ---------------------------------------------------------------------------


def test_every_request_carries_mandatory_user_agent(tmp_path) -> None:
    url = submissions_url(320193)
    client = _client({url: FakeResponse(payload=SUBMISSIONS_PAYLOAD)}, tmp_path)
    client.fetch_submissions(320193)
    getter = client._http  # noqa: SLF001 — asserting on the injected fake
    assert getter.calls, "no request recorded"
    for _, headers in getter.calls:
        assert headers is not None
        assert headers["User-Agent"] == "quant-research contact: sensordyme@gmail.com"


def test_throttle_spaces_requests_at_8_per_second(tmp_path) -> None:
    url = submissions_url(320193)
    sleeps: list[float] = []
    client = _client({url: FakeResponse(payload=SUBMISSIONS_PAYLOAD)}, tmp_path, sleeps=sleeps)
    for _ in range(3):
        client.fetch_submissions(320193)
    # Frozen clock: each send is scheduled >= previous send + 1/8 s, so the
    # 2nd and 3rd requests must wait 0.125 s and 0.250 s respectively.
    assert sleeps == pytest.approx([0.125, 0.250])


def test_fetch_filing_is_cache_first(tmp_path) -> None:
    filename = "edgar/data/320193/0000320193-26-000012.txt"
    url = "https://www.sec.gov/Archives/" + filename
    client = _client({url: FakeResponse(content=FILING_TXT)}, tmp_path)

    first = client.fetch_filing(filename)
    assert len(client._http.calls) == 1  # noqa: SLF001
    assert list(first["transaction_code"]) == ["P"]
    assert list(first["asof_date"]) == [date(2026, 2, 10)]
    # Accepted 20:30 ET -> available same evening 20:35 ET, via the full path.
    assert pd.Timestamp(first["available_at"].iloc[0]) == pd.Timestamp(
        datetime(2026, 2, 10, 20, 35, tzinfo=ET_TZ)
    )

    second = client.fetch_filing(filename)
    assert len(client._http.calls) == 1, "cache hit must not touch HTTP"
    assert list(second["transaction_code"]) == ["P"]
    assert list(second["shares"]) == [5000.0]
    assert pd.Timestamp(second["available_at"].iloc[0]) == pd.Timestamp(
        datetime(2026, 2, 10, 20, 35, tzinfo=ET_TZ)
    )


def test_fetch_quarter_index_is_cache_first(tmp_path) -> None:
    url = "https://www.sec.gov/Archives/edgar/full-index/2026/QTR1/form.idx"
    client = _client({url: FakeResponse(content=FORM_IDX_FIXED)}, tmp_path)
    first = client.fetch_quarter_index(2026, 1)
    second = client.fetch_quarter_index(2026, 1)
    assert len(client._http.calls) == 1  # noqa: SLF001
    assert list(second["cik"]) == list(first["cik"]) == [320193, 7654321]


def test_parse_filing_txt_extracts_acceptance_and_xml() -> None:
    filing = parse_filing_txt(FILING_TXT)
    assert filing.accession_number == "0000320193-26-000012"
    assert filing.acceptance_datetime == datetime(2026, 2, 10, 20, 30, tzinfo=ET_TZ)
    assert len(filing.transactions) == 1
    assert filing.transactions[0].transaction_code == "P"


def test_http_error_is_loud(tmp_path) -> None:
    url = submissions_url(320193)
    client = _client({url: FakeResponse(content=b"forbidden", status_code=403)}, tmp_path)
    with pytest.raises(RuntimeError, match="403"):
        client.fetch_submissions(320193)


# ---------------------------------------------------------------------------
# Backfill orchestration: symbols -> CIKs -> quarter indexes -> filings
# ---------------------------------------------------------------------------

INDEX_URL_2026Q1 = "https://www.sec.gov/Archives/edgar/full-index/2026/QTR1/form.idx"
AAPL_FILING_URL = "https://www.sec.gov/Archives/edgar/data/320193/0000320193-26-000012.txt"
BETA_FILING_URL = "https://www.sec.gov/Archives/edgar/data/7654321/0007654321-26-000003.txt"

# BETA's filing: same SGML wrapper, issuer swapped to BETA/7654321.
BETA_FILING_TXT = FILING_TXT.replace(b"0000320193-26-000012", b"0007654321-26-000003").replace(
    b"<issuerCik>0000320193</issuerCik>", b"<issuerCik>0007654321</issuerCik>"
).replace(b"<issuerTradingSymbol>AAPL</issuerTradingSymbol>",
          b"<issuerTradingSymbol>BETA</issuerTradingSymbol>")

COMPANY_TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 7654321, "ticker": "BETA", "title": "Beta Corp"},
    "2": {"cik_str": 999, "ticker": "AAPL", "title": "Impostor (dup, must lose)"},
}

BACKFILL_TODAY = date(2026, 8, 22)  # 2026Q3 — 2026Q1 is a completed quarter


def test_parse_company_tickers_first_wins() -> None:
    mapping = parse_company_tickers(COMPANY_TICKERS_PAYLOAD)
    assert mapping == {"AAPL": 320193, "BETA": 7654321}


def test_fetch_company_ciks_resolves_and_is_loud_on_missing(tmp_path) -> None:
    url = EdgarConfig().company_tickers_url
    client = _client({url: FakeResponse(payload=COMPANY_TICKERS_PAYLOAD)}, tmp_path)
    assert client.fetch_company_ciks(["aapl", "BETA"]) == {"AAPL": 320193, "BETA": 7654321}
    with pytest.raises(KeyError, match="ZZZZ"):
        client.fetch_company_ciks(["AAPL", "ZZZZ"])


def test_backfill_filings_filters_by_cik_and_date_and_reports_failures(tmp_path) -> None:
    routes = {
        INDEX_URL_2026Q1: FakeResponse(content=FORM_IDX_FIXED),
        AAPL_FILING_URL: FakeResponse(content=FILING_TXT),
        # BETA's .txt is malformed: must be skipped LOUDLY, not abort the run.
        BETA_FILING_URL: FakeResponse(content=b"<html>not a filing</html>"),
    }
    client = _client(routes, tmp_path)
    summary = client.backfill_filings(
        {"AAPL": 320193, "BETA": 7654321},
        date(2026, 1, 1),
        date(2026, 3, 31),
        today=BACKFILL_TODAY,
    )
    # The 10-K/8-K/4A index rows and non-requested CIKs never fetch; the AAPL
    # Form 4 lands in the cache; BETA's malformed txt is reported, not raised.
    assert summary.filings_by_symbol == {"AAPL": 1, "BETA": 0}
    assert summary.failed == ("edgar/data/7654321/0007654321-26-000003.txt",)
    assert (tmp_path / "filings" / "accession=0000320193-26-000012.parquet").exists()


def test_backfill_filings_date_window_excludes_out_of_range(tmp_path) -> None:
    routes = {
        INDEX_URL_2026Q1: FakeResponse(content=FORM_IDX_FIXED),
        BETA_FILING_URL: FakeResponse(content=BETA_FILING_TXT),
    }
    client = _client(routes, tmp_path)
    # Window starts Feb 1: AAPL's Jan 8 filing is out of range, BETA's Feb 11 is in.
    summary = client.backfill_filings(
        {"AAPL": 320193, "BETA": 7654321},
        date(2026, 2, 1),
        date(2026, 3, 31),
        today=BACKFILL_TODAY,
    )
    assert summary.filings_by_symbol == {"AAPL": 0, "BETA": 1}
    assert summary.failed == ()
    cached = pd.read_parquet(tmp_path / "filings" / "accession=0007654321-26-000003.parquet")
    assert list(cached["symbol"]) == ["BETA"]


def test_backfill_filings_cache_is_the_checkpoint(tmp_path) -> None:
    """A re-run refetches NOTHING already on disk — interruption resumes free."""
    routes = {
        INDEX_URL_2026Q1: FakeResponse(content=FORM_IDX_FIXED),
        AAPL_FILING_URL: FakeResponse(content=FILING_TXT),
        BETA_FILING_URL: FakeResponse(content=BETA_FILING_TXT),
    }
    client = _client(routes, tmp_path)
    args = ({"AAPL": 320193, "BETA": 7654321}, date(2026, 1, 1), date(2026, 3, 31))
    first = client.backfill_filings(*args, today=BACKFILL_TODAY)
    assert first.filings_by_symbol == {"AAPL": 1, "BETA": 1}
    calls_after_first = len(client._http.calls)  # noqa: SLF001
    assert calls_after_first == 3  # 1 index + 2 filings

    second = client.backfill_filings(*args, today=BACKFILL_TODAY)
    assert second.filings_by_symbol == first.filings_by_symbol
    assert len(client._http.calls) == calls_after_first, "resume must not re-touch HTTP"  # noqa: SLF001


def test_backfill_filings_rejects_current_incomplete_quarter(tmp_path) -> None:
    client = _client({}, tmp_path)
    with pytest.raises(ValueError, match="current"):
        client.backfill_filings(
            {"AAPL": 320193},
            date(2026, 1, 1),
            date(2026, 2, 20),
            today=date(2026, 2, 15),  # still inside 2026Q1
        )


# ---------------------------------------------------------------------------
# Features: net insider dollars per 21d + officer-buy event flag
# ---------------------------------------------------------------------------


def _txn_row(
    symbol: str,
    asof: date,
    code: str,
    shares: float,
    price: float,
    officer: bool,
    available: datetime,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "transaction_code": code,
        "shares": shares,
        "price": price,
        "is_officer": officer,
        "asof_date": asof,
        "available_at": available,
    }


@pytest.fixture
def txn_frame() -> pd.DataFrame:
    rows = [
        # Director buy Jan 5, but FILED LATE: public only Jan 9 at 10:10 ET.
        _txn_row("AAPL", date(2026, 1, 5), "P", 10_000, 10.0, False,
                 datetime(2026, 1, 9, 10, 10, tzinfo=ET_TZ)),
        # Officer sale Jan 15, filed same evening.
        _txn_row("AAPL", date(2026, 1, 15), "S", 2_000, 25.0, True,
                 datetime(2026, 1, 15, 18, 35, tzinfo=ET_TZ)),
        # Officer open-market buy Feb 4 (the event flag) + an award (code A,
        # must contribute zero dollars).
        _txn_row("AAPL", date(2026, 2, 4), "P", 1_000, 30.0, True,
                 datetime(2026, 2, 4, 12, 10, tzinfo=ET_TZ)),
        _txn_row("AAPL", date(2026, 2, 4), "A", 5_000, 0.0, True,
                 datetime(2026, 2, 4, 12, 10, tzinfo=ET_TZ)),
        # Second symbol, kept independent by the groupby.
        _txn_row("MSFT", date(2026, 1, 20), "P", 100, 50.0, False,
                 datetime(2026, 1, 20, 17, 5, tzinfo=ET_TZ)),
    ]
    return pd.DataFrame(rows)


def test_net_insider_dollars_21d_window(txn_frame: pd.DataFrame) -> None:
    feats = insider_features(txn_frame, window_days=21)
    aapl = feats[feats["symbol"] == "AAPL"].set_index("asof_date")
    col = "net_insider_dollars_21d"

    # Jan 5: +10,000 * 10 = +100,000.
    assert aapl.loc[date(2026, 1, 5), col] == pytest.approx(100_000.0)
    # Jan 15: window (Dec 25, Jan 15] holds the Jan 5 buy and the sale:
    # +100,000 - 50,000 = +50,000.
    assert aapl.loc[date(2026, 1, 15), col] == pytest.approx(50_000.0)
    # Feb 4: window (Jan 14, Feb 4] DROPS the Jan 5 buy, keeps the Jan 15
    # sale, adds the Feb 4 buy; the award contributes ZERO:
    # -50,000 + 30,000 = -20,000.
    assert aapl.loc[date(2026, 2, 4), col] == pytest.approx(-20_000.0)

    msft = feats[feats["symbol"] == "MSFT"].set_index("asof_date")
    assert msft.loc[date(2026, 1, 20), col] == pytest.approx(5_000.0)


def test_officer_buy_event_flag(txn_frame: pd.DataFrame) -> None:
    feats = insider_features(txn_frame, window_days=21)
    aapl = feats[feats["symbol"] == "AAPL"].set_index("asof_date")
    assert bool(aapl.loc[date(2026, 1, 5), "officer_buy"]) is False  # director buy
    assert bool(aapl.loc[date(2026, 1, 15), "officer_buy"]) is False  # officer SALE
    assert bool(aapl.loc[date(2026, 2, 4), "officer_buy"]) is True  # officer P


def test_feature_available_at_is_max_over_window(txn_frame: pd.DataFrame) -> None:
    """The windowed sum is unknowable until its LAST filing is public: a
    late-filed transaction drags the whole window's available_at later."""
    feats = insider_features(txn_frame, window_days=21)
    aapl = feats[feats["symbol"] == "AAPL"].set_index("asof_date")
    # Jan 5 row: the transaction was filed late -> available only Jan 9.
    assert pd.Timestamp(aapl.loc[date(2026, 1, 5), "available_at"]) == pd.Timestamp(
        datetime(2026, 1, 9, 10, 10, tzinfo=ET_TZ)
    )
    # Jan 15 row: max(Jan 9 10:10, Jan 15 18:35) = Jan 15 18:35.
    assert pd.Timestamp(aapl.loc[date(2026, 1, 15), "available_at"]) == pd.Timestamp(
        datetime(2026, 1, 15, 18, 35, tzinfo=ET_TZ)
    )


def test_insider_features_empty_input() -> None:
    feats = insider_features(pd.DataFrame(), window_days=21)
    assert feats.empty
    assert "net_insider_dollars_21d" in feats.columns
    assert "officer_buy" in feats.columns


def test_transactions_frame_point_in_time_columns() -> None:
    txns = parse_ownership_xml(OFFICER_XML)
    acceptance = parse_acceptance_datetime("20260206163012")
    frame = transactions_frame(txns, acceptance)
    assert {"asof_date", "available_at"} <= set(frame.columns)
    assert list(frame["asof_date"]) == [date(2026, 2, 6), date(2026, 2, 6)]
    expected = datetime(2026, 2, 6, 16, 35, 12, tzinfo=ET_TZ)
    assert all(pd.Timestamp(v) == pd.Timestamp(expected) for v in frame["available_at"])


def test_config_rejects_blank_user_agent() -> None:
    with pytest.raises(ValueError):
        EdgarConfig(user_agent="   ")
