"""Catalyst provider + reaction-session semantics (audit D-047/D-128/D-212)."""

from datetime import date, datetime

from catalyst.core.types import Catalyst, CatalystType
from catalyst.data.catalysts import StaticEconomicCalendar, resolve_reaction_session


def _cat(when: datetime, type_=CatalystType.EARNINGS) -> Catalyst:
    return Catalyst(symbol="SPY", type=type_, when=when, source="test")


class TestReactionSession:
    def test_amc_earnings_react_next_session(self):
        # Tuesday 2024-06-04 16:05 (after close) -> Wednesday
        c = _cat(datetime(2024, 6, 4, 16, 5))
        assert resolve_reaction_session(c) == date(2024, 6, 5)

    def test_bmo_earnings_react_same_session(self):
        c = _cat(datetime(2024, 6, 4, 8, 0))
        assert resolve_reaction_session(c) == date(2024, 6, 4)

    def test_exactly_1600_is_after_close(self):
        c = _cat(datetime(2024, 6, 4, 16, 0))
        assert resolve_reaction_session(c) == date(2024, 6, 5)

    def test_fomc_1400_reacts_same_day(self):
        c = _cat(datetime(2024, 6, 12, 14, 0), CatalystType.FOMC)
        assert resolve_reaction_session(c) == date(2024, 6, 12)

    def test_friday_amc_reacts_monday(self):
        c = _cat(datetime(2024, 6, 7, 16, 30))     # Friday after close
        assert resolve_reaction_session(c) == date(2024, 6, 10)

    def test_weekend_event_rolls_forward(self):
        c = _cat(datetime(2024, 6, 8, 9, 0))       # Saturday
        assert resolve_reaction_session(c) == date(2024, 6, 10)


class TestStaticCalendarDedup:
    def test_same_day_cpi_fomc_collapse(self, tmp_path):
        """2024-06-12 had BOTH CPI (08:30) and FOMC (14:00): one catalyst per
        symbol per session, the earlier event kept (audit D-128)."""
        (tmp_path / "cpi.csv").write_text(
            "release_date,reference_month,time\n2024-06-12,May 2024,08:30 AM\n")
        (tmp_path / "fomc.csv").write_text(
            "decision_date\n2024-06-12\n")
        cal = StaticEconomicCalendar(tmp_path, ["SPY"])
        cats = cal.get_catalyst_calendar(date(2024, 6, 1), date(2024, 6, 30))
        assert len(cats) == 1
        assert cats[0].type is CatalystType.CPI    # 08:30 beats 14:00

    def test_distinct_days_kept(self, tmp_path):
        (tmp_path / "cpi.csv").write_text(
            "release_date,reference_month,time\n2024-06-12,May 2024,08:30 AM\n")
        (tmp_path / "fomc.csv").write_text(
            "decision_date\n2024-07-31\n")
        cal = StaticEconomicCalendar(tmp_path, ["SPY"])
        cats = cal.get_catalyst_calendar(date(2024, 6, 1), date(2024, 8, 31))
        assert len(cats) == 2


class TestShippedCalendarsAreClean:
    def test_no_phantom_2025_shutdown_rows(self):
        """The releases that never happened must stay gone (audit D-018)."""
        text = open("config/calendars/cpi.csv").read()
        for phantom in ("2025-10-15", "2025-11-13", "2025-12-10"):
            assert phantom not in text
        assert "2025-10-24" in text     # the real (delayed) September release
        assert "2025-12-18" in text     # the real (delayed) November release

    def test_one_release_per_reference_month(self):
        import csv as _csv
        with open("config/calendars/cpi.csv") as f:
            months = [r["reference_month"] for r in _csv.DictReader(f)]
        dupes = {m for m in months if months.count(m) > 1}
        assert not dupes, f"duplicate CPI reference months: {dupes}"
