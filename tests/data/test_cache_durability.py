"""ParquetCache durability contract (audit D-046/D-211)."""

import pandas as pd

from catalyst.data.cache import ParquetCache


class TestAtomicity:
    def test_round_trip(self, tmp_path):
        c = ParquetCache(tmp_path)
        df = pd.DataFrame({"a": [1, 2]})
        c.put("cat", "key1", df)
        got = c.get("cat", "key1")
        pd.testing.assert_frame_equal(got, df)

    def test_no_tmp_residue(self, tmp_path):
        c = ParquetCache(tmp_path)
        c.put("cat", "key1", pd.DataFrame({"a": [1]}))
        residue = [p for p in (tmp_path / "cat").iterdir() if ".tmp" in p.name]
        assert residue == []

    def test_corrupt_file_quarantined_not_fatal(self, tmp_path):
        c = ParquetCache(tmp_path)
        c.put("cat", "key1", pd.DataFrame({"a": [1]}))
        # tear the file
        path = next((tmp_path / "cat").glob("key1*.parquet"))
        path.write_bytes(b"NOT A PARQUET FILE")
        assert c.get("cat", "key1") is None            # miss, not crash
        assert list((tmp_path / "cat").glob("*.corrupt-*")), "evidence preserved"
        # and the slot is writable again
        c.put("cat", "key1", pd.DataFrame({"a": [9]}))
        assert c.get("cat", "key1")["a"].tolist() == [9]


class TestKeyCollisions:
    def test_sanitized_keys_stay_distinct(self, tmp_path):
        """BRK/B and BRK B both sanitize to BRK_B — they must not share a file
        (audit D-211)."""
        c = ParquetCache(tmp_path)
        c.put("cat", "BRK/B_2024", pd.DataFrame({"v": [1]}))
        c.put("cat", "BRK B_2024", pd.DataFrame({"v": [2]}))
        assert c.get("cat", "BRK/B_2024")["v"].tolist() == [1]
        assert c.get("cat", "BRK B_2024")["v"].tolist() == [2]

    def test_clean_keys_keep_legacy_paths(self, tmp_path):
        """Keys needing no sanitization must keep their exact filename so the
        existing warm cache stays valid."""
        c = ParquetCache(tmp_path)
        c.put("cat", "SPY_20240603", pd.DataFrame({"v": [1]}))
        assert (tmp_path / "cat" / "SPY_20240603.parquet").exists()
