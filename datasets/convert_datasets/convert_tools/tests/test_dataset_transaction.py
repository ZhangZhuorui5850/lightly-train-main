from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import dataset_transaction as transaction  # noqa: E402


@pytest.mark.skipif(transaction.fcntl is None, reason="POSIX file locking is unavailable")
def test_staged_output_rejects_concurrent_writer(tmp_path: Path) -> None:
    output = tmp_path / "merged"
    with transaction.staged_output(output) as first_stage:
        (first_stage / "sentinel").write_text("first", encoding="utf-8")
        with pytest.raises(RuntimeError, match="另一进程"):
            with transaction.staged_output(output):
                pass

    assert (output / "sentinel").read_text(encoding="utf-8") == "first"
