"""Run the native reconstruction recipe as an end user, without a legacy runtime."""

import json
import subprocess
import sys
from pathlib import Path


def test_portable_example_reconstructs_equity_and_futures_after_pruning_and_relocation():
    example = Path(__file__).parents[1] / "examples/portable_snapshot_migration.py"
    completed = subprocess.run(
        [sys.executable, str(example)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["assertions"] == "passed"
    assert result["equity_stable_ids"] == [17, 17, 17, 17]
    assert result["futures_selected_ids"] == [23]
    assert result["tables"] == [
        "equity_actions",
        "equity_identities",
        "equity_membership",
        "equity_prices",
        "futures_definitions",
        "futures_prices",
    ]
