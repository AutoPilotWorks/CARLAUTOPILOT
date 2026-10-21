import json

import pytest

from carlautopilot.cli.dryrun import main, run_dryrun


def test_dryrun_tiny(tmp_path):
    s = main(["--work-dir", str(tmp_path / "w"), "--episodes", "8", "--heldout", "2", "--stride", "8", "--epochs", "1"])
    assert s["heldout"]["episodes"] == 2 and s["samples"] > 0
    data = json.loads((tmp_path / "w" / "summary.json").read_text(encoding="utf-8"))
    assert data["train_episodes"] + data["val_episodes"] == 8
    assert (tmp_path / "w" / "eval" / "report.md").is_file()
    assert (tmp_path / "w" / "run" / "checkpoint.pt").is_file()
    with pytest.raises(FileExistsError):
        run_dryrun(tmp_path / "w")
