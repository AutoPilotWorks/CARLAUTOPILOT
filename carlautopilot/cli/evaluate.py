from __future__ import annotations

import argparse
import json
from pathlib import Path

from carlautopilot.policies import ConstantPolicy, PathFollowPolicy
from carlautopilot.runner import run_suite
from carlautopilot.scenario import ScenarioSpec
from carlautopilot.scenarios_lib import standard_suite
from carlautopilot.sim import FakeEnv

POLICIES = {"pathfollow": PathFollowPolicy, "constant": ConstantPolicy}


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Evaluate a policy on a scenario suite with the fake environment")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--policy", choices=sorted(POLICIES), default="pathfollow")
    p.add_argument("--scenarios-dir", help="directory of scenario JSON files (default: built-in standard suite)")
    p.add_argument("--title", default="Evaluation report")
    a = p.parse_args(argv)
    if a.scenarios_dir:
        scenarios = [ScenarioSpec.load(f) for f in sorted(Path(a.scenarios_dir).glob("*.json"))]
        if not scenarios:
            p.error("no scenario JSON files found")
    else:
        scenarios = standard_suite()
    result = run_suite(lambda: FakeEnv(privileged_obs=True), POLICIES[a.policy], scenarios, a.out_dir, title=a.title)
    print(json.dumps(result["aggregate"], ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
