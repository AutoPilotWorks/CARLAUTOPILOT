from __future__ import annotations

import argparse

from carlautopilot.eval.plots import plot_logs


def main(argv: list[str] | None = None) -> list[str]:
    p = argparse.ArgumentParser(description="Draw trajectory/speed/lateral-error PNGs for every episode log in a folder")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    paths = [str(x) for x in plot_logs(a.log_dir, a.out_dir)]
    print("\n".join(paths))
    return paths


if __name__ == "__main__":
    main()
