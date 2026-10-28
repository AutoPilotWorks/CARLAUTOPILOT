from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from carlautopilot.core import Action, EpisodeHeader, Event, GroundTruth, StepRecord

SCHEMA_VERSION = 1


def _record_to_dict(r: StepRecord) -> dict:
    return {
        "step": r.step,
        "t": r.t,
        "speed_mps": r.speed_mps,
        "command": r.command,
        "action": asdict(r.action),
        "truth": asdict(r.truth),
        "events": [asdict(e) for e in r.events],
        "lead_distance_m": r.lead_distance_m,
        **({} if r.percept is None else {"percept": list(r.percept)}),
    }


def _record_from_dict(d: dict) -> StepRecord:
    return StepRecord(
        step=int(d["step"]),
        t=float(d["t"]),
        speed_mps=float(d["speed_mps"]),
        command=int(d["command"]),
        action=Action(**d["action"]),
        truth=GroundTruth(**d["truth"]),
        events=tuple(Event(**e) for e in d.get("events", [])),
        lead_distance_m=None if d.get("lead_distance_m") is None else float(d["lead_distance_m"]),
        percept=None if d.get("percept") is None else tuple(d["percept"]),
    )


class EpisodeLogWriter:
    def __init__(self, path: str | Path, overwrite: bool = False):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {self.path}")
        self._file = None
        self._header_written = False

    def __enter__(self) -> "EpisodeLogWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write_header(self, header: EpisodeHeader) -> None:
        if self._file is None:
            raise RuntimeError("use EpisodeLogWriter as a context manager")
        if self._header_written:
            raise RuntimeError("header already written")
        self._file.write(json.dumps(asdict(header), ensure_ascii=False) + "\n")
        self._header_written = True

    def write(self, record: StepRecord) -> None:
        if self._file is None or not self._header_written:
            raise RuntimeError("write the header first")
        self._file.write(json.dumps(_record_to_dict(record), ensure_ascii=False) + "\n")


def read_episode(path: str | Path) -> tuple[EpisodeHeader, list[StepRecord]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty episode log")
    h = json.loads(lines[0])
    if h.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {h.get('schema_version')}")
    header = EpisodeHeader(
        scenario_name=h["scenario_name"],
        seed=h["seed"],
        dt=float(h["dt"]),
        policy_name=h["policy_name"],
        schema_version=int(h["schema_version"]),
    )
    return header, [_record_from_dict(json.loads(line)) for line in lines[1:] if line.strip()]
