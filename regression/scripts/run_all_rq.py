"""Run all RQ1/2/3 aligned configs sequentially with per-config logs and resume."""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
LOG_DIR = REG / "train_log" / "run_all_rq"

CONFIGS = [
    "config.rq1.normal_v_abnormal.image.yaml",
    "config.rq1.normal_v_abnormal.fusion.yaml",
    "config.rq2a.angle_3class.image.yaml",
    "config.rq2a.angle_3class.fusion.yaml",
    "config.rq2b.angle_binary.image.yaml",
    "config.rq2b.angle_binary.fusion.yaml",
    "config.rq2c.angle_reg.image.yaml",
    "config.rq2c.angle_reg.fusion.yaml",
    "config.rq3.oi_emphysema.image.yaml",
    "config.rq3.oi_emphysema.fusion.yaml",
]


def load_summary() -> dict:
    path = LOG_DIR / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_summary(summary: dict) -> None:
    (LOG_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    for config in CONFIGS:
        if summary.get(config, {}).get("status") == "done":
            print(f"[skip] {config} already done", flush=True)
            continue
        log_path = LOG_DIR / f"{Path(config).stem}.log"
        print(f"[run ] {config} -> {log_path}", flush=True)
        started = datetime.now().isoformat(timespec="seconds")
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, "train.py", "--config", config],
                cwd=REG, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        uuid = None
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if "Run UUID:" in line or "Weights saved to:" in line:
                uuid = line.strip().split()[-1]
        summary[config] = {
            "status": "done" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "uuid": uuid,
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "log": str(log_path),
        }
        save_summary(summary)
        state = "ok  " if proc.returncode == 0 else "FAIL"
        print(f"[{state}] {config} rc={proc.returncode}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for config in CONFIGS:
        info = summary.get(config, {})
        print(f"  {info.get('status', 'pending'):8} {config}  uuid={info.get('uuid')}",
              flush=True)


if __name__ == "__main__":
    main()
