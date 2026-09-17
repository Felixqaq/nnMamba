"""Wait for the existing runner, then load the corrected orchestration once."""
import subprocess
import time
from pathlib import Path

pid_path = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260915/runner.pid')
pid = int(pid_path.read_text().strip())
proc = Path(f'/proc/{pid}/stat')
identity = proc.read_text().split()[21] if proc.exists() else None
while proc.exists():
    try:
        if proc.read_text().split()[21] != identity:
            break
    except FileNotFoundError:
        break
    time.sleep(5)
subprocess.run(['bash', '/mnt/d/Felix/Hospital/nnMamba/regression/experiments/ratio5_20260915/launch.sh'], check=True)
