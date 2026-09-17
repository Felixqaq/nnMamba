"""Measure the paper-sized MONAI 3D Hybrid-Mamba without changing existing code."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from experiments.full_hybrid_window_distillation.model import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    model = build().cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    x = torch.randn(args.batch, 1, 32, 512, 512, device="cuda")
    result = {"gpu": torch.cuda.get_device_name(), "batch": args.batch,
              "shape": list(x.shape), "parameters": sum(p.numel() for p in model.parameters()),
              "amp": "disabled_fp32", "status": "running"}
    durations = []
    try:
        for step in range(6):
            torch.cuda.synchronize()
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=False):
                logits = model(x).flatten()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
            assert torch.isfinite(loss), "Non-finite loss"
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            print(f"step={step} seconds={seconds:.3f}", flush=True)
            if step >= 2:
                durations.append(seconds)
        result.update(status="complete", seconds_per_patient=sum(durations)/len(durations)/args.batch,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        result["thirteen_stages_40ep_train_only_hours"] = result["seconds_per_patient"] * 461 * 40 * 13 / 3600
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


