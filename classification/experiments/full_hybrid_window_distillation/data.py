"""Reuse verified HU arrays without modifying the completed SE-ResNet run."""
import json
import os
from pathlib import Path
from experiments.full_window_distillation.data import prepare as prepare_original


def prepare(records: list, source_records: dict, cache: Path, fingerprint: str, deadline) -> dict:
    source = cache.parent.parent / 'seresnet50_full777'
    if source.is_dir():
        previous = json.loads((source / 'config.json').read_text())
        current = json.loads((cache.parent / 'config.json').read_text())
        for key in ('manifest_sha256', 'source_sha256', 'shape', 'sampling', 'windows'):
            if previous[key] != current[key]:
                raise ValueError(f'Cannot reuse cache: {key} differs')
        cache.mkdir(exist_ok=True)
        for row in records:
            deadline()
            for suffix in ('.npy', '.json'):
                target = cache / (row['patient_id'] + suffix)
                if not target.exists():
                    os.link(source / 'cache' / target.name, target)
        if not (cache / 'stats.json').exists():
            (cache / 'stats.json').write_bytes((source / 'cache/stats.json').read_bytes())
    return prepare_original(records, source_records, cache, fingerprint, deadline)
