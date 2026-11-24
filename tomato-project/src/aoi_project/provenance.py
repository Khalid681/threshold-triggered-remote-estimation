from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(root='results/runs', name=None, overwrite=False):
    root = Path(root).resolve()
    name = name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Run name must be a single safe directory name')
    output = root/name
    if output.exists():
        if not overwrite:
            raise FileExistsError(f'{output} already exists; use a unique name or explicit --overwrite')
        # Preserve even explicitly replaced runs as complete previous results.
        backup = root/(name+'.previous-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
        output.rename(backup)
    output.mkdir(parents=True)
    for child in ['raw','summaries','figures','reports','logs']:
        (output/child).mkdir()
    return output


def write_metadata(output, config, source_config=None, splits=()):
    used = output/'config_used.json'
    used.write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    versions = {d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'unavailable', True
    code = [Path(x) for x in ['preprocess_tomato.py','run_agriculture.py','run_baseline.py','paired_analysis.py','plot_results.py','quality_checks.py']]
    code += list(Path('src/aoi_project').glob('*.py'))
    metadata = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), git_commit=commit, git_dirty=dirty,
                    python_version=platform.python_version(), platform=platform.platform(), package_versions=versions,
                    config_sha256=sha256(used), source_config_sha256=sha256(source_config) if source_config else None,
                    source_sha256={str(p):sha256(p) for p in code if p.exists()},
                    feedback_model=config.get('feedback_model'), channel_model=config.get('channel_model'),
                    dataset_sha256=sha256(config['data_path']) if 'data_path' in config else None,
                    nodes=len({s.node_id for s in splits}), segments=len(splits),
                    thresholds=len(config.get('threshold_multipliers', [])), evaluation_seeds=len(config.get('evaluation_seeds', [])),
                    calibration_seeds=len(config.get('calibration_seeds', [])),
                    conditions=len(config.get('threshold_multipliers', []))*len(config.get('loss_probabilities', [])), status='running')
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    (output/'environment.txt').write_text('Python '+platform.python_version()+'\n'+'\n'.join(f'{k}=={v}' for k,v in sorted(versions.items()))+'\n', encoding='utf-8')
    if splits:
        import pandas as pd
        rows = []
        for split in splits:
            row = dict(node_id=split.node_id, signal=split.signal, segment_id=split.segment_id,
                       nominal_interval_minutes=split.nominal_interval_minutes)
            for name in ['train','validation','test']:
                frame = getattr(split,name)
                row.update({name+'_start':str(frame.timestamp.min()), name+'_end':str(frame.timestamp.max()), name+'_records':len(frame)})
            rows.append(row)
        pd.DataFrame(rows).to_csv(output/'reports'/'split_manifest.csv', index=False)
    return metadata


def finish_run(output, metadata, status='complete'):
    metadata.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), status=status)
    (output/'run_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
