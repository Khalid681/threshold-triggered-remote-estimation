import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('preprocess_tomato', Path(__file__).parents[2] / 'preprocess_tomato.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_preprocessing_preserves_source_and_flags_invalid_values(tmp_path):
    source = tmp_path / 'source.csv'
    pd.DataFrame({'ts_generation': [1748730000000, 1748728800000, 1748729400000],
                  'generated_by': ['b', 'a', 'a'], 'line': [1, 1, 1],
                  'temperature': [25, 24, 23], 'humidity': [655.35, 0, 20]}).to_csv(source, index=False)
    before = source.read_bytes()
    output = tmp_path / 'clean.csv'
    result = module.preprocess(source, output, tmp_path / 'reports')
    clean = pd.read_csv(output)
    assert source.read_bytes() == before
    assert result['invalid_or_missing_moisture'] == 1
    assert clean.Node_ID.tolist() == [1, 1, 2]
    assert clean.Moist.iloc[0] == 0
    assert clean.zero_moisture_flag.iloc[0]
    assert pd.isna(clean.Moist.iloc[2])
    assert clean.Temp.tolist() == [24, 23, 25]


def test_preprocessing_rejects_duplicate_device_timestamps(tmp_path):
    source = tmp_path / 'duplicate.csv'
    pd.DataFrame({'ts_generation': [1748728800000]*2, 'generated_by': ['a']*2,
                  'line': [1]*2, 'temperature': [24]*2, 'humidity': [20]*2}).to_csv(source, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        module.preprocess(source, tmp_path/'clean.csv', tmp_path/'reports')
