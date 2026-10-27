import os
import zipfile
from pathlib import Path

from django.conf import settings

from core.data_loader import load_all_data


REQUIRED_CSVS = {
    'clients.csv',
    'client_data.csv',
    'poisoned_clients.csv',
    'group_assignments.csv',
    'model_updates.csv',
    'test_results.csv',
}


def extract_dataset_zip(uploaded_file):
    """Store and extract a FedGT dataset ZIP under MEDIA_ROOT/datasets."""
    dataset_root = Path(settings.MEDIA_ROOT) / 'datasets'
    dataset_root.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_root / uploaded_file.name
    with zip_path.open('wb+') as dest:
        for chunk in uploaded_file.chunks():
            dest.write(chunk)

    extract_dir = dataset_root / Path(uploaded_file.name).stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    csv_dir = _find_csv_dir(extract_dir)
    present = {p.name for p in csv_dir.glob('*.csv')}
    missing = sorted(REQUIRED_CSVS - present)
    if missing:
        raise ValueError(f"Dataset ZIP is missing: {', '.join(missing)}")

    load_all_data(base_dir=str(csv_dir), replace=True)
    return {
        'extract_dir': os.fspath(extract_dir),
        'csv_dir': os.fspath(csv_dir),
        'loaded_files': sorted(REQUIRED_CSVS),
    }


def _find_csv_dir(root):
    for path in [root, *root.rglob('*')]:
        if path.is_dir():
            names = {p.name for p in path.glob('*.csv')}
            if REQUIRED_CSVS.issubset(names):
                return path
    raise ValueError('No folder in the ZIP contains the complete FedGT CSV set.')
