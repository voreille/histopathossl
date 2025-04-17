import random
from pathlib import Path

from torchvision import transforms


project_dir = Path(__file__).parents[2].resolve()


def split_patients(patient_ids, seed=42):
    """
    Splits the patient IDs into train, validation, and test sets.
    """
    random.seed(seed)
    random.shuffle(patient_ids)
    n_patients = len(patient_ids)

    train_split = int(n_patients * 0.7)
    val_split = int(n_patients * 0.85)

    train_patient_ids = patient_ids[:train_split]
    val_patient_ids = patient_ids[train_split:val_split]
    test_patient_ids = patient_ids[val_split:]

    return train_patient_ids, val_patient_ids, test_patient_ids


def get_patient_from_tcga_id(tcga_id):
    """
    Extracts the patient ID from the TCGA ID.
    """
    # Example TCGA ID: TCGA-XX-XXXX
    # Extract the first part (TCGA-XX)
    patient_id = "-".join(tcga_id.split("-")[:3])
    return patient_id


def get_train_val_test_tile_paths(data_dir="data/tcga-ut", magnification_key=5):
    data_dir = project_dir / data_dir
    present_subdataset = [f.name for f in data_dir.iterdir() if f.is_dir()]
    print(f"Present subdatasets: {present_subdataset}")

    wsi_dir_paths = list(data_dir.glob(f"./*/{magnification_key}/*/"))
    patient_ids = set([get_patient_from_tcga_id(f.name) for f in wsi_dir_paths])
    patient_ids = list(patient_ids)
    patient_ids.sort()

    train_patient_ids, val_patient_ids, test_patient_ids = split_patients(
        patient_ids, seed=42
    )

    tile_paths = list(data_dir.glob(f"./*/{magnification_key}/*/*.jpg"))
    print(f"Total tiles: {len(tile_paths)}")

    train_tile_paths = []
    val_tile_paths = []
    test_tile_paths = []

    for tile_path in tile_paths:
        patient_id = get_patient_from_tcga_id(tile_path.parent.name)
        if patient_id in train_patient_ids:
            train_tile_paths.append(tile_path)
        elif patient_id in val_patient_ids:
            val_tile_paths.append(tile_path)
        elif patient_id in test_patient_ids:
            test_tile_paths.append(tile_path)

    train_tiles = len(train_tile_paths)
    val_tiles = len(val_tile_paths)
    test_tiles = len(test_tile_paths)
    print(
        f"Split tiles: Train: {train_tiles} ({train_tiles / len(tile_paths):.1%}), "
        f"Val: {val_tiles} ({val_tiles / len(tile_paths):.1%}), "
        f"Test: {test_tiles} ({test_tiles / len(tile_paths):.1%})"
    )

    return train_tile_paths, val_tile_paths, test_tile_paths


def get_eval_transformations():
    """
    Returns the transformations for evaluation.
    """
    return transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
