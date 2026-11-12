import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def make_and_save_synthetic(
    n_samples_per_class: int = 500,
    n_features: int = 4,
    complexity: str = "medium",  # "easy", "medium", "complex", "very_complex"
    use_projection_matrix: bool = False,  # Applies EPCO-style linear projection
    random_state: int = 42,
    filename: str = "datasets/test.csv"
):
    n_classes = 5 #hardcodowane póki co, w odczycie do wyboru
    # class_sep ułatwia, flip_y utrudnia (przemieszanie klas), clusters_per_class utrudnia (ile grup w klasie)
    complexity_presets = {
        "easy": {
            "class_sep": 1.8,
            "flip_y": 0.0,
            "n_clusters_per_class": 1,
        },
        "medium": {
                "class_sep": 1.0,
                "flip_y": 0.01,
                "n_clusters_per_class": 1,
            },
        "complex": {
            "class_sep": 0.6,
            "flip_y": 0.05,
            "n_clusters_per_class": 1,
        }
            
    }

    params = complexity_presets.get(complexity.lower(), complexity_presets["medium"])
    n_clusters = params["n_clusters_per_class"]

    # Ensure 2**n_informative >= n_classes * n_clusters_per_class
    required_informative = int(np.ceil(np.log2(n_classes * n_clusters)))
    n_informative = max(2, required_informative)
    effective_n_features = max(n_features, n_informative)

    n_samples=n_samples_per_class*n_classes
    X_raw, y_raw = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=n_clusters,
        class_sep=params["class_sep"],
        flip_y=params["flip_y"],
        random_state=random_state,
    )
    print("Informative :")
    print(n_informative)
    print("All: ")
    print(n_features)

    # 2. EPCO Concept: Apply linear projection matrix transformation X' = X @ P
    if use_projection_matrix:
        np.random.seed(random_state)
        # Draw random d x d transformation matrix P ~ N(0, 3) as described in EPCO
        d = X_raw.shape[1]
        projection_matrix = np.random.normal(loc=0.0, scale=3.0, size=(d, d))
        X_raw = X_raw @ projection_matrix
    
    # 1. Create feature column names (e.g., Feature_1, Feature_2, ...)
    n_features = X_raw.shape[1]
    feature_names = [f"Feature_{i+1}" for i in range(n_features)]

    # 2. Build the DataFrame from X
    df = pd.DataFrame(X_raw, columns=feature_names)

    # 3. Add the target vector y as a new column
    df["target"] = y_raw

    # 4. Save to CSV
    df.to_csv(filename, index=False)

def load_dataset(
    dataset_name: str = "iris",
    target_classes: list = ["0", "1"],
    target_features: list = [0, 1, 2, 3],
    n_samples: int = 1000,
    n_classes: int = 2,
    n_features: int = 4,
    complexity: str = "medium",  # "easy", "medium", "complex", "very_complex"
    use_projection_matrix: bool = False,  # Applies EPCO-style linear projection
    random_state: int = 42,
    partition: str | None = None,
    return_groups: bool = False,
):
    name_clean = dataset_name.lower().strip().replace(" ", "_").replace("-", "_")
    target_features_int = [int(f) for f in target_features]

    if name_clean == "iris":
        iris = load_iris()
        X_raw = iris.data
        y_raw = iris.target

    elif "cornus" in name_clean:
        if "4" in name_clean or "four" in name_clean:
            file_path = "datasets/cornus_four_features.csv"
        elif "5" in name_clean or "five" in name_clean:
            file_path = "datasets/cornus_five_features.csv"
        else:
            raise ValueError(f"Nieokreślony wariant Cornus w nazwie: {dataset_name}")

        df = pd.read_csv(file_path, header=1)
        X_raw = df.iloc[:, :-1].values.astype(float)
        y_raw = df.iloc[:, -1].values

    elif "synthetic" in name_clean:
        if "easy" in name_clean:
            if "3" in name_clean or "three" in name_clean:
                file_path = "datasets/synthetic_easy_three_features.csv"
            elif "4" in name_clean or "four" in name_clean:
                file_path = "datasets/synthetic_easy_four_features.csv"
            elif "5" in name_clean or "five" in name_clean:
                file_path = "datasets/synthetic_easy_five_features.csv"
            else:
                raise ValueError(f"Nieokreślony wariant Synthetic w nazwie: {dataset_name}")
        elif "medium" in name_clean:
            if "3" in name_clean or "three" in name_clean:
                file_path = "datasets/synthetic_medium_three_features.csv"
            elif "4" in name_clean or "four" in name_clean:
                file_path = "datasets/synthetic_medium_four_features.csv"
            elif "5" in name_clean or "five" in name_clean:
                file_path = "datasets/synthetic_medium_five_features.csv"
            else:
                raise ValueError(f"Nieokreślony wariant Synthetic w nazwie: {dataset_name}")
        elif "complex" in name_clean:
            if "3" in name_clean or "three" in name_clean:
                file_path = "datasets/synthetic_complex_three_features.csv"
            elif "4" in name_clean or "four" in name_clean:
                file_path = "datasets/synthetic_complex_four_features.csv"
            elif "5" in name_clean or "five" in name_clean:
                file_path = "datasets/synthetic_complex_five_features.csv"
            else:
                raise ValueError(f"Nieokreślony wariant Synthetic w nazwie: {dataset_name}")
        else:
            raise ValueError(f"Nieokreślony wariant Synthetic w nazwie: {dataset_name}")

        df = pd.read_csv(file_path, header=1)
        X_raw = df.iloc[:, :-1].values.astype(float)
        y_raw = df.iloc[:, -1].values

    elif (file_path := (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / f"{name_clean.removesuffix('.csv')}.csv"
    )).is_file():

        df = pd.read_csv(file_path)
        if name_clean.startswith("speech_ae_") and "partition" in df.columns:
            df = df[df["partition"] == "fit"].copy()

        if df.empty:
            raise ValueError(f"Pusty dataset: {file_path}")

        if "target" not in df.columns:
            raise ValueError(f"Brak kolumny 'target' w {file_path}")

        metadata_cols = {"target", "partition", "speaker_id", "sample_id"}
        feature_cols = [
            col for col in df.columns
            if col not in metadata_cols
        ]

        if not feature_cols:
            raise ValueError("Brak kolumn z cechami")

        X_raw = df[feature_cols].to_numpy(dtype=float)
        y_raw = df["target"].to_numpy()

        if not np.isfinite(X_raw).all():
            raise ValueError("Dataset zawiera NaN lub Inf")


    else:
        raise ValueError(
            f"Nieznany dataset: '{dataset_name}'. Wybierz 'iris', 'cornus f4', 'cornus 5f' lub 'synthetic'."
        )

    # Bezpieczna filtracja klas niezależnie od typu (int vs str)
    y_str = y_raw.astype(str)
    target_classes_str = [str(c) for c in target_classes]

    mask = np.isin(y_str, target_classes_str)
    X_filtered = X_raw[mask]
    y_filtered = y_raw[mask]

    # Wybór cech
    X_selected = X_filtered[:, target_features_int]

    return X_selected, y_filtered

def stratified_kfold(X, y, seed=42, n_splits=5):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        yield X_train, X_test, y_train, y_test


