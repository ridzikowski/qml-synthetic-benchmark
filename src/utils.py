import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

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
        # 1. Map requested complexity levels to make_classification hyperparameters
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
            },
            "very_complex": {
                "class_sep": 0.3,
                "flip_y": 0.10,
                "n_clusters_per_class": 2,
            },
        }

        params = complexity_presets.get(complexity.lower(), complexity_presets["medium"])
        n_clusters = params["n_clusters_per_class"]

        # Ensure 2**n_informative >= n_classes * n_clusters_per_class
        required_informative = int(np.ceil(np.log2(n_classes * n_clusters)))
        n_informative = max(2, required_informative)
        effective_n_features = max(n_features, max(target_features_int) + 1, n_informative)

        X_raw, y_raw = make_classification(
            n_samples=n_samples,
            n_features=effective_n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=0,
            n_clusters_per_class=n_clusters,
            class_sep=params["class_sep"],
            flip_y=params["flip_y"],
            random_state=random_state,
        )

        # 2. EPCO Concept: Apply linear projection matrix transformation X' = X @ P
        if use_projection_matrix:
            np.random.seed(random_state)
            # Draw random d x d transformation matrix P ~ N(0, 3) as described in EPCO
            d = X_raw.shape[1]
            projection_matrix = np.random.normal(loc=0.0, scale=3.0, size=(d, d))
            X_raw = X_raw @ projection_matrix

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