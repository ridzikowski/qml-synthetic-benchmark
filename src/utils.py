import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import MinMaxScaler


def load_dataset(
    dataset_name: str = "iris",
    target_classes: list = ["0", "1"],
    target_features: list = [0, 1, 2, 3],
    n_samples: int = 1000,
    n_classes: int = 2,
    n_features: int = 4,
    n_clusters_per_class: int = 1,
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

    elif any(
        k in name_clean
        for k in ["synthetic", "generated", "make_classification", "synthetic_classification"]
    ):
        # Calculate n_informative so 2**n_informative >= n_classes * n_clusters_per_class
        required_informative = int(np.ceil(np.log2(n_classes * n_clusters_per_class)))
        n_informative = max(2, required_informative)

        # Ensure n_features is large enough for requested indices and informative features
        effective_n_features = max(n_features, max(target_features_int) + 1, n_informative)

        X_raw, y_raw = make_classification(
            n_samples=n_samples,
            n_features=effective_n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=0,
            n_clusters_per_class=n_clusters_per_class,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Nieznany dataset: '{dataset_name}'. Wybierz 'iris', 'coronus f4', 'coronus 5f' lub 'synthetic'."
        )

    # Bezpieczna filtracja klas niezależnie od typu (int vs str)
    y_str = y_raw.astype(str)
    target_classes_str = [str(c) for c in target_classes]

    mask = np.isin(y_str, target_classes_str)
    X_filtered = X_raw[mask]
    y_filtered = y_raw[mask]

    # Wybór cech
    X_selected = X_filtered[:, target_features_int]

    # Normalizacja do zakresu [0, 1]
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_selected)

    return X_normalized, y_filtered