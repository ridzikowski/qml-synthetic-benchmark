import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from statsmodels.multivariate.manova import MANOVA
from src.utils import load_dataset


def run_analysis_and_visualization(
    dataset_name: str = "synthetic",
    target_classes: list = ["0", "1", "2", "4"],
    target_features: list = [0, 1, 2, 3],
    **loader_kwargs,
):
    print("=" * 60)
    print(f"RUNNING ANALYSIS FOR: '{dataset_name}'")
    print("=" * 60)

    # 1. Load dataset via load_dataset function
    X, y = load_dataset(
        dataset_name=dataset_name,
        target_classes=target_classes,
        target_features=target_features,
        **loader_kwargs,
    )

    if X is None or len(X) == 0:
        print("Data loading failed or returned an empty set.")
        return

    # Convert normalized numpy arrays to DataFrame for statsmodels & pandas operations
    feature_names = [f"feature_{f}" for f in target_features]
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y

    unique_classes = np.unique(y)

    # # 2. Compute Mean and Standard Deviation per class
    # print("\n--- FEATURE DESCRIPTIVE STATISTICS ---")
    # for cls in unique_classes:
    #     filtered_data = df[df["label"] == cls]
    #     print(f"\nClass / Label: {cls}")
    #     for feature in feature_names:
    #         mean = filtered_data[feature].mean()
    #         std_dev = filtered_data[feature].std()
    #         print(
    #             f"  Feature '{feature}': Mean = {mean:.4f}, Std Dev = {std_dev:.4f}"
    #         )

    # # 3. Shapiro-Wilk Test for Normality
    # print("\n--- SHAPIRO-WILK TEST ---")
    # for cls in unique_classes:
    #     filtered_data = df[df["label"] == cls]
    #     print(f"\nClass / Label: {cls}")
    #     for feature in feature_names:
    #         data_to_test = filtered_data[feature].dropna()
    #         if len(data_to_test) >= 3:
    #             stat, p_val = stats.shapiro(data_to_test)
    #             print(f"  {feature}: stat={stat:.4f}, p-value={p_val:.4e}")
    #         else:
    #             print(
    #                 f"  {feature}: Not enough samples (n < 3) for Shapiro test"
    #             )

    # # 4. One-Way ANOVA
    # print("\n--- ONE-WAY ANOVA ---")
    # for feature in feature_names:
    #     groups = [
    #         df[df["label"] == cls][feature].dropna() for cls in unique_classes
    #     ]
    #     if len(groups) > 1 and all(len(g) > 0 for g in groups):
    #         stat, p_val = stats.f_oneway(*groups)
    #         print(f"{feature}: F-stat = {stat:.4f}, p-value = {p_val:.4e}")
    #     else:
    #         print(f"{feature}: Insufficient class variation for ANOVA")

    # # 5. MANOVA
    # print("\n--- MANOVA ---")
    # if len(feature_names) >= 1 and len(unique_classes) > 1:
    #     formula = f"{' + '.join(feature_names)} ~ C(label)"
    #     try:
    #         manova = MANOVA.from_formula(formula, data=df)
    #         print(manova.mv_test())
    #     except Exception as e:
    #         print(f"Error executing MANOVA: {e}")

    # # 6. Feature Correlation Matrix
    # plt.figure(figsize=(8, 6))
    # corr_matrix = df[feature_names].corr()
    # sns.heatmap(
    #     corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1
    # )
    # plt.title(f"Feature Correlation Matrix - {dataset_name}")
    # plt.tight_layout()
    # corr_filename = f"correlation_matrix_{dataset_name}.png"
    # plt.savefig(corr_filename, dpi=300)
    # plt.close()
    # print(f"\nSaved plot: {corr_filename}")

    # 7. Decision Boundaries (SVM on 2D PCA projection)
    if X.shape[1] >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        clf.fit(X_pca, y)

        x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
        y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Dynamic colormaps for supporting arbitrary class counts
        n_classes = len(unique_classes)
        color_palette = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 3)))[
            :n_classes
        ]

        cmap_points = ListedColormap(color_palette)
        # Add opacity to background regions
        bg_colors = [
            (r, g, b, 0.3) for r, g, b, _ in color_palette
        ]
        cmap_background = ListedColormap(bg_colors)

        # Convert labels to contiguous 0..N-1 indices for smooth colormap rendering
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        y_idx = np.array([class_to_idx[val] for val in y])
        Z_idx = np.vectorize(class_to_idx.get)(Z)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z_idx, alpha=0.3, cmap=cmap_background)
        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y_idx,
            cmap=cmap_points,
            edgecolors="k",
            s=50,
        )

        # handles = [
        #     plt.Line2D(
        #         [0],
        #         [0],
        #         marker="o",
        #         color="w",
        #         markerfacecolor=color_palette[i],
        #         markersize=10,
        #         markeredgecolor="k",
        #     )
        #     for i in range(n_classes)
        # ]
        # plt.legend(
        #     handles,
        #     [f"Class {c}" for c in unique_classes],
        #     title="Classes",
        #     loc="best",
        # )

        # plt.xlabel("Principal Component 1")
        # plt.ylabel("Principal Component 2")
        # plt.title(
        #     f"Decision Boundaries (SVM on 2D PCA) - {dataset_name}"
        # )
        plt.tight_layout()
        boundary_filename = f"results/decision_boundaries/decision_boundaries_{dataset_name}.png"
        plt.savefig(boundary_filename, dpi=300)
        plt.close()
        print(f"Saved plot: {boundary_filename}")


if __name__ == "__main__":
    # Test 1: Standard Iris Dataset
    # run_analysis_and_visualization(
    #     dataset_name="iris",
    #     target_classes=["0", "1", "2"],
    #     target_features=[0, 1, 2, 3],
    # )

    # Test 2: Synthetic Dataset generated on-the-fly
    datasets = [
        "synthetic_complex_three_features",
        "synthetic_medium_three_features",
        "synthetic_easy_three_features",
        "synthetic_complex_four_features",
        "synthetic_medium_four_features",
        "synthetic_easy_four_features",
        "synthetic_complex_five_features",
        "synthetic_medium_five_features",
        "synthetic_easy_five_features",
    ]
    for dataset in datasets:
        if dataset in [
            "synthetic_complex_three_features",
            "synthetic_medium_three_features",
            "synthetic_easy_three_features",
        ]:
            target_features = [0, 1, 2]
        elif dataset in [
            "synthetic_complex_four_features",
            "synthetic_medium_four_features",
            "synthetic_easy_four_features",
        ]:
            target_features = [0, 1, 2, 3]
        elif dataset in [
            "synthetic_complex_five_features",
            "synthetic_medium_five_features",
            "synthetic_easy_five_features",
        ]:
            target_features = [0, 1, 2, 3, 4]
        else:
            target_features = [0, 1, 2] 

        run_analysis_and_visualization(
            dataset_name=dataset,
            target_features=target_features,
            target_classes=["0", "1", "2", "3", "4"]
        )