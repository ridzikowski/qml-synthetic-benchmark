import os.path

import pandas as pd
import warnings
from src.utils import load_dataset, stratified_kfold
import numpy as np
import time

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def evaluate_classical_models(X, y, n_splits=5, random_state=42, output_file=None, verbose = True):

    models = {
        "SVM": lambda: SVC(kernel="rbf", random_state=random_state),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=random_state),
        "K-NN": lambda: KNeighborsClassifier(n_neighbors=5),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=random_state)
    }

    results = {}

    for name, create_model in models.items():

        scores = {
            "acc": [],
            "f1": [],
            "prec": [],
            "rec": []
        }

        for X_train, X_test, y_train, y_test in stratified_kfold(
            X, y, seed=random_state, n_splits=n_splits
        ):
            model = create_model()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores["acc"].append(accuracy_score(y_test, y_pred))
            scores["f1"].append(f1_score(y_test, y_pred, average="macro", zero_division=0))
            scores["prec"].append(precision_score(y_test, y_pred, average="macro", zero_division=0))
            scores["rec"].append(recall_score(y_test, y_pred, average="macro", zero_division=0))

        results[name] = {
            metric: np.mean(values)
            for metric, values in scores.items()
        }

        r = results[name]
        if verbose:
            print(
                f"{name:<20} | "
                f"Acc: {r['acc']:.4f} | "
                f"F1: {r['f1']:.4f} | "
                f"Prec: {r['prec']:.4f} | "
                f"Recall: {r['rec']:.4f}"
            )

        if output_file:
            with open(output_file, "a") as f:
                f.write(
                    f"{name},{r['acc']:.4f},{r['f1']:.4f},"
                    f"{r['prec']:.4f},{r['rec']:.4f}\n"
                )

    return results

def calculate_average_metrics(csv_file_path):
    column_names = ['Model', 'Accuracy', 'Macro-F1', 'Macro-Prec', 'Macro-Recall']

    df = pd.read_csv(csv_file_path, header=None, names=column_names)

    df['Model'] = df['Model'].astype(str).str.strip()

    metrics = ['Accuracy', 'Macro-F1', 'Macro-Prec', 'Macro-Recall']
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=metrics)

    avg_metrics = df.groupby('Model').mean(numeric_only=True).round(4)
    avg_metrics = avg_metrics.sort_values(by='Accuracy', ascending=False)
    return avg_metrics


def create_latex_table(results, output_file='results/classical_baselines_results_avg.tex', caption="M", label="tab:M"):
    with open(output_file, 'w') as f:
        f.write("\\begin{table*}[h!]\n")
        f.write("\\centering\n")
        f.write("\\caption{")
        f.write(caption)
        f.write("}\n")

        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write(
            "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Macro-F1} & \\textbf{Macro-Prec} & \\textbf{Macro-Recall} \\\\\n")
        f.write("\\midrule\n")
        for model, metrics in results.iterrows():
            f.write(
                f"{model}&{metrics['Accuracy']}&{metrics['Macro-F1']}&{metrics['Macro-Prec']}&{metrics['Macro-Recall']}\\\\ \n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

        f.write("\\label{")
        f.write(label)
        f.write("}\n")
        f.write("\\end{table*}\n")

def calculate_means():
    folder = Path("results")
    metric = "Accuracy"
    results = []

    for file in sorted(folder.glob("classical_baselines_results_*.csv")):
        df = pd.read_csv(file)

        mean_score = df.groupby("Model")[metric].mean().mean()
        mean_f1_score = df.groupby("Model")["Macro-F1"].mean().mean()
        mean_prec_score = df.groupby("Model")["Macro-Prec"].mean().mean()
        mean_recall_score = df.groupby("Model")["Macro-Recall"].mean().mean()

        name = file.stem.replace(
            "classical_baselines_results_", ""
        )

        results.append({
            "Dataset": name,
            "Mean": mean_score,
            "Mean Macro-F1": mean_f1_score,
            "Mean Macro-Prec": mean_prec_score,
            "Mean Macro-Recall": mean_recall_score
        })

    summary = pd.DataFrame(results)
    print("\n--- MEAN ACCURACY SUMMARY ---")
    print(summary)

    plt.figure(figsize=(12, 7))

    bars = plt.barh(
        summary["Dataset"],
        summary["Mean"]
    )

    plt.bar_label(
        bars,
        labels=[f"{v:.2%}" for v in summary["Mean"]],
        padding=5
    )

    plt.xlim(0, 1.10)
    plt.xlabel("Mean Accuracy")
    plt.ylabel("Dataset")
    plt.title("Mean Classification Accuracy by Dataset")
    plt.gca().invert_yaxis()

    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig("mean_accuracy.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    # verbose = False
    # datasets = [
    #     "synthetic_complex_three_features",
    #     "synthetic_medium_three_features",
    #     "synthetic_easy_three_features",
    #     "synthetic_complex_four_features",
    #     "synthetic_medium_four_features",
    #     "synthetic_easy_four_features",
    #     "synthetic_complex_five_features",
    #     "synthetic_medium_five_features",
    #     "synthetic_easy_five_features",
    # ]
    # target_classes = ["0", "1", "2", "3", "4"]
    # target_features = [0,1,2]
    

    # for dataset in datasets:
    #     if "four" in dataset:
    #         target_features = [0, 1, 2, 3]
    #     elif "five" in dataset:
    #         target_features = [0, 1, 2, 3, 4]
    #     else:
    #         target_features = [0, 1, 2]
    #     start_time = time.time()
    #     output_file = f'results/classical_baselines_results_{dataset}.csv'
    #     if output_file:
    #         with open(output_file, 'w') as f:
    #             f.write("Model,Accuracy,Macro-F1,Macro-Prec,Macro-Recall\n")
    #     X, y = load_dataset(
    #         dataset_name=dataset,
    #         target_features=target_features,
    #         target_classes=target_classes,
    #     )

    #     if X is not None:
    #         seeds = [10, 25, 42, 55, 73, 88, 99, 101, 202, 303]
    #         all_rf_acc = []

    #         for seed in seeds:
    #             if verbose:
    #                 print(f"Seed: {seed}")
    #             evaluate_classical_models(X, y, n_splits=5, random_state=seed, output_file=output_file, verbose=verbose)


    #     results = calculate_average_metrics(output_file)
    #     end_time = time.time()-start_time
    #     print(f"\n--- AVERAGE METRICS OVER ALL SEEDS FOR {dataset}, ELAPSED TIME {end_time:.2f} s ---")
    #     print(results)
    calculate_means()
    # create_latex_table(results, output_file='results/classical_baselines_results_avg_3vs4_5_features.tex')

