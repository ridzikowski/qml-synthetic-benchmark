import pandas as pd
import warnings
from src.utils import load_dataset, stratified_kfold
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


if __name__ == "__main__":
    verbose = False
    test_1 = {
        "dataset_name": "cornus5f",
        "target_features": [0, 1, 2, 3, 4],
    }
    test_2 = {
        "dataset_name": "cornus4f",
        "target_features": [0, 1, 2, 3],
    }
    test_objects = [test_1, test_2]
    for test_object in test_objects:
        output_file = f'results/classical_baselines_results_{test_object["dataset_name"]}.csv'
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Model,Accuracy,Macro-F1,Macro-Prec,Macro-Recall\n")

        X, y = load_dataset(
            dataset_name=test_object["dataset_name"],
            target_features=test_object["target_features"],
        )

        if X is not None:
            seeds = [10, 25, 42, 55, 73, 88, 99, 101, 202, 303]
            all_rf_acc = []

            for seed in seeds:
                if verbose:
                    print(f"Seed: {seed}")
                evaluate_classical_models(X, y, n_splits=5, random_state=seed, output_file=output_file, verbose=verbose)


        results = calculate_average_metrics(output_file)
        print(f"\n--- AVERAGE METRICS OVER ALL SEEDS FOR {test_object["dataset_name"]} ---")
        print(results)

    # create_latex_table(results, output_file='results/classical_baselines_results_avg_3vs4_5_features.tex')

