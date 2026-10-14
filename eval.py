import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Importy z Twojego projektu
from src.backend import get_backend
from src.circuit import build_circuit
from src.optimizer import (
    compute_accuracy,
    compute_loss,
    predict_circuit,
)
from src.utils import load_dataset


def load_model_weights(weights_path: str) -> np.ndarray:
    """Wczytuje zapisane wagi modelu z pliku .npz."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Nie znaleziono pliku wag: {weights_path}")

    weights_data = np.load(weights_path)
    if "best_weights" in weights_data:
        return weights_data["best_weights"]
    elif "final_weights" in weights_data:
        return weights_data["final_weights"]
    else:
        first_key = weights_data.files[0]
        return weights_data[first_key]


def evaluate_on_backend(
    weights_path: str,
    dataset_name: str,
    target_backend: str,
    feature_map_type: str,
    ansatz_type: str,
    reps: int,
    seed: int,
    target_classes: list,
    target_features: list,
    shots: int = 1024,
):
    """Przeprowadza ewaluację wytrenowanego modelu na wybranym backendzie kwantowym."""
    # 1. Pobranie backendu i trybu pracy (sampler/device)
    backend_obj, backend_mode = get_backend(target_backend)

    # 2. Wczytanie danych i odtworzenie dokładnego podziału walidacyjnego (X_val, y_val)
    X, y = load_dataset(dataset_name, target_classes, target_features)

    unique_classes = sorted(list(set(y)))
    class_to_idx = {val: idx for idx, val in enumerate(unique_classes)}
    y_mapped = np.array([class_to_idx[val] for val in y], dtype=int)

    _, X_val, _, y_val = train_test_split(
        X, y_mapped, test_size=0.2, random_state=seed, stratify=y_mapped
    )

    # 3. Odtworzenie obwodu kwantowego
    full_qc, fmap, ansatz = build_circuit(
        feature_map_type=feature_map_type,
        ansatz_type=ansatz_type,
        num_qubits=X.shape[1],
        reps=reps,
    )

    # 4. Wczytanie wag modelu
    best_weights = load_model_weights(weights_path)

    # 5. Predykcja za pomocą funkcji predict_circuit
    val_probs = predict_circuit(
        full_qc,
        fmap,
        ansatz,
        X_val,
        best_weights,
        backend=backend_obj,
        backend_mode=backend_mode,
        shots=shots,
        num_classes=len(unique_classes),
    )

    # 6. Wyliczenie metryk oraz Confusion Matrix
    val_acc = compute_accuracy(y_val, val_probs)
    val_loss = compute_loss(y_val, val_probs)

    preds = np.argmax(val_probs, axis=1)
    cm = confusion_matrix(y_val, preds, labels=list(range(len(unique_classes))))

    # Wyliczanie dodatkowych metryk macro
    macro_prec = precision_score(y_val, preds, average="macro", zero_division=0)
    macro_rec = recall_score(y_val, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

    return val_acc, val_loss, macro_prec, macro_rec, macro_f1, cm, len(unique_classes)


def run_benchmark_evaluation(
    csv_path: str,
    dataset: str,
    ansatz: str,
    layers: int,
    seed_selection: str,
    target_backend: str,
    repeats: int,
    target_classes: list,
    target_features: list,
    feature_map_type: str,
    shots: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Nie znaleziono pliku CSV z wynikami: {csv_path}")

    df = pd.read_csv(csv_path)

    mask = (
        (df["dataset"].astype(str).str.lower() == dataset.lower())
        & (df["ansatz"].astype(str).str.lower() == ansatz.lower())
        & (df["layers"].astype(int) == int(layers))
    )

    if seed_selection.lower() != "all":
        mask = mask & (df["seed"].astype(str) == str(seed_selection))

    filtered_df = df[mask]

    if filtered_df.empty:
        print(
            f"❌ Brak dopasowań w pliku CSV dla: dataset='{dataset}', ansatz='{ansatz}', layers={layers}, seed='{seed_selection}'."
        )
        return

    print("=" * 80)
    print(f"🚀 ROZPOCZĘCIE EWALUACJI DLA BACKENDU: '{target_backend}'")
    print(
        f"   Dataset: {dataset} | Ansatz: {ansatz} | Warstwy: {layers} | Klasy: {target_classes} | Seed: {seed_selection} | Powtórzenia: {repeats}"
    )
    print("=" * 80)

    eval_results = []
    all_seed_cms = []
    n_classes_ref = len(target_classes)

    for _, row in filtered_df.iterrows():
        current_seed = int(row["seed"])
        weights_path = row["weights_path"]

        print(f"\n🔍 Testowanie modelu (Seed={current_seed}) na backendzie {target_backend}...")
        print(f"   Ścieżka do wag: {weights_path}")

        accs, losses, precs, recs, f1s, repeat_cms = [], [], [], [], [], []

        for rep in range(repeats):
            print(f"   • Uruchamianie powtórzenia [{rep + 1}/{repeats}]...")
            acc, loss, prec, rec, f1, cm, n_cls = evaluate_on_backend(
                weights_path=weights_path,
                dataset_name=dataset,
                target_backend=target_backend,
                feature_map_type=feature_map_type,
                ansatz_type=ansatz,
                reps=layers,
                seed=current_seed,
                target_classes=target_classes,
                target_features=target_features,
                shots=shots,
            )

            accs.append(acc)
            losses.append(loss)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            repeat_cms.append(cm)
            n_classes_ref = n_cls

        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
        mean_loss, std_loss = float(np.mean(losses)), float(np.std(losses))
        mean_prec, std_prec = float(np.mean(precs)), float(np.std(precs))
        mean_rec, std_rec = float(np.mean(recs)), float(np.std(recs))
        mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))

        avg_cm_for_seed = np.mean(repeat_cms, axis=0)
        all_seed_cms.append(avg_cm_for_seed)

        print(
            f"   ✅ Wyniki Seed={current_seed}: Accuracy = {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%, "
            f"F1 = {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%, "
            f"Prec = {mean_prec * 100:.2f}% ± {std_prec * 100:.2f}%, "
            f"Recall = {mean_rec * 100:.2f}% ± {std_rec * 100:.2f}%, "
            f"Loss = {mean_loss:.4f} ± {std_loss:.4f}"
        )

        eval_results.append(
            {
                "dataset": dataset,
                "ansatz": ansatz,
                "layers": layers,
                "seed": current_seed,
                "target_backend": target_backend,
                "repeats": repeats,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_macro_precision": mean_prec,
                "std_macro_precision": std_prec,
                "mean_macro_recall": mean_rec,
                "std_macro_recall": std_rec,
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "weights_path": weights_path,
            }
        )

    results_df = pd.DataFrame(eval_results)

    # Obliczenia podsumowujące (dla wszystkich testowanych seedów)
    overall_mean_acc = float(results_df["mean_accuracy"].mean())
    overall_std_acc = (
        float(results_df["mean_accuracy"].std())
        if len(results_df) > 1
        else float(results_df["std_accuracy"].iloc[0])
    )

    overall_mean_prec = float(results_df["mean_macro_precision"].mean())
    overall_std_prec = (
        float(results_df["mean_macro_precision"].std())
        if len(results_df) > 1
        else float(results_df["std_macro_precision"].iloc[0])
    )

    overall_mean_rec = float(results_df["mean_macro_recall"].mean())
    overall_std_rec = (
        float(results_df["mean_macro_recall"].std())
        if len(results_df) > 1
        else float(results_df["std_macro_recall"].iloc[0])
    )

    overall_mean_f1 = float(results_df["mean_macro_f1"].mean())
    overall_std_f1 = (
        float(results_df["mean_macro_f1"].std())
        if len(results_df) > 1
        else float(results_df["std_macro_f1"].iloc[0])
    )

    overall_mean_loss = float(results_df["mean_loss"].mean())
    overall_std_loss = (
        float(results_df["mean_loss"].std())
        if len(results_df) > 1
        else float(results_df["std_loss"].iloc[0])
    )

    cm_array = np.array(all_seed_cms)
    overall_cm_mean = np.mean(cm_array, axis=0)
    overall_cm_std = np.std(cm_array, axis=0)

    # 1. ZAPIS WYNIKÓW DO PLIKU CSV
    file_prefix = f"{dataset}_{ansatz}_{layers}l_{target_backend}_seed_{seed_selection}"
    out_csv_path = os.path.join(out_dir, f"eval_results_{file_prefix}.csv")

    summary_row = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "ansatz": ansatz,
                "layers": layers,
                "seed": "OVERALL_MEAN",
                "target_backend": target_backend,
                "repeats": repeats,
                "mean_accuracy": overall_mean_acc,
                "std_accuracy": overall_std_acc,
                "mean_macro_precision": overall_mean_prec,
                "std_macro_precision": overall_std_prec,
                "mean_macro_recall": overall_mean_rec,
                "std_macro_recall": overall_std_rec,
                "mean_macro_f1": overall_mean_f1,
                "std_macro_f1": overall_std_f1,
                "mean_loss": overall_mean_loss,
                "std_loss": overall_std_loss,
                "weights_path": "AGGREGATED",
            }
        ]
    )

    final_df = pd.concat([results_df, summary_row], ignore_index=True)
    final_df.to_csv(out_csv_path, index=False)
    print(f"\n💾 Wyniki ewaluacji zapisano do pliku CSV: {out_csv_path}")

    # 2. GENEROWANIE I ZAPIS WYKRESU CONFUSION MATRIX (PNG)
    plt.figure(figsize=(5.5, 4.5))

    annot_matrix = np.empty(overall_cm_mean.shape, dtype=object)
    for i in range(overall_cm_mean.shape[0]):
        for j in range(overall_cm_mean.shape[1]):
            annot_matrix[i, j] = f"{overall_cm_mean[i, j]:.1f}\n± {overall_cm_std[i, j]:.1f}"

    labels = [f"Class {c}" for c in target_classes[:n_classes_ref]]

    sns.heatmap(
        overall_cm_mean,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 11},
    )

    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()

    out_png_path = os.path.join(out_dir, f"confusion_matrix_{file_prefix}.png")
    plt.savefig(out_png_path, dpi=300)
    plt.close()

    print(f"🖼️ Wykres Confusion Matrix zapisano w: {out_png_path}")
    print("\n" + "=" * 80)
    print("PODSUMOWANIE:")
    print(f" • Średnia Dokładność (Accuracy): {overall_mean_acc * 100:.2f}% ± {overall_std_acc * 100:.2f}%")
    print(f" • Średni Macro Precision       : {overall_mean_prec * 100:.2f}% ± {overall_std_prec * 100:.2f}%")
    print(f" • Średni Macro Recall          : {overall_mean_rec * 100:.2f}% ± {overall_std_rec * 100:.2f}%")
    print(f" • Średni Macro F1-Score        : {overall_mean_f1 * 100:.2f}% ± {overall_std_f1 * 100:.2f}%")
    print(f" • Średnia Strata (Loss)        : {overall_mean_loss:.4f} ± {overall_std_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ewaluacja wyuczonych modeli QML na wskazanym backendzie kwantowym."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="benchmark_backends_results.csv",
        help="Plik CSV z wynikami i ścieżkami do wag (domyślnie: benchmark_backends_results.csv)",
    )
    parser.add_argument(
        "--dataset", type=str, default="iris", help="Nazwa datasetu (np. iris, cornus4f)"
    )
    parser.add_argument(
        "--ansatz", type=str, default="realamplitudes", help="Typ anzatza (np. realamplitudes, spider)"
    )
    parser.add_argument(
        "--layers", type=int, default=10, help="Liczba warstw (reps)"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="all",
        help="Wybór seeda ('all' lub konkretny seed np. '42')",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="FakeOdra",
        choices=["FakeOdra", "Odra5", "Garnet", "FakeGarnet", "Garnet_Hardware", "sampler", "Adonis", "Aphrodite"],
        help="Docelowy backend kwantowy (np. FakeOdra, Odra5, Garnet)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Liczba powtórzeń ewaluacji dla wyznaczenia odchylenia standardowego",
    )
    parser.add_argument(
        "--target_classes",
        nargs="+",
        default=["0", "1"],
        help="Etykiety docelowych klas (domyślnie dla 2 klas: '0' '1')",
    )
    parser.add_argument(
        "--target_features",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Indeksy cech wejściowych (domyślnie: 0 1 2 3)",
    )
    parser.add_argument(
        "--fmap",
        type=str,
        default="zzfeaturemap",
        help="Typ feature mapy (domyślnie: zzfeaturemap)",
    )
    parser.add_argument(
        "--shots", type=int, default=1024, help="Liczba shotów"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out_eval",
        help="Katalog wyjściowy do zapisu wyników CSV oraz PNG",
    )

    args = parser.parse_args()

    run_benchmark_evaluation(
        csv_path=args.csv,
        dataset=args.dataset,
        ansatz=args.ansatz,
        layers=args.layers,
        seed_selection=args.seed,
        target_backend=args.backend,
        repeats=args.repeats,
        target_classes=args.target_classes,
        target_features=args.target_features,
        feature_map_type=args.fmap,
        shots=args.shots,
        out_dir=args.out_dir,
    )
    #python eval.py --dataset easy3_synthetic --csv out_synthetic_easy3f/benchmark_backends_results.csv --ansatz spider --layers 5 --seed 42 --backend Odra5 