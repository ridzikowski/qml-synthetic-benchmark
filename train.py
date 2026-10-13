import itertools
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.backend import draw_backend, get_backend
from src.circuit import build_circuit
from src.optimizer import (
    compute_accuracy,
    compute_loss,
    optimize_circuit,
    plot_history,
    predict_circuit,
)
from src.utils import load_dataset


def train(
    dataset_name: str = "iris",
    target_classes: list = ["0", "1"],
    target_features: list =  [0, 1, 2], # [1, 2, 3, 4],
    feature_map_type: str = "zzfeaturemap",
    ansatz_type: str = "realamplitudes",
    reps: int = 2,
    optimizer_type: str = "adam",
    backend_type: str = "FakeOdra",
    shots: int = 1024,
    seed: int = 42,
    max_iters: int = 40,
    batch_size: int = 8,
    out_dir: str = "out",
    verbose: bool = True,
) -> dict:

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Pobranie backendu i trybu z backend.py
    backend_obj, backend_mode = get_backend(backend_type)

    # Opcjonalnie wyrysowanie mapy połączeń
    try:
        draw_backend(backend_obj, out_dir)
    except Exception:
        pass

    # 2. Wczytanie danych i podział
    X, y = load_dataset(dataset_name, target_classes, target_features)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # 3. Budowa obwodu
    full_qc, fmap, ansatz = build_circuit(
        feature_map_type=feature_map_type,
        ansatz_type=ansatz_type,
        num_qubits=X.shape[1],
        reps=reps,
    )

    # 4. Optymalizacja na wskazanym backendzie kwantowym
    start_time = time.time()
    final_w, best_w, history = optimize_circuit(
        full_circuit=full_qc,
        feature_map=fmap,
        ansatz=ansatz,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        optimizer_type=optimizer_type,
        batch_size=batch_size,
        max_iters=max_iters,
        seed=seed,
        shots=shots,
        backend=backend_obj,
        backend_mode=backend_mode,
        verbose=verbose,
    )
    training_time = time.time() - start_time
    # 5. Ewaluacja walidacyjna
    val_probs = predict_circuit(
        full_qc,
        fmap,
        ansatz,
        X_val,
        best_w,
        num_classes=len(target_classes),
        backend=backend_obj,
        backend_mode=backend_mode,
        shots=shots,
    )
    final_val_acc = compute_accuracy(y_val, val_probs)
    final_val_loss = compute_loss(y_val, val_probs)

    # 6. Zapis plików (.npz, .csv, .png)
    base_name = f"{timestamp}_{dataset_name}_{backend_type}_{optimizer_type}_{ansatz_type}_{reps}l_s{seed}"

    weights_path = os.path.join(out_dir, f"weights_{base_name}.npz")
    np.savez(weights_path, best_weights=best_w, final_weights=final_w)

    history_path = os.path.join(out_dir, f"history_{base_name}.csv")
    pd.DataFrame(history).to_csv(history_path, index_label="iteration")

    plot_path = plot_history(
        history,
        filename=f"plot_{base_name}.png",
        out_dir=out_dir,
        title=f"{dataset_name.upper()} | Backend: {backend_type} | {ansatz_type} ({reps}L)",
    )

    return {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "backend": backend_type,
        "optimizer": optimizer_type,
        "ansatz": ansatz_type,
        "layers": reps,
        "seed": seed,
        "training_time_sec": training_time,
        "val_loss": final_val_loss,
        "val_acc": final_val_acc,
        "weights_path": weights_path,
        "history_path": history_path,
        "plot_path": plot_path,
    }


def benchmark(
    datasets: list = ["cornus4f"],
    target_features: list =  [0, 1, 2, 3],
    backend_types: list = ["sampler", "FakeOdra", "FakeGarnet"],
    ansatze: list = ["spider", "realamplitudes"],
    layers_list: list = [1, 2],
    optimizers: list = ["adam"],
    seeds: list = [42],
    out_dir: str = "out",
    max_iters: int = 40,
    batch_size: int = 8,
):
    results = []
    total_runs = (
        len(datasets)
        * len(backend_types)
        * len(ansatze)
        * len(layers_list)
        * len(optimizers)
        * len(seeds)
    )

    print(
        f"\n🚀 Uruchamianie benchmarku: łącznie {total_runs} kombinacji"
        " eksperymentów...\n"
    )

    csv_path = os.path.join(out_dir, "benchmark_backends_results.csv")

    # Usuwamy poprzedni plik wyników (jeśli istnieje), aby nie mieszać starych danych
    if os.path.exists(csv_path):
        os.remove(csv_path)
    run_idx = 1
    for ds, b_type, ans, l, opt, sd in itertools.product(
        datasets, backend_types, ansatze, layers_list, optimizers, seeds
    ):
        print(
            f"--- Run [{run_idx}/{total_runs}]: DS={ds} | Backend={b_type} |"
            f" Ansatz={ans} | L={l} | Opt={opt} | Seed={sd} ---"
        )

        res = train(
            dataset_name=ds,
            target_features=target_features,
            backend_type=b_type,
            ansatz_type=ans,
            reps=l,
            optimizer_type=opt,
            seed=sd,
            out_dir=out_dir,
            verbose=True,
            max_iters=max_iters,
            batch_size=batch_size,
        )
        results.append(res)
        # Zapis pojedynczego wyniku na bieżąco do pliku CSV
        pd.DataFrame([res]).to_csv(
            csv_path,
            mode="a",
            header=not os.path.exists(csv_path),
            index=False,
        )
        print(
            f"   Finished [{run_idx}/{total_runs}]: Val Loss ="
            f" {res['val_loss']:.4f} | Val Acc = {res['val_acc'] * 100:.2f}%\n"
        )
        run_idx += 1

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "benchmark_backends_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ BENCHMARK ZAKOŃCZONY! Raport zapisano w: {csv_path}\n")
    return pd.DataFrame(results)


if __name__ == "__main__":
    df_report = benchmark(
        datasets=["easy3_synthetic"],
        target_features =  [0, 1, 2],
        backend_types=["FakeOdra"],
        ansatze=["realamplitudes"],
        layers_list=[10],
        optimizers=["adam"],
        seeds=[10, 42],
        max_iters=250,
        batch_size=8,
        out_dir = "out_synthetic_easy3f"
    )