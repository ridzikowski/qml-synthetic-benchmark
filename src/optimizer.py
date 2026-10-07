import os
import matplotlib.pyplot as plt
import numpy as np
from src.quantum import quantum_multi_job


def interpret_counts_as_class_probabilities(
    counts, num_classes=2, assignment_mode="block"
):
    logit_vector = np.zeros(num_classes)

    if not counts:
        return np.ones(num_classes) / num_classes

    bitstrings = list(counts.keys())

    if assignment_mode == "block":
        bitstrings = sorted(bitstrings, key=lambda b: int(b, 2))
        block_size = int(np.ceil(len(bitstrings) / num_classes))

        def class_index(i, bitstring):
            return min(i // block_size, num_classes - 1)

    elif assignment_mode == "modulo":

        def class_index(i, bitstring):
            return int(bitstring, 2) % num_classes

    else:
        raise ValueError("assignment_mode must be 'modulo' or 'block'")

    for i, bitstring in enumerate(bitstrings):
        logit_vector[class_index(i, bitstring)] += counts[bitstring]

    total = np.sum(logit_vector)
    if total > 0:
        return logit_vector / total
    else:
        return np.ones(num_classes) / num_classes


def compute_loss(y_true, y_pred_probs, eps=1e-10):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_pred_probs, dtype=float)

    if probs.ndim == 1:
        return float(-np.log(probs[y_true] + eps))

    N = len(y_true)
    correct_probs = probs[np.arange(N), y_true]
    return float(-np.mean(np.log(correct_probs + eps)))


def compute_accuracy(y_true, y_pred_probs):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_pred_probs, dtype=float)

    if probs.ndim == 1:
        preds = np.argmax(probs)
    else:
        preds = np.argmax(probs, axis=1)

    return float(np.mean(preds == y_true))


def predict_circuit(
    full_circuit,
    feature_map,
    ansatz,
    X,
    theta,
    backend=None,
    backend_mode="sampler",
    shots=1024,
    num_classes=2,
    assignment_mode="block",
    **kwargs,
):
    X = np.asarray(X, dtype=float)
    param_sets = [np.concatenate([x_i, theta]) for x_i in X]

    counts_list = quantum_multi_job(
        param_sets, full_circuit, backend, mode=backend_mode, shots=shots
    )

    probs = [
        interpret_counts_as_class_probabilities(
            c, num_classes=num_classes, assignment_mode=assignment_mode
        )
        for c in counts_list
    ]
    return np.array(probs)


def compute_gradient(
    full_circuit,
    X_batch,
    y_batch,
    theta,
    backend,
    backend_mode,
    num_classes=2,
    delta=1e-2,
    shots=1024,
    assignment_mode="block",
):
    num_params = len(theta)
    param_sets = []

    # Przygotowanie zestawów parametrów dla całej paczki batch
    for x_k in X_batch:
        for j in range(num_params):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[j] += delta
            theta_minus[j] -= delta
            param_sets.append(np.concatenate([x_k, theta_plus]))
            param_sets.append(np.concatenate([x_k, theta_minus]))

    counts_list = quantum_multi_job(
        param_sets, full_circuit, backend, mode=backend_mode, shots=shots
    )

    grad = np.zeros(num_params)
    batch_size = len(X_batch)

    idx = 0
    for k, y_k in enumerate(y_batch):
        for j in range(num_params):
            probs_plus = interpret_counts_as_class_probabilities(
                counts_list[idx], num_classes, assignment_mode
            )
            probs_minus = interpret_counts_as_class_probabilities(
                counts_list[idx + 1], num_classes, assignment_mode
            )

            loss_plus = compute_loss(y_k, probs_plus)
            loss_minus = compute_loss(y_k, probs_minus)

            grad[j] += (loss_plus - loss_minus) / (2 * delta)
            idx += 2

    return grad / batch_size


def optimize_circuit(
    full_circuit,
    feature_map,
    ansatz,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    optimizer_type="adam",
    learning_rate=0.01,
    batch_size=8,
    max_iters=40,
    seed=42,
    shots=1024,
    backend=None,
    backend_mode="sampler",
    verbose=True,
    assignment_mode="block",
    **kwargs,
):
    if seed is not None:
        np.random.seed(seed)

    num_params = len(ansatz.parameters)
    num_classes = len(np.unique(y_train))

    theta = np.random.uniform(-np.pi, np.pi, size=num_params)
    best_loss = float("inf")
    best_weights = theta.copy()

    history = {"loss": [], "accuracy": []}
    has_val = X_val is not None and y_val is not None
    if has_val:
        history["val_loss"] = []
        history["val_accuracy"] = []

    opt_type = optimizer_type.lower().strip()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for it in range(1, max_iters + 1):
        indices = np.random.choice(
            len(X_train), size=min(batch_size, len(X_train)), replace=False
        )
        X_batch, y_batch = X_train[indices], y_train[indices]

        # Obliczenie gradientu
        grad = compute_gradient(
            full_circuit,
            X_batch,
            y_batch,
            theta,
            backend,
            backend_mode,
            num_classes=num_classes,
            shots=shots,
            assignment_mode=assignment_mode,
        )

        # Aktualizacja wag
        if opt_type == "sgd":
            theta -= learning_rate * grad
        elif opt_type == "adam":
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**it)
            v_hat = v / (1 - beta2**it)
            theta -= learning_rate * m_hat / (np.sqrt(v_hat) + eps_adam)
        else:
            raise ValueError(f"Nieobsługiwany optymalizator: {optimizer_type}")

        # Ocena treningowa
        train_probs = predict_circuit(
            full_circuit,
            feature_map,
            ansatz,
            X_train,
            theta,
            backend=backend,
            backend_mode=backend_mode,
            shots=shots,
            num_classes=num_classes,
            assignment_mode=assignment_mode,
        )
        train_l = compute_loss(y_train, train_probs)
        train_a = compute_accuracy(y_train, train_probs)

        history["loss"].append(train_l)
        history["accuracy"].append(train_a)

        current_eval_loss = train_l

        if has_val:
            val_probs = predict_circuit(
                full_circuit,
                feature_map,
                ansatz,
                X_val,
                theta,
                backend=backend,
                backend_mode=backend_mode,
                shots=shots,
                num_classes=num_classes,
                assignment_mode=assignment_mode,
            )
            val_l = compute_loss(y_val, val_probs)
            val_a = compute_accuracy(y_val, val_probs)

            history["val_loss"].append(val_l)
            history["val_accuracy"].append(val_a)
            current_eval_loss = val_l

        if current_eval_loss < best_loss:
            best_loss = current_eval_loss
            best_weights = theta.copy()

        if verbose and (
            it % max(1, max_iters // 10) == 0 or it == 1 or it == max_iters
        ):
            msg = f"Iteracja {it}/{max_iters} | Train Loss: {train_l:.4f} | Train Acc: {train_a:.4f}"
            if has_val:
                msg += f" | Val Loss: {val_l:.4f} | Val Acc: {val_a:.4f}"
            print(msg)

    return theta, best_weights, history


def plot_history(
    history,
    filename="training_history.png",
    out_dir="out",
    title="Optimization Curves",
):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)

    iters = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(
        iters, history["loss"], label="Train Loss", color="#1f77b4", linewidth=2
    )
    if "val_loss" in history and len(history["val_loss"]) > 0:
        ax1.plot(
            iters,
            history["val_loss"],
            label="Val Loss",
            color="#ff7f0e",
            linestyle="--",
            linewidth=2,
        )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(
        iters,
        history["accuracy"],
        label="Train Accuracy",
        color="#2ca02c",
        linewidth=2,
    )
    if "val_accuracy" in history and len(history["val_accuracy"]) > 0:
        ax2.plot(
            iters,
            history["val_accuracy"],
            label="Val Accuracy",
            color="#d62728",
            linestyle="--",
            linewidth=2,
        )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path