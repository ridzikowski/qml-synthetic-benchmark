import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_dataset


# 1. Define a Simple Neural Network (Multi-Layer Perceptron)
class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 2. Helper to train and evaluate the PyTorch model
def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.01,
):
    num_classes = len(np.unique(y))
    input_dim = X.shape[1]

    # Convert y to integer labels
    unique_labels = np.unique(y)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_idx = np.array([label_map[item] for item in y])

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )

    # Convert to PyTorch Tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )

    model = SimpleNN(input_dim=input_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Test accuracy evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_te_t)
            _, test_preds = torch.max(test_outputs, 1)
            test_acc = (test_preds == y_te_t).sum().item() / y_te_t.size(0)

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_train_acc)
        history["test_acc"].append(test_acc)

    return model, history, (X_test, y_test)


# 3. Decision Boundary Visualizer using 2D PCA Projection
def plot_pca_decision_boundary(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    complexity: str,
    output_dir: str,
):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    # Train a fast 2D proxy mesh using PCA coordinates to plot boundary
    x_min, x_max = X_pca[:, 0].min() - 0.2, X_pca[:, 0].max() + 0.2
    y_min, y_max = X_pca[:, 1].min() - 0.2, X_pca[:, 1].max() + 0.2
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
    )

    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    # Inverse transform grid points to full input feature dimension
    grid_orig = pca.inverse_transform(grid_2d)

    model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_orig, dtype=torch.float32)
        preds = torch.argmax(model(grid_tensor), dim=1).numpy()

    Z = preds.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_test,
        cmap="coolwarm",
        edgecolors="k",
        s=40,
    )
    plt.colorbar(scatter, label="Class")
    plt.title(f"Feature Space & NN Boundary (PCA 2D) - {complexity.upper()}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()

    save_path = os.path.join(
        output_dir, f"feature_boundary_{complexity}.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()


# 4. Main Execution
if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    complexities = ["easy", "medium", "complex", "very_complex"]
    num_classes = 3
    target_features = [0, 1, 2, 3]

    fig_curves, axes = plt.subplots(
        len(complexities), 2, figsize=(12, 3 * len(complexities))
    )

    for idx, comp in enumerate(complexities):
        print(f"--> Training Neural Network on Complexity: '{comp}'")

        # Load dataset
        X, y = load_dataset(
            dataset_name="synthetic",
            target_classes=["0", "1", "2"],
            target_features=target_features,
            n_samples=1200,
            n_classes=num_classes,
            n_features=4,
            complexity=comp,
            random_state=42,
        )

        # Train model
        model, history, (X_test, y_test) = train_and_evaluate(
            X, y, epochs=100, batch_size=32, lr=0.01
        )

        # Plot Loss Curve
        axes[idx, 0].plot(
            history["train_loss"], label="Train Loss", color="crimson"
        )
        axes[idx, 0].set_title(f"Loss Curve ({comp.upper()})")
        axes[idx, 0].set_xlabel("Epochs")
        axes[idx, 0].set_ylabel("Cross Entropy Loss")
        axes[idx, 0].grid(True, linestyle="--", alpha=0.5)

        # Plot Accuracy Curves
        axes[idx, 1].plot(
            history["train_acc"], label="Train Acc", color="blue"
        )
        axes[idx, 1].plot(
            history["test_acc"],
            label="Test Acc",
            color="green",
            linestyle="--",
        )
        axes[idx, 1].set_title(f"Accuracy ({comp.upper()})")
        axes[idx, 1].set_xlabel("Epochs")
        axes[idx, 1].set_ylabel("Accuracy")
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, linestyle="--", alpha=0.5)

        # Generate and save feature visualization plot
        plot_pca_decision_boundary(
            model, X_test, y_test, comp, output_dir
        )

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_loss_accuracy_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    print(f"\nSaved combined metrics plot: {curves_path}")
    print(f"Saved feature boundary plots inside: '{output_dir}/'")