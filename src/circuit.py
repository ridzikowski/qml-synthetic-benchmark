import os
from typing import Tuple, Union
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import (
    EfficientSU2,
    RealAmplitudes,
    ZFeatureMap,
    ZZFeatureMap,
)


def spider_variational_circuit(
    num_qubits: int = 4, num_layers: int = 2, verbose: bool = False
) -> QuantumCircuit:
    theta = ParameterVector(r"$\theta$", num_qubits + num_qubits * num_layers)
    qc = QuantumCircuit(num_qubits, name="SpiderAnsatz")
    param_idx = 0
    mid = num_qubits // 2

    for q in range(num_qubits):
        qc.ry(theta[param_idx], q)
        param_idx += 1

    if verbose:
        qc.barrier()

    for _ in range(num_layers):
        for q in range(num_qubits):
            if q != mid:
                qc.cx(mid, q)

        if verbose:
            qc.barrier()

        for q in range(num_qubits):
            qc.rx(theta[param_idx], q % num_qubits)
            param_idx += 1

        if verbose:
            qc.barrier()

    return qc


def build_circuit(
    feature_map_type: str = "zfeaturemap",
    ansatz_type: str = "spider",
    num_qubits: int = 4,
    reps: int = 2,
    verbose: bool = False,
) -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    fmap_clean = (
        feature_map_type.lower().strip().replace("_", "").replace(" ", "")
    )
    ansatz_clean = (
        ansatz_type.lower().strip().replace("_", "").replace(" ", "")
    )

    # 1. Wybór Feature Mapy
    if fmap_clean in ["z", "zfeaturemap"]:
        feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
    elif fmap_clean in ["zz", "zzfeaturemap"]:
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
    else:
        raise ValueError(
            f"Nieznany typ feature mapy: '{feature_map_type}'. "
            f"Wybierz 'zfeaturemap' lub 'zzfeaturemap'."
        )

    if ansatz_clean in ["spider"]:
        ansatz = spider_variational_circuit(
            num_qubits=num_qubits, num_layers=reps, verbose=verbose
        )
    elif ansatz_clean in ["realamplitudes"]:
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps)
    elif ansatz_clean in ["efficientsu2"]:
        ansatz = EfficientSU2(num_qubits=num_qubits, reps=reps)
    else:
        raise ValueError(
            f"Nieznany typ anzatza: '{ansatz_type}'. "
            f"Wybierz 'spider', 'realamplitudes' lub 'efficientsu2'."
        )

    full_circuit = feature_map.compose(ansatz)

    full_circuit = full_circuit.decompose()
    full_circuit.measure_all()
    return full_circuit, feature_map, ansatz


def save_ansatz(
    ansatz: QuantumCircuit,
    filename: str = "ansatz.png",
    out_dir: str = "out",
    output_format: str = "mpl",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)

    if output_format == "text" or filename.endswith(".txt"):
        circuit_text = str(ansatz.draw(output="text"))
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(circuit_text)
    else:
        fig = ansatz.draw(output=output_format)
        if hasattr(fig, "savefig"):
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

    print(f"Obwód zapisano w: {save_path}")
    return save_path