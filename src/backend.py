import os
from dotenv import load_dotenv
from qiskit_aer.primitives import SamplerV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler import CouplingMap

try:
    from iqm.qiskit_iqm import IQMProvider, IQMFakeAdonis, IQMFakeAphrodite, IQMFakeGarnet
except ImportError:
    pass
from rustworkx.visualization import mpl_draw
from rustworkx import spring_layout

def get_backend(backend_type="FakeOdra"):
    if backend_type == "sampler":
        return SamplerV2(), "sampler"

    elif backend_type == "FakeOdra":
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.02, 2), ["cx"]
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.002, 1), ["h", "rx", "ry", "rz"]
        )
        coupling_map = CouplingMap([[0, 2], [1, 2], [2, 3], [2, 4]])
        noisy_sampler = SamplerV2(
            options=dict(
                backend_options=dict(
                    noise_model=noise_model, coupling_map=coupling_map
                )
            )
        )
        return noisy_sampler, "sampler"

    elif backend_type == "FakeGarnet":
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.03, 2), ["cx"]
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(0.004, 1), ["h", "rx", "ry", "rz"]
        )
        coupling_map = CouplingMap([
            [0, 1],
            [0, 3],
            [1, 4],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [7, 2],
            [8, 3],
            [9, 4],
            [10, 5],
            [11, 6],
            [7, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [7, 12],
            [8, 13],
            [9, 14],
            [15, 10],
            [16, 11],
            [12, 13],
            [13, 14],
            [15, 14],
            [15, 16],
            [17, 13],
            [18, 14],
            [15, 19],
            [17, 18],
            [18, 19],
        ])
        noisy_sampler = SamplerV2(
            options=dict(
                backend_options=dict(
                    noise_model=noise_model, coupling_map=coupling_map
                )
            )
        )
        return noisy_sampler, "sampler"

    elif backend_type == "Adonis":
        return IQMFakeAdonis(), "device"

    elif backend_type == "Aphrodite":
        return IQMFakeAphrodite(), "device"

    elif backend_type == "Garnet":
        return IQMFakeGarnet(), "device"

    elif backend_type == "Odra5":
        load_dotenv("ODRA_TOKEN.env")
        os.environ["IQM_TOKEN"] = os.getenv("IQM_TOKEN")
        server_url = os.getenv("SERVER")
        provider = IQMProvider(server_url)
        return provider.get_backend(), "device"

    elif backend_type == "Garnet_Hardware":
        load_dotenv("IQM_TOKEN.env")
        os.environ["IQM_TOKEN"] = os.getenv("IQM_TOKEN")
        provider = IQMProvider(
            "https://resonance.iqm.tech/", 
            quantum_computer="garnet"
        )
        return provider.get_backend(), "device"

    else:
        raise ValueError(f"Nieznany backend_type: {backend_type}")

def draw_backend(backend, out_dir = "out"):
    try:
        if type(backend) == SamplerV2:
            backend = backend._backend
        fig = mpl_draw(backend.coupling_map.graph, arrows=True, with_labels=True, node_color='#32a8a4', pos=spring_layout(backend.coupling_map.graph, num_iter=500))
        fig.savefig(f'{out_dir}/backend_coupling_map.png')
    except Exception as e:
        print(f"Error drawing backend: {e}")