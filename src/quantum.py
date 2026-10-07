from qiskit import transpile

def quantum_multi_job(
    param_sets, circ, backend, mode="sampler", shots=1024
):
    params = list(circ.parameters)

    if mode == "sampler":
        pubs = [(circ, full_params) for full_params in param_sets]
        job = backend.run(pubs, shots=shots)
        result = job.result()

        counts_list = []
        for pub_res in result:
            data_bin = pub_res.data
            reg_name = list(data_bin.keys())[0]
            counts_list.append(getattr(data_bin, reg_name).get_counts())

        return counts_list

    elif mode == "device":
        transpiled_circs = []
        for full_params in param_sets:
            assignments = dict(zip(params, full_params))
            assigned_circ = circ.assign_parameters(assignments)
            trans = transpile(assigned_circ, backend=backend)
            transpiled_circs.append(trans)

        job = backend.run(transpiled_circs, shots=shots)
        result = job.result()

        counts = result.get_counts()
        return counts if isinstance(counts, list) else [counts]

    else:
        raise ValueError(f"Nieznany tryb backendu: {mode}")