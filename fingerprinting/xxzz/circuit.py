import json
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister, assemble
from qiskit_aer import Aer, AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session, SamplerOptions
from qtcodes import XXZZQubit
import time
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import mode
import logging
import asyncio
import pickle
import qiskit.qpy as qpy
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

global api_token, service, backends
api_token = 'be5e713d2a8fa9228e96fbb61a84af3337e4730a2d7a174eaeeb8c625f964b516a29f367490a0fa50ec5de4dff1be879e018c6f87e906a15b379519049c9868a'

service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
backends =  service.backends()[:-1]

def run_on_backend(qubit, backend):
    result = {}
    qpy_file = f'data/{qubit.name}_{backend.name}.qpy'
    try:
        logging.info(f'Backend: {backend.name}')
    
        # Transpile and save the new QPY file 
        logging.info(f'\tTranspiling circuit for backend: {backend.name}')
        qc = transpile(qubit.circ, backend, optimization_level=0)  # Set optimization level to 0
        logging.info('\t\tTranspilation complete')

        with open(qpy_file, 'wb') as qpy_f:
            qpy.dump(qc, qpy_f)
        logging.info(f'\t\tSaved transpiled circuit to {qpy_file}')

        # Run the job with three different mappings
        for mapping_label in ["A", "B", "C"]:
            # Generate a permutation of initial layout for Mappings B and C
            random_layout = np.random.permutation(backend.num_qubits)[:qc.num_qubits]
            
            # Transpile the circuit with the specified initial layout
            qc_mapped = transpile(qc, backend, initial_layout=random_layout, optimization_level=0)
            logging.info(f'\tRunning job with mapping {mapping_label}...')

            # Run the job on the backend
            job = backend.run(qc_mapped, memory=True, shots=5000, job_tags=[f"{qubit.name}_{backend.name}_{mapping_label}"])
            result_counts = job.result().get_counts()
            logging.info(f'\t\tGot results for mapping {mapping_label}!')

            # Store result counts in the dictionary
            result[f"{backend.name}_{mapping_label}"] = result_counts
            logging.info(f'\t\tStored result for {backend.name} with mapping {mapping_label}')

    except Exception as e:
        logging.error(f'{backend.name}: {str(e)}')

    return result

def run_circ(qubit_name, backends):
    # Load the qubit object from the pickle file
    pkl_file = f'data/{qubit_name}.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            qubit = pickle.load(f)
        logging.info(f"Loaded qubit: {qubit.name} from {pkl_file}")
    except Exception as e:
        logging.error(f"Failed to load qubit from {pkl_file}: {e}")
        return

    results = {}
    # Open a backup file to write results incrementally
    with open(f'{qubit.name}_backup.txt', 'a') as backup_file:

        # Use ThreadPoolExecutor to run the function on each backend concurrently
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_on_backend, qubit, backend): backend for backend in backends}
            for future in as_completed(futures):
                backend = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.update(result)
                        # Write the result to the backup file immediately
                        for mapping_label in ["A", "B", "C"]:
                            if f"{backend.name}_{mapping_label}" in result:
                                backup_file.write(f'{backend.name}_{mapping_label}: {result[f"{backend.name}_{mapping_label}"]}\n')
                        backup_file.flush()
                        logging.info(f'Backup stored for {backend.name}')
                except Exception as e:
                    logging.error(f'Error while processing {backend.name}: {e}')

    # After processing all backends, write the results to the JSON file
    with open(f'{qubit.name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f'Stored results in: {qubit.name}_results.json')
    
def main(qubit_name, backends):
    # Ensure logging is set up
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  
    # Run the circuit with the given qubit and backends
    run_circ(qubit_name, backends)

if __name__ == "__main__":
    # Set up argument parsing for the qubit name
    parser = argparse.ArgumentParser(description="Run quantum circuit on multiple backends with specified mappings.")
    parser.add_argument("qubit_name", type=str, help="Name of the qubit (used to load/save files).")
    
    args = parser.parse_args()
    main(args.qubit_name, backends)