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

class IgnoreMessagesFilter(logging.Filter):
    def filter(self, record):
        # Customize the condition to exclude specific messages
        return "Gate calibration for instruction" not in record.getMessage()

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    filename='output_day2.log',  # Specify your log file name here
    filemode='w',           # 'w' for overwriting, 'a' for appending
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()
logger.addFilter(IgnoreMessagesFilter())


global api_token, service, backends
api_token = 'be5e713d2a8fa9228e96fbb61a84af3337e4730a2d7a174eaeeb8c625f964b516a29f367490a0fa50ec5de4dff1be879e018c6f87e906a15b379519049c9868a'

service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
#backends = [service.backends()[1]]

def run_on_backend(qubit, backend):
    result = {}
    qpy_file = f'data/{qubit.name}_{backend.name}.qpy'
    
    try:
        logging.info(f'*Backend: {backend.name}')
        
        if os.path.exists(qpy_file):
            # Load the existing transpiled circuit from QPY file
            logging.info(f'*\tLoading existing QPY file: {qpy_file}')
            with open(qpy_file, 'rb') as qpy_f:
                qc = qpy.load(qpy_f)[0]  # Load and take the first circuit from the QPY file    
        else:
            # Transpile and save the new QPY file 
            logging.info(f'*\tTranspiling circuit for backend: {backend.name}')
            qc = transpile(qubit.circ, backend, optimization_level=0)  # Set optimization level to 0
            logging.info('\t\tTranspilation complete')

            with open(qpy_file, 'wb') as qpy_f:
                qpy.dump(qc, qpy_f)
            logging.info(f'*\t\tSaved transpiled circuit to {qpy_file}')

        # Prepare backend mapping for three different layouts
        backend_mapping = {}
        for mapping_label in ["A", "B", "C"]:     
            layout_file = f'data/{qubit.name}_{backend.name}_{mapping_label}_layout.json' 
            
            # Load existing layout if it exists, else create a new random layout
            if os.path.exists(layout_file):
                with open(layout_file, 'r') as layout_f:
                    backend_mapping = json.load(layout_f)
                    logging.info(f'*\tLoaded existing backend layouts from {layout_file}')
            else:
                # Generate a random layout for the new transpilation
                random_layout = np.random.permutation(backend.num_qubits)[:qc.num_qubits]
                backend_mapping[backend.name] = random_layout.tolist()
                logging.info(f'*\tGenerated random layout for {backend.name}: {random_layout}')

                # Save backend mappings (layouts) to the JSON file
                with open(layout_file, 'w') as layout_f:
                    json.dump(backend_mapping, layout_f, indent=4)
                logging.info(f'*\tSaved backend layouts to {layout_file}')

            # Use the layout (existing or newly generated) for transpilation
            qc_mapped = transpile(qc, backend, initial_layout=backend_mapping[backend.name], optimization_level=0)
            logging.info(f'*\tRunning job with mapping {mapping_label}...')

            # Run the job on the backend
            job = backend.run(qc_mapped, memory=True, shots=5000, job_tags=[f"{qubit.name}_{backend.name}_{mapping_label}"])
            result_counts = job.result().get_counts()
            logging.info(f'*\t\tGot results for mapping {mapping_label}!')

            # Store result counts in the dictionary
            result[f"{backend.name}_{mapping_label}"] = result_counts
            logging.info(f'*\t\tStored result for {backend.name} with mapping {mapping_label}')

    except Exception as e:
        logging.error(f'Error while running on backend {backend.name}: {str(e)}')

    return result

def run_circ(qubit_name, backends):
    # Load the qubit object from the pickle file
    pkl_file = f'data/{qubit_name}.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            qubit = pickle.load(f)
        logging.info(f'*Loaded qubit: {qubit.name} from {pkl_file}')
    except Exception as e:
        logging.error(f"Failed to load qubit from {pkl_file}: {e}")
        return

    results = {}
    # Open a backup file to write results incrementally
    with open(f'{qubit.name}_backup_day2.txt', 'a') as backup_file:

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
                        logging.info(f'*Backup stored for {backend.name}')
                except Exception as e:
                    logging.error(f'Error while processing {backend.name}: {e}')

    # After processing all backends, write the results to the JSON file
    with open(f'{qubit.name}_results_day2.json', 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f'*Stored results in: {qubit.name}_results_day2.json')
    
def main(qubit_name, backends):
    # Ensure logging is set up
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  
    # Run the circuit with the given qubit and backends
    run_circ(qubit_name, backends)

if __name__ == "__main__":
    # Set up argument parsing for the qubit name and backends
    parser = argparse.ArgumentParser(description="Run quantum circuit on multiple backends with specified mappings.")
    parser.add_argument("qubit_name", type=str, help="Name of the qubit (used to load/save files).")
    parser.add_argument("--backends", nargs="*", type=str, help="Space-separated list of backends to use (optional).")

    args = parser.parse_args()
    
    backends =  service.backends()[:-1]
    # Filter available backends based on user input, if provided
    if args.backends:
        backends = [backend for backend in backends if backend.name in args.backends]
    logging.info(backends)

    main(args.qubit_name, backends)