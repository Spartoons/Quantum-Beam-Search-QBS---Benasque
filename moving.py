
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np

def choose_path(psi: np.ndarray,rep, statistics= False):
    n_qubits = int(np.ceil(np.log2(len(psi))))
    target_length = 2 ** n_qubits
    if len(psi) < target_length:
        psi = np.pad(psi, (0, target_length - len(psi)), 'constant', constant_values=0)
    
    psi = psi / np.linalg.norm(psi)
    psi_inv = psi[::-1]
    H = np.outer(psi,psi_inv) + np.outer(psi_inv, psi)
    op = Operator(H)
    H_op = SparsePauliOp.from_operator(op)
    U_op = PauliEvolutionGate(H_op, time=0.5*rep)
    qb = QuantumRegister(n_qubits)
    cb = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qb)
    qc.initialize(psi, qb)
    qc.append(U_op,qb)
    qc.measure_all()
    backend = AerSimulator(method='statevector', shots = 1)
    qc_trans = transpile(qc, backend)
    job = backend.run(qc_trans)
    result = job.result()
    counts = result.get_counts()
    bitstring = list(counts.keys())[0]
    return int(bitstring, 2)
    