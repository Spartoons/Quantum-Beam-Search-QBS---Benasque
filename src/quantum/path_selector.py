"""
Quantum Path Selector Module

This module implements quantum amplitude encoding for path selection
using Qiskit. It maps classical path scores to quantum probability
amplitudes and uses quantum measurement to select paths.

The algorithm creates a quantum circuit that encodes path desirability
scores into probability amplitudes, then samples from this distribution
to make path selection decisions.
"""

from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np


def choose_path(psi: np.ndarray, rep: int = 1, statistics: bool = False) -> int:
    """
    Select a path using quantum amplitude encoding.
    
    This function encodes classical path scores into a quantum state,
    applies a time evolution operator, and measures the result to
    probabilistically select a path based on the encoded scores.
    
    Args:
        psi: Array of path scores (lower is better). Will be normalized
             and encoded as quantum probability amplitudes.
        rep: Number of repetitions for the time evolution (default: 1).
             Higher values can amplify differences between paths.
        statistics: If True, return full measurement statistics instead
                   of just the selected index (default: False).
    
    Returns:
        int: Index of the selected path, or measurement counts if statistics=True.
    
    Example:
        >>> scores = np.array([0.8, 0.3, 0.5, 0.9])  # Path scores
        >>> selected = choose_path(scores, rep=1)
        >>> print(f"Selected path index: {selected}")
    
    Note:
        The function automatically pads the input to the next power of 2
        to fit the quantum register size. It also handles normalization
        and creates a Hermitian operator for time evolution.
    """
    # Calculate required number of qubits
    n_qubits = int(np.ceil(np.log2(len(psi))))
    target_length = 2 ** n_qubits
    
    # Pad psi to fit quantum register size (power of 2)
    if len(psi) < target_length:
        psi = np.pad(psi, (0, target_length - len(psi)), 'constant', constant_values=0)
    
    # Normalize the state vector
    psi = psi / np.linalg.norm(psi)
    
    # Create inverted copy for Hamiltonian construction
    psi_inv = psi[::-1]
    
    # Construct Hermitian operator: H = |psi⟩⟨psi_inv| + |psi_inv⟩⟨psi|
    H = np.outer(psi, psi_inv) + np.outer(psi_inv, psi)
    
    # Convert to Qiskit Operator and then to Pauli representation
    op = Operator(H)
    H_op = SparsePauliOp.from_operator(op)
    
    # Create time evolution gate
    U_op = PauliEvolutionGate(H_op, time=0.5 * rep)
    
    # Build quantum circuit
    qb = QuantumRegister(n_qubits)
    cb = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qb)
    
    # Initialize quantum state with classical scores
    qc.initialize(psi, qb)
    
    # Apply time evolution
    qc.append(U_op, qb)
    
    # Measure all qubits
    qc.measure_all()
    
    # Run on simulator
    backend = AerSimulator(method='statevector', shots=1)
    qc_trans = transpile(qc, backend)
    job = backend.run(qc_trans)
    result = job.result()
    counts = result.get_counts()
    
    # Extract measurement result
    bitstring = list(counts.keys())[0]
    
    if statistics:
        return counts
    
    return int(bitstring, 2)
