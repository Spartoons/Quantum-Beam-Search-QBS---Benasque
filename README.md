# 🏔️ Quantum-Beam-Search: Assisted Alpine Routing

![Status](https://img.shields.io/badge/Status-Work%20in%20Progress%20%E2%9A%92%EF%B8%8F-orange)
![Event](https://img.shields.io/badge/Event-Spring%20School%20NTQC%202026-blue)
![Hardware](https://img.shields.io/badge/Hardware-Mare%20Nostrum5%20Ona%20(4q)-purple)

**Q-Route** is a B2B SaaS prototype designed for tourism operators in the Benasque Valley. It generates highly optimized, personalized one-day hiking routes by combining classical graph heuristics with **Quantum Amplitude Encoding** to escape local minima in complex orienteering problems.

## ⚠️ Hackathon Notice
> **We are currently in the coding phase of the NTQC 2026 Hackathon!** > *The code in this repository is actively being pushed. Final results, quantum circuit diagrams, and execution logs from the Mare Nostrum5 / Qmio hardware will be uploaded before the Friday 9:30 AM deadline.*

## 🗺️ The Challenge
Designing a mountain route is a constrained variant of the **Orienteering Problem**. We must maximize the "scenic value" (landmarks, peaks) while strictly adhering to user constraints (time, elevation budget, winter/summer gear) across 25 nodes in the Pyrenees. Classical greedy algorithms often fail to find the global optimum, getting trapped in local sub-optimal loops.

## ⚛️ Our Hybrid Architecture
Instead of forcing a massive 25-node graph onto a near-term QPU, we designed a highly scalable **Quantum-Assisted Lookahead Search**:

1. **Classical 2-Step Lookahead:** We use `NetworkX` and `Pandas` to explore all valid paths 2 steps ahead of the hiker's current node, dynamically filtering by the user's gear and time constraints.
2. **Heuristic Scoring:** Each path is assigned a classical desirability score based on distance, elevation gain, and landmark value.
3. **Quantum Amplitude Encoding:** We normalize these scores and map them to the probability amplitudes of a 3-qubit or 4-qubit parameterized circuit.
4. **Quantum Sampling:** We execute the circuit natively on the **Mare Nostrum5 Ona (4-qubit)** architecture. The quantum measurement collapses into our next move. 



*Why this works:* By mapping heuristics to quantum amplitudes, we heavily bias the algorithm toward the best paths while maintaining a quantum-mechanical probability of exploring sub-optimal branches, perfectly balancing exploitation and exploration!

## 🛠️ Tech Stack
* **Quantum:** `Qiskit`, `Qiskit-Aer`
* **Classical Routing:** `NetworkX`
* **Data Processing:** `Pandas`, `NumPy`
* **Target Hardware:** Mare Nostrum5 Ona (4 qubits), Qmio (32 qubits)

## 🚀 How to Run (Coming Soon)
```bash
# Clone the repo
git clone [https://github.com/yourusername/Q-Route-Benasque.git](https://github.com/yourusername/Q-Route-Benasque.git)
cd Q-Route-Benasque

# Install requirements
pip install -r requirements.txt

# Run the hybrid solver
python main.py --user_profile beginner --max_hours 6
