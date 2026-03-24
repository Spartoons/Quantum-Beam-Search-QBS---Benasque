# 🏔️ Quantum Beam Search (QBS) - Benasque

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6133BD)](https://qiskit.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Event](https://img.shields.io/badge/Event-NTQC%202026%20Spring%20School-blue)](https://www.icfo.eu/ntqc/)

> **Quantum-Assisted Hiking Route Optimization in the Pyrenees**

A hybrid quantum-classical algorithm that generates optimized hiking routes through Benasque Valley by combining classical graph traversal with quantum amplitude encoding. Built during the NTQC 2026 Hackathon at the Spring School on Quantum Computing.

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/Spartoons/Quantum-Beam-Search-QBS---Benasque.git
cd Quantum-Beam-Search-QBS---Benasque

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

Then open `http://localhost:5000` in your browser.

---

## 🎯 The Problem

The **Orienteering Problem** in mountain hiking: Given 25 locations in Benasque Valley (peaks, towns, lakes, refugios), find the optimal route that:

- ✅ Maximizes scenic value (landmarks, peaks, views)
- ✅ Respects time budget constraints
- ✅ Considers elevation gain/loss
- ✅ Adapts to seasonal conditions (winter/summer)
- ✅ Accounts for terrain difficulty

Classical greedy algorithms often get trapped in local optima, missing better routes that require temporarily accepting suboptimal paths.

---

## ⚛️ Our Solution: Quantum-Assisted Lookahead Search

Instead of forcing the entire 25-node graph onto limited quantum hardware, we designed a scalable hybrid architecture:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTUM-CLASSICAL HYBRID                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Classical  │───▶│   Classical  │───▶│   Quantum    │      │
│  │   Graph      │    │   Heuristic  │    │   Sampling   │      │
│  │   2-Step     │    │   Scoring    │    │   (Qiskit)   │      │
│  │   Lookahead  │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Path Selection Decision                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Classical 2-Step Lookahead**: Use NetworkX to explore all valid paths 2 steps ahead from the current position
2. **Heuristic Scoring**: Assign scores based on distance, elevation gain, and landmark value
3. **Quantum Amplitude Encoding**: Map normalized scores to probability amplitudes of a 3-4 qubit circuit
4. **Quantum Sampling**: Execute on Qiskit Aer simulator
5. **Measurement Collapse**: Quantum measurement determines the next move

**Key Insight**: By mapping heuristics to quantum amplitudes, we bias toward optimal paths while maintaining quantum probability for exploration—balancing exploitation and exploration naturally.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Quantum Computing** | Qiskit, Qiskit-Aer |
| **Classical Routing** | NetworkX, Pandas, NumPy |
| **Web Framework** | Flask, Gunicorn |
| **Frontend** | Leaflet.js, Font Awesome, Turf.js |

---

## 📁 Repository Structure

```
Quantum-Beam-Search-QBS---Benasque/
├── data/                    # Dataset (CSV files)
│   └── raw/                 # Original data
│       ├── coordinates.csv  # GPS coordinates
│       ├── distances.csv    # Travel times matrix
│       ├── elevations.csv   # Elevation data
│       ├── node_types.csv   # Location classifications
│       ├── places.csv       # Location names
│       └── terrain.csv      # Seasonal terrain difficulty
│
├── docs/                    # Documentation
│   ├── hackathon/           # Hackathon materials
│   └── images/              # Screenshots and diagrams
│
├── notebooks/               # Jupyter notebooks
│   ├── exploration.ipynb    # Main analysis
│   └── exploration_annika.ipynb
│
├── src/                     # Source code
│   ├── app.py               # Flask application
│   ├── quantum/             # Quantum modules
│   │   └── path_selector.py # Quantum path selection
│   ├── classical/           # Classical modules
│   └── templates/
│       └── index.html       # Web interface
│
├── tests/                   # Unit tests
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── run.py                   # Entry point
```

---

## 🚀 Usage

### Web Interface

The easiest way to interact with the algorithm is through the web interface:

```bash
python run.py
```

Then navigate to `http://localhost:5000` and:

1. Select difficulty level (Easy/Medium/Hard)
2. Toggle season (Winter/Summer)
3. Choose whether to allow snow routes
4. Click "Run Algorithm" to see the optimized path

### API Endpoints

#### GET `/api/data`
Returns all nodes and edges for map visualization.

**Response:**
```json
{
  "nodes": [...],
  "edges": [...]
}
```

#### POST `/api/calculate_path`
Calculates an optimized route based on user preferences.

**Request:**
```json
{
  "difficulty": "medium",
  "season": "winter",
  "allow_snow": true
}
```

**Response:**
```json
{
  "path": [[lat, lon], ...],
  "details": [...],
  "total_time": 5.5
}
```

### Programmatic Usage

```python
from src.quantum.path_selector import choose_path
import numpy as np

# Define path scores (lower is better)
scores = np.array([0.8, 0.3, 0.5, 0.9])

# Get quantum-selected path index
selected_index = choose_path(scores, rep=1)
print(f"Selected path: {selected_index}")
```

---

## 📊 Dataset

The project uses real data from 25 locations in Benasque Valley, Spain:

| Data File | Description | Records |
|-----------|-------------|---------|
| `places.csv` | Location names (Pico Aneto, Cerler, etc.) | 25 |
| `coordinates.csv` | GPS coordinates (lat, lon, alt) | 25 |
| `distances.csv` | Travel time matrix (hours:minutes) | 25×25 |
| `elevations.csv` | Elevation in meters | 25 |
| `node_types.csv` | Categories (Peak, Town, Lake, etc.) | 25 |
| `terrain.csv` | Winter/summer difficulty ratings | 25 |

---

## 👥 Team

This project was developed during the **NTQC 2026 Spring School Hackathon** in Benasque, Spain by:

- Aran Oliveras (@Spartoons)
- Marcos Arroyo (@MArroyoSanchez)
- Anna Ekstrøm (@AnnaEkstroem)
- Annika Weisberg (@Annika-ee)

---

## 🙏 Acknowledgments

Thanks to the organizers of the NTQC 2026 Spring School and the Benasque Center for Science for hosting the event.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- 📊 **Hackathon Info**: [NTQC 2026](https://www.icfo.eu/ntqc/)
- ⚛️ **Qiskit**: [https://qiskit.org](https://qiskit.org)
- 🗺️ **Benasque Valley**: [Wikipedia](https://en.wikipedia.org/wiki/Benasque)

---

<p align="center">
  <sub>Built with ❤️ and ⚛️ in Benasque, Spain</sub>
</p>
