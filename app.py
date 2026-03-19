from flask import Flask, render_template, jsonify, request
import pandas as pd
import networkx as nx
import math
import random
import os
import numpy as np
from moving import choose_path

app = Flask(__name__)

# Global variables to store our graph and frontend data in memory
G = nx.Graph()
FRONTEND_NODES = []
FRONTEND_EDGES = []

# ==========================================
# 1. GRAPH INITIALIZATION
# ==========================================
def parse_time(time_str):
    if pd.isna(time_str) or str(time_str).strip() in ['X', '']: return None
    time_str = str(time_str).strip()
    if "'" in time_str:
        h, m = time_str.split("'")
        return int(h) + int(m) / 60.0
    try: return float(time_str)
    except ValueError: return None

def initialize_graph():
    global G, FRONTEND_NODES, FRONTEND_EDGES # Added FRONTEND_EDGES
    G.clear()
    FRONTEND_NODES.clear()
    FRONTEND_EDGES.clear() # Added this to prevent infinite drawing on reload
    
    try:
        dist_path = 'distances.csv' if os.path.exists('distances.csv') else '194_hackaton_info/distances.csv'
        
        # We add .fillna('') to completely prevent the JSON NaN crash!
        df_distances = pd.read_csv(dist_path, skiprows=2, index_col=0).fillna('')
        coords = pd.read_csv('Untitled map.csv').fillna('')
        types = pd.read_csv('node_types.csv', header=None).fillna('').iloc[0].tolist()
        
        # Load BOTH seasons from terrain.csv
        terrains = pd.read_csv('terrain.csv', header=None).fillna('')
        winter_terrains = terrains.iloc[0].tolist() if len(terrains) > 0 else []
        summer_terrains = terrains.iloc[1].tolist() if len(terrains) > 1 else winter_terrains
        
        type_weights = {'Urban': 0.2, 'Trail': 0.4, 'Mountain': 0.6, 'Snow': 0.8}
        
        # 1. Build Nodes and frontend data
        for i, row in coords.iterrows():
            if str(row['X']) == '' or str(row['Y']) == '':
                continue
                
            node_id = str(i + 1)
            
            n_type = str(types[i]).strip() if i < len(types) and str(types[i]) != '' else 'Landmark'

            # Extract both terrains
            terrain_w = str(winter_terrains[i]).strip() if i < len(winter_terrains) and str(winter_terrains[i]) != '' else 'Mountain'
            terrain_s = str(summer_terrains[i]).strip() if i < len(summer_terrains) and str(summer_terrains[i]) != '' else 'Mountain'

            G.add_node(
                node_id,
                name=f"Node {node_id}",
                type=n_type,
                elevation=0, 
                terrain_winter=terrain_w,
                terrain_summer=terrain_s,
                type_val_winter=type_weights.get(terrain_w, 0.6),
                type_val_summer=type_weights.get(terrain_s, 0.6)
            )
            
            FRONTEND_NODES.append({
                'id': i, 'graph_id': node_id, 'name': f"Node {node_id}",
                'lat': row['Y'], 'lon': row['X'], 'type': n_type,
                'terrain_winter': terrain_w, 'terrain_summer': terrain_s
            })

        # 2. Build Edges
        for i in df_distances.index:
            for j in df_distances.columns:
                val = df_distances.loc[i, j]
                time_hours = parse_time(val)
                if time_hours is not None:
                    G.add_edge(str(i), str(j), weight=time_hours)
                    
        # 3. Build Frontend Edges
        for u, v in G.edges():
            node_u = next((n for n in FRONTEND_NODES if n['graph_id'] == u), None)
            node_v = next((n for n in FRONTEND_NODES if n['graph_id'] == v), None)
            if node_u and node_v:
                lat_u = float(str(node_u['lat']).replace(',', '.'))
                lon_u = float(str(node_u['lon']).replace(',', '.'))
                lat_v = float(str(node_v['lat']).replace(',', '.'))
                lon_v = float(str(node_v['lon']).replace(',', '.'))
                FRONTEND_EDGES.append([[lat_u, lon_u], [lat_v, lon_v]])
                    
        print(f"[*] Graph initialized SUCCESS: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    except Exception as e:
        print(f"[!] CRITICAL Error loading data: {e}")

# ==========================================
# 2. EPSILON HEURISTIC & PATHFINDING
# ==========================================
def calculate_epsilon(path, user_profile):
    cost = 0.0
    season = user_profile.get('season', 'winter') # Get the current season
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        # Select the correct mathematical weight based on the season!
        val_key = 'type_val_winter' if season == 'winter' else 'type_val_summer'
        val_u = float(G.nodes[u].get(val_key, 0.6))
        val_v = float(G.nodes[v].get(val_key, 0.6))
        mean_type = np.mean([val_u, val_v]) 
        
        time_val = float(G[u][v].get("weight", 0.1))
        elev_u = float(G.nodes[u].get("elevation", 0.0))
        elev_v = float(G.nodes[v].get("elevation", 0.0))
        dif_height = elev_v - elev_u  
        
        cost += (mean_type * time_val * dif_height)
        
    return cost

def tensor_network_heuristic(epsilon_scores, difficulty):
    """
    Acts as a Tensor Network layer. 
    It takes the raw Epsilon effort and transforms it into Quantum Amplitudes (psi) 
    based on how hard the user wants the route to be.
    """
    # Convert to numpy tensor
    scores = np.array(epsilon_scores, dtype=float)
    
    # If paths are identical in effort, avoid division by zero
    if np.all(scores == scores[0]):
        return np.ones_like(scores)

    # Tensor Transformation based on Difficulty
    if difficulty == 'easy':
        # Easy: We want lowest effort to have the HIGHEST amplitude
        # Shift scores to be positive, then invert
        scores = scores - np.min(scores) + 1e-5
        psi = 1.0 / scores
    elif difficulty == 'hard':
        # Hard: We want the highest effort to have the HIGHEST amplitude
        psi = scores - np.min(scores) + 1e-5
    else: # medium
        # Medium: We want efforts closest to the average to have the HIGHEST amplitude
        target = np.mean(scores)
        distances_from_mean = np.abs(scores - target)
        psi = 1.0 / (distances_from_mean + 1e-5)
        
    # Return the normalized amplitudes for the Quantum Walker
    return psi / np.linalg.norm(psi)

def quantum_selector(possible_paths, user_profile):
    print(f"  -> Evaluating paths for {user_profile.get('season')}...")
    epsilon_scores = []
    valid_paths = []
    
    terrain_key = 'terrain_winter' if user_profile.get('season', 'winter') == 'winter' else 'terrain_summer'
    
    for path in possible_paths:
        # Filter snow based on the active season's terrain
        has_snow = any(G.nodes[n].get(terrain_key) == 'Snow' for n in path[1:])
        if not user_profile.get('allow_snow', True) and has_snow:
            continue 
            
        valid_paths.append(path)
        # Pass user_profile into epsilon now!
        epsilon_scores.append(calculate_epsilon(path, user_profile))
        
    if not valid_paths:
        return possible_paths[0] # Fallback if constraints block everything

    # 2. Tensor Network: Convert Epsilon to Quantum Amplitudes based on difficulty
    psi = tensor_network_heuristic(epsilon_scores, user_profile.get('difficulty', 'medium'))
    
    # 3. Quantum Walker: Execute your Qiskit circuit in moving.py
    # `choose_path` runs Time Evolution and returns something like {'010': 1}
    counts = choose_path(psi, rep=1)
    
    # Extract the binary string measured by the quantum computer
    measured_binary = list(counts.keys())[0]
    
    # Convert binary back to a standard integer index
    measured_idx = int(measured_binary, 2)
    
    # Handle Qubit padding (if 5 paths exist, Qiskit pads to 8. If 7 is measured, modulo it)
    if measured_idx >= len(valid_paths):
        measured_idx = measured_idx % len(valid_paths)
        
    selected_path = valid_paths[measured_idx]
    print(f"  -> Quantum Walker Collapsed on Index {measured_idx} (Epsilon Score: {epsilon_scores[measured_idx]:.2f})")
    
    return selected_path

def get_2_step_paths(current_node, visited_nodes):
    paths = []
    for n1 in G.neighbors(current_node):
        if n1 in visited_nodes: continue
        has_n2 = False
        for n2 in G.neighbors(n1):
            if n2 != current_node and n2 not in visited_nodes:
                paths.append([current_node, n1, n2])
                has_n2 = True
        if not has_n2:
            paths.append([current_node, n1])
    return paths

def placeholder_selector(possible_paths, user_profile):
    """
    Evaluates paths using Epsilon and returns the absolute best one.
    This is where `moving.choose_path` will be inserted later!
    """
    best_path = None
    best_score = -float('inf')
    
    for path in possible_paths:
        score = calculate_epsilon(path, user_profile)
        if score > best_score:
            best_score = score
            best_path = path
            
    # If all paths are invalid (e.g., snow blocked), just pick a random valid step backward
    if best_path is None:
        return random.choice(possible_paths) 
        
    return best_path

# ==========================================
# 3. FLASK ROUTES
# ==========================================
@app.before_request
def startup():
    if not FRONTEND_NODES:
        initialize_graph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"nodes": FRONTEND_NODES, "edges": FRONTEND_EDGES})

@app.route('/api/calculate_path', methods=['POST'])
def calculate_path():
    data = request.json
    
    # Extract user profile from frontend
    user_profile = {
        'allow_snow': data.get('allow_snow', True),
        'difficulty': data.get('difficulty', 'medium'),
        'season': data.get('season', 'winter'), # Added this line!
        'time_budget': 10 if data.get('difficulty') == 'hard' else 6
    }
    
    start_node = "3" # Benasque
    route = [start_node]
    current_node = start_node
    target_steps = 5 if user_profile['difficulty'] == 'hard' else 3
    
    print(f"\n--- Starting Backend Route Generation (Snow allowed: {user_profile['allow_snow']}) ---")
    
    # 2-Step Lookahead Loop
    for step in range(target_steps):
        possible_paths = get_2_step_paths(current_node, route)
        if not possible_paths:
            print("Dead end reached.")
            break
            
        selected_path = quantum_selector(possible_paths, user_profile)
        next_node = selected_path[1] # Take exactly one step forward
        
        route.append(next_node)
        current_node = next_node
        print(f"Step {step+1}: Moved to {G.nodes[next_node].get('name', next_node)}")
        
    # Convert Graph IDs ('1', '2') back to Frontend Array Indices (0, 1)
    # This ensures your frontend draws the polyline correctly!
    # Convert Graph IDs ('1', '2') into [lat, lon] coordinates for Leaflet to draw
    path_coords = []
    for r_id in route:
        for f_node in FRONTEND_NODES:
            if f_node['graph_id'] == r_id:
                # Ensure coordinates are strict floats (fixes comma vs dot decimal issues)
                lat = float(str(f_node['lat']).replace(',', '.'))
                lon = float(str(f_node['lon']).replace(',', '.'))
                path_coords.append([lat, lon])
                break

    print(f"Final Route Coords for Frontend: {path_coords}")
    return jsonify({"path": path_coords})

if __name__ == '__main__':
    initialize_graph()
    app.run(debug=True, port=5000)