"""
Flask Web Application for Quantum Benasque Routing

This module provides a web interface and API for the quantum-classical
hybrid hiking route optimizer. It handles graph initialization,
route calculation, and serves the frontend.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import networkx as nx
import math
import random
import os
import numpy as np

# Import quantum path selector
try:
    from quantum.path_selector import choose_path
except ImportError:
    # Fallback for direct execution
    from src.quantum.path_selector import choose_path

app = Flask(__name__)

# Global variables to store our graph and frontend data in memory
G = nx.Graph()
FRONTEND_NODES = []
FRONTEND_EDGES = []

# Data directory configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


def get_data_path(filename):
    """Get the full path to a data file."""
    return os.path.join(DATA_DIR, filename)


def parse_time(time_str):
    """
    Parse time string in format "H'MM" or float to hours.
    
    Args:
        time_str: Time string (e.g., "4'50", "1'25") or float
    
    Returns:
        float: Time in hours, or None if parsing fails
    """
    if pd.isna(time_str) or str(time_str).strip() in ['X', '']:
        return None
    time_str = str(time_str).strip()
    if "'" in time_str:
        h, m = time_str.split("'")
        return int(h) + int(m) / 60.0
    try:
        return float(time_str)
    except ValueError:
        return None


def initialize_graph():
    """
    Initialize the graph from CSV data files.
    
    Loads node data (coordinates, types, elevations, terrain) and edge data
    (distances/travel times) to build a NetworkX graph for route planning.
    Also prepares frontend-compatible node and edge data.
    """
    global G, FRONTEND_NODES, FRONTEND_EDGES
    G.clear()
    FRONTEND_NODES.clear()
    FRONTEND_EDGES.clear()
    
    try:
        # Load data files
        df_distances = pd.read_csv(
            get_data_path('distances.csv'), 
            skiprows=2, 
            index_col=0
        ).fillna('')
        
        coords = pd.read_csv(get_data_path('coordinates.csv')).fillna('')
        types = pd.read_csv(get_data_path('node_types.csv'), header=None).fillna('').iloc[0].tolist()
        
        # Load both seasons from terrain.csv
        terrains = pd.read_csv(get_data_path('terrain.csv'), header=None).fillna('')
        winter_terrains = terrains.iloc[0].tolist() if len(terrains) > 0 else []
        summer_terrains = terrains.iloc[1].tolist() if len(terrains) > 1 else winter_terrains

        # Load place names
        try:
            with open(get_data_path('places.csv'), 'r', encoding='utf-8') as f:
                places_names = [name.strip() for name in f.read().split(',') if name.strip()]
        except FileNotFoundError:
            places_names = []

        # Load elevations
        try:
            with open(get_data_path('elevations.csv'), 'r', encoding='utf-8') as f:
                elevations_data = [float(e.strip()) for e in f.read().split(',') if e.strip()]
        except FileNotFoundError:
            elevations_data = []
        
        # Terrain type weights for scoring
        type_weights = {
            'Urban': 0.2, 
            'Trail': 0.4, 
            'Mountain': 0.6, 
            'Snow': 0.8
        }
        
        # Build nodes and frontend data
        for i, row in coords.iterrows():
            if str(row['X']) == '' or str(row['Y']) == '':
                continue
                
            node_id = str(i + 1)
            
            n_type = str(types[i]).strip() if i < len(types) and str(types[i]) != '' else 'Landmark'
            n_name = places_names[i] if i < len(places_names) else f"Node {node_id}"
            n_elev = elevations_data[i] if i < len(elevations_data) else 0

            # Extract both terrains
            terrain_w = str(winter_terrains[i]).strip() if i < len(winter_terrains) and str(winter_terrains[i]) != '' else 'Mountain'
            terrain_s = str(summer_terrains[i]).strip() if i < len(summer_terrains) and str(summer_terrains[i]) != '' else 'Mountain'

            G.add_node(
                node_id,
                name=n_name,
                type=n_type,
                elevation=n_elev, 
                terrain_winter=terrain_w,
                terrain_summer=terrain_s,
                type_val_winter=type_weights.get(terrain_w, 0.6),
                type_val_summer=type_weights.get(terrain_s, 0.6)
            )
            
            FRONTEND_NODES.append({
                'id': i, 
                'graph_id': node_id, 
                'name': n_name,
                'lat': row['Y'], 
                'lon': row['X'], 
                'type': n_type,
                'terrain_winter': terrain_w, 
                'terrain_summer': terrain_s
            })

        # Build edges from distance matrix
        for i in df_distances.index:
            for j in df_distances.columns:
                val = df_distances.loc[i, j]
                time_hours = parse_time(val)
                if time_hours is not None:
                    G.add_edge(str(i), str(j), weight=time_hours)
                
        # Build frontend edges
        for u, v in G.edges():
            node_u = next((n for n in FRONTEND_NODES if n['graph_id'] == u), None)
            node_v = next((n for n in FRONTEND_NODES if n['graph_id'] == v), None)
            if node_u and node_v:
                lat_u = float(str(node_u['lat']).replace(',', '.'))
                lon_u = float(str(node_u['lon']).replace(',', '.'))
                lat_v = float(str(node_v['lat']).replace(',', '.'))
                lon_v = float(str(node_v['lon']).replace(',', '.'))
                FRONTEND_EDGES.append([[lat_u, lon_u], [lat_v, lon_v]])
                    
        print(f"[*] Graph initialized: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        raise


def tensor_network_heuristic(epsilon_scores, difficulty):
    """
    Convert epsilon scores to probability distribution based on difficulty.
    
    Args:
        epsilon_scores: Array of path cost scores
        difficulty: 'easy', 'medium', or 'hard'
    
    Returns:
        Normalized probability distribution
    """
    scores = np.array(epsilon_scores, dtype=float)
    
    # Handle identical scores
    if np.all(scores == scores[0]):
        psi = np.ones_like(scores)
        return psi / np.linalg.norm(psi)

    if difficulty == 'easy':
        # Prefer shorter/easier paths
        scores = scores - np.min(scores) + 1e-5
        psi = 1.0 / scores
    elif difficulty == 'hard':
        # Prefer more challenging paths
        psi = scores - np.min(scores) + 1e-5
    else:  # medium
        # Prefer paths close to average
        target = np.mean(scores)
        distances_from_mean = np.abs(scores - target)
        psi = 1.0 / (distances_from_mean + 1e-5)
        
    return psi / np.linalg.norm(psi)


def calculate_epsilon(G_req, path, user_profile, current_route):
    """
    Calculate the cost (epsilon) of a path.
    
    Lower scores indicate better paths. Considers terrain difficulty,
    travel time, elevation change, and penalizes backtracking.
    
    Args:
        G_req: The graph
        path: List of node IDs
        user_profile: User preferences dict
        current_route: Current route to check for backtracking
    
    Returns:
        float: Path cost score
    """
    cost = 0.0
    season = user_profile.get('season', 'winter')
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        
        val_key = 'type_val_winter' if season == 'winter' else 'type_val_summer'
        val_u = float(G_req.nodes[u].get(val_key, 0.6))
        val_v = float(G_req.nodes[v].get(val_key, 0.6))
        mean_type = np.mean([val_u, val_v])
        
        edge_data = G_req[u][v]
        time_val = float(edge_data.get("weight", 0.1))
        
        elev_u = float(G_req.nodes[u].get("elevation", 0.0))
        elev_v = float(G_req.nodes[v].get("elevation", 0.0))
        
        # Use absolute elevation difference
        dif_height = abs(elev_v - elev_u)
        if dif_height < 1:
            dif_height = 1.0
        
        step_cost = (mean_type * time_val * dif_height)
        
        # Penalize backtracking
        if v in current_route:
            step_cost *= 10.0
            
        cost += step_cost
        
    return cost


def quantum_selector(G_req, possible_paths, user_profile, current_route):
    """
    Select a path using quantum-assisted decision making.
    
    Args:
        G_req: The graph
        possible_paths: List of candidate paths
        user_profile: User preferences
        current_route: Current route for backtracking check
    
    Returns:
        Selected path (list of node IDs)
    """
    print(f"  -> Evaluating {len(possible_paths)} paths with Quantum Pipeline...")
    
    if not possible_paths:
        return []
        
    if len(possible_paths) == 1:
        print("  -> Only 1 path available. Deterministic bypass.")
        return possible_paths[0]

    # Calculate scores for all paths
    epsilon_scores = []
    for path in possible_paths:
        epsilon_scores.append(calculate_epsilon(G_req, path, user_profile, current_route))
        
    # Convert to quantum probability distribution
    psi = tensor_network_heuristic(epsilon_scores, user_profile.get('difficulty', 'medium'))
    
    try:
        # Quantum selection
        counts = choose_path(psi, rep=1)
        measured_binary = list(counts.keys())[0]
        measured_idx = int(measured_binary, 2)
    except Exception as e:
        print(f"[!] Quantum Simulator Exception: {e}")
        print("[!] Failing gracefully to Classical Random Selector...")
        measured_idx = random.randint(0, len(possible_paths) - 1)
        
    # Bounds check
    if measured_idx >= len(possible_paths):
        measured_idx = measured_idx % len(possible_paths)
        
    selected_path = possible_paths[measured_idx]
    print(f"  -> Collapsed on Index {measured_idx} (Epsilon Score: {epsilon_scores[measured_idx]:.2f})")
    
    return selected_path


def get_2_step_paths(G_req, current_node):
    """
    Get all valid 2-step paths from the current node.
    
    Args:
        G_req: The graph
        current_node: Starting node ID
    
    Returns:
        List of paths (each path is a list of node IDs)
    """
    paths = []
    if current_node not in G_req:
        return paths

    for n1 in G_req.neighbors(current_node):
        has_n2 = False
        for n2 in G_req.neighbors(n1):
            if n2 != current_node:
                paths.append([current_node, n1, n2])
                has_n2 = True
        if not has_n2:
            paths.append([current_node, n1])
    return paths


# ==========================================
# FLASK ROUTES
# ==========================================

@app.before_request
def startup():
    """Initialize graph on first request if not already done."""
    if not FRONTEND_NODES:
        initialize_graph()


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/data', methods=['GET'])
def get_data():
    """
    Get all nodes and edges for map visualization.
    
    Returns:
        JSON with nodes and edges arrays
    """
    return jsonify({"nodes": FRONTEND_NODES, "edges": FRONTEND_EDGES})


@app.route('/api/calculate_path', methods=['POST'])
def calculate_path():
    """
    Calculate an optimized route based on user preferences.
    
    Request Body:
        - difficulty: 'easy', 'medium', or 'hard'
        - season: 'winter' or 'summer'
        - allow_snow: boolean
    
    Returns:
        JSON with path coordinates, details, and total time
    """
    data = request.json
    
    user_profile = {
        'allow_snow': data.get('allow_snow', True),
        'difficulty': data.get('difficulty', 'medium'),
        'season': data.get('season', 'winter')
    }
    
    start_node = "3"  # Benasque
    G_req = G.copy()
    
    season = user_profile['season']
    terrain_key = 'terrain_winter' if season == 'winter' else 'terrain_summer'
    snow_identifiers = ['Snow', 'S']
    city_identifiers = ['Urban', 'Town', 'City', 'U']

    # Remove snow nodes if not allowed
    if not user_profile['allow_snow']:
        snow_nodes = [
            n for n in list(G_req.nodes) 
            if G_req.nodes[n].get(terrain_key) in snow_identifiers 
            or G_req.nodes[n].get('type') in snow_identifiers
        ]
        G_req.remove_nodes_from(snow_nodes)
        
    city_nodes = [
        n for n in list(G_req.nodes) 
        if G_req.nodes[n].get('type') in city_identifiers 
        or G_req.nodes[n].get(terrain_key) in city_identifiers 
        or "Besurta" in G_req.nodes[n].get('name', '')
    ]
    
    # Super node contraction for medium/hard difficulty
    super_node = "SUPER_CITY"
    entry_exit_map = {}

    if user_profile['difficulty'] == 'easy':
        # Easy: Isolate cities only
        non_city_nodes = [n for n in list(G_req.nodes) if n not in city_nodes]
        if start_node in non_city_nodes:
            non_city_nodes.remove(start_node)
        G_req.remove_nodes_from(non_city_nodes)
        start_node_algo = start_node
        
    else:
        # Medium/Hard: Contract cities into super node
        if len(city_nodes) > 1:
            G_req.add_node(
                super_node, 
                type='Urban', 
                elevation=0, 
                type_val_winter=0.2, 
                type_val_summer=0.2, 
                name="City Network"
            )
            
            # Rewire edges to point to super node
            for city in city_nodes:
                for neighbor in list(G_req.neighbors(city)):
                    if neighbor not in city_nodes:
                        weight = G_req[city][neighbor]['weight']
                        
                        if G_req.has_edge(super_node, neighbor):
                            if weight < G_req[super_node][neighbor]['weight']:
                                G_req[super_node][neighbor]['weight'] = weight
                                entry_exit_map[neighbor] = city
                        else:
                            G_req.add_edge(super_node, neighbor, weight=weight)
                            entry_exit_map[neighbor] = city

            G_req.remove_nodes_from(city_nodes)
            start_node_algo = super_node
        else:
            start_node_algo = start_node

    # Execute hybrid search
    route = [start_node_algo]
    current_node = start_node_algo
    target_steps = 5 if user_profile['difficulty'] == 'hard' else 3
    
    for step in range(target_steps):
        possible_paths = get_2_step_paths(G_req, current_node)
        if not possible_paths:
            break
            
        selected_path = quantum_selector(G_req, possible_paths, user_profile, route)
        if not selected_path:
            break
            
        next_node = selected_path[1]
        route.append(next_node)
        current_node = next_node
        
    # Return trip
    if current_node != start_node_algo:
        try:
            return_path = nx.shortest_path(
                G_req, 
                source=current_node, 
                target=start_node_algo, 
                weight='weight'
            )
            route.extend(return_path[1:])
        except nx.NetworkXNoPath:
            pass

    # Decompress super node back to reality
    if start_node_algo == super_node:
        final_route = []
        for i in range(len(route)):
            curr = route[i]
            if curr == super_node:
                if i == 0:  # Start of hike
                    if len(route) > 1:
                        exit_city = entry_exit_map[route[1]]
                        path = nx.shortest_path(G, source=start_node, target=exit_city, weight='weight')
                        final_route.extend(path)
                    else:
                        final_route.append(start_node)
                        
                elif i == len(route) - 1:  # End of hike
                    entry_city = entry_exit_map[route[i - 1]]
                    final_route.append(entry_city)
                    path = nx.shortest_path(G, source=entry_city, target=start_node, weight='weight')
                    if len(path) > 1:
                        final_route.extend(path[1:])
                        
                else:  # Passing through mid-hike
                    entry_city = entry_exit_map[route[i - 1]]
                    exit_city = entry_exit_map[route[i + 1]]
                    final_route.append(entry_city)
                    path = nx.shortest_path(G, source=entry_city, target=exit_city, weight='weight')
                    if len(path) > 1:
                        final_route.extend(path[1:])
            else:
                final_route.append(curr)
    else:
        final_route = route

    # Prepare response
    path_details = []
    total_time = 0.0
    path_coords = []

    for idx, r_id in enumerate(final_route):
        f_node = next((n for n in FRONTEND_NODES if n['graph_id'] == r_id), None)
        if f_node:
            lat = float(str(f_node['lat']).replace(',', '.'))
            lon = float(str(f_node['lon']).replace(',', '.'))
            path_coords.append([lat, lon])
        
        step_time = 0.0
        if idx > 0:
            prev_id = final_route[idx - 1]
            if G.has_edge(prev_id, r_id):
                step_time = float(G[prev_id][r_id].get('weight', 0.0))
                total_time += step_time
                
        node_data = G.nodes[r_id]
        path_details.append({
            'name': node_data.get('name', f"Node {r_id}"),
            'elevation': node_data.get('elevation', 0),
            'step_time': step_time
        })

    return jsonify({
        "path": path_coords, 
        "details": path_details, 
        "total_time": total_time
    })


if __name__ == '__main__':
    initialize_graph()
    app.run(debug=True, port=5000)
