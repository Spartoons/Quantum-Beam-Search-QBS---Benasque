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

        try:
            with open('places.csv', 'r', encoding='utf-8') as f:
                # Read the single line, split by comma, and clean up whitespace
                places_names = [name.strip() for name in f.read().split(',') if name.strip()]
        except FileNotFoundError:
            places_names = [] # Fallback if file doesn't exist yet

        try:
            with open('elevations.csv', 'r', encoding='utf-8') as f:
                elevations_data = [float(e.strip()) for e in f.read().split(',') if e.strip()]
        except FileNotFoundError:
            elevations_data = []
        
        type_weights = {'Urban': 0.2, 'Trail': 0.4, 'Mountain': 0.6, 'Snow': 0.8}
        
        # 1. Build Nodes and frontend data
        for i, row in coords.iterrows():
            if str(row['X']) == '' or str(row['Y']) == '':
                continue
                
            node_id = str(i + 1)
            
            n_type = str(types[i]).strip() if i < len(types) and str(types[i]) != '' else 'Landmark'

            n_name = places_names[i] if i < len(places_names) else f"Nosde {node_id}"
            n_elev = elevations_data[i] if i < len(elevations_data) else "-" # <-- NEW

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
                'id': i, 'graph_id': node_id, 'name': n_name,
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
def tensor_network_heuristic(epsilon_scores, difficulty):
    scores = np.array(epsilon_scores, dtype=float)
    
    # FIX 1: Properly normalize the identical-scores fallback
    if np.all(scores == scores[0]):
        psi = np.ones_like(scores)
        return psi / np.linalg.norm(psi) 

    if difficulty == 'easy':
        scores = scores - np.min(scores) + 1e-5
        psi = 1.0 / scores
    elif difficulty == 'hard':
        psi = scores - np.min(scores) + 1e-5
    else: 
        target = np.mean(scores)
        distances_from_mean = np.abs(scores - target)
        psi = 1.0 / (distances_from_mean + 1e-5)
        
    return psi / np.linalg.norm(psi)

def calculate_epsilon(G_req, path, user_profile, current_route):
    cost = 0.0
    season = user_profile.get('season', 'winter') 
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        val_key = 'type_val_winter' if season == 'winter' else 'type_val_summer'
        val_u = float(G_req.nodes[u].get(val_key, 0.6))
        val_v = float(G_req.nodes[v].get(val_key, 0.6))
        mean_type = np.mean([val_u, val_v]) 
        
        edge_data = G_req[u][v]
        time_val = float(edge_data.get("weight", 0.1))
        
        elev_u = float(G_req.nodes[u].get("elevation", 0.0))
        elev_v = float(G_req.nodes[v].get("elevation", 0.0))
        
        # FIX 2: Absolute value so going downhill (negative) doesn't break the tensor score
        dif_height = abs(elev_v - elev_u)  
        if dif_height < 1: 
            dif_height = 1.0
        
        step_cost = (mean_type * time_val * dif_height)
        
        # FIX 3: THE BACKTRACK PENALTY
        # If the node is already in our route, make it 10x more expensive.
        if v in current_route:
            step_cost *= 10.0
            
        cost += step_cost
        
    return cost

def quantum_selector(G_req, possible_paths, user_profile, current_route):
    print(f"  -> Evaluating {len(possible_paths)} paths with Quantum Pipeline...")
    
    if not possible_paths:
        return []
        
    if len(possible_paths) == 1:
        print(f"  -> Only 1 path available. Deterministic bypass.")
        return possible_paths[0]

    epsilon_scores = []
    for path in possible_paths:
        # Pass the route down to the heuristic
        epsilon_scores.append(calculate_epsilon(G_req, path, user_profile, current_route))
        
    psi = tensor_network_heuristic(epsilon_scores, user_profile.get('difficulty', 'medium'))
    
    try:
        counts = choose_path(psi, rep=1)
        measured_binary = list(counts.keys())[0]
        measured_idx = int(measured_binary, 2)
    except Exception as e:
        print(f"[!] Quantum Simulator Exception: {e}")
        print(f"[!] Failing gracefully to Classical Random Selector...")
        measured_idx = random.randint(0, len(possible_paths) - 1)
        
    if measured_idx >= len(possible_paths):
        measured_idx = measured_idx % len(possible_paths)
        
    selected_path = possible_paths[measured_idx]
    print(f"  -> Collapsed on Index {measured_idx} (Epsilon Score: {epsilon_scores[measured_idx]:.2f})")
    
    return selected_path

def get_2_step_paths(G_req, current_node):
    paths = []
    # Safety check in case the current node was deleted by constraints
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
    
    user_profile = {
        'allow_snow': data.get('allow_snow', True),
        'difficulty': data.get('difficulty', 'medium'),
        'season': data.get('season', 'winter')
    }
    
    start_node = "3" # Benasque
    G_req = G.copy()
    
    season = user_profile['season']
    terrain_key = 'terrain_winter' if season == 'winter' else 'terrain_summer'
    snow_identifiers = ['Snow', 'S']
    city_identifiers = ['Urban', 'Town', 'City', 'U']

    # RULE 1: Remove Snow nodes entirely
    if not user_profile['allow_snow']:
        snow_nodes = [n for n in list(G_req.nodes) if G_req.nodes[n].get(terrain_key) in snow_identifiers or G_req.nodes[n].get('type') in snow_identifiers]
        G_req.remove_nodes_from(snow_nodes)
        
    city_nodes = [n for n in list(G_req.nodes) if G_req.nodes[n].get('type') in city_identifiers or G_req.nodes[n].get(terrain_key) in city_identifiers or "Besurta" in G_req.nodes[n].get('name', '')]
    
    # ==========================================
    # RULE 2 & 3: SUPER NODE CONTRACTION
    # ==========================================
    super_node = "SUPER_CITY"
    entry_exit_map = {} # Remembers which real city connects to which mountain

    if user_profile['difficulty'] == 'easy':
        # EASY: Isolate Cities
        non_city_nodes = [n for n in list(G_req.nodes) if n not in city_nodes]
        if start_node in non_city_nodes: non_city_nodes.remove(start_node)
        G_req.remove_nodes_from(non_city_nodes)
        start_node_algo = start_node
        
    else:
        # MEDIUM/HARD: True Contraction
        if len(city_nodes) > 1:
            # 1. Create the massive Super City
            G_req.add_node(super_node, type='Urban', elevation=0, type_val_winter=0.2, type_val_summer=0.2, name="City Network")
            
            # 2. Re-wire all mountain/trail edges to point to the Super City instead
            for city in city_nodes:
                for neighbor in list(G_req.neighbors(city)):
                    if neighbor not in city_nodes:
                        weight = G_req[city][neighbor]['weight']
                        
                        # Save the shortest entry/exit point
                        if G_req.has_edge(super_node, neighbor):
                            if weight < G_req[super_node][neighbor]['weight']:
                                G_req[super_node][neighbor]['weight'] = weight
                                entry_exit_map[neighbor] = city
                        else:
                            G_req.add_edge(super_node, neighbor, weight=weight)
                            entry_exit_map[neighbor] = city

            # 3. Destroy the original internal cities so the algorithm can't get stuck in them
            G_req.remove_nodes_from(city_nodes)
            start_node_algo = super_node
        else:
            start_node_algo = start_node

    # ==========================================
    # EXECUTE HYBRID SEARCH
    # ==========================================
    route = [start_node_algo]
    current_node = start_node_algo
    target_steps = 5 if user_profile['difficulty'] == 'hard' else 3
    
    for step in range(target_steps):
        possible_paths = get_2_step_paths(G_req, current_node)
        if not possible_paths: break
            
        selected_path = quantum_selector(G_req, possible_paths, user_profile, route)
        if not selected_path: break
            
        next_node = selected_path[1] 
        route.append(next_node)
        current_node = next_node
        
    # Return Trip
    if current_node != start_node_algo:
        try:
            return_path = nx.shortest_path(G_req, source=current_node, target=start_node_algo, weight='weight')
            route.extend(return_path[1:]) 
        except nx.NetworkXNoPath:
            pass

    # ==========================================
    # DECOMPRESS SUPER NODE BACK TO REALITY
    # ==========================================
    if start_node_algo == super_node:
        final_route = []
        for i in range(len(route)):
            curr = route[i]
            if curr == super_node:
                if i == 0: # Start of hike: leaving the city
                    if len(route) > 1:
                        exit_city = entry_exit_map[route[1]]
                        path = nx.shortest_path(G, source=start_node, target=exit_city, weight='weight')
                        final_route.extend(path)
                    else:
                        final_route.append(start_node)
                        
                elif i == len(route) - 1: # End of hike: returning to city
                    entry_city = entry_exit_map[route[i-1]]
                    final_route.append(entry_city)
                    path = nx.shortest_path(G, source=entry_city, target=start_node, weight='weight')
                    if len(path) > 1: final_route.extend(path[1:])
                        
                else: # Passing through the city network mid-hike
                    entry_city = entry_exit_map[route[i-1]]
                    exit_city = entry_exit_map[route[i+1]]
                    final_route.append(entry_city)
                    path = nx.shortest_path(G, source=entry_city, target=exit_city, weight='weight')
                    if len(path) > 1: final_route.extend(path[1:])
            else:
                final_route.append(curr)
    else:
        final_route = route

    # ==========================================
    # PREPARE RICH UI RESPONSE
    # ==========================================
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
            prev_id = final_route[idx-1]
            if G.has_edge(prev_id, r_id):
                step_time = float(G[prev_id][r_id].get('weight', 0.0))
                total_time += step_time
                
        node_data = G.nodes[r_id]
        path_details.append({
            'name': node_data.get('name', f"Node {r_id}"),
            'elevation': node_data.get('elevation', 0),
            'step_time': step_time
        })

    return jsonify({"path": path_coords, "details": path_details, "total_time": total_time})

if __name__ == '__main__':
    initialize_graph()
    app.run(debug=True, port=5000)