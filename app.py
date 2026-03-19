from flask import Flask, render_template, jsonify, request
import pandas as pd

app = Flask(__name__)

# --- Helper to load data once ---
def get_network_data():
    try:
        # Load CSVs
        coords = pd.read_csv('Untitled map.csv')
        types = pd.read_csv('node_types.csv', header=None).iloc[0].tolist()
        terrains = pd.read_csv('terrain.csv', header=None).iloc[0].tolist()
        
        nodes = []
        for i, row in coords.iterrows():
            if pd.notna(row['X']) and pd.notna(row['Y']):
                nodes.append({
                    'id': i,
                    'lat': row['Y'],
                    'lon': row['X'],
                    'type': str(types[i]).strip() if i < len(types) else 'Landmark',
                    'terrain': str(terrains[i]).strip() if i < len(terrains) else 'Mountain'
                })
        return nodes
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    nodes = get_network_data()
    return jsonify(nodes)

@app.route('/api/calculate_path', methods=['POST'])
def calculate_path():
    data = request.json
    difficulty = data.get('difficulty', 'medium')
    allow_snow = data.get('allow_snow', True)
    
    # -------------------------------------------------------------
    # ALGORITHM PLACEHOLDER
    # Here is where you will eventually put your pathfinding logic 
    # (like Dijkstra or A*). For now, we return a mock path of node IDs.
    # -------------------------------------------------------------
    
    nodes = get_network_data()
    if not nodes:
        return jsonify({"path": []})
        
    # Mock paths based on difficulty
    if difficulty == 'easy':
        mock_path_ids = [0, 1, 2] # Just an example sequence
    elif difficulty == 'hard':
        mock_path_ids = [0, 3, 5, 8, 12] 
    else:
        mock_path_ids = [0, 2, 4, 6]

    # If snow is restricted, pretend the algorithm routes around it
    if not allow_snow:
        mock_path_ids = [idx for idx in mock_path_ids if nodes[idx]['type'] != 'Snow']

    # Return the exact coordinates for the frontend to draw
    path_coords = [[nodes[i]['lat'], nodes[i]['lon']] for i in mock_path_ids if i < len(nodes)]
    
    return jsonify({"path": path_coords})

if __name__ == '__main__':
    app.run(debug=True, port=8000)