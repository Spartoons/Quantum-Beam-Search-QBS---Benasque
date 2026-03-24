# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis and algorithm development.

## Notebooks

### exploration.ipynb
Main exploration notebook containing:
- Data loading and visualization
- Graph construction and analysis
- Algorithm prototyping
- Route visualization

### exploration_annika.ipynb
Team member Annika's exploration notebook with:
- Alternative algorithm approaches
- Data analysis experiments
- Feature engineering attempts

## Usage

```bash
# Start Jupyter
jupyter notebook

# Or with specific port
jupyter notebook --port 8888
```

## Data Dependencies

Notebooks expect data files in `../data/raw/`:
- `coordinates.csv`
- `distances.csv`
- `elevations.csv`
- `node_types.csv`
- `places.csv`
- `terrain.csv`

## Output

Generated visualizations should be saved to `../docs/images/` for use in documentation.
