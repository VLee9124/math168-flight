# math168-flight

Flight-network analysis for U.S. airport routes, with a focus on the FAA Core 30 network.

## Edge Criticality Analysis

Use `src/edge_analysis.py` to compute edge-level metrics on directed routes between Core 30 airports.

The script outputs:
- route passengers
- edge betweenness centrality
- passenger share of the whole Core 30 network
- directional imbalance vs reverse route
- a composite `criticality_score` that blends flow and structural importance

Run:

`python src/edge_analysis.py --routes-csv T100.csv --output-csv edge_metrics_core30.csv --top-n 20`

## Edge Visual Storytelling (New)

Use `src/edge_visuals.py` to generate a full set of visuals that highlight edge semantics and route significance.

Run:

`python src/edge_visuals.py --routes-csv T100.csv --output-dir visuals --top-n 20`

This command creates:
- `visuals/edge_top_critical_routes.png`
- `visuals/edge_semantic_quadrants.png`
- `visuals/edge_directional_imbalance.png`
- `visuals/edge_critical_corridor_network.png`
- `visuals/edge_metrics_core30.csv`
