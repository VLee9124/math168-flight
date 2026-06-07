# math168-flight

Flight-network analysis for U.S. airport routes, with a focus on the FAA Core 30 network.

## Data Sources

- **T100 Domestic Market (All Carriers)**: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=GED
- **OurAirports**: https://ourairports.com/ 

## Setup

1. Clone the repository:
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Analysis and Visualizations

### Core 30 Airport Network Analysis

Run the Core 30 analysis using `python monthly/core30.py` to compute node-level metrics for the Core 30 airport network. Monthly snapshot plots are stored in `temp/` and include 2 types of plots, centrality plots and geographic route maps for a MONTH YEAR. 

The script also contains helper functions to generate visualizations for an overall summary of the Core 30 network and monthly averages. 

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
