import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


ROUTES_CSV = "./T100.csv"
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

#print(pd.read_csv(ROUTES_CSV, low_memory=False))

# Core 30 airports designated by FAA; set for hash lookup
CORE_30 = {"ATL", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW", "EWR", "FLL", "HNL", "IAD", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO", "MDW", "MEM", "MIA", "MSP", "ORD", "PHL", "PHX", "SAN", "SEA", "SFO", "SLC", "TPA"}

def load_airports(airports_url: str) -> pd.DataFrame:
    airports = pd.read_csv(airports_url)

    # Filter out non US airports 
    airports = airports[
        (airports["iso_country"] == "US")
        & airports["iata_code"].notna()
        & (airports["iata_code"].str.len() == 3)
    ].copy()
    
    airports["iata_code"] = airports["iata_code"].str.upper() # 3 Capitalized Letter IATA code

    return airports
  
def load_routes(routes_csv: str) -> pd.DataFrame:
    routes = pd.read_csv(routes_csv, low_memory=False)

    routes.columns = (
        routes.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    routes = routes[["ORIGIN", "DEST"]].dropna()

    routes["ORIGIN"] = routes["ORIGIN"].astype(str).str.strip().str.upper()
    routes["DEST"] = routes["DEST"].astype(str).str.strip().str.upper()

    routes = routes[routes["ORIGIN"] != routes["DEST"]]

    return routes
  
  
# Test: Build the Core 30 graph
def build_graph() -> nx.DiGraph:
    airports = load_airports(OURAIRPORTS_URL)
    routes = load_routes(ROUTES_CSV)

    G = nx.DiGraph()

    # Add all airports as nodes
    for _, row in airports.iterrows():
        if row["iata_code"] in CORE_30:
            G.add_node(row["iata_code"])

    # Add all routes as edges
    for _, row in routes.iterrows():
        if row["ORIGIN"] in CORE_30 and row["DEST"] in CORE_30:
            G.add_edge(row["ORIGIN"], row["DEST"])

    return G

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx


def draw(G, pos, measures, measure_name):
    """
    Input:
      G: NetworkX graph
      pos: Position dictionary
      measures: Dictionary of node measures e.g. betweenness centrality {node: measure_value}, degree centrality {node: measure_value}, etc.
      measure_name: Name of the measure to display
    """
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=250,
        cmap=plt.cm.plasma,
        node_color=list(measures.values()),
        nodelist=measures.keys(),
    )
    nodes.set_norm(mcolors.Normalize(vmin=min(measures.values()), vmax=max(measures.values())))
    labels = nx.draw_networkx_labels(G, pos, font_size=8)
    edges = nx.draw_networkx_edges(G, pos)

    plt.title(measure_name)

    plt.colorbar(nodes, ax=plt.gca())
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    graph = build_graph()
    # Swap the measures in 3rd argument to visualize different centrality measures
    draw(graph, nx.spring_layout(graph, seed=675), nx.betweenness_centrality(graph), "Flight Routes Between Core 30 US Airports")
    
    # Debugging: Print the number of nodes and edges
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

