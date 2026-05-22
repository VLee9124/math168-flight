import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os

from main import build_graph
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

ROUTES_CSV = "T100.csv"
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

CORE_30 = {"ATL", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW", "EWR",
           "FLL", "HNL", "IAD", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO",
           "MDW", "MEM", "MIA", "MSP", "ORD", "PHL", "PHX", "SAN", "SEA",
           "SFO", "SLC", "TPA"}

def build_weighted_graph() -> nx.DiGraph:
    airports = pd.read_csv(OURAIRPORTS_URL)
    airports = airports[
        (airports["iso_country"] == "US")
        & airports["iata_code"].notna()
        & (airports["iata_code"].str.len() == 3)
    ].copy()
    airports["iata_code"] = airports["iata_code"].str.upper()

    routes = pd.read_csv(ROUTES_CSV, low_memory=False)
    routes.columns = routes.columns.str.strip().str.upper().str.replace(" ", "_")
    routes = routes[["ORIGIN", "DEST", "PASSENGERS"]].dropna()
    routes["ORIGIN"] = routes["ORIGIN"].astype(str).str.strip().str.upper()
    routes["DEST"] = routes["DEST"].astype(str).str.strip().str.upper()
    routes = routes[routes["ORIGIN"] != routes["DEST"]]

    # Filter to Core 30 only
    routes = routes[routes["ORIGIN"].isin(CORE_30) & routes["DEST"].isin(CORE_30)]

    # Aggregate passengers per route
    route_weights = routes.groupby(["ORIGIN", "DEST"])["PASSENGERS"].sum().reset_index()

    G = nx.DiGraph()
    for iata in CORE_30:
        G.add_node(iata)

    for _, row in route_weights.iterrows():
        if row["PASSENGERS"] > 0:
            G.add_edge(row["ORIGIN"], row["DEST"], weight=row["PASSENGERS"])

    # Remove isolated nodes
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    return G

# Weighted centrality measures
def compute_centralities(G):
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness = nx.closeness_centrality(G)
    degree = nx.in_degree_centrality(G)
    hubs, authorities = nx.hits(G)
    return pagerank, betweenness, closeness, degree, hubs, authorities

def print_rankings(pagerank, betweenness, closeness, degree, hubs, authorities, top_n=10):
    def top(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n{'='*40}\nTOP {top_n} BY PAGERANK\n{'='*40}")
    for a, s in top(pagerank): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} BY BETWEENNESS\n{'='*40}")
    for a, s in top(betweenness): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} BY CLOSENESS\n{'='*40}")
    for a, s in top(closeness): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} BY IN-DEGREE CENTRALITY\n{'='*40}")
    for a, s in top(degree): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} HUBS (HITS)\n{'='*40}")
    for a, s in top(hubs): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} AUTHORITIES (HITS)\n{'='*40}")
    for a, s in top(authorities): print(f"  {a}: {s:.4f}")

def plot_centrality(G : nx.DiGraph, measures: dict, title: str, save_path: str = "snapshots/"):
    pos = nx.spring_layout(G, seed=42)
    values = list(measures.values())

    plt.figure(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=400,
        cmap=plt.cm.plasma,
        node_color=values,
    )
    nodes.set_norm(mcolors.Normalize(vmin=min(values), vmax=max(values)))
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white")
    nx.draw_networkx_edges(G, pos, alpha=0.4, arrows=True, arrowsize=10)
    plt.colorbar(nodes)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "WEIGHTED_" + title.replace(' ', '_').lower() + ".png", dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    print("Building weighted graph...")
    G = build_weighted_graph()
    print(f"Graph: {G.number_of_nodes()} airports, {G.number_of_edges()} routes")

    print("Computing centralities...")
    pagerank, betweenness, closeness, degree, hubs, authorities = compute_centralities(G)

    print_rankings(pagerank, betweenness, closeness, degree, hubs, authorities)

    plot_centrality(G, pagerank, "PageRank — Core 30 US Airports (weighted by passengers)")
    plot_centrality(G, betweenness, "Betweenness — Core 30 US Airports (weighted)")
    plot_centrality(G, closeness, "Closeness — Core 30 US Airports")
    plot_centrality(G, degree, "In-Degree — Core 30 US Airports")
    plot_centrality(G, hubs, "HITS Hubs — Core 30 US Airports")
    plot_centrality(G, authorities, "HITS Authorities — Core 30 US Airports")

# Unweighted centrality measures (for comparison)
def compute_centralities(G):
    pagerank = nx.pagerank(G, alpha=0.85)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree = nx.in_degree_centrality(G)
    hubs, authorities = nx.hits(G)
    return pagerank, betweenness, closeness, degree, hubs, authorities

def print_rankings(pagerank, betweenness, closeness, degree, hubs, authorities, top_n=10):
    def top(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n{'='*40}\nTOP {top_n} BY PAGERANK\n{'='*40}")
    for a, s in top(pagerank): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} BY BETWEENNESS\n{'='*40}")
    for a, s in top(betweenness): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} BY CLOSENESS\n{'='*40}")
    for a, s in top(closeness): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} BY IN-DEGREE CENTRALITY\n{'='*40}")
    for a, s in top(degree): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} HUBS (HITS)\n{'='*40}")
    for a, s in top(hubs): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} AUTHORITIES (HITS)\n{'='*40}")
    for a, s in top(authorities): print(f"  {a}: {s:.4f}")

def plot_centrality(G: nx.DiGraph, measures: dict, title: str, save_path: str = "snapshots/"):
    pos = nx.spring_layout(G, seed=42)
    values = list(measures.values())
    
    plt.figure(figsize=(12, 8))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=400,
        cmap=plt.cm.plasma,
        node_color=values,
    )
    nodes.set_norm(mcolors.Normalize(vmin=min(values), vmax=max(values)))
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white")
    nx.draw_networkx_edges(G, pos, alpha=0.4, arrows=True, arrowsize=10)
    plt.colorbar(nodes)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "UNWEIGHTED_" + title.replace(' ', '_').lower() + ".png", dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    print("Building graph...")
    G = build_graph()
    print(f"Graph: {G.number_of_nodes()} airports, {G.number_of_edges()} routes")

    print("Computing centralities...")
    pagerank, betweenness, closeness, degree, hubs, authorities = compute_centralities(G)

    print_rankings(pagerank, betweenness, closeness, degree, hubs, authorities)

    # TODO: CONVERT THIS INTO MATPLOTLIB SUBPLOTS INSTEAD OF SEPARATE PLOTS
    plot_centrality(G, pagerank, "PageRank — Core 30 US Airports")
    plot_centrality(G, betweenness, "Betweenness Centrality — Core 30 US Airports")
    plot_centrality(G, closeness, "Closeness Centrality — Core 30 US Airports")
    plot_centrality(G, degree, "In-Degree Centrality — Core 30 US Airports")
    plot_centrality(G, hubs, "HITS Hubs — Core 30 US Airports")
    plot_centrality(G, authorities, "HITS Authorities — Core 30 US Airports")
    
