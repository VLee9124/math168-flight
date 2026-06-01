import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

CORE_30 = {"ATL", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW", "EWR",
           "FLL", "HNL", "IAD", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO",
           "MDW", "MEM", "MIA", "MSP", "ORD", "PHL", "PHX", "SAN", "SEA",
           "SFO", "SLC", "TPA"}

def build_weighted_graph() -> nx.DiGraph:
    dfs = []
    
    # 2018-2020 files (no PASSENGERS column — use 1 as proxy)
    files_2018_2020 = [
        "monthly/T_T100 2018.csv",
        "monthly/T_T100 2019.csv",
        "monthly/T_T100 2020.csv",
    ]
    for path in files_2018_2020:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.upper()
        df = df[["ORIGIN", "DEST"]].dropna()
        df["ORIGIN"] = df["ORIGIN"].astype(str).str.strip().str.upper()
        df["DEST"] = df["DEST"].astype(str).str.strip().str.upper()
        df["PASSENGERS"] = 1
        dfs.append(df)
    
    # 2021-2025 + 2026 files (have PASSENGERS column)
    files_with_passengers = [
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2021.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2022.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2023.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2024.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2025.csv",
        "T100.csv",
    ]
    for path in files_with_passengers:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
        df = df[["ORIGIN", "DEST", "PASSENGERS"]].dropna()
        df["ORIGIN"] = df["ORIGIN"].astype(str).str.strip().str.upper()
        df["DEST"] = df["DEST"].astype(str).str.strip().str.upper()
        dfs.append(df)
    
    routes = pd.concat(dfs, ignore_index=True)
    routes = routes[routes["ORIGIN"] != routes["DEST"]]
    routes = routes[routes["ORIGIN"].isin(CORE_30) & routes["DEST"].isin(CORE_30)]
    route_weights = routes.groupby(["ORIGIN", "DEST"])["PASSENGERS"].sum().reset_index()
    
    G = nx.DiGraph()
    for iata in CORE_30:
        G.add_node(iata)
    for _, row in route_weights.iterrows():
        if row["PASSENGERS"] > 0:
            G.add_edge(row["ORIGIN"], row["DEST"], weight=row["PASSENGERS"])
    
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


def plot_all_centralities(G, pagerank, betweenness, closeness, degree, hubs, authorities):
    pos = nx.spring_layout(G, seed=42)
    
    measures = [
        (pagerank, "PageRank"),
        (betweenness, "Betweenness Centrality"),
        (closeness, "Closeness Centrality"),
        (degree, "In-Degree Centrality"),
        (hubs, "HITS Hubs"),
        (authorities, "HITS Authorities"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (measure, title) in enumerate(measures):
        ax = axes[i]
        values = [measure[node] for node in G.nodes()]
        
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=400,
            cmap=plt.cm.plasma,
            node_color=values,
            ax=ax
        )
        nodes.set_norm(mcolors.Normalize(vmin=min(values), vmax=max(values)))
        nx.draw_networkx_labels(G, pos, font_size=7, font_color="white", ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=8, ax=ax)
        plt.colorbar(nodes, ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    
    plt.suptitle("Centrality Measures — FAA Core 30 US Airports (Weighted by Passengers)", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("centrality_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved as centrality_comparison.png")

def compute_graph_stats(G):
    print(f"\n{'='*40}")
    print("GRAPH-LEVEL STATISTICS")
    print(f"{'='*40}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Average shortest path length and diameter (on undirected version)
    G_undirected = G.to_undirected()
    if nx.is_connected(G_undirected):
        avg_path = nx.average_shortest_path_length(G_undirected)
        diameter = nx.diameter(G_undirected)
        print(f"  Average Shortest Path Length: {avg_path:.4f}")
        print(f"  Diameter: {diameter}")
    else:
        largest_cc = G_undirected.subgraph(max(nx.connected_components(G_undirected), key=len))
        avg_path = nx.average_shortest_path_length(largest_cc)
        diameter = nx.diameter(largest_cc)
        print(f"  Average Shortest Path Length (largest component): {avg_path:.4f}")
        print(f"  Diameter (largest component): {diameter}")

def plot_degree_distribution(G):
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(in_degrees, bins=10, color="mediumpurple", edgecolor="black")
    axes[0].set_title("In-Degree Distribution — Core 30 Airports")
    axes[0].set_xlabel("In-Degree")
    axes[0].set_ylabel("Number of Airports")

    axes[1].hist(out_degrees, bins=10, color="darkorange", edgecolor="black")
    axes[1].set_title("Out-Degree Distribution — Core 30 Airports")
    axes[1].set_xlabel("Out-Degree")
    axes[1].set_ylabel("Number of Airports")

    plt.tight_layout()
    plt.savefig("degree_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved as degree_distribution.png")

def cut_set_analysis(G):
    print(f"\n{'='*40}")
    print("CUT SET ANALYSIS")
    print(f"{'='*40}")
    
    # Baseline stats
    G_undirected = G.to_undirected()
    baseline_components = nx.number_connected_components(G_undirected)
    largest_cc_size = len(max(nx.connected_components(G_undirected), key=len))
    print(f"  Baseline components: {baseline_components}")
    print(f"  Baseline largest component size: {largest_cc_size}")
    
    # Remove each airport one at a time and measure impact
    results = []
    for airport in G.nodes():
        G_temp = G.copy()
        G_temp.remove_node(airport)
        G_temp_undirected = G_temp.to_undirected()
        
        components = nx.number_connected_components(G_temp_undirected)
        largest_cc = len(max(nx.connected_components(G_temp_undirected), key=len))
        edges_lost = G.number_of_edges() - G_temp.number_of_edges()
        
        results.append({
            "airport": airport,
            "components": components,
            "largest_cc": largest_cc,
            "edges_lost": edges_lost
        })
    
    # Sort by largest_cc ascending (most disruptive first)
    results.sort(key=lambda x: x["largest_cc"])
    
    print(f"\n  TOP 10 MOST DISRUPTIVE AIRPORTS TO REMOVE:")
    print(f"  {'Airport':<10} {'Components':<12} {'Largest CC':<12} {'Edges Lost'}")
    print(f"  {'-'*50}")
    for r in results[:10]:
        print(f"  {r['airport']:<10} {r['components']:<12} {r['largest_cc']:<12} {r['edges_lost']}")

def plot_cut_set(G):
    G_undirected = G.to_undirected()
    
    results = []
    for airport in G.nodes():
        G_temp = G.copy()
        G_temp.remove_node(airport)
        largest_cc = len(max(nx.connected_components(G_temp.to_undirected()), key=len))
        edges_lost = G.number_of_edges() - G_temp.number_of_edges()
        results.append({"airport": airport, "largest_cc": largest_cc, "edges_lost": edges_lost})
    
    results.sort(key=lambda x: x["largest_cc"])
    airports = [r["airport"] for r in results]
    largest_ccs = [r["largest_cc"] for r in results]
    edges_lost = [r["edges_lost"] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1 - largest connected component after removal
    colors = ["red" if a in ("MEM", "HNL") else "steelblue" for a in airports]
    axes[0].barh(airports, largest_ccs, color=colors)
    axes[0].set_xlabel("Largest Connected Component Size")
    axes[0].set_title("Network Resilience: Largest CC After Airport Removal")
    axes[0].axvline(x=29, color="black", linestyle="--", label="Baseline (30 nodes)")
    axes[0].legend()
    
    # Plot 2 - edges lost
    colors2 = ["red" if a in ("MEM", "HNL") else "darkorange" for a in airports]
    axes[1].barh(airports, edges_lost, color=colors2)
    axes[1].set_xlabel("Number of Routes Lost")
    axes[1].set_title("Routes Lost After Airport Removal")
    
    plt.suptitle("Cut Set Analysis — FAA Core 30 Airports", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cut_set_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved as cut_set_analysis.png")

if __name__ == "__main__":
    print("Building weighted graph...")
    G = build_weighted_graph()
    print(f"Graph: {G.number_of_nodes()} airports, {G.number_of_edges()} routes")

    print("Computing centralities...")
    pagerank, betweenness, closeness, degree, hubs, authorities = compute_centralities(G)

    print_rankings(pagerank, betweenness, closeness, degree, hubs, authorities)
    plot_all_centralities(G, pagerank, betweenness, closeness, degree, hubs, authorities)
    compute_graph_stats(G)
    plot_degree_distribution(G)
    cut_set_analysis(G)
    plot_cut_set(G)

