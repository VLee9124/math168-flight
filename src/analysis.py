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
    all_files = [
        "monthly/T_T100 2018.csv",
        "monthly/T_T100 2019.csv",
        "monthly/T_T100 2020.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2021.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2022.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2023.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2024.csv",
        "monthly/T_T100D_MARKET_ALL_CARRIER 2_2025.csv",
        "T100.csv",
    ]
    
    dfs = []
    for path in all_files:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
        df = df[["ORIGIN", "DEST"]].dropna()
        df["ORIGIN"] = df["ORIGIN"].astype(str).str.strip().str.upper()
        df["DEST"] = df["DEST"].astype(str).str.strip().str.upper()
        df = df[df["ORIGIN"] != df["DEST"]]
        df["DEPARTURES"] = 1  # each row = 1 service record (carrier x route x month)
        dfs.append(df)
    
    routes = pd.concat(dfs, ignore_index=True)
    routes = routes[routes["ORIGIN"].isin(CORE_30) & routes["DEST"].isin(CORE_30)]
    
    # Sum rows per route = total service-record count (frequency proxy)
    # PAPER NOTE: this weight is the number of carrier-month service records on a
    # route, NOT a literal departure count. Describe it as a "service-frequency
    # proxy" in Methods/Limitations rather than "flight departures".
    route_weights = routes.groupby(["ORIGIN", "DEST"])["DEPARTURES"].sum().reset_index()
    
    G = nx.DiGraph()
    for iata in CORE_30:
        G.add_node(iata)
    
    for _, row in route_weights.iterrows():
        if row["DEPARTURES"] > 0:
            G.add_edge(row["ORIGIN"], row["DEST"], weight=row["DEPARTURES"])
    
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    return G

# Weighted centrality measures
def compute_centralities(G):
    # WEIGHTED: PageRank uses raw frequency weights.
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")
    # WEIGHTED: invert weights so high frequency = short distance for betweenness.
    for u, v, d in G.edges(data=True):
        d["inv_weight"] = 1.0 / d["weight"] if d["weight"] > 0 else float("inf")
    betweenness = nx.betweenness_centrality(G, weight="inv_weight")
    # UNWEIGHTED (hop-based): closeness ignores weights here.
    closeness = nx.closeness_centrality(G)
    # UNWEIGHTED by definition: in-degree counts incoming edges, not weights.
    degree = nx.in_degree_centrality(G)
    # WEIGHTED: nx.hits uses edge weights via the adjacency matrix (verified).
    hubs, authorities = nx.hits(G)
    return pagerank, betweenness, closeness, degree, hubs, authorities

def print_rankings(pagerank, betweenness, closeness, degree, hubs, authorities, top_n=10):
    def top(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def summarize_saturated(d, label):
        # FIX 3: closeness and in-degree saturate at the max in a near-complete
        # graph, so a "top N" list is an arbitrary slice of the ties. Report how
        # many are tied at the maximum and list only the airports BELOW it (the
        # real signal), instead of a meaningless ranking.
        mx = max(d.values())
        tied = sorted(k for k, v in d.items() if abs(v - mx) < 1e-9)
        below = sorted(((k, v) for k, v in d.items() if abs(v - mx) >= 1e-9),
                       key=lambda x: x[1])
        print(f"\n{'='*40}\n{label}\n{'='*40}")
        print(f"  {len(tied)} of {len(d)} airports tied at the maximum ({mx:.4f}):")
        print(f"    {', '.join(tied)}")
        if below:
            print(f"  Below maximum (the structural outliers):")
            for a, s in below:
                print(f"    {a}: {s:.4f}")
        else:
            print("  No airports fall below the maximum.")

    print(f"\n{'='*40}\nTOP {top_n} BY PAGERANK\n{'='*40}")
    for a, s in top(pagerank): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} BY BETWEENNESS\n{'='*40}")
    for a, s in top(betweenness): print(f"  {a}: {s:.4f}")

    # FIX 3: closeness and in-degree reported as saturation summaries, not rankings.
    # PAPER NOTE: do NOT present a "top 10 closeness/in-degree" table. State that
    # ~28 airports tie at 1.0 (near-complete graph) and that closeness adds no
    # information beyond in-degree here. HNL (and the one degree-28 airport) are
    # the only nodes below the max.
    summarize_saturated(closeness, "CLOSENESS CENTRALITY (saturated; not a ranking)")
    summarize_saturated(degree, "IN-DEGREE CENTRALITY (saturated; not a ranking)")

    print(f"\n{'='*40}\nTOP {top_n} HUBS (HITS)\n{'='*40}")
    for a, s in top(hubs): print(f"  {a}: {s:.4f}")

    print(f"\n{'='*40}\nTOP {top_n} AUTHORITIES (HITS)\n{'='*40}")
    for a, s in top(authorities): print(f"  {a}: {s:.4f}")


def plot_all_centralities(G, pagerank, betweenness, closeness, degree, hubs, authorities):
    pos = nx.spring_layout(G, seed=42)
    
    measures = [
        (pagerank, "PageRank (freq-weighted)"),
        (betweenness, "Betweenness (inverse-freq weighted)"),
        (closeness, "Closeness (unweighted)"),
        (degree, "In-Degree (unweighted)"),
        (hubs, "HITS Hubs (freq-weighted)"),
        (authorities, "HITS Authorities (freq-weighted)"),
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
    
    # FIX 4: blanket "Weighted by Flight Departures" overclaimed — closeness and
    # in-degree are not weighted. Title now says "where applicable" and each panel
    # is labelled with its own weighting above.
    # PAPER NOTE: update the Fig 2 caption to match — name which measures are
    # frequency-weighted (PageRank, betweenness, HITS) vs structural (closeness,
    # in-degree).
    plt.suptitle("Centrality Measures — FAA Core 30 US Airports (frequency-weighted where applicable)", 
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

def _cut_set_results(G):
    """Remove each airport once; record fragmentation and routes lost."""
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
            "edges_lost": edges_lost,
        })
    return results

def cut_set_analysis(G):
    print(f"\n{'='*40}")
    print("CUT SET ANALYSIS")
    print(f"{'='*40}")
    
    G_undirected = G.to_undirected()
    n = G.number_of_nodes()
    baseline_components = nx.number_connected_components(G_undirected)
    largest_cc_size = len(max(nx.connected_components(G_undirected), key=len))
    print(f"  Baseline components: {baseline_components}")
    print(f"  Baseline largest component size: {largest_cc_size}")
    
    results = _cut_set_results(G)
    
    # FIX 1 + FIX 2: the network never fragments (no cut vertices), so a
    # "most disruptive" ranking by largest component is meaningless (everything
    # ties at n-1). Report robustness explicitly, then rank by ROUTES LOST, the
    # only quantity that actually varies.
    # PAPER NOTE: reframe the resilience section. Correct story = "the Core 30
    # network has no single point of failure; removing any one airport leaves the
    # other 29 fully connected." Do NOT claim MDW/HNL are critical cut points.
    fragmenting = [r for r in results if r["largest_cc"] < n - 1]
    if fragmenting:
        print(f"\n  CUT VERTICES (removal fragments the network): "
              f"{[r['airport'] for r in fragmenting]}")
    else:
        print(f"\n  No cut vertices: removing any single airport leaves all "
              f"{n - 1} remaining airports connected (largest CC = {n - 1}).")
    
    # Rank by routes lost (descending) — this is what actually differs.
    results.sort(key=lambda x: x["edges_lost"], reverse=True)
    
    print(f"\n  AIRPORTS RANKED BY ROUTES LOST ON REMOVAL "
          f"(network stays connected throughout):")
    print(f"  {'Airport':<10} {'Components':<12} {'Largest CC':<12} {'Routes Lost'}")
    print(f"  {'-'*50}")
    for r in results[:10]:
        print(f"  {r['airport']:<10} {r['components']:<12} {r['largest_cc']:<12} {r['edges_lost']}")
    # Surface the least-connected airport explicitly (the structural outlier).
    least = min(results, key=lambda x: x["edges_lost"])
    print(f"\n  Fewest routes lost (most peripheral): "
          f"{least['airport']} ({least['edges_lost']} routes)")

def plot_cut_set(G):
    results = _cut_set_results(G)
    
    # FIX 1 + FIX 2: order by routes lost (the quantity that varies); no more
    # hard-coded red highlight for MDW/HNL, which the corrected results do not
    # support. Single colour per panel.
    results.sort(key=lambda x: x["edges_lost"])  # ascending => largest at top in barh
    airports = [r["airport"] for r in results]
    largest_ccs = [r["largest_cc"] for r in results]
    edges_lost = [r["edges_lost"] for r in results]
    n = G.number_of_nodes()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1 - largest connected component after removal (flat at n-1 = robust)
    axes[0].barh(airports, largest_ccs, color="steelblue")
    axes[0].set_xlabel("Largest Connected Component Size")
    axes[0].set_title("Network Resilience: Largest CC After Airport Removal")
    axes[0].axvline(x=n - 1, color="black", linestyle="--",
                    label=f"No fragmentation ({n - 1} nodes)")
    axes[0].legend()
    
    # Plot 2 - routes lost
    axes[1].barh(airports, edges_lost, color="darkorange")
    axes[1].set_xlabel("Number of Routes Lost")
    axes[1].set_title("Routes Lost After Airport Removal")
    
    plt.suptitle("Cut Set Analysis — FAA Core 30 Airports (no single point of failure)",
                 fontsize=14, fontweight="bold")
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