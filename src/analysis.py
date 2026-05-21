import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from main import build_graph  # imports Victor's build_graph from src/main.py

def compute_centralities(G):
    pagerank = nx.pagerank(G, alpha=0.85)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    return pagerank, betweenness, closeness

def print_rankings(pagerank, betweenness, closeness, top_n=10):
    print(f"\n{'='*40}")
    print(f"TOP {top_n} AIRPORTS BY PAGERANK")
    print(f"{'='*40}")
    for airport, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"  {airport}: {score:.4f}")

    print(f"\n{'='*40}")
    print(f"TOP {top_n} AIRPORTS BY BETWEENNESS CENTRALITY")
    print(f"{'='*40}")
    for airport, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"  {airport}: {score:.4f}")

    print(f"\n{'='*40}")
    print(f"TOP {top_n} AIRPORTS BY CLOSENESS CENTRALITY")
    print(f"{'='*40}")
    for airport, score in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"  {airport}: {score:.4f}")

def plot_centrality(G, measures, title):
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
    plt.show()

if __name__ == "__main__":
    print("Building graph...")
    G = build_graph()
    print(f"Graph: {G.number_of_nodes()} airports, {G.number_of_edges()} routes")

    print("Computing centralities...")
    pagerank, betweenness, closeness = compute_centralities(G)

    print_rankings(pagerank, betweenness, closeness)

    plot_centrality(G, pagerank, "PageRank Centrality — Core 30 US Airports")
    plot_centrality(G, betweenness, "Betweenness Centrality — Core 30 US Airports")
    plot_centrality(G, closeness, "Closeness Centrality — Core 30 US Airports")