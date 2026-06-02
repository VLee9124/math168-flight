import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import math
import os
import multiprocessing as mp

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

CORE_30 = {
    "ATL", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW", "EWR", "FLL",
    "HNL", "IAD", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO", "MDW", "MEM",
    "MIA", "MSP", "ORD", "PHL", "PHX", "SAN", "SEA", "SFO", "SLC", "TPA",
}


def load_airport_coords() -> dict:
    """Fetch lat/lon for Core 30 airports from OurAirports. Called once at startup."""
    airports = pd.read_csv(OURAIRPORTS_URL)
    airports = airports[
        (airports["iso_country"] == "US")
        & airports["iata_code"].isin(CORE_30)
    ][["iata_code", "municipality", "latitude_deg", "longitude_deg"]]

    return {
        row["iata_code"]: {
            "city":      row["municipality"],
            "latitude":  row["latitude_deg"],
            "longitude": row["longitude_deg"],
        }
        for _, row in airports.iterrows()
    }


def load_routes(routes_csv: str, month: int = None) -> pd.DataFrame:
    routes = pd.read_csv(routes_csv, low_memory=False)
    routes.columns = routes.columns.str.strip().str.upper().str.replace(" ", "_")
    routes = routes[["ORIGIN", "DEST", "YEAR", "MONTH"]].dropna()
    routes["ORIGIN"] = routes["ORIGIN"].astype(str).str.strip().str.upper()
    routes["DEST"]   = routes["DEST"].astype(str).str.strip().str.upper()
    routes = routes[routes["ORIGIN"] != routes["DEST"]]
    if month is not None:
        routes = routes[routes["MONTH"] == month]

    # Each row = 1 flight record, sum rows per route to get total flights
    routes["DEPARTURES_PERFORMED"] = 1
    return (
        routes.groupby(["ORIGIN", "DEST", "YEAR", "MONTH"], as_index=False)
        ["DEPARTURES_PERFORMED"].sum()
    )


def build_graph(routes_csv: str, airport_coords: dict, month: int = None) -> nx.DiGraph:
    routes = load_routes(routes_csv, month)
    routes = routes[
        routes["ORIGIN"].isin(CORE_30) & routes["DEST"].isin(CORE_30)
    ]

    G = nx.DiGraph()

    for iata, attrs in airport_coords.items():
        G.add_node(iata, **attrs)

    for _, row in routes.iterrows():
        o, d = row["ORIGIN"], row["DEST"]
        w = float(row["DEPARTURES_PERFORMED"])
        if G.has_edge(o, d):
            G[o][d]["weight"] += w
        else:
            G.add_edge(o, d, weight=w)

    return G


def display_graph(G: nx.DiGraph, year: int, month: int, save_path: str = None) -> None:
    pos = {n: (d["longitude"], d["latitude"]) for n, d in G.nodes(data=True)}

    out_weights = {
        n: sum(d["weight"] for _, _, d in G.out_edges(n, data=True))
        for n in G.nodes()
    }
    max_w      = max(out_weights.values()) if out_weights else 1
    node_sizes = [300 + 1200 * (out_weights.get(n, 0) / max_w) for n in G.nodes()]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_ew  = max(weights) if weights else 1
    widths  = [0.3 + 2.5 * (w / max_ew) for w in weights]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=widths, arrows=True, arrowsize=8,
        alpha=0.35, edge_color="#5DBB63",
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes, node_color="#5DBB63", alpha=0.9,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7, font_color="white", font_weight="bold",
    )

    ax.set_title(f"Core 30 U.S. Airport Route Graph — {year}-{month:02d}",
                 color="white", fontsize=14, pad=12)
    ax.set_xlabel("Longitude", color="gray")
    ax.set_ylabel("Latitude",  color="gray")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    saved {save_path}")
    else:
        plt.show()


def print_departure_stats(G: nx.DiGraph) -> pd.DataFrame:
    """Print total departures and arrivals per airport."""
    stats = {}
    for node in G.nodes():
        total_out = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
        total_in  = sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
        stats[node] = {"departures": total_out, "arrivals": total_in}

    df = pd.DataFrame(stats).T.sort_values("departures", ascending=False)
    print(df)
    return df


def compute_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """Compute key centrality measures for all airports."""
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness   = nx.closeness_centrality(G)
    pagerank    = nx.pagerank(G, weight="weight")
    in_degree   = nx.in_degree_centrality(G)
    out_degree  = nx.out_degree_centrality(G)

    df = pd.DataFrame({
        "betweenness": betweenness,
        "closeness":   closeness,
        "pagerank":    pagerank,
        "in_degree":   in_degree,
        "out_degree":  out_degree,
    }).sort_values("pagerank", ascending=False)

    print(df.round(4))
    return df


def plot_centrality_heatmaps(G: nx.DiGraph, centrality_df: pd.DataFrame,
                             year: int = None, month: int = None,
                             save_path: str = None) -> None:
    """Draw one network heatmap per centrality measure, colouring nodes by score.

    Mirrors the centrality subplot grid from analysis.py, driven by the
    DataFrame produced by compute_centrality().
    """
    # Spring layout (matching analysis.py) rather than the geographic layout used
    # by display_graph(): a true map crams the Northeast hub cluster (BOS/LGA/EWR/
    # PHL/DCA/IAD/JFK/BWI) into an unreadable blob, so an even spread keeps every
    # node label legible.
    pos = nx.spring_layout(G, seed=42)

    measures = list(centrality_df.columns)
    ncols = 3
    nrows = math.ceil(len(measures) / ncols)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    fig.patch.set_facecolor("#0d1117")
    axes = axes.flatten()

    nodes_in_order = list(G.nodes())

    for i, measure in enumerate(measures):
        ax = axes[i]
        ax.set_facecolor("#0d1117")
        values = [centrality_df.loc[n, measure] for n in nodes_in_order]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            alpha=0.25, arrows=True, arrowsize=8,
            edge_color="#888", connectionstyle="arc3,rad=0.08",
        )
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=nodes_in_order, node_color=values,
            node_size=420, cmap=plt.cm.plasma,
        )
        nodes.set_norm(mcolors.Normalize(vmin=min(values), vmax=max(values)))

        # White labels with a dark outline so they stay legible on both the
        # dark (low) and bright-yellow (high) ends of the plasma colormap.
        labels = nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=7, font_color="white", font_weight="bold",
        )
        for text in labels.values():
            text.set_path_effects([
                path_effects.withStroke(linewidth=2.5, foreground="black"),
            ])

        cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="gray", labelsize=7)
        cbar.outline.set_edgecolor("#333")

        ax.set_title(measure.replace("_", " ").title(),
                     color="white", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Hide any unused axes (e.g. 5 measures in a 2x3 grid).
    for j in range(len(measures), len(axes)):
        axes[j].axis("off")

    when = f" — {year}-{month:02d}" if year and month else ""
    fig.suptitle(f"Centrality Measures — Core 30 U.S. Airports{when}",
                 color="white", fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    saved {save_path}")
    else:
        plt.show()


def _draw_monthly_plot(task: tuple) -> None:
    file, year, month, airport_coords, output_dir = task
    print(f"  → Month {year}-{month:02d}")

    G = build_graph(file, airport_coords, month=month)
    path = os.path.join(output_dir, f"snapshot_{year}_{month:02d}.png")
    display_graph(G, year, month, save_path=path)


def generate_monthly_plots(output_dir: str = "."):
    """Generate and save a graph image for every month across all configured files."""
    airport_coords = load_airport_coords()
    print(f"Loaded coordinates for {len(airport_coords)} airports")
    os.makedirs(output_dir, exist_ok=True)

    files = [
        r"monthly/T_T100 2018.csv",
        r"monthly/T_T100 2019.csv",
        r"monthly/T_T100 2020.csv",
        r"monthly/T_T100D_MARKET_ALL_CARRIER 2_2021.csv",
        r"monthly/T_T100D_MARKET_ALL_CARRIER 2_2022.csv",
        r"monthly/T_T100D_MARKET_ALL_CARRIER 2_2023.csv",
        r"monthly/T_T100D_MARKET_ALL_CARRIER 2_2024.csv",
        r"monthly/T_T100D_MARKET_ALL_CARRIER 2_2025.csv",
        r"monthly/T_T100 2026.csv",
    ]

    tasks = []
    for file in files:
        print(f"\nProcessing {file}")
        df     = pd.read_csv(file, low_memory=False)
        year   = int(df["YEAR"].iloc[0])
        months = sorted(df["MONTH"].dropna().unique().astype(int))

        tasks.extend((file, year, month, airport_coords, output_dir) for month in months)

    # Multiprocessing to avoid sequential draw calls
    with mp.Pool() as pool:
        pool.map(_draw_monthly_plot, tasks)


if __name__ == "__main__":
    airport_coords = load_airport_coords()
    G = build_graph("monthly/T_T100 2018.csv", airport_coords, month=1)

    print(f"Airports : {G.number_of_nodes()}")
    print(f"Routes   : {G.number_of_edges()}")

    print("\n--- Departure Stats ---")
    print_departure_stats(G)

    print("\n--- Centrality Measures ---")
    centrality_df = compute_centrality(G)

    os.makedirs("temp", exist_ok=True)
    display_graph(G, 2018, 1, save_path="temp/core30_jan2018.png")
    plot_centrality_heatmaps(G, centrality_df, year=2018, month=1,
                             save_path="temp/core30_centrality_jan2018.png")

    # generate_monthly_plots(output_dir="temp/")
