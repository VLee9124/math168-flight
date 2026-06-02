import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from edge_analysis import (
    build_weighted_graph,
    compute_edge_metrics,
    load_core30_route_weights,
)


def _route_label(row: pd.Series) -> str:
    return f"{row['origin']}->{row['dest']}"


def _semantic_class(row: pd.Series, p_cut: float, b_cut: float) -> str:
    p_high = row["passengers_norm"] >= p_cut
    b_high = row["betweenness_norm"] >= b_cut
    if p_high and b_high:
        return "Critical Corridors"
    if p_high and not b_high:
        return "Heavy Local Pipes"
    if (not p_high) and b_high:
        return "Structural Bridges"
    return "Peripheral Links"


def plot_top_critical_routes(edge_metrics: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    top = edge_metrics.head(top_n).copy()
    top = top.sort_values("criticality_score", ascending=True)
    top["label"] = top.apply(_route_label, axis=1)

    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.barh(
        top["label"],
        top["criticality_score"],
        color=plt.cm.viridis(top["passenger_share"] / max(top["passenger_share"].max(), 1e-9)),
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Top Critical Directed Routes (Core 30)")
    ax.set_xlabel("Composite Criticality Score")
    ax.set_ylabel("Route")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    for bar, score in zip(bars, top["criticality_score"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_quadrants(edge_metrics: pd.DataFrame, output_path: Path, annotate_top_n: int = 12) -> None:
    df = edge_metrics.copy()
    p_cut = float(df["passengers_norm"].median())
    b_cut = float(df["betweenness_norm"].median())
    df["semantic_class"] = df.apply(_semantic_class, axis=1, p_cut=p_cut, b_cut=b_cut)
    df["label"] = df.apply(_route_label, axis=1)

    palette = {
        "Critical Corridors": "#d62728",
        "Heavy Local Pipes": "#ff7f0e",
        "Structural Bridges": "#1f77b4",
        "Peripheral Links": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(11.5, 8))
    for cls, g in df.groupby("semantic_class"):
        sizes = 80 + 520 * g["criticality_score"]
        ax.scatter(
            g["passengers_norm"],
            g["betweenness_norm"],
            s=sizes,
            c=palette[cls],
            alpha=0.75,
            label=cls,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.axvline(p_cut, color="black", linestyle="--", alpha=0.5)
    ax.axhline(b_cut, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Edge Semantics Map: Flow vs Structural Importance")
    ax.set_xlabel("Normalized Passenger Volume")
    ax.set_ylabel("Normalized Edge Betweenness")
    ax.grid(alpha=0.2, linestyle=":")

    to_annotate = df.head(annotate_top_n)
    for _, row in to_annotate.iterrows():
        ax.annotate(
            row["label"],
            (row["passengers_norm"], row["betweenness_norm"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_directional_imbalance(edge_metrics: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    df = edge_metrics.copy()
    df["label"] = df.apply(_route_label, axis=1)
    df = df.reindex(df["directional_imbalance"].abs().sort_values(ascending=False).index).head(top_n)
    df = df.sort_values("directional_imbalance")

    colors = ["#4c78a8" if x < 0 else "#e45756" for x in df["directional_imbalance"]]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(df["label"], df["directional_imbalance"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", linewidth=1.1)
    ax.set_title("Strongest Directional Imbalances in Core 30 Routes")
    ax.set_xlabel("Passenger Difference vs Reverse Route")
    ax.set_ylabel("Directed Route")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_critical_corridor_network(
    G: nx.DiGraph,
    edge_metrics: pd.DataFrame,
    output_path: Path,
    top_n: int = 35,
) -> None:
    top = edge_metrics.head(top_n)
    top_edges = {(row["origin"], row["dest"]): row["criticality_score"] for _, row in top.iterrows()}
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for (u, v), score in top_edges.items():
        if G.has_edge(u, v):
            H.add_edge(u, v, criticality=score, weight=G[u][v]["weight"])

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(H, seed=42, k=1.1)
    node_sizes = [600 if node in {"ATL", "DFW", "ORD", "LAX", "JFK"} else 280 for node in H.nodes()]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color="#2ca02c", alpha=0.85, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=8, font_color="white", ax=ax)

    edge_scores = [H[u][v]["criticality"] for u, v in H.edges()]
    widths = [1.0 + 6.0 * s for s in edge_scores]
    nx.draw_networkx_edges(
        H,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=10,
        width=widths,
        edge_color=edge_scores,
        edge_cmap=plt.cm.plasma,
        alpha=0.9,
        connectionstyle="arc3,rad=0.08",
    )

    ax.set_title("Critical Corridor Network (Top Directed Edges by Criticality)")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_edge_visuals(routes_csv: str, output_dir: str, top_n: int = 20) -> None:
    route_weights = load_core30_route_weights(routes_csv)
    G = build_weighted_graph(route_weights)
    edge_metrics = compute_edge_metrics(G)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "edge_metrics_core30.csv"
    edge_metrics.to_csv(metrics_path, index=False)

    plot_top_critical_routes(
        edge_metrics=edge_metrics,
        output_path=out / "edge_top_critical_routes.png",
        top_n=top_n,
    )
    plot_semantic_quadrants(
        edge_metrics=edge_metrics,
        output_path=out / "edge_semantic_quadrants.png",
        annotate_top_n=min(12, top_n),
    )
    plot_directional_imbalance(
        edge_metrics=edge_metrics,
        output_path=out / "edge_directional_imbalance.png",
        top_n=top_n,
    )
    plot_critical_corridor_network(
        G=G,
        edge_metrics=edge_metrics,
        output_path=out / "edge_critical_corridor_network.png",
        top_n=max(25, top_n),
    )

    print(f"Saved metrics and visuals to {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate edge-semantics visuals for Core 30 airport routes."
    )
    parser.add_argument(
        "--routes-csv",
        default="T100.csv",
        help="Path to route CSV with ORIGIN, DEST, PASSENGERS columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="visuals",
        help="Directory where PNG figures and CSV metrics will be saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many top routes to emphasize in ranking plots.",
    )
    args = parser.parse_args()
    generate_edge_visuals(args.routes_csv, args.output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
