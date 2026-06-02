import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


ROUTES_CSV = "T100.csv"

CORE_30 = {
    "ATL", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW", "EWR", "FLL",
    "HNL", "IAD", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO", "MDW", "MEM",
    "MIA", "MSP", "ORD", "PHL", "PHX", "SAN", "SEA", "SFO", "SLC", "TPA",
}


def load_core30_route_weights(routes_csv: str) -> pd.DataFrame:
    routes = pd.read_csv(routes_csv, low_memory=False)
    routes.columns = routes.columns.str.strip().str.upper().str.replace(" ", "_")

    expected_columns = {"ORIGIN", "DEST", "PASSENGERS"}
    missing_columns = expected_columns - set(routes.columns)
    if missing_columns:
        raise ValueError(
            "Missing required columns in routes CSV: "
            + ", ".join(sorted(missing_columns))
        )

    routes = routes[["ORIGIN", "DEST", "PASSENGERS"]].dropna()
    routes["ORIGIN"] = routes["ORIGIN"].astype(str).str.strip().str.upper()
    routes["DEST"] = routes["DEST"].astype(str).str.strip().str.upper()
    routes["PASSENGERS"] = pd.to_numeric(routes["PASSENGERS"], errors="coerce")
    routes = routes.dropna(subset=["PASSENGERS"])
    routes = routes[routes["PASSENGERS"] > 0]
    routes = routes[routes["ORIGIN"] != routes["DEST"]]
    routes = routes[
        routes["ORIGIN"].isin(CORE_30) & routes["DEST"].isin(CORE_30)
    ]

    grouped = (
        routes.groupby(["ORIGIN", "DEST"], as_index=False)["PASSENGERS"]
        .sum()
        .rename(columns={"PASSENGERS": "passengers"})
    )
    return grouped


def build_weighted_graph(route_weights: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for airport in CORE_30:
        G.add_node(airport)

    for _, row in route_weights.iterrows():
        G.add_edge(
            row["ORIGIN"],
            row["DEST"],
            weight=float(row["passengers"]),
        )

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    return G


def _normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(1.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_edge_metrics(G: nx.DiGraph) -> pd.DataFrame:
    betweenness = nx.edge_betweenness_centrality(G, weight=None)
    total_passengers = sum(data["weight"] for _, _, data in G.edges(data=True))

    records = []
    for u, v, data in G.edges(data=True):
        passengers = float(data["weight"])
        reverse_passengers = float(G[v][u]["weight"]) if G.has_edge(v, u) else 0.0
        records.append(
            {
                "origin": u,
                "dest": v,
                "passengers": passengers,
                "edge_betweenness": betweenness[(u, v)],
                "passenger_share": passengers / total_passengers if total_passengers else 0.0,
                "reverse_passengers": reverse_passengers,
                "directional_imbalance": passengers - reverse_passengers,
            }
        )

    metrics = pd.DataFrame(records)
    metrics["passengers_norm"] = _normalize(metrics["passengers"])
    metrics["betweenness_norm"] = _normalize(metrics["edge_betweenness"])
    metrics["criticality_score"] = (
        0.5 * metrics["passengers_norm"] + 0.5 * metrics["betweenness_norm"]
    )
    metrics = metrics.sort_values("criticality_score", ascending=False).reset_index(drop=True)
    return metrics


def print_top_edges(edge_metrics: pd.DataFrame, top_n: int = 15) -> None:
    cols = [
        "origin",
        "dest",
        "passengers",
        "edge_betweenness",
        "passenger_share",
        "directional_imbalance",
        "criticality_score",
    ]
    print(f"\nTop {top_n} critical directed routes (Core 30):")
    print(edge_metrics[cols].head(top_n).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge-level route criticality analysis for FAA Core 30 airports."
    )
    parser.add_argument(
        "--routes-csv",
        default=ROUTES_CSV,
        help="Path to route CSV (default: T100.csv)",
    )
    parser.add_argument(
        "--output-csv",
        default="edge_metrics_core30.csv",
        help="Path to write edge metrics CSV (default: edge_metrics_core30.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top routes to print (default: 15)",
    )
    args = parser.parse_args()

    route_weights = load_core30_route_weights(args.routes_csv)
    G = build_weighted_graph(route_weights)
    edge_metrics = compute_edge_metrics(G)
    print_top_edges(edge_metrics, top_n=args.top_n)

    output_path = Path(args.output_csv)
    edge_metrics.to_csv(output_path, index=False)
    print(f"\nSaved edge metrics to {output_path.resolve()}")


if __name__ == "__main__":
    main()
