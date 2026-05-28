import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


ROUTES_CSV = "T100.csv"
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def load_us_airports() -> pd.DataFrame:
    airports = pd.read_csv(OURAIRPORTS_URL)

    # Filter out non US airports 
    airports = airports[
        (airports["iso_country"] == "US")
        & airports["iata_code"].notna()
        & (airports["iata_code"].str.len() == 3)
    ].copy()
    
    airports["iata_code"] = airports["iata_code"].str.upper()

    return airports


def load_routes(routes_csv: str, year: int = None, month: int = None) -> pd.DataFrame:
    routes = pd.read_csv(routes_csv, low_memory=False)

    print(routes.head())
    print(routes.columns)

    routes.columns = (
        routes.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    routes = routes[["ORIGIN", "DEST", "YEAR", "MONTH"]].dropna()

    routes["ORIGIN"] = routes["ORIGIN"].astype(str).str.strip().str.upper()
    routes["DEST"] = routes["DEST"].astype(str).str.strip().str.upper()

    routes = routes[routes["ORIGIN"] != routes["DEST"]]

    print(routes["YEAR"].unique())
    print(routes["MONTH"].unique())

    if year is not None:
        routes = routes[routes["YEAR"] == year]

    if month is not None:
        routes = routes[routes["MONTH"] == month]

    print(routes["YEAR"].unique())
    print(routes["MONTH"].unique())

    return routes


def build_graph(routes_csv: str, year: int = None, month: int = None) -> nx.DiGraph:
    airports = load_us_airports()
    routes = load_routes(routes_csv, year, month)

    airport_lookup = airports.set_index("iata_code").to_dict(orient="index")
    valid_airports = set(airport_lookup.keys())

    # Keep only routes where both airports exist in OurAirports
    routes = routes[
        routes["ORIGIN"].isin(valid_airports)
        & routes["DEST"].isin(valid_airports)
    ]

    G = nx.DiGraph()

    for iata, attrs in airport_lookup.items():
        G.add_node(
            iata,
            name=attrs.get("name"),
            city=attrs.get("municipality"),
            airport_type=attrs.get("type"),
            latitude=attrs.get("latitude_deg"),
            longitude=attrs.get("longitude_deg"),
        )

    for _, row in routes.iterrows():
        G.add_edge(row["ORIGIN"], row["DEST"])

    # Remove airports with no routes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    return G


def display_graph(G: nx.DiGraph) -> None:
    pos = {
        airport: (
            data["longitude"],
            data["latitude"],
        )
        for airport, data in G.nodes(data=True)
        if pd.notna(data.get("longitude")) and pd.notna(data.get("latitude"))
    }

    # Keep only nodes with valid coordinates
    H = G.subgraph(pos.keys()).copy()

    plt.figure(figsize=(14, 9))

    nx.draw_networkx_edges(
        H,
        pos,
        arrows=True,
        arrowsize=6,
        alpha=0.2,
        width=0.6,
    )

    nx.draw_networkx_nodes(
        H,
        pos,
        node_size=75,
        node_color='#5DBB63',
    )

    nx.draw_networkx_labels(
        H,
        pos,
        font_size=10,
    )

    plt.title("U.S. Airport Route Graph")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    plt.tight_layout()
    plt.style.use('fast')
    
    plt.show()


if __name__ == "__main__":
    graph = build_graph(ROUTES_CSV, year=2026, month=1)
    print(f"Airports: {graph.number_of_nodes()}")
    print(f"Routes: {graph.number_of_edges()}")

    display_graph(graph)